"""Opt-in real checkpoint parity: python tests/test_model_parity.py
--model-path PATH.
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path

# Set the optional framework environment before importing torch.
os.environ.setdefault('USE_TF', '0')
import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--backend', default='reference')
    parser.add_argument(
        '--dtype', choices=['bfloat16', 'float32'], default='bfloat16'
    )
    parser.add_argument('--output')
    args = parser.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import (
        model_parallel_cuda_manual_seed,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from jiuge_training.model import configure, model_provider
    import jiuge_training.model
    import jiuge_training.weights
    import jiuge_training.lora
    from importlib import import_module

    norm_source = Path(inspect.getfile(import_module('jiuge_ops.rms_norm')))
    sources = [
        Path(__file__),
        Path(inspect.getfile(jiuge_training.model)),
        Path(inspect.getfile(jiuge_training.weights)),
        Path(inspect.getfile(jiuge_training.lora)),
        norm_source,
        norm_source.with_name('_triton.py'),
        Path(args.model_path) / 'config.json',
        Path(args.model_path) / 'configuration_fm9g.py',
        Path(args.model_path) / 'modeling_fm9g.py',
    ]
    source_sha256 = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    hf = json.loads((Path(args.model_path) / 'config.json').read_text())
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    dtype = getattr(torch, args.dtype)
    ids = torch.tensor(
        [[1, 101, 102, 300, 400, 200, 50, 2] * ((args.tokens + 7) // 8)],
        device='cuda',
    )[:, : args.tokens]
    reference_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    # FM9G's config constructor overrides the requested implementation when
    # flash is installed.
    reference_config._attn_implementation = 'eager'
    reference = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=reference_config,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation='eager',
        )
        .cuda()
        .eval()
    )
    reference_attention = type(reference.model.layers[0].self_attn).__name__
    reference_source = Path(
        inspect.getfile(type(reference.model.layers[0].self_attn))
    )
    source_sha256[str(reference_source)] = hashlib.sha256(
        reference_source.read_bytes()
    ).hexdigest()
    if reference_attention != 'FM9GAttention':
        raise RuntimeError(
            f'Expected explicit HF eager reference, got {reference_attention}'
        )
    hf_layers, mg_layers = {}, {}

    def capture(store, name, transpose=False):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if transpose:
                value = value.transpose(0, 1)
            store[name] = value.detach().float().cpu()

        return hook

    for i, layer in enumerate(reference.model.layers):
        layer.register_forward_hook(capture(hf_layers, str(i)))
        if i == 0:
            layer.input_layernorm.register_forward_hook(
                capture(hf_layers, 'norm0')
            )
            layer.self_attn.register_forward_hook(capture(hf_layers, 'attn0'))
            layer.mlp.register_forward_hook(capture(hf_layers, 'mlp0'))
    with torch.no_grad():
        expected = reference(ids, use_cache=False).logits.float().cpu()
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.init_process_group(
        'nccl', init_method='tcp://127.0.0.1:29687', rank=0, world_size=1
    )
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    model_parallel_cuda_manual_seed(1234)
    cfg = TransformerConfig(
        num_layers=hf['num_hidden_layers'],
        hidden_size=hf['hidden_size'],
        num_attention_heads=hf['num_attention_heads'],
        num_query_groups=hf.get(
            'num_key_value_heads', hf['num_attention_heads']
        ),
        ffn_hidden_size=hf['intermediate_size'],
        normalization='RMSNorm',
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        pipeline_dtype=dtype,
    )
    settings = dict(
        model=dict(path=args.model_path, smoke=False),
        task=dict(name='cpt', load=''),
        backend=dict(rmsnorm=args.backend),
    )
    configure(settings)
    model = model_provider(config=cfg).cuda().eval()
    for i, layer in enumerate(model.decoder.layers):
        layer.register_forward_hook(capture(mg_layers, str(i), True))
        if i == 0:
            layer.input_layernorm.register_forward_hook(
                capture(mg_layers, 'norm0', True)
            )
            layer.self_attention.register_forward_hook(
                capture(mg_layers, 'attn0', True)
            )
            layer.mlp.register_forward_hook(capture(mg_layers, 'mlp0', True))
    positions = torch.arange(ids.shape[1], device='cuda').unsqueeze(0)
    mask = torch.triu(
        torch.ones(
            1, 1, ids.shape[1], ids.shape[1], device='cuda', dtype=torch.bool
        ),
        diagonal=1,
    )
    with torch.no_grad():
        actual = (
            model(ids, positions, mask).float().cpu()[..., : hf['vocab_size']]
        )
    difference = (actual - expected).abs()
    result = dict(
        model=args.model_path,
        backend=args.backend,
        hf_attention=reference_attention,
        source_sha256=source_sha256,
        dtype=args.dtype,
        tokens=args.tokens,
        mean_abs=float(difference.mean()),
        max_abs=float(difference.max()),
        relative_l2=float(
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected)
        ),
        top1_agreement=float(
            (actual.argmax(-1) == expected.argmax(-1)).float().mean()
        ),
        hf_loss=float(
            torch.nn.functional.cross_entropy(
                expected[:, :-1].reshape(-1, hf['vocab_size']),
                ids[:, 1:].cpu().reshape(-1),
            )
        ),
        megatron_loss=float(
            torch.nn.functional.cross_entropy(
                actual[:, :-1].reshape(-1, hf['vocab_size']),
                ids[:, 1:].cpu().reshape(-1),
            )
        ),
        load_report=model.hf_load_report,
        layer_relative_l2={
            name: float(
                torch.linalg.vector_norm(value - hf_layers[name])
                / torch.linalg.vector_norm(hf_layers[name])
            )
            for name, value in mg_layers.items()
        },
    )
    result['passed'] = (
        result['relative_l2'] < 0.03
        and abs(result['hf_loss'] - result['megatron_loss']) < 0.1
        and result['top1_agreement'] >= 0.875
    )
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    torch.distributed.destroy_process_group()
    if not result['passed']:
        raise SystemExit('HF/Megatron BF16 parity failed')


if __name__ == '__main__':
    main()
