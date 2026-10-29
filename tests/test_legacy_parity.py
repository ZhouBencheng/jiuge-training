"""Opt-in original CPM9G eager/Megatron parity on a real legacy checkpoint."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault('USE_TF', '0')
import torch  # noqa: E402


def native_layers(upstream, destination):
    """Load only the upstream pure-PyTorch layer package, without BMTrain."""
    prefix = '9G-Train/cpm/native_layers/'
    names = subprocess.check_output(
        [
            'git',
            '-C',
            upstream,
            'ls-tree',
            '-r',
            '--name-only',
            'master',
            prefix,
        ],
        text=True,
    ).splitlines()
    sources = {}
    for name in names:
        if not name.endswith('.py'):
            continue
        raw = subprocess.check_output(
            ['git', '-C', upstream, 'show', f'master:{name}']
        )
        target = destination / name.removeprefix(prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        sources[name] = hashlib.sha256(raw).hexdigest()
    module_name = '_jiuge_parity_native_layers'
    spec = importlib.util.spec_from_file_location(
        module_name,
        destination / '__init__.py',
        submodule_search_locations=[str(destination)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, sources


class NativeReference(torch.nn.Module):
    """CPM9GTorch causal forward using its original native layer modules."""

    def __init__(self, layers, config, dtype):
        super().__init__()
        self.encoder = layers.Encoder(
            num_layers=config['num_layers'],
            dim_model=config['dim_model'],
            dim_ff=config['dim_ff'],
            num_heads=config['num_heads'],
            num_kv_heads=config['num_kv_heads'],
            dim_head=config['dim_head'],
            activate_fn=config['activate_fn'],
            dtype=dtype,
            eps=config['eps'],
            dropout_p=0.0,
            scale=False,
            mask_modules=None,
            use_flash_attn=False,
        )
        options = dict(
            vocab_size=config['vocab_size'],
            embedding_size=config['dim_model'],
            scale=False,
            dtype=dtype,
        )
        self.input_embedding = layers.Embedding(**options)
        self.lm_head = layers.Embedding(**options)
        self.position_bias = layers.RotaryEmbeddingESM(
            dim=config['dim_head'],
            dtype=dtype,
            base=config.get('base', 10000),
            persistent=False,
            mixed_precision=True,
        )

    def forward(self, ids):
        length = ids.shape[1]
        mask = torch.ones(length, length, device=ids.device, dtype=torch.bool)
        hidden = self.encoder(
            self.input_embedding(ids),
            attention_mask=mask.tril().unsqueeze(0),
            position_bias=self.position_bias,
            pos_bias_type='rotary',
        )
        return self.lm_head.projection(hidden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--dtype', choices=['float16', 'bfloat16'], default='float16'
    )
    parser.add_argument(
        '--backend', choices=['reference', 'triton'], default='reference'
    )
    args = parser.parse_args()
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import (
        model_parallel_cuda_manual_seed,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from jiuge_training.legacy import LEGACY_FORMAT, model_config
    from jiuge_training.model import configure, model_provider

    root = Path(args.model_path)
    raw = json.loads((root / 'config.json').read_text())
    normalized = model_config(raw, 8)
    dtype = getattr(torch, args.dtype)
    torch.cuda.set_device(0)
    ids = torch.tensor([[1, 101, 102, 300, 400, 200, 50, 2]], device='cuda')
    layer_outputs = {'native': {}, 'megatron': {}}

    def capture(backend, index, transpose=False):
        def hook(module, inputs, outputs):
            value = outputs[0] if isinstance(outputs, tuple) else outputs
            if transpose:
                value = value.transpose(0, 1)
            layer_outputs[backend][index] = value.detach().float().cpu()

        return hook

    with tempfile.TemporaryDirectory() as directory:
        layers, sources = native_layers(args.upstream, Path(directory))
        with torch.device('meta'):
            reference = NativeReference(layers, raw, dtype)
        files = list(root.glob('*.pt'))
        if len(files) != 1:
            raise ValueError('Expected exactly one legacy checkpoint')
        state = torch.load(
            files[0], weights_only=True, mmap=True, map_location='cpu'
        )
        reference.load_state_dict(state, strict=True, assign=True)
        del state
        reference = reference.cuda().eval()
        # Keep mixed-precision RoPE's inverse-frequency buffer in FP32.
        for parameter in reference.parameters():
            parameter.data = parameter.data.to(dtype=dtype)
        assert reference.position_bias.inv_freq.dtype == torch.float32
        assert reference.position_bias.dtype == dtype
        for index, layer in enumerate(reference.encoder.layers):
            layer.register_forward_hook(capture('native', index))
        with torch.no_grad():
            expected = reference(ids).float().cpu()
        del reference
        gc.collect()
        torch.cuda.empty_cache()

    torch.distributed.init_process_group(
        'nccl', init_method='tcp://127.0.0.1:29689', rank=0, world_size=1
    )
    try:
        ps.initialize_model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)
        cfg = TransformerConfig(
            num_layers=raw['num_layers'],
            hidden_size=raw['dim_model'],
            num_attention_heads=raw['num_heads'],
            num_query_groups=raw['num_kv_heads'],
            ffn_hidden_size=raw['dim_ff'],
            normalization='RMSNorm',
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            use_cpu_initialization=True,
            gradient_accumulation_fusion=False,
        )
        configure(
            dict(
                model=dict(
                    path=args.model_path, checkpoint_format=LEGACY_FORMAT
                ),
                task=dict(name='cpt', load='', seq_length=8),
                backend=dict(rmsnorm=args.backend),
            )
        )
        model = model_provider(config=cfg).cuda().eval()
        for index, layer in enumerate(model.decoder.layers):
            layer.register_forward_hook(capture('megatron', index, True))
        positions = torch.arange(8, device='cuda').unsqueeze(0)
        mask = torch.ones(1, 1, 8, 8, device='cuda', dtype=torch.bool).triu(1)
        with torch.no_grad():
            actual = model(ids, positions, mask).float().cpu()

        def loss(logits):
            return float(
                torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, normalized['vocab_size']),
                    ids[:, 1:].cpu().reshape(-1),
                )
            )

        def relative(a, b):
            return float(
                torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)
            )

        sources.update(
            {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [Path(__file__), root / 'config.json']
                + sorted((Path(__file__).parents[1] / 'src').rglob('*.py'))
            }
        )
        result = dict(
            reference='Official CPM9G native_layers causal eager harness',
            model=args.model_path,
            dtype=args.dtype,
            backend=args.backend,
            tokens=8,
            source_sha256=sources,
            relative_l2=relative(actual, expected),
            max_abs=float((actual - expected).abs().max()),
            native_loss=loss(expected),
            megatron_loss=loss(actual),
            top1_agreement=float(
                (actual.argmax(-1) == expected.argmax(-1)).float().mean()
            ),
            layer_relative_l2={
                index: relative(value, layer_outputs['native'][index])
                for index, value in layer_outputs['megatron'].items()
            },
            load_report=model.hf_load_report,
        )
        result['passed'] = (
            result['relative_l2'] < 0.03
            and abs(result['native_loss'] - result['megatron_loss']) < 0.1
            and result['top1_agreement'] >= 0.875
        )
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
        if not result['passed']:
            raise SystemExit('Native CPM9G/Megatron parity failed')
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
