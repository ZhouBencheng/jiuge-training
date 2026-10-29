"""Real MCore full-recompute regression: torchrun --nproc-per-node=4
this_file --pp 1|2.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

# Set the optional framework environment before importing torch.
os.environ.setdefault('USE_TF', '0')
import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pp', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel.schedules import (
        get_forward_backward_func,
    )
    from megatron.core.tensor_parallel.random import (
        model_parallel_cuda_manual_seed,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from jiuge_training.model import configure, model_provider

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group('nccl')
    world, rank = (
        torch.distributed.get_world_size(),
        torch.distributed.get_rank(),
    )
    tp = world // args.pp
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=args.pp
    )
    model_parallel_cuda_manual_seed(1234)
    cfg = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        ffn_hidden_size=128,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=args.pp,
        normalization='RMSNorm',
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        recompute_granularity='full',
        recompute_method='uniform',
        recompute_num_layers=1,
    )
    configure(
        dict(
            model=dict(
                smoke=True,
                layers=4,
                hidden_size=64,
                num_attention_heads=4,
                num_query_groups=2,
                ffn_hidden_size=128,
                vocab_size=128,
                tie_word_embeddings=True,
            ),
            task=dict(
                name='lora',
                seq_length=8,
                lora_rank=4,
                lora_alpha=8,
                lora_dropout=0.0,
                seed=1234,
            ),
            backend=dict(rmsnorm='reference'),
        )
    )
    model = (
        model_provider(
            pre_process=ps.is_pipeline_first_stage(),
            post_process=ps.is_pipeline_last_stage(),
            config=cfg,
        )
        .cuda()
        .train()
    )
    base = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    trainable = {
        name: value
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    assert trainable and all(
        name.endswith(('.lora_A', '.lora_B')) for name in trainable
    )
    bs = {
        name: value
        for name, value in trainable.items()
        if name.endswith('.lora_B')
    }
    assert all(torch.count_nonzero(value) == 0 for value in bs.values())
    embedding_requires_grad = []
    if model.pre_process:
        model.embedding.register_forward_hook(
            lambda module, inputs, output: embedding_requires_grad.append(
                output.requires_grad
            )
        )
    optimizer = torch.optim.SGD(trainable.values(), lr=0.001)
    tokens = torch.tensor([[1, 11, 23, 42, 57, 66, 78, 99]], device='cuda')
    labels = tokens.roll(-1, dims=-1)
    positions = torch.arange(8, device='cuda').unsqueeze(0)
    mask = torch.triu(
        torch.ones(1, 1, 8, 8, device='cuda', dtype=torch.bool), diagonal=1
    )

    def forward_step(iterator, stage):
        output = stage(
            tokens,
            positions,
            mask,
            labels=labels if stage.post_process else None,
        )

        def loss_func(losses):
            loss = losses.float().sum()
            assert torch.isfinite(loss)
            return (
                loss,
                torch.tensor(losses.numel(), device='cuda', dtype=torch.int),
                {'loss': loss.detach()},
            )

        return output, loss_func

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=None,
            model=model,
            num_microbatches=2,
            seq_length=8,
            micro_batch_size=1,
            forward_only=False,
        )
        for name, value in bs.items():
            assert (
                value.grad is not None
                and torch.isfinite(value.grad).all()
                and torch.count_nonzero(value.grad) > 0
            ), name
        optimizer.step()
        for name, value in bs.items():
            assert torch.count_nonzero(value) > 0, name
        for name, value in model.named_parameters():
            if name in base:
                assert value.grad is None and not value.requires_grad, name
                torch.testing.assert_close(value, base[name], atol=0, rtol=0)
    if model.pre_process:
        assert embedding_requires_grad and all(embedding_requires_grad)
        model.eval()
        with torch.no_grad():
            embedded = model.embedding(tokens, positions)
        assert not embedded.requires_grad
    if args.pp == 1:
        model.train()
        with torch.no_grad():
            with_recompute = model(tokens, positions, mask)
            cfg.recompute_granularity = None
            without_recompute = model(tokens, positions, mask)
        torch.testing.assert_close(
            with_recompute, without_recompute, rtol=0, atol=0
        )
    result = dict(
        rank=rank,
        tp=tp,
        pp=args.pp,
        stage=ps.get_pipeline_model_parallel_rank(),
        adapters_B=len(bs),
        updated_B=sum(
            bool(torch.count_nonzero(value)) for value in bs.values()
        ),
        base_frozen=True,
        steps=2,
        full_recompute=True,
    )
    results = [None] * world
    torch.distributed.all_gather_object(results, result)
    if rank == 0:
        root = Path(__file__).resolve().parents[1]
        sources = [
            root / 'src/jiuge_training/model.py',
            root / 'src/jiuge_training/lora.py',
            Path(__file__),
        ]
        print(
            json.dumps(
                dict(
                    passed=True,
                    ranks=results,
                    source_sha256={
                        str(path): hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        for path in sources
                    },
                )
            ),
            flush=True,
        )
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
