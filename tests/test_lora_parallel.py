"""Opt-in NCCL TP gradient test: torchrun --standalone --nproc-per-node=4
this_file.
"""

import json
import hashlib
import inspect
import os
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F


def main():
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from megatron.core.tensor_parallel.random import (
        model_parallel_cuda_manual_seed,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from jiuge_training.lora import ParallelLoRA, save_adapter, load_adapter

    sources = [Path(__file__), Path(inspect.getfile(ParallelLoRA))]
    fingerprints = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group('nccl')
    world, rank = (
        torch.distributed.get_world_size(),
        torch.distributed.get_rank(),
    )
    ps.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    group = ps.get_tensor_model_parallel_group()
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=world,
        tensor_model_parallel_size=world,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
    )
    cases = []
    for kind in ('column', 'row'):
        torch.manual_seed(42)
        full_weight = torch.randn(12, 16, device='cpu').cuda() * 0.1
        a = (torch.randn(3, 16, device='cpu').cuda() * 0.1).requires_grad_()
        b = (torch.randn(12, 3, device='cpu').cuda() * 0.1).requires_grad_()
        x = torch.randn(4, 2, 16, device='cpu').cuda().requires_grad_()
        reference = F.linear(x, full_weight) + F.linear(F.linear(x, a), b) * 2
        if kind == 'column':
            base = ColumnParallelLinear(
                16,
                12,
                config=cfg,
                init_method=cfg.init_method,
                bias=False,
                gather_output=False,
                skip_bias_add=True,
                tp_group=group,
            ).cuda()
            base.weight.data.copy_(full_weight.chunk(world, dim=0)[rank])
            local_input = x.detach().clone().requires_grad_()
            expected_output = reference.chunk(world, dim=-1)[rank]
        else:
            base = RowParallelLinear(
                16,
                12,
                config=cfg,
                init_method=cfg.init_method,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                tp_group=group,
            ).cuda()
            base.weight.data.copy_(full_weight.chunk(world, dim=1)[rank])
            local_input = (
                x.detach().chunk(world, dim=-1)[rank].clone().requires_grad_()
            )
            expected_output = reference
        adapter = ParallelLoRA(base, kind, 3, 6).cuda()
        with torch.no_grad():
            torch.testing.assert_close(
                adapter(local_input)[0], base(local_input)[0], rtol=0, atol=0
            )
            adapter.lora_A.copy_(
                a if kind == 'column' else a.chunk(world, dim=1)[rank]
            )
            adapter.lora_B.copy_(
                b.chunk(world, dim=0)[rank] if kind == 'column' else b
            )
        output = adapter(local_input)[0]
        torch.testing.assert_close(
            output, expected_output, atol=2e-6, rtol=2e-5
        )
        output.square().sum().backward()
        reference.square().sum().backward()
        torch.testing.assert_close(
            adapter.lora_A.grad,
            a.grad if kind == 'column' else a.grad.chunk(world, dim=1)[rank],
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            adapter.lora_B.grad,
            b.grad.chunk(world, dim=0)[rank] if kind == 'column' else b.grad,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            local_input.grad,
            x.grad if kind == 'column' else x.grad.chunk(world, dim=-1)[rank],
            atol=2e-5,
            rtol=2e-5,
        )
        assert base.weight.grad is None and not base.weight.requires_grad
        # Wrap so named parameter keys use the same .lora_A/B naming as a full
        # model.
        wrapped = torch.nn.Sequential(adapter)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.pt'
            save_adapter(wrapped, path)
            with torch.no_grad():
                adapter.lora_B.zero_()
            load_adapter(wrapped, path)
            torch.testing.assert_close(
                adapter(local_input)[0], output.detach(), rtol=0, atol=0
            )
        # Nonzero dropout: replicated factors keep identical TP gradients while
        # row feature shards use independent masks (column inputs are
        # replicated).
        adapter.zero_grad(set_to_none=True)
        adapter.dropout.p = 0.25
        recorded = []
        hook = adapter.dropout.register_forward_hook(
            lambda module, inputs, result: recorded.append(result.detach())
        )
        ones = torch.ones(16, 4, local_input.shape[-1], device='cuda')
        adapter(ones)[0].square().sum().backward()
        hook.remove()
        replicated = (
            adapter.lora_A.grad if kind == 'column' else adapter.lora_B.grad
        )
        gathered = [torch.empty_like(replicated) for _ in range(world)]
        torch.distributed.all_gather(gathered, replicated, group=group)
        for value in gathered:
            torch.testing.assert_close(value, replicated, rtol=2e-5, atol=2e-5)
        masks = [torch.empty_like(recorded[0]) for _ in range(world)]
        torch.distributed.all_gather(masks, recorded[0], group=group)
        if kind == 'column':
            assert all(torch.equal(masks[0], mask) for mask in masks[1:])
        elif world > 1:
            assert any(not torch.equal(masks[0], mask) for mask in masks[1:])
        cases.append(kind)
    torch.distributed.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    'passed': True,
                    'world_size': world,
                    'cases': cases,
                    'source_sha256': fingerprints,
                    'checks': [
                        'zero_init',
                        'dense_forward',
                        'A_gradient',
                        'B_gradient',
                        'input_gradient',
                        'base_frozen',
                        'adapter_restore',
                        'dropout_replicated_gradients',
                        'dropout_mask_sharding',
                    ],
                }
            ),
            flush=True,
        )
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
