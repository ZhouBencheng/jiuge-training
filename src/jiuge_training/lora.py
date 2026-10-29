"""Tensor-parallel LoRA with explicit replicated/sharded factor semantics."""

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ParallelLoRA(nn.Module):
    """Column: B is sharded, A is replicated. Row: A sharded, B replicated.

    Column copy-backward sums A/input gradients; row reduce-forward sums
    the rank-local low-rank activations. Replicated factors thus receive
    full gradients.
    """

    def __init__(self, base, kind, rank, alpha, dropout=0.0, seed=1234):
        super().__init__()
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError(
                'LoRA rank/alpha must be positive; dropout must be in [0, 1)'
            )
        if kind not in ('column', 'row'):
            raise ValueError('kind must be column or row')
        if getattr(base, 'sequence_parallel', False):
            raise ValueError('Sequence-parallel LoRA is not supported')
        if kind == 'column' and getattr(base, 'gather_output', False):
            raise ValueError('LoRA targets require gather_output=False')
        if kind == 'row' and not getattr(base, 'input_is_parallel', True):
            raise ValueError('Row LoRA targets require input_is_parallel=True')
        self.base = base
        self.kind = kind
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.tp_group = getattr(base, 'tp_group', None)
        world = self.tp_group.size() if self.tp_group is not None else 1
        tp_rank = self.tp_group.rank() if self.tp_group is not None else 0
        out_local, in_local = base.weight.shape
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, in_local))
        self.lora_B = nn.Parameter(base.weight.new_zeros(out_local, rank))
        # Construct the same global A on every rank, then select a row-parallel
        # shard.
        generator = torch.Generator(device='cpu').manual_seed(seed)
        full_input = in_local * world if kind == 'row' else in_local
        initial = torch.empty(rank, full_input, dtype=torch.float32)
        initial.uniform_(
            -1 / math.sqrt(full_input),
            1 / math.sqrt(full_input),
            generator=generator,
        )
        if kind == 'row':
            initial = initial[:, tp_rank * in_local : (tp_rank + 1) * in_local]
        with torch.no_grad():
            self.lora_A.copy_(initial)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        for name, parameter in (
            ('lora_A', self.lora_A),
            ('lora_B', self.lora_B),
        ):
            sharded = (kind == 'row' and name == 'lora_A') or (
                kind == 'column' and name == 'lora_B'
            )
            parameter.tensor_model_parallel = sharded
            parameter.partition_dim = 1 if name == 'lora_A' else 0
            parameter.partition_stride = 1
            parameter.allreduce = True

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, input_, *args, **kwargs):
        output, bias = self.base(input_, *args, **kwargs)
        if self.dropout.p and self.training and input_.is_cuda:
            from megatron.core.tensor_parallel.random import (
                get_cuda_rng_tracker,
                get_data_parallel_rng_tracker_name,
            )

            # In particular, column A must see identical dropout on every TP
            # rank.
            tracker = get_cuda_rng_tracker()
            stream = (
                tracker.fork(get_data_parallel_rng_tracker_name())
                if self.kind == 'column'
                else tracker.fork()
            )
            with stream:
                dropped = self.dropout(input_)
        else:
            dropped = self.dropout(input_)
        low_rank = F.linear(dropped, self.lora_A)
        if self.tp_group is not None and self.tp_group.size() > 1:
            from megatron.core.tensor_parallel.mappings import (
                copy_to_tensor_model_parallel_region,
                reduce_from_tensor_model_parallel_region,
            )

            if self.kind == 'column':
                low_rank = copy_to_tensor_model_parallel_region(
                    low_rank, group=self.tp_group
                )
            else:
                low_rank = reduce_from_tensor_model_parallel_region(
                    low_rank, group=self.tp_group
                )
        return output + F.linear(low_rank, self.lora_B) * self.scaling, bias

    def backward_dw(self):
        return self.base.backward_dw()

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        from megatron.core.transformer.utils import (
            make_sharded_tensors_for_checkpoint,
        )

        result = self.base.sharded_state_dict(
            prefix + 'base.', sharded_offsets, metadata
        )
        axes = {'lora_A': 1} if self.kind == 'row' else {'lora_B': 0}
        result.update(
            make_sharded_tensors_for_checkpoint(
                {'lora_A': self.lora_A, 'lora_B': self.lora_B},
                prefix,
                axes,
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata['dp_cp_group'],
            )
        )
        return result


def attach_lora(model, rank=8, alpha=16.0, dropout=0.0, seed=1234):
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    if any(isinstance(module, ParallelLoRA) for module in model.modules()):
        raise ValueError('LoRA is already attached')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = []
    for name, module in list(model.named_modules()):
        # Embeddings/lm_head stay frozen; all attention and MLP projections
        # adapt.
        if not name.startswith('decoder.layers.'):
            continue
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            parent_name, child = name.rsplit('.', 1)
            # Layer numbers are global across PP, so seed is independent of PP
            # partition.
            local_index = int(name.split('.')[2])
            global_name = name.replace(
                f'decoder.layers.{local_index}.',
                'decoder.layers.'
                f'{model.decoder.layers[local_index].layer_number - 1}.',
                1,
            )
            module_seed = seed + int.from_bytes(
                hashlib.sha256(global_name.encode()).digest()[:4], 'little'
            )
            wrapper = ParallelLoRA(
                module,
                (
                    'column'
                    if isinstance(module, ColumnParallelLinear)
                    else 'row'
                ),
                rank,
                alpha,
                dropout,
                module_seed,
            )
            setattr(model.get_submodule(parent_name), child, wrapper)
            targets.append(global_name)
    if not targets:
        raise ValueError('No attention/MLP parallel linear modules found')
    model.lora_config = dict(
        rank=rank, alpha=alpha, dropout=dropout, seed=seed, targets=targets
    )
    return targets


def adapter_state_dict(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if name.endswith(('.lora_A', '.lora_B'))
    }


def _layout(model):
    groups = getattr(model, 'pg_collection', None)
    return dict(
        tp=groups.tp.size() if groups else 1,
        tp_rank=groups.tp.rank() if groups else 0,
        pp=groups.pp.size() if groups else 1,
        pp_rank=groups.pp.rank() if groups else 0,
        hf_config=getattr(model, 'hf_config', None),
        lora_config=getattr(model, 'lora_config', None),
    )


def save_adapter(model, path):
    """Save this TP/PP shard; caller gives each shard its own filename."""
    state = adapter_state_dict(model)
    if not state:
        raise ValueError('No adapters to save')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(
        {'format': 'jiuge-lora-v1', 'layout': _layout(model), 'state': state},
        temporary,
    )
    temporary.replace(path)


def load_adapter(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != 'jiuge-lora-v1' or payload[
        'layout'
    ] != _layout(model):
        raise ValueError('Adapter configuration or TP/PP layout mismatch')
    expected = adapter_state_dict(model)
    if expected.keys() != payload['state'].keys():
        raise ValueError('Adapter key mismatch')
    parameters = dict(model.named_parameters())
    for name, value in payload['state'].items():
        if value.shape != parameters[name].shape:
            raise ValueError(f'Adapter shape mismatch: {name}')
    with torch.no_grad():
        for name, value in payload['state'].items():
            parameters[name].copy_(value)
