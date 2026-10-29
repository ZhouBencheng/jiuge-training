"""Strict, rank-local Hugging Face -> Megatron weight import."""

import json
from pathlib import Path

import torch


class CheckpointReader:
    """Map checkpoint files, fetching only the requested tensor slice."""

    def __init__(self, path, legacy_layers=None):
        self.path = Path(path)
        self.handles = {}
        self.source_names = {}
        if legacy_layers is not None:
            from .legacy import weight_names

            files = sorted(self.path.glob('*.pt'))
            if len(files) != 1:
                raise ValueError('Expected exactly one legacy CPM9G .pt file')
            state = torch.load(
                files[0], weights_only=True, mmap=True, map_location='cpu'
            )
            self.source_names = weight_names(legacy_layers)
            expected, actual = set(self.source_names.values()), set(state)
            if actual != expected:
                raise ValueError(
                    'Legacy key mismatch: '
                    f'missing={sorted(expected - actual)}, '
                    f'unexpected={sorted(actual - expected)}'
                )
            self.handles[files[0].name] = state
            self.weight_map = {key: files[0].name for key in self.source_names}
            self.kind = 'torch'
            return
        indexes = list(self.path.glob('*.safetensors.index.json'))
        if indexes:
            self.weight_map = json.loads(indexes[0].read_text())['weight_map']
            self.kind = 'safetensors'
        elif list(self.path.glob('*.safetensors')):
            from safetensors import safe_open

            self.weight_map = {}
            for file in self.path.glob('*.safetensors'):
                with safe_open(
                    str(file), framework='pt', device='cpu'
                ) as handle:
                    self.weight_map.update(
                        {key: file.name for key in handle.keys()}
                    )
            self.kind = 'safetensors'
        else:
            files = list(self.path.glob('pytorch_model*.bin'))
            if len(files) != 1:
                raise ValueError(
                    'Expected safetensors or one mmap-capable '
                    'pytorch_model.bin'
                )
            self.handles[files[0].name] = torch.load(
                files[0], weights_only=True, mmap=True, map_location='cpu'
            )
            self.weight_map = {
                key: files[0].name for key in self.handles[files[0].name]
            }
            self.kind = 'torch'

    def tensor(self, name, index=None):
        file = self.weight_map[name]
        if self.kind == 'torch':
            value = self.handles[file][self.source_names.get(name, name)]
            return value if index is None else value[index]
        from safetensors import safe_open

        with safe_open(
            str(self.path / file), framework='pt', device='cpu'
        ) as handle:
            return (
                handle.get_tensor(name)
                if index is None
                else handle.get_slice(name)[index]
            )


def packed_qkv_slice(
    reader, prefix, suffix, heads, groups, head_dim, rank, world
):
    """Megatron packs [Q heads of group, K head, V head], then shards rows.

    This also matches MCore's all-gather strategy when TP > KV heads.
    """
    queries = heads // groups
    total = (heads + 2 * groups) * head_dim
    if total % world:
        raise ValueError('QKV packed width must divide TP')
    start, stop = rank * total // world, (rank + 1) * total // world
    parts, offset = [], 0
    for group in range(groups):
        for projection, count, source_start in (
            ('q_proj', queries * head_dim, group * queries * head_dim),
            ('k_proj', head_dim, group * head_dim),
            ('v_proj', head_dim, group * head_dim),
        ):
            left, right = max(start, offset), min(stop, offset + count)
            if left < right:
                index = slice(
                    source_start + left - offset, source_start + right - offset
                )
                parts.append(
                    reader.tensor(f'{prefix}.{projection}.{suffix}', index)
                )
            offset += count
    return torch.cat(parts, dim=0)


def expected_hf_keys(hf):
    keys = {'model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'}
    qkv_bias = hf.get('model_type') == 'fm9g7b' or hf.get(
        'attention_bias', False
    )
    for i in range(hf['num_hidden_layers']):
        prefix = f'model.layers.{i}'
        keys.update(
            f'{prefix}.{key}.weight'
            for key in (
                'input_layernorm',
                'post_attention_layernorm',
                'self_attn.q_proj',
                'self_attn.k_proj',
                'self_attn.v_proj',
                'self_attn.o_proj',
                'mlp.gate_proj',
                'mlp.up_proj',
                'mlp.down_proj',
            )
        )
        if qkv_bias:
            keys.update(
                f'{prefix}.self_attn.{key}_proj.bias'
                for key in ('q', 'k', 'v')
            )
    return keys


def load_hf_weights(model, path, hf=None):
    """Import exactly this pipeline stage and TP shard, before attaching
    LoRA.
    """
    hf = hf or json.loads((Path(path) / 'config.json').read_text())
    legacy = hf.get('checkpoint_format') == 'cpm9g-bmtrain'
    reader = CheckpointReader(
        path, legacy_layers=hf['num_hidden_layers'] if legacy else None
    )
    expected = expected_hf_keys(hf)
    # Some tied checkpoints omit lm_head; the architecture must explicitly say
    # tied.
    tied = bool(hf.get('tie_word_embeddings', hf.get('model_type') == 'fm9g'))
    if tied and 'lm_head.weight' not in reader.weight_map:
        expected.remove('lm_head.weight')
    actual = set(reader.weight_map)
    # Rotary frequencies are derived deterministically, not trainable weights.
    derived = {key for key in actual if key.endswith('.rotary_emb.inv_freq')}
    if actual - derived != expected:
        raise ValueError(
            f'HF key mismatch: missing={sorted(expected - actual)}, '
            f'unexpected={sorted(actual - expected - derived)}'
        )
    rank, world = model.pg_collection.tp.rank(), model.pg_collection.tp.size()
    assigned = set()

    def put(name, value):
        target = model.get_parameter(name)
        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(
                f'{name}: target {tuple(target.shape)} != '
                f'source {tuple(value.shape)}'
            )
        with torch.no_grad():
            target.copy_(value)
        assigned.add(name)

    def vocab(name, source):
        target = model.get_parameter(name)
        start = rank * target.shape[0]
        stop = min(start + target.shape[0], hf['vocab_size'])
        value = torch.zeros_like(target, device='cpu')
        if start < stop:
            value[: stop - start] = reader.tensor(source, slice(start, stop))
        put(name, value)

    if model.pre_process:
        vocab('embedding.word_embeddings.weight', 'model.embed_tokens.weight')
    for local, layer in enumerate(model.decoder.layers):
        source = f'model.layers.{layer.layer_number - 1}'
        dest = f'decoder.layers.{local}'
        for target, origin in (
            ('input_layernorm', 'input_layernorm'),
            ('pre_mlp_layernorm', 'post_attention_layernorm'),
        ):
            put(
                f'{dest}.{target}.weight',
                reader.tensor(f'{source}.{origin}.weight'),
            )
        for suffix in ('weight', 'bias'):
            name = f'{dest}.self_attention.linear_qkv.{suffix}'
            if name in dict(model.named_parameters()):
                put(
                    name,
                    packed_qkv_slice(
                        reader,
                        source + '.self_attn',
                        suffix,
                        hf['num_attention_heads'],
                        hf.get(
                            'num_key_value_heads', hf['num_attention_heads']
                        ),
                        hf.get(
                            'head_dim',
                            hf['hidden_size'] // hf['num_attention_heads'],
                        ),
                        rank,
                        world,
                    ),
                )
        head_width = (
            hf.get('head_dim', hf['hidden_size'] // hf['num_attention_heads'])
            * hf['num_attention_heads']
            // world
        )
        put(
            f'{dest}.self_attention.linear_proj.weight',
            reader.tensor(
                f'{source}.self_attn.o_proj.weight',
                (
                    slice(None),
                    slice(rank * head_width, (rank + 1) * head_width),
                ),
            ),
        )
        width = hf['intermediate_size'] // world
        index = slice(rank * width, (rank + 1) * width)
        put(
            f'{dest}.mlp.linear_fc1.weight',
            torch.cat(
                [
                    reader.tensor(f'{source}.mlp.gate_proj.weight', index),
                    reader.tensor(f'{source}.mlp.up_proj.weight', index),
                ],
                dim=0,
            ),
        )
        put(
            f'{dest}.mlp.linear_fc2.weight',
            reader.tensor(
                f'{source}.mlp.down_proj.weight', (slice(None), index)
            ),
        )
    if model.post_process:
        put(
            'decoder.final_layernorm.weight',
            reader.tensor('model.norm.weight'),
        )
        if 'output_layer.weight' in dict(model.named_parameters()):
            vocab(
                'output_layer.weight',
                'model.embed_tokens.weight' if tied else 'lm_head.weight',
            )
    missing = set(dict(model.named_parameters())) - assigned
    if missing:
        raise ValueError(
            f'Uninitialized Megatron parameters: {sorted(missing)}'
        )
    model.hf_load_report = {
        'path': str(path),
        'tp_rank': rank,
        'loaded_parameters': len(assigned),
        'strict': True,
    }
    return model.hf_load_report
