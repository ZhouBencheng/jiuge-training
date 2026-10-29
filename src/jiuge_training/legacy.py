"""Original CPM9G BMTrain configuration, weight names and vocabulary."""

import json
from pathlib import Path

LEGACY_FORMAT = 'cpm9g-bmtrain'


def model_config(raw, seq_length):
    """Translate audited legacy fields without changing the source config."""
    if raw.get('scale', True) is not False:
        raise ValueError('Legacy CPM9G requires scale=false')
    if raw.get('activate_fn', 'silu') != 'silu':
        raise ValueError('Legacy CPM9G requires SiLU gated MLP')
    if raw.get('mask_modules') is not None or raw.get('dropout_p', 0) != 0:
        raise ValueError('Legacy CPM9G requires all layers and zero dropout')
    heads = raw['num_heads']
    groups = raw.get('num_kv_heads', heads)
    if groups != heads or raw['dim_head'] * heads != raw['dim_model']:
        raise ValueError('Legacy CPM9G adapter requires standard MHA')
    return {
        'model_type': 'cpm9g_legacy',
        'checkpoint_format': LEGACY_FORMAT,
        'hidden_size': raw['dim_model'],
        'num_hidden_layers': raw['num_layers'],
        'num_attention_heads': heads,
        'num_key_value_heads': groups,
        'head_dim': raw['dim_head'],
        'intermediate_size': raw['dim_ff'],
        'vocab_size': raw['vocab_size'],
        'rms_norm_eps': raw.get('eps', 1e-5),
        'rope_theta': raw.get('base', 10000),
        'max_position_embeddings': seq_length,
        'tie_word_embeddings': False,
        'hidden_act': 'silu',
    }


def weight_names(layers):
    """Map canonical projection names to the original BMTrain state dict."""
    names = {
        'model.embed_tokens.weight': 'input_embedding.weight',
        'model.norm.weight': 'encoder.output_layernorm.weight',
        'lm_head.weight': 'lm_head.weight',
    }
    projections = {
        'input_layernorm': 'self_att.layernorm_before_attention',
        'post_attention_layernorm': 'ffn.layernorm_before_ffn',
        'self_attn.q_proj': 'self_att.self_attention.project_q',
        'self_attn.k_proj': 'self_att.self_attention.project_k',
        'self_attn.v_proj': 'self_att.self_attention.project_v',
        'self_attn.o_proj': 'self_att.self_attention.attention_out',
        'mlp.gate_proj': 'ffn.ffn.w_in.w_0',
        'mlp.up_proj': 'ffn.ffn.w_in.w_1',
        'mlp.down_proj': 'ffn.ffn.w_out',
    }
    for layer in range(layers):
        for canonical, original in projections.items():
            names[f'model.layers.{layer}.{canonical}.weight'] = (
                f'encoder.layers.{layer}.{original}.weight'
            )
    return names


class LegacyTokenizer:
    """Official longest-prefix encoding with UTF-8 byte fallback.

    The vocabulary contains JSON strings, one per physical line. Special
    tokens are excluded from ordinary matching, as in CPM9GTokenizer.
    """

    pad_token_id = None

    def __init__(self, path):
        root = Path(path)
        with (root / 'vocab.txt').open(encoding='utf-8') as stream:
            tokens = [
                json.loads(line.strip()) for line in stream if line.strip()
            ]
        if len(tokens) != len(set(tokens)):
            raise ValueError('Legacy vocabulary contains duplicate tokens')
        config = json.loads((root / 'config.json').read_text())
        if len(tokens) != config['vocab_size']:
            raise ValueError('Legacy vocabulary size differs from config')
        ids = {token: i for i, token in enumerate(tokens)}
        special = ['<unk>', '<s>', '</s>']
        byte_tokens = [f'<0x{value:02X}>' for value in range(256)]
        missing = set(special + byte_tokens) - ids.keys()
        if missing:
            raise ValueError(f'Legacy vocabulary missing tokens: {missing}')
        self.unk_token_id, self.bos_token_id, self.eos_token_id = (
            ids[token] for token in special
        )
        self.byte_ids = [ids[token] for token in byte_tokens]
        excluded = set(special + byte_tokens)
        self.encoder = {
            token: index
            for token, index in ids.items()
            if token not in excluded
        }
        self.max_lengths = {}
        for token in self.encoder:
            if not token:
                raise ValueError('Legacy vocabulary contains an empty token')
            self.max_lengths[token[0]] = max(
                self.max_lengths.get(token[0], 0), len(token)
            )
        self.vocab_size = len(tokens)

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError(
                'Legacy encoding requires explicit special tokens'
            )
        # The upstream encoder substitutes unknown IDs beyond this boundary.
        # Reject the record so training cannot silently consume that fallback.
        if len(text) > 20480:
            raise ValueError('Legacy tokenizer input exceeds 20480 characters')
        result, start = [], 0
        while start < len(text):
            end = min(len(text), start + self.max_lengths.get(text[start], 0))
            while end > start and text[start:end] not in self.encoder:
                end -= 1
            if end > start:
                result.append(self.encoder[text[start:end]])
                start = end
            else:
                result.extend(
                    self.byte_ids[b] for b in text[start].encode('utf-8')
                )
                start += 1
        return result
