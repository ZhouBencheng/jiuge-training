"""FM9G adapters for the pinned Megatron Core GPTModel implementation."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

_settings = None


class HFCompatibleAttention(torch.nn.Module):
    """Local PyTorch attention preserving HF's BF16 matmul-then-scale
    order.

    Scores materialize as S²; this is the verified reference backend,
    not flash attention.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.dropout = config.attention_dropout

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        **kwargs,
    ):
        if attention_bias is not None or packed_seq_params is not None:
            raise ValueError(
                'Reference attention does not support packed '
                'sequences or attention bias'
            )
        # FM9G 4B/8B rotate Q/K in FP32, then round once before attention GEMM.
        query, key = query.to(value.dtype), key.to(value.dtype)
        q, k, v = (
            tensor.permute(1, 2, 0, 3) for tensor in (query, key, value)
        )
        repeats = q.shape[1] // k.shape[1]
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(
                    scores.shape[-2:], device=scores.device, dtype=torch.bool
                ),
                diagonal=1,
            )
        scores = scores.masked_fill(
            attention_mask, torch.finfo(scores.dtype).min
        )
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        probabilities = F.dropout(
            probabilities, p=self.dropout, training=self.training
        )
        context = (
            torch.matmul(probabilities, v).permute(2, 0, 1, 3).contiguous()
        )
        return context.flatten(2)


class LegacyAttention(torch.nn.Module):
    """CPM9GTorch eager attention after Q/K scaling and low-precision RoPE."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.dropout = config.attention_dropout

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        **kwargs,
    ):
        if attention_bias is not None or packed_seq_params is not None:
            raise ValueError('Legacy attention requires ordinary causal input')
        q, k, v = (
            tensor.permute(1, 2, 0, 3) for tensor in (query, key, value)
        )
        scores = torch.matmul(q, k.transpose(-1, -2))
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(
                    scores.shape[-2:], device=scores.device, dtype=torch.bool
                ),
                diagonal=1,
            )
        scores = scores.masked_fill(attention_mask, float('-inf'))
        probabilities = F.softmax(scores, dim=-1)
        probabilities = probabilities.masked_fill(attention_mask, 0)
        probabilities = F.dropout(
            probabilities, p=self.dropout, training=self.training
        )
        context = torch.matmul(probabilities, v).permute(2, 0, 1, 3)
        return context.contiguous().flatten(2)


def configure(settings):
    global _settings
    _settings = settings


def scaled_bias_dropout_add(scale):
    def factory(training, fused):
        def apply(x_with_bias, residual, prob):
            x, bias = x_with_bias
            if bias is not None:
                x = x + bias
            return residual + F.dropout(x, p=prob, training=training) * scale

        return apply

    return factory


def model_provider(
    pre_process=True,
    post_process=True,
    config=None,
    pg_collection=None,
    **kwargs,
):
    """Build this PP stage, load its HF/TP shard, then optionally attach
    LoRA.
    """
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from megatron.core.transformer.attention import (
        SelfAttention,
        SelfAttentionSubmodules,
    )
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_block import (
        TransformerBlockSubmodules,
        get_num_layers_to_build,
    )
    from megatron.core.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerSubmodules,
    )
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args
    from jiuge_ops import ReferenceRMSNorm, TritonRMSNorm

    if _settings is None:
        raise RuntimeError('Call configure(settings) before model_provider')
    settings = _settings
    model_settings, task = settings['model'], settings['task']
    smoke = model_settings.get('smoke', False)
    if smoke:
        hf = dict(
            model_type='fm9g',
            hidden_size=model_settings['hidden_size'],
            num_hidden_layers=model_settings['layers'],
            num_attention_heads=model_settings['num_attention_heads'],
            num_key_value_heads=model_settings['num_query_groups'],
            intermediate_size=model_settings['ffn_hidden_size'],
            vocab_size=model_settings['vocab_size'],
            rms_norm_eps=1e-5,
            rope_theta=1000000.0,
            max_position_embeddings=task['seq_length'],
            scale_emb=12.0,
            scale_depth=1.4,
            dim_model_base=256,
            tie_word_embeddings=model_settings.get(
                'tie_word_embeddings', False
            ),
        )
    else:
        hf = json.loads(
            (Path(model_settings['path']) / 'config.json').read_text()
        )
        if model_settings.get('checkpoint_format') == 'cpm9g-bmtrain':
            from .legacy import model_config

            hf = model_config(hf, task['seq_length'])
        # The supplied FM9G configuration class defaults this to True (4B omits
        # it).
        hf.setdefault('tie_word_embeddings', hf.get('model_type') == 'fm9g')
        dimensions = dict(
            layers=hf['num_hidden_layers'],
            hidden_size=hf['hidden_size'],
            num_attention_heads=hf['num_attention_heads'],
            num_query_groups=hf.get(
                'num_key_value_heads', hf['num_attention_heads']
            ),
            ffn_hidden_size=hf['intermediate_size'],
            vocab_size=hf['vocab_size'],
        )
        for name, actual in dimensions.items():
            if name in model_settings and model_settings[name] != actual:
                raise ValueError(
                    f'Model preset {name} differs from HF checkpoint: '
                    f'{model_settings[name]} != {actual}'
                )
    if (
        'tie_word_embeddings' in model_settings
        and model_settings['tie_word_embeddings'] != hf['tie_word_embeddings']
    ):
        raise ValueError(
            'Selected embedding tying differs from the actual HF architecture'
        )
    if hf.get('model_type') not in ('fm9g', 'fm9g7b', 'cpm9g_legacy'):
        raise ValueError('Only the audited FM9G model families are supported')
    if (
        hf.get('hidden_act', 'silu') != 'silu'
        or hf.get('mlp_bias', False)
        or hf.get('use_sliding_window', False)
    ):
        raise ValueError(
            'This adapter supports dense SiLU gated MLP '
            'without MLP bias or sliding window'
        )
    cfg = config or core_transformer_config_from_args(get_args())
    if not smoke:
        # Every parameter is supplied by HF or resume; avoid initializing
        # billions of values.
        cfg.perform_initialization = False
    for key, value in dict(
        num_layers=hf['num_hidden_layers'],
        hidden_size=hf['hidden_size'],
        num_attention_heads=hf['num_attention_heads'],
        num_query_groups=hf.get(
            'num_key_value_heads', hf['num_attention_heads']
        ),
        ffn_hidden_size=hf['intermediate_size'],
        kv_channels=hf.get(
            'head_dim', hf['hidden_size'] // hf['num_attention_heads']
        ),
        normalization='RMSNorm',
        layernorm_epsilon=hf['rms_norm_eps'],
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=(
            hf.get('model_type') == 'fm9g7b' or hf.get('attention_bias', False)
        ),
        attention_dropout=hf.get('attention_dropout', 0.0),
        hidden_dropout=0.0,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        masked_softmax_fusion=False,
        apply_rope_fusion=False,
        attention_softmax_in_fp32=True,
        rotary_interleaved=False,
        apply_query_key_layer_scaling=False,
        softmax_scale=None,
    ).items():
        setattr(cfg, key, value)
    if cfg.context_parallel_size != 1 or cfg.sequence_parallel:
        raise ValueError('Adapter supports CP=1 and sequence_parallel=False')
    if (
        hf['num_attention_heads'] % cfg.tensor_model_parallel_size
        or hf['intermediate_size'] % cfg.tensor_model_parallel_size
    ):
        raise ValueError('Attention heads and FFN width must divide TP')
    norm = {'reference': ReferenceRMSNorm, 'triton': TritonRMSNorm}[
        settings['backend']['rmsnorm']
    ]

    class FP32RotarySelfAttention(SelfAttention):
        def get_query_key_value_tensors(self, *args, **kwargs):
            query, key, value = super().get_query_key_value_tensors(
                *args, **kwargs
            )
            return query.float(), key.float(), value

    attention = (
        FP32RotarySelfAttention
        if hf.get('model_type') == 'fm9g'
        else SelfAttention
    )

    class LegacySelfAttention(SelfAttention):
        def get_query_key_value_tensors(self, *args, **kwargs):
            query, key, value = super().get_query_key_value_tensors(
                *args, **kwargs
            )
            divisor = math.sqrt(math.sqrt(query.shape[-1]))
            return query / divisor, key / divisor, value

    legacy = hf.get('model_type') == 'cpm9g_legacy'
    if legacy:
        attention = LegacySelfAttention
    scale = hf.get('scale_depth', math.sqrt(cfg.num_layers)) / math.sqrt(
        cfg.num_layers
    )
    bda = scaled_bias_dropout_add(scale)
    layer_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=norm,
            self_attention=ModuleSpec(
                module=attention,
                params={'attn_mask_type': AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=(
                        LegacyAttention if legacy else HFCompatibleAttention
                    ),
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=bda,
            pre_mlp_layernorm=norm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                ),
            ),
            mlp_bda=bda,
        ),
    )
    pp_rank = (
        pg_collection.pp.rank()
        if pg_collection is not None
        else parallel_state.get_pipeline_model_parallel_rank()
    )
    block_spec = TransformerBlockSubmodules(
        layer_specs=[layer_spec]
        * get_num_layers_to_build(cfg, pp_rank=pp_rank),
        layer_norm=norm,
    )
    tp = cfg.tensor_model_parallel_size
    padded_vocab = math.ceil(hf['vocab_size'] / tp) * tp
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=block_spec,
        vocab_size=padded_vocab,
        max_sequence_length=hf['max_position_embeddings'],
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
        share_embeddings_and_output_weights=hf.get(
            'tie_word_embeddings', False
        ),
        position_embedding_type='rope',
        rotary_base=hf['rope_theta'],
        pg_collection=pg_collection,
    )
    # Local hooks preserve standard state_dict keys and Megatron's PP
    # interface.
    if pre_process:

        def scale_embedding(module, args, output):
            output = output * hf.get('scale_emb', 1.0)
            if (
                task['name'] == 'lora'
                and cfg.recompute_granularity == 'full'
                and module.training
                and torch.is_grad_enabled()
            ):
                # Reentrant checkpoint inputs must carry an autograd edge even
                # though the embedding table is frozen. This is an activation,
                # not a trainable parameter or an unfreezing of the base model.
                output.requires_grad_(True)
            return output

        model.embedding.register_forward_hook(scale_embedding)
    if post_process:
        divisor = hf['hidden_size'] / hf.get(
            'dim_model_base', hf['hidden_size']
        )
        model.output_layer.register_forward_pre_hook(
            lambda module, args: (args[0] / divisor, *args[1:])
        )
        start = model.pg_collection.tp.rank() * (padded_vocab // tp)

        def mask_padding(module, args, output):
            logits, bias = output
            offset = 0 if logits.shape[-1] == padded_vocab else start
            ids = torch.arange(logits.shape[-1], device=logits.device) + offset
            return (
                logits.masked_fill(ids >= hf['vocab_size'], float('-inf')),
                bias,
            )

        model.output_layer.register_forward_hook(mask_padding)
    rope = hf.get('rope_scaling')
    if rope:
        if rope.get('rope_type') != 'longrope':
            raise ValueError(f'Unsupported RoPE scaling: {rope}')
        original = rope['original_max_position_embeddings']
        # This checkpoint has amplitude=1 and equal factors. Do not silently
        # generalize.
        if (
            hf['max_position_embeddings'] != original
            or rope['short_factor'] != rope['long_factor']
        ):
            raise ValueError(
                'LongRoPE requires amplitude=1 and equal short/long factors'
            )
        factors = torch.tensor(
            rope['short_factor'],
            dtype=torch.float32,
            device=model.rotary_pos_emb.inv_freq.device,
        )
        if factors.shape != model.rotary_pos_emb.inv_freq.shape or torch.any(
            factors <= 0
        ):
            raise ValueError('Invalid LongRoPE factors')
        model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq / factors
    model.hf_config = hf
    if not smoke and not task.get('load'):
        from .weights import load_hf_weights

        load_hf_weights(model, model_settings['path'], hf)
    if task['name'] == 'lora':
        from .lora import attach_lora

        attach_lora(
            model,
            rank=task['lora_rank'],
            alpha=task['lora_alpha'],
            dropout=task['lora_dropout'],
            seed=task.get('seed', 1234),
        )
    return model
