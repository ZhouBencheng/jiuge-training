"""RMSNorm seam: y = (x / sqrt(mean(x**2) + eps)) * weight.

Variance uses FP32; normalized activations round to x.dtype before
multiplying the weight, matching the supplied FM9G implementation.
Output has x.dtype. The last dimension is normalized. Weights may have
the input dtype or FP32, on the same device. Triton supports first-order
autograd only and never falls back to PyTorch.
"""

import math

import torch
from torch import nn

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def rms_norm(x, weight, eps=1e-6, backend="reference"):
    if x.ndim < 1 or x.shape[-1] == 0 or x.numel() == 0:
        raise ValueError(
            "RMSNorm requires a nonempty input with a hidden dimension"
        )
    if weight.shape != (x.shape[-1],):
        raise ValueError("RMSNorm weight must match the last input dimension")
    if x.dtype not in _DTYPES or weight.dtype not in (x.dtype, torch.float32):
        raise TypeError(
            'RMSNorm supports FP16/BF16/FP32 input and '
            'same-dtype or FP32 weight'
        )
    if x.device != weight.device:
        raise ValueError("RMSNorm input and weight must share a device")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm eps must be finite and positive")
    if backend == "reference":
        xf = x.float()
        normalized = (
            x * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        ).to(x.dtype)
        return (normalized * weight).to(x.dtype)
    if backend == "triton":
        if x.device.type != "cuda":
            raise RuntimeError("Triton RMSNorm requires a CUDA/CoreX device")
        try:
            from ._triton import triton_rms_norm
        except ImportError as exc:
            raise RuntimeError(
                "Triton RMSNorm was requested but Triton is unavailable"
            ) from exc
        return triton_rms_norm(x, weight, eps)
    raise ValueError(f"Unknown RMSNorm backend: {backend!r}")


class ReferenceRMSNorm(nn.Module):
    """Megatron ModuleSpec Adapter: (config, hidden_size, eps) ->
    forward(x).

    Parameters start on CPU unless device is supplied, like
    WrappedTorchNorm. Megatron may subsequently move/cast the module. No
    persistent/fused-norm promise is made. Sequence-parallel gradient
    reduction remains Megatron's job.
    """

    backend = "reference"

    def __init__(
        self, hidden_size, eps=1e-6, config=None, *, device=None, dtype=None
    ):
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if config is not None:
            if getattr(config, "normalization", "RMSNorm") != "RMSNorm":
                raise ValueError("jiuge_ops adapters only implement RMSNorm")
            if getattr(config, "layernorm_zero_centered_gamma", False):
                raise ValueError("zero-centered gamma is not supported")
            if getattr(config, "memory_efficient_layer_norm", False):
                raise ValueError(
                    "memory_efficient_layer_norm is not supported"
                )
        dtype = dtype or getattr(config, "params_dtype", torch.float32)
        if dtype not in _DTYPES:
            raise TypeError("RMSNorm parameter dtype must be FP16/BF16/FP32")
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        self.weight.sequence_parallel = getattr(
            config, "sequence_parallel", False
        )

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps, self.backend)


class TritonRMSNorm(ReferenceRMSNorm):
    """Explicit Triton Adapter; compilation/runtime incompatibilities
    propagate.
    """

    backend = "triton"
