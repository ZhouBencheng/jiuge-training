"""Private Triton kernels. FP32 reductions, no atomics, no global patching."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _forward(X, W, Y, R, N: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    w = tl.load(W + c, c < N, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    tl.store(R + row, r)
    h = (x * r).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * N + c, h * w, c < N)


@triton.jit
def _backward_x(X, W, G, R, DX, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    w = tl.load(W + c, c < N, 0).to(tl.float32)
    g = tl.load(G + row * N + c, c < N, 0).to(tl.float32)
    r = tl.load(R + row)
    h = x * r
    gw = (g * w).to(X.dtype.element_ty).to(tl.float32)
    direct = (r * gw).to(X.dtype.element_ty).to(tl.float32)
    variance = (
        (-r * h * tl.sum(h * gw, axis=0) / N)
        .to(X.dtype.element_ty)
        .to(tl.float32)
    )
    dx = direct + variance
    tl.store(DX + row * N + c, dx, c < N)


@triton.jit
def _weight_products(
    X,
    G,
    R,
    PRODUCTS,
    M: tl.constexpr,
    N: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    group = tl.program_id(0)
    rows = group * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(X + rows[:, None] * N + cols[None, :], mask, 0).to(tl.float32)
    g = tl.load(G + rows[:, None] * N + cols[None, :], mask, 0).to(tl.float32)
    r = tl.load(R + rows, rows < M, 0)
    h = (x * r[:, None]).to(X.dtype.element_ty).to(tl.float32)
    # Materialize in weight.dtype: CoreX loses an intermediate 16-bit product
    # rounding when multiply+extend+reduce are kept in the same kernel.
    tl.store(PRODUCTS + rows[:, None] * N + cols[None, :], g * h, mask)


@triton.jit
def _backward_weight_partial(
    PRODUCTS,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    group = tl.program_id(0)
    rows = group * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    product = tl.load(
        PRODUCTS + rows[:, None] * N + cols[None, :], mask, 0
    ).to(tl.float32)
    partial = tl.sum(product, axis=0)
    tl.store(P + group * N + cols, partial, cols < N)


@triton.jit
def _backward_weight_reduce(
    P,
    DW,
    GROUPS: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.arange(0, B)
    cols = tl.program_id(0) * BC + tl.arange(0, BC)
    p = tl.load(
        P + rows[:, None] * N + cols[None, :],
        (rows[:, None] < GROUPS) & (cols[None, :] < N),
        0,
    )
    tl.store(DW + cols, tl.sum(p, axis=0), cols < N)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        weight = weight.contiguous()
        n = x.shape[-1]
        m = x.numel() // n
        if n > 16384 or m > 65536:
            raise ValueError(
                'Triton RMSNorm lab limit: hidden_size <= 16384 '
                'and rows <= 65536'
            )
        y = torch.empty_like(x)
        r = torch.empty(m, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(n)
        warps = min(8, max(1, block // 256))
        with torch.cuda.device(x.device):
            _forward[(m,)](x, weight, y, r, n, eps, block, num_warps=warps)
        ctx.save_for_backward(x, weight, r)
        ctx.block, ctx.warps = block, warps
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight, r = ctx.saved_tensors
        g = grad_output.contiguous()
        n = x.shape[-1]
        m = x.numel() // n
        dx = dw = None
        with torch.cuda.device(x.device):
            if ctx.needs_input_grad[0]:
                dx = torch.empty_like(x)
                _backward_x[(m,)](
                    x, weight, g, r, dx, n, ctx.block, num_warps=ctx.warps
                )
            if ctx.needs_input_grad[1]:
                groups = triton.cdiv(m, 32)
                products = torch.empty(
                    x.shape, device=x.device, dtype=weight.dtype
                )
                partial = torch.empty(
                    (groups, n), device=x.device, dtype=torch.float32
                )
                dw = torch.empty_like(weight)
                _weight_products[(groups, triton.cdiv(n, 64))](
                    x, g, r, products, m, n, 32, 64, num_warps=4
                )
                _backward_weight_partial[(groups, triton.cdiv(n, 64))](
                    products, partial, m, n, 32, 64, num_warps=4
                )
                _backward_weight_reduce[(triton.cdiv(n, 32),)](
                    partial,
                    dw,
                    groups,
                    n,
                    triton.next_power_of_2(groups),
                    32,
                    num_warps=4,
                )
        return dx, dw, None


def triton_rms_norm(x, weight, eps):
    return _RMSNorm.apply(x, weight, eps)
