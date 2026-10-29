#!/usr/bin/env python3
"""Compare actual RMSNorm backends at identical precision; JSON to stdout.

Run from lab root: PYTHONPATH=src python scripts/benchmark_ops.py Kernel
compilation and warmup are excluded. CUDA event time includes device
work between events; synchronized wall time includes Python/launch
overhead. No performance claim is inferred from a single shape or
sample.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from jiuge_ops import rms_norm


def summary(samples):
    values = sorted(samples)
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.mean(values),
        "p90_ms": values[min(len(values) - 1, int(len(values) * 0.9))],
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 1 or args.rows < 1 or args.hidden < 1:
        parser.error("rows, hidden, warmup, samples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA/CoreX GPU")
    import triton

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    dtype = getattr(torch, args.dtype)
    x = torch.randn(
        args.rows, args.hidden, device=device, dtype=dtype, requires_grad=True
    )
    w = torch.randn(
        args.hidden, device=device, dtype=dtype, requires_grad=True
    )
    g = torch.randn_like(x)
    expected = rms_norm(x, w)
    expected_grads = torch.autograd.grad(expected, (x, w), g)
    actual = rms_norm(x, w, backend="triton")
    actual_grads = torch.autograd.grad(actual, (x, w), g)
    tol = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[
        dtype
    ]
    errors = {}
    for name, a, b in zip(
        ("output", "input_grad", "weight_grad"),
        (actual, *actual_grads),
        (expected, *expected_grads),
    ):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)
        errors[name + "_max_abs"] = (a.float() - b.float()).abs().max().item()
    prop = torch.cuda.get_device_properties(device)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        *sorted(
            path
            for path in (root / "src" / "jiuge_ops").glob("*.py")
            if not path.name.startswith("._")
        ),
    ]
    result = {
        "config": vars(args),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        },
        "python": platform.python_version(),
        "device": prop.name,
        "device_memory_bytes": prop.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "precision": "FP32 accumulation; input/output/weight dtype="
        + args.dtype,
        "numerical_check": errors,
        "results": {},
    }
    for mode in ("forward", "forward_backward"):
        result["results"][mode] = {}
        for backend in ("reference", "triton"):

            def run():
                if mode == "forward":
                    with torch.no_grad():
                        return rms_norm(x, w, backend=backend)
                y = rms_norm(x, w, backend=backend)
                return torch.autograd.grad(y, (x, w), g)

            for _ in range(args.warmup):
                run()
            torch.cuda.synchronize(device)
            gpu_samples, wall_samples = [], []
            start, end = torch.cuda.Event(
                enable_timing=True
            ), torch.cuda.Event(enable_timing=True)
            for _ in range(args.samples):
                torch.cuda.synchronize(device)
                wall_start = time.perf_counter()
                start.record()
                run()
                end.record()
                end.synchronize()
                wall_samples.append((time.perf_counter() - wall_start) * 1000)
                gpu_samples.append(start.elapsed_time(end))
            result["results"][mode][backend] = {
                "cuda_event": summary(gpu_samples),
                "synchronized_wall": summary(wall_samples),
            }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
