#!/usr/bin/env python3
"""Emit a bounded, non-sensitive JSON report for the local CoreX training
runtime.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

COREX_LINK = Path("/usr/local/corex")
PACKAGE_NAMES = (
    "torch",
    "triton",
    "transformers",
    "peft",
    "accelerate",
    "flash-attn",
    "bitsandbytes",
    "apex",
    "megatron-core",
    "deepspeed",
    "transformer-engine",
)


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def megatron_revisions() -> dict[str, str | None]:
    source = Path("/private/Megatron-LM")
    checked_out = None
    if source.is_dir():
        try:
            checked_out = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={source.resolve()}",
                    "-C",
                    str(source),
                    "rev-parse",
                    "HEAD",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    return {
        "source_path": str(source),
        "declared_fixed_head": "5be9626709af2722333bf54797c954c09edeada3",
        "checked_out_head": checked_out,
    }


def main() -> int:
    report: dict[str, object] = {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version.split()[0],
        },
        "corex": {
            "path": str(COREX_LINK),
            "resolved_path": (
                str(COREX_LINK.resolve()) if COREX_LINK.exists() else None
            ),
            "cuda_compatibility": None,
        },
        "packages": package_versions(),
        "megatron": megatron_revisions(),
        "driver_available": False,
        "device_count": 0,
        "devices": [],
        "errors": [],
    }

    try:
        import torch

        report["torch"] = torch.__version__
        report[('corex')][
            ('cuda_compatibility')
        ] = torch.version.cuda  # type: ignore[index]
        available = bool(torch.cuda.is_available())
        report["driver_available"] = available
        if available:
            count = torch.cuda.device_count()
            report["device_count"] = count
            report["devices"] = [
                {"index": i, "name": torch.cuda.get_device_name(i)}
                for i in range(count)
            ]
    except Exception as exc:  # Keep diagnostics bounded and omit host paths.
        report[('errors')].append(
            f"torch_probe: {type(exc).__name__}"
        )  # type: ignore[union-attr]

    try:
        smi = subprocess.run(
            ["/usr/local/iluvatar/bin/ixsmi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        match = re.search(r"Driver Version:\s*([^\s|]+)", smi.stdout)
        report["driver_version"] = match.group(1) if match else None
        report["ixsmi_exit_code"] = smi.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        report["driver_version"] = None
        report[('errors')].append(
            f"ixsmi_probe: {type(exc).__name__}"
        )  # type: ignore[union-attr]

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["driver_available"] and report["device_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
