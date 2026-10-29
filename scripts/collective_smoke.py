#!/usr/bin/env python3
"""Short four-rank NCCL collective and point-to-point validation."""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

import torch
import torch.distributed as dist


def main() -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 4:
        raise RuntimeError(
            f"expected four torchrun ranks, received {world_size}"
        )
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"expected at least {world_size} visible devices, "
            f"got {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = int(os.environ["RANK"])
    started = time.monotonic()
    checks: list[str] = []
    initialized = False
    try:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=30),
        )
        initialized = True

        reduced = torch.tensor([rank + 1], dtype=torch.int64, device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        if reduced.item() != 10:
            raise AssertionError(
                f"all_reduce expected 10, got {reduced.item()}"
            )
        checks.append("all_reduce")

        local = torch.tensor([rank], dtype=torch.int64, device=device)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        values = [int(value.item()) for value in gathered]
        if values != [0, 1, 2, 3]:
            raise AssertionError(
                f"all_gather expected [0, 1, 2, 3], got {values}"
            )
        checks.append("all_gather")

        if rank == 0:
            payload = torch.tensor(
                [20261029], dtype=torch.int64, device=device
            )
            dist.send(payload, dst=1, tag=17)
        elif rank == 1:
            received = torch.empty([1], dtype=torch.int64, device=device)
            dist.recv(received, src=0, tag=17)
            if received.item() != 20261029:
                raise AssertionError(
                    f"send/recv payload mismatch: {received.item()}"
                )
        checks.append("send_recv")

        dist.barrier()
        torch.cuda.synchronize(device)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "backend": "nccl",
                        "world_size": world_size,
                        "devices": [
                            torch.cuda.get_device_name(i)
                            for i in range(world_size)
                        ],
                        "checks": checks,
                        "elapsed_seconds": round(
                            time.monotonic() - started, 3
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": type(exc).__name__,
                    "detail": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
