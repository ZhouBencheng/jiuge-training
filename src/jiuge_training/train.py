"""Pinned Megatron pretrain loop with task-specific masks and model
provider.
"""

import argparse
import functools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("USE_TF", "0")
CONFIG = None


def configure_worker_caches():
    """Isolate compiler writes by global rank before importing the runtime."""
    rank = os.environ.get("RANK")
    if rank is None:
        return
    rank = int(rank)
    root = Path(__file__).resolve().parents[2] / ".cache"
    for variable, directory in (
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
    ):
        base = Path(os.environ.get(variable, str(root / directory)))
        os.environ[variable] = str(base.expanduser() / f"rank-{rank}")


def dataset_provider(num_samples):
    import torch

    cfg = CONFIG
    task, model = cfg["task"], cfg["model"]

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, name, length):
            self.length = int(length)
            self.rows = None
            if not task["synthetic_data"]:
                with (
                    Path(task["data_path"]) / (name + ".jsonl")
                ).open() as stream:
                    self.rows = [json.loads(line) for line in stream]
                if not self.rows:
                    raise ValueError(f"empty {name} dataset")

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            size = task["seq_length"]
            if self.rows is None:
                generator = torch.Generator().manual_seed(
                    task.get("seed", 1234) + index
                )
                tokens = torch.randint(
                    1, model["vocab_size"], (size + 1,), generator=generator
                )
                mask = torch.ones(size)
                if task["name"] == "lora":
                    mask[: size // 2] = 0
            else:
                row = self.rows[index % len(self.rows)]
                tokens = torch.tensor(row["tokens"], dtype=torch.long)
                mask = torch.tensor(row["loss_mask"], dtype=torch.float32)
                if (
                    tokens.numel() != size + 1
                    or mask.numel() != size
                    or mask.sum() == 0
                ):
                    raise ValueError(
                        "invalid prepared sample shape or empty loss mask"
                    )
            return {
                "tokens": tokens[:-1].clone(),
                "labels": tokens[1:].clone(),
                "loss_mask": mask,
                "position_ids": torch.arange(size, dtype=torch.long),
            }

    return (
        Dataset("train", num_samples[0]),
        Dataset("validation", max(1, num_samples[1])),
        None,
    )


def loss_func(mask, output):
    import torch

    loss = (output.float().reshape(-1) * mask.float().reshape(-1)).sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite language-model loss")
    count = mask.sum().detach().to(torch.int)
    return (
        loss,
        count,
        {"lm loss": torch.stack((loss.detach(), count.float()))},
    )


def forward_step(data_iterator, model, **kwargs):
    import torch
    from megatron.core import parallel_state as ps, tensor_parallel
    from megatron.training import get_args

    args = get_args()
    if ps.is_pipeline_first_stage() or ps.is_pipeline_last_stage():
        batch = (
            next(data_iterator)
            if ps.get_tensor_model_parallel_rank() == 0
            else None
        )
        packed = tensor_parallel.broadcast_data(
            ["tokens", "labels", "position_ids"], batch, torch.int64
        )
        masks = tensor_parallel.broadcast_data(
            ["loss_mask"], batch, torch.float32
        )
        tokens, labels, positions = (
            packed[k] for k in ("tokens", "labels", "position_ids")
        )
        mask = masks["loss_mask"]
    else:
        tokens = labels = positions = mask = None
    attention_mask = torch.triu(
        torch.ones(
            (1, 1, args.seq_length, args.seq_length),
            device="cuda",
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    output = model(tokens, positions, attention_mask, labels=labels)
    return output, functools.partial(loss_func, mask)


def adapter_training_validity(models, *, resumed):
    """Audit fresh B updates after timing; nonzero resumed B is not an
    update proof.
    """
    import torch

    total = nonzero = 0
    zero_names, nonfinite_names, trainable_base = [], [], []
    for model in models:
        for name, parameter in model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in ("lora_A", "lora_B") and parameter.requires_grad:
                trainable_base.append(name)
            if leaf == "lora_B":
                total += 1
                tensor = parameter.detach()
                if not bool(torch.isfinite(tensor).all()):
                    nonfinite_names.append(name)
                if bool(torch.count_nonzero(tensor)):
                    nonzero += 1
                else:
                    zero_names.append(name)
    passed = (
        total > 0
        and not trainable_base
        and not nonfinite_names
        and (resumed or nonzero == total)
    )
    return {
        "passed": passed,
        "resumed": resumed,
        "update_verification": (
            "not_verified_for_resume" if resumed else "all_B_nonzero_from_zero"
        ),
        "lora_B_total": total,
        "lora_B_nonzero": nonzero,
        "zero_B": zero_names,
        "nonfinite_B": nonfinite_names,
        "trainable_base_parameters": trainable_base,
    }


def main():
    global CONFIG
    configure_worker_caches()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--jiuge-config", required=True)
    ours, rest = parser.parse_known_args()
    from .config import validate

    CONFIG = validate(json.loads(Path(ours.jiuge_config).read_text()))
    from .config import validate_resume

    validate_resume(CONFIG)
    sys.argv = [sys.argv[0]] + rest
    try:
        from megatron.core.enums import ModelType
        from megatron.training import pretrain
        from megatron.training.arguments import parse_and_validate_args
        from megatron.training.argument_utils import (
            pretrain_cfg_container_from_args,
        )
        from .model import configure, model_provider
    except ImportError as exc:
        raise SystemExit(
            f"Pinned Megatron runtime dependency unavailable: {exc}. "
            "Source scripts/env.sh and use the audited vendor environment."
        ) from exc
    configure(CONFIG)
    args = parse_and_validate_args()
    cfg = pretrain_cfg_container_from_args(args, model_cfg=None)
    dataset_provider.is_distributed = False
    import torch
    import torch.distributed as distributed

    started = time.monotonic()
    completed = False
    built_models = []
    validity = None

    def tracked_provider(*args, **kwargs):
        model = model_provider(*args, **kwargs)
        built_models.append(model)
        return model

    try:
        pretrain(
            cfg,
            dataset_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            model_provider=tracked_provider,
        )
        if CONFIG["task"]["name"] == "lora":
            validity = adapter_training_validity(
                built_models, resumed=bool(CONFIG["task"].get("load"))
            )
            validity["rank"] = distributed.get_rank()
            path = (
                Path(ours.jiuge_config).parent
                / f"training-validity-rank{distributed.get_rank()}.json"
            )
            path.write_text(json.dumps(validity, indent=2) + "\n")
            print("JIUGE_ADAPTER_VALIDITY " + json.dumps(validity), flush=True)
            if not validity["passed"]:
                raise RuntimeError(
                    'LoRA training invalid: adapters did not update, '
                    'are non-finite, or base is not frozen; see '
                    'training-validity JSON'
                )
        completed = True
    finally:
        if distributed.is_initialized():
            rank = distributed.get_rank()
            metrics = {
                "rank": rank,
                "completed": completed,
                "pretrain_wall_seconds_including_load_save": time.monotonic()
                - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "adapter_validity": validity,
            }
            (
                Path(ours.jiuge_config).parent / f"metrics-rank{rank}.json"
            ).write_text(json.dumps(metrics, indent=2) + "\n")
            print("JIUGE_METRICS " + json.dumps(metrics), flush=True)
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
