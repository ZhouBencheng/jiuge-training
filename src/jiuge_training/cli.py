"""Small CLI: validate, inspect launch, prepare records, run Megatron."""

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .config import (
    ROOT,
    DEFAULT_MEGATRON,
    MEGATRON_SHA,
    command,
    load_config,
    runtime_environment,
)


def source_manifest(cfg, config_path):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    files = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted((ROOT / "src").rglob("*.py"))
        if not path.name.startswith("._")
    }
    files["requirements-bootstrap.txt"] = digest(
        ROOT / "requirements-bootstrap.txt"
    )
    environment = ROOT / "environment.lock.json"
    data_manifest = (
        None
        if cfg["task"]["synthetic_data"]
        else Path(cfg["task"]["data_path"]) / "manifest.json"
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files_sha256": files,
        "effective_config_sha256": digest(config_path),
        "megatron_commit": MEGATRON_SHA,
        "environment_lock_sha256": (
            digest(environment) if environment.is_file() else None
        ),
        "data_manifest_sha256": (
            digest(data_manifest) if data_manifest else None
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jiuge")
    sub = parser.add_subparsers(dest="action", required=True)
    data = sub.add_parser("prepare-data")
    for flag in ("source", "output", "model-path"):
        data.add_argument("--" + flag, required=True)
    data.add_argument("--task", choices=("cpt", "lora"), required=True)
    data.add_argument("--seq-length", type=int, required=True)
    data.add_argument("--seed", type=int, default=1234)
    data.add_argument("--validation-fraction", type=float, default=0.02)
    data.add_argument("--verify-alpaca", action="store_true")
    for name in ("validate", "dry-run", "train"):
        p = sub.add_parser(name)
        for flag, default in (
            ("model", "tiny"),
            ("task", "cpt-smoke"),
            ("parallel", "dp4"),
            ("backend", "reference"),
        ):
            p.add_argument("--" + flag, default=default)
        p.add_argument("--config-dir", default=str(ROOT / "configs"))
        p.add_argument(
            "--set", action="append", default=[], metavar="SECTION.KEY=JSON"
        )
        p.add_argument("--run-dir")
        p.add_argument("--megatron-root", default=DEFAULT_MEGATRON)
        p.add_argument("--node-rank", type=int, default=0)
        p.add_argument("--master-addr", default="127.0.0.1")
        p.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare-data":
            from .data import prepare_data

            print(
                json.dumps(
                    prepare_data(
                        args.source,
                        args.output,
                        args.model_path,
                        args.task,
                        args.seq_length,
                        seed=args.seed,
                        validation_fraction=args.validation_fraction,
                        verify_alpaca=args.verify_alpaca,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        overrides = {}
        for entry in args.set:
            key, value = entry.split("=", 1)
            overrides[key] = json.loads(value)
        cfg = load_config(
            args.model,
            args.task,
            args.parallel,
            args.backend,
            config_dir=args.config_dir,
            overrides=overrides,
        )
        if args.action == "validate":
            print(json.dumps(cfg, ensure_ascii=False, indent=2))
            return 0
        run_dir = Path(
            args.run_dir
            or f"runs/{args.model}-{args.task}-{args.parallel}-{args.backend}"
        ).resolve()
        if "task.save" not in overrides:
            cfg["task"]["save"] = str(run_dir / "checkpoints")
        config_path = run_dir / "config.json"
        cmd = command(
            cfg,
            config_path,
            node_rank=args.node_rank,
            master_addr=args.master_addr,
            master_port=args.master_port,
        )
        if not cfg["model"].get("smoke"):
            print(
                'PRODUCTION: shape validation does not establish '
                'memory/runtime feasibility; 32 GiB per card does '
                'not guarantee fit. Check the recorded '
                'model/layout/sequence validation results.',
                flush=True,
            )
        print(shlex.join(cmd), flush=True)
        if args.action == "dry-run":
            print(
                'Environment: source scripts/env.sh; '
                'PYTHONPATH=src:.deps:<pinned Megatron>; USE_TF=0; '
                'CUDA_DEVICE_MAX_CONNECTIONS=1'
            )
            if cfg["parallel"]["nnodes"] > 1:
                print(
                    'UNVERIFIED multi-node plan: run once per node '
                    'with distinct --node-rank and shared '
                    '--master-addr/config/checkpoint/data paths.'
                )
            return 0
        env = runtime_environment(args.megatron_root)
        from .config import validate_local_devices

        validate_local_devices(cfg, env)
        if not cfg["task"].get("synthetic_data"):
            from .data import validate_manifest

            validate_manifest(cfg)
        from .config import validate_resume, prepare_checkpoint_directory

        validate_resume(cfg)
        prepare_checkpoint_directory(cfg)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
        )
        (run_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
        manifest = source_manifest(cfg, config_path)
        (run_dir / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        return subprocess.call(cmd, env=env)
    except (
        ValueError,
        OSError,
        KeyError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"jiuge: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
