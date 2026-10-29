"""Compose four explicit TOML selections and validate the training
contract.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import tempfile

MEGATRON_SHA = "5be9626709af2722333bf54797c954c09edeada3"
DEFAULT_MEGATRON = "/private/Megatron-LM"
ROOT = Path(__file__).resolve().parents[2]


def load_config(
    model, task, parallel, backend, *, config_dir=None, overrides=None
):
    root = Path(config_dir or ROOT / "configs")
    result = {}
    for section, selection in (
        ("model", model),
        ("task", task),
        ("parallel", parallel),
        ("backend", backend),
    ):
        path = root / section / (selection + ".toml")
        with path.open("rb") as stream:
            result[section] = tomllib.load(stream)
    for key, value in (overrides or {}).items():
        section, field = key.split(".", 1)
        if section not in result or (
            field not in result[section]
            and key != "task.gradient_accumulation_steps"
        ):
            raise ValueError(f"unknown override: {key}")
        result[section][field] = value
    if not result["task"].get("data_path") and not result["task"].get(
        "synthetic_data"
    ):
        result["task"]["data_path"] = (
            f"data/processed/{model}-" f"{result['task']['name']}"
        )
    for section, field in (
        ("task", "global_batch_size"),
        ("task", "micro_batch_size"),
        ("parallel", "dp"),
    ):
        if (
            type(result[section][field]) is not int
            or result[section][field] < 1
        ):
            raise ValueError(f"{section}.{field} must be a positive integer")
    if "gradient_accumulation_steps" not in result["task"]:
        divisor = result["task"]["micro_batch_size"] * result["parallel"]["dp"]
        if type(divisor) is not int or divisor <= 0:
            raise ValueError("MBS and DP must be positive integers")
        result["task"]["gradient_accumulation_steps"] = (
            result["task"]["global_batch_size"] // divisor
        )
    return validate(result)


def validate(config):
    cfg = copy.deepcopy(config)
    m, t, p, b = (cfg[k] for k in ("model", "task", "parallel", "backend"))
    for section, keys in (
        (
            m,
            (
                "layers",
                "hidden_size",
                "num_attention_heads",
                "num_query_groups",
                "ffn_hidden_size",
                "vocab_size",
            ),
        ),
        (
            t,
            (
                "seq_length",
                "micro_batch_size",
                "global_batch_size",
                "gradient_accumulation_steps",
                "train_iters",
                "save_interval",
            ),
        ),
        (p, ("dp", "tp", "pp", "nnodes", "nproc_per_node")),
    ):
        for key in keys:
            if type(section[key]) is not int or section[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
    world = p["dp"] * p["tp"] * p["pp"]
    if world != p["nnodes"] * p["nproc_per_node"]:
        raise ValueError(
            "world size must equal DP × TP × PP and nnodes × nproc_per_node"
        )
    if m["hidden_size"] % m["num_attention_heads"]:
        raise ValueError("hidden_size must be divisible by attention heads")
    if m["num_attention_heads"] % p["tp"] or m["ffn_hidden_size"] % p["tp"]:
        raise ValueError(
            "attention heads and FFN size must be divisible by TP"
        )
    if m["num_attention_heads"] % m["num_query_groups"]:
        raise ValueError("attention heads must be divisible by KV groups")
    if max(m["num_query_groups"], p["tp"]) % min(
        m["num_query_groups"], p["tp"]
    ):
        raise ValueError(
            'KV groups and TP must divide one another '
            '(replicated KV is supported)'
        )
    first = p.get("num_layers_in_first_pipeline_stage")
    last = p.get("num_layers_in_last_pipeline_stage")
    # FM9G-4B has 62 layers; PP4 is explicitly 16/15/15/16.
    if m["layers"] == 62 and p["pp"] == 4 and first is None and last is None:
        first = last = 16
        p["num_layers_in_first_pipeline_stage"] = first
        p["num_layers_in_last_pipeline_stage"] = last
    if first is None and last is None:
        if m["layers"] % p["pp"]:
            raise ValueError(
                'layers must divide PP unless first/last stage '
                'counts are explicit'
            )
    else:
        if p["pp"] <= 2 or not first or not last or first < 1 or last < 1:
            raise ValueError(
                "explicit first/last counts require PP > 2 and positive counts"
            )
        remaining = m["layers"] - first - last
        if remaining <= 0 or remaining % (p["pp"] - 2):
            raise ValueError(
                "remaining layers must divide middle pipeline stages"
            )
    if (
        t["global_batch_size"]
        != t["micro_batch_size"] * p["dp"] * t["gradient_accumulation_steps"]
    ):
        raise ValueError(
            "GBS must equal MBS × DP × gradient_accumulation_steps"
        )
    if t["gradient_accumulation_steps"] < p["pp"]:
        raise ValueError(
            "microbatch count must be at least PP for these pipeline presets"
        )
    if t["name"] not in ("cpt", "lora"):
        raise ValueError("task.name must be cpt or lora")
    if type(t.get("lr")) not in (int, float) or not t["lr"] > 0:
        raise ValueError("lr must be a positive number")
    if t["name"] == "lora":
        alpha, dropout = t.get("lora_alpha"), t.get("lora_dropout")
        if (
            type(t.get("lora_rank")) is not int
            or t["lora_rank"] < 1
            or type(alpha) not in (int, float)
            or not alpha > 0
            or type(dropout) not in (int, float)
            or not 0 <= dropout < 1
        ):
            raise ValueError(
                "LoRA requires rank > 0, alpha > 0 and 0 <= dropout < 1"
            )
    if b.get("rmsnorm") not in ("reference", "triton"):
        raise ValueError(
            "backend.rmsnorm must be reference or triton; no silent fallback"
        )
    if m.get("checkpoint_format", "hf") not in ("hf", "cpm9g-bmtrain"):
        raise ValueError("Unsupported model checkpoint_format")
    if not m.get("smoke", False) and not m.get("path"):
        raise ValueError(
            'production training requires model.path with '
            'pretrained HF weights'
        )
    if m.get("smoke", False) != t.get("synthetic_data", False):
        raise ValueError(
            "tiny smoke model and synthetic_data must be selected together"
        )
    if not t.get("synthetic_data") and not t.get("data_path"):
        raise ValueError("production training requires prepared data_path")
    return cfg


def megatron_args(cfg):
    m, t, p = (cfg[k] for k in ("model", "task", "parallel"))
    fields = {
        "num-layers": m["layers"],
        "hidden-size": m["hidden_size"],
        "num-attention-heads": m["num_attention_heads"],
        "num-query-groups": m["num_query_groups"],
        "ffn-hidden-size": m["ffn_hidden_size"],
        "seq-length": t["seq_length"],
        "max-position-embeddings": max(
            t["seq_length"], m.get("max_position_embeddings", t["seq_length"])
        ),
        "tensor-model-parallel-size": p["tp"],
        "pipeline-model-parallel-size": p["pp"],
        "micro-batch-size": t["micro_batch_size"],
        "global-batch-size": t["global_batch_size"],
        "train-iters": t["train_iters"],
        "lr": t["lr"],
        "min-lr": t["lr"],
        "lr-decay-style": "constant",
        "weight-decay": t.get("weight_decay", 0.01),
        "save": t["save"],
        "save-interval": t["save_interval"],
        "ckpt-format": "torch",
        "tokenizer-type": "NullTokenizer",
        "vocab-size": m["vocab_size"],
        "make-vocab-size-divisible-by": 1,
        "transformer-impl": "local",
        "normalization": "RMSNorm",
        "position-embedding-type": "rope",
        "attention-dropout": 0,
        "hidden-dropout": 0,
        "num-workers": 0,
        "log-interval": 1,
        "eval-iters": t.get("eval_iters", 0),
        "eval-interval": t.get("eval_interval", 100),
        "seed": t.get("seed", 1234),
        "optimizer": "adam",
        "clip-grad": 1.0,
    }
    for key, flag in (
        (
            "num_layers_in_first_pipeline_stage",
            "decoder-first-pipeline-num-layers",
        ),
        (
            "num_layers_in_last_pipeline_stage",
            "decoder-last-pipeline-num-layers",
        ),
    ):
        if key in p:
            fields[flag] = p[key]
    if t.get("load"):
        fields["load"] = t["load"]
    args = [
        item
        for key, value in fields.items()
        for item in ("--" + key, str(value))
    ]
    args += [
        "--swiglu",
        "--disable-bias-linear",
        "--group-query-attention",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-swiglu-fusion",
        "--no-bias-dropout-fusion",
        "--no-rope-fusion",
        "--no-gradient-accumulation-fusion",
        "--no-persist-layer-norm",
        "--use-cpu-initialization",
        "--attention-softmax-in-fp32",
    ]
    if t.get("load"):
        args += [
            "--exit-on-missing-checkpoint",
            "--use-checkpoint-opt-param-scheduler",
        ]
    if not m.get("tie_word_embeddings", False):
        args.append("--untie-embeddings-and-output-weights")
    if t.get("bf16", True):
        args.append("--bf16")
    if t.get("distributed_optimizer", False):
        args.append("--use-distributed-optimizer")
    if t.get("recompute", False):
        args += [
            "--recompute-granularity",
            "full",
            "--recompute-method",
            "uniform",
            "--recompute-num-layers",
            "1",
        ]
    return args


def command(
    cfg,
    config_path,
    *,
    node_rank=0,
    master_addr="127.0.0.1",
    master_port=29500,
):
    p = cfg["parallel"]
    cmd = [
        "torchrun",
        "--nnodes",
        str(p["nnodes"]),
        "--nproc-per-node",
        str(p["nproc_per_node"]),
    ]
    if p["nnodes"] == 1:
        cmd += ["--standalone"]
    else:
        cmd += [
            "--node-rank",
            str(node_rank),
            "--master-addr",
            master_addr,
            "--master-port",
            str(master_port),
        ]
    return (
        cmd
        + ["-m", "jiuge_training.train", "--jiuge-config", str(config_path)]
        + megatron_args(cfg)
    )


def runtime_environment(megatron_root=DEFAULT_MEGATRON):
    root = Path(megatron_root)
    actual = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve()}",
            "-C",
            str(root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if actual != MEGATRON_SHA:
        raise ValueError(
            f"Megatron HEAD mismatch: expected {MEGATRON_SHA}, got {actual}"
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "src"),
            str(ROOT / ".deps"),
            str(root),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        ["/usr/local/corex/lib64", env.get("LD_LIBRARY_PATH", "")]
    )
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["USE_TF"] = "0"
    return env


def checkpoint_contract(cfg):
    model, task, parallel = (cfg[k] for k in ("model", "task", "parallel"))
    model_keys = (
        "name",
        "layers",
        "hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "ffn_hidden_size",
        "vocab_size",
        "smoke",
        "tie_word_embeddings",
    )
    architecture = {key: model[key] for key in model_keys}
    if model.get("checkpoint_format", "hf") != "hf":
        architecture["checkpoint_format"] = model["checkpoint_format"]
        architecture["legacy_vocab_sha256"] = hashlib.sha256(
            (Path(model["path"]) / "vocab.txt").read_bytes()
        ).hexdigest()
    architecture["hf_config_sha256"] = (
        None
        if model["smoke"]
        else hashlib.sha256(
            (Path(model["path"]) / "config.json").read_bytes()
        ).hexdigest()
    )
    return {
        "version": 1,
        "model": architecture,
        "task": task["name"],
        "lora": (
            {
                key: task[key]
                for key in ("lora_rank", "lora_alpha", "lora_dropout")
            }
            if task["name"] == "lora"
            else None
        ),
        "bf16": task.get("bf16", True),
        "distributed_optimizer": task.get("distributed_optimizer", False),
        "parallel": {
            key: parallel.get(key)
            for key in (
                "dp",
                "tp",
                "pp",
                "num_layers_in_first_pipeline_stage",
                "num_layers_in_last_pipeline_stage",
            )
        },
        "lr": task["lr"],
        "weight_decay": task.get("weight_decay", 0.01),
    }


def check_contract(cfg, path):
    if not path.is_file():
        raise ValueError(
            f"checkpoint contract missing: {path}; refusing unverified resume"
        )
    saved, expected = json.loads(path.read_text()), checkpoint_contract(cfg)
    differences = [key for key in expected if saved.get(key) != expected[key]]
    if differences:
        raise ValueError(
            "checkpoint contract mismatch: " + ", ".join(differences)
        )


def prepare_checkpoint_directory(cfg):
    root = Path(cfg["task"]["save"])
    contract_path = root / "jiuge_contract.json"
    if contract_path.exists():
        check_contract(cfg, contract_path)
        return
    if (root / "latest_checkpointed_iteration.txt").exists():
        raise ValueError(
            "existing checkpoint has no contract; refusing to overwrite"
        )
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=root, prefix=".contract-", delete=False
    ) as stream:
        stream.write(
            json.dumps(checkpoint_contract(cfg), sort_keys=True, indent=2)
            + "\n"
        )
        temporary = Path(stream.name)
    try:
        os.link(
            temporary, contract_path
        )  # atomic publish; never replace an existing contract
    except FileExistsError:
        check_contract(cfg, contract_path)
    finally:
        temporary.unlink()


def validate_resume(cfg):
    location = cfg["task"].get("load")
    if not location:
        return
    root = Path(location)
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise ValueError(
            'resume checkpoint tracker missing; refusing to '
            'start from uninitialized weights'
        )
    check_contract(cfg, root / "jiuge_contract.json")
    value = tracker.read_text().strip()
    if not value.isdigit() or int(value) < 1:
        raise ValueError(
            "resume requires a positive training iteration checkpoint"
        )
    checkpoint = root / f"iter_{int(value):07d}"
    for tp in range(cfg["parallel"]["tp"]):
        for pp in range(cfg["parallel"]["pp"]):
            rank = f"mp_rank_{tp:02d}"
            if cfg["parallel"]["pp"] > 1:
                rank += f"_{pp:03d}"
            filenames = (
                ("model_optim_rng.pt", "distrib_optim.pt")
                if cfg["task"].get("distributed_optimizer")
                else ("model_optim_rng.pt",)
            )
            for filename in filenames:
                path = checkpoint / rank / filename
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(
                        f"resume state shard missing or empty: {path}"
                    )


def validate_local_devices(cfg, env):
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.device_count())",
        ],
        env=env,
        text=True,
    )
    available = int(output.strip().splitlines()[-1])
    requested = cfg["parallel"]["nproc_per_node"]
    if requested > available:
        raise ValueError(
            f"local device preflight: requested {requested} processes "
            f"but only {available} visible devices; multi-node configurations "
            "are plans until matching hardware is available"
        )
    return available
