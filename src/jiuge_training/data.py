"""Deterministic record splitting and explicit next-token loss masks."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random

ALPACA_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)


def format_record(record):
    instruction = record["instruction"].strip()
    context = record.get("input", "").strip()
    prompt = "### 指令：\n" + instruction + "\n"
    if context:
        prompt += "### 输入：\n" + context + "\n"
    return prompt + "### 回答：\n", record["output"].strip()


def split_records(records, seed=1234, validation_fraction=0.02):
    if len(records) < 2 or not 0 < validation_fraction < 1:
        raise ValueError(
            "at least two records and 0 < validation_fraction < 1 required"
        )
    # Keep identical examples in one split; no train/validation duplication.
    groups = {}
    for record in records:
        key = json.dumps(record, ensure_ascii=False, sort_keys=True)
        groups.setdefault(key, record)
    unique = list(groups.values())
    if len(unique) < 2:
        raise ValueError("at least two distinct records required")
    random.Random(seed).shuffle(unique)
    n = max(1, min(len(unique) - 1, round(len(unique) * validation_fraction)))
    return unique[n:], unique[:n]


def encode_records(records, tokenizer, seq_length, task):
    if type(seq_length) is not int or seq_length < 1:
        raise ValueError("seq_length must be a positive integer")
    if task not in ("cpt", "lora"):
        raise ValueError("task must be cpt or lora")
    if tokenizer.eos_token_id is None:
        raise ValueError("model tokenizer must define eos_token_id")
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    buffer = []
    for record in records:
        prompt, response = format_record(record)
        if task == "cpt":
            # Explicit demonstration corpus: instruction, input, response, EOS.
            buffer.extend(
                tokenizer.encode(prompt + response, add_special_tokens=False)
                + [eos]
            )
            while len(buffer) >= seq_length + 1:
                yield {
                    "tokens": buffer[: seq_length + 1],
                    "loss_mask": [1] * seq_length,
                }
                buffer = buffer[seq_length:]  # keep the next-token overlap
        else:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(
                response, add_special_tokens=False
            ) + [eos]
            # Keep full answer where possible; truncate prompt from its left.
            answer_ids = answer_ids[:seq_length]
            answer_ids[-1] = eos
            prompt_ids = prompt_ids[
                -max(1, seq_length + 1 - len(answer_ids)) :
            ]
            ids = prompt_ids + answer_ids
            mask = [int(i >= len(prompt_ids)) for i in range(1, len(ids))]
            ids += [pad] * (seq_length + 1 - len(ids))
            mask += [0] * (seq_length - len(mask))
            yield {"tokens": ids, "loss_mask": mask}
    if task == "cpt" and len(buffer) > 1:
        count = len(buffer) - 1
        yield {
            "tokens": buffer + [pad] * (seq_length + 1 - len(buffer)),
            "loss_mask": [1] * count + [0] * (seq_length - count),
        }


def tokenizer_fingerprints(model_path):
    root = Path(model_path)
    names = set()
    for pattern in (
        "config.json",
        "tokenizer*",
        "special_tokens*",
        "added_tokens*",
        "vocab*",
        "merges*",
        "tokenization*.py",
    ):
        names.update(path for path in root.glob(pattern) if path.is_file())
    if not names or not (root / "config.json").is_file():
        raise ValueError(
            "model directory must contain config.json and tokenizer assets"
        )
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(names)
    }


def validate_manifest(config):
    task, model = config["task"], config["model"]
    root = Path(task["data_path"])
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["task"] != task["name"]
        or manifest["seq_length"] != task["seq_length"]
        or Path(manifest["model_path"]).resolve()
        != Path(model["path"]).resolve()
    ):
        raise ValueError(
            'prepared data manifest task/sequence/tokenizer '
            'model does not match training config'
        )
    if manifest.get("tokenizer_files_sha256") != tokenizer_fingerprints(
        model["path"]
    ):
        raise ValueError(
            "tokenizer/model config fingerprints changed; prepare data again"
        )
    if manifest.get("transformers_version") != version("transformers"):
        raise ValueError("transformers version changed; prepare data again")
    for name in ("train", "validation"):
        digest = hashlib.sha256()
        with (root / (name + ".jsonl")).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest["splits"][name]["sha256"]:
            raise ValueError(f"prepared {name} data SHA256 mismatch")
    return manifest


def prepare_data(
    source,
    output,
    model_path,
    task,
    seq_length,
    *,
    seed=1234,
    validation_fraction=0.02,
    verify_alpaca=False,
):
    if type(seq_length) is not int or seq_length < 1:
        raise ValueError("seq_length must be a positive integer")
    from transformers import AutoTokenizer

    source = Path(source)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if verify_alpaca and digest != ALPACA_SHA256:
        raise ValueError(
            f"official pinned Alpaca source SHA256 mismatch: {digest}"
        )
    records = json.loads(raw)
    train, valid = split_records(records, seed, validation_fraction)
    raw_config = json.loads((Path(model_path) / 'config.json').read_text())
    if 'dim_model' in raw_config and 'model_type' not in raw_config:
        from .legacy import LegacyTokenizer, model_config

        model_config(raw_config, seq_length)
        tokenizer = LegacyTokenizer(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    fingerprints = tokenizer_fingerprints(model_path)
    for name, rows in (("train", train), ("validation", valid)):
        count = supervised = 0
        file_digest = hashlib.sha256()
        with (output / (name + ".jsonl")).open("w") as stream:
            for sample in encode_records(rows, tokenizer, seq_length, task):
                line = json.dumps(sample, separators=(",", ":")) + "\n"
                stream.write(line)
                file_digest.update(line.encode("utf-8"))
                count += 1
                supervised += sum(sample["loss_mask"])
        if not count or not supervised:
            raise ValueError(f"empty {name} dataset")
        counts[name] = {
            "records": len(rows),
            "samples": count,
            "loss_tokens": supervised,
            "sha256": file_digest.hexdigest(),
        }
    manifest = {
        "source": str(source),
        "source_sha256": digest,
        "tokenizer_files_sha256": fingerprints,
        "transformers_version": version("transformers"),
        "model_path": str(model_path),
        "task": task,
        "seq_length": seq_length,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "source_records": len(records),
        "splits": counts,
        ('format'): (
            'instruction/input/response headings; EOS between '
            'CPT records; response+EOS only for LoRA'
        ),
        ('purpose'): (
            'Alpaca is a pipeline reproduction corpus, not '
            'evidence of domain CPT quality.'
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest
