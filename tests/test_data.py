import unittest
import tempfile
import json
import hashlib
from pathlib import Path
from unittest.mock import patch

from jiuge_training.data import split_records, encode_records, format_record


class Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 2 for c in text]


class DataTests(unittest.TestCase):
    def test_split_reproducible_disjoint_deduplicated(self):
        records = [
            {'instruction': str(i), 'output': str(i)} for i in range(10)
        ]
        a, b = split_records(records + records[:2], 42, 0.2)
        self.assertEqual((a, b), split_records(records + records[:2], 42, 0.2))
        self.assertEqual(len(a) + len(b), 10)
        self.assertFalse(
            {r['instruction'] for r in a} & {r['instruction'] for r in b}
        )

    def test_sft_only_answer_and_eos(self):
        row = {'instruction': 'x', 'input': '', 'output': 'yes'}
        sample = list(encode_records([row], Tokenizer(), 64, 'lora'))[0]
        labels = sample['tokens'][1:]
        supervised = [v for v, m in zip(labels, sample['loss_mask']) if m]
        self.assertEqual(supervised, Tokenizer().encode('yes') + [1])
        self.assertEqual(len(sample['tokens']), 65)
        self.assertEqual(len(sample['loss_mask']), 64)

    def test_truncation_keeps_eos_and_supervision(self):
        row = {'instruction': 'long' * 30, 'output': 'answer' * 30}
        sample = list(encode_records([row], Tokenizer(), 8, 'lora'))[0]
        self.assertEqual(len(sample['tokens']), 9)
        self.assertEqual(sample['tokens'][-1], 1)
        self.assertEqual(sum(sample['loss_mask']), 8)

    def test_rejects_invalid_sequence_length(self):
        from jiuge_training.data import prepare_data

        for length in (0, -1, 1.5, True):
            for task in ("cpt", "lora"):
                with self.assertRaisesRegex(ValueError, "seq_length"):
                    list(
                        encode_records(
                            [{"instruction": "x", "output": "y"}],
                            Tokenizer(),
                            length,
                            task,
                        )
                    )
                with self.assertRaisesRegex(ValueError, "seq_length"):
                    prepare_data("missing", "unused", "missing", task, length)

    def test_manifest_rejects_changed_tokenizer_or_data(self):
        from jiuge_training.data import (
            tokenizer_fingerprints,
            validate_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "tokenizer.json").write_text("original")
            splits = {}
            for name in ("train", "validation"):
                raw = b'{"tokens":[1,2],"loss_mask":[1]}\n'
                (root / (name + ".jsonl")).write_bytes(raw)
                splits[name] = {"sha256": hashlib.sha256(raw).hexdigest()}
            manifest = {
                "task": "cpt",
                "seq_length": 1,
                "model_path": str(model),
                "tokenizer_files_sha256": tokenizer_fingerprints(model),
                "transformers_version": "test",
                "splits": splits,
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            cfg = {
                "task": {
                    "name": "cpt",
                    "seq_length": 1,
                    "data_path": str(root),
                },
                "model": {"path": str(model)},
            }
            with patch("jiuge_training.data.version", return_value="test"):
                validate_manifest(cfg)
                (model / "tokenizer.json").write_text("changed")
                with self.assertRaisesRegex(ValueError, "fingerprints"):
                    validate_manifest(cfg)
                (model / "tokenizer.json").write_text("original")
                (root / "train.jsonl").write_text("changed")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    validate_manifest(cfg)

    def test_cpt_every_real_target_no_pad(self):
        row = {'instruction': 'x', 'output': 'y'}
        expected = Tokenizer().encode(''.join(format_record(row))) + [1]
        samples = list(encode_records([row], Tokenizer(), 8, 'cpt'))
        targets = [
            v
            for sample in samples
            for v, m in zip(sample['tokens'][1:], sample['loss_mask'])
            if m
        ]
        self.assertEqual(targets, expected[1:])


if __name__ == '__main__':
    unittest.main()
