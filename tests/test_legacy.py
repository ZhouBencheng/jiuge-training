"""Legacy CPM9G config, byte-fallback tokenization and checkpoint mapping."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from jiuge_training.config import (
    checkpoint_contract,
    check_contract,
    load_config,
    megatron_args,
)
from jiuge_training.data import encode_records, prepare_data
from jiuge_training.legacy import (
    LEGACY_FORMAT,
    LegacyTokenizer,
    model_config,
    weight_names,
)
from jiuge_training.model import LegacyAttention
from jiuge_training.weights import CheckpointReader, packed_qkv_slice


def tiny_config(vocab_size=263):
    return dict(
        vocab_size=vocab_size,
        dim_model=8,
        dim_ff=12,
        dim_head=4,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        scale=False,
        activate_fn='silu',
        dropout_p=0.0,
        eps=1e-5,
        half=True,
    )


def write_tokenizer(root):
    tokens = (
        ['<unk>', '<s>', '</s>']
        + [f'<0x{value:02X}>' for value in range(256)]
        + ['北', '北京', ' ', '\u2028']
    )
    (root / 'config.json').write_text(json.dumps(tiny_config(len(tokens))))
    (root / 'vocab.txt').write_text(
        ''.join(
            json.dumps(token, ensure_ascii=False) + '\n' for token in tokens
        )
    )
    return LegacyTokenizer(root)


class LegacyConfigTests(unittest.TestCase):
    def test_actual_archive_preset_is_distinct_from_thinking(self):
        config = load_config('8b-mha', 'lora', 'tp4', 'reference')
        self.assertEqual(config['model']['checkpoint_format'], LEGACY_FORMAT)
        self.assertEqual(config['model']['ffn_hidden_size'], 14336)
        self.assertEqual(config['model']['vocab_size'], 119696)
        self.assertEqual(config['model']['num_query_groups'], 32)
        self.assertIn(
            '--untie-embeddings-and-output-weights', megatron_args(config)
        )
        thinking = load_config('8b', 'lora', 'tp4', 'reference')
        self.assertEqual(thinking['model']['num_query_groups'], 2)
        self.assertEqual(thinking['model']['ffn_hidden_size'], 16384)

    def test_resume_contract_binds_legacy_format_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tokenizer(root)
            config = load_config(
                '8b-mha',
                'lora',
                'tp4',
                'reference',
                overrides={'model.path': str(root)},
            )
            contract = checkpoint_contract(config)
            self.assertEqual(
                contract['model']['checkpoint_format'], LEGACY_FORMAT
            )
            saved = root / 'contract.json'
            saved.write_text(json.dumps(contract))
            check_contract(config, saved)
            with (root / 'vocab.txt').open('a') as stream:
                stream.write('"changed"\n')
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                check_contract(config, saved)

    def test_source_config_preserves_unscaled_mha_semantics(self):
        raw = tiny_config()
        result = model_config(raw, 64)
        self.assertEqual(result['model_type'], 'cpm9g_legacy')
        self.assertEqual(result['rope_theta'], 10000)
        self.assertFalse(result['tie_word_embeddings'])
        self.assertNotIn('scale_emb', result)
        for override in (
            {'scale': True},
            {'num_kv_heads': 1},
            {'dim_head': 8},
        ):
            with (
                self.subTest(override=override),
                self.assertRaises(ValueError),
            ):
                model_config(dict(raw, **override), 64)


class LegacyTokenizerTests(unittest.TestCase):
    def test_longest_match_bytes_special_literals_and_physical_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = write_tokenizer(Path(directory))
            self.assertEqual(
                tokenizer.encode('北京北 \u2028'), [260, 259, 261, 262]
            )
            self.assertEqual(
                tokenizer.encode('🙂'),
                [byte + 3 for byte in '🙂'.encode('utf-8')],
            )
            self.assertEqual(tokenizer.encode('<s>'), [63, 118, 65])
            self.assertEqual(tokenizer.eos_token_id, 2)
            self.assertIsNone(tokenizer.pad_token_id)
            self.assertEqual(len(tokenizer.encode('北' * 20480)), 20480)
            with self.assertRaisesRegex(ValueError, '20480'):
                tokenizer.encode('北' * 20481)

    def test_prepared_sft_supervises_response_and_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = write_tokenizer(root)
            records = [
                {'instruction': '问题一', 'output': '北京'},
                {'instruction': '问题二', 'output': '北'},
            ]
            sample = list(encode_records(records[:1], tokenizer, 64, 'lora'))[
                0
            ]
            supervised = [
                token
                for token, mask in zip(
                    sample['tokens'][1:], sample['loss_mask']
                )
                if mask
            ]
            self.assertEqual(supervised, [260, 2])
            source = root / 'records.json'
            source.write_text(json.dumps(records))
            result = prepare_data(source, root / 'data', root, 'lora', 64)
            self.assertIn('vocab.txt', result['tokenizer_files_sha256'])
            self.assertEqual(result['splits']['train']['records'], 1)

    def test_rejects_changed_vocabulary_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tokenizer(root)
            with (root / 'vocab.txt').open('a') as stream:
                stream.write('"extra"\n')
            with self.assertRaisesRegex(ValueError, 'size'):
                LegacyTokenizer(root)


class LegacyCheckpointTests(unittest.TestCase):
    def test_mmap_aliases_and_tp_qkv_match_original_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = weight_names(2)
            state = {
                original: torch.arange(64).reshape(8, 8).float() + index * 100
                for index, original in enumerate(names.values())
            }
            torch.save(state, root / 'legacy.pt')
            reader = CheckpointReader(root, legacy_layers=2)
            for canonical, original in names.items():
                torch.testing.assert_close(
                    reader.tensor(canonical), state[original]
                )
            prefix = 'model.layers.0.self_attn'
            q, k, v = [
                reader.tensor(f'{prefix}.{p}_proj.weight') for p in 'qkv'
            ]
            expected = torch.cat([q[:4], k[:4], v[:4], q[4:], k[4:], v[4:]])
            actual = torch.cat(
                [
                    packed_qkv_slice(
                        reader, prefix, 'weight', 2, 2, 4, rank, 2
                    )
                    for rank in range(2)
                ]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            state['unexpected.weight'] = torch.ones(1)
            torch.save(state, root / 'legacy.pt')
            with self.assertRaisesRegex(ValueError, 'Legacy key mismatch'):
                CheckpointReader(root, legacy_layers=2)

    def test_attention_preserves_native_softmax_and_zero_mask_rows(self):
        torch.manual_seed(8)
        q, k, v = [
            torch.randn(3, 1, 2, 4, requires_grad=True) for _ in range(3)
        ]
        mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        attention = LegacyAttention(SimpleNamespace(attention_dropout=0.0))
        actual = attention(q, k, v, mask)
        scores = q.permute(1, 2, 0, 3) @ k.permute(1, 2, 3, 0)
        probabilities = scores.masked_fill(mask, float('-inf')).softmax(-1)
        expected = (probabilities @ v.permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
        torch.testing.assert_close(actual, expected.flatten(2), rtol=0, atol=0)
        actual.sum().backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in (q, k, v)))
        self.assertEqual(
            torch.count_nonzero(attention(q, k, v, torch.ones_like(mask))), 0
        )


if __name__ == '__main__':
    unittest.main()
