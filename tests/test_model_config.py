"""Production model declarations must match checkpoint architecture, before
PG setup.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from jiuge_training.model import configure, model_provider


@unittest.skipUnless(
    importlib.util.find_spec('megatron') is not None,
    'Pinned Megatron source required',
)
class ModelContractTest(unittest.TestCase):
    def test_wrong_checkpoint_architecture_is_rejected_before_initialization(
        self,
    ):
        hf = dict(
            model_type='fm9g',
            num_hidden_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            vocab_size=128,
            tie_word_embeddings=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text(json.dumps(hf))
            for override, message in (
                ({'hidden_size': 32}, 'hidden_size'),
                ({'num_query_groups': 4}, 'num_query_groups'),
                ({'tie_word_embeddings': False}, 'embedding tying'),
            ):
                configure(
                    dict(
                        model=dict(path=directory, smoke=False, **override),
                        task=dict(name='cpt'),
                        backend=dict(rmsnorm='reference'),
                    )
                )
                with (
                    self.subTest(override=override),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    model_provider()


if __name__ == '__main__':
    unittest.main()
