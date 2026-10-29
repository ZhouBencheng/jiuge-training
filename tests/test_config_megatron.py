"""Opt-in CPU parser integration against the pinned vendor Megatron
installation.
"""

import os
import subprocess
import sys
import unittest

from jiuge_training.config import (
    DEFAULT_MEGATRON,
    load_config,
    megatron_args,
    runtime_environment,
)


@unittest.skipUnless(
    os.environ.get("JIUGE_MEGATRON_INTEGRATION") == "1",
    "requires pinned Megatron/vendor environment",
)
class MegatronParserTests(unittest.TestCase):
    def test_actual_parser_builds_uneven_4b_pipeline(self):
        config = load_config("4b", "cpt", "pp4", "reference")
        env = runtime_environment(
            os.environ.get("JIUGE_MEGATRON_ROOT", DEFAULT_MEGATRON)
        )
        env.update(WORLD_SIZE="4", RANK="0", LOCAL_RANK="0")
        code = (
            '\n'
            'from megatron.training.arguments import '
            'parse_and_validate_args, '
            'core_transformer_config_from_args\n'
            'from megatron.core.transformer.transformer_block '
            'import get_num_layers_to_build\n'
            'args = parse_and_validate_args()\n'
            'cfg = core_transformer_config_from_args(args)\n'
            'assert [get_num_layers_to_build(cfg, '
            'pp_rank=rank) for rank in range(4)] == [16, 15, '
            '15, 16]\n'
            'assert args.untie_embeddings_and_output_weights '
            'is False\n'
            "print('UNEVEN_PP4_PARSER_PASS')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, *megatron_args(config)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-5000:])
        self.assertIn("UNEVEN_PP4_PARSER_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
