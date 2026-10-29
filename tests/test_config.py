import unittest
import tempfile
import json
from unittest.mock import patch
from pathlib import Path

from jiuge_training.config import load_config, validate, command


class ConfigTests(unittest.TestCase):
    def test_all_four_card_layouts(self):
        for model in ('4b', '7b', '8b'):
            for parallel in (
                'dp4',
                'dp2-tp2',
                'dp2-pp2',
                'tp4',
                'tp2-pp2',
                'pp4',
            ):
                cfg = load_config(model, 'cpt', parallel, 'reference')
                self.assertEqual(
                    cfg['parallel']['dp']
                    * cfg['parallel']['tp']
                    * cfg['parallel']['pp'],
                    4,
                )
        cfg = load_config('4b', 'cpt', 'pp4', 'reference')
        self.assertEqual(
            cfg['parallel']['num_layers_in_first_pipeline_stage'], 16
        )
        self.assertEqual(
            cfg['parallel']['num_layers_in_last_pipeline_stage'], 16
        )
        argv = command(cfg, "config.json")
        self.assertIn("--decoder-first-pipeline-num-layers", argv)
        self.assertIn("--decoder-last-pipeline-num-layers", argv)

    def test_future_layouts_and_invalid_tp(self):
        for model, layout in [
            ('4b', 'future64-4b8b'),
            ('8b', 'future64-4b8b'),
            ('7b', 'future64-7b'),
        ]:
            load_config(model, 'lora', layout, 'triton')
        with self.assertRaisesRegex(ValueError, 'heads'):
            load_config('7b', 'cpt', 'future64-4b8b', 'reference')

    def test_invalid_batch_backend_and_lora(self):
        cfg = load_config('tiny', 'lora-smoke', 'dp4', 'reference')
        for section, key, value in [
            ('task', 'global_batch_size', 15),
            ('backend', 'rmsnorm', 'auto'),
            ('task', 'lora_rank', 0),
            ('task', 'lora_alpha', 'bad'),
            ('task', 'lr', 0),
            ('parallel', 'dp', 2),
        ]:
            bad = {k: dict(v) for k, v in cfg.items()}
            bad[section][key] = value
            with self.assertRaises(ValueError):
                validate(bad)

    def test_production_cannot_select_synthetic_or_tiny(self):
        with self.assertRaises(ValueError):
            load_config('4b', 'cpt-smoke', 'dp4', 'reference')
        with self.assertRaises(ValueError):
            load_config('tiny', 'cpt', 'dp4', 'reference')

    def test_batch_override_recomputes_accumulation(self):
        cfg = load_config(
            "tiny",
            "cpt-smoke",
            "dp4",
            "reference",
            overrides={"task.global_batch_size": 8},
        )
        self.assertEqual(cfg["task"]["gradient_accumulation_steps"], 2)
        with self.assertRaisesRegex(ValueError, "GBS"):
            load_config(
                "tiny",
                "cpt-smoke",
                "dp4",
                "reference",
                overrides={
                    "task.global_batch_size": 8,
                    "task.gradient_accumulation_steps": 4,
                },
            )

    def test_resume_requires_tracker_and_all_model_shards(self):
        from jiuge_training.config import validate_resume, checkpoint_contract

        cfg = load_config("tiny", "cpt-smoke", "tp2-pp2", "reference")
        with tempfile.TemporaryDirectory() as tmp:
            cfg["task"]["load"] = tmp
            with self.assertRaisesRegex(ValueError, "tracker"):
                validate_resume(cfg)
            (Path(tmp) / "latest_checkpointed_iteration.txt").write_text("2")
            with self.assertRaisesRegex(ValueError, "contract missing"):
                validate_resume(cfg)
            (Path(tmp) / "jiuge_contract.json").write_text(
                json.dumps(checkpoint_contract(cfg))
            )
            with self.assertRaisesRegex(ValueError, "shard"):
                validate_resume(cfg)
            for tp in range(2):
                for pp in range(2):
                    shard = (
                        Path(tmp)
                        / "iter_0000002"
                        / f"mp_rank_{tp:02d}_{pp:03d}"
                        / "model_optim_rng.pt"
                    )
                    shard.parent.mkdir(parents=True)
                    shard.write_bytes(b"checkpoint")
            validate_resume(cfg)
            cfg["task"]["distributed_optimizer"] = True
            (Path(tmp) / "jiuge_contract.json").write_text(
                json.dumps(checkpoint_contract(cfg))
            )
            with self.assertRaisesRegex(ValueError, "distrib_optim"):
                validate_resume(cfg)
            for folder in (Path(tmp) / "iter_0000002").iterdir():
                (folder / "distrib_optim.pt").write_bytes(b"optimizer-state")
            validate_resume(cfg)

    def test_checkpoint_contract_blocks_semantic_changes_and_allows_backend(
        self,
    ):
        from jiuge_training.config import (
            prepare_checkpoint_directory,
            check_contract,
        )

        cfg = load_config("tiny", "lora-smoke", "dp4", "reference")
        with tempfile.TemporaryDirectory() as tmp:
            cfg["task"]["save"] = tmp
            prepare_checkpoint_directory(cfg)
            path = Path(tmp) / "jiuge_contract.json"
            original = path.read_bytes()
            prepare_checkpoint_directory(cfg)
            cfg["backend"]["rmsnorm"] = "triton"
            cfg["task"]["train_iters"] = 3
            check_contract(cfg, path)
            for section, key, value in (
                ("task", "lora_alpha", 32),
                ("task", "lora_dropout", 0.1),
                ("task", "lr", 0.001),
                ("task", "name", "cpt"),
                ("model", "tie_word_embeddings", True),
            ):
                changed = {k: dict(v) for k, v in cfg.items()}
                changed[section][key] = value
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    prepare_checkpoint_directory(changed)
                self.assertEqual(path.read_bytes(), original)

    def test_data_path_tracks_model_selection(self):
        for model in ("4b", "7b", "8b"):
            cfg = load_config(model, "lora", "dp4", "reference")
            self.assertEqual(
                cfg["task"]["data_path"], f"data/processed/{model}-lora"
            )
        cfg = load_config(
            "7b",
            "cpt",
            "tp4",
            "reference",
            overrides={"task.data_path": "/custom"},
        )
        self.assertEqual(cfg["task"]["data_path"], "/custom")

    def test_local_device_preflight_rejects_future_plan_on_four_cards(self):
        from jiuge_training.config import validate_local_devices

        cfg = load_config("tiny", "cpt-smoke", "dp4", "reference")
        with patch(
            "jiuge_training.config.subprocess.check_output", return_value="4\n"
        ) as probe:
            self.assertEqual(
                validate_local_devices(cfg, {"LD_LIBRARY_PATH": "/vendor"}), 4
            )
            self.assertEqual(
                probe.call_args.kwargs["env"]["LD_LIBRARY_PATH"], "/vendor"
            )
            future = load_config(
                "tiny", "cpt-smoke", "future64-4b8b", "reference"
            )
            with self.assertRaisesRegex(ValueError, "only 4 visible"):
                validate_local_devices(future, {})

    def test_shared_megatron_default_and_custom_root(self):
        import os
        import runpy
        from jiuge_training.config import (
            DEFAULT_MEGATRON,
            MEGATRON_SHA,
            ROOT,
            runtime_environment,
        )

        self.assertEqual(DEFAULT_MEGATRON, "/private/Megatron-LM")
        with patch(
            "jiuge_training.config.subprocess.check_output",
            return_value=MEGATRON_SHA,
        ) as git:
            for root in (
                DEFAULT_MEGATRON,
                "/custom/Megatron-LM",
                "../Megatron-LM",
            ):
                env = (
                    runtime_environment()
                    if root == DEFAULT_MEGATRON
                    else runtime_environment(root)
                )
                argv = git.call_args.args[0]
                self.assertEqual(argv[argv.index("-C") + 1], root)
                self.assertEqual(
                    [arg for arg in argv if arg.startswith("safe.directory=")],
                    [f"safe.directory={Path(root).resolve()}"],
                )
                self.assertIn(root, env["PYTHONPATH"].split(os.pathsep))
        doctor = runpy.run_path(str(ROOT / "scripts/doctor.py"))
        with (
            patch("pathlib.Path.is_dir", return_value=True),
            patch("subprocess.run") as doctor_git,
        ):
            doctor_git.return_value.stdout = MEGATRON_SHA
            report = doctor["megatron_revisions"]()
            self.assertEqual(report["source_path"], DEFAULT_MEGATRON)
            self.assertEqual(report["checked_out_head"], MEGATRON_SHA)
            argv = doctor_git.call_args.args[0]
            self.assertEqual(argv[argv.index("-C") + 1], DEFAULT_MEGATRON)
            self.assertEqual(
                [arg for arg in argv if arg.startswith("safe.directory=")],
                [f"safe.directory={Path(DEFAULT_MEGATRON).resolve()}"],
            )

    def test_command_is_argv(self):
        cfg = load_config('tiny', 'cpt-smoke', 'single', 'reference')
        cmd = command(cfg, '/tmp/path with spaces/config.json')
        self.assertIn('/tmp/path with spaces/config.json', cmd)
        self.assertIn('--standalone', cmd)
        self.assertIn('jiuge_training.train', cmd)


if __name__ == '__main__':
    unittest.main()
