import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jiuge_training.train import adapter_training_validity


class WorkerCacheTests(unittest.TestCase):
    def test_global_ranks_isolate_cache_writes_before_torch_import(self):
        code = (
            "import json, os, sys; "
            "from jiuge_training.train import configure_worker_caches; "
            "configure_worker_caches(); "
            "assert 'torch' not in sys.modules; "
            "print(json.dumps([os.environ.get(name) for name in "
            "('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')]))"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = os.environ.copy()
            env.pop("RANK", None)
            env["LOCAL_RANK"] = "0"
            bases = [str(root / "triton"), str(root / "inductor")]
            env.update(
                zip(("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"), bases)
            )

            def paths(environment):
                return json.loads(
                    subprocess.check_output(
                        [sys.executable, "-c", code],
                        env=environment,
                        text=True,
                    )
                )

            self.assertEqual(paths(env), bases)
            for rank in (0, 4):
                actual = paths(dict(env, RANK=str(rank)))
                expected = [str(Path(base) / f"rank-{rank}") for base in bases]
                self.assertEqual(actual, expected)
                for path in actual:
                    cache = Path(path)
                    cache.mkdir(parents=True)
                    (cache / "same-kernel-key").write_text(str(rank))
            for base in bases:
                for rank in (0, 4):
                    value = Path(base) / f"rank-{rank}" / "same-kernel-key"
                    self.assertEqual(value.read_text(), str(rank))
            default_env = dict(env, RANK="8")
            for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
                default_env.pop(key)
            expected_root = Path(__file__).resolve().parents[1] / ".cache"
            self.assertEqual(
                paths(default_env),
                [
                    str(expected_root / name / "rank-8")
                    for name in ("triton", "torchinductor")
                ],
            )


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "requires torch for tensor audit"
)
class AdapterValidityTests(unittest.TestCase):
    def model(self):
        import torch

        module = torch.nn.Module()
        module.register_parameter(
            "base", torch.nn.Parameter(torch.ones(2), requires_grad=False)
        )
        module.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2)))
        module.register_parameter("lora_B", torch.nn.Parameter(torch.zeros(2)))
        return module

    def test_fresh_training_requires_every_B_and_frozen_base(self):
        import torch

        first, second = self.model(), self.model()
        with torch.no_grad():
            second.lora_B.fill_(0.01)
        audit = adapter_training_validity([first, second], resumed=False)
        self.assertFalse(audit["passed"])
        self.assertEqual(
            (audit["lora_B_nonzero"], audit["lora_B_total"]), (1, 2)
        )
        with torch.no_grad():
            first.lora_B.fill_(0.01)
        self.assertTrue(
            adapter_training_validity([first, second], resumed=False)["passed"]
        )
        first.base.requires_grad_(True)
        self.assertFalse(
            adapter_training_validity([first, second], resumed=False)["passed"]
        )
        first.base.requires_grad_(False)
        with torch.no_grad():
            first.lora_B.fill_(float("nan"))
        audit = adapter_training_validity([first, second], resumed=False)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["nonfinite_B"], ["lora_B"])

    def test_resume_nonzero_B_does_not_claim_new_update(self):
        audit = adapter_training_validity([self.model()], resumed=True)
        self.assertEqual(
            audit["update_verification"], "not_verified_for_resume"
        )
        self.assertEqual(audit["lora_B_nonzero"], 0)


if __name__ == "__main__":
    unittest.main()
