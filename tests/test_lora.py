import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from jiuge_training.lora import ParallelLoRA, save_adapter, load_adapter


class TupleLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class LoRATest(unittest.TestCase):
    def test_zero_initialization_freezing_and_restore(self):
        for kind in ('column', 'row'):
            torch.manual_seed(12)
            base = TupleLinear(5, 7, bias=False)
            model = nn.Sequential(ParallelLoRA(base, kind, 3, 6))
            x = torch.randn(4, 5, requires_grad=True)
            expected = base(x)[0].detach()
            actual = model(x)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.square().mean().backward()
            self.assertIsNone(base.weight.grad)
            self.assertFalse(base.weight.requires_grad)
            self.assertGreater(model[0].lora_B.grad.abs().sum().item(), 0)
            self.assertEqual(model[0].lora_A.grad.abs().sum().item(), 0)
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad], lr=0.1
            )
            optimizer.step()
            updated = model(x)[0].detach()
            self.assertFalse(torch.equal(updated, expected))
            with tempfile.TemporaryDirectory() as directory:
                file = Path(directory) / 'adapter.pt'
                save_adapter(model, file)
                with torch.no_grad():
                    model[0].lora_B.zero_()
                load_adapter(model, file)
                torch.testing.assert_close(
                    model(x)[0], updated, rtol=0, atol=0
                )

    def test_invalid_configuration(self):
        for rank, alpha, dropout in ((0, 1, 0), (2, -1, 0), (2, 1, 1)):
            with self.assertRaises(ValueError):
                ParallelLoRA(TupleLinear(3, 4), 'column', rank, alpha, dropout)


if __name__ == '__main__':
    unittest.main()
