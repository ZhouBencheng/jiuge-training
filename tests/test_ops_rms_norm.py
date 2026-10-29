import unittest
from types import SimpleNamespace

import torch

from jiuge_ops import ReferenceRMSNorm, TritonRMSNorm, rms_norm


class RMSNormContractTest(unittest.TestCase):
    def test_hf_low_precision_intermediate_rounding(self):
        torch.manual_seed(19)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(3, 17, dtype=dtype, requires_grad=True)
            w = torch.randn(17, dtype=dtype, requires_grad=True)
            variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            normalized = (x * torch.rsqrt(variance + 1e-6)).to(dtype)
            expected = normalized * w
            actual = rms_norm(x, w)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            g = torch.randn_like(x)
            actual_grads = torch.autograd.grad(actual, (x, w), g)
            expected_grads = torch.autograd.grad(expected, (x, w), g)
            for a, b in zip(actual_grads, expected_grads):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_reference_math_and_gradients(self):
        torch.manual_seed(31)
        x = torch.randn(2, 3, 17, requires_grad=True)
        w = torch.randn(17, requires_grad=True)
        g = torch.randn_like(x)
        y = rms_norm(x, w)
        r = torch.rsqrt(x.detach().square().mean(-1, keepdim=True) + 1e-6)
        expected = x.detach() * r * w.detach()
        dx, dw = torch.autograd.grad(y, (x, w), g)
        h = x.detach() * r
        expected_dx = r * (
            g * w.detach() - h * (h * g * w.detach()).mean(-1, keepdim=True)
        )
        torch.testing.assert_close(y, expected)
        torch.testing.assert_close(dx, expected_dx)
        torch.testing.assert_close(dw, (g * h).sum((0, 1)))

    def test_megatron_constructor_and_failure_contract(self):
        cfg = SimpleNamespace(
            normalization="RMSNorm",
            params_dtype=torch.float32,
            sequence_parallel=True,
        )
        norm = ReferenceRMSNorm(config=cfg, hidden_size=17, eps=1e-5)
        self.assertTrue(norm.weight.sequence_parallel)
        self.assertEqual(list(norm.state_dict()), ["weight"])
        self.assertEqual(norm(torch.ones(2, 17)).shape, (2, 17))
        with self.assertRaisesRegex(RuntimeError, "CUDA/CoreX"):
            TritonRMSNorm(17)(torch.ones(2, 17))
        with self.assertRaises(ValueError):
            rms_norm(torch.ones(2, 17), torch.ones(16))
        with self.assertRaises(ValueError):
            rms_norm(torch.ones(2, 17), torch.ones(17), eps=0)
        cfg.layernorm_zero_centered_gamma = True
        with self.assertRaisesRegex(ValueError, "zero-centered"):
            ReferenceRMSNorm(config=cfg, hidden_size=17)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/CoreX required")
class TritonRMSNormTest(unittest.TestCase):
    def test_real_model_shapes_and_weight_gradient_rounding(self):
        for rows in (64, 128):
            for hidden in (2560, 3584, 4096):
                for dtype in (torch.bfloat16, torch.float16, torch.float32):
                    for weight_dtype in dict.fromkeys((dtype, torch.float32)):
                        with self.subTest(
                            rows=rows,
                            hidden=hidden,
                            dtype=dtype,
                            weight_dtype=weight_dtype,
                        ):
                            torch.manual_seed(2026)
                            x = torch.randn(
                                rows,
                                hidden,
                                device='cuda',
                                dtype=dtype,
                                requires_grad=True,
                            )
                            w = torch.randn(
                                hidden,
                                device='cuda',
                                dtype=weight_dtype,
                                requires_grad=True,
                            )
                            g = torch.randn_like(x)
                            expected = rms_norm(x, w)
                            expected_grads = torch.autograd.grad(
                                expected, (x, w), g
                            )
                            actual = rms_norm(x, w, backend='triton')
                            actual_grads = torch.autograd.grad(
                                actual, (x, w), g
                            )
                            tol = {
                                torch.float32: 2e-5,
                                torch.float16: 3e-3,
                                torch.bfloat16: 2e-2,
                            }[dtype]
                            for a, b in zip(
                                (actual, *actual_grads),
                                (expected, *expected_grads),
                            ):
                                torch.testing.assert_close(
                                    a, b, atol=tol, rtol=tol
                                )

    def test_forward_and_gradients(self):
        torch.manual_seed(2026)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shape in ((7, 17), (2, 3, 257), (16, 3072), (8, 4096)):
                for weight_dtype in dict.fromkeys((dtype, torch.float32)):
                    with self.subTest(
                        dtype=dtype, shape=shape, weight_dtype=weight_dtype
                    ):
                        x = torch.randn(
                            shape,
                            device="cuda",
                            dtype=dtype,
                            requires_grad=True,
                        )
                        w = torch.randn(
                            shape[-1],
                            device="cuda",
                            dtype=weight_dtype,
                            requires_grad=True,
                        )
                        g = torch.randn_like(x)
                        yr = rms_norm(x, w, backend="reference")
                        dxr, dwr = torch.autograd.grad(yr, (x, w), g)
                        yt = rms_norm(x, w, backend="triton")
                        dxt, dwt = torch.autograd.grad(yt, (x, w), g)
                        tol = {
                            torch.float32: 2e-5,
                            torch.float16: 3e-3,
                            torch.bfloat16: 2e-2,
                        }[dtype]
                        for a, b in ((yt, yr), (dxt, dxr), (dwt, dwr)):
                            torch.testing.assert_close(
                                a, b, atol=tol, rtol=tol
                            )

    def test_noncontiguous_zero_and_frozen_weight(self):
        for zero in (True, False):
            x = (
                torch.zeros(3, 7, 5, device="cuda")
                if zero
                else torch.randn(3, 7, 5, device="cuda")
            )
            x = x.transpose(1, 2).requires_grad_()
            w = torch.ones(7, device="cuda")
            g = torch.randn_like(x)
            yr = rms_norm(x, w)
            yt = rms_norm(x, w, backend="triton")
            (dxr,) = torch.autograd.grad(yr, x, g)
            (dxt,) = torch.autograd.grad(yt, x, g)
            torch.testing.assert_close(yt, yr, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(dxt, dxr, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
