"""
test_e4m3_vectorized.py
=======================
Two things in one file:

1. Correctness tests -- all three implementations must produce bit-identical
   output to _float_to_e4m3_byte_scalar (the golden reference):
     - float_to_e4m3_bytes_torch  (vectorized torch ops)
     - float_to_e4m3_bytes_c      (C loop via ctypes, shim_float_to_e4m3)

2. Benchmark -- wall-clock time for all three across a range of tensor
   sizes, printed as a table.

Run:
    python test_e4m3_vectorized.py          # correctness only
    python test_e4m3_vectorized.py --bench  # correctness + benchmark
    python test_e4m3_vectorized.py -v       # verbose per-test output
"""

from __future__ import annotations

import argparse
import sys
import time
import unittest

import numpy as np
import torch

from pi0_inout_c.ipt_mxu_model.ipt_rtl_linear import (
    float_to_e4m3_bytes as float_to_e4m3_bytes_scalar,
    _float_to_e4m3_byte_scalar,
)
from pi0_inout_c.ipt_mxu_model.ipt_rtl_linear_c import (
    float_to_e4m3_bytes_torch,
    float_to_e4m3_bytes_c,
)


# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------


def _check(values: list[float], label: str) -> tuple[bool, list[str]]:
    """
    Run all three implementations on `values`.
    Returns (all_match, failure_lines).
    Scalar is the golden reference; torch and C must match it.
    """
    t = torch.tensor(values, dtype=torch.float32)
    t_np = t.numpy().astype(np.float32)

    scalar_out = float_to_e4m3_bytes_scalar(t).tolist()
    torch_out = float_to_e4m3_bytes_torch(t).tolist()
    c_out = float_to_e4m3_bytes_c(t_np).tolist()

    failures = []
    for i, (s, tv, cv, f) in enumerate(zip(scalar_out, torch_out, c_out, values)):
        if s != tv or s != cv:
            failures.append(
                f"  [{label}][{i}]  input={f!r}"
                f"  scalar=0x{s:02x}"
                f"  torch=0x{tv:02x}"
                f"  c=0x{cv:02x}"
            )
    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestE4M3Implementations(unittest.TestCase):

    def _assert_match(self, values: list[float], label: str):
        ok, failures = _check(values, label)
        if not ok:
            self.fail(
                f"{len(failures)} mismatch(es) in '{label}':\n"
                + "\n".join(failures[:20])
                + ("\n  ..." if len(failures) > 20 else "")
            )

    # ------------------------------------------------------------------
    # Test 1: zero and signed zero
    # ------------------------------------------------------------------
    def test_zero(self):
        self._assert_match([0.0, -0.0], "zero")

    # ------------------------------------------------------------------
    # Test 2: all 256 E4M3 representable values round-trip correctly
    # ------------------------------------------------------------------
    def test_all_e4m3_roundtrip(self):
        """Every normal E4M3 value must round-trip back to its original byte.

        Subnormals (exp_field==0, frac!=0, values 0x01-0x07 and 0x81-0x87)
        and -0.0 (0x80) are excluded: _float_to_e4m3_byte_scalar flushes all
        of these to 0x00 because their magnitude is below 2^-6.  Both
        implementations match the scalar, so the flush is correct behaviour --
        the round-trip simply does not hold for subnormal inputs.
        """
        from ipt_mxu_model.fp_formats import decode_e4m3

        values = []
        expected_bytes = []
        for b in range(256):
            d = decode_e4m3(b)
            if d.is_nan or d.is_inf:
                continue
            # Skip subnormals (exp_field == 0): scalar flushes these to 0
            if ((b >> 3) & 0xF) == 0:
                continue
            values.append(d.value)
            expected_bytes.append(b)

        t = torch.tensor(values, dtype=torch.float32)
        t_np = t.numpy().astype(np.float32)

        torch_out = float_to_e4m3_bytes_torch(t).tolist()
        c_out = float_to_e4m3_bytes_c(t_np).tolist()

        failures = []
        for i, (exp, tv, cv, v) in enumerate(
            zip(expected_bytes, torch_out, c_out, values)
        ):
            if exp != tv or exp != cv:
                failures.append(
                    f"  byte=0x{exp:02x}  value={v!r}"
                    f"  torch=0x{tv:02x}  c=0x{cv:02x}"
                )
        if failures:
            self.fail(
                f"{len(failures)} round-trip failure(s):\n" + "\n".join(failures[:20])
            )

    # ------------------------------------------------------------------
    # Test 3: normal positive values across the full E4M3 exponent range
    # ------------------------------------------------------------------
    def test_normal_range(self):
        values = []
        for exp in range(-6, 9):
            for frac in range(8):
                values.append((1.0 + frac / 8.0) * (2.0**exp))
        self._assert_match(values, "normal_range")

    # ------------------------------------------------------------------
    # Test 4: normal negative values
    # ------------------------------------------------------------------
    def test_normal_range_negative(self):
        values = []
        for exp in range(-6, 9):
            for frac in range(8):
                values.append(-(1.0 + frac / 8.0) * (2.0**exp))
        self._assert_match(values, "normal_range_negative")

    # ------------------------------------------------------------------
    # Test 5: frac==8 carry (rounding bumps the exponent)
    # ------------------------------------------------------------------
    def test_rounding_carry(self):
        values = []
        for exp in range(-6, 8):
            base = (1.0 + 7.0 / 8.0) * (2.0**exp)
            values.append(base * 1.001)
            values.append(-base * 1.001)
        self._assert_match(values, "rounding_carry")

    # ------------------------------------------------------------------
    # Test 6: saturation boundary
    # ------------------------------------------------------------------
    def test_saturation(self):
        # 480.0 = (1+7/8)*2^8 is the true max, encodes to 0x7F
        # 496.0 is the midpoint where frac rounds to 8 -> exp=9 -> saturate
        values = [
            480.0,
            -480.0,  # max normal, should NOT saturate
            495.0,
            -495.0,  # just below midpoint, should NOT saturate
            496.0,
            -496.0,  # exact midpoint, saturate
            500.0,
            -500.0,
            1e6,
            -1e6,
            float("inf"),
            float("-inf"),
        ]
        self._assert_match(values, "saturation")

    # ------------------------------------------------------------------
    # Test 7: flush to zero (below 2^-6)
    # ------------------------------------------------------------------
    def test_flush_to_zero(self):
        values = [
            2.0**-6,  # exactly min normal, must NOT flush
            2.0**-6 - 1e-10,  # just below, must flush
            0.005,
            -0.005,
            1e-10,
            -1e-10,
        ]
        self._assert_match(values, "flush_to_zero")

    # ------------------------------------------------------------------
    # Test 8: NaN
    # ------------------------------------------------------------------
    def test_nan(self):
        self._assert_match([float("nan")], "nan")

    # ------------------------------------------------------------------
    # Test 9: random float32 values (broad coverage)
    # ------------------------------------------------------------------
    def test_random_float32(self):
        torch.manual_seed(0)
        values = torch.empty(4096).uniform_(-500.0, 500.0).tolist()
        self._assert_match(values, "random_float32")

    # ------------------------------------------------------------------
    # Test 10: random values in the near-zero / flush region
    # ------------------------------------------------------------------
    def test_random_near_zero(self):
        torch.manual_seed(1)
        values = torch.empty(1024).uniform_(-0.01, 0.01).tolist()
        self._assert_match(values, "random_near_zero")

    # ------------------------------------------------------------------
    # Test 11: 2-D tensor -- shape preserved, all elements match
    # ------------------------------------------------------------------
    def test_shape_preserved(self):
        torch.manual_seed(2)
        t = torch.randn(16, 32)
        t_np = t.numpy().astype(np.float32)

        scalar_out = float_to_e4m3_bytes_scalar(t)
        torch_out = float_to_e4m3_bytes_torch(t)
        c_out = torch.from_numpy(float_to_e4m3_bytes_c(t_np))

        self.assertEqual(scalar_out.shape, torch_out.shape)
        self.assertEqual(scalar_out.shape, c_out.shape)
        self.assertTrue(
            torch.equal(scalar_out, torch_out), "torch: 2-D output mismatch"
        )
        self.assertTrue(torch.equal(scalar_out, c_out), "C: 2-D output mismatch")

    # ------------------------------------------------------------------
    # Test 12: large weight-matrix shape
    # ------------------------------------------------------------------
    def test_large_weight_matrix(self):
        torch.manual_seed(3)
        t = torch.randn(1024, 1024)
        t_np = t.numpy().astype(np.float32)

        scalar_out = float_to_e4m3_bytes_scalar(t)
        torch_out = float_to_e4m3_bytes_torch(t)
        c_out = torch.from_numpy(float_to_e4m3_bytes_c(t_np))

        torch_mm = int((scalar_out != torch_out).sum().item())
        c_mm = int((scalar_out != c_out).sum().item())
        self.assertEqual(
            torch_mm, 0, f"torch: {torch_mm} mismatches in 1024x1024 matrix"
        )

        if c_mm != 0:
            # Print the failing values to help diagnose
            idxs = (scalar_out != c_out).nonzero(as_tuple=False)
            msgs = []
            for idx in idxs[:10]:
                i = tuple(idx.tolist())
                v = float(t[i].item())
                msgs.append(
                    f"  [{i}] value={v!r} "
                    f"scalar=0x{scalar_out[i].item():02x} "
                    f"c=0x{c_out[i].item():02x}"
                )
            self.fail(
                f"C: {c_mm} mismatch(es) in 1024x1024 matrix:\n" + "\n".join(msgs)
            )


# ---------------------------------------------------------------------------
# Benchmark (opt-in via --bench)
# ---------------------------------------------------------------------------


def _benchmark(shapes: list[tuple[int, ...]], n_warmup: int = 5, n_runs: int = 30):
    print("\nBenchmark: float_to_e4m3_bytes  (ms per call)")
    print(
        f"{'shape':<22} {'scalar':>10} {'torch':>10} {'C':>10}"
        f" {'torch speedup':>14} {'C speedup':>11}"
    )
    print("-" * 82)

    for shape in shapes:
        t = torch.randn(*shape)
        t_np = t.numpy().astype(np.float32)

        for _ in range(n_warmup):
            float_to_e4m3_bytes_scalar(t)
            float_to_e4m3_bytes_torch(t)
            float_to_e4m3_bytes_c(t_np)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            float_to_e4m3_bytes_scalar(t)
        scalar_ms = (time.perf_counter() - t0) / n_runs * 1000

        t0 = time.perf_counter()
        for _ in range(n_runs):
            float_to_e4m3_bytes_torch(t)
        torch_ms = (time.perf_counter() - t0) / n_runs * 1000

        t0 = time.perf_counter()
        for _ in range(n_runs):
            float_to_e4m3_bytes_c(t_np)
        c_ms = (time.perf_counter() - t0) / n_runs * 1000

        torch_speedup = scalar_ms / torch_ms if torch_ms > 0 else float("inf")
        c_speedup = scalar_ms / c_ms if c_ms > 0 else float("inf")
        shape_str = "x".join(str(d) for d in shape)

        print(
            f"{shape_str:<22} {scalar_ms:>10.2f} {torch_ms:>10.2f} {c_ms:>10.2f}"
            f" {torch_speedup:>13.1f}x {c_speedup:>10.1f}x"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench", action="store_true", help="Run benchmark after correctness tests"
    )
    args, _ = parser.parse_known_args()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestE4M3Implementations)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if args.bench and result.wasSuccessful():
        _benchmark(
            [
                (32,),
                (16, 32),
                (256, 32),
                (1024, 1024),
                (4096, 1024),
                (1024, 4096),
            ]
        )

    sys.exit(0 if result.wasSuccessful() else 1)
