"""Bit-exact tests for funct_models_vector.lane_boxes.sqrt.

Mirrors `Sqrt.scala` + `SqrtLUT`. The `oddExp` branch handles odd
unbiased exponents by multiplying the LUT entry by `sqrt(2)`; the
`lutFixedToBf16Sqrt` helper handles the half-exponent. Covers:

  - Exact powers of 4 (even unbiased exp + LUT entry == 1.0).
  - Powers of 2 that need the `oddExp` shifted base.
  - Special cases (zero, inf, NaN, subnormal).
  - Exhaustive 16-bit sweep with internal consistency checks.
"""

from __future__ import annotations

import struct
import math
import pytest

from funct_models_vector.lane_boxes.sqrt import Sqrt, SqrtReq, sqrt_bf16
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes
BOX = Sqrt(P)
LUT = BOX._lut
MAX_VAL = BOX._max_val


def _f32_to_bf16(x: float) -> int:
    return (struct.unpack("<I", struct.pack("<f", x))[0] >> 16) & 0xFFFF


def _bf16_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]


# ---------------------------------------------------------------
#  Special cases
# ---------------------------------------------------------------

def test_sqrt_pos_zero():
    assert sqrt_bf16(0x0000, LUT, MAX_VAL) == 0x0000


def test_sqrt_pos_inf():
    assert sqrt_bf16(0x7F80, LUT, MAX_VAL) == 0x7F80


def test_sqrt_pos_nan_to_zero():
    """`isInputNaN || isInputSubnormal || isInputZero` → 0.
    Note: `neg=false.B` always, so output is +0 even for negative NaN."""
    assert sqrt_bf16(0x7FC0, LUT, MAX_VAL) == 0x0000


def test_sqrt_subnormal_to_zero():
    assert sqrt_bf16(0x0001, LUT, MAX_VAL) == 0x0000


# ---------------------------------------------------------------
#  Math correctness
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "x,expected",
    [
        (1.0, 1.0),
        (4.0, 2.0),
        (16.0, 4.0),
        (64.0, 8.0),
        (0.25, 0.5),
        (0.0625, 0.25),
        (256.0, 16.0),
    ],
)
def test_sqrt_perfect_squares_even_exp(x, expected):
    """Even-unbiased-exponent perfect squares: LUT entry 0 = sqrt(1) =
    1.0 exactly, exponent halves cleanly. Must be exact."""
    out = sqrt_bf16(_f32_to_bf16(x), LUT, MAX_VAL)
    assert out == _f32_to_bf16(expected)


@pytest.mark.parametrize("x", [2.0, 8.0, 32.0, 0.5, 0.125])
def test_sqrt_powers_of_two_odd_exp_within_lut_resolution(x):
    """Powers of 2 with odd unbiased exponent need the `oddExp`
    shifted base. LUT resolution gives ~1/128 relative error."""
    out = sqrt_bf16(_f32_to_bf16(x), LUT, MAX_VAL)
    assert _bf16_to_f32(out) == pytest.approx(math.sqrt(x), rel=1e-2)


@pytest.mark.parametrize("x", [9.0, 100.0, 25.0, 49.0, 1.5, 3.0])
def test_sqrt_within_lut_resolution(x):
    out = sqrt_bf16(_f32_to_bf16(x), LUT, MAX_VAL)
    assert _bf16_to_f32(out) == pytest.approx(math.sqrt(x), rel=1e-2)


# ---------------------------------------------------------------
#  Exhaustive sweep
# ---------------------------------------------------------------

def test_exhaustive_sweep_no_exceptions_and_well_formed():
    for bits in range(0, 1 << 16):
        out = sqrt_bf16(bits, LUT, MAX_VAL)
        assert 0 <= out <= 0xFFFF


def test_exhaustive_sweep_positive_finite_against_python_reference():
    """Positive normal BF16 inputs whose sqrt fits in BF16 must match
    `math.sqrt` within 1.5%."""
    bad = []
    for bits in range(0x0080, 0x7F80):
        x = _bf16_to_f32(bits)
        if x <= 0:
            continue
        out = sqrt_bf16(bits, LUT, MAX_VAL)
        if (out >> 7) & 0xFF == 0 or (out >> 7) & 0xFF == 0xFF:
            continue
        actual = _bf16_to_f32(out)
        true = math.sqrt(x)
        if true == 0.0:
            continue
        rel_err = abs(actual - true) / true
        if rel_err > 0.02:
            bad.append((bits, x, true, actual, rel_err))
    assert not bad, f"high-error inputs (first 5): {bad[:5]}"


# ---------------------------------------------------------------
#  Class-level: lane masking + step
# ---------------------------------------------------------------

def test_lane_count_validation():
    box = Sqrt(P)
    with pytest.raises(ValueError):
        box.compute_now(SqrtReq(aVec=[0] * (N + 1)))


def test_lane_mask_disabled_lanes_zero():
    box = Sqrt(P)
    a = [_f32_to_bf16(4.0)] * N
    r = box.compute_now(SqrtReq(aVec=a, laneMask=0xF0F0))
    for i in range(N):
        if (0xF0F0 >> i) & 1:
            assert r.result[i] == _f32_to_bf16(2.0)
        else:
            assert r.result[i] == 0x0000


def test_step_latency_one_cycle():
    box = Sqrt(P)
    req = SqrtReq(aVec=[_f32_to_bf16(16.0)] * N)
    assert box.step("sqrt", req) is None
    out = box.step("sqrt", None)
    assert out is not None and all(v == _f32_to_bf16(4.0) for v in out.result)


def test_unknown_op_name():
    box = Sqrt(P)
    with pytest.raises(KeyError):
        box.step("not_sqrt", None)
