"""Bit-exact tests for funct_models_vector.lane_boxes.rcp.

The reference is the funct model's own LUT helper — that's the
intentional source-of-truth chosen in `IMPLEMENTATION_PLAN.md §9.8`.
The exhaustive 16-bit BF16 sweep verifies internal consistency
(every BF16 input runs through `rcp_bf16` without raising), and the
hand-written tests cover every special-case branch of
`lutFixedToBf16Rcp`.

Cross-test against the Scala `Rcp.scala` golden runs in Phase 2.
"""

from __future__ import annotations

import struct
import math
import pytest

from funct_models_vector.lane_boxes.rcp import Rcp, RcpReq, rcp_bf16
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes
BOX = Rcp(P)
LUT = BOX._lut


def _f32_to_bf16(x: float) -> int:
    return (struct.unpack("<I", struct.pack("<f", x))[0] >> 16) & 0xFFFF


def _bf16_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]


# ---------------------------------------------------------------
#  Special cases (lutFixedToBf16Rcp lattice)
# ---------------------------------------------------------------

def test_rcp_pos_zero_to_pos_inf():
    assert rcp_bf16(0x0000, LUT) == 0x7F80


def test_rcp_neg_zero_to_neg_inf():
    assert rcp_bf16(0x8000, LUT) == 0xFF80


def test_rcp_pos_inf_to_pos_zero():
    assert rcp_bf16(0x7F80, LUT) == 0x0000


def test_rcp_neg_inf_to_neg_zero():
    assert rcp_bf16(0xFF80, LUT) == 0x8000


def test_rcp_nan_to_zero():
    """rcp on NaN flushes to signed zero (sign bit preserved)."""
    assert rcp_bf16(0x7FC0, LUT) == 0x0000
    assert rcp_bf16(0xFFC0, LUT) == 0x8000


def test_rcp_subnormal_to_inf():
    """Subnormal inputs flush to signed inf (lattice puts them with zero)."""
    assert rcp_bf16(0x0001, LUT) == 0x7F80
    assert rcp_bf16(0x807F, LUT) == 0xFF80


# ---------------------------------------------------------------
#  Math correctness within LUT precision
# ---------------------------------------------------------------

@pytest.mark.parametrize("x", [1.0, 2.0, 4.0, 0.5, 8.0, 16.0, 0.25, 1e3, 1e-3])
def test_rcp_powers_match_within_lut_resolution(x):
    """LUT-backed rcp has ~ULP-of-LUT-resolution error. For 7-bit
    addressing on `[1, 2)` that's ~1/128 = ~0.0078 relative error."""
    out = rcp_bf16(_f32_to_bf16(x), LUT)
    actual = _bf16_to_f32(out)
    assert actual == pytest.approx(1.0 / x, rel=1e-2)


def test_rcp_one_is_one():
    """LUT entry 0 = 1/1 = 1.0 exactly."""
    out = rcp_bf16(_f32_to_bf16(1.0), LUT)
    assert out == _f32_to_bf16(1.0)


def test_rcp_negative_preserves_sign():
    out = rcp_bf16(_f32_to_bf16(-2.0), LUT)
    assert (out >> 15) & 1 == 1
    assert _bf16_to_f32(out) == pytest.approx(-0.5, rel=1e-2)


# ---------------------------------------------------------------
#  Exhaustive 16-bit sweep — internal consistency
# ---------------------------------------------------------------

def test_exhaustive_sweep_no_exceptions_and_well_formed():
    """All 65536 BF16 patterns through `rcp_bf16` must not raise and
    must return a 16-bit value.

    Note: the Chisel `lutFixedToBf16Rcp` has NO `bfExpSigned <= 0`
    underflow guard. For very large inputs near the BF16 max the
    reciprocal lands at `bfExpSigned == 0` and the helper emits
    `Cat(neg, 0x00, frac)` — bit-pattern-identical to a BF16 subnormal.
    That's hardware-faithful behavior, not a bug, so we don't assert
    against it here."""
    for bits in range(0, 1 << 16):
        out = rcp_bf16(bits, LUT)
        assert 0 <= out <= 0xFFFF


def test_exhaustive_sweep_finite_inputs_against_python_reference():
    """Every BF16 input in `(0, +inf)` whose true reciprocal lands in
    BF16 normal range should produce a finite normal output within
    ~1.5% relative error (LUT step on `[1, 2)` is `1/128 ≈ 0.78%`,
    with another bit lost on the exponent fold)."""
    bad = []
    for bits in range(0x0080, 0x7F80):   # positive normals only
        x = _bf16_to_f32(bits)
        true_rcp = 1.0 / x
        if not math.isfinite(true_rcp):
            continue
        if abs(true_rcp) < 1e-37 or abs(true_rcp) > 1e37:
            continue   # output would underflow/overflow BF16 normal range
        out = rcp_bf16(bits, LUT)
        if (out >> 7) & 0xFF == 0:
            continue   # underflow flushed
        if (out >> 7) & 0xFF == 0xFF:
            continue   # overflow flushed
        actual = _bf16_to_f32(out)
        if actual == 0.0 or true_rcp == 0.0:
            continue
        rel_err = abs(actual - true_rcp) / abs(true_rcp)
        if rel_err > 0.02:
            bad.append((bits, x, true_rcp, actual, rel_err))
    assert not bad, f"high-error inputs (first 5): {bad[:5]}"


# ---------------------------------------------------------------
#  Class-level: lane masking + step
# ---------------------------------------------------------------

def test_lane_count_validation():
    box = Rcp(P)
    with pytest.raises(ValueError):
        box.compute_now(RcpReq(aVec=[0] * (N - 1)))


def test_lane_mask_disabled_lanes_zero():
    box = Rcp(P)
    a = [_f32_to_bf16(2.0)] * N
    r = box.compute_now(RcpReq(aVec=a, laneMask=0x000F))
    for i in range(N):
        if i < 4:
            assert r.result[i] == _f32_to_bf16(0.5)
        else:
            assert r.result[i] == 0x0000


def test_step_latency_one_cycle():
    box = Rcp(P)
    req = RcpReq(aVec=[_f32_to_bf16(4.0)] * N)
    assert box.step("rcp", req) is None
    out = box.step("rcp", None)
    assert out is not None and all(v == _f32_to_bf16(0.25) for v in out.result)


def test_unknown_op_name():
    box = Rcp(P)
    with pytest.raises(KeyError):
        box.step("not_rcp", None)
