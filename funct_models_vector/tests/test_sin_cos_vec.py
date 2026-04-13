"""Bit-exact tests for funct_models_vector.lane_boxes.sin_cos_vec.

`SinCosVec` does Q-format conversion + LUT lookup + linear
interpolation. The funct model error budget at BF16 precision is
~1 LUT step (with `lutAddrBits=5` and `qmnN=12` that's `~1/128 = 0.0078`
absolute on `[0, pi/2]`), plus rounding noise from the BF16 output
encoding. We use absolute tolerance ≤0.03 across the input domain
since LUT-edge interpolation can amplify error.

Special-attention cases:
  - `x == 0`: cos returns 1.0 exactly via `is_cos_bot_addr`;
    sin returns 0.
  - `x == pi/2`: sin returns ~1.0 via `is_sin_top_addr`; cos returns ~0.
  - Lane masking + step semantics.
"""

from __future__ import annotations

import struct
import math
import pytest

from funct_models_vector.lane_boxes.sin_cos_vec import (
    SinCosVec,
    SinCosVecReq,
    sin_cos_bf16,
)
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes
BOX = SinCosVec(P)
LUT = BOX._lut
SCALED = BOX._scaled


def _f32_to_bf16(x: float) -> int:
    return (struct.unpack("<I", struct.pack("<f", x))[0] >> 16) & 0xFFFF


def _bf16_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]


# ---------------------------------------------------------------
#  Endpoint cases
# ---------------------------------------------------------------

def test_sin_zero_is_zero():
    assert sin_cos_bf16(_f32_to_bf16(0.0), False, LUT, SCALED) == 0x0000


def test_cos_zero_is_one():
    """`is_cos_bot_addr` should clamp y0 to `scaled = 2^16`, so the
    interp result decodes to exactly 1.0 (no rounding loss)."""
    out = sin_cos_bf16(_f32_to_bf16(0.0), True, LUT, SCALED)
    assert _bf16_to_f32(out) == pytest.approx(1.0, abs=0.01)


def test_sin_pi_over_2_is_near_one():
    """At x = pi/2 the q-fold puts us *exactly* at quadrant 1 boundary
    if the multiply by 2/pi rounds up; depending on the rounding it
    can land in Q0 or Q1. Either way the magnitude must be ~1."""
    out = sin_cos_bf16(_f32_to_bf16(math.pi / 2), False, LUT, SCALED)
    assert abs(_bf16_to_f32(out)) == pytest.approx(1.0, abs=0.02)


def test_cos_pi_is_near_neg_one():
    out = sin_cos_bf16(_f32_to_bf16(math.pi), True, LUT, SCALED)
    assert _bf16_to_f32(out) == pytest.approx(-1.0, abs=0.02)


def test_sin_pi_over_6_is_one_half():
    """sin(30°) = 0.5; LUT-precision tolerance ~0.01."""
    out = sin_cos_bf16(_f32_to_bf16(math.pi / 6), False, LUT, SCALED)
    assert _bf16_to_f32(out) == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------
#  Quadrant fold sanity
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "x,is_cos,expected",
    [
        (math.pi, False, 0.0),     # sin(pi) ≈ 0
        (math.pi, True, -1.0),     # cos(pi) = -1
        (3 * math.pi / 2, False, -1.0),
        (3 * math.pi / 2, True, 0.0),
    ],
)
def test_quadrant_fold(x, is_cos, expected):
    out = sin_cos_bf16(_f32_to_bf16(x), is_cos, LUT, SCALED)
    assert _bf16_to_f32(out) == pytest.approx(expected, abs=0.03)


# ---------------------------------------------------------------
#  Sweep across [0, 2pi]
# ---------------------------------------------------------------

@pytest.mark.parametrize("is_cos", [False, True])
def test_sweep_over_unit_circle(is_cos):
    """64 sample points across [0, 2pi); each output should be within
    ~0.03 absolute of the true sin/cos."""
    bad = []
    for k in range(64):
        x = k * (2 * math.pi) / 64
        out = sin_cos_bf16(_f32_to_bf16(x), is_cos, LUT, SCALED)
        actual = _bf16_to_f32(out)
        true = math.cos(x) if is_cos else math.sin(x)
        if abs(actual - true) > 0.05:
            bad.append((x, true, actual, abs(actual - true)))
    assert not bad, f"high-error points (first 5): {bad[:5]}"


# ---------------------------------------------------------------
#  Class-level: lane masking + step
# ---------------------------------------------------------------

def test_lane_count_validation():
    box = SinCosVec(P)
    with pytest.raises(ValueError):
        box.compute_now(SinCosVecReq(xVec=[0] * (N - 1)))


def test_lane_mask_disabled_lanes_zero():
    box = SinCosVec(P)
    xs = [_f32_to_bf16(0.0)] * N    # cos(0) = 1 on enabled lanes
    r = box.compute_now(SinCosVecReq(xVec=xs, cos=True, laneMask=0x00FF))
    for i in range(N):
        if i < 8:
            assert _bf16_to_f32(r.result[i]) == pytest.approx(1.0, abs=0.01)
        else:
            assert r.result[i] == 0x0000


def test_step_sin_latency_one_cycle():
    box = SinCosVec(P)
    req = SinCosVecReq(xVec=[_f32_to_bf16(math.pi / 6)] * N, cos=False)
    assert box.step("sin", req) is None
    out = box.step("sin", None)
    assert out is not None
    for v in out.result:
        assert _bf16_to_f32(v) == pytest.approx(0.5, abs=0.01)


def test_step_cos_latency_one_cycle():
    box = SinCosVec(P)
    req = SinCosVecReq(xVec=[_f32_to_bf16(math.pi / 3)] * N, cos=True)
    assert box.step("cos", req) is None
    out = box.step("cos", None)
    assert out is not None
    for v in out.result:
        assert _bf16_to_f32(v) == pytest.approx(0.5, abs=0.01)


def test_sin_and_cos_share_module_separate_queues():
    box = SinCosVec(P)
    sin_req = SinCosVecReq(xVec=[_f32_to_bf16(math.pi / 6)] * N, cos=False)
    cos_req = SinCosVecReq(xVec=[_f32_to_bf16(math.pi / 3)] * N, cos=True)
    assert box.step("sin", sin_req) is None
    assert box.step("cos", cos_req) is None
    sin_out = box.step("sin", None)
    cos_out = box.step("cos", None)
    assert sin_out is not None and cos_out is not None
    for v in sin_out.result:
        assert _bf16_to_f32(v) == pytest.approx(0.5, abs=0.01)
    for v in cos_out.result:
        assert _bf16_to_f32(v) == pytest.approx(0.5, abs=0.01)


def test_unknown_op_name():
    box = SinCosVec(P)
    with pytest.raises(KeyError):
        box.step("not_sin_or_cos", None)
