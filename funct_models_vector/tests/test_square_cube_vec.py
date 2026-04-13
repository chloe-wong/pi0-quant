"""Bit-exact tests for funct_models_vector.lane_boxes.square_cube_vec.

`square_cube_bf16` mirrors `VectorParam.scala:6-96` exactly (integer
mantissa multiply + manual leading-1 normalize + explicit exponent
adjustment). Tests cover:

  - Math correctness on simple powers-of-two and small integers.
  - Sign convention: square always +, cube preserves sign.
  - Special cases: zero / subnormal / NaN → flush to (signed) zero;
    inf → signed inf; under/overflow on exponent → flushed.
  - Lane masking and lane-count validation.
  - Cycle-accurate step() per op (square / cube share the module but
    have separate latency queues, so they don't collide).
"""

from __future__ import annotations

import struct
import numpy as np
import pytest

from funct_models_vector.lane_boxes.square_cube_vec import (
    SquareCubeVec,
    SquareCubeReq,
    square_cube_bf16,
)
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes


def _f32_to_bf16(x: float) -> int:
    return (struct.unpack("<I", struct.pack("<f", x))[0] >> 16) & 0xFFFF


def _bf16_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]


# ---------------------------------------------------------------
#  Pure helper math
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "x,expected",
    [
        (1.0, 1.0),
        (2.0, 4.0),
        (3.0, 9.0),
        (4.0, 16.0),
        (-2.0, 4.0),
        (0.5, 0.25),
        (-0.5, 0.25),
        (0.25, 0.0625),
    ],
)
def test_square_basic_math(x, expected):
    out = square_cube_bf16(_f32_to_bf16(x), is_cube=False)
    assert _bf16_to_f32(out) == pytest.approx(expected, rel=1e-2)


@pytest.mark.parametrize(
    "x,expected",
    [
        (1.0, 1.0),
        (2.0, 8.0),
        (3.0, 27.0),
        (4.0, 64.0),
        (-2.0, -8.0),
        (-3.0, -27.0),
        (0.5, 0.125),
        (-0.5, -0.125),
    ],
)
def test_cube_basic_math(x, expected):
    out = square_cube_bf16(_f32_to_bf16(x), is_cube=True)
    assert _bf16_to_f32(out) == pytest.approx(expected, rel=1e-2)


def test_square_always_unsigned():
    """Square of any negative input must have sign bit 0."""
    for bits in [0xC000, 0xBF80, 0x8001, 0xFF7F]:
        out = square_cube_bf16(bits, is_cube=False)
        assert (out >> 15) & 1 == 0, f"square({bits:#06x}) got sign-1: {out:#06x}"


def test_cube_preserves_sign():
    """Cube of negative finite value must have sign bit 1."""
    for bits in [0xC000, 0xBF80, 0xC080]:
        out = square_cube_bf16(bits, is_cube=True)
        assert (out >> 15) & 1 == 1


# ---------------------------------------------------------------
#  Special cases (VectorParam.scala:78-92)
# ---------------------------------------------------------------

def test_zero_input_square():
    assert square_cube_bf16(0x0000, is_cube=False) == 0x0000
    assert square_cube_bf16(0x8000, is_cube=False) == 0x0000   # -0 → +0


def test_zero_input_cube_preserves_sign():
    assert square_cube_bf16(0x0000, is_cube=True) == 0x0000
    assert square_cube_bf16(0x8000, is_cube=True) == 0x8000   # -0 → -0


def test_subnormal_flush_to_zero():
    """exp=0, fra!=0 → flushed (`isInputSubnormal` not flagged in the
    Scala `when` directly, but underflow handling forces it via
    `adjustedExp <= 0` since `real_exp = -127`)."""
    for bits in [0x0001, 0x007F, 0x0040]:
        assert square_cube_bf16(bits, is_cube=False) == 0x0000
        # cube preserves sign even on flush; positive sub → +0
        assert square_cube_bf16(bits, is_cube=True) == 0x0000


def test_nan_flushes_to_zero_not_propagated():
    """`isInputNaN` is in the same `when` as input zero, so NaN → 0.
    This intentionally diverges from IEEE NaN propagation."""
    for bits in [0x7FC0, 0x7FFF, 0xFFC0]:
        assert square_cube_bf16(bits, is_cube=False) == 0x0000
        # cube preserves sign of the NaN bit pattern
        sign = (bits >> 15) & 1
        assert square_cube_bf16(bits, is_cube=True) == (sign << 15)


def test_inf_inputs():
    assert square_cube_bf16(0x7F80, is_cube=False) == 0x7F80   # +inf² = +inf
    assert square_cube_bf16(0xFF80, is_cube=False) == 0x7F80   # -inf² = +inf
    assert square_cube_bf16(0x7F80, is_cube=True) == 0x7F80    # +inf³ = +inf
    assert square_cube_bf16(0xFF80, is_cube=True) == 0xFF80    # -inf³ = -inf


def test_overflow_to_inf():
    """Square of a huge BF16 → adjusted_exp >= 255 → flush to inf."""
    huge = _f32_to_bf16(1e30)   # ~2^99
    assert square_cube_bf16(huge, is_cube=False) == 0x7F80
    cube_huge = _f32_to_bf16(1e20)
    assert square_cube_bf16(cube_huge, is_cube=True) == 0x7F80


def test_underflow_to_zero():
    """Cube of a tiny BF16 → adjusted_exp <= 0 → flush to signed zero."""
    tiny = _f32_to_bf16(1e-30)
    assert square_cube_bf16(tiny, is_cube=False) == 0x0000
    neg_tiny = _f32_to_bf16(-1e-15)
    # cube path; preserves sign
    assert square_cube_bf16(neg_tiny, is_cube=True) == 0x8000


# ---------------------------------------------------------------
#  Class-level: lane masking + validation
# ---------------------------------------------------------------

def test_lane_count_validation():
    box = SquareCubeVec(P)
    with pytest.raises(ValueError):
        box.compute_now(SquareCubeReq(aVec=[0] * (N - 1)))


def test_lane_mask_disabled_lanes_zero():
    box = SquareCubeVec(P)
    a = [_f32_to_bf16(2.0)] * N
    r = box.compute_now(SquareCubeReq(aVec=a, isCube=False, laneMask=0x00FF))
    for i in range(N):
        if i < 8:
            assert r.result[i] == _f32_to_bf16(4.0)
        else:
            assert r.result[i] == 0x0000


def test_lane_mask_passthrough_to_response():
    box = SquareCubeVec(P)
    a = [_f32_to_bf16(2.0)] * N
    r = box.compute_now(SquareCubeReq(aVec=a, laneMask=0xCAFE))
    assert r.laneMask == 0xCAFE


# ---------------------------------------------------------------
#  Cycle-accurate step (latency 1)
# ---------------------------------------------------------------

def test_step_square_latency_one():
    box = SquareCubeVec(P)
    req = SquareCubeReq(aVec=[_f32_to_bf16(3.0)] * N, isCube=False)
    assert box.step("square", req) is None
    out = box.step("square", None)
    assert out is not None
    for v in out.result:
        assert v == _f32_to_bf16(9.0)


def test_step_cube_latency_one():
    box = SquareCubeVec(P)
    req = SquareCubeReq(aVec=[_f32_to_bf16(3.0)] * N, isCube=True)
    assert box.step("cube", req) is None
    out = box.step("cube", None)
    assert out is not None
    for v in out.result:
        assert v == _f32_to_bf16(27.0)


def test_square_and_cube_share_module_separate_queues():
    """Square and cube have independent latency queues, so back-to-
    back of different ops do NOT collide."""
    box = SquareCubeVec(P)
    sq = SquareCubeReq(aVec=[_f32_to_bf16(2.0)] * N, isCube=False)
    cb = SquareCubeReq(aVec=[_f32_to_bf16(2.0)] * N, isCube=True)
    assert box.step("square", sq) is None
    assert box.step("cube", cb) is None
    sq_out = box.step("square", None)
    cb_out = box.step("cube", None)
    assert sq_out is not None and all(v == _f32_to_bf16(4.0) for v in sq_out.result)
    assert cb_out is not None and all(v == _f32_to_bf16(8.0) for v in cb_out.result)


def test_unknown_op_name():
    box = SquareCubeVec(P)
    with pytest.raises(KeyError):
        box.step("not_an_op", None)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_input_no_exceptions(seed: int):
    """Sweep random BF16 patterns through both square and cube paths
    and verify no Python exceptions (no out-of-range indexing, etc.)
    and that the output is a 16-bit value."""
    rng = np.random.default_rng(seed)
    box = SquareCubeVec(P)
    for _ in range(64):
        a = [int(rng.integers(0, 1 << 16)) for _ in range(N)]
        for is_cube in [False, True]:
            r = box.compute_now(SquareCubeReq(aVec=a, isCube=is_cube))
            for v in r.result:
                assert 0 <= v <= 0xFFFF
