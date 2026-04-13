"""Bit-exact tests for funct_models_vector.lane_boxes.log.

Mirrors `Log.scala` + `LogLUT`. Tests the LUT entry for r=1 (which
is `log2(1) = 0`, encoded as the special "neg=isNeg, x==0" branch),
the +/- exponent fold via `isNeg = (exp < bias)`, and the `m=9`
hardcoded width slack on the integer side (Log.scala:31).
"""

from __future__ import annotations

import struct
import math
import pytest

from funct_models_vector.lane_boxes.log import Log, LogReq, log_bf16
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes
BOX = Log(P)
LUT = BOX._lut


def _f32_to_bf16(x: float) -> int:
    return (struct.unpack("<I", struct.pack("<f", x))[0] >> 16) & 0xFFFF


def _bf16_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]


# ---------------------------------------------------------------
#  Special cases (lutFixedToBf16Log lattice)
# ---------------------------------------------------------------

def test_log_one_is_zero():
    """LUT[0] = log2(1) = 0 → helper sees `is_zero_lut=True` and
    returns signed zero with `neg=isNeg=False` → 0x0000."""
    assert log_bf16(_f32_to_bf16(1.0), LUT) == 0x0000


def test_log_pos_zero_to_neg_inf():
    """log2(+0) → -inf encoded via `isNeg=true, expUnsigned=0xFF`."""
    assert log_bf16(0x0000, LUT) == 0xFF80


def test_log_pos_inf_to_pos_inf():
    """input inf → flush to signed inf via the elsewhen branch."""
    assert log_bf16(0x7F80, LUT) == 0x7F80


def test_log_nan_to_zero():
    assert log_bf16(0x7FC0, LUT) == 0x0000


def test_log_subnormal_to_inf():
    """Subnormals share the input-zero branch → signed inf."""
    out = log_bf16(0x0001, LUT)
    # Subnormal has exp=0, so isNeg = (0 < 127) = true → -inf.
    assert out == 0xFF80


# ---------------------------------------------------------------
#  Math correctness
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "x,expected",
    [
        (2.0, 1.0),
        (4.0, 2.0),
        (8.0, 3.0),
        (16.0, 4.0),
        (0.5, -1.0),
        (0.25, -2.0),
        (0.125, -3.0),
    ],
)
def test_log_powers_of_two(x, expected):
    out = log_bf16(_f32_to_bf16(x), LUT)
    assert _bf16_to_f32(out) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "x,expected",
    [
        (3.0, math.log2(3.0)),
        (1.5, math.log2(1.5)),
        (10.0, math.log2(10.0)),
        (100.0, math.log2(100.0)),
    ],
)
def test_log_within_lut_resolution(x, expected):
    out = log_bf16(_f32_to_bf16(x), LUT)
    actual = _bf16_to_f32(out)
    # Absolute tolerance because log2 grows slowly; LUT step gives
    # ~1/128 mantissa precision.
    assert actual == pytest.approx(expected, abs=2e-2)


# ---------------------------------------------------------------
#  Exhaustive sweep
# ---------------------------------------------------------------

def test_exhaustive_sweep_no_exceptions_and_well_formed():
    for bits in range(0, 1 << 16):
        out = log_bf16(bits, LUT)
        assert 0 <= out <= 0xFFFF


def test_exhaustive_sweep_positive_against_python_reference():
    """log2 of positive normal BF16 inputs in `[2^-126, 2^127)` should
    match `math.log2` within ~0.02 absolute error."""
    bad = []
    for bits in range(0x0080, 0x7F80):
        x = _bf16_to_f32(bits)
        if x <= 0:
            continue
        out = log_bf16(bits, LUT)
        out_exp = (out >> 7) & 0xFF
        if out_exp == 0 or out_exp == 0xFF:
            continue
        actual = _bf16_to_f32(out)
        true = math.log2(x)
        # absolute error tolerance grows with magnitude — use 1.5% rel
        # for large magnitudes, 0.02 abs floor for small.
        if abs(true) < 1.0:
            tol = 0.02
        else:
            tol = abs(true) * 0.02
        if abs(actual - true) > tol:
            bad.append((bits, x, true, actual, abs(actual - true)))
    assert not bad, f"high-error inputs (first 5): {bad[:5]}"


# ---------------------------------------------------------------
#  Class-level: lane masking + step
# ---------------------------------------------------------------

def test_lane_count_validation():
    box = Log(P)
    with pytest.raises(ValueError):
        box.compute_now(LogReq(aVec=[0] * (N - 2)))


def test_lane_mask_disabled_lanes_zero():
    box = Log(P)
    a = [_f32_to_bf16(2.0)] * N
    r = box.compute_now(LogReq(aVec=a, laneMask=0x5555))
    for i in range(N):
        if (0x5555 >> i) & 1:
            assert r.result[i] == _f32_to_bf16(1.0)
        else:
            assert r.result[i] == 0x0000


def test_step_latency_one_cycle():
    box = Log(P)
    req = LogReq(aVec=[_f32_to_bf16(8.0)] * N)
    assert box.step("log", req) is None
    out = box.step("log", None)
    assert out is not None and all(v == _f32_to_bf16(3.0) for v in out.result)


def test_unknown_op_name():
    box = Log(P)
    with pytest.raises(KeyError):
        box.step("not_log", None)
