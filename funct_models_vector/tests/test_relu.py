"""Bit-exact tests for funct_models_vector.lane_boxes.relu.

The Scala module is `Mux(inVal(15), 0, inVal)` — a pure sign-bit check —
so the reference is a one-line bitwise mask. Hand-computed cases cover the
sign-zero and NaN edge cases the user must not get wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from funct_models_vector.lane_boxes.relu import Relu, ReluReq
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes


def _ref_relu(b: int) -> int:
    return 0 if (b & 0x8000) else (b & 0xFFFF)


def test_relu_positive_passthrough():
    box = Relu(P)
    a = [0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0, 0x4100,
         0x4120, 0x4140, 0x4160, 0x4180, 0x41A0, 0x41C0, 0x41E0, 0x4200]
    r = box.compute_now(ReluReq(aVec=a))
    assert r.result == a


def test_relu_negative_to_zero():
    box = Relu(P)
    a = [0xBF80, 0xC000, 0xC040, 0xC080, 0xC0A0, 0xC0C0, 0xC0E0, 0xC100,
         0xC120, 0xC140, 0xC160, 0xC180, 0xC1A0, 0xC1C0, 0xC1E0, 0xC200]
    r = box.compute_now(ReluReq(aVec=a))
    assert r.result == [0x0000] * N


def test_relu_negative_zero_becomes_positive_zero():
    box = Relu(P)
    a = [0x8000] * N  # all -0
    r = box.compute_now(ReluReq(aVec=a))
    assert r.result == [0x0000] * N


def test_relu_positive_zero_passthrough():
    box = Relu(P)
    a = [0x0000] * N
    r = box.compute_now(ReluReq(aVec=a))
    assert r.result == [0x0000] * N


def test_relu_signed_nan_becomes_zero():
    """0xFFC0 has sign=1 mantissa!=0 — a negative NaN. Sign-bit check
    collapses it to +0. This is intentionally NOT IEEE-correct for max(x,0):
    the Scala module is a literal sign-bit mux."""
    box = Relu(P)
    a = [0xFFC0] * N
    r = box.compute_now(ReluReq(aVec=a))
    assert r.result == [0x0000] * N


def test_relu_unsigned_nan_passthrough():
    box = Relu(P)
    a = [0x7FC0] * N  # +NaN
    r = box.compute_now(ReluReq(aVec=a))
    assert r.result == [0x7FC0] * N


def test_relu_mixed_lanes():
    box = Relu(P)
    a = [
        0x3F80, 0xBF80, 0x4000, 0x8000, 0x0000, 0x4040, 0xC040, 0x7F80,
        0xFFC0, 0x7FC0, 0xBE00, 0x3E00, 0x4500, 0xC500, 0x0001, 0x8001,
    ]
    r = box.compute_now(ReluReq(aVec=a))
    assert r.result == [_ref_relu(x) for x in a]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_relu_random_matches_reference(seed: int):
    rng = np.random.default_rng(seed)
    box = Relu(P)
    for _ in range(64):
        a = [int(rng.integers(0, 1 << 16)) for _ in range(N)]
        r = box.compute_now(ReluReq(aVec=a))
        assert r.result == [_ref_relu(x) for x in a]


def test_step_relu_latency_one_cycle():
    box = Relu(P)
    req = ReluReq(aVec=[0xBF80] * N)
    assert box.step("relu", req) is None
    out = box.step("relu", None)
    assert out is not None and out.result == [0x0000] * N
