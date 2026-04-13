"""Bit-exact tests for funct_models_vector.lane_boxes.mov."""

from __future__ import annotations

import numpy as np
import pytest

from funct_models_vector.lane_boxes.mov import Mov, MovReq
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes


def test_mov_passes_lane_pattern_through():
    box = Mov(P)
    a = list(range(N))
    r = box.compute_now(MovReq(aVec=a))
    assert r.result == a


def test_mov_masks_high_bits():
    """`x & 0xFFFF` defends against accidental int>16-bit injection."""
    box = Mov(P)
    a = [0x1FFFF] * N
    r = box.compute_now(MovReq(aVec=a))
    assert r.result == [0xFFFF] * N


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mov_random(seed: int):
    rng = np.random.default_rng(seed)
    box = Mov(P)
    for _ in range(32):
        a = [int(rng.integers(0, 1 << 16)) for _ in range(N)]
        r = box.compute_now(MovReq(aVec=a))
        assert r.result == a


def test_step_mov_latency_one_cycle():
    box = Mov(P)
    req = MovReq(aVec=list(range(N)))
    assert box.step("mov", req) is None
    out = box.step("mov", None)
    assert out is not None and out.result == list(range(N))


def test_reset_clears_pipe():
    box = Mov(P)
    box.step("mov", MovReq(aVec=[0xAAAA] * N))
    box.reset()
    assert box.step("mov", None) is None
