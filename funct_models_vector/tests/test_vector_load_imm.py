"""Bit-exact tests for funct_models_vector.lane_boxes.vector_load_imm.

Mirrors the four sub-ops in VectorLoadImm.scala:
  vliAll: every lane gets imm
  vliRow: every lane on row 0; otherwise zeros
  vliCol: lane 0 only, on every row
  vliOne: lane 0 on row 0; otherwise zeros
"""

from __future__ import annotations

import pytest

from funct_models_vector.lane_boxes.vector_load_imm import (
    VectorLoadImm,
    VLIReq,
)
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes


def test_vli_all_fills_every_lane_every_row():
    box = VectorLoadImm(P)
    for row in (0, 1, 5, 31):
        r = box.compute_now(VLIReq(op="vliAll", imm=0x4000, rowIdx=row))
        assert r.result == [0x4000] * N


def test_vli_row_only_fires_on_row_zero():
    box = VectorLoadImm(P)
    assert box.compute_now(VLIReq(op="vliRow", imm=0x4000, rowIdx=0)).result == [0x4000] * N
    for row in (1, 2, 5, 31):
        r = box.compute_now(VLIReq(op="vliRow", imm=0x4000, rowIdx=row))
        assert r.result == [0x0000] * N


def test_vli_col_writes_lane_zero_every_row():
    box = VectorLoadImm(P)
    for row in (0, 1, 5, 31):
        r = box.compute_now(VLIReq(op="vliCol", imm=0x4000, rowIdx=row))
        expected = [0x0000] * N
        expected[0] = 0x4000
        assert r.result == expected


def test_vli_one_writes_lane_zero_only_on_row_zero():
    box = VectorLoadImm(P)
    r0 = box.compute_now(VLIReq(op="vliOne", imm=0x4000, rowIdx=0))
    expected = [0x0000] * N
    expected[0] = 0x4000
    assert r0.result == expected
    for row in (1, 2, 5, 31):
        r = box.compute_now(VLIReq(op="vliOne", imm=0x4000, rowIdx=row))
        assert r.result == [0x0000] * N


def test_vli_negative_imm_round_trips_as_unsigned_bits():
    """Scala SInt(16.W).asUInt keeps the bottom 16 bits, so -1 → 0xFFFF."""
    box = VectorLoadImm(P)
    r = box.compute_now(VLIReq(op="vliAll", imm=-1, rowIdx=0))
    assert r.result == [0xFFFF] * N


def test_vli_imm_masked_to_16_bits():
    box = VectorLoadImm(P)
    r = box.compute_now(VLIReq(op="vliAll", imm=0x1FFFF, rowIdx=0))
    assert r.result == [0xFFFF] * N


def test_vli_unknown_op_raises():
    box = VectorLoadImm(P)
    with pytest.raises(ValueError):
        box.compute_now(VLIReq(op="vliBananas", imm=0, rowIdx=0))


def test_step_vli_latency_one_cycle():
    box = VectorLoadImm(P)
    req = VLIReq(op="vliAll", imm=0x4000, rowIdx=0)
    assert box.step("vliAll", req) is None
    out = box.step("vliAll", None)
    assert out is not None and out.result == [0x4000] * N


def test_step_per_op_queues_are_independent():
    """vliAll and vliCol have independent latency queues; advancing one
    must not pop the other."""
    box = VectorLoadImm(P)
    box.step("vliAll", VLIReq(op="vliAll", imm=0x4000, rowIdx=0))   # vliAll t=0
    # Pump vliCol independently — should not surface vliAll's pending result.
    out_col = box.step("vliCol", VLIReq(op="vliCol", imm=0x4001, rowIdx=0))
    assert out_col is None
    # Now advance vliAll one more step → t=1, result emerges.
    out_all = box.step("vliAll", None)
    assert out_all is not None and out_all.result == [0x4000] * N
