"""Group D placeholder tests for exp / exp2 / tanh.

Per `IMPLEMENTATION_PLAN.md §4.5`, Group D ships only the class
skeleton + `pytest.skip("blocked on §6")`. These tests verify that:
  - The classes can be constructed without raising (so the engine
    wiring can register them with the dispatcher).
  - The request / response dataclasses have the right shape.
  - `compute_now` raises `NotImplementedError` (so a caller that
    accidentally drives them gets an explicit failure, not silent
    wrong output).
  - `step()` with `req=None` is safe (drain path used by the
    cycle-accurate engine harness).
"""

from __future__ import annotations

import pytest

from funct_models_vector.lane_boxes.exp import Exp, FPEXReq, FPEXResp
from funct_models_vector.lane_boxes.tanh_rec import TanhRec, TanhReq, TanhResp
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes


def test_exp_class_constructs():
    box = Exp(P)
    assert "exp" in box.LATENCIES
    assert "exp2" in box.LATENCIES


def test_tanh_class_constructs():
    box = TanhRec(P)
    assert "tanh" in box.LATENCIES


def test_exp_compute_now_raises():
    box = Exp(P)
    with pytest.raises(NotImplementedError, match="Group D"):
        box.compute_now(FPEXReq(xVec=[0] * N))


def test_tanh_compute_now_raises():
    box = TanhRec(P)
    with pytest.raises(NotImplementedError, match="Group D"):
        box.compute_now(TanhReq(xVec=[0] * N))


def test_exp_step_drain_is_safe():
    """Driving step() with req=None must not raise — the engine's
    drain phase calls every lane_box every cycle, and Group D
    placeholders need to participate without blowing up."""
    box = Exp(P)
    assert box.step("exp", None) is None
    assert box.step("exp2", None) is None


def test_tanh_step_drain_is_safe():
    box = TanhRec(P)
    assert box.step("tanh", None) is None


def test_exp_step_with_request_raises():
    box = Exp(P)
    with pytest.raises(NotImplementedError, match="Group D"):
        box.step("exp", FPEXReq(xVec=[0] * N))


def test_tanh_step_with_request_raises():
    box = TanhRec(P)
    with pytest.raises(NotImplementedError, match="Group D"):
        box.step("tanh", TanhReq(xVec=[0] * N))


@pytest.mark.skip(reason="blocked on IMPLEMENTATION_PLAN.md §6")
def test_exp_against_scala_golden():
    """Bit-exact cross-test against `ExpLane.scala`. Lands when §6
    HardFloat round-trip strategy is implemented."""


@pytest.mark.skip(reason="blocked on IMPLEMENTATION_PLAN.md §6")
def test_tanh_against_scala_golden():
    """Bit-exact cross-test against `TanhRec.scala`. Lands when §6
    HardFloat round-trip strategy is implemented."""
