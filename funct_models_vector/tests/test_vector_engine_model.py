"""Engine-level behavioral tests for `VectorEngineModel.execute()`.

The lane_box tests already cover per-op bit-exact semantics; this file
targets the dispatcher itself. Specifically: ops whose semantics depend
on engine-level state that isn't carried in `a_vec` / `b_vec` — today,
that's only the `row_idx` kwarg used by `vliRow` / `vliOne`.

Keep these tests narrow: if you need more coverage of a lane_box's own
behavior, add it to the per-lane_box test instead of here.
"""

from __future__ import annotations

from funct_models_vector.vector_engine_model import VectorEngineModel
from funct_models_vector.vector_params import VectorParams


_P = VectorParams()
_N = _P.num_lanes


def test_execute_vli_row_respects_row_idx() -> None:
    """`vliRow` fires only on row 0, broadcasts `imm` across every lane;
    any other row returns the all-zero reset pattern."""
    model = VectorEngineModel(_P)
    assert model.execute("vliRow", imm=0x4000, row_idx=0) == [0x4000] * _N
    assert model.execute("vliRow", imm=0x4000, row_idx=1) == [0x0000] * _N
    assert model.execute("vliRow", imm=0x4000, row_idx=7) == [0x0000] * _N


def test_execute_vli_one_respects_row_idx() -> None:
    """`vliOne` fires only on row 0, writes `imm` into lane 0 only;
    any other row returns all zeros."""
    model = VectorEngineModel(_P)
    expected_row0 = [0x0000] * _N
    expected_row0[0] = 0x4000
    assert model.execute("vliOne", imm=0x4000, row_idx=0) == expected_row0
    assert model.execute("vliOne", imm=0x4000, row_idx=1) == [0x0000] * _N
    assert model.execute("vliOne", imm=0x4000, row_idx=7) == [0x0000] * _N


def test_execute_vli_all_ignores_row_idx() -> None:
    """`vliAll` fills every lane on every row — `row_idx` must not
    gate it (mirrors `VectorLoadImm.scala` where the `isRow0` guard
    doesn't apply to `vliAll`)."""
    model = VectorEngineModel(_P)
    assert model.execute("vliAll", imm=0x4000, row_idx=0) == [0x4000] * _N
    assert model.execute("vliAll", imm=0x4000, row_idx=5) == [0x4000] * _N


def test_execute_vli_col_ignores_row_idx() -> None:
    """`vliCol` writes `imm` to lane 0 on every row — `row_idx` must
    not gate it."""
    model = VectorEngineModel(_P)
    expected = [0x0000] * _N
    expected[0] = 0x4000
    assert model.execute("vliCol", imm=0x4000, row_idx=0) == expected
    assert model.execute("vliCol", imm=0x4000, row_idx=5) == expected


def test_execute_row_idx_default_is_zero() -> None:
    """Callers that omit `row_idx` should see the row-0 behavior; this
    is the implicit contract the prior dispatcher had (`rowIdx=0`
    hardcoded)."""
    model = VectorEngineModel(_P)
    assert model.execute("vliRow", imm=0x4000) == [0x4000] * _N
    expected_one = [0x0000] * _N
    expected_one[0] = 0x4000
    assert model.execute("vliOne", imm=0x4000) == expected_one
