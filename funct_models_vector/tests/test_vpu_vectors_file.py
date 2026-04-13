"""Tier 2 — file-based cross-test (§8 Tier 2).

Parses the committed `src/test/resources/vpu_vectors.txt` and
`vpu_vectors_csum.txt` via `vpu_vector_file.parse_cases` and replays every
case through `VectorEngineModel.execute(...)`. For every lane the
dispatcher output must match `case.exp[i]` exactly.

Arithmetic spine ops (`add/sub/mul/relu/mov/rsum/csum/rmax/rmin/cmax/cmin/`
`pairmax/pairmin/square/cube/vli*`) are bit-exact by construction — the
reference files were regenerated from the same `VectorEngineModel`, so
any drift is either a lane_box refactor bug or a dispatcher routing bug.

Transcendentals (`rcp/sqrt/sin/cos/log/exp/exp2/tanh`) are also bit-exact
against the regenerated golden today. When the Group D real lane_boxes
land (§6), their output may diverge ≤ 1 ULP from the `_legacy_math_fallback`
path; at that point loosen `_compare_lane` for those ops to a 1-ULP band
and track the exact diverging lines. Until then, keeping the check
strict catches any unintended behavior change immediately.

The fp8pack / fp8unpack ops are intentionally absent from
`vpu_vectors.txt` (see IMPLEMENTATION_PLAN.md §4.6) — their dedicated
coverage lives in `test_fp8_pack.py` / `test_fp8_unpack.py`, and this
cross-test skips any case whose `vpu_op` is in that set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from funct_models_vector.vector_engine_model import VectorEngineModel
from funct_models_vector.vector_params import VectorParams
from funct_models_vector.vpu_vector_file import VPUTestCase, read_cases


REPO_ROOT = Path(__file__).resolve().parents[3]
REF_VECTORS = REPO_ROOT / "src" / "test" / "resources" / "vpu_vectors.txt"
REF_CSUM = REPO_ROOT / "src" / "test" / "resources" / "vpu_vectors_csum.txt"

_SKIP_OPS = frozenset({"fp8pack", "fp8unpack"})
_COL_OPS = frozenset({"csum", "cmax", "cmin"})
_VLI_OPS = frozenset({"vliOne", "vliCol", "vliRow", "vliAll"})
_BIN_OPS = frozenset({"add", "sub", "mul", "pairmax", "pairmin"})
_ROW_REDUCE_OPS = frozenset({"rsum", "rmax", "rmin"})


def _h(s: str) -> int:
    return int(s, 16) & 0xFFFF


def _load_cases(path: Path) -> list[VPUTestCase]:
    assert path.is_file(), (
        f"Missing reference file {path}. Regenerate with:\n"
        "  python -m funct_models_vector.gen_vectors "
        f"--out {path} --seed 12345 --num 58"
    )
    cases = read_cases(str(path))
    assert cases, f"{path} parsed into zero cases"
    return cases


def _dispatch(
    model: VectorEngineModel, case: VPUTestCase
) -> list[int]:
    """Mirror of `gen_vectors._build_case` dispatch logic, in reverse.
    Produces the lane vector that the model thinks is correct, shaped to
    match `case.exp` so a per-lane compare works."""
    op = case.vpu_op
    num_lanes = case.num_lanes

    if op in _VLI_OPS:
        imm = _h(case.vec_a[0])
        result = model.execute(op, imm=imm)
        imm_hex = result[0] & 0xFFFF
        if op == "vliOne":
            return [imm_hex]
        if op == "vliRow":
            return [imm_hex] * num_lanes
        # vliCol / vliAll: 32 slots
        return [imm_hex] * (2 * num_lanes)

    a_bits = [_h(h) for h in case.vec_a]
    b_bits = [_h(h) for h in case.vec_b] if case.vec_b else []

    if op in _COL_OPS:
        return model.execute(op, a_vec=a_bits)
    if op in _ROW_REDUCE_OPS or op == "mov":
        return model.execute(op, a_vec=a_bits)
    if op in _BIN_OPS:
        return model.execute(op, a_vec=a_bits, b_vec=b_bits)
    # Remaining: unary pointwise (rcp/sqrt/sin/cos/log/tanh/exp/exp2/square/cube/relu)
    return model.execute(op, a_vec=a_bits)


def _compare_lane(op: str, got: int, want: int) -> bool:
    """Bit-exact for every op today. When Group D real lane_boxes land,
    widen the transcendental branch to allow ≤ 1 ULP in BF16."""
    return (got & 0xFFFF) == (want & 0xFFFF)


def _gather_mismatches(
    model: VectorEngineModel, case: VPUTestCase
) -> list[str]:
    got = _dispatch(model, case)
    want = [_h(h) for h in case.exp]
    assert len(got) == len(want), (
        f"case {case.case_id} ({case.vpu_op}): "
        f"dispatcher returned {len(got)} lanes, file has {len(want)}"
    )
    out: list[str] = []
    for i, (g, w) in enumerate(zip(got, want)):
        if not _compare_lane(case.vpu_op, g, w):
            out.append(f"lane {i}: got 0x{g & 0xFFFF:04X}, want 0x{w:04X}")
    return out


# ----------------------------------------------------------------
#  Pytest plumbing
# ----------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> VectorEngineModel:
    return VectorEngineModel(VectorParams())


def _load_and_filter(path: Path) -> list[VPUTestCase]:
    return [c for c in _load_cases(path) if c.vpu_op not in _SKIP_OPS]


def _pp_id(case: VPUTestCase) -> str:
    return f"{case.case_id}-{case.vpu_op}"


_CASES_MAIN = _load_and_filter(REF_VECTORS) if REF_VECTORS.is_file() else []
_CASES_CSUM = _load_and_filter(REF_CSUM) if REF_CSUM.is_file() else []


def test_reference_files_parse_nonempty() -> None:
    assert _CASES_MAIN, f"vpu_vectors.txt parsed into zero dispatchable cases"
    assert _CASES_CSUM, f"vpu_vectors_csum.txt parsed into zero dispatchable cases"


@pytest.mark.parametrize("case", _CASES_MAIN, ids=[_pp_id(c) for c in _CASES_MAIN])
def test_vpu_vectors_main_case(
    model: VectorEngineModel, case: VPUTestCase
) -> None:
    mismatches = _gather_mismatches(model, case)
    if mismatches:
        head = "\n  ".join(mismatches[:8])
        more = f"\n  ... {len(mismatches) - 8} more" if len(mismatches) > 8 else ""
        pytest.fail(
            f"case {case.case_id} ({case.vpu_op}) mismatches vs "
            f"{REF_VECTORS.name}:\n  {head}{more}"
        )


@pytest.mark.parametrize("case", _CASES_CSUM, ids=[_pp_id(c) for c in _CASES_CSUM])
def test_vpu_vectors_csum_case(
    model: VectorEngineModel, case: VPUTestCase
) -> None:
    mismatches = _gather_mismatches(model, case)
    if mismatches:
        head = "\n  ".join(mismatches[:8])
        more = f"\n  ... {len(mismatches) - 8} more" if len(mismatches) > 8 else ""
        pytest.fail(
            f"case {case.case_id} ({case.vpu_op}) mismatches vs "
            f"{REF_CSUM.name}:\n  {head}{more}"
        )
