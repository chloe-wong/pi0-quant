"""Tier 3 — RTL diff cross-test (§8 Tier 3).

Parses `src/test/resources/rtl_actual_outputs.txt` — a snapshot of what
the Scala `VectorEngineTopAllOpsTest` driver produced when run against
`src/test/resources/vpu_vectors.txt` — and compares, per (case_id,
lane), the RTL's `actual_hex` against `VectorEngineModel.execute(...)`
on the same input row.

Purpose: Tier 2 pits the funct model against the Python golden
(`vpu_vectors.txt`), which was itself generated *from* the funct model
after Phase 1f — so Tier 2 only catches a regression in the dispatcher
or a lane_box relative to that self-consistency point. Tier 3 is the
complementary check: is the funct model still faithful to the actual
hardware? A disagreement here means either (a) a funct model bug
against the RTL, or (b) an RTL bug relative to what the Python golden
says it should produce.

Scope and tolerance:

- Arithmetic spine (add/sub/mul/sqrt/rcp/relu/mov/square/cube/pairmax/
  pairmin/rsum/rmax/rmin/cmax/cmin/csum/vli* and the LUT-backed ops
  log/exp/exp2) is **bit-exact**. Any mismatch here is a real bug.

- `sin` and `cos` are **bit-exact** against RTL as of the post-Phase 1f
  snapshot regeneration (2026-04-13). They were previously xfailed
  because the snapshot predated the vpu_vectors.txt regen; once the
  RTL was re-simulated against the current golden they passed cleanly.

- `tanh` remains **xfail** (expected to diverge) until the Group D
  snapshot pipeline (§6) lands. The funct model routes tanh through
  `_legacy_math_fallback` (Python `math.tanh`), which is not
  bit-for-bit identical to the HardFloat iterative approximation in
  `sp26-fp-units/vpuFUnits/`. The test is marked xfail so the suite
  stays green; removing the xfail should be the exit gate for §6.

File format (per row, post header):
```
  <case_id>  "<case_desc>"  <lane>  <vpuOp>  <actual_hex> <expected_hex> \
    <actual_float> <expected_float> <relErr%> <match> [| <inA_hex> <inA_float> <inB_hex> <inB_float>]
```

Blank lines separate cases. `actual_hex` is the RTL's 16-bit BF16 word
at that lane; `expected_hex` is the OLD Python golden (pre-Phase 1f)
and is intentionally ignored here.

The `| inA_hex inA_float inB_hex inB_float` suffix is only emitted by
`VectorEngineTopAllOpsTest.scala` on **failing** rows (`if (!ok)`
branch around line 556). In a mostly-PASS snapshot — which the
current `rtl_actual_outputs.txt` is — those columns are absent from
almost every data row. The parser treats them as optional.

Some ops emit more rows than there are output lanes (e.g., `vliAll`
has 512 rows for a 32-slot output — the Scala driver cycles the test
16× to verify stability). Those are deduped per-(lane,value) before
the compare.

Preconditions vs. output compare. The test has two checks per case,
in order: (1) *snapshot-corpus preconditions* — for each lane the
snapshot claims, the captured `vpuOp` must match the current
`vpu_vectors.txt` op, and (if the snapshot happens to have the
optional `inA`/`inB` columns for that row) the captured inputs must
match the current per-lane inputs; (2) the existing actual-hex
compare. (1) is a strict improvement over the prior "case_id is
proof of corpus alignment" assumption, but note the limitation: for
a snapshot where every sin/cos/tanh row is PASS (as today), the
op check aligns trivially and the input check has nothing to compare
against — so the stale-snapshot hypothesis in
`AUTONOMOUS_ASSUMPTIONS.md` A7 is **not** settled by this patch.
What the patch does give you is a loud failure the moment anyone
regenerates the snapshot with a different op ordering, a different
seed that produces the same case_ids but different inputs *and* any
failure rows, or any other form of corpus drift that surfaces an
inA/inB column.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

from funct_models_vector.vector_engine_model import VectorEngineModel
from funct_models_vector.vector_params import VectorParams
from funct_models_vector.vpu_vector_file import VPUTestCase, read_cases


REPO_ROOT = Path(__file__).resolve().parents[3]
REF_VECTORS = REPO_ROOT / "src" / "test" / "resources" / "vpu_vectors.txt"
REF_RTL = REPO_ROOT / "src" / "test" / "resources" / "rtl_actual_outputs.txt"


# Ops whose funct-model implementation is known to diverge from RTL today.
# sin/cos were removed 2026-04-13 after fresh RTL snapshot confirmed they
# now match. Remove "tanh" when §6 (Group D snapshot pipeline) lands.
_KNOWN_DIVERGENT_OPS = frozenset({"tanh"})

# Ops that don't exist in the file: fp8pack/fp8unpack are tested separately.
_SKIP_OPS = frozenset({"fp8pack", "fp8unpack"})

# Same op-category constants as test_vpu_vectors_file, for dispatching.
_COL_OPS = frozenset({"csum", "cmax", "cmin"})
_VLI_OPS = frozenset({"vliOne", "vliCol", "vliRow", "vliAll"})
_BIN_OPS = frozenset({"add", "sub", "mul", "pairmax", "pairmin"})
_ROW_REDUCE_OPS = frozenset({"rsum", "rmax", "rmin"})


def _h(s: str) -> int:
    return int(s, 16) & 0xFFFF


# ----------------------------------------------------------------
#  rtl_actual_outputs.txt parser
# ----------------------------------------------------------------

@dataclass(frozen=True)
class RTLLaneRow:
    """One deduped `(case_id, lane)` row from rtl_actual_outputs.txt.

    `op`, `lane`, and `actual_hex` are always populated. `in_a_hex` and
    `in_b_hex` are `None` unless the snapshot row carried the optional
    `| inA_hex ... inB_hex ...` suffix — which the Scala driver only
    emits for failing comparisons. Don't rely on those being present in
    the current PASS-heavy snapshot.
    """
    lane: int
    op: str
    actual_hex: int
    in_a_hex: int | None
    in_b_hex: int | None


def _parse_rtl_rows(path: Path) -> dict[int, dict[int, RTLLaneRow]]:
    """Returns `{case_id: {lane_idx: RTLLaneRow}}` deduped from the file.

    Retains the per-lane `vpuOp` column (always present) and the
    `inA_hex` / `inB_hex` columns (only present on failing rows, see
    module docstring) so Tier 3 can cross-check the snapshot corpus
    against the current `vpu_vectors.txt` rather than trusting
    `case_id` equality alone.

    Raises if the same `(case_id, lane)` has two distinct snapshot
    rows — that would mean the file itself is internally inconsistent.
    """
    assert path.is_file(), f"Missing RTL snapshot {path}"
    lines = path.read_text().splitlines()
    rows: dict[int, dict[int, RTLLaneRow]] = defaultdict(dict)
    for raw in lines[1:]:  # skip header
        line = raw.rstrip()
        if not line.strip():
            continue
        toks = line.split()
        try:
            hex_idx = next(i for i, t in enumerate(toks) if t.startswith("0x"))
        except StopIteration:
            continue  # malformed line, ignore
        lane = int(toks[hex_idx - 2])
        op = toks[hex_idx - 1]
        actual_hex = int(toks[hex_idx], 16) & 0xFFFF
        in_a_hex: int | None = None
        in_b_hex: int | None = None
        if "|" in toks:
            pipe_idx = toks.index("|")
            if pipe_idx + 1 < len(toks) and toks[pipe_idx + 1].startswith("0x"):
                in_a_hex = int(toks[pipe_idx + 1], 16) & 0xFFFF
            if pipe_idx + 3 < len(toks) and toks[pipe_idx + 3].startswith("0x"):
                in_b_hex = int(toks[pipe_idx + 3], 16) & 0xFFFF
        case_id = int(toks[0])
        row = RTLLaneRow(
            lane=lane,
            op=op,
            actual_hex=actual_hex,
            in_a_hex=in_a_hex,
            in_b_hex=in_b_hex,
        )
        prior = rows[case_id].get(lane)
        if prior is not None and prior != row:
            raise AssertionError(
                f"case {case_id} lane {lane}: inconsistent rtl rows "
                f"{prior!r} vs {row!r}"
            )
        rows[case_id][lane] = row
    return dict(rows)


# ----------------------------------------------------------------
#  Dispatch mirror of test_vpu_vectors_file
# ----------------------------------------------------------------

def _dispatch(model: VectorEngineModel, case: VPUTestCase) -> list[int]:
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
        return [imm_hex] * (2 * num_lanes)

    a_bits = [_h(h) for h in case.vec_a]
    b_bits = [_h(h) for h in case.vec_b] if case.vec_b else []

    if op in _COL_OPS:
        return model.execute(op, a_vec=a_bits)
    if op in _ROW_REDUCE_OPS or op == "mov":
        return model.execute(op, a_vec=a_bits)
    if op in _BIN_OPS:
        return model.execute(op, a_vec=a_bits, b_vec=b_bits)
    return model.execute(op, a_vec=a_bits)


def _expected_lane_inputs(
    case: VPUTestCase, lane_idx: int
) -> tuple[int | None, int | None]:
    """Return the `vpu_vectors.txt` inputs that should feed `lane_idx`
    for non-VLI ops. VLI ops are special because the snapshot's inA/inB
    columns (if ever populated) are not a stable projection of
    `case.vec_a` / `case.vec_b` — the `vec_a[0]` slot is the immediate,
    not a lane input — so we return `(None, None)` for them and skip
    the per-lane input cross-check. Everything else maps 1:1: `vec_a`
    / `vec_b` entries are the lane inputs."""
    if case.vpu_op in _VLI_OPS:
        return None, None
    a_bits = [_h(h) for h in case.vec_a]
    b_bits = [_h(h) for h in case.vec_b] if case.vec_b else []
    want_a = a_bits[lane_idx] if lane_idx < len(a_bits) else None
    want_b = b_bits[lane_idx] if lane_idx < len(b_bits) else None
    return want_a, want_b


def _gather_rtl_preconditions(
    case: VPUTestCase,
    rtl_lanes: dict[int, RTLLaneRow],
) -> list[str]:
    """Validate that the RTL snapshot row still describes the same op
    (and, when the optional inA/inB columns are present, the same
    inputs) as the current vector file. See module docstring for what
    this does and does not prove."""
    out: list[str] = []
    for lane_idx, row in sorted(rtl_lanes.items()):
        if row.op != case.vpu_op:
            out.append(
                f"lane {lane_idx}: snapshot op={row.op!r}, "
                f"current case op={case.vpu_op!r}"
            )
        want_a, want_b = _expected_lane_inputs(case, lane_idx)
        if (
            want_a is not None
            and row.in_a_hex is not None
            and row.in_a_hex != want_a
        ):
            out.append(
                f"lane {lane_idx}: snapshot inA=0x{row.in_a_hex:04X}, "
                f"current inA=0x{want_a:04X}"
            )
        if (
            want_b is not None
            and row.in_b_hex is not None
            and row.in_b_hex != want_b
        ):
            out.append(
                f"lane {lane_idx}: snapshot inB=0x{row.in_b_hex:04X}, "
                f"current inB=0x{want_b:04X}"
            )
    return out


def _gather_rtl_mismatches(
    model: VectorEngineModel,
    case: VPUTestCase,
    rtl_lanes: dict[int, RTLLaneRow],
) -> list[str]:
    """Compare funct model output to RTL actual at each lane from the
    snapshot. Skips lanes past the model's output length (shouldn't
    happen for any op in the current file, but keeps the check robust
    if the file is regenerated with more replicated lanes)."""
    got = _dispatch(model, case)
    out: list[str] = []
    for lane_idx, row in sorted(rtl_lanes.items()):
        if lane_idx >= len(got):
            continue
        g = got[lane_idx] & 0xFFFF
        if g != row.actual_hex:
            out.append(
                f"lane {lane_idx}: got 0x{g:04X}, want 0x{row.actual_hex:04X}"
            )
    return out


# ----------------------------------------------------------------
#  Pytest plumbing
# ----------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> VectorEngineModel:
    return VectorEngineModel(VectorParams())


def _load_vpu_cases() -> list[VPUTestCase]:
    if not REF_VECTORS.is_file():
        return []
    return [c for c in read_cases(str(REF_VECTORS)) if c.vpu_op not in _SKIP_OPS]


_VPU_CASES = _load_vpu_cases()
_RTL_ROWS = _parse_rtl_rows(REF_RTL) if REF_RTL.is_file() else {}


def _case_ids_aligned() -> list[VPUTestCase]:
    """Only cases that exist in both the vpu_vectors.txt golden and
    the rtl snapshot."""
    return [c for c in _VPU_CASES if c.case_id in _RTL_ROWS]


def _pp_id(case: VPUTestCase) -> str:
    return f"{case.case_id}-{case.vpu_op}"


_STRICT_CASES = [
    c for c in _case_ids_aligned() if c.vpu_op not in _KNOWN_DIVERGENT_OPS
]
_XFAIL_CASES = [
    c for c in _case_ids_aligned() if c.vpu_op in _KNOWN_DIVERGENT_OPS
]


def test_rtl_snapshot_present() -> None:
    assert REF_RTL.is_file(), f"Missing RTL snapshot file {REF_RTL}"
    assert _RTL_ROWS, f"{REF_RTL} parsed into zero cases"


def test_vpu_and_rtl_case_alignment() -> None:
    vpu_ids = {c.case_id for c in _VPU_CASES}
    rtl_ids = set(_RTL_ROWS.keys())
    missing_in_rtl = vpu_ids - rtl_ids
    missing_in_vpu = rtl_ids - vpu_ids
    assert not missing_in_rtl, (
        f"vpu_vectors.txt has case_ids not in the RTL snapshot: "
        f"{sorted(missing_in_rtl)} — regenerate rtl_actual_outputs.txt"
    )
    assert not missing_in_vpu, (
        f"RTL snapshot has case_ids not in vpu_vectors.txt: "
        f"{sorted(missing_in_vpu)}"
    )


@pytest.mark.parametrize("case", _STRICT_CASES, ids=[_pp_id(c) for c in _STRICT_CASES])
def test_rtl_diff_strict(
    model: VectorEngineModel, case: VPUTestCase
) -> None:
    rtl_lanes = _RTL_ROWS[case.case_id]
    preconditions = _gather_rtl_preconditions(case, rtl_lanes)
    if preconditions:
        head = "\n  ".join(preconditions[:8])
        more = (
            f"\n  ... {len(preconditions) - 8} more"
            if len(preconditions) > 8
            else ""
        )
        pytest.fail(
            f"case {case.case_id} ({case.vpu_op}) snapshot corpus drift "
            f"vs current {REF_VECTORS.name}; regenerate {REF_RTL.name} "
            f"before trusting Tier 3:\n  {head}{more}"
        )
    mismatches = _gather_rtl_mismatches(model, case, rtl_lanes)
    if mismatches:
        head = "\n  ".join(mismatches[:8])
        more = f"\n  ... {len(mismatches) - 8} more" if len(mismatches) > 8 else ""
        pytest.fail(
            f"case {case.case_id} ({case.vpu_op}) funct model != RTL "
            f"{REF_RTL.name}:\n  {head}{more}"
        )


@pytest.mark.xfail(
    reason=(
        "tanh: funct model routes through _legacy_math_fallback (Python "
        "math.tanh), which is not bit-for-bit identical to the HardFloat "
        "iterative approximation in sp26-fp-units/vpuFUnits/. "
        "Exit gate: land §6 Group D HardFloat snapshot pipeline."
    ),
    strict=True,
)
@pytest.mark.parametrize("case", _XFAIL_CASES, ids=[_pp_id(c) for c in _XFAIL_CASES])
def test_rtl_diff_known_divergent(
    model: VectorEngineModel, case: VPUTestCase
) -> None:
    rtl_lanes = _RTL_ROWS[case.case_id]
    # Precondition failures are NOT the "known divergent" behavior; they
    # indicate the snapshot itself no longer matches the current corpus,
    # which invalidates both the xfail and the pass paths. Surface them
    # as a hard assertion so the xfail marker doesn't silently swallow
    # "the test fixture is broken".
    preconditions = _gather_rtl_preconditions(case, rtl_lanes)
    assert not preconditions, (
        f"case {case.case_id} ({case.vpu_op}) no longer matches the "
        f"corpus captured in {REF_RTL.name}; regenerate the snapshot "
        f"before using Tier 3 as evidence either way:\n  "
        + "\n  ".join(preconditions[:8])
    )
    mismatches = _gather_rtl_mismatches(model, case, rtl_lanes)
    assert not mismatches, (
        f"case {case.case_id} ({case.vpu_op}) funct model != RTL "
        f"(known divergent, will be fixed by §6 Group D)"
    )
