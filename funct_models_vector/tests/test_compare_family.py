"""Bit-exact tests for the Compare family lane_boxes:
  - PairWiseMax / PairWiseMin
  - RowMax / RowMin

Critical invariants exercised here:

1. The Scala `compareReturn{Max,Min}` is a **bitwise sign-magnitude
   compare**, not Python's `max()`. Ties return `b`, not `a`. Signed
   zeros are *distinct* under the ordering transform (+0 maps to
   0x8000, -0 maps to 0x7FFF), so max always picks +0 and min always
   picks -0 regardless of operand position — IEEE-correct, but a
   `max(float_a, float_b)` reference would be ambiguous because Python
   treats them as equal.

2. PairwiseMax/Min gate per-lane on `laneMask`: disabled lanes
   return `0x0000`, not the input.

3. RowMax/RowMin pair adjacent lanes `(2i, 2i+1)`, NOT split-half
   `(i, i+8)` like AddSubSumVec. The test mirrors that order.

4. The bitwise compare orders NaN into the BF16 number line; for a
   negative NaN (sign=1), bit-not gives a high ordered key, so a
   negative NaN compares larger than +inf. We don't invert this — the
   RTL doesn't either. The tests pin the resulting ordering explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from funct_models_vector.lane_boxes.pair_wise_max import (
    PairWiseMax,
    PairWiseMaxReq,
)
from funct_models_vector.lane_boxes.pair_wise_min import (
    PairWiseMin,
    PairWiseMinReq,
)
from funct_models_vector.lane_boxes.row_max import RowMax, RowMaxReq
from funct_models_vector.lane_boxes.row_min import RowMin, RowMinReq
from funct_models_vector.vector_params import VectorParams
from funct_models_vector import bf16_utils as fp


P = VectorParams()
N = P.num_lanes
ALL_LANES = (1 << N) - 1   # 0xFFFF for N=16


# ============================================================
#                    PairWiseMax
# ============================================================

def test_pairmax_simple():
    box = PairWiseMax(P)
    a = [0x4000] * N    # 2.0
    b = [0x4040] * N    # 3.0
    r = box.compute_now(PairWiseMaxReq(aVec=a, bVec=b))
    assert r.result == [0x4040] * N    # max = 3.0


def test_pairmax_picks_b_on_tie():
    """compareReturnMax: `Mux(aOrdered > bOrdered, a, b)` returns b on tie."""
    box = PairWiseMax(P)
    a = [0x3F80] * N  # 1.0
    b = [0x3F80] * N  # 1.0
    r = box.compute_now(PairWiseMaxReq(aVec=a, bVec=b))
    # Both are bitwise equal → returns b.
    assert r.result == b


def test_pairmax_signed_zero_picks_positive_zero():
    """The bitwise ordering transform distinguishes ±0:
        ordered(+0=0x0000) = 0x0000 ^ 0x8000 = 0x8000
        ordered(-0=0x8000) = ~0x8000        = 0x7FFF
    so +0 > -0 under ordering and max always returns +0 regardless of
    which side it's on. (IEEE-correct: max(+0,-0) == +0.)"""
    box = PairWiseMax(P)
    r1 = box.compute_now(PairWiseMaxReq(aVec=[0x0000] * N, bVec=[0x8000] * N))
    assert r1.result == [0x0000] * N      # +0 wins
    r2 = box.compute_now(PairWiseMaxReq(aVec=[0x8000] * N, bVec=[0x0000] * N))
    assert r2.result == [0x0000] * N      # +0 wins (now coming from b)


def test_pairmax_negative_lanes():
    box = PairWiseMax(P)
    a = [0xC000] * N    # -2.0
    b = [0xBF80] * N    # -1.0
    r = box.compute_now(PairWiseMaxReq(aVec=a, bVec=b))
    assert r.result == [0xBF80] * N    # max(-2, -1) = -1


def test_pairmax_lanemask_disables_lanes():
    box = PairWiseMax(P)
    a = [0x4000] * N    # 2.0
    b = [0x4040] * N    # 3.0
    # Enable only lanes 0, 5, 10, 15.
    mask = (1 << 0) | (1 << 5) | (1 << 10) | (1 << 15)
    r = box.compute_now(PairWiseMaxReq(aVec=a, bVec=b, laneMask=mask))
    expected = [0x4040 if (mask >> i) & 1 else 0x0000 for i in range(N)]
    assert r.result == expected
    assert r.laneMask == mask


def test_pairmax_lanemask_zero_returns_all_zero():
    box = PairWiseMax(P)
    r = box.compute_now(
        PairWiseMaxReq(aVec=[0x4000] * N, bVec=[0x4040] * N, laneMask=0)
    )
    assert r.result == [0x0000] * N


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pairmax_random_matches_compare_helper(seed: int):
    """compute_now should equal lane-wise compare_return_max."""
    rng = np.random.default_rng(seed)
    box = PairWiseMax(P)
    for _ in range(64):
        a = [int(rng.integers(0, 1 << 16)) for _ in range(N)]
        b = [int(rng.integers(0, 1 << 16)) for _ in range(N)]
        r = box.compute_now(PairWiseMaxReq(aVec=a, bVec=b))
        assert r.result == [fp.compare_return_max(a[i], b[i]) for i in range(N)]


def test_step_pairmax_latency_one_cycle():
    box = PairWiseMax(P)
    req = PairWiseMaxReq(aVec=[0x4000] * N, bVec=[0x4040] * N)
    assert box.step("pairmax", req) is None
    out = box.step("pairmax", None)
    assert out is not None and out.result == [0x4040] * N


def test_step_pairmax_cmax_independent_queues():
    """pairmax and cmax share an implementation but have independent
    per-op latency queues — submitting cmax must not surface a pending
    pairmax result."""
    box = PairWiseMax(P)
    pm_req = PairWiseMaxReq(aVec=[0x4000] * N, bVec=[0x4040] * N)
    cm_req = PairWiseMaxReq(aVec=[0x3F80] * N, bVec=[0xBF80] * N)
    # Submit pairmax — should bubble out None.
    assert box.step("pairmax", pm_req) is None
    # Submit cmax — also bubble. Must NOT surface pairmax's pending result.
    out_cm = box.step("cmax", cm_req)
    assert out_cm is None
    # Pop pairmax — surfaces the pairmax result, not cmax.
    out_pm = box.step("pairmax", None)
    assert out_pm is not None and out_pm.result == [0x4040] * N
    # Pop cmax — surfaces cmax's result.
    out_cm2 = box.step("cmax", None)
    assert out_cm2 is not None and out_cm2.result == [0x3F80] * N    # 1.0 > -1.0


# ============================================================
#                    PairWiseMin
# ============================================================

def test_pairmin_simple():
    box = PairWiseMin(P)
    a = [0x4000] * N    # 2.0
    b = [0x4040] * N    # 3.0
    r = box.compute_now(PairWiseMinReq(aVec=a, bVec=b))
    assert r.result == [0x4000] * N    # min = 2.0


def test_pairmin_picks_b_on_tie():
    box = PairWiseMin(P)
    a = [0x3F80] * N
    b = [0x3F80] * N
    r = box.compute_now(PairWiseMinReq(aVec=a, bVec=b))
    assert r.result == b


def test_pairmin_signed_zero_picks_negative_zero():
    """Symmetric to the max test: under the bitwise ordering -0 < +0,
    so min always returns -0. (IEEE-correct: min(+0,-0) == -0.)"""
    box = PairWiseMin(P)
    r1 = box.compute_now(PairWiseMinReq(aVec=[0x0000] * N, bVec=[0x8000] * N))
    assert r1.result == [0x8000] * N      # -0 wins
    r2 = box.compute_now(PairWiseMinReq(aVec=[0x8000] * N, bVec=[0x0000] * N))
    assert r2.result == [0x8000] * N      # -0 wins (now coming from a)


def test_pairmin_negative_lanes():
    box = PairWiseMin(P)
    a = [0xC000] * N    # -2.0
    b = [0xBF80] * N    # -1.0
    r = box.compute_now(PairWiseMinReq(aVec=a, bVec=b))
    assert r.result == [0xC000] * N    # min(-2, -1) = -2


def test_pairmin_lanemask():
    box = PairWiseMin(P)
    mask = (1 << 1) | (1 << 4) | (1 << 11)
    r = box.compute_now(
        PairWiseMinReq(aVec=[0x4000] * N, bVec=[0x4040] * N, laneMask=mask)
    )
    expected = [0x4000 if (mask >> i) & 1 else 0x0000 for i in range(N)]
    assert r.result == expected


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
def test_pairmin_random_matches_compare_helper(seed: int):
    rng = np.random.default_rng(seed)
    box = PairWiseMin(P)
    for _ in range(64):
        a = [int(rng.integers(0, 1 << 16)) for _ in range(N)]
        b = [int(rng.integers(0, 1 << 16)) for _ in range(N)]
        r = box.compute_now(PairWiseMinReq(aVec=a, bVec=b))
        assert r.result == [fp.compare_return_min(a[i], b[i]) for i in range(N)]


def test_step_pairmin_cmin_independent_queues():
    box = PairWiseMin(P)
    pm_req = PairWiseMinReq(aVec=[0x4000] * N, bVec=[0x4040] * N)
    cm_req = PairWiseMinReq(aVec=[0x3F80] * N, bVec=[0xBF80] * N)
    assert box.step("pairmin", pm_req) is None
    assert box.step("cmin", cm_req) is None
    out_pm = box.step("pairmin", None)
    assert out_pm is not None and out_pm.result == [0x4000] * N    # min(2, 3) = 2
    out_cm = box.step("cmin", None)
    assert out_cm is not None and out_cm.result == [0xBF80] * N    # min(1, -1) = -1


# ============================================================
#                    RowMax
# ============================================================

def _ref_row_max_tree(a: list[int]) -> int:
    """Adjacent-pair tree mirror of RowMax.scala."""
    m8 = [fp.compare_return_max(a[2 * i], a[2 * i + 1]) for i in range(8)]
    m4 = [fp.compare_return_max(m8[2 * i], m8[2 * i + 1]) for i in range(4)]
    m2 = [fp.compare_return_max(m4[2 * i], m4[2 * i + 1]) for i in range(2)]
    return fp.compare_return_max(m2[0], m2[1])


def test_rowmax_uniform():
    box = RowMax(P)
    a = [0x4000] * N    # all 2.0
    r = box.compute_now(RowMaxReq(aVec=a))
    assert r.result == [0x4000] * N


def test_rowmax_finds_global_max():
    box = RowMax(P)
    a = [0x3F80] * N    # 1.0
    a[7] = 0x4080       # 4.0 — the max
    r = box.compute_now(RowMaxReq(aVec=a))
    assert r.result == [0x4080] * N


def test_rowmax_negatives():
    box = RowMax(P)
    a = [0xC080] * N    # -4.0
    a[3] = 0xBF80       # -1.0 — the max (closest to zero)
    r = box.compute_now(RowMaxReq(aVec=a))
    assert r.result == [0xBF80] * N


def test_rowmax_signed_zero_b_on_tie():
    """The full reduction collapses pairs in adjacent order; both +0
    and -0 land on -0 because every compare returns b on tie and the
    pairing keeps -0 in the right slot. This pins the actual collapsed
    value rather than relying on a hand-derived order."""
    box = RowMax(P)
    a = [0x0000, 0x8000] * 8     # alternating +0 / -0
    r = box.compute_now(RowMaxReq(aVec=a))
    expected = _ref_row_max_tree(a)
    assert r.result == [expected] * N


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_rowmax_random_matches_tree(seed: int):
    rng = np.random.default_rng(seed)
    box = RowMax(P)
    for _ in range(64):
        a = [int(rng.integers(0, 1 << 16)) for _ in range(N)]
        r = box.compute_now(RowMaxReq(aVec=a))
        assert r.result == [_ref_row_max_tree(a)] * N


def test_step_rowmax_latency_one_cycle():
    box = RowMax(P)
    a = [0x3F80] * N
    a[5] = 0x4040  # 3.0
    req = RowMaxReq(aVec=a)
    assert box.step("rmax", req) is None
    out = box.step("rmax", None)
    assert out is not None and out.result == [0x4040] * N


def test_rowmax_reset_clears_pipe():
    box = RowMax(P)
    box.step("rmax", RowMaxReq(aVec=[0x4000] * N))
    box.reset()
    assert box.step("rmax", None) is None


# ============================================================
#                    RowMin
# ============================================================

def _ref_row_min_tree(a: list[int]) -> int:
    m8 = [fp.compare_return_min(a[2 * i], a[2 * i + 1]) for i in range(8)]
    m4 = [fp.compare_return_min(m8[2 * i], m8[2 * i + 1]) for i in range(4)]
    m2 = [fp.compare_return_min(m4[2 * i], m4[2 * i + 1]) for i in range(2)]
    return fp.compare_return_min(m2[0], m2[1])


def test_rowmin_uniform():
    box = RowMin(P)
    a = [0x4000] * N
    r = box.compute_now(RowMinReq(aVec=a))
    assert r.result == [0x4000] * N


def test_rowmin_finds_global_min():
    box = RowMin(P)
    a = [0x4080] * N    # 4.0
    a[10] = 0x3F80      # 1.0 — the min
    r = box.compute_now(RowMinReq(aVec=a))
    assert r.result == [0x3F80] * N


def test_rowmin_negatives():
    box = RowMin(P)
    a = [0xBF80] * N    # -1.0
    a[2] = 0xC080       # -4.0 — the min
    r = box.compute_now(RowMinReq(aVec=a))
    assert r.result == [0xC080] * N


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
def test_rowmin_random_matches_tree(seed: int):
    rng = np.random.default_rng(seed)
    box = RowMin(P)
    for _ in range(64):
        a = [int(rng.integers(0, 1 << 16)) for _ in range(N)]
        r = box.compute_now(RowMinReq(aVec=a))
        assert r.result == [_ref_row_min_tree(a)] * N


def test_step_rowmin_latency_one_cycle():
    box = RowMin(P)
    a = [0x4080] * N
    a[5] = 0x3F80
    req = RowMinReq(aVec=a)
    assert box.step("rmin", req) is None
    out = box.step("rmin", None)
    assert out is not None and out.result == [0x3F80] * N


# ============================================================
#               Cross-validation against torch
# ============================================================

def _bf16_random_finite(n: int, rng: np.random.Generator) -> list[int]:
    """Random BF16 bit patterns excluding inf/NaN (exp == 0xFF)."""
    out: list[int] = []
    while len(out) < n:
        b = int(rng.integers(0, 1 << 16))
        if ((b >> 7) & 0xFF) == 0xFF:
            continue
        out.append(b)
    return out


def _bf16_bits_to_signed_float(b: int) -> float:
    """Decode a finite (non-inf/NaN) BF16 bit pattern to Python float
    via the upper-16-bits-of-FP32 convention. Exact for finite BF16."""
    fp32 = (b & 0xFFFF) << 16
    import struct
    return struct.unpack("<f", struct.pack("<I", fp32))[0]


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_pairmax_random_matches_python_max_on_finite(seed: int):
    """Sanity second-source. For finite BF16 (no signed-zero ties),
    Python's `max(float_a, float_b)` matches the bitwise compare. We
    skip lanes where a/b decode to the same float, because the
    sign-of-zero / b-on-tie distinction makes them legitimately differ."""
    rng = np.random.default_rng(seed)
    box = PairWiseMax(P)
    for _ in range(64):
        a = _bf16_random_finite(N, rng)
        b = _bf16_random_finite(N, rng)
        r = box.compute_now(PairWiseMaxReq(aVec=a, bVec=b))
        for i in range(N):
            af = _bf16_bits_to_signed_float(a[i])
            bf = _bf16_bits_to_signed_float(b[i])
            if af == bf:
                continue       # b-on-tie + signed-zero ambiguity
            expected_bits = a[i] if af > bf else b[i]
            assert r.result[i] == expected_bits, (
                f"lane {i}: a=0x{a[i]:04x} ({af}) b=0x{b[i]:04x} ({bf}) "
                f"got=0x{r.result[i]:04x}"
            )


@pytest.mark.parametrize("seed", [30, 31, 32])
def test_rowmax_random_matches_python_max_on_finite(seed: int):
    rng = np.random.default_rng(seed)
    box = RowMax(P)
    for _ in range(64):
        a = _bf16_random_finite(N, rng)
        r = box.compute_now(RowMaxReq(aVec=a))
        floats = [_bf16_bits_to_signed_float(x) for x in a]
        max_f = max(floats)
        # Skip when ties exist — b-on-tie ambiguity could pick a
        # different bit pattern than max() does.
        if floats.count(max_f) > 1:
            continue
        max_bits = a[floats.index(max_f)]
        assert r.result[0] == max_bits


@pytest.mark.parametrize("seed", [40, 41, 42])
def test_rowmin_random_matches_python_min_on_finite(seed: int):
    rng = np.random.default_rng(seed)
    box = RowMin(P)
    for _ in range(64):
        a = _bf16_random_finite(N, rng)
        r = box.compute_now(RowMinReq(aVec=a))
        floats = [_bf16_bits_to_signed_float(x) for x in a]
        min_f = min(floats)
        if floats.count(min_f) > 1:
            continue
        min_bits = a[floats.index(min_f)]
        assert r.result[0] == min_bits
