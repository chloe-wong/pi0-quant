"""Bit-exact tests for funct_models_vector.lane_boxes.col_add_vec.

ColAddVec accumulates BF16 inputs through `AddRecFN(8, 24)` — bit-
equivalent to IEEE-754 binary32 RNE add. The funct model carries the
accumulator as raw FP32 bit patterns and uses the same Python
`numpy.float32` add for verification, which is the most-correct
reference for this exact widened-recFN datapath.

Critical things this file pins down:

1. **Per-lane FP32 add, not BF16.** Calling the impl with two BF16
   inputs that round differently as BF16 vs FP32 must produce the
   FP32-precision result. Stretches the precision claim from §4.3 of
   IMPLEMENTATION_PLAN.md.

2. **Multi-row streaming via the latch.** `step("csum", req)` lagged
   one cycle; the engine reads the previous cycle's `addResultNext`.
   Tests drive a hand-traced 4-row stream and a 128-row sweep.

3. **Freeze on `isDoneReadingColSum`.** When the engine asserts
   `isDoneReadingColSum`, `addResultNext` stops updating — the
   subsequent cycle's output should still be the value from the
   pre-freeze cycle, and the math result computed during the freeze
   must NOT contaminate the latch.

4. **Bit-pattern boundary.** `bVec` field is FP32 bit pattern (uint32),
   `result` field is FP32 bit pattern (uint32). The conversion to
   BF16 happens at the engine wrapper layer, not inside the lane_box.
"""

from __future__ import annotations

import struct
from typing import List

import numpy as np
import pytest

from funct_models_vector import bf16_utils as fp
from funct_models_vector.lane_boxes.col_add_vec import (
    ColAddReq,
    ColAddResp,
    ColAddVec,
)
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes

ZERO_FP32 = [0] * N        # FP32 +0 (= recFNFromFN(8, 24, 0) once decoded)


# ------------------------------------------------------------
#                  helpers
# ------------------------------------------------------------

def _bf16_bits_to_fp32_bits(b: int) -> int:
    return (b & 0xFFFF) << 16


def _fp32_bits_to_f32(u: int) -> np.float32:
    return np.frombuffer(np.uint32(u & 0xFFFFFFFF).tobytes(), dtype=np.float32)[0]


def _f32_to_fp32_bits(x: np.float32) -> int:
    return int(np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0])


def _bf16_random_finite(n: int, rng: np.random.Generator) -> List[int]:
    """Random BF16 bit patterns with finite (non-inf/NaN) exponents."""
    out: List[int] = []
    while len(out) < n:
        b = int(rng.integers(0, 1 << 16))
        if ((b >> 7) & 0xFF) == 0xFF:
            continue
        out.append(b)
    return out


def _np_f32_accumulate(rows: List[List[int]]) -> List[int]:
    """Reference: per-lane numpy.float32 RNE accumulator over BF16 rows.
    Matches the AddRecFN(8, 24) RNE-at-every-add semantic."""
    n = len(rows[0])
    acc = [np.float32(0.0)] * n
    for row in rows:
        for i in range(n):
            af = _fp32_bits_to_f32(_bf16_bits_to_fp32_bits(row[i]))
            acc[i] = np.float32(acc[i] + af)
    return [_f32_to_fp32_bits(x) for x in acc]


# ------------------------------------------------------------
#                  compute_now hand-computed
# ------------------------------------------------------------

def test_compute_now_a_plus_zero():
    box = ColAddVec(P)
    a_bf16 = [0x3F80] * N  # 1.0
    r = box.compute_now(ColAddReq(aVec=a_bf16, bVec=ZERO_FP32))
    expected = _bf16_bits_to_fp32_bits(0x3F80)
    assert r.result == [expected] * N


def test_compute_now_a_plus_b_lane_wise():
    box = ColAddVec(P)
    a_bf16 = [0x3F80] * N                                    # 1.0
    b_fp32 = [_bf16_bits_to_fp32_bits(0x3F80)] * N           # also 1.0 in FP32 form
    r = box.compute_now(ColAddReq(aVec=a_bf16, bVec=b_fp32))
    expected = _f32_to_fp32_bits(np.float32(1.0) + np.float32(1.0))   # 2.0
    assert r.result == [expected] * N


def test_compute_now_distinct_lanes():
    box = ColAddVec(P)
    a_bf16 = list(range(0x3F00, 0x3F00 + N))    # 16 distinct BF16 patterns
    b_fp32 = [_bf16_bits_to_fp32_bits(0x4000)] * N    # 2.0
    r = box.compute_now(ColAddReq(aVec=a_bf16, bVec=b_fp32))
    expected = []
    for i in range(N):
        af = _fp32_bits_to_f32(_bf16_bits_to_fp32_bits(a_bf16[i]))
        bf = _fp32_bits_to_f32(b_fp32[i])
        expected.append(_f32_to_fp32_bits(np.float32(af + bf)))
    assert r.result == expected


def test_compute_now_ignores_is_done_reading():
    """compute_now is the pure-math path — the freeze flag is a state
    thing that only affects step()."""
    box = ColAddVec(P)
    a = [0x3F80] * N
    b = ZERO_FP32
    r1 = box.compute_now(ColAddReq(aVec=a, bVec=b, isDoneReadingColSum=False))
    r2 = box.compute_now(ColAddReq(aVec=a, bVec=b, isDoneReadingColSum=True))
    assert r1.result == r2.result


# ------------------------------------------------------------
#                  random sweep vs np.float32
# ------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_compute_now_random_matches_np_f32_per_lane(seed: int):
    rng = np.random.default_rng(seed)
    box = ColAddVec(P)
    for _ in range(64):
        a_bf16 = _bf16_random_finite(N, rng)
        # Random FP32 bVec: build it from random BF16 patterns first so
        # everything stays in the BF16-zero-padded subspace (the only
        # values the engine ever feeds back are sums of zero-padded BF16
        # rows, so non-zero low-16 bits in bVec are reachable).
        b_seed = _bf16_random_finite(N, rng)
        b_fp32 = [_bf16_bits_to_fp32_bits(x) for x in b_seed]
        # Now mutate the low 16 bits of each lane to exercise the full
        # FP32 mantissa. Skip lanes where the mutation pushes into
        # NaN/inf so the np.float32 reference doesn't go infinite.
        for i in range(N):
            extra = int(rng.integers(0, 1 << 16))
            cand = b_fp32[i] | extra
            if ((cand >> 23) & 0xFF) == 0xFF:
                continue
            b_fp32[i] = cand
        r = box.compute_now(ColAddReq(aVec=a_bf16, bVec=b_fp32))
        expected = []
        for i in range(N):
            af = _fp32_bits_to_f32(_bf16_bits_to_fp32_bits(a_bf16[i]))
            bf = _fp32_bits_to_f32(b_fp32[i])
            expected.append(_f32_to_fp32_bits(np.float32(af + bf)))
        # np.float32 may produce inf for sums that overflow; the impl
        # clamps the same way through fp32_bits_add → struct.pack with
        # OverflowError → ±inf bits. Both should agree.
        assert r.result == expected


# ------------------------------------------------------------
#                  step() — cycle-accurate latch + freeze
# ------------------------------------------------------------

def test_step_first_cycle_is_bubble():
    box = ColAddVec(P)
    req = ColAddReq(aVec=[0x3F80] * N, bVec=ZERO_FP32)
    assert box.step("csum", req) is None


def test_step_second_cycle_returns_first_sum():
    box = ColAddVec(P)
    req = ColAddReq(aVec=[0x3F80] * N, bVec=ZERO_FP32)
    box.step("csum", req)
    out = box.step("csum", None)
    assert out is not None
    expected_fp32 = _bf16_bits_to_fp32_bits(0x3F80)   # 1.0 + 0.0 = 1.0 (FP32 bits)
    assert out.result == [expected_fp32] * N


def test_step_two_row_running_sum():
    """Manually feed the 1-cycle-lagged accumulator: row0 then row1.
    Cycle 0: enqueue row0 with bVec=0, output bubble.
    Cycle 1: enqueue row1 with bVec=row0, output = row0.
    Cycle 2: idle, output = row0+row1.
    """
    box = ColAddVec(P)
    row0 = [0x3F80] * N      # 1.0
    row1 = [0x4000] * N      # 2.0

    bubble = box.step("csum", ColAddReq(aVec=row0, bVec=ZERO_FP32))
    assert bubble is None

    row0_fp32 = [_bf16_bits_to_fp32_bits(0x3F80)] * N
    out1 = box.step("csum", ColAddReq(aVec=row1, bVec=row0_fp32))
    assert out1 is not None and out1.result == row0_fp32   # latched row0

    out2 = box.step("csum", None)
    assert out2 is not None
    expected = _f32_to_fp32_bits(np.float32(1.0) + np.float32(2.0))  # 3.0
    assert out2.result == [expected] * N


def test_step_streaming_128_rows_matches_np_f32():
    """Drive 128 rows of random BF16 through the lane_box like the
    engine does (combinational `bVec ← addResultNext` feedback) and
    compare the final accumulator value to a pure np.float32 reference.

    Use `peek_result()` to read the latched value at the start of each
    cycle — this mirrors `csum.io.req.bits.bVec := csum.io.resp.bits.
    result` in `VectorEngine.scala:319`, which is a combinational wire
    feeding the lane_box's adders the latched output that was sampled
    at the end of the previous cycle.
    """
    rng = np.random.default_rng(99)
    box = ColAddVec(P)
    rows: List[List[int]] = [_bf16_random_finite(N, rng) for _ in range(128)]

    for row in rows:
        bVec = box.peek_result()       # = ZERO_FP32 on the first iteration
        box.step("csum", ColAddReq(aVec=row, bVec=bVec))

    expected = _np_f32_accumulate(rows)
    assert box.peek_result() == expected


def test_step_freeze_preserves_value_during_is_done():
    """`isDoneReadingColSum=True` must NOT update `addResultNext`. The
    output two cycles later (when the freeze cycle's valid bit
    propagates) must still be the pre-freeze sum, NOT the math the
    lane_box computed during the freeze cycle."""
    box = ColAddVec(P)
    row0 = [0x3F80] * N      # 1.0
    row1 = [0x4000] * N      # 2.0

    box.step("csum", ColAddReq(aVec=row0, bVec=ZERO_FP32))   # bubble
    row0_fp32 = [_bf16_bits_to_fp32_bits(0x3F80)] * N
    box.step("csum", ColAddReq(aVec=row1, bVec=row0_fp32))   # latch row0
    # At this point _latched should hold row0+row1 (3.0). One more
    # idle step to make sure that's the visible output.
    out = box.step("csum", None)
    assert out is not None
    expected_3 = _f32_to_fp32_bits(np.float32(3.0))
    assert out.result == [expected_3] * N

    # Now drive a freeze cycle with garbage on aVec/bVec — addResultNext
    # must keep the previous (3.0) value, not absorb the garbage sum.
    garbage_a = [0x7F00] * N    # ~big finite, intentionally unrelated
    garbage_b = [0x7F800000] * N    # +inf in FP32
    box.step("csum", ColAddReq(
        aVec=garbage_a, bVec=garbage_b, isDoneReadingColSum=True
    ))
    # Drain the valid bit one more cycle.
    out_frozen = box.step("csum", None)
    assert out_frozen is not None
    assert out_frozen.result == [expected_3] * N, (
        "addResultNext was overwritten during isDoneReadingColSum=True"
    )


def test_step_reset_clears_latch():
    box = ColAddVec(P)
    box.step("csum", ColAddReq(aVec=[0x3F80] * N, bVec=ZERO_FP32))
    box.step("csum", None)
    box.reset()
    # First call after reset should bubble out None (prev_valid is False).
    out = box.step("csum", None)
    assert out is None


def test_step_unknown_op_raises():
    box = ColAddVec(P)
    with pytest.raises(KeyError):
        box.step("rsum", None)


# ------------------------------------------------------------
#                  BF16 extraction sanity (engine-layer helper)
# ------------------------------------------------------------

def test_engine_layer_bf16_extraction_via_top_slice():
    """The engine wrapper extracts BF16 from the lane_box's FP32
    result via `bf16_upper_half_of_fp32_bits` (the raw upper-16-bit
    slice). Pin that the helper is what the engine should use."""
    box = ColAddVec(P)
    a_bf16 = [0x3F80, 0x3F81, 0x3F82, 0x3F83] * 4   # near-1.0
    r = box.compute_now(ColAddReq(aVec=a_bf16, bVec=ZERO_FP32))
    bf16_out = [fp.bf16_upper_half_of_fp32_bits(x) for x in r.result]
    # Adding zero is a no-op in FP32, so the BF16 top slice round-trips.
    assert bf16_out == a_bf16
