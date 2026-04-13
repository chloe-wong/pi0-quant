"""Bit-exact tests for funct_models_vector.lane_boxes.fp8_pack.

`FP8Pack` is the only **2 → 1 phased** lane_box on the BF16 side and
the first stateful one in the funct model whose state survives idle
cycles. This test file pins down four classes of behavior:

1. **Per-byte conversion math** — `bf16_to_e4m3_byte` is already
   validated against `RVFP8PackTest.goldenFp8ByteFromBf16`, but here
   we exercise the lane_box's `convert_row` path with edge cases
   (zero, ±inf, NaN, subnormals, reserved-NaN clamp boundary) so the
   wiring + iteration loop are pinned too.

2. **Output byte layout** — `FP8Pack.scala:122-125` packs two FP8
   bytes per 16-bit slot, low byte at bits[7:0], high byte at
   bits[15:8]. The tests use 16 distinct fp8 byte values across the
   two halves so any swap, off-by-one, or endian flip fails loudly.

3. **2 → 1 phased timing** — visible latency is 1, but the phase
   state machine adds another cycle of "no output" at the first
   pulse, so output appears on cycle K+2 of the cycle-K-second-pulse.
   Tests cover: first-pulse bubble, second-pulse-still-bubble (the
   register doesn't update io.resp until the next cycle), idle
   between halves preserving phase, back-to-back 4-input streaming.

4. **Reserved-NaN clamp + special-value handling** — the
   `(finalExpAdjusted == 8 && mantFP8 == 7)` corner cancels what
   would otherwise be the FP8 NaN bit pattern. Tests reach this case
   directly via a hand-picked BF16 value and verify the clamp.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from funct_models_vector import fp8_e4m3 as fp8
from funct_models_vector.lane_boxes.fp8_pack import (
    FP8Pack,
    FP8PackReq,
    FP8PackResp,
)
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes
HALF = N // 2


# ------------------------------------------------------------
#                  helpers
# ------------------------------------------------------------

def _row(fill: int) -> List[int]:
    return [fill] * N


def _slot(low_byte: int, high_byte: int) -> int:
    return ((high_byte & 0xFF) << 8) | (low_byte & 0xFF)


def _golden_packed_row(
    low_bytes: List[int], high_bytes: List[int]
) -> List[int]:
    """Mirror FP8Pack.scala:122-125 byte layout."""
    out = [0] * N
    for j in range(HALF):
        out[j] = _slot(low_bytes[2 * j], low_bytes[2 * j + 1])
        out[j + HALF] = _slot(high_bytes[2 * j], high_bytes[2 * j + 1])
    return out


def _bf16_random_finite(n: int, rng: np.random.Generator) -> List[int]:
    """Random BF16 bit patterns with finite (non-inf/NaN) exponents."""
    out: List[int] = []
    while len(out) < n:
        b = int(rng.integers(0, 1 << 16))
        if ((b >> 7) & 0xFF) == 0xFF:
            continue
        out.append(b)
    return out


# ------------------------------------------------------------
#                  convert_row — pure per-row math
# ------------------------------------------------------------

def test_convert_row_zero_in_zero_out():
    box = FP8Pack(P)
    bytes_out = box.convert_row(FP8PackReq(xVec=_row(0x0000), expShift=0))
    assert bytes_out == [0x00] * N


def test_convert_row_one_dot_zero_at_expshift_zero():
    box = FP8Pack(P)
    # BF16 1.0 = 0x3F80, exp=127, mant=0. unbExp=0, expAdj=0, mant8=0x80 →
    # rounded=0x08, normCarry=False, mantFP8=0. exp_fp8 = 0+7 = 7.
    # packed = (0<<7)|(7<<3)|0 = 0x38. (E4M3 1.0)
    bytes_out = box.convert_row(FP8PackReq(xVec=_row(0x3F80), expShift=0))
    assert bytes_out == [0x38] * N


def test_convert_row_inf_clamps_to_max_finite():
    box = FP8Pack(P)
    # +inf → 0x7E, -inf → 0xFE
    pos_inf = box.convert_row(FP8PackReq(xVec=_row(0x7F80), expShift=0))
    neg_inf = box.convert_row(FP8PackReq(xVec=_row(0xFF80), expShift=0))
    assert pos_inf == [0x7E] * N
    assert neg_inf == [0xFE] * N


def test_convert_row_nan_to_zero():
    box = FP8Pack(P)
    # BF16 NaN = exp=0xFF, mant!=0
    bytes_out = box.convert_row(FP8PackReq(xVec=_row(0x7FC0), expShift=0))
    assert bytes_out == [0x00] * N


def test_convert_row_subnormal_to_zero():
    box = FP8Pack(P)
    # BF16 subnormal = exp=0, mant!=0
    bytes_out = box.convert_row(FP8PackReq(xVec=_row(0x0001), expShift=0))
    assert bytes_out == [0x00] * N


def test_convert_row_distinct_lanes_match_helper():
    """Per-lane independence — each lane should call
    `bf16_to_e4m3_byte` with its own bit pattern, no cross-talk."""
    box = FP8Pack(P)
    # 16 distinct near-1.0 BF16 patterns
    xs = list(range(0x3F00, 0x3F00 + N))
    bytes_out = box.convert_row(FP8PackReq(xVec=xs, expShift=0))
    expected = [fp8.bf16_to_e4m3_byte(x, 0) for x in xs]
    assert bytes_out == expected


def test_convert_row_exp_shift_propagates():
    box = FP8Pack(P)
    xs = list(range(0x3F00, 0x3F00 + N))
    for shift in (-8, -2, 0, 2, 6):
        bytes_out = box.convert_row(FP8PackReq(xVec=xs, expShift=shift))
        expected = [fp8.bf16_to_e4m3_byte(x, shift) for x in xs]
        assert bytes_out == expected, f"expShift={shift} mismatch"


def test_convert_row_lane_count_validation():
    box = FP8Pack(P)
    with pytest.raises(ValueError, match="must have 16 lanes"):
        box.convert_row(FP8PackReq(xVec=[0] * (N - 1), expShift=0))
    with pytest.raises(ValueError, match="must have 16 lanes"):
        box.convert_row(FP8PackReq(xVec=[0] * (N + 1), expShift=0))


# ------------------------------------------------------------
#                  pack_two_rows — pure 2-row math
# ------------------------------------------------------------

def test_pack_two_rows_byte_layout_distinct_values():
    """Use 32 distinct BF16 patterns across the two rows and verify
    the output byte layout: low byte of slot j = byte 2j, high byte =
    byte 2j+1. Catches any swap/transpose/endian bug."""
    box = FP8Pack(P)
    row_low = list(range(0x3F00, 0x3F00 + N))
    row_high = list(range(0x3F10, 0x3F10 + N))
    resp = box.pack_two_rows(
        FP8PackReq(xVec=row_low, expShift=0),
        FP8PackReq(xVec=row_high, expShift=0),
    )
    expected = _golden_packed_row(
        [fp8.bf16_to_e4m3_byte(x, 0) for x in row_low],
        [fp8.bf16_to_e4m3_byte(x, 0) for x in row_high],
    )
    assert resp.result == expected


def test_pack_two_rows_low_half_is_first_input():
    """Slots 0..7 must come from the FIRST row, slots 8..15 from the
    SECOND row. The Scala uses `lowHalf` for slots 0..7 and the live
    `fp8Bytes` for slots 8..15."""
    box = FP8Pack(P)
    # Use sentinel BF16 values that map to distinct FP8 bytes
    low = _row(0x3F80)   # 1.0 → 0x38
    high = _row(0x4000)  # 2.0 → 0x40
    resp = box.pack_two_rows(
        FP8PackReq(xVec=low, expShift=0),
        FP8PackReq(xVec=high, expShift=0),
    )
    for j in range(HALF):
        assert resp.result[j] == _slot(0x38, 0x38), f"low slot {j} mismatch"
        assert resp.result[j + HALF] == _slot(0x40, 0x40), f"high slot {j} mismatch"


def test_pack_two_rows_does_not_mutate_state():
    """Pure helper must not touch _phase / _resp_valid / _low_half."""
    box = FP8Pack(P)
    box.pack_two_rows(
        FP8PackReq(xVec=_row(0x3F80), expShift=0),
        FP8PackReq(xVec=_row(0x4000), expShift=0),
    )
    assert box._phase is False
    assert box._resp_valid is False
    assert box._low_half == [0] * N
    assert box._resp_bits == [0] * N


# ------------------------------------------------------------
#                  reserved-NaN clamp boundary
# ------------------------------------------------------------

def test_reserved_nan_clamp_avoided():
    """`(finalExpAdjusted == 8 && mantFP8 == 7)` would otherwise pack
    to 0x7F (E4M3 NaN). The Scala clamps it to 0x7E (max finite). Find
    a BF16 value that hits this exact corner without going through
    the `>8` overflow branch.

    We need: finalExpAdjusted == 8 AND mantFP8 == 7 (with no normCarry).
    expAdjusted == 8 → unbExp - expShift = 8.  Pick expShift=0 → unbExp=8
    → expBF = 135 = 0x87.  For mantFP8==7 we want roundedSig in (8..15];
    smallest mantBF that gives roundedSig=15 is mant8=0xF8/0xF0 family —
    pick mantBF=0x70 → mant8=0xF0, trunc=0xF, guard=0, no inc, rounded=0xF.
    Then mantFP8 = (15-8)&0x7 = 7. So bf16 = 0x4380 | 0x70 = 0x43F0.
    """
    box = FP8Pack(P)
    bf16 = (0x87 << 7) | 0x70   # 0x43F0
    out = box.convert_row(FP8PackReq(xVec=_row(bf16), expShift=0))
    assert out == [0x7E] * N, f"expected clamped E4M3_MAX_POS, got {out[0]:#x}"

    # Same value with the sign bit set must clamp to E4M3_MAX_NEG.
    bf16_neg = bf16 | 0x8000
    out_neg = box.convert_row(FP8PackReq(xVec=_row(bf16_neg), expShift=0))
    assert out_neg == [0xFE] * N


def test_overflow_to_max_finite():
    """`finalExpAdjusted > 8` clamps to max finite, no NaN."""
    box = FP8Pack(P)
    # BF16 large finite: exp=0xFE (max finite), mant=0 → unbExp=127.
    # expShift=0 → expAdjusted=127 >> 8, way over.
    pos = box.convert_row(FP8PackReq(xVec=_row(0x7F00), expShift=0))
    neg = box.convert_row(FP8PackReq(xVec=_row(0xFF00), expShift=0))
    assert pos == [0x7E] * N
    assert neg == [0xFE] * N


# ------------------------------------------------------------
#                  step() — cycle-accurate phased FSM
# ------------------------------------------------------------

def test_step_first_pulse_bubbles():
    """Cycle 0 (first pulse) must produce no output — `respValidReg`
    is `false.B` initially and gets re-cleared by the `!phase` branch."""
    box = FP8Pack(P)
    out = box.step("fp8pack", FP8PackReq(xVec=_row(0x3F80), expShift=0))
    assert out is None
    assert box._phase is True
    assert box._resp_valid is False


def test_step_second_pulse_also_bubbles():
    """The Scala output is REGISTERED — `respBitsReg` is written at
    end of the second-pulse cycle, so io.resp.valid only goes high on
    the NEXT cycle (K+2 from the first pulse)."""
    box = FP8Pack(P)
    box.step("fp8pack", FP8PackReq(xVec=_row(0x3F80), expShift=0))
    out = box.step("fp8pack", FP8PackReq(xVec=_row(0x4000), expShift=0))
    assert out is None
    assert box._phase is False
    assert box._resp_valid is True   # latched at end of cycle 1


def test_step_third_cycle_emits_packed_row():
    """Cycle K+2 (one after the second pulse) is when the registered
    output finally appears on io.resp.valid / io.resp.bits.result."""
    box = FP8Pack(P)
    box.step("fp8pack", FP8PackReq(xVec=_row(0x3F80), expShift=0))   # cycle 0
    box.step("fp8pack", FP8PackReq(xVec=_row(0x4000), expShift=0))   # cycle 1
    out = box.step("fp8pack", None)                                  # cycle 2
    assert out is not None
    expected = _golden_packed_row(
        [fp8.bf16_to_e4m3_byte(0x3F80, 0)] * N,
        [fp8.bf16_to_e4m3_byte(0x4000, 0)] * N,
    )
    assert out.result == expected


def test_step_idle_after_emit_clears_resp_valid():
    """After the registered output is emitted on cycle K+2, the next
    idle cycle clears `respValidReg` (`}.otherwise { respValidReg :=
    false.B }`) so cycle K+3 returns None."""
    box = FP8Pack(P)
    box.step("fp8pack", FP8PackReq(xVec=_row(0x3F80), expShift=0))
    box.step("fp8pack", FP8PackReq(xVec=_row(0x4000), expShift=0))
    box.step("fp8pack", None)             # K+2: emits the row
    out = box.step("fp8pack", None)       # K+3: should bubble
    assert out is None


def test_step_idle_between_halves_preserves_phase():
    """The Scala `}.otherwise { respValidReg := false.B }` only clears
    the output reg — `phase` and `lowHalf` survive idle cycles, so the
    second pulse can arrive arbitrarily later."""
    box = FP8Pack(P)
    box.step("fp8pack", FP8PackReq(xVec=_row(0x3F80), expShift=0))   # cycle 0
    # 5 idle cycles
    for _ in range(5):
        out = box.step("fp8pack", None)
        assert out is None
        assert box._phase is True       # still waiting for second half
    box.step("fp8pack", FP8PackReq(xVec=_row(0x4000), expShift=0))   # cycle 6
    out = box.step("fp8pack", None)                                  # cycle 7
    assert out is not None
    expected = _golden_packed_row(
        [fp8.bf16_to_e4m3_byte(0x3F80, 0)] * N,
        [fp8.bf16_to_e4m3_byte(0x4000, 0)] * N,
    )
    assert out.result == expected


def test_step_streaming_two_packed_rows_back_to_back():
    """4 inputs back-to-back should produce 2 outputs at cycles 2 and 4.

      cycle 0 : req=row_a_low   → bubble
      cycle 1 : req=row_a_high  → bubble (register write at end-of-cycle)
      cycle 2 : req=row_b_low   → emits packed_a
      cycle 3 : req=row_b_high  → bubble
      cycle 4 : req=None        → emits packed_b
    """
    box = FP8Pack(P)
    a_low = _row(0x3F80)   # 1.0 → 0x38
    a_high = _row(0x4000)  # 2.0 → 0x40
    b_low = _row(0xBF80)   # -1.0 → 0xB8
    b_high = _row(0xC000)  # -2.0 → 0xC0

    out0 = box.step("fp8pack", FP8PackReq(xVec=a_low, expShift=0))
    out1 = box.step("fp8pack", FP8PackReq(xVec=a_high, expShift=0))
    out2 = box.step("fp8pack", FP8PackReq(xVec=b_low, expShift=0))
    out3 = box.step("fp8pack", FP8PackReq(xVec=b_high, expShift=0))
    out4 = box.step("fp8pack", None)

    assert out0 is None
    assert out1 is None
    assert out2 is not None
    assert out3 is None
    assert out4 is not None

    expected_a = _golden_packed_row([0x38] * N, [0x40] * N)
    expected_b = _golden_packed_row([0xB8] * N, [0xC0] * N)
    assert out2.result == expected_a
    assert out4.result == expected_b


def test_step_random_streaming_matches_pure_helper():
    """64 random pairs streamed through `step()` should match
    `pack_two_rows()` applied to the same pairs in the same order."""
    rng = np.random.default_rng(7)
    box = FP8Pack(P)
    pairs: List[tuple[List[int], List[int]]] = []
    for _ in range(64):
        pairs.append((
            _bf16_random_finite(N, rng),
            _bf16_random_finite(N, rng),
        ))
    expected = [
        FP8Pack(P).pack_two_rows(
            FP8PackReq(xVec=lo, expShift=0),
            FP8PackReq(xVec=hi, expShift=0),
        ).result
        for (lo, hi) in pairs
    ]

    actual: List[List[int]] = []
    for (lo, hi) in pairs:
        box.step("fp8pack", FP8PackReq(xVec=lo, expShift=0))
        out = box.step("fp8pack", FP8PackReq(xVec=hi, expShift=0))
        # The packed row from this pair is registered NOW but not
        # visible until the next step() call.
        out2 = box.step("fp8pack", None)
        assert out is None and out2 is not None
        actual.append(out2.result)

    assert actual == expected


def test_step_random_streaming_with_exp_shift():
    """Same as above but exercise expShift propagation through the FSM."""
    rng = np.random.default_rng(13)
    box = FP8Pack(P)
    for shift in (-4, -1, 0, 3, 7):
        box.reset()
        lo = _bf16_random_finite(N, rng)
        hi = _bf16_random_finite(N, rng)
        box.step("fp8pack", FP8PackReq(xVec=lo, expShift=shift))
        box.step("fp8pack", FP8PackReq(xVec=hi, expShift=shift))
        out = box.step("fp8pack", None)
        assert out is not None
        expected = _golden_packed_row(
            [fp8.bf16_to_e4m3_byte(x, shift) for x in lo],
            [fp8.bf16_to_e4m3_byte(x, shift) for x in hi],
        )
        assert out.result == expected, f"expShift={shift} mismatch"


def test_step_reset_clears_phase_and_output():
    box = FP8Pack(P)
    box.step("fp8pack", FP8PackReq(xVec=_row(0x3F80), expShift=0))
    assert box._phase is True
    box.reset()
    assert box._phase is False
    assert box._resp_valid is False
    assert box._low_half == [0] * N
    assert box._resp_bits == [0] * N
    # Now first pulse after reset behaves exactly like a fresh box.
    out = box.step("fp8pack", FP8PackReq(xVec=_row(0x3F80), expShift=0))
    assert out is None
    assert box._phase is True


def test_step_unknown_op_raises():
    box = FP8Pack(P)
    with pytest.raises(KeyError):
        box.step("fp8unpack", None)


def test_peek_resp_returns_none_when_invalid():
    box = FP8Pack(P)
    assert box.peek_resp() is None
    box.step("fp8pack", FP8PackReq(xVec=_row(0x3F80), expShift=0))
    # After first pulse, _resp_valid is still False.
    assert box.peek_resp() is None


def test_peek_resp_returns_latched_after_second_pulse():
    box = FP8Pack(P)
    box.step("fp8pack", FP8PackReq(xVec=_row(0x3F80), expShift=0))
    box.step("fp8pack", FP8PackReq(xVec=_row(0x4000), expShift=0))
    # End-of-second-pulse: _resp_valid is True, _resp_bits is the packed row
    # — even though step() returned None for the second pulse.
    peek = box.peek_resp()
    assert peek is not None
    expected = _golden_packed_row([0x38] * N, [0x40] * N)
    assert peek.result == expected
