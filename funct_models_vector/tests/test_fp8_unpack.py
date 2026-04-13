"""Bit-exact tests for funct_models_vector.lane_boxes.fp8_unpack.

`FP8Unpack` is the dual of `FP8Pack`: **1 → 2 phased + 32-deep input
queue**. The funct model has to mirror three pieces of state:

1. The `sIdle / sLow / sHigh` FSM that emits a low-half BF16 row then
   a high-half BF16 row from one packed input.
2. The 32-deep `Queue(FP8UnpackReq)` that absorbs back-to-back inputs
   while the FSM drains the previous packed row over two cycles.
3. The combinational `io.resp.valid := (state === sLow || state ===
   sHigh)` and `io.resp.bits.result := outVec` wires.

This test file pins:

- **2-cycle visible latency**. One input enqueue → first BF16 row two
  cycles later, second BF16 row three cycles later.
- **Queue 1-cycle enq→deq latency**. The `Queue` default
  `pipe=false flow=false` means an item enqueued at cycle K is only
  visible on `deq.valid` starting cycle K+1, which is what gives
  `FP8Unpack` its 2-cycle floor.
- **Back-to-back streaming**. Sustained input rate is absorbed by the
  queue; outputs come out at 2-per-input rate offset by 2 cycles.
- **Queue overflow assertion**. Raised when an enqueue arrives at a
  cycle where the queue is already at 32 (mirrors the Scala
  `assert(!io.req.valid || reqQ.io.enq.ready)`).
- **Reserved NaN + subnormal flush**. `fp8 == 0xFF/7F` (exp=0xF
  mant=0x7) and `expFP8 == 0` flush to signed zero, matching
  `FP8Unpack.scala:82-83`.
- **Round-trip pack → unpack** at `expShift = 0` recovers the
  E4M3-quantized BF16 value, exercised through the lane_box wrappers
  end-to-end.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from funct_models_vector import fp8_e4m3 as fp8
from funct_models_vector.lane_boxes.fp8_pack import FP8Pack, FP8PackReq
from funct_models_vector.lane_boxes.fp8_unpack import (
    FP8Unpack,
    FP8UnpackReq,
    FP8UnpackResp,
)
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes
HALF = N // 2


# ------------------------------------------------------------
#                  helpers
# ------------------------------------------------------------

def _slot(low_byte: int, high_byte: int) -> int:
    return ((high_byte & 0xFF) << 8) | (low_byte & 0xFF)


def _packed_row_from_bytes(
    low_bytes: List[int], high_bytes: List[int]
) -> List[int]:
    """Build the 16-slot packed row layout the engine uses, given 16
    low-half FP8 bytes and 16 high-half FP8 bytes."""
    out = [0] * N
    for j in range(HALF):
        out[j] = _slot(low_bytes[2 * j], low_bytes[2 * j + 1])
        out[j + HALF] = _slot(high_bytes[2 * j], high_bytes[2 * j + 1])
    return out


def _bf16_random_finite(n: int, rng: np.random.Generator) -> List[int]:
    out: List[int] = []
    while len(out) < n:
        b = int(rng.integers(0, 1 << 16))
        if ((b >> 7) & 0xFF) == 0xFF:
            continue
        out.append(b)
    return out


def _packed_row_for_unpack(
    bf16_low: List[int], bf16_high: List[int], exp_shift: int
) -> List[int]:
    """End-to-end: BF16 → FP8 (using the pack helper) → packed-slot
    layout. The "low" BF16 row becomes the low 16 FP8 bytes; the
    "high" BF16 row becomes the high 16 FP8 bytes. This is exactly
    what `FP8Pack.pack_two_rows` produces, so a downstream
    `FP8Unpack.step` should drain it back."""
    low_bytes = [fp8.bf16_to_e4m3_byte(x, exp_shift) for x in bf16_low]
    high_bytes = [fp8.bf16_to_e4m3_byte(x, exp_shift) for x in bf16_high]
    return _packed_row_from_bytes(low_bytes, high_bytes)


# ------------------------------------------------------------
#                  pure unpack helpers
# ------------------------------------------------------------

def test_unpack_low_row_matches_helper():
    box = FP8Unpack(P)
    # 32 distinct FP8 bytes spread across 16 packed slots
    fp8_bytes = list(range(0x10, 0x10 + 32))
    packed = _packed_row_from_bytes(fp8_bytes[:16], fp8_bytes[16:])
    resp = box.unpack_low_row(FP8UnpackReq(xVec=packed, expShift=0))
    expected = [fp8.e4m3_byte_to_bf16(b, 0) for b in fp8_bytes[:16]]
    assert resp.result == expected


def test_unpack_high_row_matches_helper():
    box = FP8Unpack(P)
    fp8_bytes = list(range(0x10, 0x10 + 32))
    packed = _packed_row_from_bytes(fp8_bytes[:16], fp8_bytes[16:])
    resp = box.unpack_high_row(FP8UnpackReq(xVec=packed, expShift=0))
    expected = [fp8.e4m3_byte_to_bf16(b, 0) for b in fp8_bytes[16:]]
    assert resp.result == expected


def test_unpack_both_rows_returns_low_then_high():
    box = FP8Unpack(P)
    fp8_bytes = list(range(0x10, 0x10 + 32))
    packed = _packed_row_from_bytes(fp8_bytes[:16], fp8_bytes[16:])
    low_resp, high_resp = box.unpack_both_rows(
        FP8UnpackReq(xVec=packed, expShift=0)
    )
    assert low_resp.result == [
        fp8.e4m3_byte_to_bf16(b, 0) for b in fp8_bytes[:16]
    ]
    assert high_resp.result == [
        fp8.e4m3_byte_to_bf16(b, 0) for b in fp8_bytes[16:]
    ]


def test_unpack_does_not_mutate_state():
    box = FP8Unpack(P)
    packed = _packed_row_from_bytes([0x38] * N, [0x40] * N)
    box.unpack_both_rows(FP8UnpackReq(xVec=packed, expShift=0))
    assert box._state == "idle"
    assert len(box._queue) == 0
    assert box._exp_buf == 0
    assert box._input_buf == [0] * (2 * N)


def test_unpack_special_values_signed_zero():
    box = FP8Unpack(P)
    # Subnormal FP8 (exp=0, mant!=0) and reserved NaN (exp=0xF, mant=0x7)
    # both flush to signed zero, sign preserved.
    bytes_low = [0x01, 0x81, 0x7F, 0xFF, 0x07, 0x87] + [0x00] * 10
    packed = _packed_row_from_bytes(bytes_low, [0x00] * N)
    resp = box.unpack_low_row(FP8UnpackReq(xVec=packed, expShift=0))
    expected_special = [
        0x0000,  # +subnormal → +0
        0x8000,  # -subnormal → -0
        0x0000,  # +reserved NaN → +0
        0x8000,  # -reserved NaN → -0
        0x0000,  # +subnormal mant=7 → +0
        0x8000,  # -subnormal mant=7 → -0
    ]
    assert resp.result[:6] == expected_special
    assert resp.result[6:] == [0x0000] * 10


def test_unpack_lane_count_validation():
    box = FP8Unpack(P)
    with pytest.raises(ValueError, match="must have 16 slots"):
        box.unpack_low_row(FP8UnpackReq(xVec=[0] * (N - 1), expShift=0))


# ------------------------------------------------------------
#                  step() — cycle-accurate FSM + queue
# ------------------------------------------------------------

def test_step_idle_returns_none():
    box = FP8Unpack(P)
    assert box.step("fp8unpack", None) is None
    assert box.step("fp8unpack", None) is None


def test_step_one_input_emits_two_outputs_after_two_cycle_latency():
    """Cycle 0: enqueue. Cycle 1: queue.deq.valid → consume into bufs,
    state→sLow at end-of-cycle. Cycle 2: state=sLow, output = low.
    Cycle 3: state=sHigh, output = high. Cycle 4: idle."""
    box = FP8Unpack(P)
    fp8_bytes = list(range(0x10, 0x10 + 32))
    packed = _packed_row_from_bytes(fp8_bytes[:16], fp8_bytes[16:])

    out0 = box.step("fp8unpack", FP8UnpackReq(xVec=packed, expShift=0))
    out1 = box.step("fp8unpack", None)
    out2 = box.step("fp8unpack", None)
    out3 = box.step("fp8unpack", None)
    out4 = box.step("fp8unpack", None)

    expected_low = [fp8.e4m3_byte_to_bf16(b, 0) for b in fp8_bytes[:16]]
    expected_high = [fp8.e4m3_byte_to_bf16(b, 0) for b in fp8_bytes[16:]]

    assert out0 is None
    assert out1 is None
    assert out2 is not None and out2.result == expected_low
    assert out3 is not None and out3.result == expected_high
    assert out4 is None


def test_step_back_to_back_two_inputs_pipelines_through_queue():
    """4 cycles of input + 4 cycles of drain. Two packed rows in,
    four BF16 rows out. The queue absorbs the back-to-back enqueues."""
    box = FP8Unpack(P)
    bytes_a = list(range(0x10, 0x10 + 32))
    bytes_b = list(range(0x40, 0x40 + 32))
    packed_a = _packed_row_from_bytes(bytes_a[:16], bytes_a[16:])
    packed_b = _packed_row_from_bytes(bytes_b[:16], bytes_b[16:])

    outs: List = []
    outs.append(box.step("fp8unpack", FP8UnpackReq(xVec=packed_a, expShift=0)))   # 0
    outs.append(box.step("fp8unpack", FP8UnpackReq(xVec=packed_b, expShift=0)))   # 1
    outs.append(box.step("fp8unpack", None))                                      # 2
    outs.append(box.step("fp8unpack", None))                                      # 3
    outs.append(box.step("fp8unpack", None))                                      # 4
    outs.append(box.step("fp8unpack", None))                                      # 5
    outs.append(box.step("fp8unpack", None))                                      # 6 (drain)

    expected_a_low = [fp8.e4m3_byte_to_bf16(b, 0) for b in bytes_a[:16]]
    expected_a_high = [fp8.e4m3_byte_to_bf16(b, 0) for b in bytes_a[16:]]
    expected_b_low = [fp8.e4m3_byte_to_bf16(b, 0) for b in bytes_b[:16]]
    expected_b_high = [fp8.e4m3_byte_to_bf16(b, 0) for b in bytes_b[16:]]

    assert outs[0] is None
    assert outs[1] is None
    assert outs[2] is not None and outs[2].result == expected_a_low
    assert outs[3] is not None and outs[3].result == expected_a_high
    assert outs[4] is not None and outs[4].result == expected_b_low
    assert outs[5] is not None and outs[5].result == expected_b_high
    assert outs[6] is None


def test_step_idle_after_input_drains_to_idle():
    """After two output beats, the FSM should return to idle and stay
    there. Cycle 4 onward is None."""
    box = FP8Unpack(P)
    packed = _packed_row_from_bytes([0x38] * N, [0x40] * N)
    box.step("fp8unpack", FP8UnpackReq(xVec=packed, expShift=0))
    box.step("fp8unpack", None)  # consume → state=low at end
    box.step("fp8unpack", None)  # output low, state=high at end
    box.step("fp8unpack", None)  # output high, state=idle at end
    for _ in range(4):
        assert box.step("fp8unpack", None) is None
    assert box._state == "idle"


def test_step_late_second_input_uses_queue_path():
    """If the second input arrives AFTER the first has fully drained,
    the FSM must come back out of idle. Tests the
    `is(sIdle) { when(deq.valid) { ... } }` re-entry branch."""
    box = FP8Unpack(P)
    bytes_a = [0x38] * 16 + [0x40] * 16
    bytes_b = [0x48] * 16 + [0x50] * 16
    packed_a = _packed_row_from_bytes(bytes_a[:16], bytes_a[16:])
    packed_b = _packed_row_from_bytes(bytes_b[:16], bytes_b[16:])

    box.step("fp8unpack", FP8UnpackReq(xVec=packed_a, expShift=0))   # 0: enq A
    box.step("fp8unpack", None)                                      # 1: deq A
    out2 = box.step("fp8unpack", None)                               # 2: A_low
    out3 = box.step("fp8unpack", None)                               # 3: A_high
    # Now state is idle. Drop a second input.
    box.step("fp8unpack", FP8UnpackReq(xVec=packed_b, expShift=0))   # 4: enq B
    box.step("fp8unpack", None)                                      # 5: deq B
    out6 = box.step("fp8unpack", None)                               # 6: B_low
    out7 = box.step("fp8unpack", None)                               # 7: B_high

    assert out2 is not None and out2.result == [
        fp8.e4m3_byte_to_bf16(b, 0) for b in bytes_a[:16]
    ]
    assert out3 is not None and out3.result == [
        fp8.e4m3_byte_to_bf16(b, 0) for b in bytes_a[16:]
    ]
    assert out6 is not None and out6.result == [
        fp8.e4m3_byte_to_bf16(b, 0) for b in bytes_b[:16]
    ]
    assert out7 is not None and out7.result == [
        fp8.e4m3_byte_to_bf16(b, 0) for b in bytes_b[16:]
    ]


def test_step_exp_shift_buffered_per_packed_row():
    """`expBuf` is a Reg, not combinational. Each packed input
    snapshots its own expShift, and that value is what the BF16
    conversion uses for both the low and high beats — even if
    subsequent inputs carry a different expShift."""
    box = FP8Unpack(P)
    bytes_a = [0x38] * 32
    bytes_b = [0x40] * 32
    packed_a = _packed_row_from_bytes(bytes_a[:16], bytes_a[16:])
    packed_b = _packed_row_from_bytes(bytes_b[:16], bytes_b[16:])

    box.step("fp8unpack", FP8UnpackReq(xVec=packed_a, expShift=2))   # 0: enq A@+2
    box.step("fp8unpack", FP8UnpackReq(xVec=packed_b, expShift=-3))  # 1: enq B@-3, deq A
    out_a_low = box.step("fp8unpack", None)                          # 2: A_low @ +2
    out_a_high = box.step("fp8unpack", None)                         # 3: A_high @ +2, deq B
    out_b_low = box.step("fp8unpack", None)                          # 4: B_low @ -3
    out_b_high = box.step("fp8unpack", None)                         # 5: B_high @ -3

    assert out_a_low.result == [fp8.e4m3_byte_to_bf16(0x38, 2)] * N
    assert out_a_high.result == [fp8.e4m3_byte_to_bf16(0x38, 2)] * N
    assert out_b_low.result == [fp8.e4m3_byte_to_bf16(0x40, -3)] * N
    assert out_b_high.result == [fp8.e4m3_byte_to_bf16(0x40, -3)] * N


def test_step_random_streaming_matches_unpack_helper():
    """Stream 16 random packed rows and compare each pair of step
    outputs to `unpack_both_rows()` applied to the same input."""
    rng = np.random.default_rng(21)
    box = FP8Unpack(P)
    inputs: List[FP8UnpackReq] = []
    for _ in range(16):
        bf16_lo = _bf16_random_finite(N, rng)
        bf16_hi = _bf16_random_finite(N, rng)
        packed = _packed_row_for_unpack(bf16_lo, bf16_hi, exp_shift=0)
        inputs.append(FP8UnpackReq(xVec=packed, expShift=0))

    expected: List = []
    for req in inputs:
        lo, hi = FP8Unpack(P).unpack_both_rows(req)
        expected.append(lo.result)
        expected.append(hi.result)

    actual: List = []
    # Drive 16 inputs back-to-back, then drain.
    for req in inputs:
        out = box.step("fp8unpack", req)
        if out is not None:
            actual.append(out.result)
    # Drain the rest of the queue.
    while True:
        out = box.step("fp8unpack", None)
        if out is None and len(actual) == len(expected):
            break
        if out is not None:
            actual.append(out.result)

    assert actual == expected


def test_step_reset_clears_state():
    box = FP8Unpack(P)
    packed = _packed_row_from_bytes([0x38] * N, [0x40] * N)
    box.step("fp8unpack", FP8UnpackReq(xVec=packed, expShift=0))
    box.step("fp8unpack", None)
    box.reset()
    assert box._state == "idle"
    assert len(box._queue) == 0
    assert box._exp_buf == 0
    assert box._input_buf == [0] * (2 * N)
    # First call after reset is None (queue empty).
    assert box.step("fp8unpack", None) is None


def test_step_unknown_op_raises():
    box = FP8Unpack(P)
    with pytest.raises(KeyError):
        box.step("fp8pack", None)


# ------------------------------------------------------------
#                  queue overflow
# ------------------------------------------------------------

def test_queue_overflow_raises():
    """Mirrors the Scala `assert(!io.req.valid || enq.ready)`. Filling
    the queue to depth 32 then attempting to enqueue must raise. We
    bypass the FSM by appending directly so the test is deterministic
    — naturally trickling 33 inputs through `step()` would NOT trip
    the assert because the consumer drains alongside."""
    box = FP8Unpack(P)
    packed = _packed_row_from_bytes([0x38] * N, [0x40] * N)
    req = FP8UnpackReq(xVec=packed, expShift=0)
    for _ in range(box.QUEUE_DEPTH):
        box._queue.append(req)
    with pytest.raises(RuntimeError, match="queue overflow"):
        box.step("fp8unpack", req)


def test_queue_at_31_accepts_one_more():
    """Boundary check — exactly QUEUE_DEPTH-1 entries leaves space for
    one more enqueue, which should NOT raise."""
    box = FP8Unpack(P)
    packed = _packed_row_from_bytes([0x38] * N, [0x40] * N)
    req = FP8UnpackReq(xVec=packed, expShift=0)
    for _ in range(box.QUEUE_DEPTH - 1):
        box._queue.append(req)
    box.step("fp8unpack", req)   # should succeed; no raise
    # Exactly at capacity now (after the FSM dequeued one and we added one).
    # The FSM consumed one (state went idle → low at the end of the cycle).
    # Net: started 31, dequeued 1 (→30), enqueued 1 (→31). Sanity-check.
    assert len(box._queue) == 31


def test_queue_overflow_no_req_does_not_raise():
    """Idle cycles never attempt to enqueue, so a full queue + step(None)
    drains as normal without raising."""
    box = FP8Unpack(P)
    packed = _packed_row_from_bytes([0x38] * N, [0x40] * N)
    req = FP8UnpackReq(xVec=packed, expShift=0)
    for _ in range(box.QUEUE_DEPTH):
        box._queue.append(req)
    # An idle cycle should pop one (sIdle → sLow consumption) without raising.
    box.step("fp8unpack", None)
    assert len(box._queue) == box.QUEUE_DEPTH - 1


# ------------------------------------------------------------
#                  round-trip pack → unpack
# ------------------------------------------------------------

def test_round_trip_pack_unpack_at_zero_shift():
    """Pack two BF16 rows, feed the packed result into Unpack, and
    verify each lane round-trips to its E4M3-quantized BF16 value.
    This is bit-exact equality only because both pack and unpack use
    `expShift=0` and the round-trip composition `e4m3_byte_to_bf16(
    bf16_to_e4m3_byte(x, 0), 0)` is what we expect."""
    pack = FP8Pack(P)
    unpack = FP8Unpack(P)

    bf16_low = list(range(0x3F00, 0x3F00 + N))
    bf16_high = list(range(0x4000, 0x4000 + N))

    packed = pack.pack_two_rows(
        FP8PackReq(xVec=bf16_low, expShift=0),
        FP8PackReq(xVec=bf16_high, expShift=0),
    )
    lo, hi = unpack.unpack_both_rows(FP8UnpackReq(xVec=packed.result, expShift=0))

    expected_low = [
        fp8.e4m3_byte_to_bf16(fp8.bf16_to_e4m3_byte(x, 0), 0) for x in bf16_low
    ]
    expected_high = [
        fp8.e4m3_byte_to_bf16(fp8.bf16_to_e4m3_byte(x, 0), 0) for x in bf16_high
    ]
    assert lo.result == expected_low
    assert hi.result == expected_high


def test_round_trip_e4m3_exact_values_unchanged():
    """Pick BF16 values that are exactly representable in E4M3 (e.g.
    1.0, 2.0, 4.0). The pack→unpack round trip should preserve them
    bit-for-bit at expShift=0."""
    pack = FP8Pack(P)
    unpack = FP8Unpack(P)
    # 1.0=0x3F80, 2.0=0x4000, 4.0=0x4080, 8.0=0x4100, -1.0=0xBF80, -2.0=0xC000
    exact = [0x3F80, 0x4000, 0x4080, 0x4100, 0xBF80, 0xC000, 0x3F80, 0x4000] * 2
    packed = pack.pack_two_rows(
        FP8PackReq(xVec=exact, expShift=0),
        FP8PackReq(xVec=exact, expShift=0),
    )
    lo, hi = unpack.unpack_both_rows(FP8UnpackReq(xVec=packed.result, expShift=0))
    assert lo.result == exact
    assert hi.result == exact


def test_round_trip_through_step_streaming():
    """Same round-trip but driven through cycle-accurate `step()` on
    both sides. Pack via 2-input phased FSM, then feed the registered
    output into Unpack's queue and drain over 4 cycles."""
    pack = FP8Pack(P)
    unpack = FP8Unpack(P)
    bf16_low = list(range(0x3F00, 0x3F00 + N))
    bf16_high = list(range(0x4000, 0x4000 + N))

    pack.step("fp8pack", FP8PackReq(xVec=bf16_low, expShift=0))
    pack.step("fp8pack", FP8PackReq(xVec=bf16_high, expShift=0))
    packed_resp = pack.step("fp8pack", None)
    assert packed_resp is not None

    unpack.step("fp8unpack", FP8UnpackReq(xVec=packed_resp.result, expShift=0))
    unpack.step("fp8unpack", None)        # cycle 1: deq into bufs
    out_low = unpack.step("fp8unpack", None)   # cycle 2: low row
    out_high = unpack.step("fp8unpack", None)  # cycle 3: high row

    expected_low = [
        fp8.e4m3_byte_to_bf16(fp8.bf16_to_e4m3_byte(x, 0), 0) for x in bf16_low
    ]
    expected_high = [
        fp8.e4m3_byte_to_bf16(fp8.bf16_to_e4m3_byte(x, 0), 0) for x in bf16_high
    ]
    assert out_low is not None and out_low.result == expected_low
    assert out_high is not None and out_high.result == expected_high
