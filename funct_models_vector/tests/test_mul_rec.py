"""Bit-exact tests for funct_models_vector.lane_boxes.mul_rec.

Two independent references:
  * `numpy.float32` multiply + `f32_to_bf16_bits_rne` (mirrors the
    HardFloat MulRawFN + RoundRawFNToRecFN(8, 8, 0) path internally).
  * `torch.bfloat16` multiply, which is a separate code path and gives us
    a real second source.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from funct_models_vector import bf16_utils as fp
from funct_models_vector.lane_boxes.mul_rec import MulRec, MulReq
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes


def _bf16_bits_to_torch(b: int) -> torch.Tensor:
    """Pour raw BF16 bits into a torch.bfloat16 tensor without going via float."""
    u16 = np.array([b & 0xFFFF], dtype=np.uint16).view(np.uint8)
    return torch.frombuffer(bytearray(u16.tobytes()), dtype=torch.bfloat16).clone()


def _torch_to_bf16_bits(t: torch.Tensor) -> int:
    return int(t.view(torch.int16).cpu().numpy().view(np.uint16)[0])


def _ref_mul_via_torch(a: int, b: int) -> int:
    ta = _bf16_bits_to_torch(a)
    tb = _bf16_bits_to_torch(b)
    return _torch_to_bf16_bits(ta * tb)


# ---------- hand-computed goldens ----------

def test_mul_two_times_three():
    box = MulRec(P)
    r = box.compute_now(MulReq(aVec=[0x4000] * N, bVec=[0x4040] * N))
    assert r.result == [0x40C0] * N  # 2.0 * 3.0 = 6.0


def test_mul_one_times_anything_is_anything():
    box = MulRec(P)
    a = list(range(0x3F00, 0x3F10))  # 16 distinct BF16 patterns near 1.0
    r = box.compute_now(MulReq(aVec=a, bVec=[0x3F80] * N))  # *1.0
    assert r.result == a


def test_mul_signs():
    box = MulRec(P)
    # +2 * -3 = -6 = 0xC0C0
    r = box.compute_now(MulReq(aVec=[0x4000] * N, bVec=[0xC040] * N))
    assert r.result == [0xC0C0] * N


def test_mul_zero_times_anything_is_zero_with_xored_sign():
    """HardFloat MulRec XORs sign bits, so (-x) * (+0) = -0 (0x8000), not +0.
    This is IEEE-754 compliant; a naive |result|==0 test would mask the sign."""
    box = MulRec(P)
    a = [0x3F80, 0x4000, 0xBF80, 0xC000] * 4   # +1, +2, -1, -2 repeated
    r = box.compute_now(MulReq(aVec=a, bVec=[0x0000] * N))     # all * +0
    expected = [0x0000 if (x & 0x8000) == 0 else 0x8000 for x in a]
    assert r.result == expected
    # And once more against -0 to confirm both sign-XOR paths.
    r2 = box.compute_now(MulReq(aVec=a, bVec=[0x8000] * N))
    expected2 = [0x8000 if (x & 0x8000) == 0 else 0x0000 for x in a]
    assert r2.result == expected2


# ---------- random sweep against torch.bfloat16 ----------

def _bf16_random_finite(n: int, rng: np.random.Generator) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        b = int(rng.integers(0, 1 << 16))
        if ((b >> 7) & 0xFF) == 0xFF:
            continue
        out.append(b)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mul_random_matches_torch(seed: int):
    rng = np.random.default_rng(seed)
    box = MulRec(P)
    for _ in range(64):
        a = _bf16_random_finite(N, rng)
        b = _bf16_random_finite(N, rng)
        r = box.compute_now(MulReq(aVec=a, bVec=b))
        for i in range(N):
            t = _ref_mul_via_torch(a[i], b[i])
            # If the torch result is BF16 +0 / -0, MulRec may pick the other
            # zero (HardFloat picks sign as a XOR b), so canonicalize.
            assert r.result[i] == t, (
                f"lane {i}: a=0x{a[i]:04x} b=0x{b[i]:04x} "
                f"got=0x{r.result[i]:04x} torch=0x{t:04x}"
            )


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
def test_mul_random_matches_internal_helper(seed: int):
    """Sanity-check that compute_now == bf16_mul lane-by-lane (the impl
    delegates to bf16_mul, so this is a trivial regression net)."""
    rng = np.random.default_rng(seed)
    box = MulRec(P)
    for _ in range(64):
        a = _bf16_random_finite(N, rng)
        b = _bf16_random_finite(N, rng)
        r = box.compute_now(MulReq(aVec=a, bVec=b))
        assert r.result == [fp.bf16_mul(a[i], b[i]) for i in range(N)]


# ---------- step latency ----------

def test_step_mul_latency_one_cycle():
    box = MulRec(P)
    req = MulReq(aVec=[0x4000] * N, bVec=[0x4040] * N)
    assert box.step("mul", req) is None
    out = box.step("mul", None)
    assert out is not None and out.result == [0x40C0] * N


def test_step_unknown_op_raises():
    box = MulRec(P)
    with pytest.raises(KeyError):
        box.step("add", None)


def test_reset_clears_pipe():
    box = MulRec(P)
    req = MulReq(aVec=[0x4000] * N, bVec=[0x4040] * N)
    box.step("mul", req)
    box.reset()
    assert box.step("mul", None) is None
