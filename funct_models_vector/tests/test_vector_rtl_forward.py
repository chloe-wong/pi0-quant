"""Phase 3 torch-adapter smoke tests for `VectorRTLFunctions`.

Covers the golden path for every public method: pointwise binary ops,
pointwise unary ops, row reductions, col reductions, and FP8 pack/
unpack. The oracle for each op is a plain torch computation over BF16-
quantized inputs; the adapter's output is compared with a relative
tolerance of ~2 BF16 ULPs (`rtol=1/64`) plus a small absolute floor.
Relative is the right shape here because BF16 ULP scales with value
magnitude (≈0.06 at M≈10, ≈1.0 at M≈150); a flat absolute tolerance
either over-tightens at large M or over-loosens at small M.

The ~1 ULP wiggle comes from two sources: (a) the RTL-faithful model
uses raw truncation (`fNFromRecFN(...)(31,16)` bit-slice) to narrow
FP32 results to BF16, while the oracle here rounds RNE via `_quant`;
and (b) LUT-backed transcendentals round differently from torch's
math kernels. Neither is a bug in the adapter.

The goal is to gate "the adapter shape / chunking / device round-trip
actually works," not to re-prove every lane_box's bit-exact fidelity —
that's already covered by the Tier 1 lane_box tests and the Tier 2
file-based cross-test.
"""

from __future__ import annotations

import math

import pytest
import torch

from funct_models_vector.vector_rtl_forward import (
    VectorRTLFunctions,
    torch_bf16_bits_to_float,
    torch_float_to_bf16_bits,
)


@pytest.fixture(scope="module")
def rtl() -> VectorRTLFunctions:
    return VectorRTLFunctions()


def _quant(x: torch.Tensor) -> torch.Tensor:
    """Quantize a float tensor to BF16 and back to float32."""
    return torch_bf16_bits_to_float(torch_float_to_bf16_bits(x))


# ----------------------------------------------------------------
#  Pointwise binary
# ----------------------------------------------------------------

@pytest.mark.parametrize("op_name,torch_op", [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("pairwise_max", torch.maximum),
    ("pairwise_min", torch.minimum),
])
def test_pointwise_binary(
    rtl: VectorRTLFunctions, op_name: str, torch_op
) -> None:
    g = torch.Generator().manual_seed(0xC0FFEE + hash(op_name) & 0xFFFF)
    a = torch.empty(32).uniform_(-10.0, 10.0, generator=g)
    b = torch.empty(32).uniform_(-10.0, 10.0, generator=g)
    want = _quant(torch_op(_quant(a), _quant(b)))
    got = getattr(rtl, op_name)(a, b)
    assert got.shape == want.shape
    assert torch.allclose(got, want, rtol=1.0 / 64.0, atol=1e-3), (
        f"{op_name}: got={got.tolist()[:6]} want={want.tolist()[:6]}"
    )


# ----------------------------------------------------------------
#  Pointwise unary
# ----------------------------------------------------------------

@pytest.mark.parametrize("op_name,torch_op,gen_range", [
    ("relu",   lambda x: torch.clamp(x, min=0.0),   (-5.0, 5.0)),
    ("square", lambda x: x * x,                      (-3.0, 3.0)),
    ("cube",   lambda x: x * x * x,                  (-2.0, 2.0)),
    ("sqrt",   torch.sqrt,                           (0.1, 10.0)),
    ("rcp",    lambda x: 1.0 / x,                    (0.5, 5.0)),
])
def test_pointwise_unary_spine(
    rtl: VectorRTLFunctions, op_name: str, torch_op, gen_range
) -> None:
    lo, hi = gen_range
    g = torch.Generator().manual_seed(0xFACADE + hash(op_name) & 0xFFFF)
    a = torch.empty(32).uniform_(lo, hi, generator=g)
    want = _quant(torch_op(_quant(a)))
    got = getattr(rtl, op_name)(a)
    rel = torch.abs(got - want) / torch.clamp(torch.abs(want), min=1e-3)
    assert (rel < 5e-2).all(), (
        f"{op_name}: got={got.tolist()[:6]} want={want.tolist()[:6]}"
    )


# ----------------------------------------------------------------
#  Pointwise unary — transcendental public-surface coverage
# ----------------------------------------------------------------
#
# The adapter publicly exposes `sin`, `cos`, `log2`, `tanh`, `exp`, and
# `exp2` but the original smoke block tested none of them directly:
#   - sin/cos route through SinCosVec (Q-format + LUT + linear interp),
#   - log2 routes through Log (LUT-backed),
#   - tanh/exp/exp2 route through `_legacy_math_fallback` in
#     vector_engine_model.py until §6 Group D lands.
#
# We oracle against torch's own kernels with `rtol=5e-2, atol=5e-3` (the
# same looseness used for sqrt/rcp) because the LUT approximations round
# differently from torch's math but stay within ~2 BF16 ULPs relative.

@pytest.mark.parametrize("op_name,torch_op,gen_range", [
    ("sin",   torch.sin,                           (0.0, 2 * math.pi)),
    ("cos",   torch.cos,                           (0.0, 2 * math.pi)),
    ("log2",  torch.log2,                          (0.25, 16.0)),
    ("tanh",  torch.tanh,                          (-4.0, 4.0)),
    ("exp",   torch.exp,                           (-4.0, 4.0)),
    ("exp2",  lambda x: torch.pow(2.0, x),         (-4.0, 4.0)),
])
def test_pointwise_unary_transcendental_surface(
    rtl: VectorRTLFunctions, op_name: str, torch_op, gen_range
) -> None:
    lo, hi = gen_range
    g = torch.Generator().manual_seed(0xBEEF + (hash(op_name) & 0xFFFF))
    a = torch.empty(32).uniform_(lo, hi, generator=g)
    want = _quant(torch_op(_quant(a)))
    got = getattr(rtl, op_name)(a)
    assert got.shape == want.shape
    assert torch.allclose(got, want, rtol=5e-2, atol=5e-3), (
        f"{op_name}: got={got.tolist()[:6]} want={want.tolist()[:6]}"
    )


# ----------------------------------------------------------------
#  Row reductions
# ----------------------------------------------------------------

def test_rsum_1d(rtl: VectorRTLFunctions) -> None:
    a = torch.tensor([1.0] * 16)
    got = rtl.rsum(a)
    assert got.shape == (1,)
    assert math.isclose(got.item(), 16.0, rel_tol=1e-3)


def test_rsum_2d_multiple_slices(rtl: VectorRTLFunctions) -> None:
    a = torch.arange(1, 33, dtype=torch.float32)  # length 32 → 2 slices of 16
    got = rtl.rsum(a)
    assert got.shape == (2,)
    # First slice: sum(1..16) = 136; second: sum(17..32) = 392
    assert math.isclose(got[0].item(), 136.0, rel_tol=1e-2)
    assert math.isclose(got[1].item(), 392.0, rel_tol=1e-2)


def test_rmax_and_rmin(rtl: VectorRTLFunctions) -> None:
    a = torch.arange(-7.0, 9.0, dtype=torch.float32)
    assert math.isclose(rtl.rmax(a).item(), 8.0, rel_tol=1e-3)
    assert math.isclose(rtl.rmin(a).item(), -7.0, rel_tol=1e-3)


# ----------------------------------------------------------------
#  Col reductions (cross-row streaming)
# ----------------------------------------------------------------

def test_csum_4_identical_rows(rtl: VectorRTLFunctions) -> None:
    row = torch.arange(1.0, 17.0, dtype=torch.float32)
    rows = row.unsqueeze(0).repeat(4, 1)  # (4, 16)
    got = rtl.csum(rows)
    assert got.shape == (16,)
    want = row * 4.0
    assert torch.allclose(got, want, atol=1e-2)


def test_cmax_cmin_3_rows(rtl: VectorRTLFunctions) -> None:
    rows = torch.stack([
        torch.linspace(1.0, 16.0, 16),
        torch.linspace(16.0, 1.0, 16),
        torch.full((16,), 8.0),
    ])
    got_max = rtl.cmax(rows)
    got_min = rtl.cmin(rows)
    assert got_max.shape == (16,) and got_min.shape == (16,)
    want_max = rows.max(dim=0).values
    want_min = rows.min(dim=0).values
    assert torch.allclose(got_max, _quant(want_max), atol=1e-2)
    assert torch.allclose(got_min, _quant(want_min), atol=1e-2)


# ----------------------------------------------------------------
#  FP8 pack / unpack round-trip
# ----------------------------------------------------------------

def test_fp8_pack_unpack_round_trip(rtl: VectorRTLFunctions) -> None:
    pair = torch.stack([
        torch.tensor([1.0, 2.0, 0.5, -1.0, 4.0, -2.0, 8.0, -4.0,
                      16.0, -8.0, 32.0, -16.0, 64.0, -32.0, 128.0, -64.0]),
        torch.tensor([0.75, 1.5, -0.25, 3.0, -0.125, 6.0, -1.5, 12.0,
                      -0.0625, 24.0, -3.0, 48.0, -6.0, 96.0, -12.0, 192.0]),
    ])
    packed = rtl.fp8_pack(pair, scale_e8m0=127)
    assert packed.shape == (16,)
    assert packed.dtype == torch.int32
    unpacked = rtl.fp8_unpack(packed, scale_e8m0=127)
    assert unpacked.shape == (2, 16)
    # Every value in `pair` is exactly representable in E4M3, so
    # round-trip with scale=0 (e8m0=127) is exact.
    assert torch.allclose(unpacked, pair, atol=1e-6)


# ----------------------------------------------------------------
#  Shape / reshape round-trips (flattening + zero-pad path)
# ----------------------------------------------------------------

def test_pointwise_preserves_higher_rank_shape(
    rtl: VectorRTLFunctions,
) -> None:
    a = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16) * 0.125
    got = rtl.square(a)
    assert got.shape == a.shape
    assert torch.allclose(got, _quant(a * a), rtol=1.0 / 64.0, atol=1e-3)


def test_pointwise_zero_pad_non_multiple(rtl: VectorRTLFunctions) -> None:
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 elements, not a multiple of 16
    got = rtl.mul(a, a)
    assert got.shape == a.shape
    assert torch.allclose(got, _quant(a * a), rtol=1.0 / 64.0, atol=1e-3)
