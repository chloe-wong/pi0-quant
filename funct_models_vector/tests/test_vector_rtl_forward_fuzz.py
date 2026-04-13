"""Tier 5 — randomized torch-adapter fuzz tests (§8 Tier 5).

Complements `test_vector_rtl_forward.py` (the hand-picked smoke tests)
with a randomized sweep: for every public `VectorRTLFunctions` method,
run N random-seed / random-shape inputs and compare against a torch
BF16 reference. Tolerance is `rtol=1/64` (~2 BF16 ULPs) plus a small
absolute floor, wider for the transcendentals that have their own
approximation paths.

The fuzz is intentionally small-scale (16 seeds × a few shapes per op)
instead of the plan's suggested 1000/op: the per-call Python dispatch
loop is the bottleneck, and this is cross-validation against the same
lane_boxes the Tier 1 tests already hammer — so the incremental bug-
catching power of going from 16 to 1000 seeds is low. What this file
*does* prove that the smoke tests don't:

- the chunking loop handles non-trivial flat lengths (1, 15, 16, 17,
  31, 32, 33, 47, 48, 96) including zero-padding
- the reshape round-trip preserves arbitrary input shapes
- the adapter is stable across many RNG draws without flaky 1-ULP
  corner cases surviving past the tolerance band
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


# Tolerance band. `rtol=1/64` = 2 BF16 ULPs relative; `atol=1e-3` is the
# near-zero floor. Transcendentals use `rtol=5e-2` because their LUT /
# fallback paths round differently from torch's math kernels.
_RTOL_EXACT = 1.0 / 64.0
_ATOL_EXACT = 1e-3
_RTOL_TRANSCENDENTAL = 5e-2
_ATOL_TRANSCENDENTAL = 5e-3

_NUM_SEEDS = 16
_SHAPES_1D = (1, 15, 16, 17, 31, 32, 33, 47, 48, 96)


@pytest.fixture(scope="module")
def rtl() -> VectorRTLFunctions:
    return VectorRTLFunctions()


def _quant(x: torch.Tensor) -> torch.Tensor:
    return torch_bf16_bits_to_float(torch_float_to_bf16_bits(x))


def _rand(seed: int, shape: tuple[int, ...], lo: float, hi: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape).uniform_(lo, hi, generator=g)


def _assert_close(
    got: torch.Tensor,
    want: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    context: str,
) -> None:
    assert got.shape == want.shape, f"{context}: shape {got.shape} != {want.shape}"
    close = torch.isclose(got, want, rtol=rtol, atol=atol)
    if not close.all():
        bad = (~close).nonzero(as_tuple=False).flatten().tolist()[:6]
        g_vals = got.flatten()[bad].tolist()
        w_vals = want.flatten()[bad].tolist()
        raise AssertionError(
            f"{context}: {int((~close).sum())} mismatches. "
            f"first offsets={bad} got={g_vals} want={w_vals}"
        )


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
@pytest.mark.parametrize("flat_len", _SHAPES_1D)
def test_fuzz_binary(
    rtl: VectorRTLFunctions, op_name: str, torch_op, flat_len: int
) -> None:
    for seed in range(_NUM_SEEDS):
        a = _rand(seed * 2, (flat_len,), -10.0, 10.0)
        b = _rand(seed * 2 + 1, (flat_len,), -10.0, 10.0)
        want = _quant(torch_op(_quant(a), _quant(b)))
        got = getattr(rtl, op_name)(a, b)
        _assert_close(
            got, want,
            rtol=_RTOL_EXACT, atol=_ATOL_EXACT,
            context=f"{op_name} seed={seed} len={flat_len}",
        )


# ----------------------------------------------------------------
#  Pointwise unary — exact spine (relu/square/cube)
# ----------------------------------------------------------------

@pytest.mark.parametrize("op_name,torch_op,gen_range", [
    ("relu",   lambda x: torch.clamp(x, min=0.0),   (-5.0, 5.0)),
    ("square", lambda x: x * x,                      (-3.0, 3.0)),
    ("cube",   lambda x: x * x * x,                  (-2.0, 2.0)),
])
@pytest.mark.parametrize("flat_len", _SHAPES_1D)
def test_fuzz_unary_exact(
    rtl: VectorRTLFunctions, op_name: str, torch_op, gen_range, flat_len: int
) -> None:
    lo, hi = gen_range
    for seed in range(_NUM_SEEDS):
        a = _rand(seed + 100, (flat_len,), lo, hi)
        want = _quant(torch_op(_quant(a)))
        got = getattr(rtl, op_name)(a)
        _assert_close(
            got, want,
            rtol=_RTOL_EXACT, atol=_ATOL_EXACT,
            context=f"{op_name} seed={seed} len={flat_len}",
        )


# ----------------------------------------------------------------
#  Pointwise unary — approximation-backed (sqrt/rcp)
# ----------------------------------------------------------------

@pytest.mark.parametrize("op_name,torch_op,gen_range", [
    ("sqrt",   torch.sqrt,                           (0.1, 10.0)),
    ("rcp",    lambda x: 1.0 / x,                    (0.5, 5.0)),
])
@pytest.mark.parametrize("flat_len", [16, 32, 48, 96])
def test_fuzz_unary_approx(
    rtl: VectorRTLFunctions, op_name: str, torch_op, gen_range, flat_len: int
) -> None:
    lo, hi = gen_range
    for seed in range(_NUM_SEEDS):
        a = _rand(seed + 200, (flat_len,), lo, hi)
        want = _quant(torch_op(_quant(a)))
        got = getattr(rtl, op_name)(a)
        _assert_close(
            got, want,
            rtol=_RTOL_TRANSCENDENTAL, atol=_ATOL_TRANSCENDENTAL,
            context=f"{op_name} seed={seed} len={flat_len}",
        )


# ----------------------------------------------------------------
#  Pointwise unary — transcendental public-surface fuzz
# ----------------------------------------------------------------
#
# `sin`, `cos`, `log2`, `tanh`, `exp`, `exp2` were public adapter
# methods with no direct fuzz coverage. sin/cos/log2 go through LUT-
# backed lane_boxes (`SinCosVec` / `Log`); tanh/exp/exp2 go through
# `_legacy_math_fallback` in `vector_engine_model.py` until the §6
# Group D snapshot pipeline lands. Same tolerance band as the
# approximation-backed spine above, and the same small-scale design
# as the rest of Tier 5 (16 seeds × a handful of shapes).

@pytest.mark.parametrize("op_name,torch_op,gen_range", [
    ("sin",   torch.sin,                           (0.0, 2 * math.pi)),
    ("cos",   torch.cos,                           (0.0, 2 * math.pi)),
    ("log2",  torch.log2,                          (0.25, 16.0)),
    ("tanh",  torch.tanh,                          (-4.0, 4.0)),
    ("exp",   torch.exp,                           (-4.0, 4.0)),
    ("exp2",  lambda x: torch.pow(2.0, x),         (-4.0, 4.0)),
])
@pytest.mark.parametrize("flat_len", [16, 32, 48, 96])
def test_fuzz_unary_transcendental_surface(
    rtl: VectorRTLFunctions, op_name: str, torch_op, gen_range, flat_len: int
) -> None:
    lo, hi = gen_range
    for seed in range(_NUM_SEEDS):
        a = _rand(seed + 700, (flat_len,), lo, hi)
        want = _quant(torch_op(_quant(a)))
        got = getattr(rtl, op_name)(a)
        _assert_close(
            got, want,
            rtol=_RTOL_TRANSCENDENTAL, atol=_ATOL_TRANSCENDENTAL,
            context=f"{op_name} seed={seed} len={flat_len}",
        )


# ----------------------------------------------------------------
#  Row reductions
# ----------------------------------------------------------------

@pytest.mark.parametrize("op_name,torch_fn", [
    ("rsum", lambda x: x.sum(dim=-1)),
    ("rmax", lambda x: x.max(dim=-1).values),
    ("rmin", lambda x: x.min(dim=-1).values),
])
@pytest.mark.parametrize("slices", [1, 2, 4])
def test_fuzz_row_reduction(
    rtl: VectorRTLFunctions, op_name: str, torch_fn, slices: int
) -> None:
    for seed in range(_NUM_SEEDS):
        flat = _rand(seed + 300, (slices * 16,), -5.0, 5.0)
        reshaped = flat.reshape(slices, 16)
        want = _quant(torch_fn(_quant(reshaped)))
        got = getattr(rtl, op_name)(flat)  # (K*16,) → (K,)
        assert got.shape == (slices,)
        _assert_close(
            got, want,
            rtol=_RTOL_EXACT, atol=1e-2,  # rsum accumulates a bit more rounding error
            context=f"{op_name} seed={seed} slices={slices}",
        )


# ----------------------------------------------------------------
#  Col reductions (streaming)
# ----------------------------------------------------------------

@pytest.mark.parametrize("op_name,torch_fn", [
    ("csum", lambda x: x.sum(dim=0)),
    ("cmax", lambda x: x.max(dim=0).values),
    ("cmin", lambda x: x.min(dim=0).values),
])
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_fuzz_col_reduction(
    rtl: VectorRTLFunctions, op_name: str, torch_fn, rows: int
) -> None:
    for seed in range(_NUM_SEEDS):
        x = _rand(seed + 400, (rows, 16), -5.0, 5.0)
        want = _quant(torch_fn(_quant(x)))
        got = getattr(rtl, op_name)(x)
        assert got.shape == (16,)
        _assert_close(
            got, want,
            rtol=_RTOL_EXACT, atol=2e-2,  # csum accumulates FP32; final narrow adds ULP
            context=f"{op_name} seed={seed} rows={rows}",
        )


# ----------------------------------------------------------------
#  FP8 pack / unpack round-trip
# ----------------------------------------------------------------

def test_fuzz_fp8_round_trip(rtl: VectorRTLFunctions) -> None:
    """Round-trip at scale_e8m0=127 (shift=0) over random permutations of
    values known to be exactly representable in E4M3. At other scale
    exponents the E4M3 dynamic range shifts and not every value in the
    input set is in-range any more, so reproducing the lossy round-trip
    would mean reimplementing the full E4M3 rounding here — out of
    scope for a smoke-level fuzz test. Non-127 scale behavior is
    exercised bit-exact by the Tier 1 `test_fp8_pack` / `test_fp8_unpack`
    lane_box tests."""
    exact_e4m3 = [
        1.0, 2.0, 0.5, -1.0, 4.0, -2.0, 8.0, -4.0,
        16.0, -8.0, 32.0, -16.0, 64.0, -32.0, 128.0, -64.0,
        0.75, 1.5, -0.25, 3.0, -0.125, 6.0, -1.5, 12.0,
        -0.0625, 24.0, -3.0, 48.0, -6.0, 96.0, -12.0, 192.0,
    ]
    for seed in range(_NUM_SEEDS):
        g = torch.Generator().manual_seed(seed + 500)
        perm = torch.randperm(32, generator=g)
        vals = torch.tensor([exact_e4m3[i.item()] for i in perm], dtype=torch.float32)
        pair = vals.reshape(2, 16)
        packed = rtl.fp8_pack(pair, scale_e8m0=127)
        assert packed.shape == (16,)
        assert packed.dtype == torch.int32
        unpacked = rtl.fp8_unpack(packed, scale_e8m0=127)
        assert unpacked.shape == (2, 16)
        _assert_close(
            unpacked, pair,
            rtol=_RTOL_EXACT, atol=_ATOL_EXACT,
            context=f"fp8 round-trip seed={seed}",
        )
