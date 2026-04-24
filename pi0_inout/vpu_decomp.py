"""
vpu_decomp.py
-------------
VPU decomposition functions for composite ops (LayerNorm, Softmax, GELU, SiLU)
and the atomic-op dispatch table (_VEC_FM_DISPATCH) for VectorRTLFunctions.

These are wrappers around VectorRTLFunctions (vrf) methods — the funct_models_vector
files are not modified or imported at module level.  A vrf instance is passed in
at call time so this file has no dependency on funct_models_vector being importable
until patch_vector_ops actually constructs one.

Row-reduction convention
------------------------
vrf.rsum(x) / vrf.rmax(x) consume TWO physical registers (2 * num_lanes = 32
elements) per reduce pass, returning shape (..., K // (2 * num_lanes)).
To get a full row sum/max we iterate the reduction until the last dim reaches 1
(_vpu_rsum_all / _vpu_rmax_all).  Padding:
  - rsum pads with zeros  (neutral element for addition)
  - rmax pads with -inf   (neutral element for max)
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Multi-stage row-reduction helpers
# ---------------------------------------------------------------------------

def _vpu_rsum_all(vrf, x: torch.Tensor) -> torch.Tensor:
    """Full row sum via staged VPU rsum.  (..., K) -> (..., 1).

    Each rsum call consumes 2 * num_lanes = 32 elements per reduce pass
    (two physical registers), returning (..., K // 32).
    """
    nl = vrf.num_lanes
    pair_width = 2 * nl          # 32 — two physical registers per reduce
    for _ in range(64):          # safety cap; log32(K) <= 7 for any realistic K
        if x.shape[-1] <= 1:
            break
        k = x.shape[-1]
        if k % pair_width != 0:
            extra = pair_width - (k % pair_width)
            pad = torch.zeros(*x.shape[:-1], extra, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        x = vrf.rsum(x)          # (..., k_padded // pair_width)
    return x                     # (..., 1)


def _vpu_rmax_all(vrf, x: torch.Tensor) -> torch.Tensor:
    """Full row max via staged VPU rmax.  (..., K) -> (..., 1).

    Each rmax call consumes 2 * num_lanes = 32 elements per reduce pass.
    Padding uses -inf so padding slots never win the max.
    """
    nl = vrf.num_lanes
    pair_width = 2 * nl          # 32 — two physical registers per reduce
    for _ in range(64):
        if x.shape[-1] <= 1:
            break
        k = x.shape[-1]
        if k % pair_width != 0:
            extra = pair_width - (k % pair_width)
            pad = torch.full(
                (*x.shape[:-1], extra), float("-inf"), dtype=x.dtype, device=x.device
            )
            x = torch.cat([x, pad], dim=-1)
        x = vrf.rmax(x)          # (..., k_padded // pair_width)
    return x                     # (..., 1)


# ---------------------------------------------------------------------------
# Composite VPU decompositions
# ---------------------------------------------------------------------------

def _vpu_rsqrt(vrf, x: torch.Tensor) -> torch.Tensor:
    """VPU sequence: sqrt → rcp.  Hardware has no rsqrt instruction."""
    return vrf.rcp(vrf.sqrt(x))


def _vpu_layer_norm(
    vrf,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """VPU-accurate 1-D LayerNorm over the last dimension.

    VPU sequence (per row of K elements):
      rsum÷K → mean
      sub(x, mean)
      mul(xc, xc) → xc²
      rsum÷K → var
      add(var, eps); sqrt; rcp → inv_std
      mul(xc, inv_std)          → x_norm
      mul(x_norm, weight); add(bias)
    """
    x = x.bfloat16()
    K = x.shape[-1]
    inv_K = 1.0 / K

    # ── Mean ──────────────────────────────────────────────────────────────────
    total = _vpu_rsum_all(vrf, x)                          # (..., 1)
    mean  = vrf.mul(total, torch.full_like(total, inv_K))  # (..., 1)

    # ── Center ────────────────────────────────────────────────────────────────
    x_c = vrf.sub(x, mean.expand_as(x))                   # (..., K)

    # ── Variance ──────────────────────────────────────────────────────────────
    sq      = vrf.mul(x_c, x_c)                           # (..., K)
    sq_sum  = _vpu_rsum_all(vrf, sq)                       # (..., 1)
    var     = vrf.mul(sq_sum, torch.full_like(sq_sum, inv_K))

    # ── Normalise ─────────────────────────────────────────────────────────────
    var_eps = vrf.add(var, torch.full_like(var, eps))
    inv_std = vrf.rcp(vrf.sqrt(var_eps))                   # (..., 1)
    x_norm  = vrf.mul(x_c, inv_std.expand_as(x_c))        # (..., K)

    # ── Affine ────────────────────────────────────────────────────────────────
    w = weight.bfloat16().expand_as(x_norm)
    out = vrf.mul(x_norm, w)
    if bias is not None:
        b_t = bias.bfloat16().expand_as(out)
        out = vrf.add(out, b_t)
    return out


def _vpu_softmax(vrf, x: torch.Tensor, dim: int) -> torch.Tensor:
    """VPU-accurate softmax over `dim`.

    VPU sequence: rmax; sub; exp; rsum; rcp; mul
    exp uses math fallback (HardFloat port pending) — otherwise HW-accurate.
    """
    x = x.bfloat16()
    ndim = x.dim()
    d = dim if dim >= 0 else ndim + dim

    # Move target dim to last for row-reduction
    if d != ndim - 1:
        x = x.transpose(d, -1).contiguous()
        transposed = True
    else:
        transposed = False

    x_max     = _vpu_rmax_all(vrf, x)                     # (..., 1)
    x_shifted = vrf.sub(x, x_max.expand_as(x))
    exp_x     = vrf.exp(x_shifted)
    exp_sum   = _vpu_rsum_all(vrf, exp_x)                  # (..., 1)
    inv_sum   = vrf.rcp(exp_sum)
    out       = vrf.mul(exp_x, inv_sum.expand_as(exp_x))

    if transposed:
        out = out.transpose(d, -1).contiguous()
    return out


def _vpu_gelu_tanh(vrf, x: torch.Tensor) -> torch.Tensor:
    """VPU-accurate GELU (tanh approximation).

    VPU sequence: mul,mul→x³; mul(0.044715); add(x); mul(√(2/π)); tanh; add(1); mul(0.5); mul(x)
    tanh uses TanhRec LUT — HW-accurate.
    """
    x = x.bfloat16()
    c1           = torch.full_like(x, 0.044715)
    sqrt_2_over_pi = torch.full_like(x, math.sqrt(2.0 / math.pi))
    one          = torch.full_like(x, 1.0)
    half         = torch.full_like(x, 0.5)

    x3     = vrf.mul(vrf.mul(x, x), x)                    # x³
    inner  = vrf.add(x, vrf.mul(x3, c1))                  # x + 0.044715·x³
    scaled = vrf.mul(inner, sqrt_2_over_pi)                # √(2/π)·(...)
    t      = vrf.tanh(scaled)
    gate   = vrf.mul(vrf.add(t, one), half)                # 0.5·(1 + tanh)
    return vrf.mul(x, gate)


def _vpu_silu(vrf, x: torch.Tensor) -> torch.Tensor:
    """VPU-accurate SiLU: x · σ(x).

    VPU sequence: neg(x); exp(−x); add(1); rcp → σ(x); mul(x, σ)
    exp uses math fallback (HardFloat port pending).
    """
    x = x.bfloat16()
    neg_x   = vrf.sub(torch.zeros_like(x), x)
    exp_neg = vrf.exp(neg_x)
    denom   = vrf.add(torch.full_like(exp_neg, 1.0), exp_neg)
    sigmoid = vrf.rcp(denom)
    return vrf.mul(x, sigmoid)


# ---------------------------------------------------------------------------
# _pow_dispatch — handles pow.Tensor_Scalar
# ---------------------------------------------------------------------------

def _pow_dispatch(vrf, a: torch.Tensor, exp) -> torch.Tensor:
    """Map pow(a, 2) → vrf.square, pow(a, 3) → vrf.cube, else PyTorch fallback."""
    if exp == 2 or exp == 2.0:
        return vrf.square(a)
    if exp == 3 or exp == 3.0:
        return vrf.cube(a)
    # Uncommon exponent — VPU has no general power unit; fall back to PyTorch.
    # _in_quant_guard is True when this is called, so the dispatch re-enters
    # __torch_dispatch__ and immediately passes through.
    return torch.pow(a, exp)


# ---------------------------------------------------------------------------
# Atomic pointwise dispatch table
# Used by the else-branch in VectorQuantMode.__torch_dispatch__ for ops that
# do not require dimension-aware reduction handling.
# Signature: fn(vrf, *bf16_args) -> tensor
# ---------------------------------------------------------------------------

def _bcast_both(a: torch.Tensor, b) -> tuple:
    """Expand both a and b to their common broadcast shape."""
    if not isinstance(b, torch.Tensor):
        return a, torch.full_like(a, float(b))
    if a.shape == b.shape:
        return a, b
    a_exp, b_exp = torch.broadcast_tensors(a, b)
    return a_exp.contiguous(), b_exp.contiguous()


_VEC_FM_DISPATCH: dict = {
    # ── Binary elementwise ────────────────────────────────────────────────────
    # _bcast_both handles general broadcasting (e.g. outer-product masks);
    # Scalar variants handle literal scalars separately.
    torch.ops.aten.add.Tensor:
        lambda vrf, a, b, *rest: vrf.add(*_bcast_both(a, b)),
    torch.ops.aten.add.Scalar:
        # alpha scaling (3rd arg) folded into scalar: a + alpha*b
        lambda vrf, a, b, *rest: vrf.add(
            a, torch.full_like(a, float(b) * (float(rest[0]) if rest else 1.0))
        ),
    torch.ops.aten.sub.Tensor:
        lambda vrf, a, b, *rest: vrf.sub(*_bcast_both(a, b)),
    torch.ops.aten.sub.Scalar:
        lambda vrf, a, b, *rest: vrf.sub(
            a, torch.full_like(a, float(b) * (float(rest[0]) if rest else 1.0))
        ),
    torch.ops.aten.mul.Tensor:
        lambda vrf, a, b: vrf.mul(*_bcast_both(a, b)),
    torch.ops.aten.mul.Scalar:
        lambda vrf, a, b: vrf.mul(a, torch.full_like(a, b)),
    torch.ops.aten.div.Tensor:
        # VPU: rcp then mul (two lane-box passes)
        lambda vrf, a, b: (lambda ae, be: vrf.mul(ae, vrf.rcp(be)))(*_bcast_both(a, b)),
    torch.ops.aten.pow.Tensor_Scalar:
        _pow_dispatch,
    # ── Unary elementwise ─────────────────────────────────────────────────────
    torch.ops.aten.reciprocal.default:
        lambda vrf, a: vrf.rcp(a),
    torch.ops.aten.sqrt.default:
        lambda vrf, a: vrf.sqrt(a),
    torch.ops.aten.neg.default:
        lambda vrf, a: vrf.sub(torch.zeros_like(a), a),
    torch.ops.aten.sin.default:
        lambda vrf, a: vrf.sin(a),
    torch.ops.aten.cos.default:
        lambda vrf, a: vrf.cos(a),
    torch.ops.aten.tanh.default:
        lambda vrf, a: vrf.tanh(a),
    torch.ops.aten.exp.default:
        lambda vrf, a: vrf.exp(a),
    torch.ops.aten.exp2.default:
        lambda vrf, a: vrf.exp2(a),
    torch.ops.aten.log2.default:
        lambda vrf, a: vrf.log2(a),
}
