"""
quant_types.py
--------------
Quantization format definitions for the four formats under test:
BF16 (baseline), FP16, FP8-E4M3, FP8-E5M2.

Three FP8 scaling modes are supported (selected via set_fp8_mode):

  "scaled"   (default) — per-tensor absmax scaling.
      scale = max(|x|) / fp8_max
      Matches NVIDIA Transformer Engine behavior.

  "clamped"  — no scaling; values outside the representable range are
      clamped to ±fp8_max, and subnormals below fp8_min are flushed to zero.
      Tests raw FP8 without any dynamic range adjustment.

  "mx"       — MX-compliant power-of-two block scaling (OCP MX spec §6.3).
      scale = largest power-of-two <= max(|x|) / fp8_max_po2
      where fp8_max_po2 is the largest power-of-two representable in the
      element type.  Out-of-range normals are clamped after scaling.

BFLOAT16 is the baseline (native model dtype): a bf16->bf16 round-trip is
a no-op for bf16 weights/activations, giving zero quantization RMSE.

FP8 notes:
    torch.float8_e4m3fn  — 1 sign, 4 exponent, 3 mantissa bits.
                           Finite range ±448.  NaN represented; no ±Inf.
    torch.float8_e5m2    — 1 sign, 5 exponent, 2 mantissa bits.
                           Finite range ±57344.  Has NaN and ±Inf.
    Both require PyTorch >= 2.1.
"""

from __future__ import annotations

import math
import torch
from enum import Enum


# ---------------------------------------------------------------------------
# Format enum
# ---------------------------------------------------------------------------

class QuantFormat(str, Enum):
    BFLOAT16    = "bfloat16"      # baseline — no-op for bf16 models
    FLOAT16     = "float16"
    FLOAT8_E4M3 = "float8_e4m3"  # torch.float8_e4m3fn
    FLOAT8_E5M2 = "float8_e5m2"  # torch.float8_e5m2


# Map QuantFormat → torch.dtype
TORCH_DTYPE: dict[QuantFormat, torch.dtype] = {
    QuantFormat.BFLOAT16:    torch.bfloat16,
    QuantFormat.FLOAT16:     torch.float16,
    QuantFormat.FLOAT8_E4M3: torch.float8_e4m3fn,
    QuantFormat.FLOAT8_E5M2: torch.float8_e5m2,
}

# Format metadata for reporting
FORMAT_BITS: dict[QuantFormat, dict] = {
    QuantFormat.BFLOAT16:    {"total": 16, "exp": 8,  "mantissa": 7},
    QuantFormat.FLOAT16:     {"total": 16, "exp": 5,  "mantissa": 10},
    QuantFormat.FLOAT8_E4M3: {"total": 8,  "exp": 4,  "mantissa": 3},
    QuantFormat.FLOAT8_E5M2: {"total": 8,  "exp": 5,  "mantissa": 2},
}


# ---------------------------------------------------------------------------
# FP8 constants
# ---------------------------------------------------------------------------

_FP8_FORMATS = {QuantFormat.FLOAT8_E4M3, QuantFormat.FLOAT8_E5M2}

# Max representable finite value
_FP8_MAX: dict[QuantFormat, float] = {
    QuantFormat.FLOAT8_E4M3: 448.0,
    QuantFormat.FLOAT8_E5M2: 57344.0,
}

# Smallest positive normal (for flush-to-zero in clamped mode)
# E4M3: bias=7, min_exp=1-7=-6, min_normal = 2^-6 = 0.015625
# E5M2: bias=15, min_exp=1-15=-14, min_normal = 2^-14 ≈ 6.1e-5
_FP8_MIN_NORMAL: dict[QuantFormat, float] = {
    QuantFormat.FLOAT8_E4M3: 2**-6,    # 0.015625
    QuantFormat.FLOAT8_E5M2: 2**-14,   # ~6.1e-5
}

# Largest power-of-two representable in each element type
# E4M3: mantissa=1.000 (implicit 1 + 0 frac bits used), max_exp = 2^(15-bias-1)
#        Actually: max finite = 448 = 1.75 * 2^8, so largest po2 = 2^8 = 256
# E5M2: max finite = 57344 = 1.75 * 2^15, so largest po2 = 2^15 = 32768
_FP8_MAX_PO2: dict[QuantFormat, float] = {
    QuantFormat.FLOAT8_E4M3: 256.0,    # 2^8
    QuantFormat.FLOAT8_E5M2: 32768.0,  # 2^15
}


# ---------------------------------------------------------------------------
# FP8 scaling mode (module-level state)
# ---------------------------------------------------------------------------

_fp8_mode: str = "scaled"  # one of: "scaled", "clamped", "mx"


def set_fp8_mode(mode: str) -> None:
    """Set the FP8 quantization mode: 'scaled', 'clamped', or 'mx'."""
    global _fp8_mode
    if mode not in ("scaled", "clamped", "mx"):
        raise ValueError(f"Unknown FP8 mode: {mode!r}. Must be 'scaled', 'clamped', or 'mx'.")
    _fp8_mode = mode


def get_fp8_mode() -> str:
    """Return the current FP8 quantization mode."""
    return _fp8_mode


# ---------------------------------------------------------------------------
# Core quantization function
# ---------------------------------------------------------------------------

def quant(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    Quantize tensor `x` to format `fmt` and return in the original dtype.

    For BFLOAT16: no-op for bf16 models (baseline, zero RMSE).
    For FP8 formats: behavior depends on the current FP8 mode
        (set via set_fp8_mode).
    For FP16: raw cast (range ±65504 is sufficient for typical values).
    """
    if fmt in _FP8_FORMATS:
        if _fp8_mode == "scaled":
            return _quant_fp8_scaled(x, fmt)
        elif _fp8_mode == "clamped":
            return _quant_fp8_clamped(x, fmt)
        elif _fp8_mode == "mx":
            return _quant_fp8_mx(x, fmt)

    target = TORCH_DTYPE[fmt]
    return x.float().to(target).to(x.dtype)


# ---------------------------------------------------------------------------
# Mode 1: Per-tensor absmax scaling (default)
# ---------------------------------------------------------------------------

def _quant_fp8_scaled(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    Per-tensor absmax scaling for FP8 quantization.

    scale = max(|x|) / fp8_max
    x_scaled = x / scale          → fits within [-fp8_max, fp8_max]
    x_q = cast(x_scaled, fp8)     → round to FP8 grid
    return x_q * scale            → restore original magnitude
    """
    target = TORCH_DTYPE[fmt]
    fp8_max = _FP8_MAX[fmt]

    x_f32 = x.float()
    amax = x_f32.abs().max()

    if amax == 0:
        return x_f32.to(x.dtype)

    scale = amax / fp8_max
    x_scaled = x_f32 / scale
    x_q = x_scaled.to(target).to(torch.float32)
    return (x_q * scale).to(x.dtype)


# ---------------------------------------------------------------------------
# Mode 2: Clamped (no scaling, clamp to range, flush subnormals)
# ---------------------------------------------------------------------------

def _quant_fp8_clamped(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    Raw FP8 quantization with clamping — no scaling.

    Values > fp8_max are clamped to fp8_max (preserving sign).
    Values with |x| < fp8_min_normal are flushed to zero.
    Values within range are cast to FP8 and back (RNE rounding).
    """
    target = TORCH_DTYPE[fmt]
    fp8_max = _FP8_MAX[fmt]
    fp8_min = _FP8_MIN_NORMAL[fmt]

    x_f32 = x.float()

    # Clamp to representable range
    x_clamped = x_f32.clamp(-fp8_max, fp8_max)

    # Flush subnormals to zero
    x_clamped = torch.where(x_clamped.abs() < fp8_min, torch.zeros_like(x_clamped), x_clamped)

    # Cast through FP8 for RNE rounding on the remaining values
    x_q = x_clamped.to(target).to(torch.float32)
    return x_q.to(x.dtype)


# ---------------------------------------------------------------------------
# Mode 3: MX-compliant power-of-two block scaling (OCP MX spec §6.3)
# ---------------------------------------------------------------------------

def _quant_fp8_mx(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    MX-compliant power-of-two block scaling.

    Per OCP MX spec §6.3:
    1. X = largest power-of-two <= max(|V|) / fp8_max_po2
    2. P_i = clamp(V_i / X, ±fp8_max) cast to element dtype (RNE)

    The block scale X is a power-of-two, so division by X is exact
    (just an exponent shift — no rounding from the scaling itself).
    """
    target = TORCH_DTYPE[fmt]
    fp8_max = _FP8_MAX[fmt]
    fp8_max_po2 = _FP8_MAX_PO2[fmt]

    x_f32 = x.float()
    amax = x_f32.abs().max()

    if amax == 0:
        return x_f32.to(x.dtype)

    # Step 1: block scale = largest po2 <= amax / fp8_max_po2
    raw_scale = amax / fp8_max_po2
    # floor to power-of-two: 2^floor(log2(raw_scale))
    log2_scale = math.floor(math.log2(raw_scale.item()))
    scale = 2.0 ** log2_scale

    # Step 2: scale inputs, clamp normals that exceed max, cast to FP8
    x_scaled = x_f32 / scale
    x_scaled = x_scaled.clamp(-fp8_max, fp8_max)
    x_q = x_scaled.to(target).to(torch.float32)

    return (x_q * scale).to(x.dtype)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def all_formats() -> list[QuantFormat]:
    return list(QuantFormat)


def sweep_pairs(
    include_baseline: bool = True,
) -> list[tuple[QuantFormat, QuantFormat]]:
    """
    Return all (input_fmt, output_fmt) combinations.

    With include_baseline=True (default): 4x4 = 16 pairs including BFLOAT16.
    With include_baseline=False: 3x3 = 9 pairs, reduced formats only.
    """
    fmts = all_formats() if include_baseline else [
        f for f in QuantFormat if f != QuantFormat.BFLOAT16
    ]
    return [(inf, outf) for inf in fmts for outf in fmts]