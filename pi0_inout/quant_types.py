"""
quant_types.py
--------------
Quantization format definitions for the four formats under test:
BF16 (baseline), FP16, FP8-E4M3, FP8-E5M2.

Two FP8 scaling modes are supported (selected via set_fp8_mode):

  "po2"     (default) — per-tensor power-of-two scaling.
      scale = largest power-of-two <= max(|x|) / fp8_max_po2
      Same per-tensor scope as "scaled", but the scale is constrained to a
      power of two so that hardware can implement it as a pure exponent shift.

  "scaled"            — per-tensor absmax scaling.
      scale = max(|x|) / fp8_max          (any float value)
      Matches NVIDIA Transformer Engine behavior.

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
    PASSTHROUGH = "passthrough"   # identity — returns tensor unchanged, any dtype
    BFLOAT16    = "bfloat16"      # no-op for bf16 models; rounds fp32 to bf16 precision
    FLOAT16     = "float16"
    FLOAT8_E4M3 = "float8_e4m3"  # torch.float8_e4m3fn
    FLOAT8_E5M2 = "float8_e5m2"  # torch.float8_e5m2


# Map QuantFormat → torch.dtype  (PASSTHROUGH has no target dtype — handled separately)
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

# Largest power-of-two representable in each element type.
# Used by "po2" mode to compute the per-tensor power-of-two scale.
# E4M3: max finite = 448 = 1.75 * 2^8, so largest po2 = 2^8 = 256
# E5M2: max finite = 57344 = 1.75 * 2^15, so largest po2 = 2^15 = 32768
_FP8_MAX_PO2: dict[QuantFormat, float] = {
    QuantFormat.FLOAT8_E4M3: 256.0,    # 2^8
    QuantFormat.FLOAT8_E5M2: 32768.0,  # 2^15
}


# ---------------------------------------------------------------------------
# FP8 scaling mode (module-level state)
# ---------------------------------------------------------------------------

_fp8_mode: str = "po2"  # one of: "po2", "scaled"


def set_fp8_mode(mode: str) -> None:
    """Set the FP8 quantization mode: 'po2' or 'scaled'."""
    global _fp8_mode
    if mode not in ("po2", "scaled"):
        raise ValueError(f"Unknown FP8 mode: {mode!r}. Must be 'po2' or 'scaled'.")
    _fp8_mode = mode


def get_fp8_mode() -> str:
    """Return the current FP8 quantization mode ('po2' or 'scaled')."""
    return _fp8_mode


# ---------------------------------------------------------------------------
# Core quantization function
# ---------------------------------------------------------------------------

def quant(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    Quantize tensor `x` to format `fmt` and return in the original dtype.

    For PASSTHROUGH: returns x unchanged (true no-op, any dtype).
    For BFLOAT16/FLOAT16: round-trip cast (no-op for bf16 models at baseline).
    For FP8: per-tensor scaling using the current mode (set via set_fp8_mode).
    """
    if fmt is QuantFormat.PASSTHROUGH:
        return x

    if fmt in _FP8_FORMATS:
        if _fp8_mode == "po2":
            return _quant_fp8_po2(x, fmt)
        else:  # "scaled"
            return _quant_fp8_scaled(x, fmt)

    target = TORCH_DTYPE[fmt]
    return x.float().to(target).to(x.dtype)


# ---------------------------------------------------------------------------
# Mode 1: Per-tensor power-of-two scaling (default)
# ---------------------------------------------------------------------------

def _quant_fp8_po2(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    Per-tensor power-of-two scaling.

    The scale is constrained to a power of two so hardware can implement
    it as a pure exponent shift (no floating-point multiply needed):

    scale = 2^floor(log2(max(|x|) / fp8_max_po2))   (power-of-two)
    x_q   = clamp(x / scale, ±fp8_max) cast to fp8
    out   = x_q * scale
    """
    target = TORCH_DTYPE[fmt]
    fp8_max = _FP8_MAX[fmt]
    fp8_max_po2 = _FP8_MAX_PO2[fmt]

    x_f32 = x.float()
    amax = x_f32.abs().max()

    if amax == 0:
        return x_f32.to(x.dtype)

    raw_scale = amax / fp8_max_po2
    scale = 2.0 ** math.floor(math.log2(raw_scale.item()))

    x_scaled = (x_f32 / scale).clamp(-fp8_max, fp8_max)
    x_q = x_scaled.to(target).to(torch.float32)
    return (x_q * scale).to(x.dtype)


# ---------------------------------------------------------------------------
# Mode 2: Per-tensor absmax scaling
# ---------------------------------------------------------------------------

def _quant_fp8_scaled(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    Per-tensor absmax scaling.

    scale = max(|x|) / fp8_max       (any float)
    x_q   = cast(x / scale, fp8)
    out   = x_q * scale
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
