"""
quant_types.py
--------------
Quantization format definitions for the four formats under test:
BF16 (baseline), FP16, FP8-E4M3, FP8-E5M2.

FP8 uses per-tensor power-of-two (po2) scaling exclusively:
    scale = 2^floor(log2(max(|x|) / fp8_max_po2))
This is hardware-compatible (pure exponent shift, no float multiply).
Scale computation and BF16→E4M3 rounding are centralised in hw_float_ops.py.

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

from .hw_float_ops import quantize_bf16_to_e4m3


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

# Largest power-of-two ≤ max finite value; used for E5M2 po2 scale.
# E4M3: scale handled via hw_float_ops.quantize_bf16_to_e4m3 (centralised).
# E5M2: max finite = 57344 = 1.75 * 2^15, largest po2 = 2^15 = 32768
_FP8_MAX_PO2: dict[QuantFormat, float] = {
    QuantFormat.FLOAT8_E4M3: 256.0,    # 2^8  — kept for reference; E4M3 uses hw_float_ops
    QuantFormat.FLOAT8_E5M2: 32768.0,  # 2^15
}

# Minimum normal magnitude: hardware flushes FP8 subnormals to zero (DAZ).
# E4M3 (bias=7):  emin = 1-7 = -6  → min_normal = 2^-6
# E5M2 (bias=15): emin = 1-15 = -14 → min_normal = 2^-14
_FP8_MIN_NORMAL: dict[QuantFormat, float] = {
    QuantFormat.FLOAT8_E4M3: 2.0 ** -6,    # 0.015625
    QuantFormat.FLOAT8_E5M2: 2.0 ** -14,   # ~6.1e-5
}


# ---------------------------------------------------------------------------
# FP8 scaling mode — po2 only
# ---------------------------------------------------------------------------

def set_fp8_mode(mode: str) -> None:
    """Accept 'po2' only.  'scaled' mode has been removed; hardware uses po2."""
    if mode != "po2":
        raise ValueError(
            f"Only 'po2' FP8 mode is supported (hardware-compatible power-of-two "
            f"scaling).  'scaled' mode has been removed.  Got: {mode!r}"
        )


def get_fp8_mode() -> str:
    """Always returns 'po2' — the only supported mode."""
    return "po2"


# ---------------------------------------------------------------------------
# Core quantization function
# ---------------------------------------------------------------------------

def quant(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    Quantize tensor `x` to format `fmt` and return in the original dtype.

    For PASSTHROUGH: returns x unchanged (true no-op, any dtype).
    For BFLOAT16/FLOAT16: round-trip cast (no-op for bf16 models at baseline).
    For FP8-E4M3: hardware-faithful po2 scaling + BF16→E4M3 RNE rounding
                  via hw_float_ops.quantize_bf16_to_e4m3.
    For FP8-E5M2: po2 scaling via PyTorch cast (E5M2 rounding not yet ported).
    """
    if fmt is QuantFormat.PASSTHROUGH:
        return x

    if fmt is QuantFormat.FLOAT8_E4M3:
        return _quant_fp8_e4m3(x)

    if fmt is QuantFormat.FLOAT8_E5M2:
        return _quant_fp8_e5m2(x)

    target = TORCH_DTYPE[fmt]
    return x.float().to(target).to(x.dtype)


# ---------------------------------------------------------------------------
# E4M3: delegates to hw_float_ops (centralised scale + rounding)
# ---------------------------------------------------------------------------

def _quant_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """
    Per-tensor po2 scaling + hardware-faithful BF16→E4M3 RNE rounding.

    Scale computation and rounding live in hw_float_ops.quantize_bf16_to_e4m3.
    Corner cases (NaN flush, inf saturate, subnormal DAZ) are handled there.
    """
    e4m3_bytes, scale_exp = quantize_bf16_to_e4m3(x)
    # Reconstruct dequantized float: (E4M3 value) * scale
    fp8_vals = e4m3_bytes.view(torch.float8_e4m3fn).to(torch.float32)
    return (fp8_vals * (2.0 ** scale_exp)).to(x.dtype)


# ---------------------------------------------------------------------------
# E5M2: po2 scaling via PyTorch cast
# ---------------------------------------------------------------------------

def _quant_fp8_e5m2(x: torch.Tensor) -> torch.Tensor:
    """
    Per-tensor po2 scaling for E5M2.  Uses PyTorch's .to(fp8) cast for
    rounding (not yet ported to a hardware-faithful implementation).

    Corner cases matched to RTL:
    - NaN inputs flushed to zero.
    - Inf inputs saturated to ±fp8_max by clamp.
    - FP8 subnormals flushed to zero (DAZ).
    """
    fp8_max     = _FP8_MAX[QuantFormat.FLOAT8_E5M2]
    fp8_max_po2 = _FP8_MAX_PO2[QuantFormat.FLOAT8_E5M2]
    target      = TORCH_DTYPE[QuantFormat.FLOAT8_E5M2]

    x_f32 = x.float().nan_to_num(nan=0.0)
    amax  = x_f32.abs().nan_to_num(posinf=0.0).max()

    if amax == 0:
        if not x_f32.any():
            return x_f32.to(x.dtype)
        scale = 1.0
    else:
        scale = 2.0 ** math.floor(math.log2((amax / fp8_max_po2).item()))

    x_scaled = (x_f32 / scale).clamp(-fp8_max, fp8_max)
    x_q = x_scaled.to(target).to(torch.float32)
    x_q[x_q.abs() < _FP8_MIN_NORMAL[QuantFormat.FLOAT8_E5M2]] = 0.0
    return (x_q * scale).to(x.dtype)


# ---------------------------------------------------------------------------
# Raw FP8 capture (for matmul IO golden-data storage)
# ---------------------------------------------------------------------------

def quant_fp8_raw(
    x: torch.Tensor,
    fmt: QuantFormat = QuantFormat.FLOAT8_E4M3,
) -> tuple[torch.Tensor, int]:
    """
    Quantize x to FP8 and return (raw_bytes, scale_exp).

    raw_bytes  : uint8 tensor, same shape as x, raw FP8 bit patterns.
                 View as torch.float8_e4m3fn (or e5m2) to interpret.
    scale_exp  : int.  scale = 2 ** scale_exp.
                 Recover quantized values:
                     raw_bytes.view(fp8_dtype) * (2 ** scale_exp)

    For FLOAT8_E4M3: delegates to hw_float_ops.quantize_bf16_to_e4m3 —
    same hardware-faithful scale and rounding used by the FM path.
    For FLOAT8_E5M2: uses PyTorch cast (rounding not yet ported).
    """
    if fmt is QuantFormat.FLOAT8_E4M3:
        return quantize_bf16_to_e4m3(x)

    # E5M2 fallback
    fp8_max     = _FP8_MAX[fmt]
    fp8_max_po2 = _FP8_MAX_PO2[fmt]
    target      = TORCH_DTYPE[fmt]

    x_f32 = x.float()
    amax  = x_f32.abs().max().item()
    if amax == 0:
        raw = torch.zeros_like(x_f32).to(target).view(torch.uint8)
        return raw, 0

    scale_exp = int(math.floor(math.log2(amax / fp8_max_po2)))
    scale     = 2.0 ** scale_exp
    x_scaled  = (x_f32 / scale).clamp(-fp8_max, fp8_max)
    raw       = x_scaled.to(target).view(torch.uint8)
    return raw, scale_exp


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
