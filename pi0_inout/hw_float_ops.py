"""
hw_float_ops.py
---------------
Centralized hardware-faithful floating-point operations for quantization.

Scale computation and BF16→E4M3 rounding live here so they can be updated
in one place when funct_models_vector becomes importable.

FUTURE:
  compute_po2_scale_exp():   replace .bfloat16().abs().max() with a
                              cmax+rmax chain via VectorRTLFunctions for
                              RTL-accurate 16-lane BF16 amax reduction.
  _rtl_bf16_to_e4m3_byte():  already RTL-accurate (funct_models_vector.fp8_e4m3).
                              Per-element loop can be replaced with
                              VectorRTLFunctions.fp8_pack() in (2,16) chunks.
"""

from __future__ import annotations

import math

import torch

from funct_models_vector.fp8_e4m3 import bf16_to_e4m3_byte as _rtl_bf16_to_e4m3_byte


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_E4M3_FP8_MAX_PO2   = 256.0  # largest power-of-two ≤ E4M3 max (448)


# ---------------------------------------------------------------------------
# Per-tensor po2 scale computation
# ---------------------------------------------------------------------------

def compute_po2_scale_exp(x: torch.Tensor) -> int:
    """Return the integer exponent for the per-tensor po2 scale.

    scale = 2^scale_exp   where   scale_exp = floor(log2(amax / 256))

    amax is computed in BF16 precision (not float32) to match the hardware
    VPU rmax which operates on BF16 values.

    Returns 0 when the tensor is all-zero (scale = 1.0).

    FUTURE: Replace the Python .bfloat16().abs().max() with a call to
    VectorRTLFunctions.rmax() processing 16-lane BF16 chunks, which will
    give a bit-exact match to the hardware scale computation.
    """
    # Compute amax in BF16 precision — matches hardware VPU input width.
    # nan_to_num flushes NaN/Inf before max (same RTL behavior as _quant_fp8_po2).
    amax = x.bfloat16().abs().nan_to_num(nan=0.0, posinf=0.0).max().item()
    if amax == 0.0:
        return 0
    return int(math.floor(math.log2(amax / _E4M3_FP8_MAX_PO2)))


# ---------------------------------------------------------------------------
# Combined: quantize BF16 tensor to E4M3 with po2 scale
# ---------------------------------------------------------------------------

def quantize_bf16_to_e4m3(
    x: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Quantize tensor x to E4M3 using hardware-faithful po2 scaling and
    BF16→E4M3 RNE rounding.

    x should be BF16 (or will be cast to BF16 first — matching hardware
    input-bus precision).

    Returns:
        e4m3_bytes  uint8 tensor, same shape as x, raw E4M3 bit patterns.
                    Interpret with: e4m3_bytes.view(torch.float8_e4m3fn)
        scale_exp   int, scale = 2 ** scale_exp.
                    Recover float values:
                        e4m3_bytes.view(torch.float8_e4m3fn).float() * 2**scale_exp

    Used by:
        quant_types._quant_fp8_po2   — format-flag quantization path
        quant_types.quant_fp8_raw    — matmul IO capture
        quant_linear.QuantLinear     — FM pre-normalization + output rescaling

    Rounding uses funct_models_vector.fp8_e4m3.bf16_to_e4m3_byte, validated
    against FP8Pack.scala. Scale computation via compute_po2_scale_exp remains
    Python; see module FUTURE comment for the VPU upgrade path.
    """
    x_bf16 = x.bfloat16()
    scale_exp = compute_po2_scale_exp(x_bf16)

    # Apply BF16→E4M3 RNE rounding element-wise via RTL-accurate function.
    # exp_shift = scale_exp folds the po2 scale into the exponent arithmetic
    # inside the RTL function — no pre-scaling of the tensor is needed.
    # FUTURE: replace per-element loop with VectorRTLFunctions.fp8_pack()
    # chunks once the (2, 16) interface is plumbed through.
    bits = x_bf16.view(torch.int16).cpu().reshape(-1).tolist()
    e4m3_flat = [_rtl_bf16_to_e4m3_byte(b, scale_exp) for b in bits]
    e4m3_bytes = torch.tensor(
        e4m3_flat, dtype=torch.uint8, device=x.device,
    ).reshape(x.shape)

    return e4m3_bytes, scale_exp
