"""
ulp_noise.py
------------
Utilities for computing 1-ULP step sizes and injecting controlled ULP noise.

We compute ULP for a *format* (fp32/fp16/bf16/fp8) at the magnitude of each value.

For a normalized value with unbiased exponent e and fraction bits f:
  ulp = 2 ** (e - f)

Using torch.frexp:
  x = m * 2**exp, with m in [0.5, 1) for non-zero finite x
  => unbiased exponent e = exp - 1

For subnormals (|x| < min_normal), spacing is constant:
  ulp_sub = 2 ** (emin - f)
where emin is the minimum normal exponent for the format.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from .quant_types import QuantFormat


class UlpNoiseMode(str, Enum):
    RANDOM = "random"  # random ±
    PLUS = "plus"      # always +ULP
    MINUS = "minus"    # always -ULP


@dataclass(frozen=True)
class UlpNoiseConfig:
    n_ulp: int = 0
    mode: UlpNoiseMode = UlpNoiseMode.RANDOM
    ulp_fmt: QuantFormat = QuantFormat.FLOAT16  # format whose ULP grid defines the step size

    def enabled(self) -> bool:
        return self.n_ulp != 0


# Minimum normal exponent (emin) and fraction bits (explicit mantissa bits) for each format.
# These are used for ULP spacing computations.
_FMT_PARAMS: dict[QuantFormat, dict[str, int]] = {
    QuantFormat.FLOAT32:     {"emin": -126, "frac_bits": 23},
    QuantFormat.FLOAT16:     {"emin": -14,  "frac_bits": 10},
    QuantFormat.BFLOAT16:    {"emin": -126, "frac_bits": 7},
    # fp8 formats (IEEE-like). torch.float8_e4m3fn has finite-only range; ULP spacing rule still follows e/f.
    QuantFormat.FLOAT8_E4M3: {"emin": -6,   "frac_bits": 3},
    QuantFormat.FLOAT8_E5M2: {"emin": -14,  "frac_bits": 2},
}


def ulp_step(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """
    Compute the 1-ULP step size (spacing) of `fmt` at each element of `x`.

    Args:
        x:   Tensor (any floating dtype). Values are interpreted in real numbers.
        fmt: Target floating-point format whose spacing is computed.

    Returns:
        Float32 tensor with same shape as x containing the local ULP step size.
        Non-finite inputs yield NaN step size.
    """
    if fmt not in _FMT_PARAMS:
        raise ValueError(f"ULP params not defined for format: {fmt}")

    params = _FMT_PARAMS[fmt]
    emin = params["emin"]
    frac_bits = params["frac_bits"]

    x_f32 = x.float()
    ax = x_f32.abs()

    finite = torch.isfinite(x_f32)

    # min_normal = 2**emin
    min_normal = torch.ldexp(torch.ones((), device=x_f32.device, dtype=torch.float32), emin)
    ulp_sub = torch.ldexp(torch.ones((), device=x_f32.device, dtype=torch.float32), emin - frac_bits)

    # torch.frexp: ax = m * 2**exp, m in [0.5, 1) for ax>0
    m, exp = torch.frexp(ax)
    # unbiased exponent for normalized numbers
    e = exp - 1
    ulp_norm = torch.ldexp(torch.ones_like(x_f32), e - frac_bits)

    # For 0 and subnormals, use constant spacing.
    step = torch.where(ax < min_normal, ulp_sub, ulp_norm)
    step = torch.where(finite, step, torch.full_like(step, float("nan")))
    return step


def inject_ulp_noise(
    y: torch.Tensor,
    *,
    n_ulp: int,
    mode: UlpNoiseMode,
    ulp_fmt: QuantFormat,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Add ±n_ulp * ULP(y, ulp_fmt) to y.

    Intended usage: inject noise into matmul outputs to emulate systematic/random rounding drift.
    """
    if n_ulp == 0:
        return y
    if n_ulp < 0:
        raise ValueError("n_ulp must be >= 0")

    y_f32 = y.float()
    step = ulp_step(y_f32, ulp_fmt)

    if mode == UlpNoiseMode.PLUS:
        sign = 1.0
    elif mode == UlpNoiseMode.MINUS:
        sign = -1.0
    elif mode == UlpNoiseMode.RANDOM:
        # Sample in {+1, -1} with equal probability.
        r = torch.randint(
            low=0,
            high=2,
            size=y_f32.shape,
            device=y_f32.device,
            generator=generator,
        )
        sign = (r * 2 - 1).to(torch.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return (y_f32 + (float(n_ulp) * step * sign)).to(y.dtype)

