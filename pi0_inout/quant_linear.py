"""
quant_linear.py
---------------
QuantLinear: a drop-in replacement for nn.Linear that applies quantization
to matmul inputs (activation + weight) and matmul output separately.

Forward pass semantics
----------------------
Given input_fmt A and output_fmt B:

    x_q      = quant(x,    A)          # activation loaded from memory in A
    W_q      = quant(W,    A)          # weight loaded from memory in A
    b_q      = quant(bias, A)          # bias loaded from memory in A
    y_accum  = x_q @ W_q^T + b_q      # accumulation in float32 (matmul + bias add)
    result   = quant(y_accum, B)       # write result to memory in B

All three inputs (activation, weight, bias) are stored parameters/values read
from memory in input_fmt.  The float32 accumulator holds the running sum
throughout.  The single output quantization to B happens once, on the final
accumulated result including the bias.

FLOAT32 input_fmt and output_fmt are identity operations (no rounding).
A FLOAT32/FLOAT32 run has exactly zero quantization RMSE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .quant_types import QuantFormat, quant
from .stats_tracker import StatsTracker, Component
from .ulp_noise import UlpNoiseConfig, inject_ulp_noise


class QuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with configurable input/output quantization.

    Attributes:
        weight:       Shared with original linear (not copied).
        bias:         Shared with original linear (not copied).
        input_fmt:    Applied to both activation x and weight W before matmul.
        output_fmt:   Applied to matmul result and bias before addition.
        component:    Architectural component tag (vision/language/action_*).
        layer_name:   Full dot-separated module path, used as stats key.
        tracker:      Optional StatsTracker for RMSE collection.
    """

    def __init__(
        self,
        linear: nn.Linear,
        input_fmt: QuantFormat,
        output_fmt: QuantFormat,
        component: Component,
        layer_name: str,
        tracker: Optional[StatsTracker] = None,
        ulp_noise: Optional[UlpNoiseConfig] = None,
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias   = linear.bias

        self.input_fmt  = input_fmt
        self.output_fmt = output_fmt
        self.component  = component
        self.layer_name = layer_name
        self.tracker    = tracker
        self.ulp_noise  = ulp_noise

        self.in_features  = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        # All arithmetic in float32
        x_f32 = x.float()
        w_f32 = self.weight.float()
        b_f32 = self.bias.float() if self.bias is not None else None

        # ── Load all inputs from memory in input_fmt ─────────────────────────
        x_q = quant(x_f32, self.input_fmt)   # activation
        w_q = quant(w_f32, self.input_fmt)   # weight
        b_q = quant(b_f32, self.input_fmt) if b_f32 is not None else None  # bias

        # ── Accumulate in float32: matmul + bias add ──────────────────────────
        # Split matmul and bias-add so we can optionally inject ULP noise into
        # the matmul output specifically (as requested by the sweep).
        y_mm = F.linear(x_q, w_q, None)
        if self.ulp_noise is not None and self.ulp_noise.enabled():
            y_mm = inject_ulp_noise(
                y_mm,
                n_ulp=self.ulp_noise.n_ulp,
                mode=self.ulp_noise.mode,
                ulp_fmt=self.ulp_noise.ulp_fmt,
            )
        y_accum = y_mm if b_q is None else (y_mm + b_q)

        # ── Write result to output memory in output_fmt ───────────────────────
        y_out = quant(y_accum, self.output_fmt)

        # ── RMSE: compare against unquantized full-precision reference ─────────
        if self.tracker is not None:
            with torch.no_grad():
                y_fp = F.linear(x_f32, w_f32, b_f32)
                self.tracker.record(
                    name=self.layer_name,
                    component=self.component,
                    y_fp=y_fp,
                    y_quant=y_out,
                )

        return y_out.to(original_dtype)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"input_fmt={self.input_fmt.value}, output_fmt={self.output_fmt.value}, "
            f"component={self.component.value}"
        )


class QuantLinearMatVec(nn.Module):
    """
    nn.Linear replacement with separate matrix/vector IO formats.

    This is meant to support experiments where:
    - matrix_in_fmt:   quantization when loading activation + weight for matmul
    - matrix_out_fmt:  quantization of the matmul output *before* bias add
    - vector_in_fmt:   quantization when loading the bias vector
    - vector_out_fmt:  quantization of the final output after bias add

    Constraint:
        vector_in_fmt must equal matrix_out_fmt (bias add consumes matmul output).

    Optional:
        ulp_noise can be injected into the raw matmul output (before matrix_out_fmt quantization),
        with ULP defined in ulp_noise.ulp_fmt.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        matrix_in_fmt: QuantFormat,
        matrix_out_fmt: QuantFormat,
        vector_out_fmt: QuantFormat,
        component: Component,
        layer_name: str,
        tracker: Optional[StatsTracker] = None,
        ulp_noise: Optional[UlpNoiseConfig] = None,
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

        self.matrix_in_fmt = matrix_in_fmt
        self.matrix_out_fmt = matrix_out_fmt
        self.vector_in_fmt = matrix_out_fmt  # enforced by design constraint
        self.vector_out_fmt = vector_out_fmt

        self.component = component
        self.layer_name = layer_name
        self.tracker = tracker
        self.ulp_noise = ulp_noise

        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        x_f32 = x.float()
        w_f32 = self.weight.float()
        b_f32 = self.bias.float() if self.bias is not None else None

        # Matrix inputs quantization (activation + weight)
        x_q = quant(x_f32, self.matrix_in_fmt)
        w_q = quant(w_f32, self.matrix_in_fmt)

        # Matmul output (float32 arithmetic on quantized-grid values)
        y_mm = F.linear(x_q, w_q, None)

        # Inject ULP noise into matmul output (before matmul-output quantization)
        if self.ulp_noise is not None and self.ulp_noise.enabled():
            y_mm = inject_ulp_noise(
                y_mm,
                n_ulp=self.ulp_noise.n_ulp,
                mode=self.ulp_noise.mode,
                ulp_fmt=self.ulp_noise.ulp_fmt,
            )

        # Matrix output quantization (feeds bias add)
        y_mm_q = quant(y_mm, self.matrix_out_fmt)

        # Vector input quantization (bias)
        b_q = quant(b_f32, self.vector_in_fmt) if b_f32 is not None else None
        y_accum = y_mm_q if b_q is None else (y_mm_q + b_q)

        # Vector output quantization (final write)
        y_out = quant(y_accum, self.vector_out_fmt)

        if self.tracker is not None:
            with torch.no_grad():
                y_fp = F.linear(x_f32, w_f32, b_f32)
                self.tracker.record(
                    name=self.layer_name,
                    component=self.component,
                    y_fp=y_fp,
                    y_quant=y_out,
                )

        return y_out.to(original_dtype)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"mat_in={self.matrix_in_fmt.value}, mat_out={self.matrix_out_fmt.value}, "
            f"vec_in={self.vector_in_fmt.value}, vec_out={self.vector_out_fmt.value}, "
            f"component={self.component.value}"
        )
