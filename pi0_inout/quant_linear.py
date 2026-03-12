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

BFLOAT16 input_fmt and output_fmt are identity operations for bf16 models.
A BFLOAT16/BFLOAT16 run has exactly zero quantization RMSE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .quant_types import QuantFormat, quant
from .stats_tracker import StatsTracker, Component


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
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias   = linear.bias

        self.input_fmt  = input_fmt
        self.output_fmt = output_fmt
        self.component  = component
        self.layer_name = layer_name
        self.tracker    = tracker

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
        y_accum = F.linear(x_q, w_q, b_q)

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
