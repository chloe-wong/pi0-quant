"""
quant_linear.py
---------------
QuantLinear: a drop-in replacement for nn.Linear that applies quantization
to matmul inputs (activation + weight) and matmul output separately.

RTL-backed variant
------------------
This version replaces F.linear(...) with AtlasLinearRTLFunction so the core
matmul/bias path uses the Python functional model of the RTL MXU.

Important limitations
---------------------
The current RTL model is hardcoded for:
  - input_fmt  = E4M3   (activations / weights / bias)
  - psumFmt    = BF16
  - outputFmt  = BF16 container, with optional runtime OutBF16 / OutE4M3

So this wrapper only supports configurations that match the RTL:
  - input_fmt  must be E4M3
  - output_fmt must be BF16 or E4M3

Forward pass semantics
----------------------
Given input_fmt A and output_fmt B:

    x_q      = quant(x,    A)            # activation loaded from memory in A
    W_q      = quant(W,    A)            # weight loaded from memory in A
    b_q      = quant(bias, A)            # bias loaded from memory in A
    y_accum  = rtl_linear(x_q, W_q, b_q) # tiled E4M3xE4M3 -> BF16/E4M3 via RTL model
    result   = quant(y_accum, B)         # optional final write-to-memory quantization

Note: the RTL model already performs its own output conversion when B is BF16 or
E4M3. The final quant(y_accum, B) is kept so this module preserves the same API
contract as the original QuantLinear and still matches the rest of the codebase.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .quant_types import QuantFormat, quant
from .stats_tracker import StatsTracker, Component
from .rtl_linear import AtlasLinearRTLFunction
from .fp_formats import OutputFmtSel


_SUPPORTED_E4M3_NAMES = {
    "e4m3",
    "fp8_e4m3",
    "ocp_e4m3",
}

_SUPPORTED_BF16_NAMES = {
    "bf16",
    "bfloat16",
}


class QuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with configurable input/output quantization.

    This RTL-backed version uses AtlasLinearRTLFunction instead of F.linear for the
    quantized path.

    Attributes:
        weight:       Shared with original linear (not copied).
        bias:         Shared with original linear (not copied).
        input_fmt:    Applied to both activation x and weight W before matmul.
        output_fmt:   Applied to final output writeback.
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
        *,
        vec_len: int = 32,
        num_lanes: int = 16,
        pipeline_depth: int = 1,
        scale_exp: int = 0,
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

        self.vec_len = vec_len
        self.num_lanes = num_lanes
        self.pipeline_depth = pipeline_depth
        self.scale_exp = scale_exp

        self._validate_rtl_formats()
        self.out_fmt_sel = self._to_output_fmt_sel(output_fmt)
        self.rtl_linear = AtlasLinearRTLFunction(
            vec_len=vec_len,
            num_lanes=num_lanes,
            pipeline_depth=pipeline_depth,
            out_fmt_sel=self.out_fmt_sel,
        )

    @staticmethod
    def _fmt_name(fmt: QuantFormat) -> str:
        return str(fmt.value).lower()

    def _validate_rtl_formats(self) -> None:
        in_name = self._fmt_name(self.input_fmt)
        out_name = self._fmt_name(self.output_fmt)

        if in_name not in _SUPPORTED_E4M3_NAMES:
            raise ValueError(
                "RTL QuantLinear only supports E4M3 input_fmt for now; "
                f"got {self.input_fmt.value!r}."
            )

        if out_name not in (_SUPPORTED_E4M3_NAMES | _SUPPORTED_BF16_NAMES):
            raise ValueError(
                "RTL QuantLinear only supports BF16 or E4M3 output_fmt for now; "
                f"got {self.output_fmt.value!r}."
            )

    @staticmethod
    def _to_output_fmt_sel(fmt: QuantFormat) -> OutputFmtSel:
        name = str(fmt.value).lower()
        if name in _SUPPORTED_BF16_NAMES:
            return OutputFmtSel.OutBF16
        if name in _SUPPORTED_E4M3_NAMES:
            return OutputFmtSel.OutE4M3
        raise ValueError(f"Unsupported RTL output format: {fmt.value!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        # All reference arithmetic starts from float32 tensors.
        x_f32 = x.float()
        w_f32 = self.weight.float()
        b_f32 = self.bias.float() if self.bias is not None else None

        # ── Load all inputs from memory in input_fmt ─────────────────────────
        x_q = quant(x_f32, self.input_fmt)
        w_q = quant(w_f32, self.input_fmt)
        b_q = quant(b_f32, self.input_fmt) if b_f32 is not None else None

        # ── Accumulate using the RTL functional model ────────────────────────
        y_accum = self.rtl_linear(
            x_q,
            w_q,
            b_q,
            scale_exp=self.scale_exp,
        )

        # ── Write result to output memory in output_fmt ──────────────────────
        # Kept for API compatibility with the original module.
        y_out = quant(y_accum, self.output_fmt)

        # ── RMSE: compare against unquantized full-precision reference ───────
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
            f"component={self.component.value}, vec_len={self.vec_len}, "
            f"num_lanes={self.num_lanes}, pipeline_depth={self.pipeline_depth}, "
            f"scale_exp={self.scale_exp}"
        )
