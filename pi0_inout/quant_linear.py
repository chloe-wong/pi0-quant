"""
quant_linear.py
---------------
QuantLinear: a drop-in replacement for nn.Linear that applies quantization
to matmul inputs (activation + weight) and matmul output separately.

Forward pass semantics
----------------------
Given mx_input_fmt A and mx_output_fmt B, and original model dtype D (e.g. bf16):

    x_q      = quant(x,    A).to(D)    # activation: quantize to A, cast back to D
    W_q      = quant(W,    A).to(D)    # weight:     quantize to A, cast back to D
    b_q      = quant(bias, A).to(D)    # bias:       quantize to A, cast back to D
    y_accum  = x_q @ W_q^T + b_q      # matmul in D — original model dtype, unchanged
    result   = quant(y_accum, B).to(D) # write result to memory in B, cast back to D

Quantization noise is baked in before the cast back to D: FP8-representable values
are a subset of BF16-representable values, so no noise is added or lost by the cast.
The matmul runs in the original model dtype, faithfully to the unpatched model.

RMSE reference is F.linear(x, w, b) in the original dtype — no fp32 upcast.
BFLOAT16/BFLOAT16 gives exactly zero RMSE since quant(x_bf16, BF16) is lossless.

If functional_model is provided, raw float32 tensors (x_f32, w_f32, b_f32) are
passed directly to it instead of applying format-flag quantization.
functional_model is mutually exclusive with non-BFLOAT16 mx_input_fmt/mx_output_fmt.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ._dispatch_guards import _in_quant_guard
from .quant_types import QuantFormat, quant
from .stats_tracker import StatsTracker, Component
from .rel_noise import inject_rel_noise


class QuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with configurable input/output quantization.

    Attributes:
        weight:           Shared with original linear (not copied).
        bias:             Shared with original linear (not copied).
        mx_input_fmt:     Applied to both activation x and weight W before matmul.
        mx_output_fmt:    Applied to matmul result and bias before addition.
        noise_injection:  Relative-error noise fraction applied to y_accum (0.0 = off).
        functional_model: Optional callable (x_f32, w_f32, b_f32) -> y_accum.
                          Mutually exclusive with non-BFLOAT16 mx_input_fmt/mx_output_fmt.
        component:        Architectural component tag (vision/language/action_*).
        layer_name:       Full dot-separated module path, used as stats key.
        tracker:          Optional StatsTracker for RMSE collection.
    """

    def __init__(
        self,
        linear: nn.Linear,
        mx_input_fmt: QuantFormat = QuantFormat.BFLOAT16,
        mx_output_fmt: QuantFormat = QuantFormat.BFLOAT16,
        component: Component = Component.UNKNOWN,
        layer_name: str = "",
        tracker: Optional[StatsTracker] = None,
        noise_injection: float = 0.0,
        functional_model=None,
    ) -> None:
        super().__init__()

        # Strict mutual exclusivity check
        if functional_model is not None:
            if mx_input_fmt != QuantFormat.BFLOAT16 or mx_output_fmt != QuantFormat.BFLOAT16:
                raise ValueError(
                    "functional_model is mutually exclusive with non-BFLOAT16 "
                    f"mx_input_fmt/mx_output_fmt. Got mx_input_fmt={mx_input_fmt}, "
                    f"mx_output_fmt={mx_output_fmt}. Set both to BFLOAT16 when using "
                    "functional_model."
                )

        self.weight = linear.weight
        self.bias   = linear.bias

        self.mx_input_fmt    = mx_input_fmt
        self.mx_output_fmt   = mx_output_fmt
        self.noise_injection = noise_injection
        self.functional_model = functional_model
        self.component       = component
        self.layer_name      = layer_name
        self.tracker         = tracker

        self.in_features  = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        w = self.weight
        b = self.bias

        if self.functional_model is not None:
            # Functional model handles quantization internally; pass tensors as-is.
            # Cast back to the layer's native dtype and device: IPT and similar
            # models operate in float32 on CPU and return CPU float32 tensors.
            y_out = self.functional_model(x, w, b).to(dtype=w.dtype, device=x.device)
        else:
            # ── Quantize inputs, cast back to original dtype ──────────────────
            x_q = quant(x.float(), self.mx_input_fmt).to(dtype)
            w_q = quant(w.float(), self.mx_input_fmt).to(dtype)
            b_q = quant(b.float(), self.mx_input_fmt).to(dtype) if b is not None else None

            # ── Matmul in original dtype — faithful to unpatched model ─────────
            y_accum = F.linear(x_q, w_q, b_q)

            # ── Noise injection ───────────────────────────────────────────────
            if self.noise_injection != 0.0:
                y_accum = inject_rel_noise(y_accum, rel_err=self.noise_injection)

            # ── Quantize output, cast back to original dtype ──────────────────
            y_out = quant(y_accum.float(), self.mx_output_fmt).to(dtype)

        # ── RMSE vs original model (native dtype, no fp32 upcast) ─────────────
        # Shield this block from VectorQuantMode interception: the arithmetic
        # inside tracker.record() (sub, pow, mean) must not be re-quantized.
        # Also cast x to w.dtype for the reference F.linear in case IPT (or any
        # functional model) returned float32 activations into a bfloat16 layer.
        if self.tracker is not None:
            _in_quant_guard.active = True
            try:
                with torch.no_grad():
                    x_ref = x.to(w.dtype) if x.dtype != w.dtype else x
                    y_ref = F.linear(x_ref, w, b)
                    self.tracker.record(
                        name=self.layer_name,
                        component=self.component,
                        y_fp=y_ref,
                        y_quant=y_out,
                    )
            finally:
                _in_quant_guard.active = False

        return y_out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"mx_input_fmt={self.mx_input_fmt.value}, mx_output_fmt={self.mx_output_fmt.value}, "
            f"component={self.component.value}"
        )
