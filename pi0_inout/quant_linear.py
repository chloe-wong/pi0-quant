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
from .quant_types import QuantFormat, quant, quant_fp8_raw
from .hw_float_ops import quantize_bf16_to_e4m3
from .stats_tracker import StatsTracker, Component
from .rel_noise import inject_rel_noise, NoiseMode
from .matmul_io_store import MatmulIOStore


def _layer_tag_from_name(layer_name: str, component: Component) -> str:
    """
    Extract a short display tag from a full module path and Component.

    Examples:
        "paligemma...language_model.model.layers.7.self_attn.q_proj", LANGUAGE
            → "language.7"
        "paligemma...gemma_expert.model.layers.3.mlp.gate_proj", ACTION_EXPERT
            → "expert.3"
        "paligemma...vision_tower.vision_model.encoder.layers.0...", VISION
            → "vision"
        "action_in_proj.weight", ACTION_HEAD
            → "action_head"
    """
    if component == Component.VISION:
        return "vision"
    if component == Component.ACTION_HEAD:
        return "action_head"
    # For LANGUAGE and ACTION_EXPERT, extract the layer index from the path
    parts = layer_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            idx = parts[i + 1]
            if component == Component.LANGUAGE:
                return f"language.{idx}"
            if component == Component.ACTION_EXPERT:
                return f"expert.{idx}"
    return component.value  # fallback


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
        noise_mode: NoiseMode = NoiseMode.UNIFORM,
        functional_model=None,
        reference_store=None,
        matmul_io_store: Optional[MatmulIOStore] = None,
        ref_input_store=None,
        trace: bool = False,
        estimated_total: int = 0,
    ) -> None:
        super().__init__()

        self.weight = linear.weight
        self.bias   = linear.bias

        self.mx_input_fmt    = mx_input_fmt
        self.mx_output_fmt   = mx_output_fmt
        self.noise_injection = noise_injection
        self.noise_mode      = noise_mode
        self.functional_model = functional_model
        self.component       = component
        self.layer_name      = layer_name
        self.tracker         = tracker

        self.in_features     = linear.in_features
        self.out_features    = linear.out_features
        self.reference_store = reference_store
        self.matmul_io_store = matmul_io_store
        self.ref_input_store = ref_input_store
        self.trace           = trace
        self.estimated_total = estimated_total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias
        # Always operate in the weight dtype (e.g. bfloat16). Some activations
        # arrive as float32 (timestep embeddings, state inputs); casting here
        # matches what the reference F.linear does and ensures passthrough gives 0 RMSE.
        dtype = w.dtype
        x = x.to(dtype)

        # ── Substitute clean reference input (golden-data mode) ───────────────
        if self.ref_input_store is not None:
            x_clean = self.ref_input_store.get(self.layer_name)
            if x_clean is not None:
                x = x_clean.to(dtype=dtype, device=x.device)

        if self.functional_model is not None:
            # ── Po2 pre-normalization ─────────────────────────────────────────
            # Compute per-tensor po2 scales for x and w in BF16 precision
            # (matches hardware VPU rmax input width).  Guard with
            # _in_quant_guard so amax() inside quantize_bf16_to_e4m3 is not
            # re-intercepted by VectorQuantMode.
            _in_quant_guard.active = True
            try:
                x_fp8, sx = quantize_bf16_to_e4m3(x)
                w_fp8, sw = quantize_bf16_to_e4m3(w)
            finally:
                _in_quant_guard.active = False

            # Pre-normalized float32 inputs: exact E4M3 values at scale=1.
            # The FM will re-quantize internally via float_to_e4m3_bytes, which
            # is idempotent on already-quantized E4M3 values.
            x_norm = x_fp8.view(torch.float8_e4m3fn).to(torch.float32)
            w_norm = w_fp8.view(torch.float8_e4m3fn).to(torch.float32)

            # Run FM without bias — bias is in output domain, not FP8 domain.
            # IPT and similar models operate in float32 on CPU.
            y_unscaled = self.functional_model(x_norm, w_norm, None)

            # Rescale output by combined input scales, add bias in output domain.
            scale = 2.0 ** (sx + sw)
            y_out = (y_unscaled * scale).to(dtype=w.dtype, device=x.device)
            if b is not None:
                y_out = y_out + b
        else:
            # ── Passthrough ───────────────────────────────────────────────────
            y_out = F.linear(x, w, b)

        # ── Fetch clean reference for tracker cumulative RMSE ────────────────
        y_clean_ref = None
        if self.tracker is not None and self.reference_store is not None:
            y_clean_ref = self.reference_store.get(self.layer_name)

        # ── RMSE vs original model (native dtype, no fp32 upcast) ─────────────
        # Shield this block from VectorQuantMode interception: the arithmetic
        # inside tracker.record() (sub, pow, mean) must not be re-quantized.
        # Also cast x to w.dtype for the reference F.linear in case IPT (or any
        # functional model) returned float32 activations into a bfloat16 layer.
        if self.tracker is not None:
            _in_quant_guard.active = True
            try:
                with torch.no_grad():
                    y_ref = F.linear(x, w, b)
                    self.tracker.record(
                        name=self.layer_name,
                        component=self.component,
                        y_fp=y_ref,
                        y_quant=y_out,
                        y_clean_ref=y_clean_ref,
                    )
            finally:
                _in_quant_guard.active = False
            if self.trace:
                last = self.tracker.calls[-1]
                seq  = last["seq"]
                rmse = last["rmse"]
                total_str = f"/~{self.estimated_total}" if self.estimated_total else ""
                pct = (f"{100 * seq // self.estimated_total:3d}%"
                       if self.estimated_total else "   ")
                tag  = _layer_tag_from_name(self.layer_name, self.component)
                proj = self.layer_name.split(".")[-1]
                print(
                    f"[MX  | {seq:5d}{total_str} | {pct}]"
                    f"  {tag:<20}  {proj:<22}  in={tuple(x.shape)}"
                    f"  rmse={rmse:.6f}",
                    flush=True,
                )

        # ── Matmul I/O tensor capture ─────────────────────────────────────────
        # Always capture raw FP8 E4M3 bytes + scale for storage, regardless of
        # mx_input_fmt or functional_model.  Guard with _in_quant_guard: amax()
        # inside quant_fp8_raw() is in TARGET_OPS and would be re-intercepted by
        # VectorQuantMode otherwise.
        if self.matmul_io_store is not None:
            _in_quant_guard.active = True
            try:
                with torch.no_grad():
                    x_fp8, x_scale = quant_fp8_raw(x)
                    w_fp8, w_scale = quant_fp8_raw(w)
                    if b is not None:
                        b_fp8, b_scale = quant_fp8_raw(b)
                    else:
                        b_fp8, b_scale = None, None
                    self.matmul_io_store.record_patched(
                        name=self.layer_name,
                        x_fp8=x_fp8, x_fp8_scale=x_scale,
                        w_fp8=w_fp8, w_fp8_scale=w_scale,
                        b_fp8=b_fp8, b_fp8_scale=b_scale,
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
