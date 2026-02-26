"""
model_patcher.py
----------------
Two patching systems for Pi0Pytorch:

1. nn.Linear replacement (patch_model / unpatch_model)
   ─────────────────────────────────────────────────────
   Walks the module tree and replaces every nn.Linear with QuantLinear.
   Covers all weight-activation matmuls:
     • Q, K, V, O projection in every attention layer (vision, language, action)
     • FFN gate_proj / up_proj / down_proj in every transformer layer
     • Action head projections (action_in_proj, action_out_proj, etc.)

2. Attention score patching (QuantAttnContext)
   ──────────────────────────────────────────────
   The attention score matmuls (Q@K^T and attn_weights@V) are NOT nn.Linear
   layers.  HuggingFace computes them inside a single fused kernel:
   F.scaled_dot_product_attention(Q, K, V, ...).

   QuantAttnContext is a context manager that temporarily replaces
   F.scaled_dot_product_attention with a quantized version.  While active,
   every SDPA call quantizes Q, K, V to input_fmt before calling the original
   kernel, and quantizes the output to output_fmt.

   Because SDPA fuses Q@K^T and attn_weights@V into one call, we cannot
   separately quantize the intermediate attention score matrix.  We can only
   quantize the inputs (Q, K, V) and the final output.  This is still
   meaningful — it tests what happens when the data flowing into the attention
   score computation is at reduced precision.

   Usage:
       with QuantAttnContext(QuantFormat.FLOAT8_E4M3, QuantFormat.FLOAT16,
                             tracker=tracker):
           actions = model.sample_actions(obs)

Component tagging for Pi0Pytorch
---------------------------------
  PI0Pytorch
  ├── paligemma_with_expert
  │   ├── paligemma
  │   │   ├── vision_tower      → VISION
  │   │   └── language_model    → LANGUAGE
  │   └── gemma_expert          → ACTION_EXPERT
  ├── action_in_proj            → ACTION_HEAD
  ├── action_out_proj           → ACTION_HEAD
  ├── state_proj                → ACTION_HEAD
  ├── action_time_mlp_{in,out}  → ACTION_HEAD
  └── time_mlp_{in,out}         → ACTION_HEAD

Tagging rules are checked in order; first match wins.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_linear import QuantLinear, QuantLinearMatVec
from .quant_types import QuantFormat, quant
from .stats_tracker import Component, StatsTracker
from .ulp_noise import UlpNoiseConfig


# ---------------------------------------------------------------------------
# Component tagging
# ---------------------------------------------------------------------------

# (path_substring, Component) — checked in order; first match wins.
_COMPONENT_RULES: list[tuple[str, Component]] = [
    # Action-head projections live directly on the Pi0Pytorch root
    ("action_in_proj",        Component.ACTION_HEAD),
    ("action_out_proj",       Component.ACTION_HEAD),
    ("action_time_mlp_in",    Component.ACTION_HEAD),
    ("action_time_mlp_out",   Component.ACTION_HEAD),
    ("state_proj",            Component.ACTION_HEAD),
    ("time_mlp_in",           Component.ACTION_HEAD),
    ("time_mlp_out",          Component.ACTION_HEAD),
    # Gemma action expert (separate transformer from language model)
    ("gemma_expert",          Component.ACTION_EXPERT),
    # Vision tower (SigLIP ViT)
    ("vision_tower",          Component.VISION),
    # Language model (Gemma inside PaliGemma)
    ("language_model",        Component.LANGUAGE),
    # Fallback for any Linear inside paligemma that doesn't match above
    ("paligemma",             Component.LANGUAGE),
]


def _infer_component(path: str) -> Component:
    """
    Determine the architectural component for a layer given its full module path.

    Args:
        path: Dot-separated module path, e.g.
              "paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj"

    Returns:
        The matching Component enum value.
    """
    for substr, component in _COMPONENT_RULES:
        if substr in path:
            return component
    return Component.UNKNOWN


# ---------------------------------------------------------------------------
# Main patching entry point
# ---------------------------------------------------------------------------

def patch_model(
    model: nn.Module,
    input_fmt: QuantFormat,
    output_fmt: QuantFormat,
    tracker: Optional[StatsTracker] = None,
    ulp_noise: Optional[UlpNoiseConfig] = None,
    skip_components: Optional[set[Component]] = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Replace every nn.Linear in `model` with a QuantLinear in-place.

    The model is modified in-place and also returned for convenience.

    Args:
        model:           The Pi0Pytorch model (or any nn.Module).
        input_fmt:       QuantFormat applied to activation + weight before the matmul.
        output_fmt:      QuantFormat applied to the matmul output.
        tracker:         Optional StatsTracker.  If provided, each QuantLinear
                         will compute RMSE against fp32 and report to the tracker.
        skip_components: Set of Components whose layers should NOT be quantized
                         (useful for ablations, e.g. skipping vision).
        verbose:         If True, print each replaced layer.

    Returns:
        The modified model (same object).
    """
    skip_components = skip_components or set()
    n_replaced = 0
    n_skipped  = 0

    for name, module in list(_iter_named_linear(model)):
        component = _infer_component(name)

        if component in skip_components:
            n_skipped += 1
            if verbose:
                print(f"  SKIP  {name}  [{component.value}]")
            continue

        # Build the QuantLinear replacement
        quant_layer = QuantLinear(
            linear=module,
            input_fmt=input_fmt,
            output_fmt=output_fmt,
            component=component,
            layer_name=name,
            tracker=tracker,
            ulp_noise=ulp_noise,
        )

        # Pre-register with tracker so summary() works even if some layers
        # never fire (e.g., conditional code paths)
        if tracker is not None:
            tracker.register(
                name=name,
                component=component,
                in_features=module.in_features,
                out_features=module.out_features,
            )

        # Replace the module in the parent
        _set_module(model, name, quant_layer)

        n_replaced += 1
        if verbose:
            print(
                f"  QUANT {name}  [{component.value}]  "
                f"in={module.in_features} out={module.out_features}"
            )

    if verbose or True:  # always print summary
        print(
            f"[patch_model] Replaced {n_replaced} nn.Linear layers "
            f"(skipped {n_skipped}).  "
            f"input_fmt={input_fmt.value}  output_fmt={output_fmt.value}"
        )

    return model


def patch_model_matvec(
    model: nn.Module,
    *,
    matrix_in_fmt: QuantFormat,
    matrix_out_fmt: QuantFormat,
    vector_out_fmt: QuantFormat,
    tracker: Optional[StatsTracker] = None,
    ulp_noise: Optional[UlpNoiseConfig] = None,
    skip_components: Optional[set[Component]] = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Replace every nn.Linear in `model` with QuantLinearMatVec in-place.

    Constraint enforced by design:
        vector_in_fmt == matrix_out_fmt
    """
    skip_components = skip_components or set()
    n_replaced = 0
    n_skipped = 0

    for name, module in list(_iter_named_linear(model)):
        component = _infer_component(name)
        if component in skip_components:
            n_skipped += 1
            if verbose:
                print(f"  SKIP  {name}  [{component.value}]")
            continue

        quant_layer = QuantLinearMatVec(
            linear=module,
            matrix_in_fmt=matrix_in_fmt,
            matrix_out_fmt=matrix_out_fmt,
            vector_out_fmt=vector_out_fmt,
            component=component,
            layer_name=name,
            tracker=tracker,
            ulp_noise=ulp_noise,
        )

        if tracker is not None:
            tracker.register(
                name=name,
                component=component,
                in_features=module.in_features,
                out_features=module.out_features,
            )

        _set_module(model, name, quant_layer)
        n_replaced += 1
        if verbose:
            print(
                f"  QUANT {name}  [{component.value}]  "
                f"in={module.in_features} out={module.out_features}"
            )

    print(
        f"[patch_model_matvec] Replaced {n_replaced} nn.Linear layers "
        f"(skipped {n_skipped}).  "
        f"mat_in={matrix_in_fmt.value}  mat_out={matrix_out_fmt.value}  vec_out={vector_out_fmt.value}"
    )
    return model


def unpatch_model(model: nn.Module) -> nn.Module:
    """
    Reverse patch_model: replace every QuantLinear back to a plain nn.Linear.

    Useful when you want to reuse the same model object for multiple
    quantization sweeps without reloading weights.
    """
    n_restored = 0
    for name, module in list(_iter_named_quant_linear(model)):
        # Reconstruct a plain nn.Linear with the same parameters
        plain = nn.Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
        )
        plain.weight = module.weight
        plain.bias   = module.bias
        _set_module(model, name, plain)
        n_restored += 1

    print(f"[unpatch_model] Restored {n_restored} QuantLinear → nn.Linear.")
    return model


# ---------------------------------------------------------------------------
# Helpers: module tree traversal
# ---------------------------------------------------------------------------

def _iter_named_linear(model: nn.Module):
    """Yield (full_dotted_name, module) for every nn.Linear in the tree."""
    for name, module in model.named_modules():
        if type(module) is nn.Linear:  # exact type, not subclass
            yield name, module


def _iter_named_quant_linear(model: nn.Module):
    """Yield (full_dotted_name, module) for every QuantLinear in the tree."""
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantLinearMatVec)):
            yield name, module


def _set_module(root: nn.Module, name: str, new_module: nn.Module) -> None:
    """
    Set the sub-module at the given dot-separated path to `new_module`.

    Example:
        _set_module(model, "paligemma.language_model.layers.0.self_attn.q_proj", quant)
    """
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# ---------------------------------------------------------------------------
# Inspection utilities
# ---------------------------------------------------------------------------

def count_layers(model: nn.Module) -> dict[str, int]:
    """
    Return a dict mapping component name → number of nn.Linear layers
    (pre-patch) or QuantLinear layers (post-patch).
    """
    counts: dict[str, int] = {c.value: 0 for c in Component}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, QuantLinear, QuantLinearMatVec)):
            comp = _infer_component(name)
            counts[comp.value] += 1
    return {k: v for k, v in counts.items() if v > 0}


def list_linear_layers(model: nn.Module) -> list[dict]:
    """
    Return metadata for every linear layer in the model.
    Useful for inspecting before patching.
    """
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, QuantLinear, QuantLinearMatVec)):
            comp = _infer_component(name)
            rows.append({
                "name":        name,
                "component":   comp.value,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "type":        type(module).__name__,
            })
    return rows


# ---------------------------------------------------------------------------
# Attention score patching via F.scaled_dot_product_attention
# ---------------------------------------------------------------------------

# The original unpatched function, saved at import time.
_orig_sdpa = F.scaled_dot_product_attention


class QuantAttnContext:
    """
    Context manager that quantizes the inputs and output of every
    F.scaled_dot_product_attention call while active.

    HuggingFace Gemma and SigLIP route all attention score computation
    through this single function, which internally computes:
        scores    = Q @ K^T / sqrt(d_head)   [+ optional mask]
        weights   = softmax(scores)
        output    = weights @ V

    These three steps are fused — we cannot quantize the intermediate
    attention score matrix separately.  What we CAN do:
        • quantize Q, K, V to input_fmt before the kernel runs
        • quantize the attended output to output_fmt after

    This covers the format effect on both inner matmuls from the outside.

    Args:
        input_fmt:  Format for Q, K, V tensors entering SDPA.
        output_fmt: Format for the attended output leaving SDPA.
        tracker:    Optional StatsTracker.  If given, records RMSE between
                    the full-precision and quantized SDPA outputs, keyed as
                    "sdpa.<component>" where component is inferred from a
                    thread-local set by the enclosing module hooks (advanced).
                    In simple usage, all SDPA calls are keyed as "sdpa".

    Usage:
        # Patch nn.Linear layers permanently with patch_model(), then wrap
        # each forward pass in QuantAttnContext for attention score quantization.
        patch_model(model, input_fmt=..., output_fmt=..., tracker=tracker)

        with QuantAttnContext(QuantFormat.FLOAT8_E4M3, QuantFormat.FLOAT16,
                              tracker=tracker):
            actions = model.sample_actions(obs)

        report = tracker.summary()
    """

    def __init__(
        self,
        input_fmt: QuantFormat,
        output_fmt: QuantFormat,
        tracker: Optional[StatsTracker] = None,
    ) -> None:
        self.input_fmt  = input_fmt
        self.output_fmt = output_fmt
        self.tracker    = tracker
        self._call_count = 0  # used as a tie-breaker key in stats

    def __enter__(self) -> "QuantAttnContext":
        input_fmt  = self.input_fmt
        output_fmt = self.output_fmt
        tracker    = self.tracker
        ctx        = self  # closure reference

        def _quant_sdpa(
            query, key, value,
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
        ):
            # Quantize Q, K, V to input_fmt
            q_q = quant(query.float(), input_fmt).to(query.dtype)
            k_q = quant(key.float(),   input_fmt).to(key.dtype)
            v_q = quant(value.float(), input_fmt).to(value.dtype)

            # Run the original SDPA with quantized inputs
            out = _orig_sdpa(q_q, k_q, v_q, attn_mask, dropout_p, is_causal, scale)

            # Quantize the output to output_fmt
            out_q = quant(out.float(), output_fmt).to(out.dtype)

            # RMSE tracking (optional)
            if tracker is not None:
                with torch.no_grad():
                    out_fp = _orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale)
                    ctx._call_count += 1
                    tracker.record(
                        name=f"sdpa.{ctx._call_count}",
                        component=Component.UNKNOWN,  # can't infer without module hooks
                        y_fp=out_fp,
                        y_quant=out_q,
                    )

            return out_q

        F.scaled_dot_product_attention = _quant_sdpa
        return self

    def __exit__(self, *_) -> None:
        F.scaled_dot_product_attention = _orig_sdpa
