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

   pass active_groups to restrict which components are quantized:
       patch_model(model, ..., active_groups={QuantGroup.VISION, QuantGroup.TRANSFORMER})

2. Attention score patching (patch_attn_sdpa / unpatch_attn_sdpa)
   ──────────────────────────────────────────────────────────────────
   The attention score matmuls (Q@K^T and attn_weights@V) are NOT nn.Linear
   layers.  HuggingFace computes them inside a single fused kernel:
   F.scaled_dot_product_attention(Q, K, V, ...).

   patch_attn_sdpa registers forward hooks on every self_attn module in the
   active groups so that F.scaled_dot_product_attention is only quantized
   when called from a module belonging to one of those groups.  Because
   LANGUAGE and ACTION_EXPERT are co-attention coupled (their layers are
   interleaved inside a single paligemma_with_expert forward pass), they are
   grouped together as QuantGroup.TRANSFORMER.

   Usage:
       active = {QuantGroup.TRANSFORMER, QuantGroup.ACTION_HEAD}
       patch_model(model, input_fmt=..., output_fmt=..., active_groups=active)
       handles = patch_attn_sdpa(model, active_groups=active, ...)
       # run inference ...
       unpatch_attn_sdpa(handles)

   Note: patch_attn_sdpa and QuantAttnContext (the legacy global context
   manager) patch the same F.scaled_dot_product_attention slot and must not
   be used simultaneously.

QuantGroup — hardware-realistic ablation boundaries
----------------------------------------------------
  VISION      SigLIP ViT — fully independent forward pass; cleanly separable
              on chip from the joint transformer block.
  TRANSFORMER PaliGemma language model + Gemma action expert — co-attention
              couples them at the layer level; on real silicon these would
              share a compute engine and precision decision.
  ACTION_HEAD Thin input/output projection MLPs (action_in_proj, action_out_proj,
              state_proj, time_mlp_*); runs after the joint pass, no attention.

Component tagging for Pi0Pytorch
---------------------------------
  PI0Pytorch
  ├── paligemma_with_expert
  │   ├── paligemma
  │   │   ├── vision_tower      → VISION       → QuantGroup.VISION
  │   │   └── language_model    → LANGUAGE     → QuantGroup.TRANSFORMER
  │   └── gemma_expert          → ACTION_EXPERT → QuantGroup.TRANSFORMER
  ├── action_in_proj            → ACTION_HEAD  → QuantGroup.ACTION_HEAD
  ├── action_out_proj           → ACTION_HEAD  → QuantGroup.ACTION_HEAD
  ├── state_proj                → ACTION_HEAD  → QuantGroup.ACTION_HEAD
  ├── action_time_mlp_{in,out}  → ACTION_HEAD  → QuantGroup.ACTION_HEAD
  └── time_mlp_{in,out}         → ACTION_HEAD  → QuantGroup.ACTION_HEAD

Tagging rules are checked in order; first match wins.
"""

from __future__ import annotations

import contextlib
import threading
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_linear import QuantLinear
from .quant_types import QuantFormat, quant
from .stats_tracker import Component, StatsTracker


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
# QuantGroup — hardware-realistic three-way ablation grouping
# ---------------------------------------------------------------------------

class QuantGroup(str, Enum):
    VISION        = "vision"         # SigLIP ViT — independent, cleanly separable
    TRANSFORMER   = "transformer"    # PaliGemma LM + action expert — co-attention coupled
    LANGUAGE      = "language"       # Gemma 2.6B LM only (software analysis; shares KV interface with action_expert)
    ACTION_EXPERT = "action_expert"  # Gemma 300M action expert only (software analysis)
    ACTION_HEAD   = "action_head"    # Thin MLPs at Pi0 root — no attention


# Maps each QuantGroup to the fine-grained Components it covers.
_GROUP_TO_COMPONENTS: dict[QuantGroup, frozenset[Component]] = {
    QuantGroup.VISION:        frozenset({Component.VISION}),
    QuantGroup.TRANSFORMER:   frozenset({Component.LANGUAGE, Component.ACTION_EXPERT}),
    QuantGroup.LANGUAGE:      frozenset({Component.LANGUAGE}),
    QuantGroup.ACTION_EXPERT: frozenset({Component.ACTION_EXPERT}),
    QuantGroup.ACTION_HEAD:   frozenset({Component.ACTION_HEAD}),
}

ALL_GROUPS: list[QuantGroup] = list(QuantGroup)


def _active_components(active_groups: set[QuantGroup]) -> set[Component]:
    """Return the set of Components covered by the given groups."""
    result: set[Component] = set()
    for g in active_groups:
        result |= _GROUP_TO_COMPONENTS[g]
    return result


# ---------------------------------------------------------------------------
# Main patching entry point
# ---------------------------------------------------------------------------

def patch_model(
    model: nn.Module,
    mx_input_fmt: QuantFormat,
    mx_output_fmt: QuantFormat,
    tracker: Optional[StatsTracker] = None,
    active_groups: Optional[set[QuantGroup]] = None,
    functional_model_factory=None,
    verbose: bool = False,
) -> nn.Module:
    """
    Replace every nn.Linear in `model` with a QuantLinear in-place.

    The model is modified in-place and also returned for convenience.

    Args:
        model:                    The Pi0Pytorch model (or any nn.Module).
        mx_input_fmt:             QuantFormat applied to activation + weight before the
                                  matmul.  Ignored when functional_model_factory is set.
        mx_output_fmt:            QuantFormat applied to the matmul output.
                                  Ignored when functional_model_factory is set.
        tracker:                  Optional StatsTracker.  If provided, each QuantLinear
                                  will compute RMSE against fp32 and report to the tracker.
        active_groups:            Which QuantGroups to quantize.  None means all groups.
                                  Pass a subset to restrict quantization, e.g.:
                                      active_groups={QuantGroup.TRANSFORMER}
        functional_model_factory: Optional callable (in_features, out_features) -> model.
                                  When provided, each QuantLinear receives a fresh
                                  functional_model instead of format-flag quantization.
                                  mx_input_fmt and mx_output_fmt are forced to BFLOAT16
                                  (QuantLinear requires this when functional_model is set).
                                  Use pi0_inout.functional_models.get_functional_model_factory
                                  to look up registered models (e.g. "ipt").
        verbose:                  If True, print each replaced layer.

    Returns:
        The modified model (same object).
    """
    use_functional = functional_model_factory is not None
    # QuantLinear requires BF16 fmts when functional_model is set
    _mx_in  = QuantFormat.BFLOAT16 if use_functional else mx_input_fmt
    _mx_out = QuantFormat.BFLOAT16 if use_functional else mx_output_fmt

    if active_groups is None:
        skip_components: set[Component] = set()
    else:
        skip_components = set(Component) - _active_components(active_groups)
    n_replaced = 0
    n_skipped  = 0

    for name, module in list(_iter_named_linear(model)):
        component = _infer_component(name)

        if component in skip_components:
            n_skipped += 1
            if verbose:
                print(f"  SKIP  {name}  [{component.value}]")
            continue

        # Build per-layer functional model if factory provided
        fm = (functional_model_factory(module.in_features, module.out_features)
              if use_functional else None)

        # Build the QuantLinear replacement
        quant_layer = QuantLinear(
            linear=module,
            mx_input_fmt=_mx_in,
            mx_output_fmt=_mx_out,
            component=component,
            layer_name=name,
            tracker=tracker,
            functional_model=fm,
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
        mode = (f"functional_model={functional_model_factory}"
                if use_functional
                else f"mx_input_fmt={mx_input_fmt.value}  mx_output_fmt={mx_output_fmt.value}")
        print(
            f"[patch_model] Replaced {n_replaced} nn.Linear layers "
            f"(skipped {n_skipped}).  {mode}"
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
        if isinstance(module, QuantLinear):
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
        if isinstance(module, (nn.Linear, QuantLinear)):
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
        if isinstance(module, (nn.Linear, QuantLinear)):
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

# Thread-local used by patch_attn_sdpa hooks to signal which Component's
# self_attn is currently executing, so the patched SDPA can gate by group.
_attn_component_local: threading.local = threading.local()


def patch_attn_sdpa(
    model: nn.Module,
    active_groups: set[QuantGroup],
    mx_input_fmt: QuantFormat,
    mx_output_fmt: QuantFormat,
    tracker: Optional[StatsTracker] = None,
) -> list:
    """
    Patch F.scaled_dot_product_attention to quantize only SDPA calls that
    originate from self_attn modules belonging to active_groups.

    Works by registering pre/post forward hooks on every self_attn module
    in the active groups' subtrees.  The pre-hook sets a thread-local flag
    to the module's Component; the patched SDPA reads it and quantizes only
    when that component belongs to an active group.  The post-hook clears
    the flag.

    This correctly handles LANGUAGE + ACTION_EXPERT co-attention: because
    both belong to QuantGroup.TRANSFORMER, their interleaved SDPA calls are
    all captured, even though their layers run inside a single joint forward.

    Args:
        model:          The (already patch_model'd) Pi0Pytorch model.
        active_groups:  Groups whose SDPA calls should be quantized.
        mx_input_fmt:   Format for Q, K, V entering SDPA.
        mx_output_fmt:  Format for the attended output.
        tracker:        Optional StatsTracker.

    Returns:
        List of hook handles — pass to unpatch_attn_sdpa() to clean up.

    Note: Do not use simultaneously with QuantAttnContext; both patch the
    same F.scaled_dot_product_attention slot.
    """
    active_groups = set(active_groups)
    active_comps  = _active_components(active_groups)
    handles: list = []
    call_count    = [0]

    # Register pre/post hooks on every self_attn module in the active subtrees.
    for name, module in model.named_modules():
        if name.split(".")[-1] != "self_attn":
            continue
        comp = _infer_component(name)
        if comp not in active_comps:
            continue

        def _make_hooks(c: Component):
            def pre(mod, inp):
                _attn_component_local.component = c
            def post(mod, inp, out):
                _attn_component_local.component = None
            return pre, post

        pre_h, post_h = _make_hooks(comp)
        handles.append(module.register_forward_pre_hook(pre_h))
        handles.append(module.register_forward_hook(post_h))

    # Patch F.scaled_dot_product_attention globally with a group-aware version.
    def _quant_sdpa(
        query, key, value,
        attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
    ):
        current_comp = getattr(_attn_component_local, "component", None)
        if current_comp is None or current_comp not in active_comps:
            return _orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)

        q_q = quant(query.float(), mx_input_fmt).to(query.dtype)
        k_q = quant(key.float(),   mx_input_fmt).to(key.dtype)
        v_q = quant(value.float(), mx_input_fmt).to(value.dtype)
        out = _orig_sdpa(q_q, k_q, v_q, attn_mask, dropout_p, is_causal, scale=scale)
        out_q = quant(out.float(), mx_output_fmt).to(out.dtype)

        if tracker is not None:
            with torch.no_grad():
                out_fp = _orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)
                call_count[0] += 1
                tracker.record(
                    name=f"sdpa.{current_comp.value}.{call_count[0]}",
                    component=current_comp,
                    y_fp=out_fp,
                    y_quant=out_q,
                )

        return out_q

    F.scaled_dot_product_attention = _quant_sdpa

    n_modules = len(handles) // 2
    print(
        f"[patch_attn_sdpa] Hooked {n_modules} self_attn modules "
        f"for groups: {[g.value for g in active_groups]}  "
        f"mx_input_fmt={mx_input_fmt.value}  mx_output_fmt={mx_output_fmt.value}"
    )
    return handles


def unpatch_attn_sdpa(handles: list) -> None:
    """
    Remove hooks registered by patch_attn_sdpa and restore the original
    F.scaled_dot_product_attention.
    """
    for h in handles:
        h.remove()
    F.scaled_dot_product_attention = _orig_sdpa
    print(f"[unpatch_attn_sdpa] Removed {len(handles)} hooks, restored original SDPA.")


class QuantAttnContext:
    """
    Legacy context manager: quantizes ALL F.scaled_dot_product_attention
    calls globally while active (no group filtering).

    Prefer patch_attn_sdpa / unpatch_attn_sdpa for new code, which
    restricts quantization to specific QuantGroups via module hooks.

    Usage:
        with QuantAttnContext(QuantFormat.FLOAT8_E4M3, QuantFormat.FLOAT16):
            actions = model.sample_actions(obs)
    """

    def __init__(
        self,
        mx_input_fmt: QuantFormat,
        mx_output_fmt: QuantFormat,
        tracker: Optional[StatsTracker] = None,
    ) -> None:
        self.mx_input_fmt   = mx_input_fmt
        self.mx_output_fmt  = mx_output_fmt
        self.tracker        = tracker
        self._call_count    = 0

    def __enter__(self) -> "QuantAttnContext":
        mx_input_fmt  = self.mx_input_fmt
        mx_output_fmt = self.mx_output_fmt
        tracker       = self.tracker
        ctx           = self

        def _quant_sdpa(
            query, key, value,
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
        ):
            q_q = quant(query.float(), mx_input_fmt).to(query.dtype)
            k_q = quant(key.float(),   mx_input_fmt).to(key.dtype)
            v_q = quant(value.float(), mx_input_fmt).to(value.dtype)
            out = _orig_sdpa(q_q, k_q, v_q, attn_mask, dropout_p, is_causal, scale=scale)
            out_q = quant(out.float(), mx_output_fmt).to(out.dtype)

            if tracker is not None:
                with torch.no_grad():
                    out_fp = _orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)
                    ctx._call_count += 1
                    tracker.record(
                        name=f"sdpa.{ctx._call_count}",
                        component=Component.UNKNOWN,
                        y_fp=out_fp,
                        y_quant=out_q,
                    )
            return out_q

        F.scaled_dot_product_attention = _quant_sdpa
        return self

    def __exit__(self, *_) -> None:
        F.scaled_dot_product_attention = _orig_sdpa
