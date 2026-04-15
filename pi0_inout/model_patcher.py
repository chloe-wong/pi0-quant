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


# ---------------------------------------------------------------------------
# OpScope — which matmul types to apply the functional model / format flags to
# ---------------------------------------------------------------------------

class OpScope(str, Enum):
    LINEAR   = "linear"     # nn.Linear weight-activation matmuls (MLP + QKV projections)
    CONV2D   = "conv2d"     # Conv2d patch-embedding (SigLIP vision encoder)
    ATTENTION = "attention" # Attention score matmuls: Q@K^T and attn_weights@V


ALL_SCOPES: list[OpScope] = list(OpScope)


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
    op_scopes: Optional[set["OpScope"]] = None,
    reference_store=None,
    noise_injection: float = 0.0,
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
        op_scopes:                Which operation types to patch.  None means all scopes.
                                  Only OpScope.LINEAR_MLP is relevant here; CONV2D and
                                  ATTENTION are handled by their own patch functions.
                                  Pass {OpScope.LINEAR} to patch only linear layers.
        verbose:                  If True, print each replaced layer.

    Returns:
        The modified model (same object).
    """
    if op_scopes is None:
        op_scopes = {OpScope.LINEAR}

    if OpScope.LINEAR not in op_scopes:
        print("[patch_model] 'linear' not in op_scopes — skipping nn.Linear patching.")
        return model

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
            noise_injection=noise_injection,
            functional_model=fm,
            reference_store=reference_store,
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
# Conv2d format-flag patching
# ---------------------------------------------------------------------------

class QuantConv2d(nn.Conv2d):
    """
    Drop-in nn.Conv2d replacement with two modes:

    Format-flag mode (functional_model is None):
        Quantizes input and weight to mx_input_fmt, runs F.conv2d, then
        quantizes the output to mx_output_fmt.

    Functional model mode (functional_model is set):
        Converts the convolution to an equivalent GEMM via im2col (F.unfold),
        calls the functional model exactly as QuantLinear does, then reshapes
        the output back to the expected spatial layout.
        Only valid for groups=1 convolutions.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        mx_input_fmt: "QuantFormat",
        mx_output_fmt: "QuantFormat",
        component: "Component",
        layer_name: str,
        tracker: Optional["StatsTracker"] = None,
        functional_model=None,
        reference_store=None,
    ) -> None:
        super().__init__(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
        )
        self.weight = conv.weight
        if conv.bias is not None:
            self.bias = conv.bias
        self.mx_input_fmt    = mx_input_fmt
        self.mx_output_fmt   = mx_output_fmt
        self.component       = component
        self.layer_name      = layer_name
        self.tracker         = tracker
        self.functional_model = functional_model
        self.reference_store  = reference_store

        # Normalize string padding ('valid'/'same') to integer tuple so the
        # im2col path and output-shape formula always see numeric values.
        if isinstance(self.padding, str):
            if self.padding == 'valid':
                self.padding = (0, 0)
            else:
                raise ValueError(
                    f"QuantConv2d does not support padding='{self.padding}' "
                    f"(layer '{layer_name}'). Only integer or 'valid' padding is supported."
                )

        if functional_model is not None and conv.groups != 1:
            raise ValueError(
                f"QuantConv2d functional model path requires groups=1 "
                f"(got groups={conv.groups} for layer '{layer_name}')"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .quant_types import quant as _quant
        from ._dispatch_guards import _in_quant_guard
        w = self.weight
        b = self.bias

        if self.functional_model is not None:
            # ── Functional model path: im2col → GEMM → reshape ──────────────
            B, C_in, H, W = x.shape
            out_H = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
            out_W = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

            # F.unfold: [B, C_in*kH*kW, out_H*out_W]
            x_col = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)
            # → [B * out_H*out_W, C_in*kH*kW]
            x_flat = x_col.permute(0, 2, 1).reshape(B * out_H * out_W, -1).to(w.dtype)
            w_flat = w.reshape(self.out_channels, -1)  # [C_out, C_in*kH*kW]

            y_out = self.functional_model(x_flat, w_flat, b).to(dtype=w.dtype, device=x.device)
            # y_out: [B*out_H*out_W, C_out] → [B, C_out, out_H, out_W]
            y_out = y_out.reshape(B, out_H * out_W, self.out_channels).permute(0, 2, 1)
            y_out = y_out.reshape(B, self.out_channels, out_H, out_W)

            if self.tracker is not None:
                _in_quant_guard.active = True
                try:
                    with torch.no_grad():
                        y_fp = F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
                        y_clean_ref = (self.reference_store.get(self.layer_name)
                                       if self.reference_store is not None else None)
                        self.tracker.record(
                            name=self.layer_name,
                            component=self.component,
                            y_fp=y_fp,
                            y_quant=y_out,
                            y_clean_ref=y_clean_ref,
                        )
                finally:
                    _in_quant_guard.active = False

            return y_out

        else:
            # ── Format-flag path ─────────────────────────────────────────────
            # Cast x to weight dtype so passthrough (BF16/BF16) gives 0 RMSE.
            # Image inputs arrive as float32; conv weights are BF16.
            x = x.to(w.dtype)
            x_q = _quant(x.float(), self.mx_input_fmt).to(w.dtype)
            w_q = _quant(w.float(), self.mx_input_fmt).to(w.dtype)
            y   = F.conv2d(x_q, w_q, b, self.stride, self.padding, self.dilation, self.groups)
            y_q = _quant(y.float(), self.mx_output_fmt).to(w.dtype)

            if self.tracker is not None:
                with torch.no_grad():
                    y_fp = F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
                    y_clean_ref = (self.reference_store.get(self.layer_name)
                                   if self.reference_store is not None else None)
                    self.tracker.record(
                        name=self.layer_name,
                        component=self.component,
                        y_fp=y_fp,
                        y_quant=y_q,
                        y_clean_ref=y_clean_ref,
                    )

            return y_q


def patch_conv2d(
    model: nn.Module,
    mx_input_fmt: "QuantFormat",
    mx_output_fmt: "QuantFormat",
    tracker: Optional["StatsTracker"] = None,
    active_groups: Optional[set[QuantGroup]] = None,
    functional_model_factory=None,
    reference_store=None,
) -> nn.Module:
    """
    Replace every nn.Conv2d in `model` with a QuantConv2d in-place.

    If functional_model_factory is provided, each QuantConv2d will use the
    functional model (im2col → GEMM) instead of format-flag quantization.
    mx_input_fmt and mx_output_fmt are forced to BFLOAT16 in that case.

    Currently the only Conv2d in Pi0 is the SigLIP patch embedding in the
    vision tower (Component.VISION).
    """
    use_functional = functional_model_factory is not None
    _mx_in  = QuantFormat.BFLOAT16 if use_functional else mx_input_fmt
    _mx_out = QuantFormat.BFLOAT16 if use_functional else mx_output_fmt

    if active_groups is None:
        skip_components: set[Component] = set()
    else:
        skip_components = set(Component) - _active_components(active_groups)

    n_replaced = 0
    for name, module in list(_iter_named_conv2d(model)):
        component = _infer_component(name)
        if component in skip_components:
            continue

        fm = functional_model_factory(module.in_channels, module.out_channels) if use_functional else None

        quant_layer = QuantConv2d(
            conv=module,
            mx_input_fmt=_mx_in,
            mx_output_fmt=_mx_out,
            component=component,
            layer_name=name,
            tracker=tracker,
            functional_model=fm,
            reference_store=reference_store,
        )
        if tracker is not None:
            tracker.register(
                name=name,
                component=component,
                in_features=module.in_channels,
                out_features=module.out_channels,
            )
        _set_module(model, name, quant_layer)
        n_replaced += 1

    mode = (f"functional_model={functional_model_factory}" if use_functional
            else f"mx_input_fmt={_mx_in.value}  mx_output_fmt={_mx_out.value}")
    print(f"[patch_conv2d] Replaced {n_replaced} nn.Conv2d layers.  {mode}")
    return model


def unpatch_conv2d(model: nn.Module) -> nn.Module:
    """Reverse patch_conv2d: restore every QuantConv2d to a plain nn.Conv2d."""
    n_restored = 0
    for name, module in list(_iter_named_quant_conv2d(model)):
        plain = nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
        plain.weight = module.weight
        if module.bias is not None:
            plain.bias = module.bias
        _set_module(model, name, plain)
        n_restored += 1

    print(f"[unpatch_conv2d] Restored {n_restored} QuantConv2d → nn.Conv2d.")
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


def _iter_named_conv2d(model: nn.Module):
    """Yield (full_dotted_name, module) for every nn.Conv2d in the tree."""
    for name, module in model.named_modules():
        if type(module) is nn.Conv2d:
            yield name, module


def _iter_named_quant_conv2d(model: nn.Module):
    """Yield (full_dotted_name, module) for every QuantConv2d in the tree."""
    for name, module in model.named_modules():
        if isinstance(module, QuantConv2d):
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
    functional_model_factory=None,
    reference_store=None,
) -> list:
    """
    Patch F.scaled_dot_product_attention to simulate attention score matmuls
    (Q@K^T and attn_weights@V) with either format-flag quantization or a
    functional model (IPT/SA).

    The fused SDPA kernel is replaced with an explicit unfused implementation:
      1. Q, K, V quantized to mx_input_fmt  (format-flag) or passed through
         (functional model quantizes internally to E4M3)
      2. scores  = Q @ K^T * scale          (batched matmul or FM loop)
      3. scores += attn_mask / causal mask
      4. attn_weights = softmax(scores)      in BF16
      5. attn_weights quantized to FP8 (E4M3, po2 scaling) — always
      6. out = attn_weights @ V              (batched matmul or FM loop)
      7. out quantized to mx_output_fmt      (format-flag only)

    Functional model path loops over (batch, head) since FM interface is 2D.
    Format-flag path uses batched torch.matmul — no loop needed.

    Works by registering pre/post forward hooks on every self_attn module in
    the active groups so the patched SDPA knows which component is executing.

    Args:
        model:                    The Pi0Pytorch model.
        active_groups:            Groups whose SDPA calls should be quantized.
        mx_input_fmt:             Format for Q, K, V (format-flag path only).
        mx_output_fmt:            Format for the output (format-flag path only).
        tracker:                  Optional StatsTracker.
        functional_model_factory: If set, uses FM for Q@K^T and attn_weights@V.
                                  mx_input_fmt and mx_output_fmt are ignored.

    Returns:
        List of hook handles — pass to unpatch_attn_sdpa() to clean up.
    """
    import math as _math

    active_groups = set(active_groups)
    active_comps  = _active_components(active_groups)
    handles: list = []
    call_count    = [0]

    # Create FM instances once — factories ignore in/out features for attention
    fm_qk = functional_model_factory(0, 0) if functional_model_factory is not None else None
    fm_av = functional_model_factory(0, 0) if functional_model_factory is not None else None

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

    def _apply_mask(scores, attn_mask, is_causal):
        """Add float/bool attn_mask or build causal mask onto scores."""
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        elif is_causal:
            Sq, Skv = scores.shape[-2], scores.shape[-1]
            causal = torch.full(
                (Sq, Skv), float("-inf"), device=scores.device, dtype=scores.dtype
            )
            causal = torch.triu(causal, diagonal=1)
            scores = scores + causal
        return scores

    # Patch F.scaled_dot_product_attention globally with a group-aware version.
    def _quant_sdpa(
        query, key, value,
        attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
    ):
        current_comp = getattr(_attn_component_local, "component", None)
        if current_comp is None or current_comp not in active_comps:
            return _orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)

        head_dim = query.size(-1)
        _scale = scale if scale is not None else (1.0 / _math.sqrt(head_dim))

        if fm_qk is not None:
            # ── Functional model path: loop over heads (B=1 at inference) ───
            H, Sq, D = query.shape[1], query.shape[2], query.shape[3]
            Skv = key.size(2)
            Dv  = value.size(3)
            y_out = torch.zeros(1, H, Sq, Dv, dtype=torch.bfloat16, device=query.device)

            for h in range(H):
                    q_bh = query[0, h].to(torch.bfloat16)   # [Sq,  D]
                    k_bh = key[0, h].to(torch.bfloat16)     # [Skv, D]
                    v_bh = value[0, h].to(torch.bfloat16)   # [Skv, Dv]

                    # Step 4: Q @ K^T via FM  →  fm(Q, K, None) = Q @ K^T
                    scores = fm_qk(q_bh, k_bh, None).to(
                        dtype=torch.bfloat16, device=query.device
                    ) * _scale  # [Sq, Skv]

                    # Step 5: mask
                    mask_bh = None
                    if attn_mask is not None:
                        if attn_mask.dim() == 4:
                            h_idx = h if attn_mask.shape[1] > 1 else 0
                            mask_bh = attn_mask[0, h_idx]
                        else:
                            mask_bh = attn_mask
                    scores = _apply_mask(scores, mask_bh, is_causal)

                    # Step 6: softmax in BF16, quantize output to FP8
                    attn_w = torch.softmax(scores, dim=-1)
                    attn_w = quant(attn_w.float(), QuantFormat.FLOAT8_E4M3).to(torch.bfloat16)

                    # Step 7: attn_weights @ V via FM
                    # fm(x, w, None) = x @ w^T; pass v_bh.T so result = attn_w @ v_bh
                    out_bh = fm_av(attn_w, v_bh.T.contiguous(), None).to(
                        dtype=torch.bfloat16, device=query.device
                    )  # [Sq, Dv]
                    y_out[0, h] = out_bh

            y_out = y_out.to(query.dtype)

        else:
            # ── Format-flag path: batched torch.matmul ───────────────────────
            dtype = query.dtype
            q_q = quant(query.float(), mx_input_fmt).to(dtype)
            k_q = quant(key.float(),   mx_input_fmt).to(dtype)
            v_q = quant(value.float(), mx_input_fmt).to(dtype)

            # Step 4: Q @ K^T
            scores = torch.matmul(q_q, k_q.transpose(-2, -1)) * _scale

            # Step 5: mask
            scores = _apply_mask(scores, attn_mask, is_causal)

            # Step 6: softmax in BF16, quantize to FP8
            attn_w = torch.softmax(scores.to(torch.bfloat16), dim=-1)
            attn_w = quant(attn_w.float(), QuantFormat.FLOAT8_E4M3).to(torch.bfloat16)

            # Step 7: attn_weights @ V
            y_out = torch.matmul(attn_w, v_q.to(torch.bfloat16))
            y_out = quant(y_out.float(), mx_output_fmt).to(dtype)

        if tracker is not None:
            with torch.no_grad():
                out_fp = _orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)
                call_count[0] += 1
                y_clean_ref = None
                if reference_store is not None:
                    raw = reference_store.get("sdpa")
                    if raw is not None:
                        y_clean_ref = raw.to(device=y_out.device, dtype=y_out.dtype)
                tracker.record(
                    name=f"sdpa.{current_comp.value}.{call_count[0]}",
                    component=current_comp,
                    y_fp=out_fp,
                    y_quant=y_out,
                    y_clean_ref=y_clean_ref,
                )

        return y_out

    F.scaled_dot_product_attention = _quant_sdpa

    n_modules = len(handles) // 2
    mode = (f"functional_model={functional_model_factory}"
            if functional_model_factory else
            f"mx_input_fmt={mx_input_fmt.value}  mx_output_fmt={mx_output_fmt.value}")
    print(
        f"[patch_attn_sdpa] Hooked {n_modules} self_attn modules "
        f"for groups: {[g.value for g in active_groups]}  {mode}"
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


# ---------------------------------------------------------------------------
# Attention score patching via modeling_gemma.eager_attention_forward
# ---------------------------------------------------------------------------

# Saved at patch time; None means unpatch_attn_eager has nothing to do.
_orig_eager_attn_forward = None


def patch_attn_eager(
    model: nn.Module,
    active_groups: set[QuantGroup],
    mx_input_fmt: QuantFormat,
    mx_output_fmt: QuantFormat,
    tracker: Optional[StatsTracker] = None,
    functional_model_factory=None,
    reference_store=None,
) -> None:
    """
    Patch modeling_gemma.eager_attention_forward to simulate Gemma attention
    score matmuls (Q@K^T and attn_weights@V) with either format-flag
    quantization or a functional model (IPT/SA).

    Covers all three Gemma attention paths in Pi0:
      - language-only layers (paligemma.language_model)
      - action-expert-only layers (gemma_expert)
      - co-attention layers (both language and expert Q/K/V concatenated)

    All three call the same modeling_gemma.eager_attention_forward, which is
    the module-level function looked up at call time — so monkey-patching it
    intercepts all three without any thread-local tricks.

    The patched implementation:
      1. repeat_kv on K, V (done inside original; must be done explicitly here)
      2. Q @ K^T  (FM loop over heads, or batched torch.matmul with quant Q/K)
      3. Scale + attention_mask
      4. Softmax in BF16
      5. FP8 (E4M3, po2 scaling) quantization of attn_weights
      6. attn_weights @ V  (FM or batched matmul)
      7. (format-flag only) quantize output to mx_output_fmt

    Returns None (no hook handles — restored via unpatch_attn_eager).
    """
    global _orig_eager_attn_forward

    from transformers.models.gemma import modeling_gemma

    active_groups = set(active_groups)
    active_comps  = _active_components(active_groups)

    # Build id(self_attn_module) → Component for O(1) lookup in the patch.
    module_to_comp: dict[int, Component] = {}
    for name, module in model.named_modules():
        if name.split(".")[-1] != "self_attn":
            continue
        comp = _infer_component(name)
        if comp not in active_comps:
            continue
        module_to_comp[id(module)] = comp

    fm_qk = functional_model_factory(0, 0) if functional_model_factory is not None else None
    fm_av = functional_model_factory(0, 0) if functional_model_factory is not None else None

    call_count = [0]
    _orig = modeling_gemma.eager_attention_forward
    _orig_eager_attn_forward = _orig

    repeat_kv = modeling_gemma.repeat_kv

    def _quant_eager_attn(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        comp = module_to_comp.get(id(module))
        if comp is None:
            return _orig(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

        key_states   = repeat_kv(key,   module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        # query:        [B, H,  Sq,  D]
        # key_states:   [B, H,  Skv, D]
        # value_states: [B, H,  Skv, Dv]

        if fm_qk is not None:
            # ── Functional model path: loop over (batch, head) ───────────────
            B, H, Sq, D = query.shape
            Skv = key_states.size(2)
            Dv  = value_states.size(3)
            y_out = torch.zeros(B, H, Sq, Dv, dtype=torch.bfloat16, device=query.device)

            for b in range(B):
                # attention_mask is [B, 1, Sq, Skv] (broadcast over heads) or [B, H, Sq, Skv]
                mask_b = attention_mask[b] if attention_mask is not None else None
                for h in range(H):
                    q_bh = query[b, h].to(torch.bfloat16)         # [Sq, D]
                    k_bh = key_states[b, h].to(torch.bfloat16)    # [Skv, D]
                    v_bh = value_states[b, h].to(torch.bfloat16)  # [Skv, Dv]

                    # Q @ K^T via FM: fm(Q, K, None) = Q @ K^T   [Sq, Skv]
                    scores = fm_qk(q_bh, k_bh, None).to(
                        dtype=torch.bfloat16, device=query.device
                    ) * scaling

                    if mask_b is not None:
                        h_idx = h if mask_b.shape[0] > 1 else 0
                        scores = scores + mask_b[h_idx, :Sq, :Skv]

                    attn_w = torch.softmax(scores, dim=-1)
                    attn_w = quant(attn_w.float(), QuantFormat.FLOAT8_E4M3).to(torch.bfloat16)

                    # attn_w @ V via FM: fm(attn_w, V^T, None) = attn_w @ V  [Sq, Dv]
                    out_bh = fm_av(attn_w, v_bh.T.contiguous(), None).to(
                        dtype=torch.bfloat16, device=query.device
                    )
                    y_out[b, h] = out_bh

            y_out = y_out.to(query.dtype)

        else:
            # ── Format-flag path: batched torch.matmul ───────────────────────
            dtype = query.dtype
            q_q = quant(query.float(), mx_input_fmt).to(dtype)
            k_q = quant(key_states.float(), mx_input_fmt).to(dtype)
            v_q = quant(value_states.float(), mx_input_fmt).to(dtype)

            scores = torch.matmul(q_q, k_q.transpose(-2, -1)) * scaling

            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :, :key_states.shape[-2]]

            attn_w = torch.softmax(scores.to(torch.bfloat16), dim=-1)
            attn_w = quant(attn_w.float(), QuantFormat.FLOAT8_E4M3).to(torch.bfloat16)

            y_out = torch.matmul(attn_w, v_q.to(torch.bfloat16))
            y_out = quant(y_out.float(), mx_output_fmt).to(dtype)

        y_out_t = y_out.transpose(1, 2).contiguous()  # [B, Sq, H, Dv]

        if tracker is not None:
            with torch.no_grad():
                ref_out, _ = _orig(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
                call_count[0] += 1
                # Look up cumulative reference from clean unpatched pass
                y_clean_ref = None
                if reference_store is not None:
                    raw = reference_store.get("eager_attn")
                    if raw is not None:
                        y_clean_ref = raw.to(device=query.device, dtype=query.dtype)
                # Both ref_out and y_out_t are [B, Sq, H, Dv]
                tracker.record(
                    name=f"eager_attn.{comp.value}.{call_count[0]}",
                    component=comp,
                    y_fp=ref_out,
                    y_quant=y_out_t,
                    y_clean_ref=y_clean_ref,
                )

        return y_out_t, None

    modeling_gemma.eager_attention_forward = _quant_eager_attn

    mode = (f"functional_model={functional_model_factory}"
            if functional_model_factory else
            f"mx_input_fmt={mx_input_fmt.value}  mx_output_fmt={mx_output_fmt.value}")
    print(
        f"[patch_attn_eager] Patched eager_attention_forward for "
        f"{len(module_to_comp)} self_attn modules "
        f"in groups: {[g.value for g in active_groups]}  {mode}"
    )


def unpatch_attn_eager() -> None:
    """Restore the original modeling_gemma.eager_attention_forward."""
    global _orig_eager_attn_forward
    if _orig_eager_attn_forward is None:
        return
    from transformers.models.gemma import modeling_gemma
    modeling_gemma.eager_attention_forward = _orig_eager_attn_forward
    _orig_eager_attn_forward = None
    print("[unpatch_attn_eager] Restored original eager_attention_forward.")


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
