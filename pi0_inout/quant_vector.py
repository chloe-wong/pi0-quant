"""
quant_vector.py
---------------
Vector-engine operation quantization for Pi0Pytorch.

Every listed vector op in every layer of the model is intercepted via
TorchDispatchMode.  With --vec-functional-model active, ops route through
VectorRTLFunctions (lane-box RTL-accurate simulation).  Without it, ops
pass through unchanged (passthrough mode).

Covered operations
------------------
  Add / Sub             (Tensor and Scalar variants)
  Multiply              (Tensor and Scalar variants)
  Div / Reciprocal
  Square / Cube         (via pow.Tensor_Scalar)
  Sqrt / Rsqrt
  Sin / Cos / Tanh / Log2 / Exp / Exp2 / Neg
  AMax / Sum reductions
  Mean reduction        (for GemmaRMSNorm)
  nn.LayerNorm          (fused aten.native_layer_norm)
  F.softmax             (fused aten._softmax)
  GELU tanh approx      (fused aten.gelu)
  SiLU                  (fused aten.silu)

Usage
-----
    from pi0_inout.quant_vector import patch_vector_ops, unpatch_vector_ops
    from funct_models_vector.vector_rtl_forward import VectorRTLFunctions

    vrf = VectorRTLFunctions(num_lanes=16)
    vec_h, vec_ctx, hook_fires = patch_vector_ops(model, active_groups=active, functional_model=vrf)
    with vec_ctx:
        actions = model.sample_actions(...)
    unpatch_vector_ops(vec_h)

Layer-tag attribution
---------------------
Forward hooks on transformer layer boundaries push (Component, layer_tag) pairs
onto a thread-local stack so __torch_dispatch__ can label each op.  layer_tag
examples: "vision", "language.7", "expert.3", "action_head", "unattributed".

All ops in TARGET_OPS are routed through VRF unconditionally — there is no
attribution gate.  Ops that fire outside any hooked module (final RMSNorm,
Euler step) are labelled "unattributed" and still go through VRF.

Re-entrant guard
----------------
A thread-local '_in_quant_guard' suppresses re-entrant dispatch so tensor
operations inside VPU decompositions are not themselves re-intercepted.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from collections import defaultdict

from ._dispatch_guards import _in_quant_guard
from .model_patcher import QuantGroup, _GROUP_TO_COMPONENTS
from .stats_tracker import Component, StatsTracker
from .vector_io_store import VectorIOStore


# ---------------------------------------------------------------------------
# Action-head module names (top-level — depth 0, no dots in name)
# ---------------------------------------------------------------------------

_ACTION_HEAD_MODULES: frozenset = frozenset({
    "action_in_proj", "action_out_proj", "action_time_mlp_in",
    "action_time_mlp_out", "state_proj", "time_mlp_in", "time_mlp_out",
})


# ---------------------------------------------------------------------------
# Layer-tag detection for vec hooks (labeling only, not gating)
# ---------------------------------------------------------------------------

def _vec_layer_tag_for_module(name: str) -> Optional[tuple[Component, str]]:
    """
    Return (Component, layer_tag) for module `name`, or None if not a hook site.

    layer_tag format:
      "vision"        — SigLIP vision encoder (hooked as a container)
      "language.N"    — language model decoder layer N
      "expert.N"      — action-expert decoder layer N
      "action_head"   — top-level action projection modules

    Rules (checked in priority order):
      1. vision_tower container — last segment == "vision_tower"
      2. language decoder layers — ...language_model.layers.<int>
      3. action-expert decoder layers — ...gemma_expert.*.layers.<int>
      4. action-head top-level linears — depth-0 names in _ACTION_HEAD_MODULES
    """
    parts = name.split(".")
    last  = parts[-1]

    if last == "vision_tower":
        return (Component.VISION, "vision")

    if (len(parts) >= 2 and last.isdigit()
            and parts[-2] == "layers" and "language_model" in name
            and "gemma_expert" not in name):
        return (Component.LANGUAGE, f"language.{last}")

    if (len(parts) >= 2 and last.isdigit()
            and parts[-2] == "layers" and "gemma_expert" in name):
        return (Component.ACTION_EXPERT, f"expert.{last}")

    if "." not in name and name in _ACTION_HEAD_MODULES:
        return (Component.ACTION_HEAD, "action_head")

    return None


from .vpu_decomp import (
    _VEC_FM_DISPATCH,
    _vpu_gelu_tanh,
    _vpu_layer_norm,
    _vpu_rmax_all,
    _vpu_rsqrt,
    _vpu_rsum_all,
    _vpu_silu,
    _vpu_softmax,
)


# ---------------------------------------------------------------------------
# Target ops: vector ops only (matmuls excluded — handled by QuantLinear)
# ---------------------------------------------------------------------------

TARGET_OPS: frozenset = frozenset({
    # ── Add / Sub ─────────────────────────────────────────────────────────────
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
    # ── Multiply ──────────────────────────────────────────────────────────────
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    # ── Square / Cube  (pow.Tensor_Scalar covers x**2, x**3, etc.) ───────────
    torch.ops.aten.pow.Tensor_Scalar,
    # ── Div / Reciprocal ──────────────────────────────────────────────────────
    torch.ops.aten.div.Tensor,
    torch.ops.aten.reciprocal.default,
    # ── Sqrt / Rsqrt ─────────────────────────────────────────────────────────
    torch.ops.aten.sqrt.default,
    torch.ops.aten.rsqrt.default,
    # ── Trig / Exp / Log ─────────────────────────────────────────────────────
    torch.ops.aten.sin.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.log2.default,
    torch.ops.aten.exp.default,
    torch.ops.aten.exp2.default,
    torch.ops.aten.neg.default,
    # ── Reductions ────────────────────────────────────────────────────────────
    torch.ops.aten.amax.default,
    torch.ops.aten.sum.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.mean.dim,
    # ── Fused composite ops (decomposed into VPU primitive sequences) ─────────
    torch.ops.aten.native_layer_norm.default,   # SigLIP nn.LayerNorm
    torch.ops.aten._softmax.default,            # all F.softmax / torch.softmax
    torch.ops.aten.gelu.default,                # Gemma / SigLIP MLP activation
    torch.ops.aten.silu.default,                # Pi0 time MLP
})


# ---------------------------------------------------------------------------
# Thread-local layer-tag stack
# ---------------------------------------------------------------------------

_vec_layer_stack: threading.local = threading.local()


def _push_layer(comp: Component, tag: str) -> None:
    if not hasattr(_vec_layer_stack, "stack"):
        _vec_layer_stack.stack = []
    _vec_layer_stack.stack.append((comp, tag))


def _pop_layer() -> None:
    if hasattr(_vec_layer_stack, "stack") and _vec_layer_stack.stack:
        _vec_layer_stack.stack.pop()


def _current_layer() -> tuple[Component, str]:
    """Returns (UNKNOWN, 'unattributed') when the stack is empty."""
    if hasattr(_vec_layer_stack, "stack") and _vec_layer_stack.stack:
        return _vec_layer_stack.stack[-1]
    return (Component.UNKNOWN, "unattributed")


# ---------------------------------------------------------------------------
# Helpers for dimension-aware reductions (amax, sum, mean)
# ---------------------------------------------------------------------------

def _reduce_with_dim(vrf, x_in: torch.Tensor, func, dim_arg, keepdim: bool) -> torch.Tensor:
    """Apply a staged VPU row-reduction (rsum or rmax) over a single dim."""
    is_max = (func is torch.ops.aten.amax.default)
    reduce_fn = _vpu_rmax_all if is_max else _vpu_rsum_all

    x_bf16 = x_in.bfloat16()
    d = (dim_arg[0] if isinstance(dim_arg, (list, tuple)) else dim_arg)
    if d < 0:
        d = x_in.dim() + d

    # Bring target dim to last position for row-reduction
    x_t = x_bf16.transpose(d, -1).contiguous()
    result = reduce_fn(vrf, x_t)               # (..., 1) — last dim is the reduced one
    result = result.transpose(d, -1).contiguous()
    if not keepdim:
        result = result.squeeze(d)
    return result.to(x_in.dtype)


# ---------------------------------------------------------------------------
# TorchDispatchMode subclass
# ---------------------------------------------------------------------------

class VectorQuantMode(TorchDispatchMode):
    """
    Context manager that routes every target vector op through VectorRTLFunctions
    when functional_model is set, or passes through unchanged otherwise.

    All ops in TARGET_OPS are intercepted unconditionally — there is no
    component gate.  Ops outside any hooked module (final RMSNorm, Euler step)
    are labelled "unattributed" in the tracker output.
    """

    def __init__(
        self,
        tracker: Optional[StatsTracker] = None,
        functional_model=None,
        verbose: bool = False,
        estimated_total: int = 0,
        active_groups: Optional[set[QuantGroup]] = None,
        io_store: Optional[VectorIOStore] = None,
        capture_mode: bool = False,
        clean_input_store=None,
        substitute_clean_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.tracker          = tracker
        self.functional_model = functional_model
        self.verbose          = verbose
        self._estimated_total = estimated_total
        self._call_count      = 0
        self._active_components: Optional[frozenset[Component]] = (
            frozenset(c for g in active_groups for c in _GROUP_TO_COMPONENTS[g])
            if active_groups is not None else None
        )
        self.io_store         = io_store
        self.capture_mode     = capture_mode
        self.clean_input_store = clean_input_store
        self.substitute_clean_inputs = substitute_clean_inputs

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Re-entrant guard: VPU decompositions use aten ops internally
        if getattr(_in_quant_guard, "active", False):
            return func(*args, **kwargs)

        # Only intercept target vector ops
        if func not in TARGET_OPS:
            return func(*args, **kwargs)

        # Skip non-floating-point tensors (bool/int index arithmetic, masks, etc.)
        # The VPU model only handles floating-point; converting bool/int to bfloat16
        # and back corrupts integer values and can return wrong dtypes (e.g. bool sum
        # flowing into a subtraction that PyTorch rejects).
        first_tensor = next((a for a in args if isinstance(a, torch.Tensor)), None)
        if first_tensor is not None and not first_tensor.is_floating_point():
            return func(*args, **kwargs)

        current_comp, layer_tag = _current_layer()

        # Component gate: skip VPU model for inactive components.
        if (self._active_components is not None
                and current_comp is not Component.UNKNOWN
                and current_comp not in self._active_components):
            return func(*args, **kwargs)

        # op_key: used by both capture mode and clean-input lookup
        op_name = getattr(
            getattr(func, "_overloadpacket", None),
            "_qualified_op_name",
            str(func),
        )
        shape_str = "_".join(str(d) for d in first_tensor.shape) if first_tensor is not None else "s"
        op_key = f"{layer_tag}.{op_name.split('::')[-1]}.{shape_str}"

        # ── Capture mode: store clean args + Pass 1 output, return unchanged ──
        if self.capture_mode:
            y = func(*args, **kwargs)
            if self.clean_input_store is not None:
                self.clean_input_store.capture(op_key, args)
                y_t = y[0] if isinstance(y, tuple) else y
                if isinstance(y_t, torch.Tensor):
                    self.clean_input_store.capture(f"{op_key}.__out__", (y_t,))
            return y

        # ── Functional model path ────────────────────────────────────────────
        if self.functional_model is not None:
            vrf = self.functional_model
            # When substitute_clean_inputs is False, FM sees real upstream args
            # (noise propagates) but the store is still consulted for the
            # comparison reference further down.
            clean_args = (self.clean_input_store.get(op_key)
                          if (self.substitute_clean_inputs and self.clean_input_store is not None)
                          else None)
            effective_args = clean_args if clean_args is not None else args

            _in_quant_guard.active = True
            try:
                out = self._dispatch_fm(func, effective_args, kwargs, vrf)

                if self.tracker is not None or self.io_store is not None:
                    y_ref_t = None
                    if self.clean_input_store is not None:
                        _stored = self.clean_input_store.get(f"{op_key}.__out__")
                        if _stored is not None:
                            y_ref_t = _stored[0]
                    # No fall back
                    out_t   = out[0]   if isinstance(out,   tuple) else out
                    if isinstance(y_ref_t, torch.Tensor) and isinstance(out_t, torch.Tensor):
                        if self.tracker is not None:
                            self._call_count += 1
                            self.tracker.record(
                                name=f"vec.{layer_tag}.{op_name}.{self._call_count}",
                                component=current_comp,
                                y_fp=y_ref_t,
                                y_quant=out_t,
                            )
                            if self.verbose:
                                last = self.tracker.calls[-1]
                                rmse = last["rmse"]
                                total_str = f"/~{self._estimated_total}" if self._estimated_total else ""
                                pct = (f"{100 * self._call_count // self._estimated_total:3d}%"
                                       if self._estimated_total else "   ")
                                op_short = op_name.split("::")[-1]
                                shape = tuple(effective_args[0].shape) if isinstance(effective_args[0], torch.Tensor) else "?"
                                print(
                                    f"[VEC | {self._call_count:5d}{total_str} | {pct}]"
                                    f"  {layer_tag:<20}  {op_short:<22}  in={shape}"
                                    f"  rmse={rmse:.6f}",
                                    flush=True,
                                )
                        if self.io_store is not None:
                            _rmse = self.tracker.calls[-1]["rmse"] if self.tracker is not None else 0.0
                            self.io_store.record(op_name, layer_tag, list(effective_args), y_ref_t, out_t, rmse=_rmse)
            finally:
                _in_quant_guard.active = False
            return out

        # ── Passthrough (no FM) ───────────────────────────────────────────────
        return func(*args, **kwargs)

    # ------------------------------------------------------------------
    def _dispatch_fm(self, func, args, kwargs, vrf):
        """Route a single op through the VPU functional model."""

        # ── Fused composite ops ──────────────────────────────────────────
        if func is torch.ops.aten.native_layer_norm.default:
            # args: (x, norm_shape, weight, bias, eps)
            x, _, weight, bias, eps = args[0], args[1], args[2], args[3], args[4]
            y = _vpu_layer_norm(vrf, x.bfloat16(), weight, bias, eps)
            # native_layer_norm returns (output, mean, rstd); only output is used
            return (y.to(x.dtype), None, None)

        if func is torch.ops.aten._softmax.default:
            # args: (x, dim, half_to_float)
            x, dim = args[0], args[1]
            y = _vpu_softmax(vrf, x.bfloat16(), dim)
            return y.to(x.dtype)

        if func is torch.ops.aten.gelu.default:
            x = args[0]
            return _vpu_gelu_tanh(vrf, x.bfloat16()).to(x.dtype)

        if func is torch.ops.aten.silu.default:
            x = args[0]
            return _vpu_silu(vrf, x.bfloat16()).to(x.dtype)

        if func is torch.ops.aten.rsqrt.default:
            x = args[0]
            return _vpu_rsqrt(vrf, x.bfloat16()).to(x.dtype)

        # ── Dimension-aware reductions ────────────────────────────────────
        if func is torch.ops.aten.mean.dim:
            # args: (x, dim_list, keepdim[, dtype])
            x_in     = args[0]
            dim_list = args[1]
            keepdim  = args[2] if len(args) > 2 else False
            d = dim_list[0] if isinstance(dim_list, (list, tuple)) else dim_list
            N = x_in.shape[d]
            sum_out = _reduce_with_dim(vrf, x_in, torch.ops.aten.sum.dim_IntList, dim_list, keepdim)
            inv_n   = torch.full_like(sum_out.bfloat16(), 1.0 / N)
            return vrf.mul(sum_out.bfloat16(), inv_n).to(x_in.dtype)

        if func is torch.ops.aten.amax.default:
            x_in    = args[0]
            dim_arg = args[1] if len(args) > 1 else None
            keepdim = args[2] if len(args) > 2 else False
            if dim_arg is None:
                flat = _vpu_rmax_all(vrf, x_in.bfloat16().flatten().unsqueeze(0))
                return flat.squeeze().to(x_in.dtype)
            return _reduce_with_dim(vrf, x_in, func, dim_arg, keepdim)

        if func is torch.ops.aten.sum.dim_IntList:
            x_in    = args[0]
            dim_arg = args[1]
            keepdim = args[2] if len(args) > 2 else False
            return _reduce_with_dim(vrf, x_in, func, dim_arg, keepdim)

        if func is torch.ops.aten.sum.default:
            x_in = args[0]
            flat = _vpu_rsum_all(vrf, x_in.bfloat16().flatten().unsqueeze(0))
            return flat.squeeze().to(x_in.dtype)

        # ── Atomic pointwise ops via dispatch table ───────────────────────
        if func not in _VEC_FM_DISPATCH:
            return func(*args, **kwargs)

        bf16_args = [
            a.bfloat16() if isinstance(a, torch.Tensor) else a
            for a in args
        ]
        method = _VEC_FM_DISPATCH[func]
        out    = method(vrf, *bf16_args)
        if isinstance(out, torch.Tensor):
            out = out.to(args[0].dtype)
        return out


# ---------------------------------------------------------------------------
# patch / unpatch
# ---------------------------------------------------------------------------

def patch_vector_ops(
    model: torch.nn.Module,
    active_groups: set[QuantGroup],
    tracker: Optional[StatsTracker] = None,
    functional_model=None,
    verbose: bool = False,
    estimated_total: int = 0,
    io_store: Optional[VectorIOStore] = None,
    capture_mode: bool = False,
    clean_input_store=None,
    substitute_clean_inputs: bool = True,
) -> tuple[list, VectorQuantMode, dict]:
    """
    Register layer-tag hooks for attribution and return a
    (handles, VectorQuantMode, hook_fires) triple.

    Hooks push (Component, layer_tag) onto a thread-local stack so
    __torch_dispatch__ can label each op.  All ops in TARGET_OPS are routed
    through VRF unconditionally — there is no attribution gate.  Ops outside
    any hook (final RMSNorm, Euler step) are labelled "unattributed".

    Args:
        model:            Pi0Pytorch model (or any nn.Module).
        active_groups:    Which QuantGroups to register labeling hooks for.
                          Does NOT gate VRF routing (all ops always go to VRF).
        tracker:          Optional StatsTracker for RMSE recording.
        functional_model: VectorRTLFunctions instance, or None for passthrough.
        verbose:          If True, print one trace line per op to stdout.
        estimated_total:  Estimated total vec-op count for % display in trace.

    Returns:
        (handles, ctx, hook_fires) where:
          handles    — list of hook handles (pass to unpatch_vector_ops)
          ctx        — VectorQuantMode instance (enter as context manager)
          hook_fires — dict[component_name, int] updated in-place by hooks
    """
    handles: list = []
    hook_fires: dict = defaultdict(int)
    hooked: list[tuple[str, str, str]] = []  # (module_path, component, layer_tag)

    for name, mod in model.named_modules():
        result = _vec_layer_tag_for_module(name)
        if result is None:
            continue
        comp, tag = result

        def _make_hooks(c: Component, t: str, fires: dict, cname: str):
            def pre(mod, inp):
                _push_layer(c, t)
                fires[cname] += 1
            def post(mod, inp, out):
                _pop_layer()
            return pre, post

        pre_h, post_h = _make_hooks(comp, tag, hook_fires, comp.value)
        handles.append(mod.register_forward_pre_hook(pre_h))
        handles.append(mod.register_forward_hook(post_h))
        hooked.append((name, comp.value, tag))

    ctx = VectorQuantMode(
        tracker=tracker,
        functional_model=functional_model,
        verbose=verbose,
        estimated_total=estimated_total,
        active_groups=active_groups,
        io_store=io_store,
        capture_mode=capture_mode,
        clean_input_store=clean_input_store,
        substitute_clean_inputs=substitute_clean_inputs,
    )

    fm_label = (
        type(functional_model).__name__ if functional_model is not None else "passthrough"
    )
    print(f"\n[patch_vector_ops] functional_model={fm_label}  verbose={verbose}")
    print(f"  {'module':<60}  {'component':<14}  layer_tag")
    print(f"  {'-'*60}  {'-'*14}  ---------")
    for mod_path, comp_name, tag in hooked:
        print(f"  {mod_path:<60}  {comp_name:<14}  {tag}")
    if not hooked:
        print("  (no modules hooked — check _vec_layer_tag_for_module vs model structure)")
    print()
    return handles, ctx, hook_fires


def unpatch_vector_ops(handles: list) -> None:
    """Remove all hooks registered by patch_vector_ops."""
    for h in handles:
        h.remove()
    print(f"[unpatch_vector_ops] Removed {len(handles)} hooks.")
