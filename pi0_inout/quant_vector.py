"""
quant_vector.py
---------------
Vector-engine operation quantization for Pi0Pytorch.

Every listed vector op in every layer of the model is intercepted via
TorchDispatchMode and quantized with configurable input/output formats,
independently from the matmul (QuantLinear / patch_attn_sdpa) quantization.

Covered operations
------------------
  Add / Sub       (2 tensor inputs)
  Multiply        (2 tensor inputs)
  Square / Cube   (pow with scalar exponent)
  Div / Reciprocal
  Sqrt
  Sin / Cos
  Tanh
  Log2
  Exp / Exp2
  AMax reduction  (max over dims, returns tensor)
  Sum reduction

Usage
-----
    from pi0_inout import patch_model, patch_attn_sdpa
    from pi0_inout.quant_vector import patch_vector_ops, unpatch_vector_ops

    active = {QuantGroup.TRANSFORMER}
    patch_model(model, mx_input_fmt=FP8, mx_output_fmt=FP16, active_groups=active)
    attn_h = patch_attn_sdpa(model, active_groups=active, ...)
    vec_h, vec_ctx = patch_vector_ops(model, active_groups=active,
                                      vec_input_fmt=FP8, vec_output_fmt=FP16)
    with vec_ctx:
        actions = model.sample_actions(...)

    unpatch_attn_sdpa(attn_h)
    unpatch_model(model)
    unpatch_vector_ops(vec_h)

Component attribution
---------------------
The same component-tagging rules from model_patcher._infer_component are
reused.  Forward hooks are placed on the top-level boundary module for each
component (the outermost module where the component tag first appears).  A
thread-local stack tracks which component is currently executing so that the
dispatch handler can gate quantization correctly.

Re-entrant guard
----------------
quant() itself calls .float() and .to(), which are aten ops that would
re-enter __torch_dispatch__.  A thread-local 'inside_quant' guard suppresses
re-entrant dispatch so quantization calls are not themselves quantized.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from ._dispatch_guards import _in_quant_guard
from .model_patcher import QuantGroup, _active_components, _infer_component
from .quant_types import QuantFormat, quant
from .stats_tracker import Component, StatsTracker


# ---------------------------------------------------------------------------
# Target ops: vector ops only (matmuls excluded — handled by QuantLinear)
# ---------------------------------------------------------------------------

TARGET_OPS: frozenset = frozenset({
    # Add / Sub
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    # Multiply
    torch.ops.aten.mul.Tensor,
    # Square / Cube  (pow.Tensor_Scalar covers x**2, x**3, etc.)
    torch.ops.aten.pow.Tensor_Scalar,
    # Div / Reciprocal
    torch.ops.aten.div.Tensor,
    torch.ops.aten.reciprocal.default,
    # Sqrt
    torch.ops.aten.sqrt.default,
    # Sin / Cos
    torch.ops.aten.sin.default,
    torch.ops.aten.cos.default,
    # Tanh
    torch.ops.aten.tanh.default,
    # Log2
    torch.ops.aten.log2.default,
    # Exp / Exp2
    torch.ops.aten.exp.default,
    torch.ops.aten.exp2.default,
    # Max reduction (amax returns a plain tensor; aten.max returns a namedtuple)
    torch.ops.aten.amax.default,
    # Sum reduction
    torch.ops.aten.sum.default,
    torch.ops.aten.sum.dim_IntList,
})


# ---------------------------------------------------------------------------
# Thread-local state
# ---------------------------------------------------------------------------

# Stack of Component tags set by component-boundary forward hooks.
# Using a stack (not a scalar) so that nested component crossings are correct:
#   entering paligemma → push LANGUAGE
#     entering vision_tower → push VISION  (top = VISION)
#   exiting  vision_tower  → pop         (top = LANGUAGE again)
_vec_component_stack: threading.local = threading.local()

# _in_quant_guard is imported from _dispatch_guards (shared with quant_linear)


def _push_component(c: Component) -> None:
    if not hasattr(_vec_component_stack, "stack"):
        _vec_component_stack.stack = []
    _vec_component_stack.stack.append(c)


def _pop_component() -> None:
    if hasattr(_vec_component_stack, "stack") and _vec_component_stack.stack:
        _vec_component_stack.stack.pop()


def _current_component() -> Optional[Component]:
    if hasattr(_vec_component_stack, "stack") and _vec_component_stack.stack:
        return _vec_component_stack.stack[-1]
    return None


# ---------------------------------------------------------------------------
# Tensor argument helpers
# ---------------------------------------------------------------------------

def _quant_val(v: object, fmt: QuantFormat) -> object:
    """Recursively quantize floating-point tensors in a value; pass others through."""
    if isinstance(v, torch.Tensor) and v.is_floating_point():
        return quant(v.float(), fmt).to(v.dtype)
    if isinstance(v, (list, tuple)):
        result = [_quant_val(x, fmt) for x in v]
        return type(v)(result)
    return v


def _quant_args(args: tuple, fmt: QuantFormat) -> tuple:
    return tuple(_quant_val(a, fmt) for a in args)


def _quant_output(out: object, fmt: QuantFormat) -> object:
    return _quant_val(out, fmt)


# ---------------------------------------------------------------------------
# TorchDispatchMode subclass
# ---------------------------------------------------------------------------

class VectorQuantMode(TorchDispatchMode):
    """
    Context manager that quantizes every target vector op to (vec_input_fmt, vec_output_fmt).

    Only ops executing within an active component (as signalled by the
    component-boundary hooks registered by patch_vector_ops) are quantized.
    All other ops pass through unchanged.
    """

    def __init__(
        self,
        active_groups: set[QuantGroup],
        vec_input_fmt: QuantFormat,
        vec_output_fmt: QuantFormat,
        tracker: Optional[StatsTracker] = None,
    ) -> None:
        super().__init__()
        self.active_comps  = _active_components(set(active_groups))
        self.vec_input_fmt  = vec_input_fmt
        self.vec_output_fmt = vec_output_fmt
        self.tracker        = tracker
        self._call_count    = 0

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Re-entrant guard: quant() itself uses aten ops — don't recurse
        if getattr(_in_quant_guard, "active", False):
            return op(*args, **kwargs)

        # Only intercept target vector ops
        if op not in TARGET_OPS:
            return op(*args, **kwargs)

        # Gate on active component
        current_comp = _current_component()
        if current_comp is None or current_comp not in self.active_comps:
            return op(*args, **kwargs)

        # Quantize inputs, run op, quantize output
        _in_quant_guard.active = True
        try:
            q_args = _quant_args(args, self.vec_input_fmt)
            out    = op(*q_args, **kwargs)
            out_q  = _quant_output(out, self.vec_output_fmt)

            if self.tracker is not None:
                with torch.no_grad():
                    out_fp = op(*args, **kwargs)
                    self._call_count += 1
                    self.tracker.record(
                        name=f"vec.{op._overloadpacket._qualified_op_name}.{current_comp.value}.{self._call_count}",
                        component=current_comp,
                        y_fp=out_fp if isinstance(out_fp, torch.Tensor) else out_fp,
                        y_quant=out_q if isinstance(out_q, torch.Tensor) else out_q,
                    )
        finally:
            _in_quant_guard.active = False

        return out_q


# ---------------------------------------------------------------------------
# patch / unpatch
# ---------------------------------------------------------------------------

def patch_vector_ops(
    model: torch.nn.Module,
    active_groups: set[QuantGroup],
    vec_input_fmt: QuantFormat,
    vec_output_fmt: QuantFormat,
    tracker: Optional[StatsTracker] = None,
) -> tuple[list, VectorQuantMode]:
    """
    Register component-boundary hooks for attribution and return a
    (handles, VectorQuantMode) pair.

    The caller is responsible for entering VectorQuantMode as a context manager
    around inference and removing hooks afterwards via unpatch_vector_ops.

    Args:
        model:          Pi0Pytorch model (or any nn.Module).
        active_groups:  Which QuantGroups to quantize.
        vec_input_fmt:  Format applied to tensor inputs of each vector op.
        vec_output_fmt: Format applied to each vector op's output.
        tracker:        Optional StatsTracker.

    Returns:
        (handles, ctx) where handles is a list of hook handles and ctx is the
        VectorQuantMode instance (not yet entered).
    """
    active_groups = set(active_groups)
    active_comps  = _active_components(active_groups)
    handles: list = []
    n_hooks = 0

    for name, mod in model.named_modules():
        comp = _infer_component(name)
        if comp == Component.UNKNOWN or comp not in active_comps:
            continue

        # Only hook at the top-level boundary for this component — i.e. the
        # outermost module where the component tag first appears.  This avoids
        # placing thousands of hooks on every sub-layer.
        parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
        parent_comp = _infer_component(parent_name)
        if parent_comp == comp:
            continue  # Inner module of same component — parent already hooks it

        def _make_hooks(c: Component):
            def pre(mod, inp):
                _push_component(c)
            def post(mod, inp, out):
                _pop_component()
            return pre, post

        pre_h, post_h = _make_hooks(comp)
        handles.append(mod.register_forward_pre_hook(pre_h))
        handles.append(mod.register_forward_hook(post_h))
        n_hooks += 1

    ctx = VectorQuantMode(active_groups, vec_input_fmt, vec_output_fmt, tracker)

    print(
        f"[patch_vector_ops] Hooked {n_hooks} component-boundary modules "
        f"for groups: {[g.value for g in active_groups]}  "
        f"vec_input_fmt={vec_input_fmt.value}  vec_output_fmt={vec_output_fmt.value}"
    )
    return handles, ctx


def unpatch_vector_ops(handles: list) -> None:
    """Remove all hooks registered by patch_vector_ops."""
    for h in handles:
        h.remove()
    print(f"[unpatch_vector_ops] Removed {len(handles)} hooks.")
