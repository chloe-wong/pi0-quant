"""
reference_store.py
------------------
Captures layer outputs from an unpatched forward pass so that QuantLinear
can compute cumulative (end-to-end propagated) RMSE alongside local RMSE.

Usage in run_eval.py:
    store = ReferenceStore()
    layer_names = {name for name, m in model.named_modules() if type(m) is nn.Linear or type(m) is nn.Conv2d}
    handles = store.register_hooks(model, layer_names)
    with torch.no_grad():
        for obs in observations:
            model.sample_actions(device, obs, num_steps=num_steps)
    for h in handles:
        h.remove()
    # store now holds reference outputs for all layers, in call order

    # Pass to patch_model:
    patch_model(model, ..., reference_store=store)

    # Before each patched forward pass, reset counters so lookup starts from 0:
    store.reset_counters()
"""

from __future__ import annotations

import torch


class ReferenceStore:
    """
    Stores reference (unpatched) layer outputs captured via forward hooks.

    For each named layer, outputs are stored in call order as a list.
    During the patched forward pass, get() returns them in the same order.
    Call reset_counters() between observations to restart the lookup index.
    """

    def __init__(self) -> None:
        # layer_name -> [tensor_call0, tensor_call1, ...]  (CPU, detached)
        self._outputs: dict[str, list[torch.Tensor]] = {}
        # per-layer call index for the current patched forward pass
        self._counters: dict[str, int] = {}

    def register_hooks(self, model: torch.nn.Module, layer_names: set[str]) -> list:
        """
        Register forward hooks on the named layers to capture their outputs.
        Returns a list of hook handles — call h.remove() on each when done.
        """
        handles = []
        for name, module in model.named_modules():
            if name not in layer_names:
                continue

            def _make_hook(n: str):
                def _hook(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        if n not in self._outputs:
                            self._outputs[n] = []
                        self._outputs[n].append(out.detach().clone())
                return _hook

            handles.append(module.register_forward_hook(_make_hook(name)))
        return handles

    def get(self, name: str) -> torch.Tensor | None:
        """
        Return the next reference output for this layer (call-order aware).
        Returns None if no reference was captured for this layer.
        """
        outputs = self._outputs.get(name)
        if not outputs:
            return None
        idx = self._counters.get(name, 0)
        if idx >= len(outputs):
            return None
        self._counters[name] = idx + 1
        return outputs[idx]

    def capture(self, name: str, tensor: torch.Tensor) -> None:
        """
        Manually store a tensor under `name` (same contract as hook-based capture).
        Use for function-level ops (e.g. eager_attention_forward, SDPA) that are not
        nn.Module outputs and can't be captured via register_hooks.
        """
        if name not in self._outputs:
            self._outputs[name] = []
        self._outputs[name].append(tensor.detach().clone())

    def reset_counters(self) -> None:
        """Reset per-layer call indices. Call before each patched forward pass."""
        self._counters.clear()

    def __len__(self) -> int:
        return sum(len(v) for v in self._outputs.values())
