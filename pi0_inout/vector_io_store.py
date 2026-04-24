"""
vector_io_store.py
------------------
Captures per-op vector I/O tensors and writes one .npz file per (layer_tag, op) pair.
Only records when a VPU functional model is active.

Captured tensors
----------------
For each intercepted vector op:

  input_0              [N, ...]  bf16 raw bits — primary tensor input
  input_1              [N, ...]  bf16 raw bits — secondary tensor input (binary ops)
  reference_output     [N, ...]  bf16 raw bits — passthrough (aten op) output
  fm_output            [N, ...]  bf16 raw bits — VPU functional model output

For native_layer_norm additionally:
  input_1 (weight)     [...]  bf16 raw bits  (static — stored once)
  input_2 (bias)       [...]  bf16 raw bits  (static — stored once)

All bf16 tensors stored as int16 raw bits.
  Reconstruct: torch.from_numpy(arr).view(torch.bfloat16)

File naming
-----------
{layer_tag}__{op_short}.npz
  e.g. language__7__add.npz, vision__native_layer_norm.npz
  (dots in layer_tag replaced with __)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch


# Input indices that are static (model parameters, same across all calls) per op.
_STATIC_INPUT_KEYS: dict[str, frozenset] = {
    "native_layer_norm": frozenset({"input_1", "input_2"}),
}


def _to_bf16_numpy(t: torch.Tensor) -> np.ndarray:
    """Store tensor as int16 raw bf16 bits.
    Reload with: torch.from_numpy(arr).view(torch.bfloat16)"""
    return t.detach().bfloat16().view(torch.int16).cpu().numpy()


def _stack_arrays(calls: list[dict], static_keys: frozenset) -> dict:
    """Stack per-call numpy arrays.

    Dynamic keys: stacked along new axis 0 → [N, ...].
    Static keys: taken from first call that has them.
    Falls back to per-call keys if shapes are inconsistent.
    """
    all_keys = {k for c in calls for k in c if c[k] is not None}
    out: dict = {}

    for key in sorted(all_keys):
        if key in static_keys:
            for c in calls:
                if c.get(key) is not None:
                    out[key] = c[key]
                    break
        else:
            frames = [c[key] for c in calls if c.get(key) is not None]
            if not frames:
                continue
            try:
                out[key] = np.stack(frames, axis=0)
            except ValueError:
                for i, arr in enumerate(frames):
                    out[f"{key}_call{i}"] = arr

    return out


class VectorIOStore:
    """
    Accumulates per-op vector I/O tensors and saves one .npz per (layer_tag, op) pair.

    Usage:
        store = VectorIOStore(out_dir / "vec_tensors")
        # Pass to patch_vector_ops(..., io_store=store)
        # Run inference with VPU functional model active
        store.save()
    """

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._calls: dict[str, list[dict]] = defaultdict(list)

    def record(
        self,
        op_name: str,
        layer_tag: str,
        inputs: List[torch.Tensor],
        ref_output: torch.Tensor,
        fm_output: torch.Tensor,
    ) -> None:
        """Record one op invocation.

        op_name   : qualified aten op name e.g. "aten::add", "aten::native_layer_norm"
        layer_tag : transformer layer label e.g. "language.7", "vision", "unattributed"
        inputs    : tensor args only (scalars already filtered out by caller)
        ref_output: passthrough (aten) output tensor
        fm_output : VPU functional model output tensor
        """
        op_short = op_name.split("::")[-1]
        key = f"{layer_tag}.{op_short}"

        entry: dict = {}
        for i, inp in enumerate(inputs):
            entry[f"input_{i}"] = _to_bf16_numpy(inp)
        entry["reference_output"] = _to_bf16_numpy(ref_output)
        entry["fm_output"]        = _to_bf16_numpy(fm_output)

        self._calls[key].append(entry)

    def save(self) -> None:
        """Write one .npz per (layer_tag, op). Call after all inference is complete."""
        if not self._calls:
            print("[VectorIOStore] No tensors recorded — nothing to save.")
            return

        for key, calls in self._calls.items():
            op_short    = key.split(".")[-1]
            static_keys = _STATIC_INPUT_KEYS.get(op_short, frozenset())
            n_calls     = len(calls)

            arrays: dict = {"n_calls": np.array(n_calls, dtype=np.int64)}
            arrays.update(_stack_arrays(calls, static_keys=static_keys))

            fname = key.replace(".", "__") + ".npz"
            np.savez(self.save_dir / fname, **arrays)

        print(
            f"[VectorIOStore] Saved {len(self._calls)} op .npz files → {self.save_dir}"
        )
