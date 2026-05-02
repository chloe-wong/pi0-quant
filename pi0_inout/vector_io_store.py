"""
vector_io_store.py
------------------
Captures per-op vector I/O tensors and writes one .npz file per (layer_tag, op) pair.
Only records when a VPU functional model is active.

Captured tensors
----------------
For each intercepted vector op, calls are stored consecutively:

  call{N}_input_0          bf16 raw bits — first arg (tensor)
  call{N}_input_1          bf16 raw bits — second arg (tensor or scalar as bf16)
  ...
  call{N}_reference_output bf16 raw bits — PyTorch/reference aten op output
  call{N}_fm_output        bf16 raw bits — VPU functional model output
  call{N}_rmse             float32 scalar — RMSE(reference_output, fm_output)

All bf16 tensors stored as int16 raw bits.
  Reconstruct: torch.from_numpy(arr).view(torch.bfloat16)
rmse stored as plain float32 scalar (not encoded).

File naming
-----------
{layer_tag}__{op_short}.npz
  e.g. language__7__add.npz, vision__native_layer_norm.npz
  (dots in layer_tag replaced with __)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, List

import numpy as np
import torch


def _to_bf16_numpy(t: torch.Tensor) -> np.ndarray:
    """Store tensor as int16 raw bf16 bits.
    Reload with: torch.from_numpy(arr).view(torch.bfloat16)"""
    return t.detach().bfloat16().view(torch.int16).cpu().numpy()


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
        # key -> list of per-call dicts {arg_key: np.ndarray or scalar}
        self._calls: dict[str, list[dict]] = defaultdict(list)

    def record(
        self,
        op_name: str,
        layer_tag: str,
        inputs: List[Any],
        ref_output: torch.Tensor,
        fm_output: torch.Tensor,
        rmse: float = 0.0,
    ) -> None:
        """Record one op invocation.

        op_name   : qualified aten op name e.g. "aten::add", "aten::native_layer_norm"
        layer_tag : transformer layer label e.g. "language.7", "vision", "unattributed"
        inputs    : all positional args (tensors and scalars)
        ref_output: PyTorch/reference aten output tensor
        fm_output : VPU functional model output tensor
        rmse      : RMSE(ref_output, fm_output) — stored as float32 scalar
        """
        op_short = op_name.split("::")[-1]
        key = f"{layer_tag}.{op_short}"

        entry: dict = {}
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                entry[f"input_{i}"] = _to_bf16_numpy(inp)
            elif isinstance(inp, (int, float, bool)):
                entry[f"input_{i}"] = _to_bf16_numpy(torch.tensor(float(inp), dtype=torch.bfloat16))
            # skip non-numeric args (e.g. None, lists of ints for dim specs)

        entry["reference_output"] = _to_bf16_numpy(ref_output)
        entry["fm_output"]        = _to_bf16_numpy(fm_output)
        entry["rmse"]             = np.float32(rmse)

        self._calls[key].append(entry)

    def save(self) -> None:
        """Write one .npz per (layer_tag, op). Call after all inference is complete."""
        if not self._calls:
            print("[VectorIOStore] No tensors recorded — nothing to save.")
            return

        for key, calls in self._calls.items():
            n_calls = len(calls)
            arrays: dict = {"n_calls": np.array(n_calls, dtype=np.int64)}

            for i, call in enumerate(calls):
                def _key_order(k):
                    if k.startswith("input_"):
                        return (0, k)
                    if k == "reference_output":
                        return (1, k)
                    if k == "fm_output":
                        return (2, k)
                    return (3, k)  # rmse
                for arg_key in sorted(call.keys(), key=_key_order):
                    arrays[f"call{i}_{arg_key}"] = call[arg_key]

            fname = key.replace(".", "__") + ".npz"
            np.savez(self.save_dir / fname, **arrays)

        print(
            f"[VectorIOStore] Saved {len(self._calls)} op .npz files → {self.save_dir}"
        )
