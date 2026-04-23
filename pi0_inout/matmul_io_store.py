"""
matmul_io_store.py
------------------
Captures per-layer matmul inputs and outputs for both the unpatched reference
pass and the patched (quantized) pass, then writes one .npz file per layer.

Captured tensors
----------------
Unpatched pass (via forward hooks registered in run_eval.py):
    unpatched_x       [N, *]  activation arriving at this layer
    unpatched_w       [out, in]  weight (stored once — static)
    unpatched_b       [out]  bias (stored once; absent if no bias)
    unpatched_y       [N, *]  F.linear(x, w, b) output

Patched pass (via QuantLinear.forward):
    patched_x_fp8        [N, *]  raw FP8 E4M3 bytes of input (uint8)
    patched_x_fp8_scale  scalar  per-tensor scale exponent for x (int8): scale = 2 ** exp
    patched_w_fp8        [out, in]  raw FP8 E4M3 bytes of weight (uint8; stored once)
    patched_w_fp8_scale  scalar  per-tensor scale exponent for w (int8; stored once): scale = 2 ** exp
    patched_b_fp8        [out]  raw FP8 E4M3 bytes of bias (uint8; stored once; absent if no bias)
    patched_b_fp8_scale  scalar  per-tensor scale exponent for b (int8; stored once; absent if no bias): scale = 2 ** exp
    patched_y_quant      [N, *]  functional model / format-flag output (int16 BF16 bits)

BF16 arrays stored as int16 raw bits (numpy lacks native bf16).
  Reconstruct: torch.from_numpy(arr).view(torch.bfloat16)
FP8 arrays stored as uint8 raw bits + int8 scale exponent.
  Reconstruct: torch.from_numpy(arr).view(torch.float8_e4m3fn) * (2 ** scale_exp)

File naming
-----------
Layer names use dots as separators (e.g.
    paligemma_with_expert.paligemma.language_model.model.layers.0.self_attn.q_proj
).  Dots are replaced with '__' so the name is a valid filename:
    paligemma_with_expert__paligemma__language_model__model__layers__0__self_attn__q_proj.npz
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_bf16_numpy(t: torch.Tensor) -> np.ndarray:
    """Store bf16 tensor as int16 raw bits (numpy lacks native bf16).
    Assumes input is already bfloat16. Reload with: torch.from_numpy(arr).view(torch.bfloat16)"""
    return t.detach().view(torch.int16).cpu().numpy()


def _maybe_bf16_numpy(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    return None if t is None else _to_bf16_numpy(t)


def _to_uint8_numpy(t: torch.Tensor) -> np.ndarray:
    """Store raw FP8 bit patterns as uint8 (one byte per element)."""
    return t.detach().view(torch.uint8).cpu().numpy()

# in case bias is none
def _maybe_uint8_numpy(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    return None if t is None else _to_uint8_numpy(t)


def _stack_calls(
    calls: List[dict],
    prefix: str,
    static_keys: set,
) -> dict:
    """
    Stack per-call dicts of numpy arrays.

    - Dynamic keys (activations, outputs): stack along new axis 0 → [N, ...]
    - Static keys (weight, bias): take from the first call that has them
    - Keys with all-None values across calls are skipped entirely

    Falls back to per-call keys (``prefix_key_callN``) if shapes are
    inconsistent across calls.
    """
    all_keys = {k for c in calls for k in c if c[k] is not None}
    out: dict = {}

    for key in sorted(all_keys):
        prefixed = f"{prefix}_{key}"
        if key in static_keys:
            for c in calls:
                if c.get(key) is not None:
                    out[prefixed] = c[key]
                    break
        else:
            frames = [c[key] for c in calls if c.get(key) is not None]
            if not frames:
                continue
            try:
                out[prefixed] = np.stack(frames, axis=0)
            except ValueError:
                # Shape mismatch across calls — fall back to per-call keys
                for i, arr in enumerate(frames):
                    out[f"{prefixed}_call{i}"] = arr

    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MatmulIOStore:
    """
    Accumulates per-layer matmul I/O tensors across inference calls and saves
    them as one .npz file per layer.

    Usage in run_eval.py:
        store = MatmulIOStore(out_dir / "tensors")
        # Register unpatched hooks before reference pass (see run_eval.py)
        # Pass store to patch_model(..., matmul_io_store=store)
        # Run inference
        store.save()
    """

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # layer_name -> list of per-call dicts
        self._unpatched: Dict[str, List[dict]] = defaultdict(list)
        self._patched:   Dict[str, List[dict]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @torch._dynamo.disable
    def record_unpatched(
        self,
        name: str,
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        y: torch.Tensor,
    ) -> None:
        """Called from the forward hook on nn.Linear during the reference pass."""
        self._unpatched[name].append({
            "x": _to_bf16_numpy(x),
            "w": _to_bf16_numpy(w),
            "b": _maybe_bf16_numpy(b),
            "y": _to_bf16_numpy(y),
        })

    @torch._dynamo.disable
    def record_patched(
        self,
        name: str,
        x_fp8: torch.Tensor,
        x_fp8_scale: int,
        w_fp8: torch.Tensor,
        w_fp8_scale: int,
        b_fp8: Optional[torch.Tensor],
        b_fp8_scale: Optional[int],
        y_quant: torch.Tensor,
    ) -> None:
        """Called from QuantLinear.forward() during the patched pass."""
        self._patched[name].append({
            "x_fp8":       _to_uint8_numpy(x_fp8),
            "x_fp8_scale": np.int8(x_fp8_scale),
            "w_fp8":       _to_uint8_numpy(w_fp8),
            "w_fp8_scale": np.int8(w_fp8_scale),
            "b_fp8":       _maybe_uint8_numpy(b_fp8),
            "b_fp8_scale": np.int8(b_fp8_scale) if b_fp8_scale is not None else None,
            "y_quant":     _to_bf16_numpy(y_quant),
        })

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write one .npz per layer. Call after all inference is complete."""
        # Only save layers that were patched — _patched is populated only for
        # active-groups layers, so this respects --active-groups filtering.
        all_names = sorted(set(self._patched))
        if not all_names:
            print("[MatmulIOStore] No tensors recorded — nothing to save.")
            return

        for name in all_names:
            arrays: dict = {}
            up_calls  = self._unpatched.get(name, [])
            pat_calls = self._patched.get(name, [])
            n_calls   = max(len(up_calls), len(pat_calls))
            arrays["n_calls"] = np.array(n_calls, dtype=np.int64)

            if up_calls:
                arrays.update(
                    _stack_calls(up_calls, prefix="unpatched", static_keys={"w", "b"})
                )
            if pat_calls:
                arrays.update(
                    _stack_calls(pat_calls, prefix="patched",
                                 static_keys={"w_fp8", "w_fp8_scale", "b_fp8", "b_fp8_scale"})
                )

            fname = name.replace(".", "__") + ".npz"
            np.savez(self.save_dir / fname, **arrays)

        print(
            f"[MatmulIOStore] Saved {len(all_names)} layer .npz files → {self.save_dir}"
        )
