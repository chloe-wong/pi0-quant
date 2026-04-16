#!/usr/bin/env python3
"""
verify_npz.py — Decode a MatmulIOStore .npz file and recover original float values.

Usage:
    python verify_npz.py path/to/layer.npz [--summary] [--call 0] [--log-dir DIR]

Requirements: torch >= 2.1, numpy

Equations
---------
Unpatched fields (unpatched_x, unpatched_w, unpatched_b, unpatched_y):
    stored as int16 raw bfloat16 bit patterns (numpy has no native bf16)
    value = reinterpret_cast<bfloat16>(int16_array)

Patched inputs/weight/bias (FP8 E4M3):
    stored as (uint8 raw-bit-pattern, int32 scale exponent)
    value = reinterpret_cast<float8_e4m3fn>(uint8_array) * (2 ** scale_exp)
    where scale_exp is ONE integer for the ENTIRE tensor (po2 mode only):
      scale_exp = floor(log2(max(|x_ij|) / 256.0))

Patched output (patched_y_quant):
    stored as int16 raw bfloat16 bit patterns — same as unpatched fields
    value = reinterpret_cast<bfloat16>(int16_array)
"""

import argparse
from pathlib import Path

import numpy as np
import torch

torch.set_printoptions(threshold=torch.inf)

# ── Reconstruction helpers ──────────────────────────────────────────────────

def decode_bf16(arr: np.ndarray) -> torch.Tensor:
    """int16 raw-bit array → native bfloat16 tensor."""
    return torch.from_numpy(arr).view(torch.bfloat16)


def decode_fp8_e4m3(raw_uint8: np.ndarray, scale_exp: int) -> tuple[torch.Tensor, int]:
    """
    uint8 raw-bit array → (native float8_e4m3fn tensor, scale_exp).

    raw_uint8  : uint8 ndarray — raw FP8-E4M3 bit patterns (1 byte per element)
    scale_exp  : int (PER-TENSOR — one exponent shared across ALL elements)
                 e.g. scale_exp=5 means scale=32.0
    Returns    : (fp8_tensor, scale_exp)
                 Dequantized value = fp8_tensor.float() * (2 ** scale_exp)
    """
    return torch.from_numpy(raw_uint8).view(torch.float8_e4m3fn), scale_exp


# ── Main ────────────────────────────────────────────────────────────────────

def load_layer(path: str, call_idx: int = 0):
    data = np.load(path)
    n_calls = int(data["n_calls"])
    print(f"File : {path}")
    print(f"Calls: {n_calls}  (showing call index {call_idx})\n")

    def get(key):
        """Return data[key][call_idx] for dynamic arrays, data[key] for static."""
        if key not in data:
            return None
        arr = data[key]
        # static arrays (w, b) have shape [out, in] or [out] — no call axis
        # dynamic arrays have shape [N, ...] — select call_idx
        if arr.ndim >= 1 and arr.shape[0] == n_calls:
            return arr[call_idx]
        return arr

    results = {}
    scale_exps = {}  # field -> scale exponent (int)

    # ── Unpatched ──
    for field in ("x", "w", "b", "y"):
        key = f"unpatched_{field}"
        raw = get(key)
        if raw is not None:
            results[f"unpatched_{field}"] = decode_bf16(raw)

    # ── Patched inputs / weight / bias (FP8) ──
    for field in ("x", "w", "b"):
        raw_key   = f"patched_{field}_fp8"
        scale_key = f"patched_{field}_fp8_scale"
        raw   = get(raw_key)
        scale = get(scale_key)
        if raw is not None and scale is not None:
            fp8_tensor, exp = decode_fp8_e4m3(raw, int(scale))
            scale_exps[f"patched_{field}"] = exp
            results[f"patched_{field}"] = fp8_tensor

    # ── Patched output ──
    raw = get("patched_y_quant")
    if raw is not None:
        results["patched_y"] = decode_bf16(raw)

    return results, scale_exps, data


def summarize(tensors: dict, scale_exps: dict):
    print(f"{'Field':<20}  {'Shape':<25}  {'min':>12}  {'max':>12}  {'mean':>12}")
    print("-" * 85)
    for name, t in tensors.items():
        # fp8 doesn't support min/max/mean — cast to float32 with scale applied
        if name in scale_exps:
            t_f = t.float() * (2.0 ** scale_exps[name])
        else:
            t_f = t
        print(f"{name:<20}  {str(tuple(t_f.shape)):<25}  "
              f"{t_f.min().item():>12.6f}  {t_f.max().item():>12.6f}  "
              f"{t_f.mean().item():>12.6f}")


def main():
    parser = argparse.ArgumentParser(description="Decode a MatmulIOStore .npz file")
    parser.add_argument("npz", help="Path to .npz file")
    parser.add_argument("--summary", action="store_true",
                        help="Print per-tensor statistics instead of raw values")
    parser.add_argument("--call", type=int, default=0,
                        help="Which inference call to decode (default: 0)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory to save log file (default: print to stdout only). "
                             "Log file is named after the .npz stem, e.g. action_in_proj_call0.log")
    args = parser.parse_args()

    # ── Set up output (stdout + optional log file) ──
    log_file = None
    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.npz).stem
        log_path = log_dir / f"{stem}_call{args.call}.log"
        log_file = open(log_path, "w")
        print(f"Logging to {log_path}")

    def emit(*a, **kw):
        print(*a, **kw)
        if log_file is not None:
            print(*a, **kw, file=log_file)

    tensors, scale_exps, _ = load_layer(args.npz, call_idx=args.call)

    if args.summary:
        lines = []
        lines.append(f"{'Field':<20}  {'Shape':<25}  {'min':>12}  {'max':>12}  {'mean':>12}  {'scale_exp':>10}")
        lines.append("-" * 97)
        for name, t in tensors.items():
            if name in scale_exps:
                t_f = t.float() * (2.0 ** scale_exps[name])
            else:
                t_f = t
            exp_str = str(scale_exps[name]) if name in scale_exps else ""
            lines.append(
                f"{name:<20}  {str(tuple(t_f.shape)):<25}  "
                f"{t_f.min().item():>12.6f}  {t_f.max().item():>12.6f}  "
                f"{t_f.mean().item():>12.6f}  {exp_str:>10}"
            )
        emit("\n".join(lines))
    else:
        for name, t in tensors.items():
            exp_str = f"  scale_exp={scale_exps[name]}  (scale={2**scale_exps[name]})" if name in scale_exps else ""
            emit(f"\n=== {name}  shape={tuple(t.shape)}{exp_str} ===")
            emit(str(t))

    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
