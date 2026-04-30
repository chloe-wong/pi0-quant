#!/usr/bin/env python3
"""
decode_vec_npz.py — Decode a VectorIOStore .npz file and recover original float values.

Usage:
    python decode_vec_npz.py path/to/layer.npz [--summary] [--call 0] [--log-dir DIR]

Requirements: torch >= 2.1, numpy

Schema
------
All tensors stored as int16 raw bfloat16 bit patterns.
Reconstruct with: torch.from_numpy(arr).view(torch.bfloat16)

Per-call keys (call index N):
  call{N}_input_0          first positional arg
  call{N}_input_1          second positional arg (if present)
  ...
  call{N}_reference_output passthrough (aten op) output
  call{N}_fm_output        VPU functional model output
  call{N}_rmse             float32 scalar — RMSE(reference_output, fm_output)

File naming: {layer_tag}__{op_short}.npz  (dots in layer_tag → __)
"""

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.set_printoptions(threshold=torch.inf)


def decode_bf16(arr: np.ndarray) -> torch.Tensor:
    """int16 raw-bit array → native bfloat16 tensor."""
    return torch.from_numpy(arr).view(torch.bfloat16)


def _field_sort_key(field: str) -> tuple:
    if field.startswith("input_"):
        return (0, field)
    if field == "reference_output":
        return (1, field)
    if field == "fm_output":
        return (2, field)
    return (3, field)  # rmse


def load_call(path: str, call_idx: int = 0):
    data = np.load(path)
    n_calls = int(data["n_calls"])
    print(f"File : {path}")
    print(f"Op   : {Path(path).stem.rsplit('__', 1)[-1]}")
    print(f"Calls: {n_calls}  (showing call index {call_idx})\n")

    prefix = f"call{call_idx}_"
    fields = [k[len(prefix):] for k in data.files if k.startswith(prefix)]
    fields.sort(key=_field_sort_key)

    tensors: OrderedDict[str, Any] = OrderedDict()
    for field in fields:
        if field == "rmse":
            tensors[field] = float(data[f"{prefix}{field}"])
        else:
            tensors[field] = decode_bf16(data[f"{prefix}{field}"])

    return tensors, n_calls


def summarize(tensors: dict, *, emit=print):
    lines = []
    if "rmse" in tensors:
        lines.append(f"RMSE : {tensors['rmse']:.8f}\n")
    lines.append(f"{'Field':<20}  {'Shape':<25}  {'min':>12}  {'max':>12}  {'mean':>12}")
    lines.append("-" * 85)
    for name, t in tensors.items():
        if name == "rmse":
            continue
        t_f = t.float()
        lines.append(
            f"{name:<20}  {str(tuple(t_f.shape)):<25}  "
            f"{t_f.min().item():>12.6f}  {t_f.max().item():>12.6f}  "
            f"{t_f.mean().item():>12.6f}"
        )

    emit("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Decode a VectorIOStore .npz file")
    parser.add_argument("npz", help="Path to .npz file")
    parser.add_argument("--summary", action="store_true",
                        help="Print per-tensor statistics instead of raw values")
    parser.add_argument("--call", type=int, default=0,
                        help="Which inference call to decode (default: 0)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory to save log file (default: print to stdout only). "
                             "Log file is named after the .npz stem, e.g. vision__add_call0.log")
    parser.add_argument("--all-calls", action="store_true",
                        help="Decode all calls in the file (ignores --call)")
    args = parser.parse_args()

    data = np.load(args.npz)
    n_calls = int(data["n_calls"])
    call_indices = range(n_calls) if args.all_calls else [args.call]

    stem = Path(args.npz).stem
    log_file = None
    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        suffix = "all" if args.all_calls else f"call{args.call}"
        log_path = log_dir / f"{stem}_{suffix}.log"
        log_file = open(log_path, "w")
        print(f"Logging to {log_path}")

    def emit(*a, **kw):
        print(*a, **kw)
        if log_file is not None:
            print(*a, **kw, file=log_file)

    for call_idx in call_indices:
        emit(f"\n{'='*60}\n CALL {call_idx}\n{'='*60}")
        tensors, _ = load_call(args.npz, call_idx=call_idx)

        if args.summary:
            summarize(tensors, emit=emit)
        else:
            for name, t in tensors.items():
                if name == "rmse":
                    emit(f"\n=== rmse ===\n{t:.8f}")
                else:
                    emit(f"\n=== {name}  shape={tuple(t.shape)} ===")
                    emit(str(t))

    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
