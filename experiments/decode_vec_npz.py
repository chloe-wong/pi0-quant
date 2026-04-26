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

File naming: {layer_tag}__{op_short}.npz  (dots in layer_tag → __)
"""

import argparse
from collections import OrderedDict
from pathlib import Path

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
    return (2, field)  # fm_output


def load_call(path: str, call_idx: int = 0):
    data = np.load(path)
    n_calls = int(data["n_calls"])
    print(f"File : {path}")
    print(f"Op   : {Path(path).stem.rsplit('__', 1)[-1]}")
    print(f"Calls: {n_calls}  (showing call index {call_idx})\n")

    prefix = f"call{call_idx}_"
    fields = [k[len(prefix):] for k in data.files if k.startswith(prefix)]
    fields.sort(key=_field_sort_key)

    tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
    for field in fields:
        tensors[field] = decode_bf16(data[f"{prefix}{field}"])

    return tensors, n_calls


def summarize(tensors: dict, *, emit=print):
    lines = []
    lines.append(f"{'Field':<20}  {'Shape':<25}  {'min':>12}  {'max':>12}  {'mean':>12}")
    lines.append("-" * 85)
    for name, t in tensors.items():
        t_f = t.float()
        lines.append(
            f"{name:<20}  {str(tuple(t_f.shape)):<25}  "
            f"{t_f.min().item():>12.6f}  {t_f.max().item():>12.6f}  "
            f"{t_f.mean().item():>12.6f}"
        )

    if "reference_output" in tensors and "fm_output" in tensors:
        ref = tensors["reference_output"].float()
        fm  = tensors["fm_output"].float()
        rmse = (ref - fm).pow(2).mean().sqrt().item()
        lines.append("-" * 85)
        lines.append(f"{'RMSE(ref, fm)':<20}  {rmse:.8f}")

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
    args = parser.parse_args()

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

    tensors, _ = load_call(args.npz, call_idx=args.call)

    if args.summary:
        summarize(tensors, emit=emit)
    else:
        for name, t in tensors.items():
            emit(f"\n=== {name}  shape={tuple(t.shape)} ===")
            emit(str(t))

        if "reference_output" in tensors and "fm_output" in tensors:
            ref  = tensors["reference_output"].float()
            fm   = tensors["fm_output"].float()
            rmse = (ref - fm).pow(2).mean().sqrt().item()
            emit(f"\nRMSE(reference_output, fm_output) = {rmse:.8f}")

    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
