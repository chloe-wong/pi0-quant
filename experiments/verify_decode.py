#!/usr/bin/env python3
"""
verify_decode.py — Sanity-check that decode_npz.py's bit-reinterpretation
is lossless for both bf16 (stored as int16) and fp8_e4m3 (stored as uint8).

Usage:
    uv run experiments/verify_decode.py

Exit 0 on all PASS, exit 1 on any FAIL.
"""

import sys
import tempfile
import numpy as np
import torch

# Import the encode/decode functions under test
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from pi0_inout.matmul_io_store import _to_bf16_numpy, _to_uint8_numpy
from experiments.decode_npz import decode_bf16, decode_fp8_e4m3

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures = []


def check(name: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    msg = f"[{tag}] {name}"
    if not ok and detail:
        msg += f"\n       {detail}"
    print(msg)
    if not ok:
        failures.append(name)


# ── BF16 roundtrip ────────────────────────────────────────────────────────────

torch.manual_seed(42)
x_rand = torch.randn(4, 8).bfloat16()

# Append edge cases as a flat vector, then reshape to 2D
edge = torch.tensor(
    [0.0, -0.0,
     float("inf"), float("-inf"), float("nan"),
     torch.finfo(torch.bfloat16).max,
     torch.finfo(torch.bfloat16).tiny],
    dtype=torch.bfloat16,
)
x_bf16 = torch.cat([x_rand.flatten(), edge]).reshape(-1, 1)  # [39, 1]

print(f"\n=== bf16 input  shape={tuple(x_bf16.shape)} ===")
print(x_bf16)

# dtype check
arr = _to_bf16_numpy(x_bf16)
check("bf16 encoded dtype is int16",  arr.dtype == np.int16,  f"got {arr.dtype}")
check("bf16 encoded itemsize is 2",   arr.itemsize == 2,      f"got {arr.itemsize}")
check("bf16 encoded shape matches",   arr.shape == x_bf16.shape, f"{arr.shape} vs {x_bf16.shape}")

print(f"\n=== bf16 encoded (int16 raw bits)  shape={arr.shape} ===")
print(arr)

# save / load roundtrip
with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
    np.savez(tmp.name, x=arr)
    loaded = np.load(tmp.name)["x"]

# decode_bf16 now returns native bf16 — compare bitwise as int16 (handles NaN exactly)
x_dec_bf16 = decode_bf16(loaded)

print(f"\n=== bf16 decoded (native bf16)  shape={tuple(x_dec_bf16.shape)} ===")
print(x_dec_bf16)

check("bf16 decoded dtype is bfloat16", x_dec_bf16.dtype == torch.bfloat16,
      f"got {x_dec_bf16.dtype}")
ok = torch.all(x_bf16.view(torch.int16) == x_dec_bf16.view(torch.int16)).item()
n_mismatch = (x_bf16.view(torch.int16) != x_dec_bf16.view(torch.int16)).sum().item()
check("bf16 roundtrip bitwise exact", ok, f"mismatched elements: {n_mismatch}")

# ── FP8 E4M3 roundtrip (scale_exp = 0) ──────────────────────────────────────

x_fp8 = x_rand.to(torch.float8_e4m3fn)   # cast from random bf16

print(f"\n=== fp8 input (float8_e4m3)  shape={tuple(x_fp8.shape)} ===")
print(x_fp8)

arr_fp8 = _to_uint8_numpy(x_fp8)
check("fp8 encoded dtype is uint8",  arr_fp8.dtype == np.uint8, f"got {arr_fp8.dtype}")
check("fp8 encoded itemsize is 1",   arr_fp8.itemsize == 1,     f"got {arr_fp8.itemsize}")
check("fp8 encoded shape matches",   arr_fp8.shape == x_fp8.shape, f"{arr_fp8.shape} vs {x_fp8.shape}")

print(f"\n=== fp8 encoded (uint8 raw bits)  shape={arr_fp8.shape} ===")
print(arr_fp8)

scale_exp = 0
with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
    np.savez(tmp.name, x=arr_fp8)
    loaded_fp8 = np.load(tmp.name)["x"]

# decode_fp8_e4m3 now returns (fp8_tensor, scale_exp)
x_dec_fp8, dec_scale_exp = decode_fp8_e4m3(loaded_fp8, scale_exp)

print(f"\n=== fp8 decoded (native fp8, scale_exp={dec_scale_exp})  shape={tuple(x_dec_fp8.shape)} ===")
print(x_dec_fp8)

check("fp8 decoded dtype is float8_e4m3fn", x_dec_fp8.dtype == torch.float8_e4m3fn,
      f"got {x_dec_fp8.dtype}")
check("fp8 scale_exp passed through", dec_scale_exp == scale_exp,
      f"got {dec_scale_exp}, expected {scale_exp}")
ok = torch.all(x_fp8.view(torch.uint8) == x_dec_fp8.view(torch.uint8)).item()
n_mismatch = (x_fp8.view(torch.uint8) != x_dec_fp8.view(torch.uint8)).sum().item()
check("fp8 roundtrip bitwise exact (scale_exp=0)", ok,
      f"mismatched elements: {n_mismatch}")

# ── FP8 scale exponent passthrough tests ─────────────────────────────────────

for exp in (3, -4):
    _, returned_exp = decode_fp8_e4m3(arr_fp8, exp)
    check(f"fp8 scale_exp passed through correctly (scale_exp={exp})",
          returned_exp == exp, f"got {returned_exp}")
    # Verify dequantized values are correct using the returned scale
    x_dec_t, _ = decode_fp8_e4m3(arr_fp8, exp)
    dequant = x_dec_t.float() * (2.0 ** exp)
    expected = x_fp8.float() * (2.0 ** exp)
    ok = torch.all(dequant == expected).item()
    check(f"fp8 dequantized values correct (scale_exp={exp})", ok,
          f"max abs diff: {(dequant - expected).abs().max().item()}")

# ── Summary ───────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"{len(failures)} test(s) FAILED: {failures}")
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)
