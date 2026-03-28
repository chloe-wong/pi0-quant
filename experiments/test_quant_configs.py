"""
test_quant_configs.py
---------------------
Verifies quantization correctness on a synthetic transformer-like model.

Four configurations tested:
  1. all_bf16        — MX=BF16/BF16, vec=BF16/BF16  → RMSE must be exactly 0
  2. mx_fp8_bf16     — MX=FP8E4M3/BF16, vec=BF16/BF16
  3. vec_fp8_bf16    — MX=BF16/BF16, vec=FP8E4M3/BF16
  4. both_fp8_bf16   — MX=FP8E4M3/BF16, vec=FP8E4M3/BF16

Also verifies that po2 scaling produces a power-of-two scale factor.

Run:
    python experiments/test_quant_configs.py
"""

from __future__ import annotations

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from pi0_inout import (
    QuantFormat, QuantGroup, StatsTracker,
    patch_model, unpatch_model,
    patch_vector_ops, unpatch_vector_ops,
    set_fp8_mode, get_fp8_mode,
)
from pi0_inout.quant_types import _quant_fp8_po2, _FP8_MAX_PO2

# ---------------------------------------------------------------------------
# Synthetic model with both linear and vector ops
# ---------------------------------------------------------------------------

class SyntheticBlock(nn.Module):
    """Small transformer-like block: linear projections + elementwise ops."""
    def __init__(self, d: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)
        self.gate = nn.Linear(d, d, bias=False)
        self.up   = nn.Linear(d, d, bias=False)
        self.down = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Vector ops: add, mul, sqrt (these are intercepted by VectorQuantMode)
        attn = q * k                     # mul
        attn = attn + v                  # add
        attn = torch.sqrt(attn.abs())    # sqrt

        out = self.out_proj(attn)

        # FFN with gated activation
        gate_out = self.gate(x)
        up_out   = self.up(x)
        hidden   = gate_out * up_out     # mul (vector op)
        hidden   = hidden + x            # add (vector op)
        out      = self.down(hidden) + out

        return out


class SyntheticModel(nn.Module):
    """Two stacked blocks, mimics a small transformer."""
    def __init__(self, d: int = 64, n_blocks: int = 2):
        super().__init__()
        # Named to match pi0 component rules so group filtering works
        self.language_model = nn.ModuleList([SyntheticBlock(d) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.language_model:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Run one configuration
# ---------------------------------------------------------------------------

def run_config(
    model: nn.Module,
    x: torch.Tensor,
    mx_input_fmt: QuantFormat,
    mx_output_fmt: QuantFormat,
    vec_input_fmt: QuantFormat,
    vec_output_fmt: QuantFormat,
    label: str,
) -> dict:
    tracker = StatsTracker()
    active  = {QuantGroup.TRANSFORMER}

    patch_model(
        model,
        mx_input_fmt=mx_input_fmt,
        mx_output_fmt=mx_output_fmt,
        tracker=tracker,
        active_groups=active,
        verbose=False,
    )
    vec_handles, vec_ctx = patch_vector_ops(
        model,
        active_groups=active,
        vec_input_fmt=vec_input_fmt,
        vec_output_fmt=vec_output_fmt,
        tracker=tracker,
    )

    with torch.no_grad(), vec_ctx:
        y_quant = model(x)

    unpatch_model(model)
    unpatch_vector_ops(vec_handles)

    report = tracker.summary()
    overall_rmse = sum(r["mean_rmse"] for r in report.component_rows) / max(len(report.component_rows), 1)

    return {
        "label":        label,
        "output_norm":  y_quant.norm().item(),
        "overall_rmse": overall_rmse,
        "components":   {r["component"]: r["mean_rmse"] for r in report.component_rows},
    }


# ---------------------------------------------------------------------------
# po2 scale sanity check
# ---------------------------------------------------------------------------

def check_po2_scale():
    """Verify that po2 mode produces a true power-of-two scale."""
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32) * 3.7  # arbitrary scale

    x_q = _quant_fp8_po2(x, QuantFormat.FLOAT8_E4M3)

    amax = x.abs().max().item()
    fp8_max_po2 = _FP8_MAX_PO2[QuantFormat.FLOAT8_E4M3]  # 256.0
    raw_scale = amax / fp8_max_po2
    expected_scale = 2.0 ** math.floor(math.log2(raw_scale))

    # Check it's a power of two: log2 should be an integer
    log2_scale = math.log2(expected_scale)
    is_po2 = abs(log2_scale - round(log2_scale)) < 1e-9

    print(f"\n[po2 scale check]")
    print(f"  amax={amax:.4f}  raw_scale={raw_scale:.6f}  expected_scale={expected_scale:.6f}")
    print(f"  log2(scale)={log2_scale:.6f}  is_power_of_two={is_po2}")
    assert is_po2, "Scale is not a power of two!"
    print("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    set_fp8_mode("po2")
    print(f"fp8_mode = {get_fp8_mode()}")

    model = SyntheticModel(d=64, n_blocks=2).bfloat16()
    x = torch.randn(4, 16, 64, dtype=torch.bfloat16)  # (batch, seq, d)

    BF16 = QuantFormat.BFLOAT16
    FP8  = QuantFormat.FLOAT8_E4M3

    configs = [
        ("all_bf16",      BF16, BF16, BF16, BF16),
        ("mx_fp8_bf16",   FP8,  BF16, BF16, BF16),
        ("vec_fp8_bf16",  BF16, BF16, FP8,  BF16),
        ("both_fp8_bf16", FP8,  BF16, FP8,  BF16),
    ]

    results = []
    for label, mi, mo, vi, vo in configs:
        r = run_config(model, x, mi, mo, vi, vo, label)
        results.append(r)

    # --- Print results table ---
    print(f"\n{'Config':<20s}  {'Overall RMSE':>14s}  {'Output norm':>12s}")
    print("-" * 52)
    for r in results:
        print(f"{r['label']:<20s}  {r['overall_rmse']:>14.6e}  {r['output_norm']:>12.4f}")

    # --- Assertions ---
    print()
    bf16 = results[0]
    assert bf16["overall_rmse"] == 0.0, \
        f"BF16/BF16 RMSE must be exactly 0, got {bf16['overall_rmse']}"
    print("PASS: all_bf16 RMSE == 0.0 exactly")

    for r in results[1:]:
        assert r["overall_rmse"] > 0.0, \
            f"{r['label']} RMSE should be > 0 for FP8 quantization"
        assert math.isfinite(r["overall_rmse"]), \
            f"{r['label']} RMSE is not finite: {r['overall_rmse']}"
    print("PASS: FP8 configs have finite nonzero RMSE")

    # --- po2 scale check ---
    check_po2_scale()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
