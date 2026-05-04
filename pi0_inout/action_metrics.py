"""
action_metrics.py
-----------------
Rich action-level metrics for quantization evaluation.

The existing eval harness computes a single scalar action RMSE.  This module
breaks that down into per-timestep, per-dimension, per-sample, and percentile
statistics so you can concretely say whether injected error is "okay" —
not just "how much" error there is.

Usage:
    metrics = compute_action_metrics(reference_actions, quantized_actions)
    metrics.print_report()

    thresholds = ActionThresholds(max_rmse=0.05, max_rel_rmse=0.02)
    verdict = metrics.check(thresholds)
    print(verdict)          # PASS or FAIL with details

Baseline variance (model's intrinsic noise floor):
    from pi0_inout import compute_baseline_variance
    baseline = compute_baseline_variance(model, observations, infer_fn, n_seeds=10)
    baseline.print_report()

    # Auto-generate thresholds: quantization error must be < 1x the model's own noise
    thresholds = baseline.to_thresholds(k=1.0)
    verdict = metrics.check(thresholds)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Thresholds (user-configurable acceptance criteria)
# ---------------------------------------------------------------------------

@dataclass
class ActionThresholds:
    """
    Acceptance criteria for action-level error.

    Set any field to None to skip that check.  All error values use the same
    units as the action space (typically normalized to roughly [-1, 1]).

    Example — typical starting point for a droid joint-position policy:
        ActionThresholds(
            max_rmse=0.05,
            max_rel_rmse=0.02,
            max_p99_abs_error=0.15,
            max_per_dim_rmse=0.10,
            max_per_step_rmse=0.10,
        )
    """
    max_rmse: Optional[float] = None
    max_rel_rmse: Optional[float] = None
    max_p99_abs_error: Optional[float] = None
    max_per_dim_rmse: Optional[float] = None
    max_per_step_rmse: Optional[float] = None


@dataclass
class CheckResult:
    """Outcome of ActionMetrics.check() against thresholds."""
    passed: bool
    failures: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.passed:
            return "PASS — all thresholds met"
        lines = ["FAIL — thresholds exceeded:"]
        for f in self.failures:
            lines.append(f"  - {f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

@dataclass
class ActionMetrics:
    """
    Comprehensive action-level error metrics computed from reference vs
    quantized action predictions.

    Action tensors are assumed to be shape [action_horizon, action_dim]
    (batch dim squeezed).
    """
    # Overall
    overall_rmse: float
    overall_rel_rmse: float              # rmse / rms(reference)
    overall_mae: float                   # mean absolute error

    # Per-timestep: shape [action_horizon]
    per_step_rmse: List[float]
    per_step_rel_rmse: List[float]

    # Per-dimension: shape [action_dim]
    per_dim_rmse: List[float]
    per_dim_rel_rmse: List[float]

    # Per-sample: shape [n_observations]
    per_sample_rmse: List[float]

    # Percentiles of per-element |error|
    percentiles: Dict[str, float]        # {"p50": ..., "p90": ..., "p95": ..., "p99": ...}

    # Worst case
    max_abs_error: float
    max_abs_error_location: Tuple[int, int, int]  # (sample_idx, step_idx, dim_idx)

    # Metadata
    n_observations: int
    action_horizon: int
    action_dim: int
    dim_labels: Optional[List[str]] = None

    # -----------------------------------------------------------------------
    # Pass/fail
    # -----------------------------------------------------------------------

    def check(self, thresholds: ActionThresholds) -> CheckResult:
        """Check metrics against acceptance thresholds."""
        failures = []

        if thresholds.max_rmse is not None and self.overall_rmse > thresholds.max_rmse:
            failures.append(
                f"overall_rmse={self.overall_rmse:.4e} > max_rmse={thresholds.max_rmse:.4e}"
            )
        if thresholds.max_rel_rmse is not None and self.overall_rel_rmse > thresholds.max_rel_rmse:
            failures.append(
                f"overall_rel_rmse={self.overall_rel_rmse:.4e} > max_rel_rmse={thresholds.max_rel_rmse:.4e}"
            )
        if thresholds.max_p99_abs_error is not None and self.percentiles["p99"] > thresholds.max_p99_abs_error:
            failures.append(
                f"p99_abs_error={self.percentiles['p99']:.4e} > max_p99_abs_error={thresholds.max_p99_abs_error:.4e}"
            )
        if thresholds.max_per_dim_rmse is not None:
            worst_dim = max(self.per_dim_rmse)
            worst_idx = self.per_dim_rmse.index(worst_dim)
            if worst_dim > thresholds.max_per_dim_rmse:
                label = self.dim_labels[worst_idx] if self.dim_labels else f"dim_{worst_idx}"
                failures.append(
                    f"per_dim_rmse[{label}]={worst_dim:.4e} > max_per_dim_rmse={thresholds.max_per_dim_rmse:.4e}"
                )
        if thresholds.max_per_step_rmse is not None:
            worst_step = max(self.per_step_rmse)
            worst_idx = self.per_step_rmse.index(worst_step)
            if worst_step > thresholds.max_per_step_rmse:
                failures.append(
                    f"per_step_rmse[t={worst_idx}]={worst_step:.4e} > max_per_step_rmse={thresholds.max_per_step_rmse:.4e}"
                )

        return CheckResult(passed=len(failures) == 0, failures=failures)

    # -----------------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------------

    def print_report(self, show_all_dims: bool = False, show_all_steps: bool = False) -> None:
        """Pretty-print a multi-section action metrics report."""
        print(f"\n{'='*64}")
        print("ACTION METRICS REPORT")
        print(f"{'='*64}")
        print(f"  observations:    {self.n_observations}")
        print(f"  action_horizon:  {self.action_horizon}")
        print(f"  action_dim:      {self.action_dim}")

        # --- Overall ---------------------------------------------------------
        print(f"\n--- Overall ---")
        print(f"  RMSE:           {self.overall_rmse:.6e}")
        print(f"  Relative RMSE:  {self.overall_rel_rmse:.6e}  ({self.overall_rel_rmse * 100:.4f}%)")
        print(f"  MAE:            {self.overall_mae:.6e}")
        print(f"  Max |error|:    {self.max_abs_error:.6e}  "
              f"@ (sample={self.max_abs_error_location[0]}, "
              f"step={self.max_abs_error_location[1]}, "
              f"dim={self.max_abs_error_location[2]})")

        # --- Percentiles -----------------------------------------------------
        print(f"\n--- Error Percentiles (|error|) ---")
        for k in ("p50", "p90", "p95", "p99"):
            print(f"  {k:>4s}:  {self.percentiles[k]:.6e}")

        # --- Per-timestep ----------------------------------------------------
        print(f"\n--- Per-Timestep RMSE ---")
        n_show = self.action_horizon if show_all_steps else min(self.action_horizon, 5)
        for t in range(n_show):
            rel = self.per_step_rel_rmse[t]
            print(f"  t={t:<3d}  rmse={self.per_step_rmse[t]:.6e}  rel={rel:.6e}")
        if n_show < self.action_horizon:
            print(f"  ... ({self.action_horizon - n_show} more steps, "
                  f"max rmse={max(self.per_step_rmse):.6e} at t={self.per_step_rmse.index(max(self.per_step_rmse))})")

        # --- Per-dimension ---------------------------------------------------
        print(f"\n--- Per-Dimension RMSE ---")
        n_show_d = self.action_dim if show_all_dims else min(self.action_dim, 8)
        # Show the worst dims first
        indexed = sorted(enumerate(self.per_dim_rmse), key=lambda x: -x[1])
        for rank, (idx, rmse) in enumerate(indexed[:n_show_d]):
            label = self.dim_labels[idx] if self.dim_labels else f"dim_{idx}"
            rel = self.per_dim_rel_rmse[idx]
            print(f"  #{rank+1:<2d}  {label:<12s}  rmse={rmse:.6e}  rel={rel:.6e}")
        if n_show_d < self.action_dim:
            print(f"  ... ({self.action_dim - n_show_d} more dims)")

        # --- Per-sample (just summary) ---------------------------------------
        print(f"\n--- Per-Sample RMSE ---")
        if self.per_sample_rmse:
            s = sorted(self.per_sample_rmse)
            print(f"  min:     {s[0]:.6e}")
            print(f"  median:  {s[len(s)//2]:.6e}")
            print(f"  max:     {s[-1]:.6e}")
            worst_idx = self.per_sample_rmse.index(s[-1])
            print(f"  worst sample: #{worst_idx}")

        print(f"{'='*64}")

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "overall_rmse":            self.overall_rmse,
            "overall_rel_rmse":        self.overall_rel_rmse,
            "overall_mae":             self.overall_mae,
            "per_step_rmse":           self.per_step_rmse,
            "per_step_rel_rmse":       self.per_step_rel_rmse,
            "per_dim_rmse":            self.per_dim_rmse,
            "per_dim_rel_rmse":        self.per_dim_rel_rmse,
            "per_sample_rmse":         self.per_sample_rmse,
            "percentiles":             self.percentiles,
            "max_abs_error":           self.max_abs_error,
            "max_abs_error_location":  list(self.max_abs_error_location),
            "n_observations":          self.n_observations,
            "action_horizon":          self.action_horizon,
            "action_dim":              self.action_dim,
            "dim_labels":              self.dim_labels,
        }


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_action_metrics(
    reference: List[torch.Tensor],
    quantized: List[torch.Tensor],
    dim_labels: Optional[List[str]] = None,
) -> ActionMetrics:
    """
    Compute comprehensive action-level metrics from paired reference/quantized
    action tensors.

    Args:
        reference:  List of fp32 action tensors, each [action_horizon, action_dim]
                    (or [1, action_horizon, action_dim] — leading batch dim is squeezed).
        quantized:  Corresponding quantized action tensors, same shapes.
        dim_labels: Optional human-readable names for each action dimension
                    (e.g. ["x", "y", "z", "rx", "ry", "rz", "gripper"]).

    Returns:
        ActionMetrics with all breakdowns.
    """
    assert len(reference) == len(quantized), (
        f"reference length {len(reference)} != quantized length {len(quantized)}"
    )
    assert len(reference) > 0, "Need at least one observation"

    # Normalize to [action_horizon, action_dim] and stack
    refs, quants = [], []
    for r, q in zip(reference, quantized):
        r_f = r.float().cpu()
        q_f = q.float().cpu()
        # Squeeze leading batch dim if present
        if r_f.dim() == 3 and r_f.shape[0] == 1:
            r_f = r_f.squeeze(0)
        if q_f.dim() == 3 and q_f.shape[0] == 1:
            q_f = q_f.squeeze(0)
        assert r_f.shape == q_f.shape, f"Shape mismatch: {r_f.shape} vs {q_f.shape}"
        refs.append(r_f)
        quants.append(q_f)

    action_horizon = refs[0].shape[0]
    action_dim = refs[0].shape[1] if refs[0].dim() > 1 else 1
    n_obs = len(refs)

    # Stack into [n_obs, action_horizon, action_dim]
    ref_stack = torch.stack(refs)     # [N, H, D]
    q_stack   = torch.stack(quants)   # [N, H, D]
    errors    = ref_stack - q_stack   # [N, H, D]
    abs_errors = errors.abs()

    # --- Overall -------------------------------------------------------------
    overall_mse = errors.pow(2).mean().item()
    overall_rmse = math.sqrt(max(overall_mse, 0.0))

    ref_ms = ref_stack.pow(2).mean().item()
    overall_rel_rmse = (
        math.sqrt(max(overall_mse, 0.0) / ref_ms) if ref_ms > 0 else float("nan")
    )

    overall_mae = abs_errors.mean().item()

    # --- Per-timestep: mean over (n_obs, action_dim) -------------------------
    # errors: [N, H, D] -> mse per step: [H]
    per_step_mse = errors.pow(2).mean(dim=(0, 2))       # [H]
    per_step_rmse = per_step_mse.sqrt().tolist()

    ref_step_ms = ref_stack.pow(2).mean(dim=(0, 2))     # [H]
    per_step_rel_rmse = [
        math.sqrt(max(m, 0.0) / max(r, 1e-30))
        for m, r in zip(per_step_mse.tolist(), ref_step_ms.tolist())
    ]

    # --- Per-dimension: mean over (n_obs, action_horizon) --------------------
    per_dim_mse = errors.pow(2).mean(dim=(0, 1))        # [D]
    per_dim_rmse = per_dim_mse.sqrt().tolist()

    ref_dim_ms = ref_stack.pow(2).mean(dim=(0, 1))      # [D]
    per_dim_rel_rmse = [
        math.sqrt(max(m, 0.0) / max(r, 1e-30))
        for m, r in zip(per_dim_mse.tolist(), ref_dim_ms.tolist())
    ]

    # --- Per-sample: mean over (action_horizon, action_dim) ------------------
    per_sample_mse = errors.pow(2).mean(dim=(1, 2))     # [N]
    per_sample_rmse = per_sample_mse.sqrt().tolist()

    # --- Percentiles of |error| ----------------------------------------------
    flat_abs = abs_errors.flatten()
    percentiles = {}
    for name, q in [("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]:
        percentiles[name] = torch.quantile(flat_abs, q).item()

    # --- Max absolute error --------------------------------------------------
    max_abs_error = flat_abs.max().item()
    max_idx = flat_abs.argmax().item()
    # Unravel: flat index -> (sample, step, dim)
    sample_idx = max_idx // (action_horizon * action_dim)
    remainder  = max_idx %  (action_horizon * action_dim)
    step_idx   = remainder // action_dim
    dim_idx    = remainder %  action_dim
    max_abs_error_location = (sample_idx, step_idx, dim_idx)

    return ActionMetrics(
        overall_rmse=overall_rmse,
        overall_rel_rmse=overall_rel_rmse,
        overall_mae=overall_mae,
        per_step_rmse=per_step_rmse,
        per_step_rel_rmse=per_step_rel_rmse,
        per_dim_rmse=per_dim_rmse,
        per_dim_rel_rmse=per_dim_rel_rmse,
        per_sample_rmse=per_sample_rmse,
        percentiles=percentiles,
        max_abs_error=max_abs_error,
        max_abs_error_location=max_abs_error_location,
        n_observations=n_obs,
        action_horizon=action_horizon,
        action_dim=action_dim,
        dim_labels=dim_labels,
    )


# ---------------------------------------------------------------------------
# Baseline variance — FP32 model's intrinsic noise floor
# ---------------------------------------------------------------------------

@dataclass
class BaselineVariance:
    """
    Measures the FP32 model's own action variance across diffusion seeds.

    The idea: run the same observation N times with different
    torch.manual_seed() values.  The resulting spread in actions is the
    model's intrinsic noise — quantization error below this floor is
    indistinguishable from the model's own randomness.

    Use to_thresholds(k) to auto-generate ActionThresholds where each
    limit is k * the corresponding baseline statistic.
    """
    # Overall std of actions across seeds (averaged over all elements)
    overall_std: float

    # Per-timestep: std at each step in the horizon, averaged over dims and obs
    per_step_std: List[float]           # [action_horizon]

    # Per-dimension: std for each action dim, averaged over steps and obs
    per_dim_std: List[float]            # [action_dim]

    # Per-sample: std for each observation, averaged over steps and dims
    per_sample_std: List[float]         # [n_observations]

    # Percentiles of per-element std (across all obs/step/dim positions)
    percentiles: Dict[str, float]       # {"p50": ..., "p90": ..., "p95": ..., "p99": ...}

    # Metadata
    n_observations: int
    n_seeds: int
    action_horizon: int
    action_dim: int
    dim_labels: Optional[List[str]] = None

    def to_thresholds(self, k: float = 1.0) -> ActionThresholds:
        """
        Generate ActionThresholds from baseline variance.

        k is a multiplier:
            k=1.0 — quant error must be smaller than the model's own noise (strict)
            k=2.0 — quant error can be up to 2x the model's noise (moderate)
            k=0.5 — quant error must be half the model's noise (very strict)
        """
        return ActionThresholds(
            max_rmse=k * self.overall_std,
            max_per_step_rmse=k * max(self.per_step_std),
            max_per_dim_rmse=k * max(self.per_dim_std),
            max_p99_abs_error=k * self.percentiles["p99"],
        )

    def print_report(self) -> None:
        print(f"\n{'='*64}")
        print("BASELINE VARIANCE REPORT (FP32 model noise floor)")
        print(f"{'='*64}")
        print(f"  observations:    {self.n_observations}")
        print(f"  seeds per obs:   {self.n_seeds}")
        print(f"  action_horizon:  {self.action_horizon}")
        print(f"  action_dim:      {self.action_dim}")

        print(f"\n--- Overall ---")
        print(f"  std:  {self.overall_std:.6e}")

        print(f"\n--- Seed Variance Percentiles (per-element std) ---")
        for k in ("p50", "p90", "p95", "p99"):
            print(f"  {k:>4s}:  {self.percentiles[k]:.6e}")

        print(f"\n--- Per-Timestep std ---")
        n_show = min(self.action_horizon, 5)
        for t in range(n_show):
            print(f"  t={t:<3d}  std={self.per_step_std[t]:.6e}")
        if n_show < self.action_horizon:
            print(f"  ... ({self.action_horizon - n_show} more, "
                  f"max={max(self.per_step_std):.6e} at t={self.per_step_std.index(max(self.per_step_std))})")

        print(f"\n--- Per-Dimension std (worst first) ---")
        n_show_d = min(self.action_dim, 8)
        indexed = sorted(enumerate(self.per_dim_std), key=lambda x: -x[1])
        for rank, (idx, std) in enumerate(indexed[:n_show_d]):
            label = self.dim_labels[idx] if self.dim_labels else f"dim_{idx}"
            print(f"  #{rank+1:<2d}  {label:<12s}  std={std:.6e}")
        if n_show_d < self.action_dim:
            print(f"  ... ({self.action_dim - n_show_d} more dims)")

        print(f"\n--- Suggested thresholds ---")
        for k_val, label in [(0.5, "strict"), (1.0, "moderate"), (2.0, "lenient")]:
            t = self.to_thresholds(k_val)
            print(f"  k={k_val} ({label}):  max_rmse={t.max_rmse:.4e}  "
                  f"max_p99={t.max_p99_abs_error:.4e}")

        print(f"{'='*64}")

    def to_dict(self) -> dict:
        return {
            "overall_std":      self.overall_std,
            "per_step_std":     self.per_step_std,
            "per_dim_std":      self.per_dim_std,
            "per_sample_std":   self.per_sample_std,
            "percentiles":      self.percentiles,
            "n_observations":   self.n_observations,
            "n_seeds":          self.n_seeds,
            "action_horizon":   self.action_horizon,
            "action_dim":       self.action_dim,
            "dim_labels":       self.dim_labels,
        }


def compute_baseline_variance_from_actions(
    actions_per_obs: List[List[torch.Tensor]],
    dim_labels: Optional[List[str]] = None,
) -> BaselineVariance:
    """
    Compute baseline variance from pre-collected multi-seed action tensors.

    Args:
        actions_per_obs: Outer list is per observation, inner list is per seed.
                         Each tensor is [action_horizon, action_dim] (or [1, H, D]).
        dim_labels:      Optional human-readable names for each action dimension.

    Returns:
        BaselineVariance with all breakdowns.
    """
    n_obs = len(actions_per_obs)
    assert n_obs > 0, "Need at least one observation"
    n_seeds = len(actions_per_obs[0])
    assert n_seeds >= 2, f"Need at least 2 seeds for variance, got {n_seeds}"

    # Normalize shapes and stack: [n_obs, n_seeds, H, D]
    all_actions = []
    for obs_actions in actions_per_obs:
        assert len(obs_actions) == n_seeds, (
            f"All observations must have same number of seeds, "
            f"got {len(obs_actions)} vs {n_seeds}"
        )
        seed_tensors = []
        for a in obs_actions:
            a_f = a.float().cpu()
            if a_f.dim() == 3 and a_f.shape[0] == 1:
                a_f = a_f.squeeze(0)
            seed_tensors.append(a_f)
        all_actions.append(torch.stack(seed_tensors))  # [n_seeds, H, D]

    stacked = torch.stack(all_actions)  # [n_obs, n_seeds, H, D]
    action_horizon = stacked.shape[2]
    action_dim = stacked.shape[3]

    # Std across seeds (dim=1), giving [n_obs, H, D]
    per_element_std = stacked.std(dim=1)  # unbiased (Bessel's correction)

    # Overall: mean of per-element std
    overall_std = per_element_std.mean().item()

    # Per-timestep: mean over (n_obs, D)
    per_step_std = per_element_std.mean(dim=(0, 2)).tolist()  # [H]

    # Per-dimension: mean over (n_obs, H)
    per_dim_std = per_element_std.mean(dim=(0, 1)).tolist()   # [D]

    # Per-sample: mean over (H, D)
    per_sample_std = per_element_std.mean(dim=(1, 2)).tolist()  # [n_obs]

    # Percentiles of per-element std
    flat_std = per_element_std.flatten()
    percentiles = {}
    for name, q in [("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]:
        percentiles[name] = torch.quantile(flat_std, q).item()

    return BaselineVariance(
        overall_std=overall_std,
        per_step_std=per_step_std,
        per_dim_std=per_dim_std,
        per_sample_std=per_sample_std,
        percentiles=percentiles,
        n_observations=n_obs,
        n_seeds=n_seeds,
        action_horizon=action_horizon,
        action_dim=action_dim,
        dim_labels=dim_labels,
    )
