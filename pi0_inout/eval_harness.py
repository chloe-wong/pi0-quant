"""
eval_harness.py
---------------
Benchmark-agnostic evaluation harness for quantization testing.

Usage pattern
-------------
1. Load your Pi0Pytorch model (however you normally do it).
2. Create a QuantConfig describing the input/output formats to test.
3. Call run_quantization_eval(model, observations, reference_actions, config).
4. Inspect the returned EvalResult for RMSE stats.

The harness:
- Runs the unquantized model to collect fp32 reference activations per layer
  and fp32 reference action outputs.
- Patches the model with QuantLinear layers.
- Runs the quantized model on the same inputs.
- Computes action RMSE (the most robotics-relevant metric) and per-layer RMSE.
- Unpatches the model, restoring it to its original state.

If you already have reference_actions from the unquantized model (e.g., from a
prior run), you can pass them in directly to avoid the second forward pass.

"observations" format
---------------------
This harness is intentionally benchmark-agnostic.  An "observation" is any
dict (or any type) that your model's forward() / sample_actions() accepts.
You pass a callable `infer_fn` that takes (model, observation) → actions.

For Pi0Pytorch:
    def infer_fn(model, obs):
        with torch.no_grad():
            return model.sample_actions(obs)

For the WebSocket serving path, you don't need this harness at all — just
patch the model before starting the server and inspect tracker.summary() after
the episode.
"""

from __future__ import annotations

import copy
import time
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .quant_types import QuantFormat
from .model_patcher import patch_model, patch_model_matvec, unpatch_model, count_layers
from .stats_tracker import StatsTracker, StatsReport, Component
from .ulp_noise import UlpNoiseConfig, UlpNoiseMode


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QuantConfig:
    """
    Describes one quantization experiment.

    input_fmt:   Format for both activations and weights entering the matmul.
    output_fmt:  Format for the matmul output.
    skip_components: Components to leave at full precision (useful for ablations).
    label:       Human-readable name for this configuration.
    """
    input_fmt:  QuantFormat = QuantFormat.FLOAT32
    output_fmt: QuantFormat = QuantFormat.FLOAT32
    ulp_noise: Optional[UlpNoiseConfig] = None
    skip_components: frozenset[Component] = field(default_factory=frozenset)
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"in={self.input_fmt.value}_out={self.output_fmt.value}"


@dataclass
class MatVecQuantConfig:
    """
    Quantization config with separate matrix/vector IO formats.

    Constraint:
        vector_in_fmt == matrix_out_fmt (enforced by patch_model_matvec).
    """
    matrix_in_fmt: QuantFormat = QuantFormat.FLOAT32
    matrix_out_fmt: QuantFormat = QuantFormat.FLOAT32
    vector_out_fmt: QuantFormat = QuantFormat.FLOAT32
    ulp_noise: Optional[UlpNoiseConfig] = None
    skip_components: frozenset[Component] = field(default_factory=frozenset)
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = (
                f"mat_in={self.matrix_in_fmt.value}_mat_out={self.matrix_out_fmt.value}"
                f"_vec_out={self.vector_out_fmt.value}"
            )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    config: QuantConfig
    # Action-level RMSE across all evaluated observations
    action_rmse: float
    action_rmse_per_component: Dict[str, float]  # keyed by component name
    # Optional: maximum absolute action error in degrees (if interpreted as radians)
    max_abs_error_deg: Optional[float] = None
    # Per-layer and per-component stats from the StatsTracker
    stats_report: StatsReport
    # Timing
    n_observations: int
    inference_time_s: float

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"Config: {self.config.label}")
        print(f"  Observations: {self.n_observations}")
        print(f"  Inference time: {self.inference_time_s:.2f}s")
        print(f"\n  Action RMSE (overall): {self.action_rmse:.6e}")
        print(f"\n  Action RMSE by component of origin:")
        for comp, rmse in self.action_rmse_per_component.items():
            print(f"    {comp:16s}: {rmse:.6e}")
        print()
        self.stats_report.print(show_layers=False)

    def to_dict(self) -> dict:
        return {
            "config":           self.config.label,
            "input_fmt":        self.config.input_fmt.value,
            "output_fmt":       self.config.output_fmt.value,
            "action_rmse":      self.action_rmse,
            "action_rmse_by_component": self.action_rmse_per_component,
            "max_abs_error_deg": self.max_abs_error_deg,
            "n_observations":   self.n_observations,
            "inference_time_s": self.inference_time_s,
            "layers":           self.stats_report.layer_rows,
            "components":       self.stats_report.component_rows,
        }


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def run_quantization_eval(
    model: nn.Module,
    observations: List[Any],
    infer_fn: Callable[[nn.Module, Any], torch.Tensor],
    config: QuantConfig,
    reference_actions: Optional[List[torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> EvalResult:
    """
    Evaluate the effect of quantization on model action predictions.

    Args:
        model:             The Pi0Pytorch model.  WILL BE PATCHED IN-PLACE then
                           unpatched before returning (so original is restored).
        observations:      List of observations to evaluate on.
        infer_fn:          Callable (model, observation) → actions tensor.
                           Should run in torch.no_grad().
        config:            QuantConfig specifying the quantization experiment.
        reference_actions: Pre-computed fp32 reference actions.  If None, the
                           unpatched model is run first to collect them.
        device:            Torch device.  If None, inferred from model.
        verbose:           Print progress.

    Returns:
        EvalResult with full RMSE statistics.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # --- Step 1: collect fp32 reference actions (if not provided) -----------
    if reference_actions is None:
        if verbose:
            print(f"[eval] Collecting fp32 reference actions for {len(observations)} observations...")
        reference_actions = _collect_actions(model, observations, infer_fn)

    assert len(reference_actions) == len(observations), (
        f"reference_actions length {len(reference_actions)} != "
        f"observations length {len(observations)}"
    )

    # --- Step 2: patch model -------------------------------------------------
    tracker = StatsTracker()
    patch_model(
        model=model,
        input_fmt=config.input_fmt,
        output_fmt=config.output_fmt,
        tracker=tracker,
        ulp_noise=config.ulp_noise,
        skip_components=set(config.skip_components),
        verbose=False,
    )

    # --- Step 3: run quantized inference ------------------------------------
    if verbose:
        print(f"[eval] Running quantized inference [{config.label}]...")

    quant_actions = []
    t0 = time.perf_counter()
    for i, obs in enumerate(observations):
        with torch.no_grad():
            acts = infer_fn(model, obs)
        quant_actions.append(acts)
        if verbose and (i + 1) % max(1, len(observations) // 10) == 0:
            print(f"  {i+1}/{len(observations)}")
    elapsed = time.perf_counter() - t0

    # --- Step 4: compute action RMSE ----------------------------------------
    action_rmse = _compute_action_rmse(reference_actions, quant_actions)
    max_abs_error_deg = _compute_max_abs_error_deg(reference_actions, quant_actions)

    # Action RMSE split by which component contributes most is complex to
    # compute directly (would need feature attribution).  Instead, we report
    # the action RMSE alongside per-component layer RMSE and let the user
    # correlate them.  The action_rmse_per_component dict here means:
    # "the mean RMSE of layers in this component" — a proxy for sensitivity.
    comp_rows = tracker.summary().component_rows
    action_rmse_per_component = {
        row["component"]: row["mean_rmse"] for row in comp_rows
    }

    # --- Step 5: unpatch model ----------------------------------------------
    unpatch_model(model)

    return EvalResult(
        config=config,
        action_rmse=action_rmse,
        action_rmse_per_component=action_rmse_per_component,
        max_abs_error_deg=max_abs_error_deg,
        stats_report=tracker.summary(),
        n_observations=len(observations),
        inference_time_s=elapsed,
    )


def run_quantization_eval_matvec(
    model: nn.Module,
    observations: List[Any],
    infer_fn: Callable[[nn.Module, Any], torch.Tensor],
    config: MatVecQuantConfig,
    reference_actions: Optional[List[torch.Tensor]] = None,
    *,
    rmse_threshold: float = 0.4,
    max_deg_threshold: Optional[float] = 20.0,
    verbose: bool = True,
) -> dict:
    """
    Evaluate matrix/vector quantization + optional ULP noise injection.

    Returns a dict containing action RMSE and whether thresholds are violated.
    """
    if reference_actions is None:
        if verbose:
            print(f"[eval-matvec] Collecting fp32 reference actions for {len(observations)} observations...")
        reference_actions = _collect_actions(model, observations, infer_fn)

    tracker = StatsTracker()
    patch_model_matvec(
        model=model,
        matrix_in_fmt=config.matrix_in_fmt,
        matrix_out_fmt=config.matrix_out_fmt,
        vector_out_fmt=config.vector_out_fmt,
        tracker=tracker,
        ulp_noise=config.ulp_noise,
        skip_components=set(config.skip_components),
        verbose=False,
    )

    if verbose:
        print(f"[eval-matvec] Running quantized inference [{config.label}]...")
    quant_actions = []
    t0 = time.perf_counter()
    for obs in observations:
        with torch.no_grad():
            quant_actions.append(infer_fn(model, obs))
    elapsed = time.perf_counter() - t0

    action_rmse = _compute_action_rmse(reference_actions, quant_actions)
    max_abs_error_deg = _compute_max_abs_error_deg(reference_actions, quant_actions)

    comp_rows = tracker.summary().component_rows
    component_rmse = {row["component"]: row["mean_rmse"] for row in comp_rows}

    unpatch_model(model)

    violates_rmse = (not math.isnan(action_rmse)) and action_rmse >= rmse_threshold
    violates_deg = (
        max_deg_threshold is not None
        and max_abs_error_deg is not None
        and (not math.isnan(max_abs_error_deg))
        and max_abs_error_deg >= max_deg_threshold
    )

    return {
        "config": config.label,
        "matrix_in_fmt": config.matrix_in_fmt.value,
        "matrix_out_fmt": config.matrix_out_fmt.value,
        "vector_out_fmt": config.vector_out_fmt.value,
        "ulp_noise": None if config.ulp_noise is None else {
            "n_ulp": config.ulp_noise.n_ulp,
            "mode": config.ulp_noise.mode.value,
            "ulp_fmt": config.ulp_noise.ulp_fmt.value,
        },
        "action_rmse": action_rmse,
        "max_abs_error_deg": max_abs_error_deg,
        "violates_rmse": violates_rmse,
        "violates_deg": violates_deg,
        "component_rmse": component_rmse,
        "n_observations": len(observations),
        "inference_time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Sweep: evaluate multiple QuantConfigs in sequence
# ---------------------------------------------------------------------------

def run_sweep(
    model: nn.Module,
    observations: List[Any],
    infer_fn: Callable[[nn.Module, Any], torch.Tensor],
    configs: List[QuantConfig],
    verbose: bool = True,
) -> List[EvalResult]:
    """
    Run run_quantization_eval for each config in `configs`, collecting fp32
    reference actions once and reusing them across all configs.

    Args:
        model:       Model to evaluate. Modified in-place per config and restored.
        observations: List of observations.
        infer_fn:    Callable (model, obs) → actions.
        configs:     List of QuantConfig instances to sweep over.
        verbose:     Print progress.

    Returns:
        List of EvalResult, one per config.
    """
    if verbose:
        print(f"[sweep] Collecting fp32 reference for {len(observations)} observations...")
    reference_actions = _collect_actions(model, observations, infer_fn)

    results = []
    for i, cfg in enumerate(configs):
        if verbose:
            print(f"\n[sweep] Config {i+1}/{len(configs)}: {cfg.label}")
        result = run_quantization_eval(
            model=model,
            observations=observations,
            infer_fn=infer_fn,
            config=cfg,
            reference_actions=reference_actions,
            verbose=verbose,
        )
        results.append(result)
        if verbose:
            result.print_summary()

    return results


def default_sweep_configs(
    input_formats: Optional[List[QuantFormat]] = None,
    output_formats: Optional[List[QuantFormat]] = None,
) -> List[QuantConfig]:
    """
    Build the full 5x5 grid of (input_fmt × output_fmt) QuantConfigs.

    Default: 25 combinations of
        {FLOAT32, FP16, BF16, FP8-E4M3, FP8-E5M2} × same.

    FLOAT32 × FLOAT32 is the identity baseline (expected RMSE = 0).

    Args:
        input_formats:  Override the default input format list.
        output_formats: Override the default output format list.

    Returns:
        List of 25 QuantConfig instances (or len(input) × len(output) if overridden).
    """
    if input_formats is None:
        input_formats = [
            QuantFormat.FLOAT32,      # baseline — zero RMSE, use to verify correctness
            QuantFormat.FLOAT16,
            QuantFormat.BFLOAT16,
            QuantFormat.FLOAT8_E4M3,
            QuantFormat.FLOAT8_E5M2,
        ]
    if output_formats is None:
        output_formats = [
            QuantFormat.FLOAT32,
            QuantFormat.FLOAT16,
            QuantFormat.BFLOAT16,
            QuantFormat.FLOAT8_E4M3,
            QuantFormat.FLOAT8_E5M2,
        ]

    configs = []
    for inf in input_formats:
        for outf in output_formats:
            configs.append(QuantConfig(
                input_fmt=inf,
                output_fmt=outf,
                label=f"in={inf.value}_out={outf.value}",
            ))
    return configs


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

def results_to_dataframe(results: List[EvalResult]):
    """
    Flatten a list of EvalResult into a single summary DataFrame (pandas).
    Each row is one (config, component) pair with action_rmse and layer stats.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pip install pandas to use results_to_dataframe()")

    rows = []
    for r in results:
        base = {
            "config":           r.config.label,
            "input_fmt":        r.config.input_fmt.value,
            "output_fmt":       r.config.output_fmt.value,
            "action_rmse":      r.action_rmse,
            "max_abs_error_deg": r.max_abs_error_deg,
            "n_observations":   r.n_observations,
            "inference_time_s": r.inference_time_s,
        }
        for comp_row in r.stats_report.component_rows:
            rows.append({**base, **comp_row})

    return pd.DataFrame(rows)


def save_results(results: List[EvalResult], path: str) -> None:
    """Save a list of EvalResult to a JSON file."""
    import json
    data = [r.to_dict() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[save_results] Saved {len(results)} results to {path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_actions(
    model: nn.Module,
    observations: List[Any],
    infer_fn: Callable[[nn.Module, Any], torch.Tensor],
) -> List[torch.Tensor]:
    """Run model on all observations and return list of action tensors."""
    actions = []
    with torch.no_grad():
        for obs in observations:
            acts = infer_fn(model, obs)
            actions.append(acts.detach().cpu())
    return actions


def _compute_action_rmse(
    reference: List[torch.Tensor],
    quantized: List[torch.Tensor],
) -> float:
    """
    Compute overall RMSE between reference and quantized action predictions.

    Each element in the lists is an action tensor of shape [..., action_horizon, action_dim]
    or any shape — we just flatten and compare.
    """
    total_se = 0.0
    total_n  = 0
    for ref, quant in zip(reference, quantized):
        ref_f   = ref.float().cpu()
        quant_f = quant.float().cpu()
        diff    = ref_f - quant_f
        total_se += diff.pow(2).sum().item()
        total_n  += diff.numel()

    if total_n == 0:
        return float("nan")
    return math.sqrt(total_se / total_n)


def _compute_max_abs_error_deg(
    reference: List[torch.Tensor],
    quantized: List[torch.Tensor],
) -> float:
    """
    Compute the maximum absolute elementwise action error, reported in degrees.

    Assumption: action tensors are in radians (common for joint angles). If your
    action space is normalized (e.g. [-1, 1]) this number won't be meaningful.
    """
    max_abs = 0.0
    for ref, quant in zip(reference, quantized):
        ref_f = ref.float().cpu()
        quant_f = quant.float().cpu()
        diff = (ref_f - quant_f).abs()
        if diff.numel() == 0:
            continue
        v = diff.max().item()
        if v > max_abs:
            max_abs = v
    return max_abs * (180.0 / math.pi)
