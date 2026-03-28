"""
stats_tracker.py
----------------
Accumulates per-layer and per-component RMSE statistics across inference calls.

Design:
- Each QuantLinear calls tracker.record() after every forward pass.
- StatsTracker accumulates running mean-squared error (numerically stable via
  Welford's online algorithm) so we can average across many inference steps
  without storing all outputs.
- At the end of evaluation, call tracker.summary() to get a dict with:
    - Per-layer stats (name, component, rmse, n_calls, tensor_elements)
    - Per-component aggregate stats (mean RMSE and std across layers)
  These can be trivially converted to a DataFrame for analysis.

Component taxonomy (matching Pi0's architecture):
    VISION         — SigLIP ViT encoder layers
    LANGUAGE       — PaliGemma Gemma language model layers
    ACTION_EXPERT  — Gemma action-expert transformer layers
    ACTION_HEAD    — Thin action projection MLPs at the Pi0 level
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Component taxonomy
# ---------------------------------------------------------------------------

class Component(str, Enum):
    VISION        = "vision"
    LANGUAGE      = "language"
    ACTION_EXPERT = "action_expert"
    ACTION_HEAD   = "action_head"
    UNKNOWN       = "unknown"


# ---------------------------------------------------------------------------
# Per-layer running statistics (Welford's online algorithm)
# ---------------------------------------------------------------------------

@dataclass
class LayerStats:
    """Welford online mean of squared-error per layer."""
    name: str
    component: Component
    in_features: int
    out_features: int

    # Internal Welford accumulators — error
    _n: int   = field(default=0, repr=False)
    _mean_mse: float = field(default=0.0, repr=False)
    _M2: float = field(default=0.0, repr=False)  # for variance across calls
    # Welford accumulator — fp32 reference scale (for relative RMSE)
    _mean_fp_ms: float = field(default=0.0, repr=False)

    def update(self, y_fp: torch.Tensor, y_quant: torch.Tensor) -> None:
        """
        Update running stats with one forward-pass batch.

        We compute MSE over the full output tensor (all batch elements, all
        sequence positions, all output features), then treat that scalar as
        one new observation for Welford's running mean/variance.
        Also tracks mean-square of y_fp for relative RMSE denominator.
        """
        with torch.no_grad():
            y_fp_f = y_fp.float()
            diff    = y_fp_f - y_quant.float()
            mse     = diff.pow(2).mean().item()
            fp_ms   = y_fp_f.pow(2).mean().item()  # scale of fp32 reference

        # Welford update — error
        self._n += 1
        delta = mse - self._mean_mse
        self._mean_mse += delta / self._n
        delta2 = mse - self._mean_mse
        self._M2 += delta * delta2

        # Welford update — fp32 scale (same counter)
        delta_fp = fp_ms - self._mean_fp_ms
        self._mean_fp_ms += delta_fp / self._n

    @property
    def n_calls(self) -> int:
        return self._n

    @property
    def rmse(self) -> float:
        """Root mean squared error averaged over all observed calls."""
        if self._n == 0:
            return float("nan")
        return math.sqrt(max(self._mean_mse, 0.0))

    @property
    def rel_rmse(self) -> float:
        """
        Relative RMSE: rmse / rms(y_fp32).

        Normalises by the typical magnitude of the fp32 activations so that
        values from layers with different output scales are comparable.
        A value of 0.01 means the quantization error is 1% of the activation RMS.
        Returns nan if no calls recorded or fp32 output is all-zero.
        """
        if self._n == 0 or self._mean_fp_ms <= 0:
            return float("nan")
        return math.sqrt(max(self._mean_mse, 0.0) / self._mean_fp_ms)

    @property
    def mse(self) -> float:
        if self._n == 0:
            return float("nan")
        return self._mean_mse

    @property
    def mse_std(self) -> float:
        """Std of per-call MSE values (Welford variance)."""
        if self._n < 2:
            return float("nan")
        return math.sqrt(self._M2 / (self._n - 1))

    def to_dict(self) -> dict:
        return {
            "layer":        self.name,
            "component":    self.component.value,
            "in_features":  self.in_features,
            "out_features": self.out_features,
            "n_calls":      self._n,
            "rmse":         self.rmse,
            "rel_rmse":     self.rel_rmse,
            "mse":          self.mse,
            "mse_std":      self.mse_std,
        }


# ---------------------------------------------------------------------------
# Per-component aggregate
# ---------------------------------------------------------------------------

@dataclass
class ComponentStats:
    component: Component
    n_layers: int
    mean_rmse: float      # mean of per-layer absolute RMSE
    std_rmse: float
    max_rmse: float
    min_rmse: float
    mean_rel_rmse: float  # mean of per-layer relative RMSE (rmse / rms_fp32)
    total_calls: int

    def to_dict(self) -> dict:
        return {
            "component":    self.component.value,
            "n_layers":     self.n_layers,
            "mean_rmse":    self.mean_rmse,
            "std_rmse":     self.std_rmse,
            "max_rmse":     self.max_rmse,
            "min_rmse":     self.min_rmse,
            "mean_rel_rmse": self.mean_rel_rmse,
            "total_calls":  self.total_calls,
        }


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class StatsTracker:
    """
    Central store for quantization RMSE statistics across all QuantLinear layers.

    Usage:
        tracker = StatsTracker()
        # Pass to model_patcher.patch_model(..., tracker=tracker)
        # Run inference (possibly many steps)
        report = tracker.summary()
        report.print()
        df = report.to_dataframe()   # requires pandas

    Per-call records:
        tracker.calls — list of dicts, one per record() call, in order:
            {seq, name, component, rmse, ref_rms, y_quant_rms}
        seq is a global counter incremented on every record() call.
    """

    def __init__(self) -> None:
        self._layers: Dict[str, LayerStats] = {}
        self.calls: List[dict] = []      # per-call records in execution order
        self._seq: int = 0               # global sequence counter

    def register(
        self,
        name: str,
        component: Component,
        in_features: int,
        out_features: int,
    ) -> None:
        """Called by model_patcher when building QuantLinear layers."""
        if name not in self._layers:
            self._layers[name] = LayerStats(
                name=name,
                component=component,
                in_features=in_features,
                out_features=out_features,
            )

    def record(
        self,
        name: str,
        component: Component,
        y_fp: torch.Tensor,
        y_quant: torch.Tensor,
    ) -> None:
        """Called by QuantLinear.forward() after each matmul."""
        if name not in self._layers:
            # Auto-register if not pre-registered.
            # 0-D scalar outputs (e.g. from sum/amax reductions) have no feature dim.
            feat = y_quant.shape[-1] if y_quant.dim() > 0 else 1
            self._layers[name] = LayerStats(
                name=name,
                component=component,
                in_features=feat,
                out_features=feat,
            )
        self._layers[name].update(y_fp, y_quant)

        # Per-call record (execution order)
        with torch.no_grad():
            y_fp_f = y_fp.float()
            y_q_f  = y_quant.float()
            diff   = y_fp_f - y_q_f
            mse    = diff.pow(2).mean().item()
            fp_ms  = y_fp_f.pow(2).mean().item()
            q_ms   = y_q_f.pow(2).mean().item()
        self.calls.append({
            "seq":         self._seq,
            "name":        name,
            "component":   component.value if isinstance(component, Component) else str(component),
            "rmse":        math.sqrt(max(mse, 0.0)),
            "ref_rms":     math.sqrt(max(fp_ms, 0.0)),
            "quant_rms":   math.sqrt(max(q_ms, 0.0)),
        })
        self._seq += 1

    def reset(self) -> None:
        """Reset all accumulated statistics (e.g., between evaluation episodes)."""
        for stats in self._layers.values():
            stats._n = 0
            stats._mean_mse = 0.0
            stats._M2 = 0.0
            stats._mean_fp_ms = 0.0
        self.calls.clear()
        self._seq = 0

    def layer_rows(self) -> List[dict]:
        """Return a list of dicts (one per layer), sorted by component then name."""
        return [
            s.to_dict()
            for s in sorted(self._layers.values(), key=lambda s: (s.component.value, s.name))
        ]

    def component_rows(self) -> List[dict]:
        """Return per-component aggregate stats."""
        by_component: Dict[Component, List[LayerStats]] = defaultdict(list)
        for s in self._layers.values():
            by_component[s.component].append(s)

        rows = []
        for comp in Component:
            layers = by_component.get(comp, [])
            if not layers:
                continue
            rmse_values     = [s.rmse     for s in layers if not math.isnan(s.rmse)]
            rel_rmse_values = [s.rel_rmse for s in layers if not math.isnan(s.rel_rmse)]
            if not rmse_values:
                continue
            rows.append(ComponentStats(
                component=comp,
                n_layers=len(layers),
                mean_rmse=_safe_mean(rmse_values),
                std_rmse=_safe_std(rmse_values),
                max_rmse=max(rmse_values),
                min_rmse=min(rmse_values),
                mean_rel_rmse=_safe_mean(rel_rmse_values),
                total_calls=sum(s.n_calls for s in layers),
            ).to_dict())
        return rows

    def summary(self) -> "StatsReport":
        return StatsReport(
            layer_rows=self.layer_rows(),
            component_rows=self.component_rows(),
        )


# ---------------------------------------------------------------------------
# Report object
# ---------------------------------------------------------------------------

@dataclass
class StatsReport:
    layer_rows: List[dict]
    component_rows: List[dict]

    def print(self, show_layers: bool = False) -> None:
        """Pretty-print to stdout."""
        print("\n=== Per-Component RMSE ===")
        _print_table(self.component_rows, [
            ("component", 14),
            ("n_layers",  8),
            ("mean_rmse", 12),
            ("std_rmse",  12),
            ("max_rmse",  12),
            ("min_rmse",  12),
            ("total_calls", 12),
        ])

        if show_layers:
            print("\n=== Per-Layer RMSE ===")
            _print_table(self.layer_rows, [
                ("component",   14),
                ("layer",       60),
                ("n_calls",      8),
                ("rmse",        12),
                ("mse_std",     12),
            ])

    def to_dataframe(self):
        """Convert to pandas DataFrames. Requires pandas."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pip install pandas to use to_dataframe()")
        return (
            pd.DataFrame(self.layer_rows),
            pd.DataFrame(self.component_rows),
        )

    def to_dict(self) -> dict:
        return {
            "layers":     self.layer_rows,
            "components": self.component_rows,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")

def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _safe_mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

def _fmt(v, width: int) -> str:
    if isinstance(v, float):
        return f"{v:.4e}".ljust(width)
    return str(v).ljust(width)

def _print_table(rows: list[dict], cols: list[tuple[str, int]]) -> None:
    header = "  ".join(name.ljust(w) for name, w in cols)
    sep    = "  ".join("-" * w for _, w in cols)
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(_fmt(row.get(name, ""), w) for name, w in cols)
        print(line)
