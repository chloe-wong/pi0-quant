"""
noise_sweep.py
--------------
Helpers for sweeping ULP noise injection levels and matrix/vector format combos.

This is intentionally model-agnostic: you provide the model, observations, and an infer_fn.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

import torch
import torch.nn as nn

from .eval_harness import MatVecQuantConfig, run_quantization_eval_matvec
from .quant_types import QuantFormat
from .stats_tracker import Component
from .ulp_noise import UlpNoiseConfig


DEFAULT_FORMATS_4 = [
    QuantFormat.FLOAT32,
    QuantFormat.FLOAT16,
    QuantFormat.BFLOAT16,
    QuantFormat.FLOAT8_E4M3,
]


@dataclass(frozen=True)
class NoiseSweepSpec:
    rmse_threshold: float = 0.4
    max_deg_threshold: Optional[float] = 20.0
    max_n_ulp: int = 8


def matvec_format_grid(
    fmts: Optional[List[QuantFormat]] = None,
) -> List[tuple[QuantFormat, QuantFormat, QuantFormat]]:
    """
    Return all (matrix_in_fmt, matrix_out_fmt, vector_out_fmt) tuples.

    Constraint is handled by construction:
        vector_in_fmt == matrix_out_fmt
    """
    fmts = fmts or list(DEFAULT_FORMATS_4)
    return [(mi, mo, vo) for mi in fmts for mo in fmts for vo in fmts]


def run_noise_sweep_for_config(
    *,
    model: nn.Module,
    observations: List[Any],
    infer_fn: Callable[[nn.Module, Any], torch.Tensor],
    base_cfg: MatVecQuantConfig,
    spec: NoiseSweepSpec,
    verbose: bool = True,
) -> List[dict]:
    """
    For a single (mat_in, mat_out, vec_out) config, run increasing ±n ULP noise until failure.
    """
    # Collect reference once (unquantized).
    reference_actions: List[torch.Tensor] = []
    with torch.no_grad():
        for obs in observations:
            reference_actions.append(infer_fn(model, obs).detach().cpu())
    results: List[dict] = []

    for mode in spec.modes:
        if verbose:
            print(f"\n[noise-sweep] mode={mode.value}  base={base_cfg.label}")

        for n in range(1, spec.max_n_ulp + 1):
            cfg = copy.deepcopy(base_cfg)
            # Default: define ULP in the matmul-output format grid (common interpretation)
            cfg.ulp_noise = UlpNoiseConfig(n_ulp=n, mode=mode, ulp_fmt=cfg.matrix_out_fmt)
            cfg.label = f"{base_cfg.label}_ulp={n}_{mode.value}"

            out = run_quantization_eval_matvec(
                model=model,
                observations=observations,
                infer_fn=infer_fn,
                config=cfg,
                reference_actions=reference_actions,
                rmse_threshold=spec.rmse_threshold,
                max_deg_threshold=spec.max_deg_threshold,
                verbose=verbose,
            )
            results.append(out)

            if out["violates_rmse"] or out["violates_deg"]:
                if verbose:
                    print(
                        f"[noise-sweep] stop: n_ulp={n} violates "
                        f"(rmse={out['violates_rmse']}, deg={out['violates_deg']})"
                    )
                break

    return results


def run_noise_sweep_grid(
    *,
    model: nn.Module,
    observations: List[Any],
    infer_fn: Callable[[nn.Module, Any], torch.Tensor],
    fmts: Optional[List[QuantFormat]] = None,
    skip_components: Optional[Iterable[Component]] = None,
    spec: Optional[NoiseSweepSpec] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Run the full grid over format combos, and for each combo sweep ULP noise levels.
    """
    spec = spec or NoiseSweepSpec()
    skip = frozenset(skip_components or [])
    all_results: List[dict] = []

    for mi, mo, vo in matvec_format_grid(fmts):
        base = MatVecQuantConfig(
            matrix_in_fmt=mi,
            matrix_out_fmt=mo,
            vector_out_fmt=vo,
            ulp_noise=None,
            skip_components=skip,
            label=f"mat_in={mi.value}_mat_out={mo.value}_vec_out={vo.value}",
        )
        all_results.extend(
            run_noise_sweep_for_config(
                model=model,
                observations=observations,
                infer_fn=infer_fn,
                base_cfg=base,
                spec=spec,
                verbose=verbose,
            )
        )

    return all_results

