"""
rel_noise.py
---------------------------
Utilities for injecting relative-error noise into matmul outputs.

Relative error noise: adds ±rel_err * |y| to each element of y,
where rel_err is a dimensionless fraction (e.g. 0.01 = 1%).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RelNoiseConfig:
    """Relative-error noise: inject ±rel_err * |y| into each matmul output element."""
    rel_err: float = 0.0

    def enabled(self) -> bool:
        return self.rel_err != 0.0


def inject_rel_noise(y: torch.Tensor, *, rel_err: float) -> torch.Tensor:
    """
    Add ±rel_err * |y| to y (relative-error noise).

    Intended usage: inject noise into matmul outputs to emulate relative rounding drift.
    Sign is chosen uniformly at random per element (stateless, no explicit RNG).
    """
    if rel_err == 0.0:
        return y
    if rel_err < 0:
        raise ValueError("rel_err must be >= 0")
    sign = torch.where(
        torch.rand_like(y) < 0.5,
        torch.tensor(-1.0, device=y.device, dtype=y.dtype),
        torch.tensor(1.0, device=y.device, dtype=y.dtype),
    )
    return y + rel_err * y.abs() * sign
