"""lane_boxes/tanh_rec.py — Group D placeholder for `TanhRec.scala`.

`TanhRec` is the BF16 vector tanh lane box. The Scala module is
genuinely complex: it wraps `TanhLUT` AND a per-lane `Tanh` (in
`dependencies/sp26-fp-units/.../vpuFUnits/Tanh.scala`) iterative
HardFloat module that drives the LUT addresses (`TanhRec.scala:53,
72-81`). Each per-lane `Tanh` has its own internal cycle accounting
that the `TanhRec` wrapper doesn't expose at the IO boundary, and
its result combines a piecewise-linear LUT interpolation with
HardFloat-style RNE rounding.

Like `Exp`, this module is a Group D placeholder — see
`IMPLEMENTATION_PLAN.md §6` for the porting strategy decision.

Visible latency assumption: 1 cycle (1 `commonState` register stage
per `TanhRec.scala:62`). Whether the per-lane `Tanh` modules add
extra cycles depends on their internal pipeline, which we have not
yet inspected; revisit when Group D ships.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..vector_params import VectorParams


@dataclass
class TanhReq:
    xVec: list[int]                     # 16 BF16 bit patterns
    laneMask: int = 0xFFFF


@dataclass
class TanhResp:
    result: list[int]                   # 16 BF16 bit patterns


class TanhRec:
    """Group D placeholder — see module docstring."""

    LATENCIES: dict[str, int] = {"tanh": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: TanhReq) -> TanhResp:
        raise NotImplementedError(
            "Tanh funct model is Group D — blocked on "
            "IMPLEMENTATION_PLAN.md §6 (per-lane Tanh module + "
            "HardFloat round-trip strategy)."
        )

    def step(self, op_name: str, req: Optional[TanhReq]) -> Optional[TanhResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"TanhRec has no op {op_name!r}; "
                f"valid: {sorted(self.LATENCIES)}"
            )
        if req is not None:
            raise NotImplementedError(
                "TanhRec.step with a live request is Group D — blocked on §6."
            )
        q = self._queues[op_name]
        q.append(None)
        return q.popleft()
