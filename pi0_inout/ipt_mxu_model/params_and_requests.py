from __future__ import annotations

from dataclasses import dataclass

from fp_formats import E4M3, BF16, E4M3ProdFmt, AtlasFPType, AddendSel, OutputFmtSel


@dataclass(frozen=True)
class InnerProductTreeParams:
    numLanes: int = 16
    vecLen: int = 32
    accumIntWidth: int = 0
    pipelineCuts: frozenset[int] = frozenset()

    def __post_init__(self):
        for c in self.pipelineCuts:
            if c < 0 or c > 3:
                raise ValueError(f"pipelineCuts must be in {{0..3}}, got {self.pipelineCuts}")

    @property
    def inputFmt(self) -> AtlasFPType:
        return E4M3

    @property
    def biasFmt(self) -> AtlasFPType:
        return E4M3

    @property
    def psumFmt(self) -> AtlasFPType:
        return BF16

    @property
    def outputFmt(self) -> AtlasFPType:
        return BF16

    @property
    def anchorHeadroom(self) -> int:
        total = self.vecLen + 1
        return total.bit_length() + 1

    @property
    def intWidth(self) -> int:
        if self.accumIntWidth > 0:
            return self.accumIntWidth
        return E4M3ProdFmt.sigWidth + self.anchorHeadroom + 15

    @property
    def expWorkWidth(self) -> int:
        max_ew = max(
            self.inputFmt.expWidth,
            E4M3ProdFmt.expWidth,
            self.biasFmt.expWidth,
            self.psumFmt.expWidth,
            self.outputFmt.expWidth,
        )
        return max_ew + 4

    @property
    def numPipeCuts(self) -> int:
        return len(self.pipelineCuts)

    @property
    def latency(self) -> int:
        return self.numPipeCuts + 1

    @staticmethod
    def withPipelineDepth(depth: int, base: "InnerProductTreeParams | None" = None) -> "InnerProductTreeParams":
        if base is None:
            base = InnerProductTreeParams()
        if depth < 1 or depth > 5:
            raise ValueError(f"depth must be 1..5, got {depth}")
        cuts = {
            1: frozenset(),
            2: frozenset({1}),
            3: frozenset({0, 2}),
            4: frozenset({0, 1, 2}),
            5: frozenset({0, 1, 2, 3}),
        }[depth]
        return InnerProductTreeParams(
            numLanes=base.numLanes,
            vecLen=base.vecLen,
            accumIntWidth=base.accumIntWidth,
            pipelineCuts=cuts,
        )


@dataclass
class ComputeReq:
    act: list[int]
    bias: list[int]
    psum: list[int]
    scaleExp: list[int]
    addendSel: AddendSel
    outFmtSel: OutputFmtSel


@dataclass
class WeightLoadReq:
    weightsDma: list[int]
    laneIdx: int
    last: bool


@dataclass
class StepResult:
    out_valid: bool
    out_bits: list[int] | None
