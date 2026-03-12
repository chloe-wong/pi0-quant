from __future__ import annotations

from collections import deque

from fp_formats import E4M3ProdFmt, wrap_signed
from params_and_requests import ComputeReq, WeightLoadReq, StepResult, InnerProductTreeParams
from converters import (
    e4m3_mul_to_prod,
    e4m3_prod_to_aligned_int,
    ieee_to_aligned_int,
    aligned_int_to_bf16,
    output_conv_stage,
)


class AnchorAccumulationTreeModel:
    def __init__(self, p: InnerProductTreeParams):
        self.p = p
        self.sentinel = -(1 << (p.expWorkWidth - 2))

    def _product_unbiased_exp(self, prod_bits: int) -> int:
        exp_bits = (prod_bits >> 7) & 0x1F
        if exp_bits == 0:
            return self.sentinel
        return exp_bits - E4M3ProdFmt.bias

    def compute_lane(
        self,
        act: list[int],
        weight_buf0: list[int],
        weight_buf1: list[int],
        bias: int,
        psum: int,
        scale_exp: int,
        buf_read_sel: bool,
        addend_sel,
        out_fmt_sel,
    ) -> int:
        p = self.p
        weights = weight_buf1 if buf_read_sel else weight_buf0

        # S0
        prod_s0 = [e4m3_mul_to_prod(a, w) for a, w in zip(act, weights)]

        bias_exp_field = (bias >> p.biasFmt.mantissaBits) & ((1 << p.biasFmt.expWidth) - 1)
        bias_is_zero = ((bias >> p.biasFmt.mantissaBits) & ((1 << (p.biasFmt.expWidth + 1)) - 1)) == 0
        bias_unb_exp = bias_exp_field - p.biasFmt.ieeeBias

        psum_exp_field = (psum >> p.psumFmt.mantissaBits) & ((1 << p.psumFmt.expWidth) - 1)
        psum_frac = psum & ((1 << p.psumFmt.mantissaBits) - 1)
        psum_is_zero = (psum_exp_field == 0) and (psum_frac == 0)
        psum_unb_exp = psum_exp_field - p.psumFmt.ieeeBias

        addend_exp = self.sentinel
        if addend_sel.name == "UseBias":
            addend_exp = self.sentinel if bias_is_zero else bias_unb_exp
        elif addend_sel.name == "UsePsum":
            addend_exp = self.sentinel if psum_is_zero else psum_unb_exp

        # S1
        prod_unb_exp = [self._product_unbiased_exp(prod) for prod in prod_s0]
        anchor = max(prod_unb_exp + [addend_exp]) + p.anchorHeadroom

        # S2
        prod_int = [e4m3_prod_to_aligned_int(prod, anchor, p.intWidth) for prod in prod_s0]
        bias_int = ieee_to_aligned_int(bias, p.biasFmt, anchor, p.intWidth)
        psum_int = ieee_to_aligned_int(psum, p.psumFmt, anchor, p.intWidth)

        addend_int = 0
        if addend_sel.name == "UseBias":
            addend_int = bias_int
        elif addend_sel.name == "UsePsum":
            addend_int = psum_int

        # S3
        prod_sum = 0
        for x in prod_int:
            prod_sum = wrap_signed(prod_sum + x, p.intWidth)
        total_int = wrap_signed(prod_sum + addend_int, p.intWidth)

        # S4 + output conversion
        bf16_result = aligned_int_to_bf16(total_int, anchor, p.intWidth)
        return output_conv_stage(bf16_result, out_fmt_sel, scale_exp)


class InnerProductTreesModel:
    def __init__(self, p: InnerProductTreeParams = InnerProductTreeParams()):
        self.p = p
        self.wEn = False
        self.wbuf0 = [[0 for _ in range(p.vecLen)] for _ in range(p.numLanes)]
        self.wbuf1 = [[0 for _ in range(p.vecLen)] for _ in range(p.numLanes)]
        self.lanes = [AnchorAccumulationTreeModel(p) for _ in range(p.numLanes)]
        self.out_queue: deque[list[int] | None] = deque([None] * p.latency)

    @property
    def buf_read_sel(self) -> bool:
        return not self.wEn

    def load_weights(self, req: WeightLoadReq) -> None:
        if len(req.weightsDma) != self.p.vecLen:
            raise ValueError(f"weightsDma length must be {self.p.vecLen}")
        if not (0 <= req.laneIdx < self.p.numLanes):
            raise ValueError("laneIdx out of range")

        row = [x & 0xFF for x in req.weightsDma]
        if self.wEn:
            self.wbuf1[req.laneIdx] = row
        else:
            self.wbuf0[req.laneIdx] = row

        if req.last:
            self.wEn = not self.wEn

    def compute_now(self, req: ComputeReq) -> list[int]:
        if len(req.act) != self.p.vecLen:
            raise ValueError(f"act length must be {self.p.vecLen}")
        if len(req.bias) != self.p.numLanes:
            raise ValueError(f"bias length must be {self.p.numLanes}")
        if len(req.psum) != self.p.numLanes:
            raise ValueError(f"psum length must be {self.p.numLanes}")
        if len(req.scaleExp) != self.p.numLanes:
            raise ValueError(f"scaleExp length must be {self.p.numLanes}")

        out = []
        for lane_idx in range(self.p.numLanes):
            lane_out = self.lanes[lane_idx].compute_lane(
                act=[x & 0xFF for x in req.act],
                weight_buf0=self.wbuf0[lane_idx],
                weight_buf1=self.wbuf1[lane_idx],
                bias=req.bias[lane_idx] & 0xFF,
                psum=req.psum[lane_idx] & 0xFFFF,
                scale_exp=req.scaleExp[lane_idx],
                buf_read_sel=self.buf_read_sel,
                addend_sel=req.addendSel,
                out_fmt_sel=req.outFmtSel,
            )
            out.append(lane_out & 0xFFFF)
        return out

    def step(
        self,
        compute_req: ComputeReq | None = None,
        weight_load_req: WeightLoadReq | None = None,
    ) -> StepResult:
        if weight_load_req is not None:
            self.load_weights(weight_load_req)

        produced = None
        if compute_req is not None:
            produced = self.compute_now(compute_req)

        self.out_queue.append(produced)
        popped = self.out_queue.popleft()
        return StepResult(out_valid=popped is not None, out_bits=popped)
