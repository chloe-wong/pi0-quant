from __future__ import annotations

from collections import deque

from fp_formats import AddendSel, E4M3ProdFmt, wrap_signed
from params_and_requests import ComputeReq, WeightLoadReq, StepResult, InnerProductTreeParams
from converters import (
    e4m3_mul_to_prod,
    e4m3_prod_to_aligned_int,
    ieee_to_aligned_int,
    aligned_int_to_bf16,
    output_conv_stage,
)


_E4M3_PROD_BIAS = E4M3ProdFmt.bias


class AnchorAccumulationTreeModel:
    def __init__(self, p: InnerProductTreeParams):
        self.p = p
        self.sentinel = -(1 << (p.expWorkWidth - 2))

        bias_fmt = p.biasFmt
        psum_fmt = p.psumFmt

        self._anchor_headroom = p.anchorHeadroom
        self._int_width = p.intWidth

        self._bias_mant_bits = bias_fmt.mantissaBits
        self._bias_exp_mask = (1 << bias_fmt.expWidth) - 1
        self._bias_zero_mask = (1 << (bias_fmt.expWidth + 1)) - 1
        self._bias_bias = bias_fmt.ieeeBias

        self._psum_mant_bits = psum_fmt.mantissaBits
        self._psum_exp_mask = (1 << psum_fmt.expWidth) - 1
        self._psum_frac_mask = (1 << psum_fmt.mantissaBits) - 1
        self._psum_bias = psum_fmt.ieeeBias

    def _product_unbiased_exp(self, prod_bits: int) -> int:
        exp_bits = (prod_bits >> 7) & 0x1F
        return self.sentinel if exp_bits == 0 else (exp_bits - _E4M3_PROD_BIAS)

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
        weights = weight_buf1 if buf_read_sel else weight_buf0

        prod_s0 = []
        max_prod_exp = self.sentinel
        for a, w in zip(act, weights):
            prod = e4m3_mul_to_prod(a, w)
            prod_s0.append(prod)

            exp_bits = (prod >> 7) & 0x1F
            prod_exp = self.sentinel if exp_bits == 0 else (exp_bits - _E4M3_PROD_BIAS)
            if prod_exp > max_prod_exp:
                max_prod_exp = prod_exp

        addend_exp = self.sentinel

        if addend_sel is AddendSel.UseBias:
            bias_exp_field = (bias >> self._bias_mant_bits) & self._bias_exp_mask
            bias_is_zero = ((bias >> self._bias_mant_bits) & self._bias_zero_mask) == 0
            if not bias_is_zero:
                addend_exp = bias_exp_field - self._bias_bias

        elif addend_sel is AddendSel.UsePsum:
            psum_exp_field = (psum >> self._psum_mant_bits) & self._psum_exp_mask
            psum_frac = psum & self._psum_frac_mask
            psum_is_zero = (psum_exp_field == 0) and (psum_frac == 0)
            if not psum_is_zero:
                addend_exp = psum_exp_field - self._psum_bias

        anchor = (max_prod_exp if max_prod_exp >= addend_exp else addend_exp) + self._anchor_headroom

        int_width = self._int_width
        prod_sum = 0
        for prod in prod_s0:
            prod_sum = wrap_signed(prod_sum + e4m3_prod_to_aligned_int(prod, anchor, int_width), int_width)

        if addend_sel is AddendSel.UseBias:
            addend_int = ieee_to_aligned_int(bias, self.p.biasFmt, anchor, int_width)
        elif addend_sel is AddendSel.UsePsum:
            addend_int = ieee_to_aligned_int(psum, self.p.psumFmt, anchor, int_width)
        else:
            addend_int = 0

        total_int = wrap_signed(prod_sum + addend_int, int_width)
        bf16_result = aligned_int_to_bf16(total_int, anchor, int_width)
        return output_conv_stage(bf16_result, out_fmt_sel, scale_exp)


class InnerProductTreesModel:
    def __init__(self, p: InnerProductTreeParams = InnerProductTreeParams()):
        self.p = p
        self.wEn = False
        self.wbuf0 = [[0] * p.vecLen for _ in range(p.numLanes)]
        self.wbuf1 = [[0] * p.vecLen for _ in range(p.numLanes)]
        self.lanes = [AnchorAccumulationTreeModel(p) for _ in range(p.numLanes)]
        self.out_queue: deque[list[int] | None] = deque([None] * p.latency)

    @property
    def buf_read_sel(self) -> bool:
        return not self.wEn

    def reset(self) -> None:
        p = self.p
        self.wEn = False
        self.wbuf0 = [[0] * p.vecLen for _ in range(p.numLanes)]
        self.wbuf1 = [[0] * p.vecLen for _ in range(p.numLanes)]
        self.out_queue = deque([None] * p.latency)

    def load_weights(self, req: WeightLoadReq) -> None:
        p = self.p
        lane_idx = req.laneIdx

        if len(req.weightsDma) != p.vecLen:
            raise ValueError(f"weightsDma length must be {p.vecLen}")
        if not (0 <= lane_idx < p.numLanes):
            raise ValueError("laneIdx out of range")

        row = [x & 0xFF for x in req.weightsDma]
        if self.wEn:
            self.wbuf1[lane_idx] = row
        else:
            self.wbuf0[lane_idx] = row

        if req.last:
            self.wEn = not self.wEn

    def compute_now(self, req: ComputeReq) -> list[int]:
        p = self.p
        num_lanes = p.numLanes
        vec_len = p.vecLen

        if len(req.act) != vec_len:
            raise ValueError(f"act length must be {vec_len}")
        if len(req.bias) != num_lanes:
            raise ValueError(f"bias length must be {num_lanes}")
        if len(req.psum) != num_lanes:
            raise ValueError(f"psum length must be {num_lanes}")
        if len(req.scaleExp) != num_lanes:
            raise ValueError(f"scaleExp length must be {num_lanes}")

        act_masked = [x & 0xFF for x in req.act]

        buf_read_sel = self.buf_read_sel
        addend_sel = req.addendSel
        out_fmt_sel = req.outFmtSel

        wbuf0 = self.wbuf0
        wbuf1 = self.wbuf1
        lanes = self.lanes
        bias = req.bias
        psum = req.psum
        scale_exp = req.scaleExp

        out = [0] * num_lanes
        for lane_idx in range(num_lanes):
            out[lane_idx] = lanes[lane_idx].compute_lane(
                act=act_masked,
                weight_buf0=wbuf0[lane_idx],
                weight_buf1=wbuf1[lane_idx],
                bias=bias[lane_idx] & 0xFF,
                psum=psum[lane_idx] & 0xFFFF,
                scale_exp=scale_exp[lane_idx],
                buf_read_sel=buf_read_sel,
                addend_sel=addend_sel,
                out_fmt_sel=out_fmt_sel,
            ) & 0xFFFF

        return out

    def step(
        self,
        compute_req: ComputeReq | None = None,
        weight_load_req: WeightLoadReq | None = None,
    ) -> StepResult:
        if weight_load_req is not None:
            self.load_weights(weight_load_req)

        produced = self.compute_now(compute_req) if compute_req is not None else None

        out_queue = self.out_queue
        out_queue.append(produced)
        popped = out_queue.popleft()
        return StepResult(out_valid=popped is not None, out_bits=popped)
