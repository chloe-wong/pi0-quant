import math
from typing import Optional

import torch

from fp_formats import AddendSel, OutputFmtSel
from params_and_requests import InnerProductTreeParams, ComputeReq, WeightLoadReq
from inner_product_trees_model import InnerProductTreesModel


_E4M3_FLOAT_LUT = torch.zeros(256, dtype=torch.float32)
for _i in range(256):
    _sign = (_i >> 7) & 1
    _exp = (_i >> 3) & 0xF
    _frac = _i & 0x7
    if 0 < _exp < 0xF:
        _val = (1.0 + (_frac / 8.0)) * (2.0 ** (_exp - 7))
        _E4M3_FLOAT_LUT[_i] = -_val if _sign else _val


def torch_float_to_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    x_f32 = x.float().contiguous()
    u32 = x_f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    upper = (u32 >> 16) & 0xFFFF
    lsb = upper & 1
    rounded = u32 + (0x7FFF + lsb)
    return ((rounded >> 16) & 0xFFFF).to(torch.int32)


def torch_bf16_bits_to_float(bits: torch.Tensor) -> torch.Tensor:
    u32 = (bits.to(torch.int32) & 0xFFFF) << 16
    return u32.view(torch.float32)


def quant_bf16_tensor(x: torch.Tensor) -> torch.Tensor:
    bits = torch_float_to_bf16_bits(x.float())
    return torch_bf16_bits_to_float(bits)


def _float_to_e4m3_byte_scalar(v: float) -> int:
    if math.isnan(v):
        return 0
    if math.isinf(v):
        return 0xFE if v < 0 else 0x7E
    if v == 0.0:
        return 0

    sign = 1 if v < 0 else 0
    a = -v if sign else v

    exp = math.floor(math.log2(a))
    if exp > 8:
        return 0xFE if sign else 0x7E
    if exp < -6:
        return 0

    mant = a / (2.0 ** exp)
    frac = int(round((mant - 1.0) * 8.0))

    if frac == 8:
        frac = 0
        exp += 1

    if exp > 8:
        return 0xFE if sign else 0x7E
    if exp < -6:
        return 0

    return (sign << 7) | (((exp + 7) & 0xF) << 3) | (frac & 0x7)


def float_to_e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    flat = x.detach().float().cpu().reshape(-1).tolist()
    out = [_float_to_e4m3_byte_scalar(v) for v in flat]
    return torch.tensor(out, dtype=torch.uint8, device=x.device).reshape(x.shape)


def e4m3_bytes_to_float(x: torch.Tensor) -> torch.Tensor:
    lut = _E4M3_FLOAT_LUT.to(x.device)
    return lut[(x.to(torch.int64) & 0xFF)]


def decode_model_output_bits(out_bits: torch.Tensor, out_fmt_sel: OutputFmtSel) -> torch.Tensor:
    if out_fmt_sel is OutputFmtSel.OutBF16:
        return torch_bf16_bits_to_float(out_bits.to(torch.int32))
    e4m3_bytes = (out_bits & 0xFF).to(torch.uint8)
    return e4m3_bytes_to_float(e4m3_bytes)


class IPTLinearRTLFunction:
    """
    Functional adapter around InnerProductTreesModel.

    Assumptions:
      - input activations / weights are quantized to E4M3 before entering the MXU
      - psum is BF16
      - bias is loaded as E4M3 if used in the first tile
      - output container is BF16-width
      - output shape matches F.linear
    """

    def __init__(
        self,
        vec_len: int = 32,
        num_lanes: int = 16,
        pipeline_depth: int = 1,
        out_fmt_sel: OutputFmtSel = OutputFmtSel.OutBF16,
    ):
        self.p = InnerProductTreeParams.withPipelineDepth(
            pipeline_depth,
            InnerProductTreeParams(numLanes=num_lanes, vecLen=vec_len),
        )
        self.out_fmt_sel = out_fmt_sel
        self._prepared_cache = None

    def _tensor_cache_key(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        return (
            t.data_ptr(),
            tuple(t.shape),
            tuple(t.stride()),
            str(t.dtype),
            str(t.device),
            getattr(t, "_version", None),
        )

    def _prepare_static_operands(
        self,
        w_q: torch.Tensor,
        b_q: Optional[torch.Tensor],
        in_features: int,
        out_features: int,
    ):
        key = (self._tensor_cache_key(w_q), self._tensor_cache_key(b_q))
        if self._prepared_cache is not None and self._prepared_cache["key"] == key:
            return self._prepared_cache

        vec_len = self.p.vecLen
        num_lanes = self.p.numLanes
        num_k_tiles = (in_features + vec_len - 1) // vec_len

        w2 = w_q.float()
        b2 = b_q.float() if b_q is not None else None

        w_e4m3 = float_to_e4m3_bytes(w2)
        w_e4m3_list = w_e4m3.cpu().tolist()

        if b2 is not None:
            b_e4m3 = float_to_e4m3_bytes(b2)
            b_e4m3_list = b_e4m3.cpu().tolist()
        else:
            b_e4m3_list = None

        zero_vec = [0] * vec_len

        prepared_weight_tiles = []
        for out_base in range(0, out_features, num_lanes):
            lane_count = min(num_lanes, out_features - out_base)
            out_tile = []
            for k_tile in range(num_k_tiles):
                k0 = k_tile * vec_len
                k1 = min(k0 + vec_len, in_features)
                tile_width = k1 - k0

                lane_rows = []
                for lane in range(num_lanes):
                    if lane < lane_count:
                        row = w_e4m3_list[out_base + lane][k0:k1]
                        if tile_width < vec_len:
                            row = row + zero_vec[tile_width:]
                    else:
                        row = zero_vec
                    lane_rows.append(row)
                out_tile.append(lane_rows)
            prepared_weight_tiles.append((out_base, lane_count, out_tile))

        prepared = {
            "key": key,
            "b_e4m3_list": b_e4m3_list,
            "prepared_weight_tiles": prepared_weight_tiles,
        }
        self._prepared_cache = prepared
        return prepared

    def __call__(
        self,
        x_q: torch.Tensor,
        w_q: torch.Tensor,
        b_q: Optional[torch.Tensor] = None,
        scale_exp: int = 0,
    ) -> torch.Tensor:
        original_shape = x_q.shape[:-1]
        in_features = x_q.shape[-1]
        out_features = w_q.shape[0]

        x2 = x_q.reshape(-1, in_features).float()
        batch = x2.shape[0]
        device = x_q.device

        vec_len = self.p.vecLen
        num_lanes = self.p.numLanes
        num_k_tiles = (in_features + vec_len - 1) // vec_len

        x_e4m3 = float_to_e4m3_bytes(x2)
        x_e4m3_list = x_e4m3.cpu().tolist()

        prepared = self._prepare_static_operands(w_q, b_q, in_features, out_features)
        b_e4m3_list = prepared["b_e4m3_list"]
        prepared_weight_tiles = prepared["prepared_weight_tiles"]

        zero_vec = [0] * vec_len
        zero_bias = [0] * num_lanes
        scale_list = [scale_exp] * num_lanes

        x_tiles = []
        for b_idx in range(batch):
            row_tiles = []
            row = x_e4m3_list[b_idx]
            for k_tile in range(num_k_tiles):
                k0 = k_tile * vec_len
                k1 = min(k0 + vec_len, in_features)
                tile = row[k0:k1]
                if len(tile) < vec_len:
                    tile = tile + zero_vec[len(tile):]
                row_tiles.append(tile)
            x_tiles.append(row_tiles)

        y_bits = torch.zeros(batch, out_features, dtype=torch.int32, device=device)

        for out_base, lane_count, out_tile_weights in prepared_weight_tiles:
            dut = InnerProductTreesModel(self.p)
            psum_bits = [[0] * num_lanes for _ in range(batch)]

            if b_e4m3_list is not None:
                bias_first_tile = zero_bias.copy()
                for lane in range(lane_count):
                    bias_first_tile[lane] = b_e4m3_list[out_base + lane]
            else:
                bias_first_tile = zero_bias

            for k_tile in range(num_k_tiles):
                lane_rows = out_tile_weights[k_tile]
                for lane in range(num_lanes):
                    dut.load_weights(
                        WeightLoadReq(
                            weightsDma=lane_rows[lane],
                            laneIdx=lane,
                            last=(lane == num_lanes - 1),
                        )
                    )

                if k_tile == 0 and b_e4m3_list is not None:
                    addend_sel = AddendSel.UseBias
                    bias_list = bias_first_tile
                elif k_tile == 0:
                    addend_sel = AddendSel.UseAct
                    bias_list = zero_bias
                else:
                    addend_sel = AddendSel.UsePsum
                    bias_list = zero_bias

                for b_idx in range(batch):
                    req = ComputeReq(
                        act=x_tiles[b_idx][k_tile],
                        bias=bias_list,
                        psum=psum_bits[b_idx],
                        scaleExp=scale_list,
                        addendSel=addend_sel,
                        outFmtSel=self.out_fmt_sel,
                    )
                    psum_bits[b_idx] = dut.compute_now(req)

            tile_bits = torch.tensor(
                [row[:lane_count] for row in psum_bits],
                dtype=torch.int32,
                device=device,
            )
            y_bits[:, out_base:out_base + lane_count] = tile_bits

        y = decode_model_output_bits(y_bits, self.out_fmt_sel)
        return y.reshape(*original_shape, out_features)
