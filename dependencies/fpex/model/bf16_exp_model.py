#!/usr/bin/env python3
"""Bit-exact functional BF16 exp model for this RTL.

This models the function implemented by:
  - src/main/scala/fpex/fpex.scala (BF16 path)
  - src/main/scala/fpex/common.scala (BF16 params)
  - src/main/scala/fpex/lut.scala
  - src/main/scala/fpex/hardfloat/RoundAnyRawFNToRecFN.scala
  - src/main/scala/fpex/hardfloat/fNFromRecFN.scala

It intentionally ignores pipeline/lanes/handshake and only models e^x for BF16.
"""

from __future__ import annotations

import argparse
import math
import struct


# HardFloat rounding mode encodings (as consumed by RoundAnyRawFNToRecFN).
ROUND_NEAR_EVEN = 0b000
ROUND_MIN_MAG = 0b001
ROUND_MIN = 0b010
ROUND_MAX = 0b011
ROUND_NEAR_MAX_MAG = 0b100
ROUND_ODD = 0b110


BF16_WORD_WIDTH = 16
BF16_EXP_WIDTH = 8
BF16_SIG_WIDTH = 8  # includes hidden bit
BF16_FRAC_WIDTH = BF16_SIG_WIDTH - 1

QMN_M = 9
QMN_N = 12
QMN_WIDTH = QMN_M + QMN_N  # 21

RLN2_Q2_12 = 5909

LUT_ADDR_BITS = 5
LUT_ENTRIES = 1 << LUT_ADDR_BITS
LUT_VAL_N = 16
LUT_VAL_M = 1
LUT_WIDTH = LUT_VAL_M + LUT_VAL_N  # 17
R_LOW_BITS = QMN_N - LUT_ADDR_BITS  # 7
LUT_TOP_ENDPOINT = (1 << LUT_WIDTH) - 1  # 131071

# BF16 ln(max finite) threshold for early +Inf.
MAX_X_EXP = 0x85
MAX_X_SIG = 0x31


def _mask(width: int) -> int:
    return (1 << width) - 1


def _u(value: int, width: int) -> int:
    return value & _mask(width)


def _s(value: int, width: int) -> int:
    value &= _mask(width)
    sign = 1 << (width - 1)
    return value - (1 << width) if (value & sign) else value


def _round_ties_away_positive(x: float) -> int:
    # Java/Scala Math.round for positive numbers.
    return int(math.floor(x + 0.5))


def _build_lut() -> list[int]:
    lut = []
    scale = 1 << LUT_VAL_N
    for i in range(LUT_ENTRIES):
        r = i / LUT_ENTRIES
        v = math.pow(2.0, r)
        lut.append(_round_ties_away_positive(v * scale))
    return lut


LUT = _build_lut()


def _low_mask(in_val: int, in_width: int, top_bound: int, bottom_bound: int) -> int:
    """Python port of hardfloat.primitives.lowMask."""
    assert top_bound != bottom_bound
    num_in_vals = 1 << in_width

    if top_bound < bottom_bound:
        return _low_mask((~in_val) & _mask(in_width), in_width, num_in_vals - 1 - top_bound, num_in_vals - 1 - bottom_bound)

    out_width = top_bound - bottom_bound

    if num_in_vals > 64:
        mid = num_in_vals // 2
        msb = (in_val >> (in_width - 1)) & 1
        lsbs = in_val & _mask(in_width - 1)
        if mid < top_bound:
            if mid <= bottom_bound:
                if msb:
                    return _low_mask(lsbs, in_width - 1, top_bound - mid, bottom_bound - mid)
                return 0
            if msb:
                left = _low_mask(lsbs, in_width - 1, top_bound - mid, 0)
                right_width = mid - bottom_bound
                right = _mask(right_width)
                return (left << right_width) | right
            return _low_mask(lsbs, in_width - 1, mid, bottom_bound)
        if msb:
            return _mask(out_width)
        return _low_mask(lsbs, in_width - 1, top_bound, bottom_bound)

    # Base branch from HardFloat:
    # shift = (BigInt(-1)<<numInVals).S>>in
    # Reverse(shift(numInVals-1-bottom, numInVals-top))
    shift_val = ((-1 << num_in_vals) >> in_val)
    lo = num_in_vals - top_bound

    sliced = 0
    for i in range(out_width):
        bit = (shift_val >> (lo + i)) & 1
        sliced |= bit << i

    rev = 0
    for i in range(out_width):
        if (sliced >> i) & 1:
            rev |= 1 << (out_width - 1 - i)
    return rev


def _raw_to_recfn_bf16(raw_s_exp_10b: int, raw_sig_11b: int, rounding_mode: int) -> int:
    """Port of RoundAnyRawFNToRecFN for this exact BF16 usage."""
    # Constants from RoundAnyRawFNToRecFN (for outExpWidth=8, outSigWidth=8).
    out_nan_exp = 7 << (BF16_EXP_WIDTH - 2)  # 448
    out_inf_exp = 6 << (BF16_EXP_WIDTH - 2)  # 384
    out_max_finite_exp = out_inf_exp - 1     # 383
    out_min_norm_exp = (1 << (BF16_EXP_WIDTH - 1)) + 2       # 130
    out_min_nonzero_exp = out_min_norm_exp - BF16_SIG_WIDTH + 1  # 123

    s_adjusted_exp = _s(raw_s_exp_10b, 10)
    adjusted_sig = _u(raw_sig_11b, 11)
    do_shift_sig_down1 = (adjusted_sig >> (BF16_SIG_WIDTH + 2)) & 1  # bit 10

    rm_near_even = rounding_mode == ROUND_NEAR_EVEN
    rm_max = rounding_mode == ROUND_MAX
    rm_near_max_mag = rounding_mode == ROUND_NEAR_MAX_MAG
    rm_odd = rounding_mode == ROUND_ODD

    # Sign is always positive in this datapath.
    round_mag_up = bool(rm_max)

    low = _low_mask(_u(s_adjusted_exp, 9), 9, out_min_norm_exp - BF16_SIG_WIDTH - 1, out_min_norm_exp)
    round_mask = (((low | do_shift_sig_down1) << 2) | 0b11) & _mask(11)
    shifted_round_mask = (round_mask >> 1) & _mask(11)
    round_pos_mask = ((~shifted_round_mask) & _mask(11)) & round_mask

    round_pos_bit = ((adjusted_sig & round_pos_mask) != 0)
    any_round_extra = ((adjusted_sig & shifted_round_mask) != 0)
    any_round = round_pos_bit or any_round_extra

    round_incr = (((rm_near_even or rm_near_max_mag) and round_pos_bit) or (round_mag_up and any_round))

    if round_incr:
        rounded_sig = (((adjusted_sig | round_mask) >> 2) + 1)
        if rm_near_even and round_pos_bit and (not any_round_extra):
            rounded_sig &= ~((round_mask >> 1) & _mask(10))
        rounded_sig &= _mask(10)
    else:
        rounded_sig = ((adjusted_sig & ((~round_mask) & _mask(11))) >> 2)
        if rm_odd and any_round:
            rounded_sig |= (round_pos_mask >> 1)
        rounded_sig &= _mask(10)

    s_rounded_exp = s_adjusted_exp + ((rounded_sig >> BF16_SIG_WIDTH) & _mask(2))

    common_exp_out = _u(s_rounded_exp, 9)
    if do_shift_sig_down1:
        common_fract_out = (rounded_sig >> 1) & _mask(BF16_FRAC_WIDTH)
    else:
        common_fract_out = rounded_sig & _mask(BF16_FRAC_WIDTH)

    common_overflow = (s_rounded_exp >> (BF16_EXP_WIDTH - 1)) >= 3
    common_total_underflow = s_rounded_exp < out_min_nonzero_exp

    if do_shift_sig_down1:
        round_mask_bit = ((round_mask >> 3) & 1) != 0
    else:
        round_mask_bit = ((round_mask >> 2) & 1) != 0

    # detectTininess is hardwired to 0 (before rounding).
    common_underflow = (common_total_underflow or (any_round and ((s_adjusted_exp >> BF16_EXP_WIDTH) <= 0) and round_mask_bit))
    common_inexact = common_total_underflow or any_round

    overflow = common_overflow

    overflow_round_mag_up = rm_near_even or rm_near_max_mag or round_mag_up
    peg_min_nonzero_mag_out = common_total_underflow and (round_mag_up or rm_odd)
    peg_max_finite_mag_out = overflow and (not overflow_round_mag_up)
    not_nan_is_inf_out = overflow and overflow_round_mag_up

    exp_out = common_exp_out
    if common_total_underflow:
        exp_out &= (~out_nan_exp) & _mask(9)
    if peg_min_nonzero_mag_out:
        exp_out &= (~((~out_min_nonzero_exp) & _mask(9))) & _mask(9)
    if peg_max_finite_mag_out:
        exp_out &= (~(1 << (BF16_EXP_WIDTH - 1))) & _mask(9)
    if not_nan_is_inf_out:
        exp_out &= (~(1 << (BF16_EXP_WIDTH - 2))) & _mask(9)
    if peg_min_nonzero_mag_out:
        exp_out |= out_min_nonzero_exp
    if peg_max_finite_mag_out:
        exp_out |= out_max_finite_exp
    if not_nan_is_inf_out:
        exp_out |= out_inf_exp

    fract_out = 0 if common_total_underflow else common_fract_out
    if peg_max_finite_mag_out:
        fract_out |= _mask(BF16_FRAC_WIDTH)

    # recFN width = sign(1) + exp(9) + frac(7), sign always 0 here.
    return ((exp_out & _mask(9)) << BF16_FRAC_WIDTH) | (fract_out & _mask(BF16_FRAC_WIDTH))


def _recfn_to_bf16(recfn_17b: int) -> int:
    """Port of fNFromRecFN for BF16."""
    sign = (recfn_17b >> (BF16_EXP_WIDTH + BF16_SIG_WIDTH)) & 1
    exp = (recfn_17b >> BF16_FRAC_WIDTH) & _mask(BF16_EXP_WIDTH + 1)  # 9 bits
    frac = recfn_17b & _mask(BF16_FRAC_WIDTH)

    is_zero = ((exp >> (BF16_EXP_WIDTH - 2)) == 0)
    is_special = ((exp >> (BF16_EXP_WIDTH - 1)) == 0b11)
    is_nan = is_special and (((exp >> (BF16_EXP_WIDTH - 2)) & 1) == 1)
    is_inf = is_special and (((exp >> (BF16_EXP_WIDTH - 2)) & 1) == 0)

    s_exp = exp  # zext
    raw_sig = ((0 << BF16_SIG_WIDTH) | ((0 if is_zero else 1) << BF16_FRAC_WIDTH) | frac) & _mask(BF16_SIG_WIDTH + 1)

    min_norm_exp = (1 << (BF16_EXP_WIDTH - 1)) + 2  # 130
    is_subnormal = s_exp < min_norm_exp
    denorm_shift_dist = (1 - (s_exp & _mask(3))) & _mask(3)
    denorm_fract = ((raw_sig >> 1) >> denorm_shift_dist) & _mask(BF16_FRAC_WIDTH)

    if is_subnormal:
        exp_out = 0
    else:
        exp_out = ((s_exp & _mask(BF16_EXP_WIDTH)) - ((1 << (BF16_EXP_WIDTH - 1)) + 1)) & _mask(BF16_EXP_WIDTH)
    if is_nan or is_inf:
        exp_out |= _mask(BF16_EXP_WIDTH)

    if is_subnormal:
        fract_out = denorm_fract
    else:
        fract_out = 0 if is_inf else (raw_sig & _mask(BF16_FRAC_WIDTH))

    return (sign << 15) | ((exp_out & _mask(BF16_EXP_WIDTH)) << BF16_FRAC_WIDTH) | (fract_out & _mask(BF16_FRAC_WIDTH))


def exp_bf16_bits(x_bits: int, rounding_mode: int = ROUND_NEAR_EVEN) -> int:
    """Compute RTL-equivalent BF16 output bits for e^x from BF16 input bits."""
    x_bits &= 0xFFFF

    sign = (x_bits >> 15) & 1
    exp = (x_bits >> BF16_FRAC_WIDTH) & _mask(BF16_EXP_WIDTH)
    frac = x_bits & _mask(BF16_FRAC_WIDTH)

    is_zero = exp == 0 and frac == 0
    is_subnorm = exp == 0 and frac != 0
    is_inf = exp == 0xFF and frac == 0
    is_nan = exp == 0xFF and frac != 0

    exp_fp_overflow = (sign == 0) and ((exp > MAX_X_EXP) or (exp == MAX_X_EXP and frac > MAX_X_SIG))

    if is_nan:
        is_sig_nan = ((frac >> (BF16_FRAC_WIDTH - 1)) & 1) == 0
        nan_frac = ((1 if is_sig_nan else 0) << (BF16_FRAC_WIDTH - 1)) | _mask(BF16_FRAC_WIDTH - 1)
        return (sign << 15) | (0xFF << BF16_FRAC_WIDTH) | nan_frac
    if is_zero or is_subnorm:
        return 0x3F80  # +1.0
    if is_inf and sign == 1:
        return 0x0000  # e^-inf = +0
    if (is_inf and sign == 0) or exp_fp_overflow:
        return 0x7F80  # +inf

    # qmnFromRawFloat (BF16): rawSig = 0b01_frac (9-bit container, numeric 8-bit).
    raw_sig = (1 << BF16_FRAC_WIDTH) | frac  # 8-bit hidden+frac
    shift = int(exp) - 122  # qmnN + unbiasedExp - (sigWidth-1)
    if shift < 0:
        mag = raw_sig >> (-shift)
    else:
        mag = raw_sig << shift

    if sign:
        qmn_value = _s(-mag, QMN_WIDTH)
    else:
        qmn_value = _s(mag, QMN_WIDTH)

    # Qmn.mul(rln2) then getKR.
    xrln2 = _s(qmn_value * RLN2_Q2_12, 35) >> QMN_N
    xrln2 = _s(xrln2, 23)
    k = _s(xrln2 >> QMN_N, 11)
    r = xrln2 & _mask(QMN_N)

    addr = (r >> R_LOW_BITS) & _mask(LUT_ADDR_BITS)
    r_lower = r & _mask(R_LOW_BITS)
    y0 = LUT[addr]
    y1 = LUT_TOP_ENDPOINT if addr == (LUT_ENTRIES - 1) else LUT[addr + 1]
    delta = y1 - y0
    delta_frac = (delta * r_lower) >> R_LOW_BITS
    pow2r = y0 + delta_frac

    # rawFloatFromQmnK for BF16.
    sig_with_gr = pow2r >> (LUT_VAL_N - (BF16_SIG_WIDTH - 1) - 2)  # >> 7
    pre_sig = sig_with_gr & _mask(BF16_SIG_WIDTH + 2)  # 10 bits
    sticky = (pow2r & _mask((LUT_VAL_N - (BF16_SIG_WIDTH - 1) - 2))) != 0
    raw_out_sig = pre_sig | (1 if sticky else 0)  # 11-bit container in RoundRaw input
    raw_out_s_exp = _s(k + (1 << BF16_EXP_WIDTH), 10)

    rec = _raw_to_recfn_bf16(raw_out_s_exp, raw_out_sig, rounding_mode & 0x7)
    return _recfn_to_bf16(rec)


def exp_bf16_float(x_float: float, rounding_mode: int = ROUND_NEAR_EVEN) -> int:
    """Compute RTL-equivalent BF16 output bits for e^x from a float input.

    The float input is converted to BF16 the same way the CLI does:
    truncate the low 16 bits of FP32 representation.
    """
    x_bits = float_to_bf16_bits_trunc(x_float)
    return exp_bf16_bits(x_bits, rounding_mode)


def bf16_bits_to_float(bits: int) -> float:
    bits &= 0xFFFF
    f32_bits = bits << 16
    return struct.unpack(">f", struct.pack(">I", f32_bits))[0]


def float_to_bf16_bits_trunc(value: float) -> int:
    f32_bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    return (f32_bits >> 16) & 0xFFFF


def _parse_int(s: str) -> int:
    return int(s, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bit-exact BF16 exp model for this RTL.")
    parser.add_argument("--x-bits", type=_parse_int, help="BF16 input bits (e.g. 0x3f80).")
    parser.add_argument("--x-float", type=float, help="Float input; converted to BF16 by truncating lower FP32 16 bits.")
    parser.add_argument("--rounding-mode", type=_parse_int, default=ROUND_NEAR_EVEN, help="3-bit HardFloat rounding mode (default 0 / RNE).")
    args = parser.parse_args()

    if (args.x_bits is None) == (args.x_float is None):
        raise SystemExit("Provide exactly one of --x-bits or --x-float.")

    if args.x_bits is not None:
        x_bits = args.x_bits & 0xFFFF
        out_bits = exp_bf16_bits(x_bits, args.rounding_mode)
    else:
        x_bits = float_to_bf16_bits_trunc(args.x_float)
        out_bits = exp_bf16_float(args.x_float, args.rounding_mode)
    print(f"in_bf16_bits=0x{x_bits:04x} ({bf16_bits_to_float(x_bits)})")
    print(f"out_bf16_bits=0x{out_bits:04x} ({bf16_bits_to_float(out_bits)})")


if __name__ == "__main__":
    main()
