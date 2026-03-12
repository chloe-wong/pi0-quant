from __future__ import annotations

import math

from fp_formats import (
    E4M3,
    BF16,
    E4M3ProdFmt,
    E4M3_MAX_NEG,
    E4M3_MAX_POS,
    decode_e4m3,
    encode_e4m3_normal,
    f32_to_bf16_bits_rne,
    round_right_shift4_rne,
    sanitize_bf16,
    wrap_signed,
)
from params_and_requests import InnerProductTreeParams


# Product model: E4M3 x E4M3 -> 13-bit custom float
# Product format: S(1) E(5, bias=13) M(7)

def pack_e4m3_prod(sign: int, exp_unb: int, mant7: int) -> int:
    exp_field = exp_unb + E4M3ProdFmt.bias
    if exp_field <= 0:
        return 0
    if exp_field >= (1 << E4M3ProdFmt.expWidth) - 1:
        exp_field = (1 << E4M3ProdFmt.expWidth) - 1
    return ((sign & 1) << 12) | ((exp_field & 0x1F) << 7) | (mant7 & 0x7F)



def e4m3_mul_to_prod(a_bits: int, b_bits: int) -> int:
    a_bits &= 0xFF
    b_bits &= 0xFF

    # field extraction
    a_sign = (a_bits >> 7) & 0x1
    a_exp  = (a_bits >> 3) & 0xF
    a_man  = a_bits & 0x7

    b_sign = (b_bits >> 7) & 0x1
    b_exp  = (b_bits >> 3) & 0xF
    b_man  = b_bits & 0x7

    # zero detect: any exp == 0 is treated as zero
    a_zero = (a_exp == 0)
    b_zero = (b_exp == 0)
    is_zero = a_zero or b_zero

    # sign
    out_sign = a_sign ^ b_sign

    if is_zero:
        return (out_sign << 12)

    # significand multiply
    a_sig = (1 << 3) | a_man   # 4 bits
    b_sig = (1 << 3) | b_man   # 4 bits
    prod_sig = a_sig * b_sig   # 8 bits

    # normalization
    need_shift = (prod_sig >> 7) & 0x1

    if need_shift:
        out_man = prod_sig & 0x7F          # prodSig(6,0)
    else:
        out_man = ((prod_sig & 0x3F) << 1) # Cat(prodSig(5,0), 0)

    # exponent: biased_out = aExp + bExp - 1 + needShift
    out_exp = a_exp + b_exp + need_shift - 1

    # pack to 13 bits: S(1) E(5) M(7)
    return ((out_sign & 0x1) << 12) | ((out_exp & 0x1F) << 7) | (out_man & 0x7F)


# Format converters


def ieee_to_aligned_int(ieee_bits: int, fmt, anchor_exp: int, int_width: int) -> int:
    sign = (ieee_bits >> (fmt.ieeeWidth - 1)) & 1
    exp_field = (ieee_bits >> fmt.mantissaBits) & ((1 << fmt.expWidth) - 1)
    frac = ieee_bits & ((1 << fmt.mantissaBits) - 1)

    is_zero = exp_field == 0 and frac == 0
    unb_exp = exp_field - fmt.ieeeBias
    full_sig = ((1 if exp_field != 0 else 0) << fmt.mantissaBits) | frac

    shift_right = anchor_exp - unb_exp - (int_width - 1 - fmt.mantissaBits)
    sig_wide = full_sig

    if shift_right >= int_width:
        magnitude = 0
    elif shift_right >= 0:
        magnitude = sig_wide >> shift_right
    elif shift_right >= -(int_width - 1):
        magnitude = sig_wide << (-shift_right)
    else:
        magnitude = 0

    magnitude &= (1 << int_width) - 1
    if is_zero:
        return 0
    return wrap_signed(-magnitude if sign else magnitude, int_width)



def e4m3_prod_to_aligned_int(prod_bits: int, anchor_exp: int, int_width: int) -> int:
    sign = (prod_bits >> 12) & 1
    exp_bits = (prod_bits >> 7) & 0x1F
    man = prod_bits & 0x7F
    is_zero = exp_bits == 0

    if is_zero:
        return 0

    sig = (1 << 7) | man
    unb_exp = exp_bits - E4M3ProdFmt.bias
    rshift = anchor_exp - unb_exp
    sig_wide = sig << max(0, int_width - E4M3ProdFmt.sigWidth)

    if rshift < 0:
        shifted = (sig_wide << (-rshift)) & ((1 << int_width) - 1)
    else:
        shifted = (sig_wide >> rshift) & ((1 << int_width) - 1)

    magnitude = shifted
    return wrap_signed(-magnitude if sign else magnitude, int_width)



def aligned_int_to_bf16(int_in: int, anchor_exp: int, int_width: int) -> int:
    int_in = wrap_signed(int_in, int_width)
    if int_in == 0:
        return 0

    scale_exp = anchor_exp - (int_width - 1)
    value = math.ldexp(float(int_in), scale_exp)
    return f32_to_bf16_bits_rne(value)


# Optional output conversion stage

def bf16_scale_to_e4m3(bf16_bits: int, scale_exp: int) -> int:
    bf16_bits &= 0xFFFF

    sign = (bf16_bits >> 15) & 1
    exp_bf16 = (bf16_bits >> 7) & 0xFF
    frac_bf16 = bf16_bits & 0x7F

    is_zero = exp_bf16 == 0 and frac_bf16 == 0
    is_sub = exp_bf16 == 0 and frac_bf16 != 0
    is_inf = exp_bf16 == 0xFF and frac_bf16 == 0
    is_nan = exp_bf16 == 0xFF and frac_bf16 != 0

    if is_zero or is_sub or is_nan:
        return 0
    if is_inf:
        return E4M3_MAX_NEG if sign else E4M3_MAX_POS

    unb_exp = exp_bf16 - BF16.ieeeBias
    scaled_unb_exp = unb_exp + scale_exp

    mant8 = (1 << 7) | frac_bf16
    rounded_norm = round_right_shift4_rne(mant8)
    norm_carry = rounded_norm == 16

    final_unb_exp = scaled_unb_exp + (1 if norm_carry else 0)
    norm_mant = 0 if norm_carry else ((rounded_norm - 8) & 0x7)

    if final_unb_exp > 8:
        return E4M3_MAX_NEG if sign else E4M3_MAX_POS
    if final_unb_exp >= -6:
        return encode_e4m3_normal(sign, final_unb_exp, norm_mant)
    return 0



def output_conv_stage(bf16_bits: int, out_fmt_sel, scale_exp: int) -> int:
    bf16_sanitized = sanitize_bf16(bf16_bits)
    if out_fmt_sel.name == "OutBF16":
        return bf16_sanitized
    e4m3 = bf16_scale_to_e4m3(bf16_sanitized, scale_exp)
    return e4m3 & 0xFF
