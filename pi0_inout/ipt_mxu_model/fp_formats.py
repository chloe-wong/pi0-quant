from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import struct


@dataclass(frozen=True)
class AtlasFPType:
    name: str
    ieeeWidth: int
    expWidth: int
    mantissaBits: int
    ieeeBias: int

    @property
    def sigWidth(self) -> int:
        return 1 + self.mantissaBits


E4M3 = AtlasFPType("E4M3", ieeeWidth=8, expWidth=4, mantissaBits=3, ieeeBias=7)
BF16 = AtlasFPType("BF16", ieeeWidth=16, expWidth=8, mantissaBits=7, ieeeBias=127)


@dataclass(frozen=True)
class E4M3ProdFmtDesc:
    width: int = 13
    expWidth: int = 5
    mantissaBits: int = 7
    bias: int = 13

    @property
    def sigWidth(self) -> int:
        return 1 + self.mantissaBits


E4M3ProdFmt = E4M3ProdFmtDesc()


class AddendSel(Enum):
    UseAct = 0
    UseBias = 1
    UsePsum = 2


class OutputFmtSel(Enum):
    OutBF16 = 0
    OutE4M3 = 1


BF16_MAX_POS = 0x7F7F
BF16_MAX_NEG = 0xFF7F
E4M3_MAX_POS = 0x7E
E4M3_MAX_NEG = 0xFE


F32_SIGN_MASK = 0x80000000
F32_EXP_MASK = 0x7F800000
F32_FRAC_MASK = 0x007FFFFF


@dataclass(frozen=True)
class DecodedFloat:
    sign: int
    exp_field: int
    frac: int
    is_zero: bool
    is_sub: bool
    is_inf: bool
    is_nan: bool
    unb_exp: int | None
    sig: float | None
    value: float | None



def u32_to_float(x: int) -> float:
    return struct.unpack(">f", struct.pack(">I", x & 0xFFFFFFFF))[0]



def float_to_u32(x: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(x)))[0]



def sign_extend(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)



def clamp_signed(value: int, bits: int) -> int:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return max(lo, min(hi, value))



def wrap_signed(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value



def decode_e4m3(bits: int) -> DecodedFloat:
    bits &= 0xFF
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    frac = bits & 0x7

    is_zero = exp == 0 and frac == 0
    is_sub = exp == 0 and frac != 0
    is_inf = exp == 0xF and frac == 0
    is_nan = exp == 0xF and frac != 0

    if is_zero:
        v = -0.0 if sign else 0.0
        return DecodedFloat(sign, exp, frac, True, False, False, False, None, None, v)
    if is_sub:
        unb_exp = 1 - E4M3.ieeeBias
        sig = frac / 8.0
        v = ((-1.0) ** sign) * sig * (2.0 ** unb_exp)
        return DecodedFloat(sign, exp, frac, False, True, False, False, unb_exp, sig, v)
    if is_inf:
        v = float("-inf") if sign else float("inf")
        return DecodedFloat(sign, exp, frac, False, False, True, False, None, None, v)
    if is_nan:
        return DecodedFloat(sign, exp, frac, False, False, False, True, None, None, float("nan"))

    unb_exp = exp - E4M3.ieeeBias
    sig = 1.0 + frac / 8.0
    v = ((-1.0) ** sign) * sig * (2.0 ** unb_exp)
    return DecodedFloat(sign, exp, frac, False, False, False, False, unb_exp, sig, v)



def encode_e4m3_normal(sign: int, unb_exp: int, mant3: int) -> int:
    return ((sign & 1) << 7) | (((unb_exp + E4M3.ieeeBias) & 0xF) << 3) | (mant3 & 0x7)



def f32_to_bf16_bits_rne(x: float) -> int:
    u = float_to_u32(x)
    exp = (u >> 23) & 0xFF
    frac = u & 0x7FFFFF

    if exp == 0xFF:
        if frac != 0:
            return 0x7FC0
        return (u >> 16) & 0xFFFF

    upper = u >> 16
    lsb = upper & 1
    round_bias = 0x7FFF + lsb
    rounded = u + round_bias
    return (rounded >> 16) & 0xFFFF



def bf16_bits_to_f32(bits: int) -> float:
    return u32_to_float((bits & 0xFFFF) << 16)



def round_right_shift4_rne(x: int) -> int:
    trunc = (x >> 4) & 0xF
    guard = (x >> 3) & 1
    sticky = 1 if (x & 0x7) != 0 else 0
    lsb = trunc & 1
    inc = 1 if (guard and (sticky or lsb)) else 0
    return trunc + inc



def sanitize_bf16(bits: int) -> int:
    bits &= 0xFFFF
    sign = (bits >> 15) & 1
    exp = (bits >> 7) & 0xFF
    frac = bits & 0x7F

    is_zero = exp == 0 and frac == 0
    is_sub = exp == 0 and frac != 0
    is_inf = exp == 0xFF and frac == 0
    is_nan = exp == 0xFF and frac != 0

    if is_zero:
        return bits
    if is_sub or is_nan:
        return 0
    if is_inf:
        return BF16_MAX_NEG if sign else BF16_MAX_POS
    return bits
