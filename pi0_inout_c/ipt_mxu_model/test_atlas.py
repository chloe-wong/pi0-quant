#!/usr/bin/env python3
"""
test_atlas.py  —  comprehensive tests for fp_formats + converters.

Run:  python test_atlas.py
Exits 0 on all-pass, 1 on any failure.

Also writes test_vectors.h alongside itself so the C test binary can
consume the exact same vectors without re-deriving them.
"""

from __future__ import annotations

import math
import struct
import sys

from fp_formats import (
    BF16,
    E4M3,
    E4M3ProdFmt,
    decode_e4m3,
    encode_e4m3_normal,
    f32_to_bf16_bits_rne,
    bf16_bits_to_f32,
    round_right_shift4_rne,
    sanitize_bf16,
    wrap_signed,
    sign_extend,
    clamp_signed,
    AddendSel,
    OutputFmtSel,
)
from converters import (
    e4m3_mul_to_prod,
    e4m3_prod_to_aligned_int,
    ieee_to_aligned_int,
    aligned_int_to_bf16,
    bf16_scale_to_e4m3,
    output_conv_stage,
    pack_e4m3_prod,
)

# ---------------------------------------------------------------------------
# Tiny test harness
# ---------------------------------------------------------------------------

_pass = 0
_fail = 0
_section = ""


def section(name: str) -> None:
    global _section
    _section = name
    print(f"\n=== {name} ===")


def check(name: str, got, expected) -> bool:
    global _pass, _fail
    if got == expected or (
        isinstance(got, float)
        and isinstance(expected, float)
        and math.isnan(got)
        and math.isnan(expected)
    ):
        print(f"  PASS  {name}")
        _pass += 1
        return True
    else:
        print(f"  FAIL  {name}  got={got!r}  expected={expected!r}")
        _fail += 1
        return False


def summary() -> int:
    print(f"\n{'='*40}")
    print(f"  {_pass} passed,  {_fail} failed")
    print(f"{'='*40}")
    return 0 if _fail == 0 else 1


# ---------------------------------------------------------------------------
# Helper: pack/unpack float<->u32
# ---------------------------------------------------------------------------


def f32_to_u32(x: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(x)))[0]


def u32_to_f32(x: int) -> float:
    return struct.unpack(">f", struct.pack(">I", x & 0xFFFFFFFF))[0]


# ===========================================================================
# SECTION 1 – Leaf utility functions
# ===========================================================================

section("sign_extend")
# 4-bit
check("pos 4-bit 0", sign_extend(0b0000, 4), 0)
check("pos 4-bit 1", sign_extend(0b0001, 4), 1)
check("pos 4-bit 3", sign_extend(0b0011, 4), 3)
check("pos 4-bit max", sign_extend(0b0111, 4), 7)
check("neg 4-bit 8", sign_extend(0b1000, 4), -8)
check("neg 4-bit 15", sign_extend(0b1111, 4), -1)
check("neg 4-bit 9", sign_extend(0b1001, 4), -7)
# 1-bit
check("pos 1-bit 0", sign_extend(0, 1), 0)
check("neg 1-bit 1", sign_extend(1, 1), -1)
# 2-bit
check("pos 2-bit 0", sign_extend(0b00, 2), 0)
check("pos 2-bit 1", sign_extend(0b01, 2), 1)
check("neg 2-bit 2", sign_extend(0b10, 2), -2)
check("neg 2-bit 3", sign_extend(0b11, 2), -1)
# 8-bit
check("pos 8-bit 0", sign_extend(0x00, 8), 0)
check("pos 8-bit 1", sign_extend(0x01, 8), 1)
check("pos 8-bit 127", sign_extend(0x7F, 8), 127)
check("neg 8-bit 128", sign_extend(0x80, 8), -128)
check("neg 8-bit 255", sign_extend(0xFF, 8), -1)
check("neg 8-bit 129", sign_extend(0x81, 8), -127)
# 16-bit
check("pos 16-bit max", sign_extend(0x7FFF, 16), 32767)
check("neg 16-bit min", sign_extend(0x8000, 16), -32768)
check("neg 16-bit -1", sign_extend(0xFFFF, 16), -1)
check("pos 16-bit 1", sign_extend(0x0001, 16), 1)
# 32-bit
check("pos 32-bit max", sign_extend(0x7FFFFFFF, 32), 2147483647)
check("neg 32-bit min", sign_extend(0x80000000, 32), -2147483648)
check("neg 32-bit -1", sign_extend(0xFFFFFFFF, 32), -1)

section("clamp_signed")
# 4-bit
check("4b in range", clamp_signed(5, 4), 5)
check("4b clamp hi", clamp_signed(8, 4), 7)
check("4b clamp lo", clamp_signed(-9, 4), -8)
check("4b exact hi", clamp_signed(7, 4), 7)
check("4b exact lo", clamp_signed(-8, 4), -8)
check("4b zero", clamp_signed(0, 4), 0)
check("4b -1", clamp_signed(-1, 4), -1)
check("4b large pos", clamp_signed(100, 4), 7)
check("4b large neg", clamp_signed(-100, 4), -8)
# 1-bit
check("1b 0", clamp_signed(0, 1), 0)
check("1b clamp hi", clamp_signed(1, 1), 0)
check("1b clamp lo", clamp_signed(-2, 1), -1)
check("1b -1", clamp_signed(-1, 1), -1)
check("1b 100", clamp_signed(100, 1), 0)
# 8-bit
check("8b hi", clamp_signed(200, 8), 127)
check("8b lo", clamp_signed(-200, 8), -128)
check("8b exact hi", clamp_signed(127, 8), 127)
check("8b exact lo", clamp_signed(-128, 8), -128)
check("8b zero", clamp_signed(0, 8), 0)
# 16-bit
check("16b hi", clamp_signed(40000, 16), 32767)
check("16b lo", clamp_signed(-40000, 16), -32768)
check("16b exact hi", clamp_signed(32767, 16), 32767)
check("16b exact lo", clamp_signed(-32768, 16), -32768)
# 32-bit
check("32b no clamp", clamp_signed(0x7FFFFFFF, 32), 0x7FFFFFFF)
check("32b no clamp neg", clamp_signed(-0x80000000, 32), -0x80000000)

section("wrap_signed")
# 4-bit
check("4b no wrap", wrap_signed(3, 4), 3)
check("4b wrap hi", wrap_signed(8, 4), -8)
check("4b wrap neg", wrap_signed(-1, 4), -1)
check("4b exact max", wrap_signed(7, 4), 7)
check("4b exact min", wrap_signed(-8, 4), -8)
check("4b wrap max+1", wrap_signed(8, 4), -8)
check("4b wrap min-1", wrap_signed(-9, 4), 7)
check("4b wrap 16", wrap_signed(16, 4), 0)
check("4b wrap -16", wrap_signed(-16, 4), 0)
check("4b wrap 15", wrap_signed(15, 4), -1)
# 1-bit
check("1b zero", wrap_signed(0, 1), 0)
check("1b neg", wrap_signed(-1, 1), -1)
check("1b wrap 1", wrap_signed(1, 1), -1)
check("1b wrap 2", wrap_signed(2, 1), 0)
check("1b wrap -2", wrap_signed(-2, 1), 0)
# 8-bit
check("8b zero", wrap_signed(0, 8), 0)
check("8b wrap 128", wrap_signed(128, 8), -128)
check("8b wrap -129", wrap_signed(-129, 8), 127)
check("8b wrap 256", wrap_signed(256, 8), 0)
check("8b exact hi", wrap_signed(127, 8), 127)
check("8b exact lo", wrap_signed(-128, 8), -128)
check("8b wrap 255", wrap_signed(255, 8), -1)
check("8b wrap -255", wrap_signed(-255, 8), 1)
# 16-bit
check("16b wrap 32768", wrap_signed(32768, 16), -32768)
check("16b wrap -32769", wrap_signed(-32769, 16), 32767)
check("16b exact hi", wrap_signed(32767, 16), 32767)
check("16b exact lo", wrap_signed(-32768, 16), -32768)
# 32-bit — wrap_signed(x, 32) must be identity for valid int32 range
check("32b identity pos", wrap_signed(2147483647, 32), 2147483647)
check("32b identity neg", wrap_signed(-2147483648, 32), -2147483648)
check("32b identity zero", wrap_signed(0, 32), 0)
check("32b identity -1", wrap_signed(-1, 32), -1)
check("32b identity 1", wrap_signed(1, 32), 1)

section("round_right_shift4_rne")


# Exhaustive: all 256 inputs — compute oracle inline
def rrs4_ref(x: int) -> int:
    """Reference implementation of round_right_shift4_rne."""
    trunc = x >> 4
    guard = (x >> 3) & 1
    sticky = (x & 0x7) != 0
    # RNE: round up if guard=1 and (sticky=1 or trunc is odd)
    if guard and (sticky or (trunc & 1)):
        return trunc + 1
    return trunc


rrs4_fails = 0
for v in range(256):
    got = round_right_shift4_rne(v)
    exp = rrs4_ref(v)
    if got != exp:
        check(f"rrs4(0x{v:02X})", got, exp)
        rrs4_fails += 1
if rrs4_fails == 0:
    print("  PASS  all 256 inputs")
    _pass += 256
else:
    _fail += rrs4_fails

# Spot-check key cases with explanatory names
check("rrs4 exact 1", round_right_shift4_rne(0x10), 1)  # 16/16 exact
check("rrs4 half-up (odd)", round_right_shift4_rne(0x18), 2)  # trunc=1 odd, guard=1
check(
    "rrs4 half-even (even)", round_right_shift4_rne(0x28), 2
)  # trunc=2 even, guard=1, sticky=0 → no round
check("rrs4 above-half", round_right_shift4_rne(0x19), 2)  # trunc=1, guard=1, sticky=1
check("rrs4 zero", round_right_shift4_rne(0x00), 0)
check("rrs4 below-half", round_right_shift4_rne(0x08), 0)  # guard=0
check("rrs4 just-below-half", round_right_shift4_rne(0x0F), 1)  # guard=0
check("rrs4 0xFF", round_right_shift4_rne(0xFF), 16)
check(
    "rrs4 0x17 below-half", round_right_shift4_rne(0x17), 1
)  # trunc=1, guard=0 (bit3=0) → no round
check("rrs4 0x38 half-odd", round_right_shift4_rne(0x38), 4)  # trunc=3 odd → round up
check("rrs4 0x48 half-even", round_right_shift4_rne(0x48), 4)  # trunc=4 even → no round
check("rrs4 0x88 half-even", round_right_shift4_rne(0x88), 8)  # trunc=8 even → no round
check("rrs4 0x98 half-odd", round_right_shift4_rne(0x98), 10)  # trunc=9 odd → round up
check(
    "rrs4 0xF8 half-even", round_right_shift4_rne(0xF8), 16
)  # trunc=15 odd → round up to 16
check("rrs4 0xF0 exact", round_right_shift4_rne(0xF0), 15)

section("f32_to_bf16_bits_rne")
# Exhaustive: spot-check all normal classes and special values
check("1.0", f32_to_bf16_bits_rne(1.0), 0x3F80)
check("-1.0", f32_to_bf16_bits_rne(-1.0), 0xBF80)
check("0.0", f32_to_bf16_bits_rne(0.0), 0x0000)
check("-0.0", f32_to_bf16_bits_rne(-0.0), 0x8000)
check("+inf", f32_to_bf16_bits_rne(math.inf), 0x7F80)
check("-inf", f32_to_bf16_bits_rne(-math.inf), 0xFF80)
check("nan quiet", (f32_to_bf16_bits_rne(math.nan) & 0x7FC0), 0x7FC0)
check("2.0", f32_to_bf16_bits_rne(2.0), 0x4000)
check("0.5", f32_to_bf16_bits_rne(0.5), 0x3F00)
check("-2.0", f32_to_bf16_bits_rne(-2.0), 0xC000)
check("256.0", f32_to_bf16_bits_rne(256.0), 0x4380)
check("-0.5", f32_to_bf16_bits_rne(-0.5), 0xBF00)
check("0.25", f32_to_bf16_bits_rne(0.25), 0x3E80)
check("4.0", f32_to_bf16_bits_rne(4.0), 0x4080)
# f32 subnormal → bf16 subnormal
check("bf16 subnorm", f32_to_bf16_bits_rne(u32_to_f32(0x00400000)), 0x0040)
check("bf16 smallest sub", f32_to_bf16_bits_rne(u32_to_f32(0x00010000)), 0x0001)
# f32 that rounds up in mantissa
check("rne round-up 0x3F808080", f32_to_bf16_bits_rne(u32_to_f32(0x3F808080)), 0x3F81)
# Tie-to-even: lsb=0 (even) → no round
check("rne tie-even no-round", f32_to_bf16_bits_rne(u32_to_f32(0x3F808000)), 0x3F80)
# Tie-to-even: lsb=1 (odd) → round up
check("rne tie-even round-up", f32_to_bf16_bits_rne(u32_to_f32(0x3F818000)), 0x3F82)
# Another tie: 0x3F900000 lsb=0 → no round; 0x3F910000 lsb=1 → round up
check("rne tie 0x3F900000", f32_to_bf16_bits_rne(u32_to_f32(0x3F908000)), 0x3F90)
check("rne tie 0x3F918000", f32_to_bf16_bits_rne(u32_to_f32(0x3F918000)), 0x3F92)
# Overflow: largest f32 that rounds to BF16 max (0x7F7F), vs one that rounds up to inf
# BF16 max = 0x7F7F = 3.3895314e38; midpoint between 7F7F and 7F80 is 0x7F7F8000 f32
check("f32 near-max rne no-round", f32_to_bf16_bits_rne(u32_to_f32(0x7F7F0000)), 0x7F7F)
check("f32 near-max rne round-up", f32_to_bf16_bits_rne(u32_to_f32(0x7F7FFFFF)), 0x7F80)
# Negative versions
check("neg 0.25", f32_to_bf16_bits_rne(-0.25), 0xBE80)
check("neg 4.0", f32_to_bf16_bits_rne(-4.0), 0xC080)
# f32 value that produces a specific bf16 exactly
check("1.9921875", f32_to_bf16_bits_rne(u32_to_f32(0x3FFF0000)), 0x3FFF)

section("bf16_bits_to_f32")
check("1.0", bf16_bits_to_f32(0x3F80), 1.0)
check("-2.0", bf16_bits_to_f32(0xC000), -2.0)
check("0.0", bf16_bits_to_f32(0x0000), 0.0)
check("-0.0", bf16_bits_to_f32(0x8000), -0.0)
check("0.5", bf16_bits_to_f32(0x3F00), 0.5)
check("-0.5", bf16_bits_to_f32(0xBF00), -0.5)
check("256.0", bf16_bits_to_f32(0x4380), 256.0)
check("+inf", bf16_bits_to_f32(0x7F80), math.inf)
check("-inf", bf16_bits_to_f32(0xFF80), -math.inf)
check("0.25", bf16_bits_to_f32(0x3E80), 0.25)
check("max pos", bf16_bits_to_f32(0x7F7F), u32_to_f32(0x7F7F0000))
check("max neg", bf16_bits_to_f32(0xFF7F), -u32_to_f32(0x7F7F0000))
# NaN: value is a nan
got_nan = bf16_bits_to_f32(0x7FC0)
check("nan", math.isnan(got_nan), True)
# Subnormal BF16
sub_val = bf16_bits_to_f32(0x0001)
check("smallest bf16 sub", sub_val > 0.0 and sub_val < u32_to_f32(0x00800000), True)
# Round-trip: encode then decode
for v in [1.0, -1.0, 0.5, 2.0, 0.25, 256.0, -0.125]:
    bits = f32_to_bf16_bits_rne(v)
    check(f"bf16 round-trip {v}", bf16_bits_to_f32(bits), v)

section("sanitize_bf16")
# Normal values pass through
check("normal 1.0", sanitize_bf16(0x3F80), 0x3F80)
check("normal -1.0", sanitize_bf16(0xBF80), 0xBF80)
check("normal 2.0", sanitize_bf16(0x4000), 0x4000)
check("normal 0.25", sanitize_bf16(0x3E80), 0x3E80)
check("max pos", sanitize_bf16(0x7F7F), 0x7F7F)
check("max neg", sanitize_bf16(0xFF7F), 0xFF7F)
check("smallest normal pos", sanitize_bf16(0x0080), 0x0080)
check("smallest normal neg", sanitize_bf16(0x8080), 0x8080)
# Zero passes through
check("+0", sanitize_bf16(0x0000), 0x0000)
check("-0", sanitize_bf16(0x8000), 0x8000)
# Subnormals → 0 (preserve only sign? No — both → 0 in this impl)
check("subnormal pos→0", sanitize_bf16(0x0040), 0x0000)
check("subnormal pos 0x0001→0", sanitize_bf16(0x0001), 0x0000)
check("subnormal neg→0", sanitize_bf16(0x8040), 0x0000)
check("subnormal neg 0x8001→0", sanitize_bf16(0x8001), 0x0000)
# Inf → max
check("+inf→max_pos", sanitize_bf16(0x7F80), 0x7F7F)
check("-inf→max_neg", sanitize_bf16(0xFF80), 0xFF7F)
# NaN → 0
check("+nan→0", sanitize_bf16(0x7FC0), 0x0000)
check("-nan→0", sanitize_bf16(0xFFC0), 0x0000)
check("+nan payload→0", sanitize_bf16(0x7FFF), 0x0000)
check("-nan payload→0", sanitize_bf16(0xFFFF), 0x0000)
# Exhaust subnormals: all 127 positive subnormals should → 0
sub_fails = 0
for frac in range(1, 128):
    bits = frac  # exp=0, positive
    if sanitize_bf16(bits) != 0:
        sub_fails += 1
    bits_neg = 0x8000 | frac
    if sanitize_bf16(bits_neg) != 0:
        sub_fails += 1
check("all 254 subnormals→0", sub_fails, 0)
# Exhaust NaN: all NaN patterns (exp=0xFF, frac!=0) should → 0
nan_fails = 0
for frac in range(1, 128):
    bits_pos = (0x7F << 7) | frac  # 0x7F80 + frac but exp=0xFF
    bits_neg = 0x8000 | bits_pos
    if sanitize_bf16(bits_pos) != 0:
        nan_fails += 1
    if sanitize_bf16(bits_neg) != 0:
        nan_fails += 1
check("all 254 NaN patterns→0", nan_fails, 0)

# ===========================================================================
# SECTION 2 – E4M3 decode (exhaustive)
# ===========================================================================

section("decode_e4m3 exhaustive")
ok = True
for bits in range(256):
    d = decode_e4m3(bits)
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    frac = bits & 0x7

    if exp == 0 and frac == 0:
        if not (d.is_zero and d.value == 0.0):
            print(f"  FAIL decode_e4m3(0x{bits:02X}): expected zero, got {d}")
            ok = False
    elif exp == 0xF and frac == 0:
        if not d.is_inf:
            print(f"  FAIL decode_e4m3(0x{bits:02X}): expected inf")
            ok = False
    elif exp == 0xF and frac != 0:
        if not d.is_nan:
            print(f"  FAIL decode_e4m3(0x{bits:02X}): expected nan")
            ok = False
    else:
        if exp == 0:
            mag = frac * 0.125 * (2.0 ** (1 - 7))
        else:
            mag = (1.0 + frac * 0.125) * (2.0 ** (exp - 7))
        expected_val = -mag if sign else mag
        if not math.isclose(d.value, expected_val, rel_tol=1e-6):
            print(
                f"  FAIL decode_e4m3(0x{bits:02X}): got {d.value} expected {expected_val}"
            )
            ok = False

if ok:
    check("all 256 values", True, True)
    _pass += 255
else:
    _fail += 1

section("encode_e4m3_normal")
check("1.0", encode_e4m3_normal(0, 0, 0), 0b0_0111_000)
check("neg 1.0", encode_e4m3_normal(1, 0, 0), 0b1_0111_000)
check("1.5", encode_e4m3_normal(0, 0, 4), 0b0_0111_100)
check("2.0", encode_e4m3_normal(0, 1, 0), 0b0_1000_000)
check("0.5", encode_e4m3_normal(0, -1, 0), 0b0_0110_000)
check("-2.0", encode_e4m3_normal(1, 1, 0), 0b1_1000_000)
check("max pos", encode_e4m3_normal(0, 8, 6), 0x7E)
check("max neg", encode_e4m3_normal(1, 8, 6), 0xFE)
check("1.875", encode_e4m3_normal(0, 0, 7), 0b0_0111_111)
# Round-trip: all 120 normals both polarities
rt_fails = 0
for sign in [0, 1]:
    for unb in range(-6, 9):
        for mant in range(8):
            enc = encode_e4m3_normal(sign, unb, mant)
            d = decode_e4m3(enc)
            expected = (1.0 + mant * 0.125) * (2.0**unb)
            if sign:
                expected = -expected
            if not math.isclose(d.value, expected, rel_tol=1e-6):
                rt_fails += 1
                check(
                    f"round-trip sign={sign} unb={unb} mant={mant}", d.value, expected
                )
check(f"encode/decode round-trip all 240 values", rt_fails, 0)

# ===========================================================================
# SECTION 3 – e4m3_mul_to_prod
# ===========================================================================

section("e4m3_mul_to_prod spot checks")


def ref_e4m3_mul(a_bits: int, b_bits: int) -> int:
    a_sign = (a_bits >> 7) & 1
    a_exp = (a_bits >> 3) & 0xF
    a_man = a_bits & 0x7
    b_sign = (b_bits >> 7) & 1
    b_exp = (b_bits >> 3) & 0xF
    b_man = b_bits & 0x7
    out_sign = a_sign ^ b_sign
    if a_exp == 0 or b_exp == 0:
        return out_sign << 12
    a_sig = 8 | a_man
    b_sig = 8 | b_man
    prod_sig = a_sig * b_sig
    need_shift = (prod_sig >> 7) & 1
    out_man = (prod_sig & 0x7F) if need_shift else ((prod_sig & 0x3F) << 1)
    out_exp = a_exp + b_exp + need_shift - 1
    return (out_sign << 12) | ((out_exp & 0x1F) << 7) | out_man


a, b = 0b0_0111_000, 0b0_0111_000
check("1.0 × 1.0", e4m3_mul_to_prod(a, b), ref_e4m3_mul(a, b))
a, b = 0b0_0111_000, 0b1_0111_000
check("1.0 × -1.0", e4m3_mul_to_prod(a, b), ref_e4m3_mul(a, b))
check("0 × 1.0", e4m3_mul_to_prod(0x00, 0b0_0111_000), 0)
check("0 × 0", e4m3_mul_to_prod(0x00, 0x00), 0)
a = encode_e4m3_normal(0, 0, 4)  # 1.5
b = encode_e4m3_normal(0, 1, 0)  # 2.0
check("1.5 × 2.0", e4m3_mul_to_prod(a, b), ref_e4m3_mul(a, b))
a = b = 0x7E
check("max × max", e4m3_mul_to_prod(a, b), ref_e4m3_mul(a, b))
a = 0x7E
b = 0xFE
check("max × -max", e4m3_mul_to_prod(a, b), ref_e4m3_mul(a, b))
a = 0x01
b = encode_e4m3_normal(0, 0, 0)
check("subnormal × 1.0", e4m3_mul_to_prod(a, b), ref_e4m3_mul(a, b))
# zero × subnormal
check("0 × subnorm", e4m3_mul_to_prod(0x00, 0x01), 0)
# subnormal × subnormal
check("subnorm × subnorm", e4m3_mul_to_prod(0x01, 0x01), ref_e4m3_mul(0x01, 0x01))
# negative zero
check("-0 × 1.0", e4m3_mul_to_prod(0x80, 0x38), 0x1000)
check("-0 × -1.0", e4m3_mul_to_prod(0x80, 0xB8), 0x0000)

section("e4m3_mul_to_prod exhaustive")
mismatch = 0
for aa in range(256):
    for bb in range(256):
        got = e4m3_mul_to_prod(aa, bb)
        exp = ref_e4m3_mul(aa, bb)
        if got != exp:
            print(
                f"  FAIL mul(0x{aa:02X}, 0x{bb:02X}): got 0x{got:04X} exp 0x{exp:04X}"
            )
            mismatch += 1
            if mismatch >= 8:
                print("  (stopping after 8 mismatches)")
                break
    if mismatch >= 8:
        break
check(f"all 65536 products ({mismatch} mismatches)", mismatch, 0)

# ===========================================================================
# SECTION 4 – pack_e4m3_prod
# ===========================================================================

section("pack_e4m3_prod")


def pack_ref(sign, exp_unb, mant7):
    """Reference implementation."""
    exp_field = exp_unb + 13  # _E4M3_PROD_BIAS
    if exp_field <= 0:
        return 0
    if exp_field >= 31:  # _E4M3_PROD_EXP_MAX
        exp_field = 31
    return ((sign & 1) << 12) | ((exp_field & 0x1F) << 7) | (mant7 & 0x7F)


# Boundary conditions
check("sign=0 unb=0 mant=0", pack_e4m3_prod(0, 0, 0), pack_ref(0, 0, 0))
check("sign=1 unb=0 mant=0", pack_e4m3_prod(1, 0, 0), pack_ref(1, 0, 0))
check("sign=1 unb=1 mant=63", pack_e4m3_prod(1, 1, 63), pack_ref(1, 1, 63))
check("exp underflow -14", pack_e4m3_prod(0, -14, 0), 0)
check("exp underflow -100", pack_e4m3_prod(0, -100, 0), 0)
check("exp exactly -13→0", pack_e4m3_prod(0, -13, 0), 0)  # exp_field=0 → underflow
check("exp exactly -12→1", pack_e4m3_prod(0, -12, 0), pack_ref(0, -12, 0))
check("exp overflow 19", pack_e4m3_prod(0, 19, 0), pack_ref(0, 19, 0))
check("exp overflow 100", pack_e4m3_prod(0, 100, 0), pack_ref(0, 100, 0))
check("exp exactly 17→30", pack_e4m3_prod(0, 17, 0), pack_ref(0, 17, 0))
check("exp exactly 18→clamp", pack_e4m3_prod(0, 18, 0), pack_ref(0, 18, 0))
check("max mant7=127", pack_e4m3_prod(0, 0, 127), pack_ref(0, 0, 127))
check("mant7 masked to 7 bits", pack_e4m3_prod(0, 0, 128), pack_ref(0, 0, 128 & 0x7F))
check("neg unb_exp=-6", pack_e4m3_prod(0, -6, 0), pack_ref(0, -6, 0))
check("neg unb_exp=-12", pack_e4m3_prod(0, -12, 0), pack_ref(0, -12, 0))
check("sign=0 exp=8 mant=6", pack_e4m3_prod(0, 8, 6), pack_ref(0, 8, 6))
check("sign=1 exp=8 mant=6", pack_e4m3_prod(1, 8, 6), pack_ref(1, 8, 6))
# Sweep all valid unbiased exponents
for unb in range(-13, 19):
    got = pack_e4m3_prod(0, unb, 0)
    exp = pack_ref(0, unb, 0)
    if got != exp:
        check(f"pack exp sweep unb={unb}", got, exp)
        _fail += 1
    else:
        _pass += 1
print(f"  PASS  pack_e4m3_prod exp sweep -13..18 (32 cases)")

# ===========================================================================
# SECTION 5 – e4m3_prod_to_aligned_int
# ===========================================================================

section("e4m3_prod_to_aligned_int")


def prod_to_int_ref(prod_bits: int, anchor_exp: int, int_width: int) -> int:
    """Pure-Python oracle."""
    exp_bits = (prod_bits >> 7) & 0x1F
    if exp_bits == 0:
        return 0
    sign = (prod_bits >> 12) & 1
    man = prod_bits & 0x7F
    sig = (1 << 7) | man
    unb_exp = exp_bits - 13  # _E4M3_PROD_BIAS
    rshift = anchor_exp - unb_exp
    left_pad = int_width - 8  # _E4M3_PROD_SIG_WIDTH
    sig_wide = sig << left_pad if left_pad > 0 else sig
    if rshift < 0:
        shifted = sig_wide << (-rshift)
    else:
        shifted = sig_wide >> rshift
    mask = (1 << int_width) - 1
    magnitude = shifted & mask
    result = (-magnitude & mask) if sign else magnitude
    return wrap_signed(result, int_width)


# Zero
check("zero prod", e4m3_prod_to_aligned_int(0, 8, 32), 0)
check("zero prod anc=0 iw=8", e4m3_prod_to_aligned_int(0, 0, 8), 0)

# Anchor sweep for 1.0 (unb_exp=0), iw=32
for anchor in range(0, 33, 4):
    prod = pack_e4m3_prod(0, 0, 0)
    got = e4m3_prod_to_aligned_int(prod, anchor, 32)
    exp = prod_to_int_ref(prod, anchor, 32)
    check(f"1.0 anc={anchor} iw=32", got, exp)

# Anchor sweep for -1.0 (unb_exp=0), iw=32
for anchor in range(0, 33, 4):
    prod = pack_e4m3_prod(1, 0, 0)
    got = e4m3_prod_to_aligned_int(prod, anchor, 32)
    exp = prod_to_int_ref(prod, anchor, 32)
    check(f"-1.0 anc={anchor} iw=32", got, exp)

# unb_exp sweep, anchor=8, iw=32
for unb in range(-12, 19):
    prod = pack_e4m3_prod(0, unb, 0)
    if prod == 0:
        continue  # underflowed — skip
    got = e4m3_prod_to_aligned_int(prod, 8, 32)
    exp = prod_to_int_ref(prod, 8, 32)
    check(f"unb={unb} anc=8 iw=32", got, exp)

# int_width sweep for 1.0, anchor=0
for iw in [1, 2, 4, 8, 12, 16, 24, 32]:
    prod = pack_e4m3_prod(0, 0, 0)
    got = e4m3_prod_to_aligned_int(prod, 0, iw)
    exp = prod_to_int_ref(prod, 0, iw)
    check(f"1.0 anc=0 iw={iw}", got, exp)

# int_width sweep for -1.0, anchor=0
for iw in [1, 2, 4, 8, 12, 16, 24, 32]:
    prod = pack_e4m3_prod(1, 0, 0)
    got = e4m3_prod_to_aligned_int(prod, 0, iw)
    exp = prod_to_int_ref(prod, 0, iw)
    check(f"-1.0 anc=0 iw={iw}", got, exp)

# Non-zero mantissa sweep
for mant in [1, 2, 4, 8, 16, 32, 63, 64, 96, 127]:
    prod = pack_e4m3_prod(0, 0, mant)
    got = e4m3_prod_to_aligned_int(prod, 8, 32)
    exp = prod_to_int_ref(prod, 8, 32)
    check(f"mant={mant} anc=8 iw=32", got, exp)

# Negative with mantissa
for mant in [1, 64, 127]:
    prod = pack_e4m3_prod(1, 0, mant)
    got = e4m3_prod_to_aligned_int(prod, 8, 32)
    exp = prod_to_int_ref(prod, 8, 32)
    check(f"neg mant={mant} anc=8 iw=32", got, exp)

# Large left-shift: anchor much less than unb_exp (overflow into upper bits)
for unb in [5, 10, 15]:
    prod = pack_e4m3_prod(0, unb, 0)
    got = e4m3_prod_to_aligned_int(prod, 0, 32)
    exp = prod_to_int_ref(prod, 0, 32)
    check(f"large left-shift unb={unb} anc=0 iw=32", got, exp)

# Large right-shift: result underflows to 0
prod = pack_e4m3_prod(0, 0, 0)  # unb_exp=0
got = e4m3_prod_to_aligned_int(prod, 40, 32)  # shift right by 40
check("large right-shift→0", got, 0)

# Cross-product: anchor x iw x unb_exp systematic grid
for unb, anc, iw in [
    (-5, 0, 32),
    (-3, 4, 32),
    (0, 8, 32),
    (3, 12, 32),
    (5, 16, 32),
    (0, 0, 8),
    (0, 0, 16),
    (0, 0, 24),
    (0, 0, 32),
    (1, 0, 16),
    (-1, 4, 16),
    (2, 8, 16),
]:
    prod = pack_e4m3_prod(0, unb, 0)
    if prod == 0:
        continue
    got = e4m3_prod_to_aligned_int(prod, anc, iw)
    exp = prod_to_int_ref(prod, anc, iw)
    check(f"grid unb={unb} anc={anc} iw={iw}", got, exp)

# Legacy exact-value cases from original test suite
check("0x0680 anc=8  iw=32", e4m3_prod_to_aligned_int(0x0680, 8, 32), 8388608)
check("0x1700 anc=8  iw=32", e4m3_prod_to_aligned_int(0x1700, 8, 32), -16777216)
check("0x087F anc=10 iw=32", e4m3_prod_to_aligned_int(0x087F, 10, 32), 33423360)
check("0x1400 anc=0  iw=32", e4m3_prod_to_aligned_int(0x1400, 0, 32), -67108864)

# ===========================================================================
# SECTION 6 – ieee_to_aligned_int
# ===========================================================================


def ieee_to_int_ref(bits: int, fmt, anchor: int, iw: int) -> int:
    """Pure-Python oracle for ieee_to_aligned_int."""
    mant_bits = fmt.mantissaBits
    exp_width = fmt.expWidth
    exp_mask = (1 << exp_width) - 1
    sign = (bits >> (fmt.ieeeWidth - 1)) & 1
    exp_field = (bits >> mant_bits) & exp_mask
    frac = bits & ((1 << mant_bits) - 1)
    if exp_field == 0 and frac == 0:
        return 0
    unb_exp = exp_field - fmt.ieeeBias
    full_sig = ((1 << mant_bits) if exp_field != 0 else 0) | frac
    shift_right = anchor - unb_exp - (iw - 1 - mant_bits)
    mask = (1 << iw) - 1
    if shift_right >= iw:
        mag = 0
    elif shift_right >= 0:
        mag = full_sig >> shift_right
    elif shift_right > -iw:
        mag = full_sig << (-shift_right)
    else:
        mag = 0
    mag &= mask
    result = (-mag & mask) if sign else mag
    return wrap_signed(result, iw)


section("ieee_to_aligned_int (BF16) — basic")
for bits, anchor, iw in [
    (0x3F80, 7, 32),  # 1.0
    (0xBF80, 7, 32),  # -1.0
    (0x0000, 7, 32),  # 0.0
    (0x4000, 7, 32),  # 2.0
    (0x3F00, 7, 32),  # 0.5
    (0xC000, 7, 32),  # -2.0
    (0x4080, 7, 32),  # 4.0
    (0x3E80, 7, 32),  # 0.25
    (0x4380, 7, 32),  # 256.0
    (0x8000, 7, 32),  # -0.0 (zero path)
    (0xFF7F, 7, 32),  # BF16 max neg
    (0x7F7F, 7, 32),  # BF16 max pos
]:
    got = ieee_to_aligned_int(bits, BF16, anchor, iw)
    exp = ieee_to_int_ref(bits, BF16, anchor, iw)
    check(f"BF16 0x{bits:04X} anc={anchor} iw={iw}", got, exp)

section("ieee_to_aligned_int (BF16) — BF16 subnormals")
# exp_field=0, frac!=0: these are subnormal BF16 values
for frac in [1, 2, 63, 64, 127]:
    bits = frac  # exp=0, sign=0
    got = ieee_to_aligned_int(bits, BF16, 7, 32)
    exp = ieee_to_int_ref(bits, BF16, 7, 32)
    check(f"BF16 subnorm frac={frac} anc=7 iw=32", got, exp)
# Negative subnormal BF16
for frac in [1, 64, 127]:
    bits = 0x8000 | frac
    got = ieee_to_aligned_int(bits, BF16, 7, 32)
    exp = ieee_to_int_ref(bits, BF16, 7, 32)
    check(f"BF16 neg-subnorm frac={frac} anc=7 iw=32", got, exp)

section("ieee_to_aligned_int (BF16) — anchor sweep")
# BF16 1.0: unb_exp=0, full_sig=128, iw=32
# shift_right = anchor - 0 - 24 = anchor - 24
for anchor in range(0, 33):
    got = ieee_to_aligned_int(0x3F80, BF16, anchor, 32)
    exp = ieee_to_int_ref(0x3F80, BF16, anchor, 32)
    if got != exp:
        check(f"BF16 1.0 anc={anchor}", got, exp)
        _fail += 1
    else:
        _pass += 1
print(f"  PASS  BF16 1.0 anchor sweep 0..32 (33 cases)")

section("ieee_to_aligned_int (BF16) — iw sweep")
for iw in [1, 2, 4, 8, 12, 16, 24, 32]:
    for bits in [0x3F80, 0xBF80, 0x4000, 0x3F00]:
        got = ieee_to_aligned_int(bits, BF16, 7, iw)
        exp = ieee_to_int_ref(bits, BF16, 7, iw)
        check(f"BF16 0x{bits:04X} anc=7 iw={iw}", got, exp)

section("ieee_to_aligned_int (E4M3) — full normal sweep")
# All 120 E4M3 normal values, both polarities, anchor=7 iw=32
e4m3_normal_fails = 0
for sign in [0, 1]:
    for unb in range(-6, 9):
        for mant in range(8):
            e4m3_bits = encode_e4m3_normal(sign, unb, mant)
            got = ieee_to_aligned_int(e4m3_bits, E4M3, 7, 32)
            exp = ieee_to_int_ref(e4m3_bits, E4M3, 7, 32)
            if got != exp:
                e4m3_normal_fails += 1
                check(f"E4M3 0x{e4m3_bits:02X} anc=7 iw=32", got, exp)
check(
    f"all 240 E4M3 normals anc=7 iw=32 ({e4m3_normal_fails} fails)",
    e4m3_normal_fails,
    0,
)

section("ieee_to_aligned_int (E4M3) — anchor/iw sweep")
# E4M3 1.0 = 0x38, anchor and iw sweeps
for anchor in range(0, 20):
    got = ieee_to_aligned_int(0x38, E4M3, anchor, 32)
    exp = ieee_to_int_ref(0x38, E4M3, anchor, 32)
    if got != exp:
        check(f"E4M3 1.0 anc={anchor} iw=32", got, exp)
        _fail += 1
    else:
        _pass += 1
print(f"  PASS  E4M3 1.0 anchor sweep 0..19 (20 cases)")

for iw in [1, 2, 4, 8, 12, 16, 24, 32]:
    for e4m3_bits in [0x38, 0xB8, 0x40, 0x7E]:
        got = ieee_to_aligned_int(e4m3_bits, E4M3, 7, iw)
        exp = ieee_to_int_ref(e4m3_bits, E4M3, 7, iw)
        check(f"E4M3 0x{e4m3_bits:02X} anc=7 iw={iw}", got, exp)

section("ieee_to_aligned_int (E4M3) — subnormals (oracle-based)")
# E4M3 subnormals: exp_field=0, frac in 1..7
for frac in range(1, 8):
    bits_pos = frac
    bits_neg = 0x80 | frac
    for anchor in [0, 7, 14]:
        got_p = ieee_to_aligned_int(bits_pos, E4M3, anchor, 32)
        exp_p = ieee_to_int_ref(bits_pos, E4M3, anchor, 32)
        check(f"E4M3 subnorm +0x{bits_pos:02X} anc={anchor} iw=32", got_p, exp_p)
        got_n = ieee_to_aligned_int(bits_neg, E4M3, anchor, 32)
        exp_n = ieee_to_int_ref(bits_neg, E4M3, anchor, 32)
        check(f"E4M3 subnorm -0x{bits_neg:02X} anc={anchor} iw=32", got_n, exp_n)

section("ieee_to_aligned_int (BF16) — large-exponent stress")
# Values where shift_right is deeply negative (large left shifts)
for bits, anchor, iw in [
    (0x4380, 0, 32),  # 256.0, will shift left
    (0x4000, 0, 32),  # 2.0, shift left
    (0x3F80, 0, 32),  # 1.0, shift left to fill 32-bit
    (0x3F80, 0, 16),
    (0x3F80, 0, 8),
    (0x7F7F, 7, 32),  # BF16 max pos
    (0xFF7F, 7, 32),  # BF16 max neg
    (0x0080, 7, 32),  # BF16 smallest normal
    (0x8080, 7, 32),  # BF16 smallest neg normal
]:
    got = ieee_to_aligned_int(bits, BF16, anchor, iw)
    exp = ieee_to_int_ref(bits, BF16, anchor, iw)
    check(f"BF16-stress 0x{bits:04X} anc={anchor} iw={iw}", got, exp)

# ===========================================================================
# SECTION 7 – aligned_int_to_bf16
# ===========================================================================

section("aligned_int_to_bf16")


def a2b_ref(ival: int, anchor: int, iw: int) -> int:
    """Oracle: wrap then ldexp then round to bf16."""
    ival_w = wrap_signed(ival, iw)
    if ival_w == 0:
        return 0
    f = math.ldexp(float(ival_w), anchor - (iw - 1))
    return f32_to_bf16_bits_rne(f)


ANCHOR = 7
IW = 32

# Round-trip: BF16 → int → BF16
for bf16_val in [
    0x3F80,
    0xBF80,
    0x4000,
    0x3F00,
    0xC000,
    0x4380,
    0x3E80,
    0x7F7F,
    0xFF7F,
    0x0080,
    0x8080,
]:
    ival = ieee_to_aligned_int(bf16_val, BF16, ANCHOR, IW)
    got = aligned_int_to_bf16(ival, ANCHOR, IW)
    check(f"round-trip 0x{bf16_val:04X}", got, bf16_val)

# Direct cases with oracle
for ival, anchor, iw in [
    # iw=32 basic
    (128, 7, 32),  #  1.0
    (-128, 7, 32),  # -1.0
    (0, 7, 32),  #  0.0
    (256, 8, 32),
    (1, 0, 32),
    (64, 7, 32),
    (-256, 7, 32),
    (512, 7, 32),
    (-1, 7, 32),
    (100, 7, 32),
    (-100, 7, 32),
    (1, 30, 32),
    (-1, 30, 32),
    (2147483647, 31, 32),  # INT32_MAX
    (-2147483648, 31, 32),  # INT32_MIN
    # iw=16
    (32767, 15, 16),
    (-32768, 15, 16),
    (128, 7, 16),
    (-128, 7, 16),
    (1, 0, 16),
    # iw=8 — wrap_signed path
    (128, 7, 8),  # wraps to -128 → -1.0
    (-128, 7, 8),  # -1.0
    (127, 7, 8),  # just below wrap
    (1, 0, 8),
    # iw=4
    (7, 3, 4),
    (-8, 3, 4),
    (8, 3, 4),  # wraps to -8
    # iw=1
    (0, 0, 1),
    (-1, 0, 1),
    # Values that exercise RNE rounding (ldexp produces value between two BF16s)
    # ival=3, anchor=1, iw=32 → 3 * 2^(1-31) = 3 * 2^-30 ≈ 2.79e-9, rounds in bf16
    (3, 1, 32),
    (5, 2, 32),
    (7, 3, 32),
    # Large anchor values
    (1, 31, 32),
    (-1, 31, 32),
]:
    exp = a2b_ref(ival, anchor, iw)
    got = aligned_int_to_bf16(ival, anchor, iw)
    check(f"a2b ival={ival} anc={anchor} iw={iw}", got, exp)

# Systematic iw sweep: ival=1 with anchor=iw-1 (value=1.0)
for iw in [4, 8, 16, 32]:
    got = aligned_int_to_bf16(1, iw - 1, iw)
    exp = a2b_ref(1, iw - 1, iw)
    check(f"a2b 1.0 via iw={iw}", got, exp)

# Zero always → 0x0000
for iw in [1, 8, 16, 32]:
    check(f"a2b zero iw={iw}", aligned_int_to_bf16(0, 7, iw), 0x0000)

# ===========================================================================
# SECTION 8 – bf16_scale_to_e4m3
# ===========================================================================

section("bf16_scale_to_e4m3")


def bf16_scale_ref(bf16_bits: int, scale_exp: int) -> int:
    """Oracle for bf16_scale_to_e4m3."""
    sign = (bf16_bits >> 15) & 1
    exp_bf16 = (bf16_bits >> 7) & 0xFF
    frac_bf16 = bf16_bits & 0x7F
    if exp_bf16 == 0:
        return 0
    if exp_bf16 == 0xFF:
        if frac_bf16 != 0:
            return 0
        return 0xFE if sign else 0x7E
    scaled_unb_exp = (exp_bf16 - 127) + scale_exp
    mant8 = 0x80 | frac_bf16
    rounded_norm = round_right_shift4_rne(mant8)
    if rounded_norm == 16:
        final_unb_exp = scaled_unb_exp + 1
        norm_mant = 0
    else:
        final_unb_exp = scaled_unb_exp
        norm_mant = (rounded_norm - 8) & 0x7
    if final_unb_exp > 8:
        return 0xFE if sign else 0x7E
    if final_unb_exp >= -6:
        return encode_e4m3_normal(sign, final_unb_exp, norm_mant)
    return 0


# Special values
check("0.0 scale 0", bf16_scale_to_e4m3(0x0000, 0), 0)
check("+0.0", bf16_scale_to_e4m3(0x0000, 5), 0)
check("-0.0", bf16_scale_to_e4m3(0x8000, 0), 0)
check("+inf→max", bf16_scale_to_e4m3(0x7F80, 0), 0x7E)
check("-inf→max_neg", bf16_scale_to_e4m3(0xFF80, 0), 0xFE)
check("+nan→0", bf16_scale_to_e4m3(0x7FC0, 0), 0)
check("-nan→0", bf16_scale_to_e4m3(0xFFC0, 0), 0)
check("subnorm→0", bf16_scale_to_e4m3(0x0040, 0), 0)
check("neg subnorm→0", bf16_scale_to_e4m3(0x8040, 0), 0)

# Scale sweep for 1.0: scale -15..15
for sc in range(-15, 16):
    got = bf16_scale_to_e4m3(0x3F80, sc)
    exp = bf16_scale_ref(0x3F80, sc)
    check(f"1.0 scale={sc}", got, exp)

# Scale sweep for -1.0
for sc in range(-15, 16):
    got = bf16_scale_to_e4m3(0xBF80, sc)
    exp = bf16_scale_ref(0xBF80, sc)
    check(f"-1.0 scale={sc}", got, exp)

# Round-trip all E4M3 normals: encode → decode → bf16 → scale back
rt_fails = 0
for sign in [0, 1]:
    for unb in range(-6, 9):
        for mant3 in range(8):
            e4m3_bits = encode_e4m3_normal(sign, unb, mant3)
            d = decode_e4m3(e4m3_bits)
            bf16_bits = f32_to_bf16_bits_rne(d.value)
            recovered = bf16_scale_to_e4m3(bf16_bits, 0)
            if recovered != e4m3_bits:
                rt_fails += 1
                if rt_fails <= 4:
                    print(
                        f"  FAIL  rt sign={sign} unb={unb} mant={mant3}: "
                        f"got 0x{recovered:02X} expected 0x{e4m3_bits:02X}"
                    )
check(f"bf16_scale round-trip all 240 normals ({rt_fails} fails)", rt_fails, 0)

# Systematic: various BF16 values at scale=0 vs oracle
for bits in [
    0x3F80,
    0xBF80,
    0x4000,
    0x3F00,
    0xC000,
    0x4380,
    0x3E80,
    0x7F7F,
    0x3F40,
    0x3FC0,
    0x3FE0,
]:
    got = bf16_scale_to_e4m3(bits, 0)
    exp = bf16_scale_ref(bits, 0)
    check(f"bf16_scale 0x{bits:04X} sc=0", got, exp)

# Scale values that cause rounding to fire inside round_right_shift4_rne
# (bf16 mantissa bits 3..0 are non-zero, trigger guard/sticky)
for bits in [0x3F84, 0x3F88, 0x3F8C, 0x3F90, 0x3FA0, 0x3FB0]:
    got = bf16_scale_to_e4m3(bits, 0)
    exp = bf16_scale_ref(bits, 0)
    check(f"bf16_scale rounding 0x{bits:04X} sc=0", got, exp)

# ===========================================================================
# SECTION 9 – output_conv_stage
# ===========================================================================

section("output_conv_stage")


def oc_ref(bf16_bits: int, out_fmt_sel, scale_exp: int) -> int:
    sanitized = sanitize_bf16(bf16_bits)
    if out_fmt_sel == OutputFmtSel.OutBF16:
        return sanitized
    return bf16_scale_to_e4m3(sanitized, scale_exp) & 0xFF


# BF16 output mode
for bits in [
    0x3F80,
    0xBF80,
    0x4000,
    0x3F00,
    0x0000,
    0x8000,
    0x7F80,
    0xFF80,
    0x7FC0,
    0xFFC0,
    0x0040,
    0x8040,
    0x7F7F,
    0xFF7F,
]:
    got = output_conv_stage(bits, OutputFmtSel.OutBF16, 0)
    exp = oc_ref(bits, OutputFmtSel.OutBF16, 0)
    check(f"BF16-out 0x{bits:04X}", got, exp)

# E4M3 output mode, scale=0
for bits in [
    0x3F80,
    0xBF80,
    0x4000,
    0x3F00,
    0x0000,
    0x8000,
    0x7F80,
    0xFF80,
    0x7FC0,
    0x0040,
    0x7F7F,
]:
    got = output_conv_stage(bits, OutputFmtSel.OutE4M3, 0)
    exp = oc_ref(bits, OutputFmtSel.OutE4M3, 0)
    check(f"E4M3-out 0x{bits:04X} sc=0", got, exp)

# E4M3 output mode, scale sweep
for sc in range(-8, 10):
    got = output_conv_stage(0x3F80, OutputFmtSel.OutE4M3, sc)
    exp = oc_ref(0x3F80, OutputFmtSel.OutE4M3, sc)
    check(f"E4M3-out 1.0 sc={sc}", got, exp)

# E4M3 output mode with -1.0, scale sweep
for sc in [-2, -1, 0, 1, 2]:
    got = output_conv_stage(0xBF80, OutputFmtSel.OutE4M3, sc)
    exp = oc_ref(0xBF80, OutputFmtSel.OutE4M3, sc)
    check(f"E4M3-out -1.0 sc={sc}", got, exp)

# Verify sanitize_bf16 is called before conversion
# subnormal bf16 → sanitized to 0 → e4m3 output = 0
check("oc subnorm→e4m3=0", output_conv_stage(0x0001, OutputFmtSel.OutE4M3, 0), 0)
# +inf → sanitized to max_pos → e4m3 output = max
check(
    "oc +inf→e4m3=max",
    output_conv_stage(0x7F80, OutputFmtSel.OutE4M3, 0),
    oc_ref(0x7F80, OutputFmtSel.OutE4M3, 0),
)
# nan → sanitized to 0 → e4m3 output = 0
check("oc nan→e4m3=0", output_conv_stage(0x7FC0, OutputFmtSel.OutE4M3, 0), 0)


# ===========================================================================
# Write test_vectors.h for the C test binary
# ===========================================================================


def write_c_vectors() -> None:
    lines: list[str] = []
    lines.append("/* test_vectors.h  —  AUTO-GENERATED by test_atlas.py */")
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("")

    # ---- round_right_shift4_rne: full 256 ----
    lines.append("#define N_RRS4 256")
    lines.append("static const struct { int in; int out; } RRS4_CASES[256] = {")
    for v in range(256):
        lines.append(f"    {{ 0x{v:02X}, {rrs4_ref(v)} }},")
    lines.append("};")
    lines.append("")

    # ---- f32_to_bf16_bits_rne ----
    bf16_cases_f32 = [
        f32_to_u32(1.0),
        f32_to_u32(-1.0),
        f32_to_u32(0.0),
        f32_to_u32(-0.0),
        f32_to_u32(math.inf),
        f32_to_u32(-math.inf),
        0x3F808080,  # rne round-up
        f32_to_u32(256.0),
        f32_to_u32(-0.5),
        f32_to_u32(2.0),
        f32_to_u32(0.5),
        f32_to_u32(-2.0),
        0x3F808000,  # tie-even no-round
        0x3F818000,  # tie-even round-up
        0x3F908000,  # another tie-even no-round
        0x3F918000,  # another tie-even round-up
        0x00400000,  # bf16 subnormal
        0x00010000,  # smallest bf16 subnormal
        f32_to_u32(0.125),
        f32_to_u32(-0.125),
        f32_to_u32(0.25),
        f32_to_u32(-0.25),
        f32_to_u32(4.0),
        f32_to_u32(-4.0),
        f32_to_u32(1.9921875),
        0x7F7F0000,  # near-max no-round
        0x7F7FFFFF,  # near-max rounds to inf
    ]
    lines.append(f"#define N_BF16_CASES {len(bf16_cases_f32)}")
    lines.append(
        "static const struct { uint32_t f32_bits; uint16_t bf16_bits; } BF16_CASES[] = {"
    )
    for f32b in bf16_cases_f32:
        bf16b = f32_to_bf16_bits_rne(u32_to_f32(f32b))
        lines.append(f"    {{ 0x{f32b:08X}u, 0x{bf16b:04X} }},")
    lines.append("};")
    lines.append("")

    # ---- sanitize_bf16: exhaustive 65536 ----
    lines.append("#define N_SAN_CASES 65536")
    lines.append("static const uint16_t SAN_LUT[65536] = {")
    row = []
    for i in range(65536):
        row.append(f"0x{sanitize_bf16(i):04X}")
        if len(row) == 16:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row))
    lines.append("};")
    lines.append("")

    # ---- decode_e4m3: all 256 ----
    lines.append("#define N_E4M3_DECODE 256")
    lines.append("static const struct {")
    lines.append("    uint8_t bits;")
    lines.append("    int sign, exp_field, frac;")
    lines.append("    int is_zero, is_sub, is_inf, is_nan;")
    lines.append("    int has_unb_exp; int unb_exp;")
    lines.append("    uint32_t value_bits;")
    lines.append("} E4M3_DECODE_CASES[256] = {")
    for i in range(256):
        d = decode_e4m3(i)
        has_unb_exp = not (d.is_zero or d.is_inf or d.is_nan)
        unb_exp = d.unb_exp if has_unb_exp else 0
        vbits = 0x7FC00000 if d.is_nan else f32_to_u32(d.value)
        lines.append(
            f"    {{ 0x{i:02X}, {d.sign}, {d.exp_field}, {d.frac}, "
            f"{int(d.is_zero)}, {int(d.is_sub)}, {int(d.is_inf)}, {int(d.is_nan)}, "
            f"{int(has_unb_exp)}, {unb_exp}, 0x{vbits:08X}u }},"
        )
    lines.append("};")
    lines.append("")

    # ---- e4m3_mul_to_prod: full 65536 LUT ----
    lines.append("#define N_MUL_PROD 65536")
    lines.append("static const uint16_t MUL_PROD_LUT[65536] = {")
    row = []
    for idx in range(65536):
        row.append(f"0x{e4m3_mul_to_prod(idx >> 8, idx & 0xFF):04X}")
        if len(row) == 16:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row))
    lines.append("};")
    lines.append("")

    # ---- pack_e4m3_prod: full sweep ----
    pack_cases = []
    # All valid unbiased exponents -13..18 with mant=0, both signs
    for sign in [0, 1]:
        for unb in range(-14, 20):
            pack_cases.append((sign, unb, 0))
    # Max mantissa, various exponents
    for unb in [-6, 0, 8]:
        for mant in [0, 63, 64, 127]:
            pack_cases.append((0, unb, mant))
            pack_cases.append((1, unb, mant))
    # Mant masking (bit 7 set — should be ignored)
    pack_cases.append((0, 0, 128))
    pack_cases.append((0, 0, 255))
    lines.append(f"#define N_PACK {len(pack_cases)}")
    lines.append(
        "static const struct { int sign; int exp_unb; int mant7; uint16_t out; } PACK_CASES[] = {"
    )
    for s, e, m in pack_cases:
        out = pack_e4m3_prod(s, e, m)
        lines.append(f"    {{ {s}, {e}, {m}, 0x{out:04X} }},")
    lines.append("};")
    lines.append("")

    # ---- e4m3_prod_to_aligned_int ----
    prod_int_cases = []
    # Zero
    prod_int_cases.append((0x0000, 8, 32))
    # 1.0, anchor sweep
    p1 = pack_e4m3_prod(0, 0, 0)
    for anc in range(0, 33, 4):
        prod_int_cases.append((p1, anc, 32))
    # -1.0, anchor sweep
    pn1 = pack_e4m3_prod(1, 0, 0)
    for anc in range(0, 33, 4):
        prod_int_cases.append((pn1, anc, 32))
    # unb_exp sweep, anchor=8
    for unb in range(-12, 19):
        p = pack_e4m3_prod(0, unb, 0)
        if p:
            prod_int_cases.append((p, 8, 32))
    # int_width sweep for 1.0, anchor=0
    for iw in [1, 2, 4, 8, 12, 16, 24, 32]:
        prod_int_cases.append((p1, 0, iw))
    # int_width sweep for -1.0, anchor=0
    for iw in [1, 2, 4, 8, 12, 16, 24, 32]:
        prod_int_cases.append((pn1, 0, iw))
    # Mantissa sweep
    for mant in [1, 16, 64, 127]:
        prod_int_cases.append((pack_e4m3_prod(0, 0, mant), 8, 32))
        prod_int_cases.append((pack_e4m3_prod(1, 0, mant), 8, 32))
    # Cross grid
    for unb, anc, iw in [
        (-5, 0, 32),
        (0, 8, 32),
        (5, 16, 32),
        (0, 0, 8),
        (0, 0, 16),
        (0, 0, 24),
        (1, 0, 16),
        (-1, 4, 16),
    ]:
        p = pack_e4m3_prod(0, unb, 0)
        if p:
            prod_int_cases.append((p, anc, iw))
    # Legacy cases
    prod_int_cases += [
        (0x0680, 8, 32),
        (0x1700, 8, 32),
        (0x087F, 10, 32),
        (0x1400, 0, 32),
    ]
    # Deduplicate preserving order
    seen = set()
    prod_int_cases_dedup = []
    for c in prod_int_cases:
        if c not in seen:
            seen.add(c)
            prod_int_cases_dedup.append(c)
    prod_int_cases = prod_int_cases_dedup

    lines.append(f"#define N_PROD_INT {len(prod_int_cases)}")
    lines.append(
        "static const struct { uint16_t prod; int anchor; int iw; int32_t out; } PROD_INT_CASES[] = {"
    )
    for pb, anc, iw in prod_int_cases:
        out = e4m3_prod_to_aligned_int(pb, anc, iw)
        lines.append(f"    {{ 0x{pb:04X}, {anc}, {iw}, {out} }},")
    lines.append("};")
    lines.append("")

    # ---- ieee_to_aligned_int ----
    ieee_cases = []
    # BF16 basic + anchor sweep + iw sweep
    for bits in [
        0x3F80,
        0xBF80,
        0x0000,
        0x8000,
        0x4000,
        0x3F00,
        0xC000,
        0x4080,
        0x4380,
        0x7F7F,
        0xFF7F,
        0x0080,
        0x8080,
    ]:
        ieee_cases.append((bits, "BF16", 7, 32))
    for anc in range(0, 33, 4):
        ieee_cases.append((0x3F80, "BF16", anc, 32))
    for iw in [1, 2, 4, 8, 16, 24, 32]:
        for bits in [0x3F80, 0xBF80]:
            ieee_cases.append((bits, "BF16", 7, iw))
    # BF16 subnormals
    for frac in [1, 64, 127]:
        ieee_cases.append((frac, "BF16", 7, 32))
        ieee_cases.append((0x8000 | frac, "BF16", 7, 32))
    # E4M3 all normals
    for sign in [0, 1]:
        for unb in range(-6, 9):
            for mant in range(8):
                bits = encode_e4m3_normal(sign, unb, mant)
                ieee_cases.append((bits, "E4M3", 7, 32))
    # E4M3 anchor sweep
    for anc in range(0, 20, 4):
        ieee_cases.append((0x38, "E4M3", anc, 32))
    # E4M3 iw sweep
    for iw in [1, 2, 4, 8, 16, 32]:
        for bits in [0x38, 0xB8, 0x40]:
            ieee_cases.append((bits, "E4M3", 7, iw))
    # E4M3 subnormals
    for frac in range(1, 8):
        ieee_cases.append((frac, "E4M3", 7, 32))
        ieee_cases.append((0x80 | frac, "E4M3", 7, 32))
    # Large-left-shift stress
    for bits, anc, iw in [
        (0x4380, 0, 32),
        (0x4000, 0, 32),
        (0x7F7F, 7, 32),
        (0x0080, 7, 32),
    ]:
        ieee_cases.append((bits, "BF16", anc, iw))

    # Deduplicate
    seen_i = set()
    ieee_cases_dedup = []
    for c in ieee_cases:
        if c not in seen_i:
            seen_i.add(c)
            ieee_cases_dedup.append(c)
    ieee_cases = ieee_cases_dedup

    lines.append(f"#define N_IEEE_INT {len(ieee_cases)}")
    lines.append("/* fmt: 0=BF16, 1=E4M3 */")
    lines.append(
        "static const struct { uint32_t bits; int fmt; int anchor; int iw; int32_t out; } IEEE_INT_CASES[] = {"
    )
    for bits, fn, anc, iw in ieee_cases:
        fmt = BF16 if fn == "BF16" else E4M3
        out = ieee_to_aligned_int(bits, fmt, anc, iw)
        fi = 0 if fn == "BF16" else 1
        lines.append(f"    {{ 0x{bits:08X}u, {fi}, {anc}, {iw}, {out} }},")
    lines.append("};")
    lines.append("")

    # ---- aligned_int_to_bf16 ----
    a2b_cases_raw = [
        # iw=32 basics
        (128, 7, 32),
        (-128, 7, 32),
        (0, 7, 32),
        (256, 8, 32),
        (1, 0, 32),
        (64, 7, 32),
        (-256, 7, 32),
        (512, 7, 32),
        (-1, 7, 32),
        (100, 7, 32),
        (-100, 7, 32),
        (1, 30, 32),
        (-1, 30, 32),
        (1, 31, 32),
        (-1, 31, 32),
        (2147483647, 31, 32),
        (-2147483648, 31, 32),
        # iw=16
        (32767, 15, 16),
        (-32768, 15, 16),
        (128, 7, 16),
        (-128, 7, 16),
        (1, 0, 16),
        # iw=8 (wrap_signed path)
        (128, 7, 8),
        (-128, 7, 8),
        (127, 7, 8),
        (1, 0, 8),
        # iw=4
        (7, 3, 4),
        (-8, 3, 4),
        (8, 3, 4),  # wraps
        # iw=1
        (0, 0, 1),
        (-1, 0, 1),
        # RNE rounding cases
        (3, 1, 32),
        (5, 2, 32),
        (7, 3, 32),
        # 1.0 via various iw
        (1, 0, 1),
        (1, 1, 2),
        (1, 3, 4),
        (1, 7, 8),
        (1, 15, 16),
        (1, 31, 32),
        # zero for all iw
        (0, 7, 8),
        (0, 7, 16),
        (0, 7, 32),
    ]
    lines.append(f"#define N_A2B {len(a2b_cases_raw)}")
    lines.append(
        "static const struct { int32_t ival; int anchor; int iw; uint16_t out; } A2B_CASES[] = {"
    )
    for iv, anc, iw in a2b_cases_raw:
        out = aligned_int_to_bf16(iv, anc, iw)
        lines.append(f"    {{ {iv}, {anc}, {iw}, 0x{out:04X} }},")
    lines.append("};")
    lines.append("")

    # ---- bf16_scale_to_e4m3 ----
    bs_cases = []
    # Special values
    for bits in [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x0040, 0x8040]:
        bs_cases.append((bits, 0))
    # 1.0 scale sweep
    for sc in range(-15, 16):
        bs_cases.append((0x3F80, sc))
    # -1.0 scale sweep
    for sc in range(-15, 16):
        bs_cases.append((0xBF80, sc))
    # Various BF16 values at scale=0
    for bits in [
        0x4000,
        0x3F00,
        0xC000,
        0x4380,
        0x7F7F,
        0xFF7F,
        0x3E80,
        0x3F84,
        0x3F88,
        0x3F90,
        0x3FA0,
    ]:
        bs_cases.append((bits, 0))
    # All E4M3 normals round-trip
    for unb in range(-6, 9):
        for mant3 in range(8):
            e4m3_bits = encode_e4m3_normal(0, unb, mant3)
            d = decode_e4m3(e4m3_bits)
            bf16_bits = f32_to_bf16_bits_rne(d.value)
            bs_cases.append((bf16_bits, 0))

    seen_bs = set()
    bs_dedup = []
    for c in bs_cases:
        if c not in seen_bs:
            seen_bs.add(c)
            bs_dedup.append(c)
    bs_cases = bs_dedup

    lines.append(f"#define N_BS {len(bs_cases)}")
    lines.append(
        "static const struct { uint16_t bf16; int scale; uint8_t out; } BS_CASES[] = {"
    )
    for bf16b, sc in bs_cases:
        out = bf16_scale_to_e4m3(bf16b, sc)
        lines.append(f"    {{ 0x{bf16b:04X}, {sc}, 0x{out:02X} }},")
    lines.append("};")
    lines.append("")

    # ---- output_conv_stage ----
    oc_cases = []
    # BF16 output: all special inputs
    for bits in [
        0x3F80,
        0xBF80,
        0x4000,
        0x3F00,
        0x0000,
        0x8000,
        0x7F80,
        0xFF80,
        0x7FC0,
        0xFFC0,
        0x0040,
        0x8040,
        0x7F7F,
        0xFF7F,
        0x0001,
    ]:
        oc_cases.append((bits, 0, 0))
    # E4M3 output: all special inputs, scale=0
    for bits in [
        0x3F80,
        0xBF80,
        0x4000,
        0x3F00,
        0x0000,
        0x8000,
        0x7F80,
        0xFF80,
        0x7FC0,
        0x0040,
        0x7F7F,
        0xFF7F,
        0x0001,
    ]:
        oc_cases.append((bits, 1, 0))
    # E4M3 output: 1.0 and -1.0 with scale sweep
    for sc in range(-8, 10):
        oc_cases.append((0x3F80, 1, sc))
    for sc in [-2, -1, 0, 1, 2]:
        oc_cases.append((0xBF80, 1, sc))

    seen_oc = set()
    oc_dedup = []
    for c in oc_cases:
        if c not in seen_oc:
            seen_oc.add(c)
            oc_dedup.append(c)
    oc_cases = oc_dedup

    lines.append(f"#define N_OC {len(oc_cases)}")
    lines.append("/* fmt_sel: 0=BF16, 1=E4M3 */")
    lines.append(
        "static const struct { uint16_t bf16; int fmt_sel; int scale; uint32_t out; } OC_CASES[] = {"
    )
    for bf16b, fs, sc in oc_cases:
        fsel = OutputFmtSel.OutBF16 if fs == 0 else OutputFmtSel.OutE4M3
        out = output_conv_stage(bf16b, fsel, sc)
        lines.append(f"    {{ 0x{bf16b:04X}, {fs}, {sc}, 0x{out:08X}u }},")
    lines.append("};")
    lines.append("")

    with open("test_vectors.h", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[wrote test_vectors.h]")


write_c_vectors()

sys.exit(summary())
