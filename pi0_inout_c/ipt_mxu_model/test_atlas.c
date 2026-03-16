/* test_atlas.c
 *
 * Build:
 *   cc -O2 -Wall -Wno-unused-function -I. test_atlas.c -lm -o test_atlas
 *
 * Generate golden vectors first:
 *   python test_atlas.py
 *
 * Exits 0 on all-pass, 1 on any failure.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "fp_formats.h"
#include "converters.h"
#include "test_vectors.h"

static void print_constants(void)
{
    printf("_E4M3_PROD_BIAS      = %d  (want 13)\n", _E4M3_PROD_BIAS);
    printf("_E4M3_PROD_SIG_WIDTH = %d  (want  8)\n", _E4M3_PROD_SIG_WIDTH);
    printf("_E4M3_PROD_EXP_MAX   = %d  (want 31)\n", _E4M3_PROD_EXP_MAX);
    printf("_BF16_BIAS           = %d  (want 127)\n", _BF16_BIAS);
    printf("\n");
}

/* ---------------------------------------------------------------------------
 * Harness
 * ------------------------------------------------------------------------- */

static int g_pass = 0;
static int g_fail = 0;

static void section(const char *name)
{
    printf("\n=== %s ===\n", name);
}

#define CHECK_EQ(label, got, expected, fmt)                       \
    do                                                            \
    {                                                             \
        __typeof__(got) _g = (got);                               \
        __typeof__(expected) _e = (expected);                     \
        if (_g == _e)                                             \
        {                                                         \
            printf("  PASS  %s\n", (label));                      \
            g_pass++;                                             \
        }                                                         \
        else                                                      \
        {                                                         \
            printf("  FAIL  %s  got=" fmt "  expected=" fmt "\n", \
                   (label), _g, _e);                              \
            g_fail++;                                             \
        }                                                         \
    } while (0)

static void check_f32(const char *label, float got, float expected)
{
    if ((isnan(got) && isnan(expected)) || got == expected)
    {
        printf("  PASS  %s\n", label);
        g_pass++;
    }
    else
    {
        printf("  FAIL  %s  got=%a  expected=%a\n",
               label, (double)got, (double)expected);
        g_fail++;
    }
}

/* Batch pass counter — for exhaustive loops we don't print per-case */
static void batch_result(const char *label, int failures, int total)
{
    if (failures == 0)
    {
        printf("  PASS  %s (%d cases)\n", label, total);
        g_pass++;
    }
    else
    {
        printf("  FAIL  %s (%d / %d wrong)\n", label, failures, total);
        g_fail++;
    }
}

static void summary(void)
{
    printf("\n========================================\n");
    printf("  %d passed,  %d failed\n", g_pass, g_fail);
    printf("========================================\n");
}

static inline float _bits_to_f32(uint32_t x)
{
    float f;
    memcpy(&f, &x, 4);
    return f;
}
static inline uint32_t _f32_to_bits(float x)
{
    uint32_t u;
    memcpy(&u, &x, 4);
    return u;
}

/* ===========================================================================
 * SECTION 1 — Leaf utility functions
 * ========================================================================= */

static void test_sign_extend(void)
{
    section("sign_extend");
    /* 4-bit */
    CHECK_EQ("pos 4-bit 0", sign_extend(0x0, 4), 0, "%d");
    CHECK_EQ("pos 4-bit 1", sign_extend(0x1, 4), 1, "%d");
    CHECK_EQ("pos 4-bit 3", sign_extend(0x3, 4), 3, "%d");
    CHECK_EQ("pos 4-bit max", sign_extend(0x7, 4), 7, "%d");
    CHECK_EQ("neg 4-bit 8", sign_extend(0x8, 4), -8, "%d");
    CHECK_EQ("neg 4-bit 15", sign_extend(0xF, 4), -1, "%d");
    CHECK_EQ("neg 4-bit 9", sign_extend(0x9, 4), -7, "%d");
    /* 1-bit */
    CHECK_EQ("pos 1-bit 0", sign_extend(0, 1), 0, "%d");
    CHECK_EQ("neg 1-bit 1", sign_extend(1, 1), -1, "%d");
    /* 2-bit */
    CHECK_EQ("pos 2-bit 1", sign_extend(0x1, 2), 1, "%d");
    CHECK_EQ("neg 2-bit 2", sign_extend(0x2, 2), -2, "%d");
    CHECK_EQ("neg 2-bit 3", sign_extend(0x3, 2), -1, "%d");
    /* 8-bit */
    CHECK_EQ("pos 8-bit 0", sign_extend(0x00, 8), 0, "%d");
    CHECK_EQ("pos 8-bit 127", sign_extend(0x7F, 8), 127, "%d");
    CHECK_EQ("neg 8-bit 128", sign_extend(0x80, 8), -128, "%d");
    CHECK_EQ("neg 8-bit 255", sign_extend(0xFF, 8), -1, "%d");
    CHECK_EQ("neg 8-bit 129", sign_extend(0x81, 8), -127, "%d");
    /* 16-bit */
    CHECK_EQ("pos 16-bit max", sign_extend(0x7FFF, 16), 32767, "%d");
    CHECK_EQ("neg 16-bit min", sign_extend(0x8000, 16), -32768, "%d");
    CHECK_EQ("neg 16-bit -1", sign_extend(0xFFFF, 16), -1, "%d");
    /* 32-bit */
    CHECK_EQ("pos 32-bit max", sign_extend(0x7FFFFFFF, 32), 2147483647, "%d");
    CHECK_EQ("neg 32-bit min", sign_extend((int32_t)0x80000000, 32), -2147483648, "%d");
    CHECK_EQ("neg 32-bit -1", sign_extend(-1, 32), -1, "%d");
}

static void test_clamp_signed(void)
{
    section("clamp_signed");
    CHECK_EQ("4b in range", clamp_signed(5, 4), 5, "%d");
    CHECK_EQ("4b clamp hi", clamp_signed(8, 4), 7, "%d");
    CHECK_EQ("4b clamp lo", clamp_signed(-9, 4), -8, "%d");
    CHECK_EQ("4b exact hi", clamp_signed(7, 4), 7, "%d");
    CHECK_EQ("4b exact lo", clamp_signed(-8, 4), -8, "%d");
    CHECK_EQ("4b zero", clamp_signed(0, 4), 0, "%d");
    CHECK_EQ("4b -1", clamp_signed(-1, 4), -1, "%d");
    CHECK_EQ("4b large pos", clamp_signed(100, 4), 7, "%d");
    CHECK_EQ("4b large neg", clamp_signed(-100, 4), -8, "%d");
    CHECK_EQ("1b 0", clamp_signed(0, 1), 0, "%d");
    CHECK_EQ("1b clamp hi", clamp_signed(1, 1), 0, "%d");
    CHECK_EQ("1b clamp lo", clamp_signed(-2, 1), -1, "%d");
    CHECK_EQ("1b -1", clamp_signed(-1, 1), -1, "%d");
    CHECK_EQ("1b 100", clamp_signed(100, 1), 0, "%d");
    CHECK_EQ("8b hi", clamp_signed(200, 8), 127, "%d");
    CHECK_EQ("8b lo", clamp_signed(-200, 8), -128, "%d");
    CHECK_EQ("8b exact hi", clamp_signed(127, 8), 127, "%d");
    CHECK_EQ("8b exact lo", clamp_signed(-128, 8), -128, "%d");
    CHECK_EQ("16b hi", clamp_signed(40000, 16), 32767, "%d");
    CHECK_EQ("16b lo", clamp_signed(-40000, 16), -32768, "%d");
    CHECK_EQ("32b no clamp", clamp_signed(0x7FFFFFFF, 32), 0x7FFFFFFF, "%d");
    CHECK_EQ("32b no clamp neg", clamp_signed((int32_t)0x80000000, 32), (int32_t)0x80000000, "%d");
}

static void test_wrap_signed(void)
{
    section("wrap_signed");
    /* 4-bit */
    CHECK_EQ("4b no wrap", wrap_signed(3, 4), 3, "%d");
    CHECK_EQ("4b wrap hi", wrap_signed(8, 4), -8, "%d");
    CHECK_EQ("4b wrap neg", wrap_signed(-1, 4), -1, "%d");
    CHECK_EQ("4b exact max", wrap_signed(7, 4), 7, "%d");
    CHECK_EQ("4b exact min", wrap_signed(-8, 4), -8, "%d");
    CHECK_EQ("4b wrap max+1", wrap_signed(8, 4), -8, "%d");
    CHECK_EQ("4b wrap min-1", wrap_signed(-9, 4), 7, "%d");
    CHECK_EQ("4b wrap 16", wrap_signed(16, 4), 0, "%d");
    CHECK_EQ("4b wrap -16", wrap_signed(-16, 4), 0, "%d");
    CHECK_EQ("4b wrap 15", wrap_signed(15, 4), -1, "%d");
    /* 1-bit */
    CHECK_EQ("1b zero", wrap_signed(0, 1), 0, "%d");
    CHECK_EQ("1b neg", wrap_signed(-1, 1), -1, "%d");
    CHECK_EQ("1b wrap 1", wrap_signed(1, 1), -1, "%d");
    CHECK_EQ("1b wrap 2", wrap_signed(2, 1), 0, "%d");
    CHECK_EQ("1b wrap -2", wrap_signed(-2, 1), 0, "%d");
    /* 8-bit */
    CHECK_EQ("8b zero", wrap_signed(0, 8), 0, "%d");
    CHECK_EQ("8b wrap 128", wrap_signed(128, 8), -128, "%d");
    CHECK_EQ("8b wrap -129", wrap_signed(-129, 8), 127, "%d");
    CHECK_EQ("8b wrap 256", wrap_signed(256, 8), 0, "%d");
    CHECK_EQ("8b exact hi", wrap_signed(127, 8), 127, "%d");
    CHECK_EQ("8b exact lo", wrap_signed(-128, 8), -128, "%d");
    CHECK_EQ("8b wrap 255", wrap_signed(255, 8), -1, "%d");
    CHECK_EQ("8b wrap -255", wrap_signed(-255, 8), 1, "%d");
    /* 16-bit */
    CHECK_EQ("16b wrap 32768", wrap_signed(32768, 16), -32768, "%d");
    CHECK_EQ("16b wrap -32769", wrap_signed(-32769, 16), 32767, "%d");
    CHECK_EQ("16b exact hi", wrap_signed(32767, 16), 32767, "%d");
    CHECK_EQ("16b exact lo", wrap_signed(-32768, 16), -32768, "%d");
    /* 32-bit — identity for all valid int32 values */
    CHECK_EQ("32b identity pos", wrap_signed(2147483647, 32), 2147483647, "%d");
    CHECK_EQ("32b identity neg", wrap_signed((int32_t)(-2147483648LL), 32), (int32_t)(-2147483648LL), "%d");
    CHECK_EQ("32b identity zero", wrap_signed(0, 32), 0, "%d");
    CHECK_EQ("32b identity -1", wrap_signed(-1, 32), -1, "%d");
    CHECK_EQ("32b identity 1", wrap_signed(1, 32), 1, "%d");
}

static void test_rrs4(void)
{
    section("round_right_shift4_rne");
    /* Table-driven: full 256 cases */
    int failures = 0;
    for (int i = 0; i < N_RRS4; i++)
    {
        int got = round_right_shift4_rne(RRS4_CASES[i].in);
        if (got != RRS4_CASES[i].out)
        {
            failures++;
            if (failures <= 8)
                printf("  FAIL  rrs4(0x%02X)  got=%d  expected=%d\n",
                       RRS4_CASES[i].in, got, RRS4_CASES[i].out);
        }
    }
    batch_result("rrs4 all 256 inputs", failures, N_RRS4);
}

static void test_bf16_conv(void)
{
    section("f32_to_bf16_bits_rne");
    char label[64];
    for (int i = 0; i < N_BF16_CASES; i++)
    {
        float f = _bits_to_f32(BF16_CASES[i].f32_bits);
        uint16_t got = f32_to_bf16_bits_rne(f);
        uint16_t exp_val = BF16_CASES[i].bf16_bits;
        if (isnan(f))
        {
            int ok = ((got & 0x7FC0) == 0x7FC0);
            if (ok)
            {
                printf("  PASS  bf16 NaN\n");
                g_pass++;
            }
            else
            {
                printf("  FAIL  bf16 NaN  got=0x%04X\n", got);
                g_fail++;
            }
        }
        else
        {
            snprintf(label, sizeof label, "bf16(0x%08X)", BF16_CASES[i].f32_bits);
            CHECK_EQ(label, got, exp_val, "0x%04X");
        }
    }

    section("bf16_bits_to_f32");
    check_f32("1.0", bf16_bits_to_f32(0x3F80), 1.0f);
    check_f32("-2.0", bf16_bits_to_f32(0xC000), -2.0f);
    check_f32("0.0", bf16_bits_to_f32(0x0000), 0.0f);
    check_f32("0.5", bf16_bits_to_f32(0x3F00), 0.5f);
    check_f32("-0.5", bf16_bits_to_f32(0xBF00), -0.5f);
    check_f32("256.0", bf16_bits_to_f32(0x4380), 256.0f);
    check_f32("+inf", bf16_bits_to_f32(0x7F80), (float)INFINITY);
    check_f32("-inf", bf16_bits_to_f32(0xFF80), -(float)INFINITY);
    check_f32("0.25", bf16_bits_to_f32(0x3E80), 0.25f);
    /* NaN: check isnan */
    {
        float nan_val = bf16_bits_to_f32(0x7FC0);
        int ok = isnan(nan_val);
        if (ok)
        {
            printf("  PASS  bf16_bits_to_f32 NaN\n");
            g_pass++;
        }
        else
        {
            printf("  FAIL  bf16_bits_to_f32 NaN  got=%g\n", nan_val);
            g_fail++;
        }
    }
    /* Round-trip */
    {
        static const struct
        {
            float v;
        } rt[] = {
            {1.0f}, {-1.0f}, {0.5f}, {2.0f}, {0.25f}, {256.0f}, {-0.125f}};
        for (int i = 0; i < (int)(sizeof rt / sizeof rt[0]); i++)
        {
            uint16_t bits = f32_to_bf16_bits_rne(rt[i].v);
            float back = bf16_bits_to_f32(bits);
            check_f32("bf16 round-trip", back, rt[i].v);
        }
    }
}

static void test_sanitize_bf16(void)
{
    section("sanitize_bf16");
    /* Exhaustive: compare against full LUT */
    int failures = 0;
    for (int i = 0; i < 65536; i++)
    {
        uint16_t got = sanitize_bf16((uint16_t)i);
        uint16_t exp_val = SAN_LUT[i];
        if (got != exp_val)
        {
            failures++;
            if (failures <= 8)
                printf("  FAIL  sanitize(0x%04X)  got=0x%04X  expected=0x%04X\n",
                       i, got, exp_val);
        }
    }
    batch_result("sanitize_bf16 all 65536 inputs", failures, 65536);
}

/* ===========================================================================
 * SECTION 2 — E4M3 decode / encode
 * ========================================================================= */

static void test_decode_e4m3(void)
{
    section("decode_e4m3 exhaustive");
    int failures = 0;
    for (int i = 0; i < 256; i++)
    {
        DecodedFloat d = decode_e4m3((uint8_t)i);
        const __typeof__(E4M3_DECODE_CASES[0]) *ref = &E4M3_DECODE_CASES[i];

        int ok = 1;
        ok &= (d.sign == ref->sign);
        ok &= (d.exp_field == ref->exp_field);
        ok &= (d.frac == ref->frac);
        ok &= ((int)d.is_zero == ref->is_zero);
        ok &= ((int)d.is_sub == ref->is_sub);
        ok &= ((int)d.is_inf == ref->is_inf);
        ok &= ((int)d.is_nan == ref->is_nan);
        ok &= (!ref->has_unb_exp || d.unb_exp == ref->unb_exp);
        if (ref->is_nan)
            ok &= isnan(d.value);
        else
            ok &= (_f32_to_bits(d.value) == ref->value_bits);

        if (!ok)
        {
            failures++;
            if (failures <= 8)
                printf("  FAIL  decode_e4m3(0x%02X)\n", i);
        }
    }
    batch_result("decode_e4m3 all 256", failures, 256);
}

static void test_encode_e4m3_normal(void)
{
    section("encode_e4m3_normal");
    CHECK_EQ("1.0", encode_e4m3_normal(0, 0, 0), (uint8_t)0x38, "0x%02X");
    CHECK_EQ("-1.0", encode_e4m3_normal(1, 0, 0), (uint8_t)0xB8, "0x%02X");
    CHECK_EQ("1.5", encode_e4m3_normal(0, 0, 4), (uint8_t)0x3C, "0x%02X");
    CHECK_EQ("2.0", encode_e4m3_normal(0, 1, 0), (uint8_t)0x40, "0x%02X");
    CHECK_EQ("0.5", encode_e4m3_normal(0, -1, 0), (uint8_t)0x30, "0x%02X");
    CHECK_EQ("-2.0", encode_e4m3_normal(1, 1, 0), (uint8_t)0xC0, "0x%02X");
    CHECK_EQ("max pos", encode_e4m3_normal(0, 8, 6), (uint8_t)0x7E, "0x%02X");
    CHECK_EQ("max neg", encode_e4m3_normal(1, 8, 6), (uint8_t)0xFE, "0x%02X");
    CHECK_EQ("1.875", encode_e4m3_normal(0, 0, 7), (uint8_t)0x3F, "0x%02X");
}

/* ===========================================================================
 * SECTION 3 — multiply LUT
 * ========================================================================= */

static void test_mul_to_prod(void)
{
    section("e4m3_mul_to_prod exhaustive");
    int failures = 0;
    for (int i = 0; i < 65536; i++)
    {
        uint8_t a = (uint8_t)(i >> 8);
        uint8_t b = (uint8_t)(i & 0xFF);
        uint16_t got = e4m3_mul_to_prod(a, b);
        uint16_t exp_val = MUL_PROD_LUT[i];
        if (got != exp_val)
        {
            failures++;
            if (failures <= 8)
                printf("  FAIL  mul(0x%02X,0x%02X) got=0x%04X exp=0x%04X\n",
                       a, b, got, exp_val);
        }
    }
    batch_result("e4m3_mul_to_prod all 65536", failures, 65536);
}

/* ===========================================================================
 * SECTION 4 — pack_e4m3_prod
 * ========================================================================= */

static void test_pack_e4m3_prod(void)
{
    section("pack_e4m3_prod");
    char label[80];
    int failures = 0;
    for (int i = 0; i < N_PACK; i++)
    {
        uint16_t got = pack_e4m3_prod(
            PACK_CASES[i].sign, PACK_CASES[i].exp_unb, PACK_CASES[i].mant7);
        if (got != PACK_CASES[i].out)
        {
            failures++;
            if (failures <= 8)
            {
                snprintf(label, sizeof label, "pack sign=%d exp=%d mant=%d",
                         PACK_CASES[i].sign, PACK_CASES[i].exp_unb, PACK_CASES[i].mant7);
                printf("  FAIL  %s  got=0x%04X  expected=0x%04X\n",
                       label, got, PACK_CASES[i].out);
            }
        }
    }
    batch_result("pack_e4m3_prod", failures, N_PACK);
}

/* ===========================================================================
 * SECTION 5 — e4m3_prod_to_aligned_int
 * ========================================================================= */

static void test_prod_to_aligned_int(void)
{
    section("e4m3_prod_to_aligned_int");
    char label[80];
    int failures = 0;
    for (int i = 0; i < N_PROD_INT; i++)
    {
        int32_t got = e4m3_prod_to_aligned_int(
            PROD_INT_CASES[i].prod, PROD_INT_CASES[i].anchor, PROD_INT_CASES[i].iw);
        if (got != PROD_INT_CASES[i].out)
        {
            failures++;
            if (failures <= 8)
            {
                snprintf(label, sizeof label, "prod=0x%04X anc=%d iw=%d",
                         PROD_INT_CASES[i].prod, PROD_INT_CASES[i].anchor, PROD_INT_CASES[i].iw);
                printf("  FAIL  %s  got=%d  expected=%d\n",
                       label, got, PROD_INT_CASES[i].out);
            }
        }
    }
    batch_result("e4m3_prod_to_aligned_int", failures, N_PROD_INT);
}

/* ===========================================================================
 * SECTION 6 — ieee_to_aligned_int
 * ========================================================================= */

static void test_ieee_to_aligned_int(void)
{
    section("ieee_to_aligned_int");
    char label[80];
    int failures = 0;
    for (int i = 0; i < N_IEEE_INT; i++)
    {
        const AtlasFPType *fmt = (IEEE_INT_CASES[i].fmt == 0) ? &BF16 : &E4M3;
        int32_t got = ieee_to_aligned_int(
            IEEE_INT_CASES[i].bits, fmt,
            IEEE_INT_CASES[i].anchor, IEEE_INT_CASES[i].iw);
        if (got != IEEE_INT_CASES[i].out)
        {
            failures++;
            if (failures <= 8)
            {
                snprintf(label, sizeof label, "bits=0x%08X fmt=%s anc=%d iw=%d",
                         IEEE_INT_CASES[i].bits, fmt->name,
                         IEEE_INT_CASES[i].anchor, IEEE_INT_CASES[i].iw);
                printf("  FAIL  %s  got=%d  expected=%d\n",
                       label, got, IEEE_INT_CASES[i].out);
            }
        }
    }
    batch_result("ieee_to_aligned_int", failures, N_IEEE_INT);
}

/* ===========================================================================
 * SECTION 7 — aligned_int_to_bf16
 * ========================================================================= */

static void test_aligned_int_to_bf16(void)
{
    section("aligned_int_to_bf16");
    char label[80];
    int failures = 0;
    for (int i = 0; i < N_A2B; i++)
    {
        uint16_t got = aligned_int_to_bf16(
            A2B_CASES[i].ival, A2B_CASES[i].anchor, A2B_CASES[i].iw);
        if (got != A2B_CASES[i].out)
        {
            failures++;
            if (failures <= 8)
            {
                snprintf(label, sizeof label, "ival=%d anc=%d iw=%d",
                         A2B_CASES[i].ival, A2B_CASES[i].anchor, A2B_CASES[i].iw);
                printf("  FAIL  %s  got=0x%04X  expected=0x%04X\n",
                       label, got, A2B_CASES[i].out);
            }
        }
    }
    batch_result("aligned_int_to_bf16", failures, N_A2B);
}

/* ===========================================================================
 * SECTION 8 — bf16_scale_to_e4m3
 * ========================================================================= */

static void test_bf16_scale_to_e4m3(void)
{
    section("bf16_scale_to_e4m3");
    char label[80];
    int failures = 0;
    for (int i = 0; i < N_BS; i++)
    {
        uint8_t got = bf16_scale_to_e4m3(BS_CASES[i].bf16, BS_CASES[i].scale);
        if (got != BS_CASES[i].out)
        {
            failures++;
            if (failures <= 8)
            {
                snprintf(label, sizeof label, "bf16=0x%04X scale=%d",
                         BS_CASES[i].bf16, BS_CASES[i].scale);
                printf("  FAIL  %s  got=0x%02X  expected=0x%02X\n",
                       label, got, BS_CASES[i].out);
            }
        }
    }
    batch_result("bf16_scale_to_e4m3", failures, N_BS);
}

/* ===========================================================================
 * SECTION 9 — output_conv_stage
 * ========================================================================= */

static void test_output_conv_stage(void)
{
    section("output_conv_stage");
    char label[80];
    int failures = 0;
    for (int i = 0; i < N_OC; i++)
    {
        OutputFmtSel fs = (OC_CASES[i].fmt_sel == 0)
                              ? OutputFmtSel_OutBF16
                              : OutputFmtSel_OutE4M3;
        uint32_t got = output_conv_stage(OC_CASES[i].bf16, fs, OC_CASES[i].scale);
        if (got != OC_CASES[i].out)
        {
            failures++;
            if (failures <= 8)
            {
                snprintf(label, sizeof label, "bf16=0x%04X fmt=%d scale=%d",
                         OC_CASES[i].bf16, OC_CASES[i].fmt_sel, OC_CASES[i].scale);
                printf("  FAIL  %s  got=0x%08X  expected=0x%08X\n",
                       label, got, OC_CASES[i].out);
            }
        }
    }
    batch_result("output_conv_stage", failures, N_OC);
}

/* ===========================================================================
 * main
 * ========================================================================= */

int main(void)
{
    print_constants();

    atlas_fp_init_lut();
    atlas_acc_init_lut();

    test_sign_extend();
    test_clamp_signed();
    test_wrap_signed();
    test_rrs4();
    test_bf16_conv();
    test_sanitize_bf16();

    test_decode_e4m3();
    test_encode_e4m3_normal();

    test_mul_to_prod();

    test_pack_e4m3_prod();
    test_prod_to_aligned_int();
    test_ieee_to_aligned_int();
    test_aligned_int_to_bf16();
    test_bf16_scale_to_e4m3();
    test_output_conv_stage();

    summary();
    return (g_fail == 0) ? 0 : 1;
}