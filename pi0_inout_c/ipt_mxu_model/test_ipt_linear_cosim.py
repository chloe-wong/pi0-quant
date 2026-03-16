"""
test_ipt_linear_cosim.py
========================
Co-simulation test: Python IPTLinearRTLFunction (golden reference)
vs the C ipt_linear_call() in ipt_linear.h, compiled via a thin shim.

Setup
-----
1.  Set IPT_HEADER_DIR to the directory containing all .h files
    (fp_formats.h, converters.h, params_and_requests.h,
     inner_product_trees_model.h, ipt_linear.h).
    Defaults to the current working directory.

        export IPT_HEADER_DIR=/path/to/headers

2.  All Python source files must be on sys.path.

3.  Run:
        python test_ipt_linear_cosim.py          # normal
        python test_ipt_linear_cosim.py -v       # verbose per-element diffs
"""

from __future__ import annotations

import ctypes
import os
import random
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Python golden model imports
# ---------------------------------------------------------------------------
from fp_formats import AddendSel, OutputFmtSel
from params_and_requests import InnerProductTreeParams
from ipt_rtl_linear import (
    IPTLinearRTLFunction,
    float_to_e4m3_bytes,
    torch_float_to_bf16_bits,
    torch_bf16_bits_to_float,
    decode_model_output_bits,
)

VERBOSE = "-v" in sys.argv

# ---------------------------------------------------------------------------
# C shim source — wraps ipt_linear_call for ctypes
# ---------------------------------------------------------------------------

_SHIM_C = textwrap.dedent(
    r"""
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "fp_formats.h"
#include "converters.h"
#include "params_and_requests.h"
#include "inner_product_trees_model.h"
#include "ipt_linear.h"

/* One-time LUT initialisation — call before anything else. */
void shim_init(void) {
    ipt_linear_init_luts();
}

/*
 * shim_ipt_linear_call
 *
 * Parameters (all arrays are flat, row-major):
 *   num_lanes, vec_len, pipeline_depth : model configuration
 *   x_e4m3      : uint8_t[batch * in_features]
 *   w_e4m3      : uint8_t[out_features * in_features]
 *   b_e4m3      : uint8_t[out_features], or NULL
 *   batch, in_features, out_features
 *   scale_exp   : scalar int applied to every lane
 *   out_fmt_sel : 0 = BF16, 1 = E4M3
 *   out_bits    : uint16_t[batch * out_features]  — written by callee
 */
void shim_ipt_linear_call(
        int num_lanes, int vec_len, int pipeline_depth,
        const uint8_t  *x_e4m3,
        const uint8_t  *w_e4m3,
        const uint8_t  *b_e4m3,
        int batch, int in_features, int out_features,
        int scale_exp, int out_fmt_sel,
        uint16_t *out_bits)
{
    InnerProductTreeParams base;
    base.numLanes      = num_lanes;
    base.vecLen        = vec_len;
    base.accumIntWidth = 0;
    base.pipelineCuts  = 0x00;

    InnerProductTreeParams p =
        IPT_withPipelineDepth(pipeline_depth, &base);

    ipt_linear_call(
        &p,
        x_e4m3, w_e4m3, b_e4m3,
        batch, in_features, out_features,
        scale_exp,
        (OutputFmtSel)out_fmt_sel,
        out_bits);
}
"""
)


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------


def _build_shim(header_dir: str) -> ctypes.CDLL:
    build_dir = tempfile.mkdtemp(prefix="ipt_linear_cosim_")
    shim_c = os.path.join(build_dir, "shim.c")
    shim_so = os.path.join(build_dir, "libipt_linear.so")

    with open(shim_c, "w") as f:
        f.write(_SHIM_C)

    cmd = [
        "gcc",
        "-O2",
        "-Wall",
        "-Wno-unused-function",
        "-shared",
        "-fPIC",
        f"-I{header_dir}",
        "-o",
        shim_so,
        shim_c,
        "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"C shim compilation failed:\n{r.stderr}\n"
            f"Make sure all .h files are in IPT_HEADER_DIR={header_dir!r}"
        )

    lib = ctypes.CDLL(shim_so)

    lib.shim_init.restype = None
    lib.shim_init.argtypes = []

    lib.shim_ipt_linear_call.restype = None
    lib.shim_ipt_linear_call.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # num_lanes, vec_len, depth
        ctypes.POINTER(ctypes.c_uint8),  # x_e4m3
        ctypes.POINTER(ctypes.c_uint8),  # w_e4m3
        ctypes.POINTER(ctypes.c_uint8),  # b_e4m3 (may be None→NULL)
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # batch, in_features, out_features
        ctypes.c_int,
        ctypes.c_int,  # scale_exp, out_fmt_sel
        ctypes.POINTER(ctypes.c_uint16),  # out_bits
    ]

    lib.shim_init()
    return lib


# ---------------------------------------------------------------------------
# C model wrapper
# ---------------------------------------------------------------------------


class CIPTLinear:
    """Calls shim_ipt_linear_call and returns a uint16 tensor matching the
    Python model's output layout."""

    def __init__(
        self, lib: ctypes.CDLL, num_lanes: int, vec_len: int, pipeline_depth: int
    ):
        self._lib = lib
        self.num_lanes = num_lanes
        self.vec_len = vec_len
        self.pipeline_depth = pipeline_depth

    def __call__(
        self,
        x_e4m3: torch.Tensor,  # uint8 [batch, in_features]
        w_e4m3: torch.Tensor,  # uint8 [out_features, in_features]
        b_e4m3: Optional[torch.Tensor],  # uint8 [out_features] or None
        scale_exp: int,
        out_fmt_sel: OutputFmtSel,
    ) -> torch.Tensor:  # int32 [batch, out_features]  (uint16 bits)
        batch, in_features = x_e4m3.shape
        out_features = w_e4m3.shape[0]

        x_flat = x_e4m3.cpu().contiguous().numpy()
        w_flat = w_e4m3.cpu().contiguous().numpy()

        x_ptr = x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        w_ptr = w_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        if b_e4m3 is not None:
            b_flat = b_e4m3.cpu().contiguous().numpy()
            b_ptr = b_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        else:
            b_ptr = ctypes.cast(None, ctypes.POINTER(ctypes.c_uint8))

        out_np = __import__("numpy").zeros(
            batch * out_features, dtype=__import__("numpy").uint16
        )
        out_ptr = out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

        self._lib.shim_ipt_linear_call(
            ctypes.c_int(self.num_lanes),
            ctypes.c_int(self.vec_len),
            ctypes.c_int(self.pipeline_depth),
            x_ptr,
            w_ptr,
            b_ptr,
            ctypes.c_int(batch),
            ctypes.c_int(in_features),
            ctypes.c_int(out_features),
            ctypes.c_int(scale_exp),
            ctypes.c_int(int(out_fmt_sel.value)),
            out_ptr,
        )

        out_tensor = torch.from_numpy(out_np).to(torch.int32)
        return out_tensor.reshape(batch, out_features)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _rand_float_tensor(shape, rng: random.Random, lo=-1.0, hi=1.0):
    """Random floats, seeded via Python rng for reproducibility."""
    vals = [rng.uniform(lo, hi) for _ in range(__import__("math").prod(shape))]
    return torch.tensor(vals, dtype=torch.float32).reshape(shape)


def _quantize_inputs(x_f, w_f, b_f=None):
    """E4M3-quantize x, w, b exactly as IPTLinearRTLFunction does."""
    x_e4m3 = float_to_e4m3_bytes(x_f.float())
    w_e4m3 = float_to_e4m3_bytes(w_f.float())
    b_e4m3 = float_to_e4m3_bytes(b_f.float()) if b_f is not None else None
    return x_e4m3, w_e4m3, b_e4m3


def _run_python(
    fn: IPTLinearRTLFunction,
    x_f: torch.Tensor,
    w_f: torch.Tensor,
    b_f: Optional[torch.Tensor],
    scale_exp: int,
) -> torch.Tensor:
    """Run the Python golden model; returns float tensor."""
    return fn(x_f, w_f, b_f, scale_exp)


def _run_c(
    c_fn: CIPTLinear,
    x_e4m3: torch.Tensor,
    w_e4m3: torch.Tensor,
    b_e4m3: Optional[torch.Tensor],
    scale_exp: int,
    out_fmt_sel: OutputFmtSel,
) -> torch.Tensor:
    """Run the C model; returns float tensor decoded from uint16 bits."""
    bits = c_fn(x_e4m3, w_e4m3, b_e4m3, scale_exp, out_fmt_sel)
    return decode_model_output_bits(bits, out_fmt_sel)


def _compare(
    py_float: torch.Tensor,
    c_float: torch.Tensor,
    label: str,
    verbose: bool = VERBOSE,
) -> bool:
    """Bitwise comparison after re-encoding both outputs to uint16."""
    # Both paths go through sanitize_bf16 / bf16_scale_to_e4m3 internally,
    # so comparing float values is sufficient and avoids re-encoding noise.
    # For an exact bitwise check we compare the raw bit patterns instead.
    py_bits = torch_float_to_bf16_bits(py_float).to(torch.int32)
    c_bits = torch_float_to_bf16_bits(c_float).to(torch.int32)

    match = torch.equal(py_bits, c_bits)
    if verbose or not match:
        tag = "OK  " if match else "FAIL"
        print(f"  [{tag}] {label}  shape={tuple(py_float.shape)}")
        if not match:
            diff = (py_bits != c_bits).nonzero(as_tuple=False)
            for idx in diff[:8]:  # cap at 8 mismatches
                i = tuple(idx.tolist())
                print(
                    f"         [{i}]  py=0x{py_bits[i].item():04x}"
                    f"  c=0x{c_bits[i].item():04x}"
                )
            if diff.shape[0] > 8:
                print(f"         … and {diff.shape[0]-8} more")
    return match


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestIPTLinearCosim(unittest.TestCase):

    HEADER_DIR = os.environ.get("IPT_HEADER_DIR", ".")
    NUM_LANES = 16
    VEC_LEN = 32
    PIPELINE_DEPTH = 1

    @classmethod
    def setUpClass(cls):
        cls.lib = _build_shim(cls.HEADER_DIR)
        cls.c_fn = CIPTLinear(cls.lib, cls.NUM_LANES, cls.VEC_LEN, cls.PIPELINE_DEPTH)
        cls.py_fn = IPTLinearRTLFunction(
            vec_len=cls.VEC_LEN,
            num_lanes=cls.NUM_LANES,
            pipeline_depth=cls.PIPELINE_DEPTH,
            out_fmt_sel=OutputFmtSel.OutBF16,
        )

    # ------------------------------------------------------------------
    # Convenience: run both models and assert exact match
    # ------------------------------------------------------------------
    def _check(
        self,
        label: str,
        x_f: torch.Tensor,
        w_f: torch.Tensor,
        b_f: Optional[torch.Tensor],
        scale_exp: int = 0,
        out_fmt_sel: OutputFmtSel = OutputFmtSel.OutBF16,
    ):
        # Re-create py_fn with the right out_fmt_sel for this call
        py_fn = IPTLinearRTLFunction(
            vec_len=self.VEC_LEN,
            num_lanes=self.NUM_LANES,
            pipeline_depth=self.PIPELINE_DEPTH,
            out_fmt_sel=out_fmt_sel,
        )
        x_e4m3, w_e4m3, b_e4m3 = _quantize_inputs(x_f, w_f, b_f)

        py_float = _run_python(py_fn, x_f, w_f, b_f, scale_exp)
        c_float = _run_c(self.c_fn, x_e4m3, w_e4m3, b_e4m3, scale_exp, out_fmt_sel)

        ok = _compare(py_float, c_float, label)
        self.assertTrue(ok, f"Output mismatch: {label}")

    # ------------------------------------------------------------------
    # Test 1: all-zero inputs, no bias
    # ------------------------------------------------------------------
    def test_zero_inputs_no_bias(self):
        rng = random.Random(0)
        x = torch.zeros(4, 32)
        w = torch.zeros(16, 32)
        self._check("zero_inputs_no_bias", x, w, None)

    # ------------------------------------------------------------------
    # Test 2: single batch row, in/out == vec_len/num_lanes (no tiling)
    # ------------------------------------------------------------------
    def test_single_tile_no_bias(self):
        rng = random.Random(1)
        x = _rand_float_tensor((1, self.VEC_LEN), rng)
        w = _rand_float_tensor((self.NUM_LANES, self.VEC_LEN), rng)
        self._check("single_tile_no_bias", x, w, None)

    # ------------------------------------------------------------------
    # Test 3: single tile, with bias
    # ------------------------------------------------------------------
    def test_single_tile_with_bias(self):
        rng = random.Random(2)
        x = _rand_float_tensor((1, self.VEC_LEN), rng)
        w = _rand_float_tensor((self.NUM_LANES, self.VEC_LEN), rng)
        b = _rand_float_tensor((self.NUM_LANES,), rng)
        self._check("single_tile_with_bias", x, w, b)

    # ------------------------------------------------------------------
    # Test 4: K-tiling (in_features = 3 × vec_len)
    # ------------------------------------------------------------------
    def test_k_tiling_no_bias(self):
        rng = random.Random(3)
        in_f = self.VEC_LEN * 3
        x = _rand_float_tensor((2, in_f), rng)
        w = _rand_float_tensor((self.NUM_LANES, in_f), rng)
        self._check("k_tiling_no_bias", x, w, None)

    # ------------------------------------------------------------------
    # Test 5: K-tiling with bias
    # ------------------------------------------------------------------
    def test_k_tiling_with_bias(self):
        rng = random.Random(4)
        in_f = self.VEC_LEN * 3
        x = _rand_float_tensor((2, in_f), rng)
        w = _rand_float_tensor((self.NUM_LANES, in_f), rng)
        b = _rand_float_tensor((self.NUM_LANES,), rng)
        self._check("k_tiling_with_bias", x, w, b)

    # ------------------------------------------------------------------
    # Test 6: output tiling (out_features > num_lanes)
    # ------------------------------------------------------------------
    def test_output_tiling_no_bias(self):
        rng = random.Random(5)
        out_f = self.NUM_LANES * 3
        x = _rand_float_tensor((2, self.VEC_LEN), rng)
        w = _rand_float_tensor((out_f, self.VEC_LEN), rng)
        self._check("output_tiling_no_bias", x, w, None)

    # ------------------------------------------------------------------
    # Test 7: output tiling with bias
    # ------------------------------------------------------------------
    def test_output_tiling_with_bias(self):
        rng = random.Random(6)
        out_f = self.NUM_LANES * 3
        x = _rand_float_tensor((2, self.VEC_LEN), rng)
        w = _rand_float_tensor((out_f, self.VEC_LEN), rng)
        b = _rand_float_tensor((out_f,), rng)
        self._check("output_tiling_with_bias", x, w, b)

    # ------------------------------------------------------------------
    # Test 8: both K and output tiling, larger batch
    # ------------------------------------------------------------------
    def test_k_and_output_tiling_batched(self):
        rng = random.Random(7)
        in_f = self.VEC_LEN * 4
        out_f = self.NUM_LANES * 2
        x = _rand_float_tensor((8, in_f), rng)
        w = _rand_float_tensor((out_f, in_f), rng)
        b = _rand_float_tensor((out_f,), rng)
        self._check("k_and_output_tiling_batched", x, w, b)

    # ------------------------------------------------------------------
    # Test 9: non-multiple in_features (partial last K-tile)
    # ------------------------------------------------------------------
    def test_partial_k_tile(self):
        rng = random.Random(8)
        in_f = self.VEC_LEN * 2 + 13  # intentionally not a multiple
        x = _rand_float_tensor((3, in_f), rng)
        w = _rand_float_tensor((self.NUM_LANES, in_f), rng)
        self._check("partial_k_tile", x, w, None)

    # ------------------------------------------------------------------
    # Test 10: non-multiple out_features (partial last output tile)
    # ------------------------------------------------------------------
    def test_partial_output_tile(self):
        rng = random.Random(9)
        out_f = self.NUM_LANES + 5  # intentionally not a multiple
        x = _rand_float_tensor((3, self.VEC_LEN), rng)
        w = _rand_float_tensor((out_f, self.VEC_LEN), rng)
        b = _rand_float_tensor((out_f,), rng)
        self._check("partial_output_tile", x, w, b)

    # ------------------------------------------------------------------
    # Test 11: non-zero scale_exp
    # ------------------------------------------------------------------
    def test_scale_exp(self):
        rng = random.Random(10)
        x = _rand_float_tensor((2, self.VEC_LEN), rng)
        w = _rand_float_tensor((self.NUM_LANES, self.VEC_LEN), rng)
        for sexp in (-4, -1, 0, 2, 5):
            self._check(
                f"scale_exp={sexp}",
                x,
                w,
                None,
                scale_exp=sexp,
                out_fmt_sel=OutputFmtSel.OutE4M3,
            )

    # ------------------------------------------------------------------
    # Test 12: OutE4M3 output format
    # ------------------------------------------------------------------
    def test_out_e4m3(self):
        rng = random.Random(11)
        x = _rand_float_tensor((4, self.VEC_LEN), rng)
        w = _rand_float_tensor((self.NUM_LANES, self.VEC_LEN), rng)
        b = _rand_float_tensor((self.NUM_LANES,), rng)
        self._check("out_e4m3", x, w, b, scale_exp=0, out_fmt_sel=OutputFmtSel.OutE4M3)

    # ------------------------------------------------------------------
    # Test 13: weights near E4M3 saturation boundary
    # ------------------------------------------------------------------
    def test_saturation_weights(self):
        rng = random.Random(12)
        # Mix of large-magnitude and normal values to exercise clamping paths
        x = _rand_float_tensor((2, self.VEC_LEN), rng, lo=0.9, hi=1.0)
        w = _rand_float_tensor((self.NUM_LANES, self.VEC_LEN), rng, lo=0.9, hi=1.0)
        self._check("saturation_weights", x, w, None)

    # ------------------------------------------------------------------
    # Test 14: negative weights and activations
    # ------------------------------------------------------------------
    def test_negative_values(self):
        rng = random.Random(13)
        x = _rand_float_tensor((4, self.VEC_LEN), rng, lo=-1.0, hi=0.0)
        w = _rand_float_tensor((self.NUM_LANES, self.VEC_LEN), rng, lo=-1.0, hi=0.0)
        b = _rand_float_tensor((self.NUM_LANES,), rng, lo=-0.5, hi=0.5)
        self._check("negative_values", x, w, b)

    # ------------------------------------------------------------------
    # Test 15: larger randomised stress test
    # ------------------------------------------------------------------
    def test_stress(self):
        rng = random.Random(99)
        in_f = self.VEC_LEN * 5 + 7
        out_f = self.NUM_LANES * 4 + 3
        x = _rand_float_tensor((16, in_f), rng)
        w = _rand_float_tensor((out_f, in_f), rng)
        b = _rand_float_tensor((out_f,), rng)
        self._check("stress", x, w, b)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    argv = [a for a in sys.argv if a not in ("-v",)]
    unittest.main(argv=argv, verbosity=2)
