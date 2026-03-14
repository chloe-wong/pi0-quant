"""
ipt_rtl_linear_c.py
-------------------
C-accelerated drop-in replacement for IPTLinearRTLFunction.

On first import the module compiles ipt_linear.h (and its dependencies)
into a shared library via gcc and caches the ctypes handle as a
module-level singleton.  Subsequent imports and all QuantLinearC instances
share the same .so.

Required environment variable
------------------------------
    IPT_HEADER_DIR   Directory containing all C headers:
                         fp_formats.h
                         converters.h
                         params_and_requests.h
                         inner_product_trees_model.h
                         ipt_linear.h
                     Defaults to the directory of this file
                     (i.e. ipt_mxu_model/).

The compiled .so is written to a per-process temp directory and cleaned up
on interpreter exit.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import subprocess
import tempfile
import textwrap
from typing import Optional

import torch

from .fp_formats import OutputFmtSel

# ---------------------------------------------------------------------------
# C shim source
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

void shim_init(void) {
    ipt_linear_init_luts();
}

/*
 * shim_ipt_linear_call
 *
 * All arrays are flat row-major C arrays:
 *   x_e4m3   uint8_t  [batch * in_features]
 *   w_e4m3   uint8_t  [out_features * in_features]
 *   b_e4m3   uint8_t  [out_features]   — NULL if no bias
 *   out_bits uint16_t [batch * out_features]  — written by callee
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

    InnerProductTreeParams p = IPT_withPipelineDepth(pipeline_depth, &base);

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
# Module-level singleton state
# ---------------------------------------------------------------------------

_lib: Optional[ctypes.CDLL] = None
_build_dir: Optional[str] = None


def _cleanup() -> None:
    global _lib, _build_dir
    _lib = None
    if _build_dir and os.path.isdir(_build_dir):
        shutil.rmtree(_build_dir, ignore_errors=True)
    _build_dir = None


atexit.register(_cleanup)


def _get_lib() -> ctypes.CDLL:
    """Return the cached ctypes handle, compiling on first call."""
    global _lib, _build_dir

    if _lib is not None:
        return _lib

    # Default: the ipt_mxu_model/ directory where the .h files live
    # alongside this .py file.
    header_dir = os.environ.get(
        "IPT_HEADER_DIR",
        os.path.dirname(os.path.abspath(__file__)),
    )

    _build_dir = tempfile.mkdtemp(prefix="ipt_linear_c_")
    shim_c = os.path.join(_build_dir, "ipt_linear_shim.c")
    shim_so = os.path.join(_build_dir, "libipt_linear.so")

    with open(shim_c, "w") as fh:
        fh.write(_SHIM_C)

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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _cleanup()
        raise RuntimeError(
            "ipt_rtl_linear_c: failed to compile C shim.\n"
            f"  IPT_HEADER_DIR = {header_dir!r}\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  stderr:\n{result.stderr}"
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
        ctypes.POINTER(ctypes.c_uint8),  # b_e4m3 (NULL = no bias)
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # batch, in_features, out_features
        ctypes.c_int,
        ctypes.c_int,  # scale_exp, out_fmt_sel
        ctypes.POINTER(ctypes.c_uint16),  # out_bits (output)
    ]

    lib.shim_init()
    _lib = lib
    return _lib


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class CIPTLinearRTLFunction:
    """
    C-accelerated drop-in replacement for IPTLinearRTLFunction.

    The public interface is identical:

        fn = CIPTLinearRTLFunction(vec_len=32, num_lanes=16,
                                   pipeline_depth=1,
                                   out_fmt_sel=OutputFmtSel.OutBF16)
        y = fn(x_q, w_q, b_q, scale_exp=0)   # float32 tensor

    The first instantiation triggers a one-time gcc compilation.
    All subsequent instances reuse the cached .so.
    """

    def __init__(
        self,
        vec_len: int = 32,
        num_lanes: int = 16,
        pipeline_depth: int = 1,
        out_fmt_sel: OutputFmtSel = OutputFmtSel.OutBF16,
    ) -> None:
        self.vec_len = vec_len
        self.num_lanes = num_lanes
        self.pipeline_depth = pipeline_depth
        self.out_fmt_sel = out_fmt_sel

        # Trigger compilation now so errors surface at construction time.
        _get_lib()

    def _call_c(
        self,
        x_e4m3: torch.Tensor,
        w_e4m3: torch.Tensor,
        b_e4m3: Optional[torch.Tensor],
        scale_exp: int,
    ) -> torch.Tensor:
        import numpy as np

        lib = _get_lib()
        batch = x_e4m3.shape[0]
        in_features = x_e4m3.shape[1]
        out_features = w_e4m3.shape[0]

        x_np = x_e4m3.cpu().contiguous().numpy()
        w_np = w_e4m3.cpu().contiguous().numpy()

        x_ptr = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        w_ptr = w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        if b_e4m3 is not None:
            b_np = b_e4m3.cpu().contiguous().numpy()
            b_ptr = b_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        else:
            b_ptr = ctypes.cast(None, ctypes.POINTER(ctypes.c_uint8))

        out_np = np.zeros(batch * out_features, dtype=np.uint16)
        out_ptr = out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

        lib.shim_ipt_linear_call(
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
            ctypes.c_int(int(self.out_fmt_sel.value)),
            out_ptr,
        )

        bits = torch.from_numpy(out_np).to(torch.int32).reshape(batch, out_features)

        from .ipt_rtl_linear import decode_model_output_bits

        return decode_model_output_bits(bits, self.out_fmt_sel)

    def __call__(
        self,
        x_q: torch.Tensor,
        w_q: torch.Tensor,
        b_q: Optional[torch.Tensor] = None,
        scale_exp: int = 0,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_q       : float32 [..., in_features]   (already quantized to E4M3)
        w_q       : float32 [out_features, in_features]
        b_q       : float32 [out_features] or None
        scale_exp : integer exponent applied to every output lane

        Returns
        -------
        float32 tensor with the same leading dimensions as x_q and a final
        dimension of out_features.
        """
        original_shape = x_q.shape[:-1]
        in_features = x_q.shape[-1]

        x2 = x_q.reshape(-1, in_features).float()
        w2 = w_q.float()
        b2 = b_q.float() if b_q is not None else None

        from .ipt_rtl_linear import float_to_e4m3_bytes

        x_e4m3 = float_to_e4m3_bytes(x2)
        w_e4m3 = float_to_e4m3_bytes(w2)
        b_e4m3 = float_to_e4m3_bytes(b2) if b2 is not None else None

        y = self._call_c(x_e4m3, w_e4m3, b_e4m3, scale_exp)

        return y.reshape(*original_shape, w_q.shape[0])
