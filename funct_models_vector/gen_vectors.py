"""gen_vectors.py — CLI entry point that replaces `scripts/gen_vpu_vectors.py`.

This module generates `vpu_vectors.txt` blocks via
`VectorEngineModel(p).execute(...)` instead of an inline `compute_expected`
table. It preserves the same CLI surface as the old script
(`--out / --num / --seed / --num-lanes / --ops`) so existing Mill / sbt
hooks and developer muscle memory keep working without touching Scala.

Two behavioral deltas vs. the old script that are intentional:

1. **numpy.random is seeded.** The old script seeded only Python's
   stdlib `random`; sin/cos/tanh used `numpy.random.uniform(0, 2*pi)` which
   drew from numpy's un-seeded global state. Two runs of the old script
   with the same `--seed` produced different sin/cos/tanh cases — the
   committed `vpu_vectors.txt` was a snapshot of one such run. This
   script seeds both RNGs, making the output fully reproducible.

2. **csum goes through the FP32 widened accumulator.** The old script
   did `sum([bf16_hex_to_float(h) for h in vec])` which accumulates in
   Python float (FP64). `VectorEngineModel._exec_col_reduce` routes
   csum through `ColAddVec.compute_now` which mirrors the Chisel
   `AddRecFN(8, 24)` widened-recFN adder. A handful of rounding-edge
   cases diverge; that is the §4.3 upgrade, not a regression.

The output of this script is therefore NOT byte-identical to the
currently-committed `src/test/resources/vpu_vectors.txt`. Regeneration of
the committed file is part of Phase 1f landing (see `test_gen_vectors_cli.py`).
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path
from typing import TextIO

import numpy as np

from .vector_engine_model import VectorEngineModel
from .vector_params import VectorParams
from .vpu_vector_file import VPUTestCase, write_case


# --------------------------------------------------------------------
#  BF16 helpers — byte-identical to gen_vpu_vectors.py helpers
# --------------------------------------------------------------------

def float_to_bf16_hex(f: float) -> str:
    """Python float → BF16 hex string, RNE. Matches
    `scripts/gen_vpu_vectors.py:float_to_bf16_hex` byte-for-byte."""
    if math.isnan(f):
        return "7FC0"
    if f == float("inf"):
        return "7F80"
    if f == float("-inf"):
        return "FF80"
    fp32_int = struct.unpack(">I", struct.pack(">f", f))[0]
    lower_16 = fp32_int & 0xFFFF
    bit_16 = (fp32_int >> 16) & 1
    if lower_16 > 0x8000 or (lower_16 == 0x8000 and bit_16 == 1):
        fp32_int = (fp32_int + 0x8000) & 0xFFFFFFFF
    return f"{(fp32_int >> 16) & 0xFFFF:04X}"


def bf16_hex_to_int(h: str) -> int:
    return int(h, 16) & 0xFFFF


# --------------------------------------------------------------------
#  Random input generation — matches old script call pattern exactly
# --------------------------------------------------------------------

def generate_random_float(op: str) -> float:
    """Mirror of `scripts/gen_vpu_vectors.py:generate_random_float`.

    Important: the call sequence and library choice must match the old
    script exactly so the non-transcendental cases remain comparable
    when debugging a regen. sin/cos/tanh use numpy; everything else
    uses Python random."""
    if op in ("sqrt", "log"):
        return random.uniform(0.01, 100.0)
    if op in ("exp", "exp2", "square", "cube"):
        return random.uniform(-10.0, 10.0)
    if op in ("sin", "cos", "tanh"):
        return float(np.random.uniform(0, 2 * math.pi))
    if op == "rcp":
        val = random.uniform(-100.0, 100.0)
        return val if abs(val) > 0.01 else 1.0
    if op == "fp8pack":
        return random.uniform(-1000.0, 1000.0)
    # add, sub, mul, reductions, compares, etc.
    return random.uniform(-50.0, 50.0)


# --------------------------------------------------------------------
#  Per-case builder
# --------------------------------------------------------------------

# Non-binary ops — descriptions and vecB suppression match the old script.
_UNARY_OPS = frozenset({
    "rcp", "sqrt", "sin", "cos", "tanh", "log", "exp", "exp2",
    "square", "cube", "rmax", "rsum", "mov", "fp8pack", "fp8unpack",
    "relu", "rmin", "cmax", "cmin",
    "vliOne", "vliCol", "vliRow", "vliAll",
})

# Col reductions use 32-lane vecA built from two alternating raw streams.
_COL_OPS = frozenset({"csum", "cmax", "cmin"})

# VLI ops: vecA is a single 16-bit pattern, no random inputs needed.
_VLI_OPS = frozenset({"vliOne", "vliCol", "vliRow", "vliAll"})


def _build_inputs(op: str, num_lanes: int) -> tuple[list[str], list[str]]:
    """Return (vecA_hex, vecB_hex) drawn from the RNG exactly as the old
    script does. Each lane consumes four `generate_random_float` calls
    (raw_a, raw_b, raw_a2, raw_b2); the second pair is only retained for
    col ops."""
    vec_a_hex: list[str] = []
    vec_b_hex: list[str] = []
    for _ in range(num_lanes):
        raw_a = generate_random_float(op)
        raw_b = generate_random_float(op)
        hex_a = float_to_bf16_hex(raw_a)
        hex_b = float_to_bf16_hex(raw_b)

        raw_a2 = generate_random_float(op)
        raw_b2 = generate_random_float(op)
        hex_a2 = float_to_bf16_hex(raw_a2)
        hex_b2 = float_to_bf16_hex(raw_b2)

        vec_a_hex.append(hex_a)
        vec_b_hex.append(hex_b)
        if op in _COL_OPS:
            vec_a_hex.append(hex_a2)
            vec_b_hex.append(hex_b2)

    return vec_a_hex, vec_b_hex


def _build_case(
    case_id: int,
    op: str,
    num_lanes: int,
    model: VectorEngineModel,
) -> VPUTestCase:
    """Generate one case: inputs, golden output from the model, pretty
    description. Matches the shape of `gen_vpu_vectors.py:generate_test_case`
    closely enough that diffs against the committed file concentrate on
    sin/cos/tanh (numpy seeding delta) and csum (FP32 upgrade)."""
    vec_a_hex, vec_b_hex = _build_inputs(op, num_lanes)

    if op in _VLI_OPS:
        # VLI: vecA becomes a single 16-bit slot; exp shape depends on
        # variant and matches the historical script's quirky layout
        # (which is what the Scala drivers expect):
        #   vliOne → 1 slot  (lane 0 only on row 0)
        #   vliRow → 16 slots (full row 0 broadcast)
        #   vliCol → 32 slots (one column, broadcast across 32 row slots)
        #   vliAll → 32 slots (all lanes × 2 rows broadcast)
        imm = bf16_hex_to_int(vec_a_hex[0])
        vec_a_hex = [f"{imm:04X}"]
        vec_b_hex = ["0000"] * num_lanes
        result = model.execute(op, imm=imm)
        # `result[0]` == imm for all four VLI ops (see lane_boxes/vector_load_imm.py).
        imm_hex = f"{result[0] & 0xFFFF:04X}"
        if op == "vliOne":
            exp_hex = [imm_hex]
        elif op == "vliRow":
            exp_hex = [imm_hex] * num_lanes
        else:  # vliCol, vliAll
            exp_hex = [imm_hex] * (2 * num_lanes)
    elif op == "mov":
        # Mov is identity: exp = vecA. Route through the model anyway so
        # the dispatcher path gets exercised.
        a_bits = [bf16_hex_to_int(h) for h in vec_a_hex]
        result = model.execute("mov", a_vec=a_bits)
        exp_hex = [f"{v & 0xFFFF:04X}" for v in result]
    elif op in _COL_OPS:
        a_bits = [bf16_hex_to_int(h) for h in vec_a_hex]
        result = model.execute(op, a_vec=a_bits)
        exp_hex = [f"{v & 0xFFFF:04X}" for v in result]
    elif op in ("rsum", "rmax", "rmin"):
        a_bits = [bf16_hex_to_int(h) for h in vec_a_hex]
        result = model.execute(op, a_vec=a_bits)
        exp_hex = [f"{v & 0xFFFF:04X}" for v in result]
    else:
        # Unary pointwise (rcp/sqrt/sin/cos/tanh/log/exp/exp2/square/cube/relu)
        # or binary pointwise (add/sub/mul/pairmax/pairmin).
        a_bits = [bf16_hex_to_int(h) for h in vec_a_hex]
        b_bits = [bf16_hex_to_int(h) for h in vec_b_hex]
        result = model.execute(op, a_vec=a_bits, b_vec=b_bits)
        exp_hex = [f"{v & 0xFFFF:04X}" for v in result]

    # Old script zeros vecB for any op in its unary_ops set — mirror that
    # so the written blocks look the same.
    if op in _UNARY_OPS:
        vec_b_hex = ["0000"] * num_lanes

    desc = f"{op} unary random vectors" if op in _UNARY_OPS else f"{op} random vectors"

    return VPUTestCase(
        case_id=case_id,
        desc=desc,
        vpu_op=op,
        num_lanes=num_lanes,
        vec_a=vec_a_hex,
        vec_b=vec_b_hex,
        exp=exp_hex,
    )


# --------------------------------------------------------------------
#  Static sanity cases — hardcoded, no RNG
# --------------------------------------------------------------------

def _static_cases(num_lanes: int, ops: list[str]) -> list[VPUTestCase]:
    """Two hardcoded add cases that the old script always emits when
    'add' is in the op list. Byte-identical to
    `gen_vpu_vectors.py:main`'s `static_cases`."""
    cases = [
        VPUTestCase(
            case_id=0,
            desc="1.0 + 0.0 = 1.0 (all lanes)",
            vpu_op="add",
            num_lanes=num_lanes,
            vec_a=["3F80"] * num_lanes,
            vec_b=["0000"] * num_lanes,
            exp=["3F80"] * num_lanes,
        ),
        VPUTestCase(
            case_id=1,
            desc="1.0 + 2.0 = 3.0 (all lanes)",
            vpu_op="add",
            num_lanes=num_lanes,
            vec_a=["3F80"] * num_lanes,
            vec_b=["4000"] * num_lanes,
            exp=["4040"] * num_lanes,
        ),
    ]
    return [c for c in cases if c.vpu_op in ops]


# --------------------------------------------------------------------
#  CLI entry point
# --------------------------------------------------------------------

# Enum order matches `VPUOp` in `VectorIO.scala:19-25`, minus the
# permanently-excluded fp8pack / fp8unpack (see §4.6: those ops are
# 2→1 / 1→2 phased and don't fit the single-pulse vector file format).
ALL_OPS: list[str] = [
    "add", "sub", "mul", "rcp", "sqrt",
    "sin", "cos",
    "tanh",
    "log", "exp", "exp2",
    "square", "cube", "rmax", "rsum", "mov",
    "relu", "rmin",
    "cmax", "cmin",
    "csum",
    "pairmax", "pairmin",
    "vliOne", "vliCol", "vliRow", "vliAll",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m funct_models_vector.gen_vectors",
        description="Generate VPU test vectors via the functional VectorEngineModel",
    )
    parser.add_argument("--out", default="vpu_vectors.txt", help="Output file path")
    parser.add_argument("--num", type=int, default=50, help="Number of random cases")
    parser.add_argument("--num-lanes", type=int, default=16, help="Lanes per instruction")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for both random and numpy")
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=ALL_OPS,
        default=ALL_OPS,
        help="Operations to include (default: all non-FP8 ops)",
    )
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    p = VectorParams()
    model = VectorEngineModel(p)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    static = _static_cases(args.num_lanes, args.ops)
    with out_path.open("w") as f:
        for case in static:
            write_case(f, case)

        start_id = len(static)
        for i in range(args.num):
            op = args.ops[i % len(args.ops)]
            case = _build_case(start_id + i, op, args.num_lanes, model)
            write_case(f, case)

    total = len(static) + args.num
    print(f"Successfully wrote {total} test vectors to {args.out}")
    print(f"Included Operations: {', '.join(args.ops)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
