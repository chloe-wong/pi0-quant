"""
bench_e4m3.py
=============
Compares four implementations of float_to_e4m3_bytes:

  scalar        -- original Python loop (the baseline to beat)
  numba-serial  -- @njit without parallel=True (JIT compilation alone)
  C             -- single-threaded C loop via ctypes (gcc -O3 -march=native)
  numba-par     -- @njit(parallel=True) using all CPU cores

Also benchmarks end-to-end forward pass: IPTLinearRTLFunction (Python) vs
CIPTLinearRTLFunction (C), with cold/warm weight cache and steady-state
inference (warm weights, varying activations).

Run:
    python3 -m pi0_inout_c.ipt_mxu_model.bench_e4m3
    python3 -m pi0_inout_c.ipt_mxu_model.bench_e4m3 --runs 50
    python3 -m pi0_inout_c.ipt_mxu_model.bench_e4m3 --no-forward

Numba warmup: JIT compilation happens on the first call and is absorbed by
the warmup iterations. cache=True means subsequent process starts load from
__pycache__ with no recompilation cost.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import torch

from pi0_inout_c.ipt_mxu_model.ipt_rtl_linear import (
    float_to_e4m3_bytes as float_to_e4m3_bytes_scalar,
    IPTLinearRTLFunction,
)
from pi0_inout_c.ipt_mxu_model.ipt_rtl_linear_c import (
    NUMBA_AVAILABLE,
    float_to_e4m3_bytes_c,
    float_to_e4m3_bytes_numba,
    float_to_e4m3_bytes_numba_serial,
    CIPTLinearRTLFunction,
    _tensor_to_f32_numpy,
)
from pi0_inout_c.ipt_mxu_model.fp_formats import OutputFmtSel


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _ms(seconds: float) -> str:
    if seconds >= 0.1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds * 1000:.3f} ms"


def _time(fn, n_warmup: int, n_runs: int) -> float:
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    return (time.perf_counter() - t0) / n_runs


def _speedup(ref: float, t: float) -> str:
    return "  n/a" if t <= 0 else f"{ref / t:>6.1f}x"


# ---------------------------------------------------------------------------
# Section 1: float_to_e4m3_bytes — four-way comparison
# ---------------------------------------------------------------------------


def bench_quantization(n_warmup: int, n_runs: int) -> None:
    print(
        "\n── float_to_e4m3_bytes ──────────────────────────────────────────────────────────────"
    )
    if NUMBA_AVAILABLE:
        print("  Triggering Numba JIT compilation (absorbed into warmup)...")
        _seed = np.zeros((1, 32), dtype=np.float32)
        float_to_e4m3_bytes_numba_serial(_seed)
        float_to_e4m3_bytes_numba(_seed)
        print("  Numba JIT ready.")

    col = f"{'scalar':>10}  {'nb-serial':>10}  {'C':>10}  {'nb-par':>10}"
    sp = f"{'vs sc (nb-s)':>13}  {'vs sc (C)':>10}  {'vs sc (nb-p)':>13}  {'vs C (nb-p)':>12}"
    print(f"  {'':35} {col}  {sp}")
    print("  " + "-" * 130)

    shapes = [
        ("activation  (1, 32)", (1, 32)),
        ("activation  (8, 1024)", (8, 1024)),
        ("weight      (16, 32)", (16, 32)),
        ("weight      (1024, 32)", (1024, 32)),
        ("weight   (1024, 1024)", (1024, 1024)),
        ("weight   (4096, 1024)", (4096, 1024)),
    ]

    for label, shape in shapes:
        t = torch.randn(*shape)
        t_np = t.numpy().astype(np.float32)

        sc_s = _time(lambda: float_to_e4m3_bytes_scalar(t), n_warmup, n_runs)
        nbs_s = (
            _time(lambda: float_to_e4m3_bytes_numba_serial(t_np), n_warmup, n_runs)
            if NUMBA_AVAILABLE
            else None
        )
        c_s = _time(lambda: float_to_e4m3_bytes_c(t_np), n_warmup, n_runs)
        nbp_s = (
            _time(lambda: float_to_e4m3_bytes_numba(t_np), n_warmup, n_runs)
            if NUMBA_AVAILABLE
            else None
        )

        nbs_str = _ms(nbs_s) if nbs_s is not None else "  n/a"
        nbp_str = _ms(nbp_s) if nbp_s is not None else "  n/a"

        sp_nbs = _speedup(sc_s, nbs_s) if nbs_s is not None else "  n/a"
        sp_c = _speedup(sc_s, c_s)
        sp_nbp = _speedup(sc_s, nbp_s) if nbp_s is not None else "  n/a"
        sp_cnb = _speedup(c_s, nbp_s) if nbp_s is not None else "  n/a"

        print(
            f"  {label:<35} {_ms(sc_s):>10}  {nbs_str:>10}  {_ms(c_s):>10}  {nbp_str:>10}"
            f"  {sp_nbs:>13}  {sp_c:>10}  {sp_nbp:>13}  {sp_cnb:>12}"
        )


# ---------------------------------------------------------------------------
# Section 2: end-to-end forward pass (Python vs C, cold/warm)
# ---------------------------------------------------------------------------


def bench_forward(n_warmup: int, n_runs: int) -> None:
    print("\n── end-to-end forward pass ──────────────────────────────────────────")
    print("  [cold] = first call, weight cache empty (includes weight quantization)")
    print("  [warm] = subsequent calls, weights already cached")
    print(f"  {'':35} {'python':>10}  {'C':>10}  {'speedup':>10}")
    print("  " + "-" * 70)

    configs = [
        ("(1,   32) ->   16", 1, 32, 16),
        ("(1,  256) ->   64", 1, 256, 64),
        ("(1, 1024) -> 1024", 1, 1024, 1024),
        ("(8, 1024) -> 1024", 8, 1024, 1024),
        ("(1, 4096) -> 4096", 1, 4096, 4096),
    ]

    for label, batch, in_f, out_f in configs:
        x = torch.randn(batch, in_f)
        w = torch.randn(out_f, in_f)
        b = torch.randn(out_f)

        cold_n = max(n_runs // 5, 3)

        def py_cold():
            IPTLinearRTLFunction(
                vec_len=32,
                num_lanes=16,
                pipeline_depth=1,
                out_fmt_sel=OutputFmtSel.OutBF16,
            )(x, w, b)

        def c_cold():
            CIPTLinearRTLFunction(
                vec_len=32,
                num_lanes=16,
                pipeline_depth=1,
                out_fmt_sel=OutputFmtSel.OutBF16,
            )(x, w, b)

        py_cold_s = _time(py_cold, n_warmup=2, n_runs=cold_n)
        c_cold_s = _time(c_cold, n_warmup=2, n_runs=cold_n)
        sp = _speedup(py_cold_s, c_cold_s)
        print(
            f"  {label+'  [cold]':<35} {_ms(py_cold_s):>10}  {_ms(c_cold_s):>10}  {sp:>10}"
        )

        py_fn = IPTLinearRTLFunction(
            vec_len=32,
            num_lanes=16,
            pipeline_depth=1,
            out_fmt_sel=OutputFmtSel.OutBF16,
        )
        c_fn = CIPTLinearRTLFunction(
            vec_len=32,
            num_lanes=16,
            pipeline_depth=1,
            out_fmt_sel=OutputFmtSel.OutBF16,
        )
        py_fn(x, w, b)
        c_fn(x, w, b)

        py_warm_s = _time(lambda: py_fn(x, w, b), n_warmup, n_runs)
        c_warm_s = _time(lambda: c_fn(x, w, b), n_warmup, n_runs)
        sp = _speedup(py_warm_s, c_warm_s)
        print(
            f"  {label+'  [warm]':<35} {_ms(py_warm_s):>10}  {_ms(c_warm_s):>10}  {sp:>10}"
        )


# ---------------------------------------------------------------------------
# Section 3: steady-state inference (warm weights, fresh x, C vs C+nb-par)
# ---------------------------------------------------------------------------


def bench_activation_only(n_warmup: int, n_runs: int) -> None:
    print("\n── steady-state inference (warm weights, varying x) ────────────────")
    print("  C      = CIPTLinearRTLFunction with C activation quantization")
    print("  C+nb-p = same kernel, Numba parallel activation quantization")
    print(
        f"  {'':35} {'python':>10}  {'C':>10}  {'C+nb-p':>10}  {'vs py (C)':>11}  {'vs C (nb-p)':>12}"
    )
    print("  " + "-" * 95)

    configs = [
        ("(1,  1024) -> 1024", 1, 1024, 1024),
        ("(8,  1024) -> 1024", 8, 1024, 1024),
        ("(32, 1024) -> 1024", 32, 1024, 1024),
        ("(1,  4096) -> 4096", 1, 4096, 4096),
    ]

    for label, batch, in_f, out_f in configs:
        w = torch.randn(out_f, in_f)
        b = torch.randn(out_f)
        xs = [torch.randn(batch, in_f) for _ in range(n_runs + n_warmup)]

        py_fn = IPTLinearRTLFunction(
            vec_len=32,
            num_lanes=16,
            pipeline_depth=1,
            out_fmt_sel=OutputFmtSel.OutBF16,
        )
        c_fn = CIPTLinearRTLFunction(
            vec_len=32,
            num_lanes=16,
            pipeline_depth=1,
            out_fmt_sel=OutputFmtSel.OutBF16,
        )
        c_fn_nb = CIPTLinearRTLFunction(
            vec_len=32,
            num_lanes=16,
            pipeline_depth=1,
            out_fmt_sel=OutputFmtSel.OutBF16,
        )
        py_fn(xs[0], w, b)
        c_fn(xs[0], w, b)
        c_fn_nb(xs[0], w, b)

        idx = [0]

        def py_call():
            py_fn(xs[idx[0] % len(xs)], w, b)
            idx[0] += 1

        idx2 = [0]

        def c_call():
            c_fn(xs[idx2[0] % len(xs)], w, b)
            idx2[0] += 1

        idx3 = [0]

        def c_nb_call():
            x = xs[idx3[0] % len(xs)]
            idx3[0] += 1
            x2 = x.reshape(-1, in_f).float()
            x_np = float_to_e4m3_bytes_numba(_tensor_to_f32_numpy(x2))
            w_np, b_np = c_fn_nb._prepare_weights(w, b)
            y = c_fn_nb._call_c(x_np, w_np, b_np, x2.shape[0], in_f, out_f, 0)
            return y.reshape(batch, out_f)

        py_s = _time(py_call, n_warmup, n_runs)
        c_s = _time(c_call, n_warmup, n_runs)
        nb_s = _time(c_nb_call, n_warmup, n_runs) if NUMBA_AVAILABLE else None

        nb_str = _ms(nb_s) if nb_s is not None else "  n/a"
        sp_c = _speedup(py_s, c_s)
        sp_nb_c = _speedup(c_s, nb_s) if nb_s is not None else "  n/a"

        print(
            f"  {label:<35} {_ms(py_s):>10}  {_ms(c_s):>10}  {nb_str:>10}"
            f"  {sp_c:>11}  {sp_nb_c:>12}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--no-forward", action="store_true")
    parser.add_argument("--no-steady", action="store_true")
    args = parser.parse_args()

    nb_ver = (
        f"  |  numba {__import__('numba').__version__}"
        if NUMBA_AVAILABLE
        else "  |  numba not installed"
    )
    print(f"Benchmark  (warmup={args.warmup}, runs={args.runs})")
    print(f"torch {torch.__version__}  |  numpy {np.__version__}{nb_ver}")

    if not args.no_quant:
        bench_quantization(args.warmup, args.runs)

    if not args.no_forward:
        bench_forward(args.warmup, args.runs)

    if not args.no_steady:
        bench_activation_only(args.warmup, args.runs)

    print()


if __name__ == "__main__":
    main()
