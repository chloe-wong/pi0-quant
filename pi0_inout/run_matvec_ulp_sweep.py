"""
run_matvec_ulp_sweep.py
-----------------------
Sweep all 16 matrix input/output format combinations ({E4M3, E5M2, FP16, BF16}^2)
with increasing ULP noise injection, recording end-to-end action RMSE vs. ulp_n.

For each (input_fmt, output_fmt) pair:
  - The quantized server runs QuantLinear with those formats and ulp_n noise
    injected into each matmul output.
  - The base server (BF16, no noise) serves as the reference.
  - ULP step size is calibrated to output_fmt (--ulp-fmt = output_fmt).
  - Sweep stops when RMSE >= 0.4 (default) or max_ulp_n is reached.

Usage
-----
    python pi0_inout/run_matvec_ulp_sweep.py \\
        --checkpoint-dir /path/to/model.safetensors_dir \\
        --output-dir ./matvec_ulp_sweep

    # Quick smoke test (3 steps, 4 observations per combo):
    python pi0_inout/run_matvec_ulp_sweep.py \\
        --checkpoint-dir /path/to/ckpt \\
        --max-ulp-n 3 --n-obs 4 --output-dir /tmp/test_sweep

    # Resume an interrupted run:
    python pi0_inout/run_matvec_ulp_sweep.py \\
        --checkpoint-dir /path/to/ckpt --resume --output-dir ./matvec_ulp_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import IO, Optional

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

# Reduced formats to sweep (no FP32 — that is the base reference)
SWEEP_FORMATS = ["float8_e4m3", "float8_e5m2", "float16", "bfloat16"]

# Short display names for plot labels
_SHORT = {
    "float8_e4m3": "e4m3",
    "float8_e5m2": "e5m2",
    "float16":     "fp16",
    "bfloat16":    "bf16",
}


# ---------------------------------------------------------------------------
# Combo generation
# ---------------------------------------------------------------------------

def format_combos() -> list[tuple[str, str]]:
    return [(inf, outf) for inf in SWEEP_FORMATS for outf in SWEEP_FORMATS]


def combo_label(input_fmt: str, output_fmt: str) -> str:
    return f"{input_fmt}__{output_fmt}"


# ---------------------------------------------------------------------------
# Port utilities
# ---------------------------------------------------------------------------

def _wait_for_port(port: int, *, timeout_s: float = 120.0, interval_s: float = 1.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(interval_s)
    return False


def _pids_listening_on_port(port: int) -> set[int]:
    """Return PIDs with a TCP LISTEN socket on `port` (Linux /proc only)."""
    def _inodes(path: str) -> set[str]:
        inodes: set[str] = set()
        with suppress(FileNotFoundError, PermissionError):
            with open(path, "r") as f:
                next(f, None)
                for line in f:
                    parts = line.split()
                    if len(parts) < 10 or parts[3] != "0A":
                        continue
                    with suppress(Exception):
                        if int(parts[1].split(":")[1], 16) == port:
                            inodes.add(parts[9])
        return inodes

    inodes = _inodes("/proc/net/tcp") | _inodes("/proc/net/tcp6")
    if not inodes:
        return set()
    pids: set[int] = set()
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        fd_dir = f"/proc/{pid_str}/fd"
        with suppress(FileNotFoundError, PermissionError):
            for fd in os.listdir(fd_dir):
                with suppress(FileNotFoundError, PermissionError, OSError):
                    target = os.readlink(os.path.join(fd_dir, fd))
                    if target.startswith("socket:[") and target[8:-1] in inodes:
                        pids.add(int(pid_str))
                        break
    return pids


def _kill_listeners_on_port(port: int, *, timeout_s: float = 10.0) -> None:
    """Terminate any process currently listening on `port`."""
    pids = _pids_listening_on_port(port)
    if not pids:
        return
    logger.info("Killing %d existing listener(s) on port %d: %s", len(pids), port, pids)
    for pid in pids:
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _pids_listening_on_port(port):
            return
        time.sleep(0.1)
    for pid in _pids_listening_on_port(port):
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _stop_proc_tree(proc: subprocess.Popen) -> None:
    """Kill the process group started with setsid()."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=15)


def _start_base_server(
    *,
    python: str,
    checkpoint_dir: str,
    config: str,
    port: int,
    gpu: int,
    openpi_dir: Optional[str],
    log_path: Path,
) -> subprocess.Popen:
    serve_script = _THIS_DIR / "serve_quant.py"
    cmd = [
        python, str(serve_script),
        "--config",         config,
        "--checkpoint-dir", checkpoint_dir,
        "--port",           str(port),
        "--gpu",            str(gpu),
        "--input-fmt",      "bfloat16",
        "--output-fmt",     "bfloat16",
    ]
    if openpi_dir:
        cmd += ["--openpi-dir", openpi_dir]

    env = os.environ.copy()
    if openpi_dir:
        env["OPENPI_DIR"] = str(openpi_dir)
    # Pin base server to its own GPU so it doesn't share with quantized server
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd[cmd.index("--gpu") + 1] = "0"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w")
    logger.info("Starting base server: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )


# ---------------------------------------------------------------------------
# Per-combo sweep invocation
# ---------------------------------------------------------------------------

def _build_quant_server_template(
    *,
    python: str,
    checkpoint_dir: str,
    config: str,
    gpu: int,
    openpi_dir: Optional[str],
    input_fmt: str,
    output_fmt: str,
) -> str:
    """
    Build the --quantized-server-cmd template for run_ulp_sweep_two_servers.py.

    {ulp_n} and {quantized_port} are left as literal placeholders — they will
    be substituted by run_ulp_sweep_two_servers._replace_placeholders().
    """
    serve_script = _THIS_DIR / "serve_quant.py"
    parts = [
        python, str(serve_script),
        "--config",         config,
        "--checkpoint-dir", checkpoint_dir,
        "--port",           "{quantized_port}",
        "--gpu",            "0",   # with CUDA_VISIBLE_DEVICES={gpu}, only one GPU visible
        "--input-fmt",      input_fmt,
        "--output-fmt",     output_fmt,
        "--ulp-fmt",        output_fmt,
        "--ulp-n",          "{ulp_n}",
    ]
    if openpi_dir:
        parts += ["--openpi-dir", openpi_dir]
    cmd_str = " ".join(parts)
    # Pin quantized server to its own GPU
    return f"env CUDA_VISIBLE_DEVICES={gpu} {cmd_str}"


def _run_combo_sweep(
    *,
    python: str,
    input_fmt: str,
    output_fmt: str,
    combo_dir: Path,
    base_port: int,
    quantized_port: int,
    n_obs: int,
    seed: int,
    max_ulp_n: int,
    ulp_step: int,
    rmse_threshold: float,
    ready_timeout: float,
    checkpoint_dir: str,
    config: str,
    gpu_quant: int,
    openpi_dir: Optional[str],
    use_fixed_pi0_noise: bool,
    log_fh: IO[str],
) -> list[dict]:
    """
    Invoke run_ulp_sweep_two_servers.py for one (input_fmt, output_fmt) combo.
    Returns list of {ulp_n, rmse} dicts parsed from the inner run.log.
    """
    inner_log_dir = combo_dir / "inner_logs"
    inner_log_dir.mkdir(parents=True, exist_ok=True)

    template = _build_quant_server_template(
        python=python,
        checkpoint_dir=checkpoint_dir,
        config=config,
        gpu=gpu_quant,
        openpi_dir=openpi_dir,
        input_fmt=input_fmt,
        output_fmt=output_fmt,
    )

    sweep_script = _THIS_DIR / "run_ulp_sweep_two_servers.py"
    cmd = [
        python, str(sweep_script),
        "--base-port",          str(base_port),
        "--quantized-port",     str(quantized_port),
        "--n-obs",              str(n_obs),
        "--seed",               str(seed),
        "--start-ulp-n",        "0",
        "--max-ulp-n",          str(max_ulp_n),
        "--ulp-step",           str(ulp_step),
        "--rmse-threshold",     str(rmse_threshold),
        "--ready-timeout-s",    str(ready_timeout),
        "--log-dir",            str(inner_log_dir),
        "--kill-existing-quantized-server",   # always clean up before each ulp_n step
        "--quantized-server-cmd", template,
    ]
    if use_fixed_pi0_noise:
        cmd.append("--use-fixed-pi0-noise")

    label = combo_label(input_fmt, output_fmt)
    msg = f"\n[combo {label}] Running sweep..."
    print(msg)
    log_fh.write(msg + "\n")
    log_fh.flush()

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Echo inner stdout so user sees progress
    if result.stdout:
        sys.stdout.write(result.stdout)
        log_fh.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
        log_fh.write(result.stderr)
    log_fh.flush()

    return _parse_inner_log(inner_log_dir)


def _parse_inner_log(inner_log_dir: Path) -> list[dict]:
    """
    Find the run.log written by run_ulp_sweep_two_servers and extract
    (ulp_n, rmse) data points plus stop reason.
    """
    run_logs = sorted(inner_log_dir.glob("run-*/run.log"))
    if not run_logs:
        logger.warning("No run.log found in %s", inner_log_dir)
        return []

    text = run_logs[-1].read_text(encoding="utf-8")
    data = []
    for m in re.finditer(r'ulp_n=(\d+)\s+rmse=([0-9.eE+\-]+)', text):
        data.append({"ulp_n": int(m.group(1)), "rmse": float(m.group(2))})
    return data


def _stop_reason(inner_log_dir: Path) -> str:
    run_logs = sorted(inner_log_dir.glob("run-*/run.log"))
    if not run_logs:
        return "no_log"
    text = run_logs[-1].read_text(encoding="utf-8")
    if "threshold violated" in text:
        return "threshold_violated"
    if "non-finite" in text:
        return "nonfinite"
    return "max_reached"


def _max_tol_ulp_n(data: list[dict], threshold: float, max_ulp_n: int) -> int:
    """Highest ulp_n with rmse < threshold; max_ulp_n if never exceeded; 0 if violated at step 1."""
    passing = [d["ulp_n"] for d in data if d["rmse"] < threshold]
    if not passing:
        return 0
    best = max(passing)
    # If no data point exceeded the threshold at all, report max_ulp_n
    if all(d["rmse"] < threshold for d in data):
        return max_ulp_n
    return best


# ---------------------------------------------------------------------------
# Summary I/O
# ---------------------------------------------------------------------------

def _write_summary_json(results: list[dict], path: Path) -> None:
    path.write_text(json.dumps(results, indent=2))


def _write_summary_csv(results: list[dict], path: Path) -> None:
    rows = []
    for r in results:
        for point in r.get("data", []):
            rows.append({
                "combo":       combo_label(r["input_fmt"], r["output_fmt"]),
                "input_fmt":   r["input_fmt"],
                "output_fmt":  r["output_fmt"],
                "ulp_n":       point["ulp_n"],
                "rmse":        point["rmse"],
            })
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["combo", "input_fmt", "output_fmt", "ulp_n", "rmse"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_grid(results: list[dict], *, threshold: float, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping ulp_rmse_grid.png")
        return

    fmts = SWEEP_FORMATS
    n = len(fmts)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 3 * n), squeeze=False)
    fig.suptitle("ULP noise sweep: ulp_n vs RMSE per quantization combo", fontsize=13)

    # Build lookup: (input_fmt, output_fmt) → {data, max_tol_ulp_n}
    lookup = {(r["input_fmt"], r["output_fmt"]): r for r in results}

    for ri, inf in enumerate(fmts):
        for ci, outf in enumerate(fmts):
            ax = axes[ri][ci]
            r = lookup.get((inf, outf))
            if r and r.get("data"):
                xs = [d["ulp_n"] for d in r["data"]]
                ys = [d["rmse"]  for d in r["data"]]
                ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, color="steelblue")
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"thr={threshold}")
            ax.set_title(f"in={_SHORT[inf]}\nout={_SHORT[outf]}", fontsize=8)
            ax.set_xlabel("ulp_n", fontsize=7)
            ax.set_ylabel("RMSE", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_heatmap(results: list[dict], *, max_ulp_n: int, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy not available; skipping tolerance_hmap.png")
        return

    fmts = SWEEP_FORMATS
    n = len(fmts)
    lookup = {(r["input_fmt"], r["output_fmt"]): r for r in results}

    mat = np.full((n, n), float("nan"))
    for ri, inf in enumerate(fmts):
        for ci, outf in enumerate(fmts):
            r = lookup.get((inf, outf))
            if r is not None:
                mat[ri, ci] = r.get("max_tol_ulp_n", 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=max_ulp_n, aspect="auto")
    fig.colorbar(im, ax=ax, label="Max tolerable ulp_n (RMSE < threshold)")

    short = [_SHORT[f] for f in fmts]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short)
    ax.set_xlabel("output_fmt")
    ax.set_ylabel("input_fmt")
    ax.set_title("Max tolerable ULP noise per format combo")

    for ri in range(n):
        for ci in range(n):
            v = mat[ri, ci]
            if not (v != v):  # not NaN
                ax.text(ci, ri, str(int(v)), ha="center", va="center",
                        fontsize=10, color="black" if v < max_ulp_n * 0.7 else "white")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _find_resume_dir(output_dir: Path) -> Optional[Path]:
    runs = sorted(output_dir.glob("run-*"))
    return runs[-1] if runs else None


def _combo_done(combo_dir: Path) -> bool:
    rj = combo_dir / "results.json"
    if not rj.exists():
        return False
    try:
        data = json.loads(rj.read_text())
        return bool(data.get("data"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        force=True,
    )
    args = _parse_args()

    python = args.python or sys.executable

    # ── Determine output directory ────────────────────────────────────────
    output_root = Path(args.output_dir)
    if args.resume:
        run_dir = _find_resume_dir(output_root)
        if run_dir is None:
            logger.warning("--resume specified but no existing run found; starting fresh")
            run_dir = output_root / _timestamp_tag()
        else:
            logger.info("Resuming run: %s", run_dir)
    else:
        run_dir = output_root / _timestamp_tag()

    run_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_dir / "run.log"
    run_log_fh = run_log_path.open("a", encoding="utf-8")

    combos = format_combos()
    logger.info("Sweep: %d format combinations", len(combos))
    run_log_fh.write(f"# run_dir={run_dir}\n")
    run_log_fh.write(f"# combos={len(combos)}  n_obs={args.n_obs}  seed={args.seed}\n")
    run_log_fh.write(f"# max_ulp_n={args.max_ulp_n}  ulp_step={args.ulp_step}  "
                     f"rmse_threshold={args.rmse_threshold}\n\n")
    run_log_fh.flush()

    # ── Clear both ports before starting anything ────────────────────────
    _kill_listeners_on_port(args.base_port)
    _kill_listeners_on_port(args.quantized_port)
    openpi_dir = args.openpi_dir or str(_REPO_ROOT / "openpi")
    base_log = run_dir / "base_server.log"
    base_proc = _start_base_server(  # bfloat16 reference, no ULP noise
        python=python,
        checkpoint_dir=args.checkpoint_dir,
        config=args.config,
        port=args.base_port,
        gpu=args.gpu_base,
        openpi_dir=openpi_dir,
        log_path=base_log,
    )
    logger.info("Waiting for base server on port %d...", args.base_port)
    if not _wait_for_port(args.base_port, timeout_s=args.ready_timeout):
        logger.error("Base server did not start within %.1fs — aborting", args.ready_timeout)
        _stop_proc_tree(base_proc)
        sys.exit(1)
    logger.info("Base server ready on port %d", args.base_port)

    # ── Sweep ─────────────────────────────────────────────────────────────
    all_results: list[dict] = []

    try:
        for i, (input_fmt, output_fmt) in enumerate(combos):
            label = combo_label(input_fmt, output_fmt)
            combo_dir = run_dir / label
            combo_dir.mkdir(exist_ok=True)

            if args.resume and _combo_done(combo_dir):
                existing = json.loads((combo_dir / "results.json").read_text())
                all_results.append(existing)
                logger.info("[%d/%d] SKIP %s (already done)", i + 1, len(combos), label)
                continue

            logger.info("[%d/%d] Starting %s", i + 1, len(combos), label)
            run_log_fh.write(f"\n# combo {i+1}/{len(combos)}: {label}\n")
            run_log_fh.flush()

            data = _run_combo_sweep(
                python=python,
                input_fmt=input_fmt,
                output_fmt=output_fmt,
                combo_dir=combo_dir,
                base_port=args.base_port,
                quantized_port=args.quantized_port,
                n_obs=args.n_obs,
                seed=args.seed,
                max_ulp_n=args.max_ulp_n,
                ulp_step=args.ulp_step,
                rmse_threshold=args.rmse_threshold,
                ready_timeout=args.ready_timeout,
                checkpoint_dir=args.checkpoint_dir,
                config=args.config,
                gpu_quant=args.gpu_quant,
                openpi_dir=openpi_dir,
                use_fixed_pi0_noise=args.use_fixed_pi0_noise,
                log_fh=run_log_fh,
            )

            inner_log_dir = combo_dir / "inner_logs"
            stop = _stop_reason(inner_log_dir)
            max_tol = _max_tol_ulp_n(data, args.rmse_threshold, args.max_ulp_n)

            combo_result = {
                "input_fmt":     input_fmt,
                "output_fmt":    output_fmt,
                "combo":         label,
                "data":          data,
                "max_tol_ulp_n": max_tol,
                "stop_reason":   stop,
            }
            (combo_dir / "results.json").write_text(json.dumps(combo_result, indent=2))
            all_results.append(combo_result)

            summary_line = (
                f"{label:30s}  max_tol_ulp_n={max_tol:3d}  "
                f"n_points={len(data):2d}  stop={stop}"
            )
            print(summary_line)
            run_log_fh.write(summary_line + "\n")
            run_log_fh.flush()

            # Write running summary after each combo (safe if interrupted)
            _write_summary_json(all_results, run_dir / "summary.json")
            _write_summary_csv(all_results, run_dir / "summary.csv")

    finally:
        logger.info("Stopping base server...")
        _stop_proc_tree(base_proc)
        run_log_fh.close()

    # ── Final summary + plots ─────────────────────────────────────────────
    _write_summary_json(all_results, run_dir / "summary.json")
    _write_summary_csv(all_results, run_dir / "summary.csv")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _plot_grid(all_results, threshold=args.rmse_threshold,
               out_path=plots_dir / "ulp_rmse_grid.png")
    _plot_heatmap(all_results, max_ulp_n=args.max_ulp_n,
                  out_path=plots_dir / "tolerance_hmap.png")

    print(f"\nDone. Results in: {run_dir}")
    print(f"  summary.json     : {run_dir / 'summary.json'}")
    print(f"  ulp_rmse_grid.png: {plots_dir / 'ulp_rmse_grid.png'}")
    print(f"  tolerance_hmap   : {plots_dir / 'tolerance_hmap.png'}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp_tag() -> str:
    return time.strftime("run-%Y%m%d-%H%M%S", time.localtime())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep 16 (input_fmt x output_fmt) quantization combos with ULP noise injection, "
            "recording RMSE vs. ulp_n and generating summary plots."
        )
    )
    p.add_argument("--checkpoint-dir", required=True,
                   help="Directory containing model.safetensors (or gs:// path with warning)")
    p.add_argument("--config", default="pi05_droid_jointpos_polaris",
                   help="openpi training config name")
    p.add_argument("--base-port",      type=int, default=8000)
    p.add_argument("--quantized-port", type=int, default=8002)
    p.add_argument("--gpu-base",  type=int, default=0, help="CUDA device for base server")
    p.add_argument("--gpu-quant", type=int, default=1, help="CUDA device for quantized server")
    p.add_argument("--n-obs",      type=int,   default=16,  help="Observations per combo")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--max-ulp-n",  type=int,   default=5000)
    p.add_argument("--ulp-step",   type=int,   default=1)
    p.add_argument("--rmse-threshold", type=float, default=0.4)
    p.add_argument("--ready-timeout",  type=float, default=120.0,
                   help="Seconds to wait for a server to become ready")
    p.add_argument("--output-dir", default="matvec_ulp_sweep")
    p.add_argument("--resume", action="store_true",
                   help="Skip combos that already have results.json with data")
    p.add_argument("--python", default=None,
                   help="Python interpreter for servers (default: sys.executable)")
    p.add_argument("--openpi-dir", default=None,
                   help="Path to openpi repo root (passed to serve_quant.py). Default: repo/openpi")
    p.add_argument("--use-fixed-pi0-noise", action="store_true",
                   help="Use deterministic pi0_noise for reproducible comparisons (default: False)")
    return p.parse_args()


if __name__ == "__main__":
    main()
