"""
run_benchmark.py
----------------
Orchestrates a full sweep of all (input_fmt × output_fmt) combinations,
collecting RMSE statistics, benchmark success rates, and video outputs
for each configuration.

For each of the 25 format combinations:
  1. Start serve_quant.py on a free port (--stats-output <path>)
  2. Wait until the WebSocket server is accepting connections
  3. Run sim-evals run_eval.py for the requested number of episodes and scenes
  4. Wait for eval to finish
  5. Shut down the server (which writes the RMSE stats JSON on exit)
  6. Read and merge: RMSE stats + eval results + video paths
  7. Move to the next combination

Output layout:
  <output_dir>/
    summary.json          ← one entry per format combo, all metrics
    summary.csv           ← same, as a flat CSV (requires pandas)
    float32_float32/      ← one subdirectory per (input, output) combo
      stats.json          ← per-layer + per-component RMSE
      eval_results.json   ← success rate per scene + episode
      videos/             ← symlinked or copied from sim-evals output
    float16_float32/
      ...
    ...

Usage:
    # Minimal: run all 25 combos, 5 episodes per scene, 3 scenes
    python run_benchmark.py \\
        --sim-evals-dir /path/to/sim-evals \\
        --openpi-dir /path/to/openpi \\
        --checkpoint-dir gs://openpi-assets/checkpoints/pi05_droid_jointpos \\
        --config pi05_droid_jointpos_polaris \\
        --episodes 5 --scenes 1 2 3

    # Run only specific format pairs
    python run_benchmark.py \\
        --sim-evals-dir /path/to/sim-evals \\
        --input-fmts float32 float8_e4m3 \\
        --output-fmts float32 float16

    # Resume a partial run (skips combos that already have stats.json)
    python run_benchmark.py ... --resume

Environment variables:
    OPENPI_DATA_HOME    Path to openpi data cache (checkpoints, tokenizer)
    CUDA_VISIBLE_DEVICES  GPU to use
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format combinations
# ---------------------------------------------------------------------------

ALL_FORMATS = ["bfloat16", "float16", "float8_e4m3", "float8_e5m2"]


def format_pairs(input_fmts: list[str], output_fmts: list[str]) -> list[tuple[str, str]]:
    return [(inf, outf) for inf in input_fmts for outf in output_fmts]


def combo_name(input_fmt: str, output_fmt: str) -> str:
    return f"{input_fmt}__{output_fmt}"


# ---------------------------------------------------------------------------
# Port management
# ---------------------------------------------------------------------------

def find_free_port(start: int = 8100) -> int:
    """Find a free TCP port starting from `start`."""
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{start + 200}")


def wait_for_port(port: int, timeout: float = 300.0, interval: float = 2.0) -> bool:
    """
    Poll until something is listening on localhost:port, or timeout expires.
    Returns True if port opened, False if timed out.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_server(
    args: argparse.Namespace,
    port: int,
    input_fmt: str,
    output_fmt: str,
    stats_path: Path,
) -> subprocess.Popen:
    """
    Start serve_quant.py as a subprocess.

    Returns the Popen object (call .kill() / .wait() to clean up).

    Python selection (in priority order):
      1. --python PATH          explicit interpreter
      2. --openpi-dir PATH      use "uv run python" from the openpi directory
      3. sys.executable         fallback (must already have torch + openpi installed)
    """
    serve_script = Path(__file__).parent / "serve_quant.py"

    server_args = [
        str(serve_script),
        "--config",         args.config,
        "--checkpoint-dir", args.checkpoint_dir,
        "--port",           str(port),
        "--gpu",            str(args.gpu),
        "--input-fmt",      input_fmt,
        "--output-fmt",     output_fmt,
        "--stats-output",   str(stats_path),
    ]
    if args.openpi_dir:
        server_args += ["--openpi-dir", str(args.openpi_dir)]

    # Choose the Python interpreter
    if args.python:
        interpreter = [args.python]
    else:
        interpreter = [sys.executable]
    cmd = interpreter + server_args

    env = os.environ.copy()
    if args.openpi_data_home:
        env["OPENPI_DATA_HOME"] = args.openpi_data_home
    if args.openpi_dir:
        env["OPENPI_DIR"] = str(args.openpi_dir)

    log_path = stats_path.parent / "server.log"
    log_file = open(log_path, "w")

    logger.info(f"  Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # create new process group so we can kill the whole tree
    )
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """Send SIGTERM to the server process group and wait."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(
    args: argparse.Namespace,
    port: int,
    output_dir: Path,
) -> dict:
    """
    Run sim-evals run_eval.py (one scene at a time) and return parsed results.

    Returns a dict with keys:
        success_rate: float   (mean across scenes)
        by_scene: dict        {scene_id: {success_rate, n_episodes}}
        eval_log: str         raw stdout (last scene)
        video_dir: Path
        video_files: list[str]
    """
    sim_evals_dir = Path(args.sim_evals_dir)
    run_eval_script = sim_evals_dir / "run_eval.py"

    if not run_eval_script.exists():
        logger.warning(f"  run_eval.py not found at {run_eval_script}; skipping eval")
        return {"success_rate": None, "by_scene": {}, "eval_log": "", "video_dir": None,
                "video_files": []}

    video_out = output_dir / "videos"
    video_out.mkdir(exist_ok=True)

    # Use uv run python from the sim-evals dir so the correct venv (Isaac Sim) is used
    uv_bin = shutil.which("uv") or "uv"

    by_scene = {}
    all_videos = []
    last_log = ""

    for scene in args.scenes:
        scene_video_out = video_out / f"scene{scene}"
        scene_video_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            uv_bin, "run", "python", str(run_eval_script),
            "--port",       str(port),
            "--host",       "localhost",
            "--episodes",   str(args.episodes),
            "--scene",      str(scene),
            "--headless",
            "--output-dir", str(scene_video_out),
        ]

        logger.info(f"  Running eval scene {scene}: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=str(sim_evals_dir),
                capture_output=True,
                text=True,
                timeout=args.eval_timeout,
                env=os.environ,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"  Eval scene {scene} timed out after {args.eval_timeout}s")
            by_scene[str(scene)] = {
                "n_success": None, "n_episodes": args.episodes, "success_rate": None,
            }
            continue

        last_log = result.stdout + result.stderr
        (output_dir / f"eval_scene{scene}.log").write_text(last_log)

        m = re.search(rf"[Ss]cene\s+{scene}[:\s]+(\d+)/(\d+)", last_log)
        if m:
            n_succ  = int(m.group(1))
            n_total = int(m.group(2))
            by_scene[str(scene)] = {
                "n_success":    n_succ,
                "n_episodes":   n_total,
                "success_rate": n_succ / n_total if n_total > 0 else None,
            }
        else:
            logger.warning(f"  Could not parse success rate for scene {scene}")
            by_scene[str(scene)] = {
                "n_success": None, "n_episodes": args.episodes, "success_rate": None,
            }

        all_videos += sorted(str(p) for p in scene_video_out.glob("**/*.mp4"))

    # Aggregate overall success rate (mean across scenes)
    rates = [v["success_rate"] for v in by_scene.values() if v["success_rate"] is not None]
    overall_rate = sum(rates) / len(rates) if rates else None

    return {
        "success_rate": overall_rate,
        "by_scene":     by_scene,
        "eval_log":     last_log[:4000],
        "video_dir":    str(video_out),
        "video_files":  all_videos,
    }


# ---------------------------------------------------------------------------
# Per-combo runner
# ---------------------------------------------------------------------------

def run_one_combo(
    args: argparse.Namespace,
    input_fmt: str,
    output_fmt: str,
    output_dir: Path,
    port: int,
) -> dict:
    """
    Run the full pipeline for one (input_fmt, output_fmt) pair.
    Returns a dict with all results merged.
    """
    stats_path = output_dir / "stats.json"
    combo = combo_name(input_fmt, output_fmt)

    logger.info(f"\n{'─'*60}")
    logger.info(f"  Config: input={input_fmt}  output={output_fmt}")
    logger.info(f"  Output: {output_dir}")

    # --- Start server --------------------------------------------------------
    server_proc = start_server(args, port, input_fmt, output_fmt, stats_path)

    try:
        logger.info(f"  Waiting for server on port {port}...")
        if not wait_for_port(port, timeout=args.server_timeout):
            logger.error(f"  Server did not start within {args.server_timeout}s")
            return {
                "combo":      combo,
                "input_fmt":  input_fmt,
                "output_fmt": output_fmt,
                "error":      "server_start_timeout",
            }
        logger.info(f"  Server ready on port {port}")

        # --- Run eval --------------------------------------------------------
        eval_results = run_eval(args, port, output_dir)

    finally:
        # --- Stop server (triggers stats write on exit) ----------------------
        logger.info(f"  Stopping server...")
        stop_server(server_proc)
        logger.info(f"  Server stopped")

    # --- Read RMSE stats -----------------------------------------------------
    rmse_stats = {}
    if stats_path.exists():
        try:
            rmse_stats = json.loads(stats_path.read_text())
        except json.JSONDecodeError:
            logger.warning(f"  Could not parse stats JSON at {stats_path}")
    else:
        logger.warning(f"  No stats.json found at {stats_path} (server may have crashed)")

    # Compute aggregate RMSE per component for the summary
    component_rmse = {}
    for row in rmse_stats.get("components", []):
        component_rmse[row["component"]] = row["mean_rmse"]

    # --- Build result record -------------------------------------------------
    result = {
        "combo":        combo,
        "input_fmt":    input_fmt,
        "output_fmt":   output_fmt,
        # Benchmark
        "success_rate": eval_results.get("success_rate"),
        "by_scene":     eval_results.get("by_scene", {}),
        "video_dir":    eval_results.get("video_dir"),
        "video_files":  eval_results.get("video_files", []),
        # RMSE
        "component_rmse": component_rmse,
        "rmse_stats_path": str(stats_path),
    }

    # Save per-combo merged result
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))

    logger.info(f"  Done: success_rate={result['success_rate']}  "
                f"rmse_components={list(component_rmse.keys())}")
    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        force=True,
    )

    # Build output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(args.output_dir) / timestamp
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output root: {output_root}")

    pairs = format_pairs(args.input_fmts, args.output_fmts)
    logger.info(f"Format combinations: {len(pairs)}")
    for inf, outf in pairs:
        logger.info(f"  {inf} × {outf}")

    all_results = []
    port = find_free_port(args.port)

    for i, (input_fmt, output_fmt) in enumerate(pairs):
        name = combo_name(input_fmt, output_fmt)
        combo_dir = output_root / name
        combo_dir.mkdir(exist_ok=True)

        # Skip if already done (resume mode)
        if args.resume and (combo_dir / "result.json").exists():
            logger.info(f"[{i+1}/{len(pairs)}] Skipping {name} (already done)")
            existing = json.loads((combo_dir / "result.json").read_text())
            all_results.append(existing)
            continue

        logger.info(f"\n[{i+1}/{len(pairs)}] Running {name}")
        result = run_one_combo(args, input_fmt, output_fmt, combo_dir, port)
        all_results.append(result)

        # Write running summary after each combo (safe if interrupted)
        _write_summary(all_results, output_root)

    logger.info(f"\n{'='*60}")
    logger.info(f"All {len(pairs)} combinations complete.")
    logger.info(f"Results: {output_root}/summary.json")
    _write_summary(all_results, output_root)
    _print_summary_table(all_results)


def _write_summary(results: list[dict], output_root: Path) -> None:
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    # CSV (flat, one row per combo)
    try:
        import csv
        csv_path = output_root / "summary.csv"
        if not results:
            return
        # Flatten component_rmse into top-level columns
        flat_rows = []
        for r in results:
            row = {k: v for k, v in r.items()
                   if k not in ("component_rmse", "by_scene", "video_files")}
            for comp, rmse in r.get("component_rmse", {}).items():
                row[f"rmse_{comp}"] = rmse
            for scene, info in r.get("by_scene", {}).items():
                row[f"success_scene{scene}"] = info.get("success_rate")
            flat_rows.append(row)

        all_keys = list(dict.fromkeys(k for row in flat_rows for k in row))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_rows)
    except Exception as e:
        logger.warning(f"Could not write CSV: {e}")


def _print_summary_table(results: list[dict]) -> None:
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    header = f"{'input':12s}  {'output':12s}  {'success':8s}  {'rmse_vision':12s}  {'rmse_lang':12s}  {'rmse_act':10s}"
    print(header)
    print("-" * len(header))
    for r in results:
        rmse = r.get("component_rmse", {})
        success = r.get("success_rate")
        print(
            f"{r['input_fmt']:12s}  {r['output_fmt']:12s}  "
            f"{(f'{success:.2f}' if success is not None else 'N/A'):8s}  "
            f"{_fmtf(rmse.get('vision')):12s}  "
            f"{_fmtf(rmse.get('language')):12s}  "
            f"{_fmtf(rmse.get('action_head')):10s}"
        )


def _fmtf(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4e}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep all (input_fmt × output_fmt) combinations with benchmark + RMSE."
    )

    # Paths
    p.add_argument("--sim-evals-dir", required=True,
                   help="Path to the sim-evals repository root (contains run_eval.py)")
    p.add_argument("--openpi-dir", default=None,
                   help="Path to the openpi repository root (used to find uv env)")
    p.add_argument("--checkpoint-dir",
                   default="gs://openpi-assets/checkpoints/pi05_droid_jointpos",
                   help="Model checkpoint path or GCS URI")
    p.add_argument("--config",
                   default="pi05_droid_jointpos_polaris",
                   help="openpi training config name")
    p.add_argument("--output-dir",
                   default="./results",
                   help="Root directory for all outputs")
    p.add_argument("--openpi-data-home", default=None,
                   help="Value for OPENPI_DATA_HOME env var")
    p.add_argument("--python", default=None,
                   help="Python interpreter to use for serve_quant.py. "
                        "If not set and --openpi-dir is given, 'uv run python' is used. "
                        "Otherwise sys.executable is used (must have torch+openpi).")

    # Format sweep
    p.add_argument("--input-fmts", nargs="+", default=ALL_FORMATS,
                   choices=ALL_FORMATS,
                   help="Input formats to sweep (default: all 5)")
    p.add_argument("--output-fmts", nargs="+", default=ALL_FORMATS,
                   choices=ALL_FORMATS,
                   help="Output formats to sweep (default: all 5)")

    # Eval settings
    p.add_argument("--episodes", type=int, default=5,
                   help="Episodes per scene")
    p.add_argument("--scenes", type=int, nargs="+", default=[1, 2, 3],
                   help="Scene IDs to evaluate")
    p.add_argument("--gpu", type=int, default=0,
                   help="CUDA device index")
    p.add_argument("--port", type=int, default=8100,
                   help="Starting port to search from")

    # Timeouts
    p.add_argument("--server-timeout", type=float, default=300.0,
                   help="Seconds to wait for server to become ready")
    p.add_argument("--eval-timeout", type=float, default=3600.0,
                   help="Seconds to wait for run_eval.py to finish")

    # Control
    p.add_argument("--resume", action="store_true",
                   help="Skip combos that already have a result.json")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
