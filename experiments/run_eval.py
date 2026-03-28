"""
run_eval.py
-----------
Flexible evaluation runner: patches the Pi0 model with any combination of
quantization settings and logs per-layer RMSE to a results folder.

Matrix path — choose one:
  --mx-input-fmt / --mx-output-fmt   software format-flag quantization
  --functional-model NAME             hardware-accurate simulation (e.g. "ipt")
  (mutually exclusive; default is passthrough = bfloat16/bfloat16)

Vector path (independent of matrix path):
  --vec-input-fmt / --vec-output-fmt  (default: passthrough = bfloat16/bfloat16)

Component selection:
  --active-groups vision,language,action_expert,action_head   (default: all)

Output — written to <results-dir>/<label>/:
  config.json        exact parameters used
  chronological.csv  one row per op call in execution order
  grouped.csv        same rows sorted by (component, layer_name)
  summary.csv        per-component aggregate stats (mx and vec separately)

Usage:
    OPENPI_DIR=/scratch/chloe.wong/openpi \\
    CUDA_VISIBLE_DEVICES=0 \\
    /scratch/chloe.wong/envs/pi0/bin/python experiments/run_eval.py \\
        --label fp8_mx_only \\
        --mx-input-fmt float8_e4m3 --mx-output-fmt bfloat16 \\
        --checkpoint-dir /scratch/chloe.wong/data/pi05_base \\
        --config pi05_droid_jointpos_polaris

    # IPT functional model for matmuls, FP8 for vector ops:
    OPENPI_DIR=/scratch/chloe.wong/openpi \\
    /scratch/chloe.wong/envs/pi0/bin/python experiments/run_eval.py \\
        --label ipt_vec_fp8 \\
        --functional-model ipt \\
        --vec-input-fmt float8_e4m3 --vec-output-fmt bfloat16
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from pi0_inout._jax_stubs import inject as _inject_jax_stubs
_inject_jax_stubs()

from pi0_inout import (
    QuantFormat, QuantGroup,
    StatsTracker,
    patch_model, unpatch_model,
    patch_attn_sdpa, unpatch_attn_sdpa,
    patch_vector_ops, unpatch_vector_ops,
    get_functional_model_factory, list_functional_models,
    set_fp8_mode,
)

PASSTHROUGH = "passthrough"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_or_passthrough(s: str) -> Optional[QuantFormat]:
    """Return None for passthrough, else QuantFormat."""
    if s == PASSTHROUGH:
        return None
    return QuantFormat(s)


def _make_dummy_obs(config_ns: SimpleNamespace, device: torch.device) -> SimpleNamespace:
    H, W = 224, 224
    max_tok = config_ns.max_token_len
    return SimpleNamespace(
        images={
            "base_0_rgb":        torch.randn(1, 3, H, W, dtype=torch.float32, device=device),
            "left_wrist_0_rgb":  torch.randn(1, 3, H, W, dtype=torch.float32, device=device),
            "right_wrist_0_rgb": torch.zeros(1, 3, H, W, dtype=torch.float32, device=device),
        },
        image_masks={
            "base_0_rgb":        torch.ones(1,  dtype=torch.bool, device=device),
            "left_wrist_0_rgb":  torch.ones(1,  dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
        },
        state=torch.randn(1, 32, dtype=torch.float32, device=device),
        tokenized_prompt=      torch.zeros(1, max_tok, dtype=torch.int64, device=device),
        tokenized_prompt_mask= torch.ones(1,  max_tok, dtype=torch.bool,  device=device),
        token_ar_mask=         torch.zeros(1, max_tok, dtype=torch.bool,  device=device),
        token_loss_mask=       torch.zeros(1, max_tok, dtype=torch.bool,  device=device),
    )


def _rel_rmse(rmse: float, ref_rms: float) -> float:
    if ref_rms > 0 and math.isfinite(rmse):
        return rmse / ref_rms
    return float("nan")


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

_CHRON_FIELDS = ["seq", "tag", "layer_name", "component", "rmse", "ref_rms", "rel_rmse"]
_SUMMARY_FIELDS = [
    "tag", "component", "n_layers",
    "mean_rmse", "std_rmse", "max_rmse", "min_rmse",
    "mean_rel_rmse", "total_calls",
]


def _calls_to_rows(calls: list[dict], tag: str) -> list[dict]:
    rows = []
    for rec in calls:
        rows.append({
            "seq":        rec["seq"],
            "tag":        tag,
            "layer_name": rec["name"],
            "component":  rec["component"],
            "rmse":       rec["rmse"],
            "ref_rms":    rec["ref_rms"],
            "rel_rmse":   _rel_rmse(rec["rmse"], rec["ref_rms"]),
        })
    return rows


def _write_chronological(path: Path, mx_calls: list[dict], vec_calls: list[dict]) -> None:
    mx_rows  = _calls_to_rows(mx_calls,  "mx")
    vec_rows = _calls_to_rows(vec_calls, "vec")
    all_rows = sorted(mx_rows + vec_rows, key=lambda r: r["seq"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CHRON_FIELDS)
        w.writeheader()
        w.writerows(all_rows)


def _write_grouped(path: Path, mx_calls: list[dict], vec_calls: list[dict]) -> None:
    mx_rows  = _calls_to_rows(mx_calls,  "mx")
    vec_rows = _calls_to_rows(vec_calls, "vec")
    all_rows = sorted(mx_rows + vec_rows, key=lambda r: (r["component"], r["layer_name"], r["tag"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CHRON_FIELDS)
        w.writeheader()
        w.writerows(all_rows)


def _write_summary(path: Path, mx_tracker: StatsTracker, vec_tracker: StatsTracker) -> None:
    rows = []
    for tag, tracker in [("mx", mx_tracker), ("vec", vec_tracker)]:
        for comp_row in tracker.component_rows():
            rows.append({"tag": tag, **comp_row})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)


_COMPONENTS = ["vision", "language", "action_expert", "action_head"]

_TOP_LEVEL_FIELDS = (
    ["timestamp", "label", "elapsed_seconds", "elapsed_human",
     "mx_input", "mx_output", "vec_input", "vec_output",
     "functional_model", "active_groups"]
    + [f"mx_{c}_mean_rmse"     for c in _COMPONENTS]
    + [f"mx_{c}_mean_rel_rmse" for c in _COMPONENTS]
    + [f"vec_{c}_mean_rmse"     for c in _COMPONENTS]
    + [f"vec_{c}_mean_rel_rmse" for c in _COMPONENTS]
)


def _append_top_level_summary(
    results_dir: Path,
    config_record: dict,
    mx_tracker: StatsTracker,
    vec_tracker: StatsTracker,
) -> None:
    """Append one row to <results_dir>/all_runs_summary.csv."""
    path = results_dir / "all_runs_summary.csv"
    write_header = not path.exists()

    # Build component lookup: {tag: {component: row}}
    comp_lookup: dict[str, dict[str, dict]] = {"mx": {}, "vec": {}}
    for tag, tracker in [("mx", mx_tracker), ("vec", vec_tracker)]:
        for row in tracker.component_rows():
            comp_lookup[tag][row["component"]] = row

    mp = config_record["matrix_path"]
    vp = config_record["vector_path"]
    elapsed_s = config_record.get("elapsed_seconds", float("nan"))
    elapsed_td = str(datetime.timedelta(seconds=int(elapsed_s))) if math.isfinite(elapsed_s) else ""

    row: dict = {
        "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
        "label":           config_record["label"],
        "elapsed_seconds": elapsed_s,
        "elapsed_human":   elapsed_td,
        "mx_input":        mp.get("mx_input_fmt") or "passthrough",
        "mx_output":       mp.get("mx_output_fmt") or "passthrough",
        "vec_input":       vp.get("vec_input_fmt") or "passthrough",
        "vec_output":      vp.get("vec_output_fmt") or "passthrough",
        "functional_model": mp.get("functional_model") or "",
        "active_groups":   "|".join(config_record.get("active_groups", [])),
    }
    for c in _COMPONENTS:
        mx_row  = comp_lookup["mx"].get(c,  {})
        vec_row = comp_lookup["vec"].get(c, {})
        row[f"mx_{c}_mean_rmse"]      = mx_row.get("mean_rmse",     float("nan"))
        row[f"mx_{c}_mean_rel_rmse"]  = mx_row.get("mean_rel_rmse", float("nan"))
        row[f"vec_{c}_mean_rmse"]     = vec_row.get("mean_rmse",     float("nan"))
        row[f"vec_{c}_mean_rel_rmse"] = vec_row.get("mean_rel_rmse", float("nan"))

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TOP_LEVEL_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _print_intermediate(
    label: str,
    mx_tracker: StatsTracker,
    vec_tracker: StatsTracker,
    elapsed_s: float,
) -> None:
    """Print a compact per-component RMSE table to stdout."""
    total_calls = mx_tracker._seq + vec_tracker._seq
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_s)))
    print(f"\n  elapsed={elapsed_str}  layer_calls={total_calls}")
    print(f"  {'component':<14} {'mx_rel_rmse':>12}  {'mx_rmse':>12}  {'vec_rel_rmse':>13}  {'vec_rmse':>12}")
    print(f"  {'-'*14} {'-'*12}  {'-'*12}  {'-'*13}  {'-'*12}")

    mx_by_comp  = {r["component"]: r for r in mx_tracker.component_rows()}
    vec_by_comp = {r["component"]: r for r in vec_tracker.component_rows()}
    components  = ["vision", "language", "action_expert", "action_head"]
    for c in components:
        mx  = mx_by_comp.get(c,  {})
        vec = vec_by_comp.get(c, {})
        mx_rel  = mx.get("mean_rel_rmse", float("nan"))
        mx_abs  = mx.get("mean_rmse",     float("nan"))
        vec_rel = vec.get("mean_rel_rmse", float("nan"))
        vec_abs = vec.get("mean_rmse",     float("nan"))
        print(
            f"  {c:<14} {mx_rel:>12.4e}  {mx_abs:>12.4e}  {vec_rel:>13.4e}  {vec_abs:>12.4e}"
        )


def _start_heartbeat(
    mx_tracker: StatsTracker,
    vec_tracker: StatsTracker,
    t0: float,
    stop_event: threading.Event,
    interval_s: int = 30,
) -> threading.Thread:
    """Background thread: prints a one-liner every `interval_s` seconds."""
    def _loop():
        while not stop_event.wait(timeout=interval_s):
            elapsed = time.monotonic() - t0
            calls   = mx_tracker._seq + vec_tracker._seq
            print(
                f"  [heartbeat] elapsed={datetime.timedelta(seconds=int(elapsed))}  "
                f"layer_calls={calls}",
                flush=True,
            )
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(
    model: nn.Module,
    observations: list,
    device: torch.device,
    active_groups: set[QuantGroup],
    mx_input_fmt: Optional[QuantFormat],
    mx_output_fmt: Optional[QuantFormat],
    vec_input_fmt: Optional[QuantFormat],
    vec_output_fmt: Optional[QuantFormat],
    functional_model_name: Optional[str],
    num_steps: int,
    t0: float,
) -> tuple[StatsTracker, StatsTracker]:
    """
    Patch model, run observations, unpatch.  Returns (mx_tracker, vec_tracker).
    """
    mx_tracker  = StatsTracker()
    vec_tracker = StatsTracker()

    # Resolve effective formats (passthrough = BF16 no-op)
    _mx_in  = mx_input_fmt  or QuantFormat.BFLOAT16
    _mx_out = mx_output_fmt or QuantFormat.BFLOAT16
    _vi     = vec_input_fmt  or QuantFormat.BFLOAT16
    _vo     = vec_output_fmt or QuantFormat.BFLOAT16

    # Resolve functional model factory
    fm_factory = None
    if functional_model_name is not None:
        fm_factory = get_functional_model_factory(functional_model_name)

    patch_model(
        model,
        mx_input_fmt=_mx_in,
        mx_output_fmt=_mx_out,
        tracker=mx_tracker,
        active_groups=active_groups,
        functional_model_factory=fm_factory,
    )
    attn_handles = patch_attn_sdpa(
        model,
        active_groups=active_groups,
        mx_input_fmt=_mx_in,
        mx_output_fmt=_mx_out,
        tracker=mx_tracker,
    )
    vec_handles, vec_ctx = patch_vector_ops(
        model,
        active_groups=active_groups,
        vec_input_fmt=_vi,
        vec_output_fmt=_vo,
        tracker=vec_tracker,
    )

    n_obs = len(observations)
    stop_heartbeat = threading.Event()
    _start_heartbeat(mx_tracker, vec_tracker, t0, stop_heartbeat)

    with torch.no_grad(), vec_ctx:
        for i, obs in enumerate(observations):
            print(f"\n[obs {i + 1}/{n_obs}] running ({num_steps} diffusion steps)...", flush=True)
            obs_t0 = time.monotonic()
            model.sample_actions(str(device), obs, num_steps=num_steps)
            obs_elapsed = time.monotonic() - obs_t0
            print(f"[obs {i + 1}/{n_obs}] done in {obs_elapsed:.1f}s", flush=True)
            _print_intermediate(
                f"obs {i + 1}/{n_obs}",
                mx_tracker, vec_tracker,
                elapsed_s=time.monotonic() - t0,
            )

    stop_heartbeat.set()
    unpatch_model(model)
    unpatch_attn_sdpa(attn_handles)
    unpatch_vector_ops(vec_handles)

    return mx_tracker, vec_tracker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch Pi0 with quantization settings and log per-layer RMSE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Identity
    parser.add_argument("--label", required=True,
                        help="Run label — used as the output folder name under results-dir")

    # Model loading
    parser.add_argument("--checkpoint-dir", default="/scratch/chloe.wong/data/pi05_base")
    parser.add_argument("--config", default="pi05_droid_jointpos_polaris")
    parser.add_argument("--gpu",    type=int, default=0)

    # Eval settings
    parser.add_argument("--n-obs",  type=int, default=4,
                        help="Number of random observations to run")
    parser.add_argument("--steps",  type=int, default=10,
                        help="Diffusion steps per sample_actions call")

    # Matrix path (mutually exclusive)
    mx_group = parser.add_mutually_exclusive_group()
    mx_group.add_argument("--functional-model", metavar="NAME",
                          help=f"Hardware-accurate model for matmuls. "
                               f"Available: {list_functional_models()}")
    mx_group.add_argument("--mx-input-fmt", metavar="FMT",
                          help="Format for matmul inputs (activation + weight). "
                               "Use 'passthrough' for no-op.")
    parser.add_argument("--mx-output-fmt", metavar="FMT", default=PASSTHROUGH,
                        help="Format for matmul outputs. Use 'passthrough' for no-op.")

    # Vector path
    parser.add_argument("--vec-input-fmt",  metavar="FMT", default=PASSTHROUGH,
                        help="Format for vector op inputs. Use 'passthrough' for no-op.")
    parser.add_argument("--vec-output-fmt", metavar="FMT", default=PASSTHROUGH,
                        help="Format for vector op outputs. Use 'passthrough' for no-op.")

    # Component selection
    all_group_names = [g.value for g in QuantGroup]
    parser.add_argument("--active-groups", metavar="G1,G2,...",
                        default=",".join(all_group_names),
                        help=f"Comma-separated groups to quantize. "
                             f"Choices: {all_group_names}")

    # Output
    parser.add_argument("--results-dir",
                        default=str(_REPO / "experiments" / "results"),
                        help="Root directory for results (default: <repo>/experiments/results)")
    parser.add_argument("--fp8-mode", default="po2", choices=["po2", "abs"],
                        help="FP8 scaling mode: po2=power-of-two, abs=absmax")

    args = parser.parse_args()

    # ── Validate ────────────────────────────────────────────────────────────
    if args.functional_model is not None and args.mx_output_fmt != PASSTHROUGH:
        parser.error("--mx-output-fmt has no effect with --functional-model")

    active_groups: set[QuantGroup] = set()
    for g in args.active_groups.split(","):
        g = g.strip()
        try:
            active_groups.add(QuantGroup(g))
        except ValueError:
            parser.error(f"Unknown group '{g}'. Choices: {all_group_names}")

    mx_input_fmt  = _fmt_or_passthrough(args.mx_input_fmt or PASSTHROUGH)
    mx_output_fmt = _fmt_or_passthrough(args.mx_output_fmt)
    vec_input_fmt  = _fmt_or_passthrough(args.vec_input_fmt)
    vec_output_fmt = _fmt_or_passthrough(args.vec_output_fmt)

    set_fp8_mode(args.fp8_mode)

    # ── Device / model ───────────────────────────────────────────────────────
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    from pi0_inout.serve_quant import load_pi0_pytorch, _get_model_config
    config_ns = _get_model_config(args.config)
    print(f"Loading model: {args.config}  checkpoint: {args.checkpoint_dir}")
    model = load_pi0_pytorch(args.config, args.checkpoint_dir, device)
    model.eval()

    torch.manual_seed(0)
    observations = [_make_dummy_obs(config_ns, device) for _ in range(args.n_obs)]
    print(f"Observations: {args.n_obs}  steps: {args.steps}")

    # ── Build config record ──────────────────────────────────────────────────
    config_record = {
        "label":               args.label,
        "checkpoint_dir":      args.checkpoint_dir,
        "model_config":        args.config,
        "n_obs":               args.n_obs,
        "steps":               args.steps,
        "gpu":                 args.gpu,
        "fp8_mode":            args.fp8_mode,
        "active_groups":       [g.value for g in active_groups],
        "matrix_path": {
            "functional_model": args.functional_model,
            "mx_input_fmt":     args.mx_input_fmt,
            "mx_output_fmt":    args.mx_output_fmt,
        },
        "vector_path": {
            "vec_input_fmt":  args.vec_input_fmt,
            "vec_output_fmt": args.vec_output_fmt,
        },
    }

    # ── Run ──────────────────────────────────────────────────────────────────
    print(f"\nRunning config: {args.label}")
    t0 = time.monotonic()
    mx_tracker, vec_tracker = run(
        model=model,
        observations=observations,
        device=device,
        active_groups=active_groups,
        mx_input_fmt=mx_input_fmt,
        mx_output_fmt=mx_output_fmt,
        vec_input_fmt=vec_input_fmt,
        vec_output_fmt=vec_output_fmt,
        functional_model_name=args.functional_model,
        num_steps=args.steps,
        t0=t0,
    )
    elapsed_s = time.monotonic() - t0
    config_record["elapsed_seconds"] = round(elapsed_s, 2)

    # ── Write outputs ────────────────────────────────────────────────────────
    out_dir = Path(args.results_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, default=str)
    )

    _write_chronological(
        out_dir / "chronological.csv",
        mx_tracker.calls, vec_tracker.calls,
    )
    _write_grouped(
        out_dir / "grouped.csv",
        mx_tracker.calls, vec_tracker.calls,
    )
    _write_summary(
        out_dir / "summary.csv",
        mx_tracker, vec_tracker,
    )

    _append_top_level_summary(
        Path(args.results_dir), config_record, mx_tracker, vec_tracker
    )

    # ── Print summary to stdout ───────────────────────────────────────────────
    elapsed_td = datetime.timedelta(seconds=int(elapsed_s))
    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed_td} ({elapsed_s:.1f}s)")
    print(f"Results: {out_dir}")
    print(f"Top-level summary: {Path(args.results_dir) / 'all_runs_summary.csv'}")
    print("\n-- Matrix path --")
    mx_tracker.summary().print()
    print("\n-- Vector path --")
    vec_tracker.summary().print()


if __name__ == "__main__":
    main()
