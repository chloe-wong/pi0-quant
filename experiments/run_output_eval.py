"""
run_output_eval.py
------------------
Like run_eval.py, but measures RMSE on the **final action output** rather than
per-layer activations.  Runs the model twice on the same observations:
  1. Unpatched (baseline) — collects reference action tensors.
  2. Patched (quantized)  — collects quantized action tensors.
Then computes per-observation and overall RMSE between the two.

Matrix path — choose one:
  --mx-input-fmt / --mx-output-fmt   software format-flag quantization
  --functional-model NAME             hardware-accurate simulation (e.g. "ipt")
  (mutually exclusive; default is passthrough = bfloat16/bfloat16)

Vector path (independent of matrix path):
  --vec-input-fmt / --vec-output-fmt  (default: passthrough = bfloat16/bfloat16)

Component selection:
  --active-groups vision,language,action_expert,action_head   (default: all)

Output — written to <results-dir>/<label>/:
  config.json               exact parameters used
  action_rmse.csv           per-observation rmse, ref_rms, rel_rmse + overall row
Appended to <results-dir>/all_runs_output_summary.csv for cross-run comparison.

Usage:
    OPENPI_DIR=/scratch/chloe.wong/openpi \\
    CUDA_VISIBLE_DEVICES=0 \\
    /scratch/chloe.wong/envs/pi0/bin/python experiments/run_output_eval.py \\
        --label fp8_mx_only \\
        --mx-input-fmt float8_e4m3 --mx-output-fmt bfloat16 \\
        --checkpoint-dir /scratch/chloe.wong/data/pi05_base \\
        --config pi05_droid_jointpos_polaris

    # Sanity check — bfloat16/bfloat16 must give exactly 0 RMSE:
    /scratch/chloe.wong/envs/pi0/bin/python experiments/run_output_eval.py \\
        --label passthrough \\
        --n-obs 2 --steps 5
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
from typing import Optional

import numpy as np
import torch

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
from pi0_inout.eval_harness import _compute_action_rmse
from pi0_inout.serve_quant import _load_norm_stats, Pi0PyTorchPolicy

PASSTHROUGH = "passthrough"


# ---------------------------------------------------------------------------
# Helpers (mirrored from run_eval.py)
# ---------------------------------------------------------------------------

def _fmt_or_passthrough(s: str) -> Optional[QuantFormat]:
    if s == PASSTHROUGH:
        return None
    return QuantFormat(s)


def _make_dummy_obs(rng: np.random.Generator) -> dict:
    """Random observation dict compatible with Pi0PyTorchPolicy.infer()."""
    return {
        "observation/exterior_image_1_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left":      rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position":        rng.random(7).astype(np.float32),
        "observation/gripper_position":      rng.random(1).astype(np.float32),
        "prompt": "Grab the object",
    }


# ---------------------------------------------------------------------------
# Progress helpers (mirrored from run_eval.py)
# ---------------------------------------------------------------------------

def _start_heartbeat(
    label: str,
    t0: float,
    stop_event: threading.Event,
    interval_s: int = 30,
    mx_tracker: Optional[StatsTracker] = None,
    vec_tracker: Optional[StatsTracker] = None,
) -> threading.Thread:
    def _loop():
        while not stop_event.wait(timeout=interval_s):
            elapsed = time.monotonic() - t0
            calls = (mx_tracker._seq if mx_tracker else 0) + (vec_tracker._seq if vec_tracker else 0)
            msg = (
                f"  [heartbeat] {label}  elapsed={datetime.timedelta(seconds=int(elapsed))}"
                f"  layer_calls={calls}"
            )
            print(msg, flush=True)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Action collection
# ---------------------------------------------------------------------------

def _collect_actions(
    policy: Pi0PyTorchPolicy,
    observations: list,
    num_steps: int,
    label: str,
    t0: float,
) -> list[torch.Tensor]:
    """Run policy on all observations; return list of CPU action tensors."""
    actions = []
    n = len(observations)
    stop = threading.Event()
    _start_heartbeat(label, t0, stop)
    for i, obs in enumerate(observations):
        print(f"\n[{label}] obs {i+1}/{n} ({num_steps} diffusion steps)...", flush=True)
        obs_t0 = time.monotonic()
        result = policy.infer(obs)
        act = torch.from_numpy(result["actions"])
        print(f"[{label}] obs {i+1}/{n} done in {time.monotonic()-obs_t0:.1f}s", flush=True)
        actions.append(act)
    stop.set()
    return actions


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(
    policy: Pi0PyTorchPolicy,
    observations: list,
    active_groups: set[QuantGroup],
    mx_input_fmt: Optional[QuantFormat],
    mx_output_fmt: Optional[QuantFormat],
    vec_input_fmt: Optional[QuantFormat],
    vec_output_fmt: Optional[QuantFormat],
    functional_model_name: Optional[str],
    num_steps: int,
    t0: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Collect baseline actions (unpatched) then quantized actions (patched).
    Returns (baseline_actions, quant_actions) — both lists of CPU tensors.
    """
    model = policy.model

    # --- Baseline pass -------------------------------------------------------
    print("\n=== Baseline (unpatched) pass ===")
    baseline_actions = _collect_actions(policy, observations, num_steps, "baseline", t0)

    # --- Resolve effective formats -------------------------------------------
    _mx_in  = mx_input_fmt  or QuantFormat.BFLOAT16
    _mx_out = mx_output_fmt or QuantFormat.BFLOAT16
    _vi     = vec_input_fmt  or QuantFormat.BFLOAT16
    _vo     = vec_output_fmt or QuantFormat.BFLOAT16

    fm_factory = None
    if functional_model_name is not None:
        fm_factory = get_functional_model_factory(functional_model_name)

    # --- Patch ---------------------------------------------------------------
    mx_tracker = StatsTracker()
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
        tracker=StatsTracker(),
    )

    # --- Quantized pass ------------------------------------------------------
    print("\n=== Quantized pass ===")
    quant_actions: list[torch.Tensor] = []
    n = len(observations)
    stop = threading.Event()
    _start_heartbeat("quantized", t0, stop, mx_tracker=mx_tracker)
    with vec_ctx:
        for i, obs in enumerate(observations):
            print(f"\n[quantized] obs {i+1}/{n} ({num_steps} diffusion steps)...", flush=True)
            obs_t0 = time.monotonic()
            result = policy.infer(obs)
            act = torch.from_numpy(result["actions"])
            print(f"[quantized] obs {i+1}/{n} done in {time.monotonic()-obs_t0:.1f}s", flush=True)
            quant_actions.append(act)
    stop.set()

    # --- Unpatch -------------------------------------------------------------
    unpatch_model(model)
    unpatch_attn_sdpa(attn_handles)
    unpatch_vector_ops(vec_handles)

    return baseline_actions, quant_actions


# ---------------------------------------------------------------------------
# CSV / summary writers
# ---------------------------------------------------------------------------

_ACTION_RMSE_FIELDS = ["obs_idx", "rmse", "ref_rms", "rel_rmse"]
_OUTPUT_SUMMARY_FIELDS = [
    "timestamp", "label", "overall_rmse", "rel_rmse", "elapsed_seconds", "elapsed_human",
    "mx_input", "mx_output", "vec_input", "vec_output", "functional_model", "active_groups",
    "n_obs", "steps",
]


def _write_action_rmse(
    path: Path,
    baseline_actions: list[torch.Tensor],
    quant_actions: list[torch.Tensor],
) -> tuple[float, float]:
    """Write per-observation action_rmse.csv; return (overall_rmse, ref_rms)."""
    rows = []
    total_se = 0.0
    total_n  = 0
    total_ref_ms = 0.0

    for i, (ref, quant) in enumerate(zip(baseline_actions, quant_actions)):
        ref_f   = ref.float()
        quant_f = quant.float()
        diff    = ref_f - quant_f
        se      = diff.pow(2).sum().item()
        n       = diff.numel()
        ref_ms  = ref_f.pow(2).mean().item()
        rmse_i  = math.sqrt(se / n)
        ref_rms = math.sqrt(max(ref_ms, 0.0))
        rows.append({
            "obs_idx": i,
            "rmse":    rmse_i,
            "ref_rms": ref_rms,
            "rel_rmse": rmse_i / ref_rms if ref_rms > 0 else float("nan"),
        })
        total_se     += se
        total_n      += n
        total_ref_ms += ref_ms

    overall_rmse = math.sqrt(total_se / total_n) if total_n > 0 else float("nan")
    overall_ref_rms = math.sqrt(max(total_ref_ms / len(baseline_actions), 0.0))
    rows.append({
        "obs_idx": "overall",
        "rmse":    overall_rmse,
        "ref_rms": overall_ref_rms,
        "rel_rmse": overall_rmse / overall_ref_rms if overall_ref_rms > 0 else float("nan"),
    })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_ACTION_RMSE_FIELDS)
        w.writeheader()
        w.writerows(rows)

    return overall_rmse, overall_ref_rms


def _append_output_summary(
    results_dir: Path,
    config_record: dict,
    overall_rmse: float,
    overall_ref_rms: float,
) -> None:
    path = results_dir / "all_runs_output_summary.csv"
    write_header = not path.exists()

    mp = config_record["matrix_path"]
    vp = config_record["vector_path"]
    elapsed_s = config_record.get("elapsed_seconds", float("nan"))
    elapsed_td = str(datetime.timedelta(seconds=int(elapsed_s))) if math.isfinite(elapsed_s) else ""
    rel_rmse = overall_rmse / overall_ref_rms if overall_ref_rms > 0 else float("nan")

    row = {
        "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
        "label":            config_record["label"],
        "overall_rmse":     overall_rmse,
        "rel_rmse":         rel_rmse,
        "elapsed_seconds":  elapsed_s,
        "elapsed_human":    elapsed_td,
        "mx_input":         mp.get("mx_input_fmt")  or "passthrough",
        "mx_output":        mp.get("mx_output_fmt") or "passthrough",
        "vec_input":        vp.get("vec_input_fmt")  or "passthrough",
        "vec_output":       vp.get("vec_output_fmt") or "passthrough",
        "functional_model": mp.get("functional_model") or "",
        "active_groups":    "|".join(config_record.get("active_groups", [])),
        "n_obs":            config_record["n_obs"],
        "steps":            config_record["steps"],
    }
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_OUTPUT_SUMMARY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure final action output RMSE of base vs quantized Pi0.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--label", required=True,
                        help="Run label — used as the output folder name under results-dir")

    parser.add_argument("--checkpoint-dir", default="/scratch/chloe.wong/data/pi05_base")
    parser.add_argument("--config", default="pi05_droid_jointpos_polaris")
    parser.add_argument("--gpu",    type=int, default=0)

    parser.add_argument("--n-obs",  type=int, default=4)
    parser.add_argument("--steps",  type=int, default=10)

    mx_group = parser.add_mutually_exclusive_group()
    mx_group.add_argument("--functional-model", metavar="NAME",
                          help=f"Hardware-accurate model for matmuls. "
                               f"Available: {list_functional_models()}")
    mx_group.add_argument("--mx-input-fmt", metavar="FMT",
                          help="Format for matmul inputs. Use 'passthrough' for no-op.")
    parser.add_argument("--mx-output-fmt", metavar="FMT", default=PASSTHROUGH,
                        help="Format for matmul outputs. Use 'passthrough' for no-op.")

    parser.add_argument("--vec-input-fmt",  metavar="FMT", default=PASSTHROUGH)
    parser.add_argument("--vec-output-fmt", metavar="FMT", default=PASSTHROUGH)

    all_group_names = [g.value for g in QuantGroup]
    parser.add_argument("--active-groups", metavar="G1,G2,...",
                        default=",".join(all_group_names),
                        help=f"Comma-separated groups to quantize. Choices: {all_group_names}")

    parser.add_argument("--results-dir",
                        default=str(_REPO / "experiments" / "results"))
    parser.add_argument("--fp8-mode", default="po2", choices=["po2", "abs"])
    parser.add_argument("--norm-stats-dir", metavar="DIR", default=None,
                        help="Directory containing norm_stats.json "
                             "(default: tries <checkpoint-dir>/assets/droid/)")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Path to paligemma_tokenizer.model "
                             "(default: searches ~/Desktop and ~/.cache/openpi/)")
    parser.add_argument("--use-quantile-norm", action="store_true")

    args = parser.parse_args()

    # ── Validate ─────────────────────────────────────────────────────────────
    if args.functional_model is not None and args.mx_output_fmt != PASSTHROUGH:
        parser.error("--mx-output-fmt has no effect with --functional-model")

    active_groups: set[QuantGroup] = set()
    for g in args.active_groups.split(","):
        g = g.strip()
        try:
            active_groups.add(QuantGroup(g))
        except ValueError:
            parser.error(f"Unknown group '{g}'. Choices: {all_group_names}")

    mx_input_fmt   = _fmt_or_passthrough(args.mx_input_fmt or PASSTHROUGH)
    mx_output_fmt  = _fmt_or_passthrough(args.mx_output_fmt)
    vec_input_fmt  = _fmt_or_passthrough(args.vec_input_fmt)
    vec_output_fmt = _fmt_or_passthrough(args.vec_output_fmt)

    set_fp8_mode(args.fp8_mode)

    # ── Device / model ────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    from pi0_inout.serve_quant import load_pi0_pytorch, _get_model_config
    config_ns = _get_model_config(args.config)
    print(f"Loading model: {args.config}  checkpoint: {args.checkpoint_dir}")
    model = load_pi0_pytorch(args.config, args.checkpoint_dir, device)
    model.eval()

    # ── Norm stats ────────────────────────────────────────────────────────────
    norm_stats = None
    if args.norm_stats_dir:
        norm_stats = _load_norm_stats(args.norm_stats_dir)
        print(f"Loaded norm stats from {args.norm_stats_dir}  (keys: {list(norm_stats.keys())})")
    else:
        print("WARNING: no norm stats found — actions will be in normalized space.", file=sys.stderr)

    is_joint_position = "jointpos" in args.config
    policy = Pi0PyTorchPolicy(
        model=model,
        device=device,
        norm_stats=norm_stats,
        use_quantile_norm=args.use_quantile_norm,
        is_joint_position=is_joint_position,
        max_token_len=config_ns.max_token_len,
        tokenizer_path=args.tokenizer_path,
    )

    rng = np.random.default_rng(0)
    observations = [_make_dummy_obs(rng) for _ in range(args.n_obs)]
    print(f"Observations: {args.n_obs}  steps: {args.steps}")

    # ── Config record ─────────────────────────────────────────────────────────
    config_record = {
        "label":          args.label,
        "checkpoint_dir": args.checkpoint_dir,
        "model_config":   args.config,
        "n_obs":          args.n_obs,
        "steps":          args.steps,
        "gpu":            args.gpu,
        "fp8_mode":       args.fp8_mode,
        "active_groups":  [g.value for g in active_groups],
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

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"\nRunning config: {args.label}")
    t0 = time.monotonic()
    baseline_actions, quant_actions = run(
        policy=policy,
        observations=observations,
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

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_dir = Path(args.results_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(config_record, indent=2, default=str))

    overall_rmse, overall_ref_rms = _write_action_rmse(
        out_dir / "action_rmse.csv",
        baseline_actions,
        quant_actions,
    )

    _append_output_summary(
        Path(args.results_dir), config_record, overall_rmse, overall_ref_rms,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    elapsed_td = datetime.timedelta(seconds=int(elapsed_s))
    rel_rmse = overall_rmse / overall_ref_rms if overall_ref_rms > 0 else float("nan")
    print(f"\n{'='*60}")
    print(f"Elapsed:      {elapsed_td} ({elapsed_s:.1f}s)")
    print(f"Results:      {out_dir}")
    print(f"Top-level summary: {Path(args.results_dir) / 'all_runs_output_summary.csv'}")
    print(f"\nOverall action RMSE:     {overall_rmse:.6e}")
    print(f"Relative action RMSE:    {rel_rmse:.6e}  (rmse / rms(baseline))")
    print(f"\nPer-observation breakdown:")
    print(f"  {'obs':>4}  {'rmse':>12}  {'ref_rms':>12}  {'rel_rmse':>12}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*12}")
    for i, (ref, quant) in enumerate(zip(baseline_actions, quant_actions)):
        ref_f   = ref.float()
        quant_f = quant.float()
        diff    = ref_f - quant_f
        rmse_i  = math.sqrt(diff.pow(2).mean().item())
        rms_i   = math.sqrt(ref_f.pow(2).mean().item())
        rel_i   = rmse_i / rms_i if rms_i > 0 else float("nan")
        print(f"  {i:>4}  {rmse_i:>12.4e}  {rms_i:>12.4e}  {rel_i:>12.4e}")


if __name__ == "__main__":
    main()
