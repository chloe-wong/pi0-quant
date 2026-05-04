"""
run_eval_vec_io.py
------------------
Evaluate Pi0 vector-op accuracy and optionally capture per-op I/O tensors.

Requires --vec-functional-model vector when --save-tensors is set (tensor capture
only happens when the VPU functional model is active — no FM means no fm_output
to store).

Vector path:
  --vec-functional-model vector   route all vector ops through VectorRTLFunctions
                                  (default: passthrough — RMSE is all zeros)

Component selection:
  --active-groups vision,language,action_expert,action_head   (default: all)

Op scope selection:
  --ops linear,conv2d,attention   (default: linear)
  Linear ops are patched as passthrough (BF16) — this script focuses on vec RMSE.

Output — written to <results-dir>/<label>/:
  config.json        exact parameters used
  chronological.csv  one row per op call in execution order
  grouped.csv        same rows sorted by (component, layer_name)
  summary.csv        per-component aggregate stats (mx and vec separately)
  worst_layers.csv   top-20 layers by local rel RMSE
  vec_tensors/       (when --save-tensors) one .npz per (layer_tag, op)
                     keys: n_calls, input_0[, input_1], reference_output, fm_output

Usage:
    OPENPI_DIR=/scratch/chloe.wong/openpi \\
    CUDA_VISIBLE_DEVICES=0 \\
    python experiments/run_eval_vec_io.py \\
        --label vec_io_vpu \\
        --vec-functional-model vector \\
        --n-obs 1 --steps 2 \\
        --save-tensors
"""

from __future__ import annotations

import argparse
import contextlib
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

import numpy as np
import sentencepiece
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
    patch_attn_eager, unpatch_attn_eager,
    patch_attn_siglip_eager, unpatch_attn_siglip_eager,
    patch_vector_ops, unpatch_vector_ops,
    VectorIOStore,
    get_functional_model_factory, list_functional_models,
)
from pi0_inout.model_patcher import OpScope, ALL_SCOPES, patch_conv2d, unpatch_conv2d
from pi0_inout.reference_store import ReferenceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _load_real_obs(
    obs_dir: str,
    config_ns: SimpleNamespace,
    checkpoint_dir: str,
    device: torch.device,
    obs_file: str | None = None,
    norm_stats_dir: str | None = None,
) -> list[SimpleNamespace]:
    from pi0_inout.serve_quant import _load_norm_stats

    obs_dir = Path(obs_dir)
    if obs_file is not None:
        npz_path = Path(obs_file)
        if not npz_path.is_absolute():
            npz_path = obs_dir / npz_path
        if not npz_path.exists():
            raise FileNotFoundError(f"obs file not found: {npz_path}")
        npz_files = [npz_path]
    else:
        npz_files = sorted(obs_dir.glob("obs_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No obs_*.npz files found in {obs_dir}")

    norm_stats = None
    _norm_dir = norm_stats_dir if norm_stats_dir is not None else checkpoint_dir
    try:
        norm_stats = _load_norm_stats(_norm_dir)
    except FileNotFoundError:
        print(f"[warn] norm_stats.json not found in {_norm_dir}; skipping normalization")

    tokenizer_path = None
    for candidate in [
        Path.home() / "Desktop" / "paligemma_tokenizer.model",
        Path.home() / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model",
    ]:
        if candidate.exists():
            tokenizer_path = str(candidate)
            break
    if tokenizer_path is None:
        raise FileNotFoundError(
            "Cannot find paligemma_tokenizer.model. "
            "Place it at ~/.cache/openpi/big_vision/paligemma_tokenizer.model"
        )
    tokenizer = sentencepiece.SentencePieceProcessor(model_proto=open(tokenizer_path, "rb").read())

    max_tok = config_ns.max_token_len
    observations = []

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=False)

        def _img(arr):
            t = torch.from_numpy(arr.copy()).to(device)
            t = t.permute(2, 0, 1).unsqueeze(0).float()
            t = t / 255.0 * 2.0 - 1.0
            if t.shape[2:] != (224, 224):
                t = torch.nn.functional.interpolate(
                    t, size=(224, 224), mode="bilinear", align_corners=False
                )
            return t

        base_img  = _img(data["right_image"])
        wrist_img = _img(data["wrist_image"])
        zero_img  = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device)

        joint = data["joint_position"].astype(np.float32).flatten()
        grip  = data["gripper_position"].astype(np.float32).flatten()
        raw_state = np.concatenate([joint, grip])
        if norm_stats is not None and "state" in norm_stats:
            s = norm_stats["state"]
            mean = s.mean[:raw_state.shape[0]].astype(np.float32)
            std  = s.std[:raw_state.shape[0]].astype(np.float32)
            norm_state = (raw_state - mean) / (std + 1e-6)
        else:
            norm_state = raw_state
        state_padded = np.zeros(32, dtype=np.float32)
        state_padded[:norm_state.shape[0]] = norm_state
        state = torch.from_numpy(state_padded).unsqueeze(0).to(device)

        prompt_text = data["prompt"].item()
        if isinstance(prompt_text, bytes):
            prompt_text = prompt_text.decode("utf-8")
        prompt_text = prompt_text.strip().replace("_", " ").replace("\n", " ")
        if prompt_text:
            tokens = tokenizer.encode(prompt_text, add_bos=True) + tokenizer.encode("\n")
        else:
            tokens = []
        tok_len = min(len(tokens), max_tok)
        tokens = tokens[:tok_len]
        pad_len = max_tok - tok_len
        tokens_padded = tokens + [0] * pad_len
        mask_list = [True] * tok_len + [False] * pad_len

        observations.append(SimpleNamespace(
            _source=str(npz_path),
            images={
                "base_0_rgb":        base_img,
                "left_wrist_0_rgb":  wrist_img,
                "right_wrist_0_rgb": zero_img,
            },
            image_masks={
                "base_0_rgb":        torch.ones(1,  dtype=torch.bool, device=device),
                "left_wrist_0_rgb":  torch.ones(1,  dtype=torch.bool, device=device),
                "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
            },
            state=state,
            tokenized_prompt=      torch.tensor([tokens_padded], dtype=torch.int64, device=device),
            tokenized_prompt_mask= torch.tensor([mask_list],      dtype=torch.bool,  device=device),
            token_ar_mask=         torch.zeros(1, max_tok, dtype=torch.bool, device=device),
            token_loss_mask=       torch.zeros(1, max_tok, dtype=torch.bool, device=device),
        ))

    print(f"Loaded {len(observations)} real observations from {obs_dir}")
    return observations


def _rel_rmse(rmse: float, ref_rms: float) -> float:
    if ref_rms > 0 and math.isfinite(rmse):
        return rmse / ref_rms
    return float("nan")


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

_CHRON_FIELDS = [
    "seq", "tag", "layer_name", "component",
    "rmse", "ref_rms", "rel_rmse",
    "cumulative_rmse", "cumulative_rel_rmse",
]
_SUMMARY_FIELDS = [
    "tag", "component", "n_layers",
    "mean_rmse", "std_rmse", "max_rmse", "min_rmse",
    "mean_rel_rmse", "std_rel_rmse", "max_rel_rmse", "max_rel_rmse_layer",
    "total_calls",
    "mean_cumulative_rmse", "mean_cumulative_rel_rmse",
]
_WORST_LAYERS_FIELDS = ["rank", "tag", "component", "layer_name", "rel_rmse", "rmse", "n_calls"]


def _calls_to_rows(calls: list[dict], tag: str) -> list[dict]:
    rows = []
    for rec in calls:
        cum_rmse = rec.get("cumulative_rmse", float("nan"))
        cum_ref  = rec.get("cumulative_ref_rms", float("nan"))
        rows.append({
            "seq":                 rec["seq"],
            "tag":                 tag,
            "layer_name":          rec["name"],
            "component":           rec["component"],
            "rmse":                rec["rmse"],
            "ref_rms":             rec["ref_rms"],
            "rel_rmse":            _rel_rmse(rec["rmse"], rec["ref_rms"]),
            "cumulative_rmse":     cum_rmse,
            "cumulative_rel_rmse": _rel_rmse(cum_rmse, cum_ref),
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


def _write_worst_layers(path: Path, mx_tracker: StatsTracker, vec_tracker: StatsTracker, top_n: int = 20) -> None:
    rows = []
    for tag, tracker in [("mx", mx_tracker), ("vec", vec_tracker)]:
        for layer in tracker.layer_rows():
            rel = layer.get("rel_rmse", float("nan"))
            if math.isfinite(rel):
                rows.append({"tag": tag, **layer})
    rows.sort(key=lambda r: r["rel_rmse"], reverse=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_WORST_LAYERS_FIELDS)
        w.writeheader()
        for rank, r in enumerate(rows[:top_n], 1):
            w.writerow({
                "rank":       rank,
                "tag":        r["tag"],
                "component":  r["component"],
                "layer_name": r["layer"],
                "rel_rmse":   r["rel_rmse"],
                "rmse":       r["rmse"],
                "n_calls":    r["n_calls"],
            })


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
     "vec_functional_model", "active_groups", "ops"]
    + [f"vec_{c}_mean_rmse"         for c in _COMPONENTS]
    + [f"vec_{c}_mean_rel_rmse"     for c in _COMPONENTS]
    + [f"vec_{c}_std_rel_rmse"      for c in _COMPONENTS]
    + [f"vec_{c}_max_rel_rmse"      for c in _COMPONENTS]
    + [f"vec_{c}_max_rel_rmse_layer" for c in _COMPONENTS]
)


def _append_top_level_summary(
    results_dir: Path,
    config_record: dict,
    vec_tracker: StatsTracker,
) -> None:
    path = results_dir / "all_runs_summary.csv"
    write_header = not path.exists()

    vec_comp = {r["component"]: r for r in vec_tracker.component_rows()}
    elapsed_s = config_record.get("elapsed_seconds", float("nan"))
    elapsed_td = str(datetime.timedelta(seconds=int(elapsed_s))) if math.isfinite(elapsed_s) else ""

    row: dict = {
        "timestamp":            datetime.datetime.now().isoformat(timespec="seconds"),
        "label":                config_record["label"],
        "elapsed_seconds":      elapsed_s,
        "elapsed_human":        elapsed_td,
        "vec_functional_model": config_record.get("vec_functional_model") or "passthrough",
        "active_groups":        "|".join(config_record.get("active_groups", [])),
        "ops":                  "|".join(config_record.get("ops", [])),
    }
    for c in _COMPONENTS:
        r = vec_comp.get(c, {})
        row[f"vec_{c}_mean_rmse"]          = r.get("mean_rmse",          float("nan"))
        row[f"vec_{c}_mean_rel_rmse"]      = r.get("mean_rel_rmse",      float("nan"))
        row[f"vec_{c}_std_rel_rmse"]       = r.get("std_rel_rmse",       float("nan"))
        row[f"vec_{c}_max_rel_rmse"]       = r.get("max_rel_rmse",       float("nan"))
        row[f"vec_{c}_max_rel_rmse_layer"] = r.get("max_rel_rmse_layer", "")

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
    hook_fires: Optional[dict] = None,
) -> None:
    total_calls = mx_tracker._seq + vec_tracker._seq
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_s)))
    print(f"\n  elapsed={elapsed_str}  layer_calls={total_calls}", end="")
    if hook_fires:
        fires_str = "  hooks: " + "  ".join(
            f"{k}={v}" for k, v in sorted(hook_fires.items()) if v > 0
        )
        print(fires_str, end="")
    print()
    print(f"  {'component':<14} {'vec_rel_rmse':>13}  {'vec_rmse':>12}")
    print(f"  {'-'*14} {'-'*13}  {'-'*12}")
    vec_by_comp = {r["component"]: r for r in vec_tracker.component_rows()}
    for c in ["vision", "language", "action_expert", "action_head"]:
        vec = vec_by_comp.get(c, {})
        print(
            f"  {c:<14} {vec.get('mean_rel_rmse', float('nan')):>13.4e}"
            f"  {vec.get('mean_rmse', float('nan')):>12.4e}"
        )


def _start_heartbeat(
    mx_tracker: StatsTracker,
    vec_tracker: StatsTracker,
    t0: float,
    stop_event: threading.Event,
    hook_fires: Optional[dict] = None,
    interval_s: int = 30,
) -> threading.Thread:
    def _loop():
        while not stop_event.wait(timeout=interval_s):
            elapsed = time.monotonic() - t0
            calls   = mx_tracker._seq + vec_tracker._seq
            fires_str = ""
            if hook_fires:
                fires_str = "  hooks: " + "  ".join(
                    f"{k}={v}" for k, v in sorted(hook_fires.items()) if v > 0
                )
            print(
                f"  [heartbeat] elapsed={datetime.timedelta(seconds=int(elapsed))}  "
                f"mx={mx_tracker._seq}  vec={vec_tracker._seq}{fires_str}",
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
    op_scopes: set[OpScope],
    vec_functional_model_name: Optional[str],
    num_steps: int,
    t0: float,
    trace: bool = False,
    vector_io_store: Optional[VectorIOStore] = None,
    propagate_noise: bool = False,
    out_dir: Optional[Path] = None,
    functional_model_name: Optional[str] = None,
) -> tuple[StatsTracker, StatsTracker]:
    """Patch model, run observations, unpatch. Returns (mx_tracker, vec_tracker)."""
    mx_tracker  = StatsTracker()
    vec_tracker = StatsTracker()

    vec_fm = None
    if vec_functional_model_name is not None:
        from funct_models_vector.vector_rtl_forward import VectorRTLFunctions
        vec_fm = VectorRTLFunctions(num_lanes=16)

    fm_factory = None
    if functional_model_name is not None:
        fm_factory = get_functional_model_factory(functional_model_name)

    # ── Capture reference layer outputs for cumulative RMSE ──────────────────
    ref_store = ReferenceStore()
    layer_names = {
        name for name, m in model.named_modules()
        if type(m) is nn.Linear or type(m) is nn.Conv2d
    }
    ref_hooks = ref_store.register_hooks(model, layer_names)

    # Capture clean vector-op args during the reference pass for error-free RMSE replay.
    # When propagate_noise is set, skip Pass 1 entirely so FM outputs propagate forward
    # in Pass 2 as inputs to subsequent ops, letting quantization noise accumulate end-to-end.
    clean_input_store = ReferenceStore() if (vec_fm is not None and not propagate_noise) else None
    if clean_input_store is not None:
        cap_handles, cap_ctx, _ = patch_vector_ops(
            model,
            active_groups=active_groups,
            functional_model=None,
            capture_mode=True,
            clean_input_store=clean_input_store,
        )

    if OpScope.ATTENTION in op_scopes:
        import torch.nn.functional as _F_ref
        from transformers.models.gemma import modeling_gemma as _mg_ref

        _ref_orig_eager = _mg_ref.eager_attention_forward
        _ref_orig_sdpa  = _F_ref.scaled_dot_product_attention

        def _ref_capture_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            out, w = _ref_orig_eager(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            ref_store.capture("eager_attn", out)
            return out, w

        def _ref_capture_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            out = _ref_orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)
            ref_store.capture("sdpa", out)
            return out

        _mg_ref.eager_attention_forward    = _ref_capture_eager
        _F_ref.scaled_dot_product_attention = _ref_capture_sdpa

    _cap_ctx = cap_ctx if clean_input_store is not None else contextlib.nullcontext()
    with torch.no_grad(), _cap_ctx:
        for i, obs in enumerate(observations):
            torch.manual_seed(i)
            ref_store.reset_counters()
            model.sample_actions(str(device), obs, num_steps=num_steps)
    for h in ref_hooks:
        h.remove()
    if clean_input_store is not None:
        unpatch_vector_ops(cap_handles)
        clean_input_store.reset_counters()

    if OpScope.ATTENTION in op_scopes:
        _mg_ref.eager_attention_forward    = _ref_orig_eager
        _F_ref.scaled_dot_product_attention = _ref_orig_sdpa

    print(f"[reference_store] Captured {len(ref_store)} reference layer outputs.")

    # ── Patch model (linear/conv2d/attention; FM applies if --functional-model is set) ──
    patch_model(
        model,
        mx_input_fmt=QuantFormat.BFLOAT16,
        mx_output_fmt=QuantFormat.BFLOAT16,
        tracker=mx_tracker,
        active_groups=active_groups,
        functional_model_factory=fm_factory,
        op_scopes=op_scopes,
        reference_store=ref_store,
        trace=trace,
    )
    from pi0_inout.quant_linear import QuantLinear as _QL
    _n_mx = sum(1 for _, m in model.named_modules() if isinstance(m, _QL))
    _estimated_total = _n_mx * (1 + num_steps) * len(observations) * 2
    for _, m in model.named_modules():
        if isinstance(m, _QL):
            m.estimated_total = _estimated_total

    if OpScope.CONV2D in op_scopes:
        patch_conv2d(
            model,
            mx_input_fmt=QuantFormat.BFLOAT16,
            mx_output_fmt=QuantFormat.BFLOAT16,
            tracker=mx_tracker,
            active_groups=active_groups,
            functional_model_factory=fm_factory,
            reference_store=ref_store,
        )
    if OpScope.ATTENTION in op_scopes:
        attn_handles = patch_attn_sdpa(
            model,
            active_groups=active_groups,
            tracker=mx_tracker,
            functional_model_factory=fm_factory,
            reference_store=ref_store,
        )
        patch_attn_eager(
            model,
            active_groups=active_groups,
            tracker=mx_tracker,
            functional_model_factory=fm_factory,
            reference_store=ref_store,
        )
        patch_attn_siglip_eager(
            model,
            active_groups=active_groups,
            tracker=mx_tracker,
            functional_model_factory=fm_factory,
            reference_store=ref_store,
        )
    else:
        attn_handles = []

    vec_handles, vec_ctx, hook_fires = patch_vector_ops(
        model,
        active_groups=active_groups,
        tracker=vec_tracker,
        functional_model=vec_fm,
        verbose=trace,
        estimated_total=_estimated_total,
        io_store=vector_io_store,
        clean_input_store=clean_input_store,
    )

    n_obs = len(observations)
    stop_heartbeat = threading.Event()
    _start_heartbeat(mx_tracker, vec_tracker, t0, stop_heartbeat, hook_fires=hook_fires)

    with torch.no_grad(), vec_ctx:
        for i, obs in enumerate(observations):
            src = getattr(obs, "_source", "dummy")
            print(f"\n[obs {i + 1}/{n_obs}] source={src}  running ({num_steps} diffusion steps)...", flush=True)
            obs_t0 = time.monotonic()
            torch.manual_seed(i)
            ref_store.reset_counters()
            actions = model.sample_actions(str(device), obs, num_steps=num_steps)
            if out_dir is not None:
                np.save(str(out_dir / f"actions_{i:04d}.npy"),
                        actions.detach().float().cpu().numpy())
            obs_elapsed = time.monotonic() - obs_t0
            print(f"[obs {i + 1}/{n_obs}] done in {obs_elapsed:.1f}s", flush=True)
            _print_intermediate(
                f"obs {i + 1}/{n_obs}",
                mx_tracker, vec_tracker,
                elapsed_s=time.monotonic() - t0,
                hook_fires=hook_fires,
            )

    stop_heartbeat.set()
    unpatch_model(model)
    if OpScope.CONV2D in op_scopes:
        unpatch_conv2d(model)
    unpatch_attn_sdpa(attn_handles)
    unpatch_attn_eager()
    unpatch_attn_siglip_eager()
    unpatch_vector_ops(vec_handles)

    if vector_io_store is not None:
        vector_io_store.save()

    return mx_tracker, vec_tracker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Pi0 vector-op accuracy and optionally capture I/O tensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--label", required=True,
                        help="Run label — used as the output folder name under results-dir")

    # Model loading
    parser.add_argument("--checkpoint-dir", default="/scratch/chloe.wong/data/pi05_base")
    parser.add_argument("--config", default="pi05_droid_jointpos_polaris")
    parser.add_argument("--gpu",    type=int, default=0)

    # Eval settings
    parser.add_argument("--n-obs",  type=int, default=4,
                        help="Number of random observations (ignored when --obs-dir is set)")
    parser.add_argument("--obs-dir", metavar="DIR", default=None,
                        help="Directory of obs_*.npz files. Uses real observations when set.")
    parser.add_argument("--obs-file", metavar="FILE", default=None,
                        help="Single obs_*.npz file. Requires --obs-dir. Overrides globbing.")
    parser.add_argument("--norm-stats-dir", metavar="DIR", default=None,
                        help="Directory containing norm_stats.json. Defaults to --checkpoint-dir.")
    parser.add_argument("--steps",  type=int, default=10,
                        help="Diffusion steps per sample_actions call")

    # Vector path
    parser.add_argument("--vec-functional-model", metavar="NAME",
                        choices=["vector"],
                        default=None,
                        help="Route all vector ops through VectorRTLFunctions. "
                             "Required when --save-tensors is set.")

    # Matrix path
    parser.add_argument("--functional-model", metavar="NAME",
                        default=None,
                        help=f"Hardware-accurate model for matmuls (linear/conv2d/attention). "
                             f"Available: {list_functional_models()}. Default: BF16 passthrough.")

    # Op scope selection
    all_scope_names = [s.value for s in ALL_SCOPES]
    parser.add_argument("--ops", metavar="OP1,OP2,...",
                        default="linear",
                        help=f"Comma-separated op types. Choices: {all_scope_names}  (default: linear)")

    # Component selection
    all_group_names = [g.value for g in QuantGroup]
    parser.add_argument("--active-groups", metavar="G1,G2,...",
                        default=",".join(all_group_names),
                        help=f"Comma-separated groups. Choices: {all_group_names}")

    # Output
    parser.add_argument("--results-dir",
                        default=str(_REPO / "experiments" / "results"),
                        help="Root directory for results (default: <repo>/experiments/results)")
    parser.add_argument("--save-tensors", action="store_true",
                        help="Save per-op vector I/O tensors to "
                             "<results-dir>/<label>/vec_tensors/ (one .npz per (layer, op)). "
                             "Requires --vec-functional-model.")
    parser.add_argument("--propagate-noise", action="store_true",
                        help="Propagate FM outputs forward as inputs to subsequent ops, so "
                             "quantization noise accumulates end-to-end. The final action chunk "
                             "(with cumulative noise) is captured per obs to actions_<i>.npy — "
                             "use this for measuring noise at the final layer. "
                             "Default off: two-pass clean-input mode measures each layer in isolation. "
                             "Incompatible with --save-tensors.")
    parser.add_argument("--trace", action="store_true",
                        help="Print one line per op as it fires with shape and RMSE.")

    args = parser.parse_args()

    if args.save_tensors and args.vec_functional_model is None:
        parser.error("--save-tensors requires --vec-functional-model (no FM means no fm_output to store)")
    if args.propagate_noise and args.save_tensors:
        parser.error("--propagate-noise is incompatible with --save-tensors "
                     "(per-op reference outputs are not meaningful when noise propagates)")

    op_scopes: set[OpScope] = set()
    for s in args.ops.split(","):
        s = s.strip()
        try:
            op_scopes.add(OpScope(s))
        except ValueError:
            parser.error(f"Unknown op scope '{s}'. Choices: {all_scope_names}")

    active_groups: set[QuantGroup] = set()
    for g in args.active_groups.split(","):
        g = g.strip()
        try:
            active_groups.add(QuantGroup(g))
        except ValueError:
            parser.error(f"Unknown group '{g}'. Choices: {all_group_names}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    from pi0_inout.serve_quant import load_pi0_pytorch, _get_model_config
    config_ns = _get_model_config(args.config)
    print(f"Loading model: {args.config}  checkpoint: {args.checkpoint_dir}")
    model = load_pi0_pytorch(args.config, args.checkpoint_dir, device)
    model.eval()
    if "sample_actions" in model.__dict__:
        del model.sample_actions
        print("[run_eval_vec_io] Removed torch.compile wrapper from sample_actions (eager mode).")

    torch.manual_seed(0)
    if args.obs_dir is not None:
        observations = _load_real_obs(args.obs_dir, config_ns, args.checkpoint_dir, device,
                                      obs_file=args.obs_file,
                                      norm_stats_dir=args.norm_stats_dir)
        src = args.obs_file if args.obs_file else f"all obs_*.npz in {args.obs_dir}"
        print(f"[obs] real observations from: {src}")
    else:
        observations = [_make_dummy_obs(config_ns, device) for _ in range(args.n_obs)]
        print(f"[obs] using {len(observations)} dummy (random) observations")
    print(f"Observations: {len(observations)}  steps: {args.steps}")

    config_record = {
        "label":                args.label,
        "checkpoint_dir":       args.checkpoint_dir,
        "model_config":         args.config,
        "n_obs":                len(observations),
        "obs_dir":              args.obs_dir,
        "steps":                args.steps,
        "gpu":                  args.gpu,
        "active_groups":        [g.value for g in active_groups],
        "ops":                  [s.value for s in op_scopes],
        "vec_functional_model": args.vec_functional_model or "passthrough",
    }

    out_dir = Path(args.results_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    vector_io_store = VectorIOStore(out_dir / "vec_tensors") if args.save_tensors else None

    print(f"\nRunning config: {args.label}")
    t0 = time.monotonic()
    mx_tracker, vec_tracker = run(
        model=model,
        observations=observations,
        device=device,
        active_groups=active_groups,
        op_scopes=op_scopes,
        vec_functional_model_name=args.vec_functional_model,
        num_steps=args.steps,
        t0=t0,
        trace=args.trace,
        vector_io_store=vector_io_store,
        propagate_noise=args.propagate_noise,
        out_dir=out_dir,
        functional_model_name=args.functional_model,
    )
    elapsed_s = time.monotonic() - t0
    config_record["elapsed_seconds"] = round(elapsed_s, 2)

    (out_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, default=str)
    )
    _write_chronological(out_dir / "chronological.csv", mx_tracker.calls, vec_tracker.calls)
    _write_grouped(out_dir / "grouped.csv", mx_tracker.calls, vec_tracker.calls)
    _write_summary(out_dir / "summary.csv", mx_tracker, vec_tracker)
    _write_worst_layers(out_dir / "worst_layers.csv", mx_tracker, vec_tracker, top_n=20)
    _append_top_level_summary(Path(args.results_dir), config_record, vec_tracker)

    elapsed_td = datetime.timedelta(seconds=int(elapsed_s))
    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed_td} ({elapsed_s:.1f}s)")
    print(f"Results: {out_dir}")
    if vector_io_store is not None:
        print(f"Tensors: {out_dir / 'vec_tensors'}")
    print(f"Top-level summary: {Path(args.results_dir) / 'all_runs_summary.csv'}")
    print("\n-- Vector path --")
    vec_tracker.summary().print()


if __name__ == "__main__":
    main()
