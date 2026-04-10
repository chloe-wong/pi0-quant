"""
run_eval.py
-----------
Flexible evaluation runner: patches the Pi0 model with any combination of
quantization settings and logs per-layer RMSE to a results folder.

Op scope selection — --ops OP1,OP2,...  (default: linear)
  linear      nn.Linear weight-activation matmuls: all Q/K/V/O projections,
              MLP gate/up/down_proj, and action-head projections.
  conv2d      nn.Conv2d patch embedding in the SigLIP vision encoder (one layer).
  attention   Attention score matmuls Q@K^T and attn_weights@V:
                - SigLIP ViT: via F.scaled_dot_product_attention (patch_attn_sdpa)
                - Gemma language model: via eager_attention_forward (patch_attn_eager)
                - Gemma action expert: via eager_attention_forward (patch_attn_eager)
                - Co-attention (language + expert joint): same eager path
              Softmax runs in BF16; attn_weights are always quantized to FP8
              E4M3 before the AV matmul (hardware faithful).

  Together, --ops linear,conv2d,attention covers all active matmuls in Pi0
  inference. The only excluded ops are the RoPE frequency precomputation
  (no learned weights, negligible FLOPs) and lm_head (never called in Pi0).

Matrix path — choose one:
  --mx-input-fmt / --mx-output-fmt   software format-flag quantization
  --functional-model NAME             hardware-accurate simulation
                                      available: ipt, ipt_numba, ipt_c, systolic_c
  (mutually exclusive; default is passthrough = bfloat16/bfloat16)

Vector path (independent of matrix path):
  --vec-input-fmt / --vec-output-fmt  (default: passthrough = bfloat16/bfloat16)

Component selection:
  --active-groups vision,language,action_expert,action_head   (default: all)

Output — written to <results-dir>/<label>/:
  config.json        exact parameters used
  chronological.csv  one row per op call in execution order (local + cumulative RMSE)
  grouped.csv        same rows sorted by (component, layer_name)
  summary.csv        per-component aggregate stats (mx and vec separately)
  worst_layers.csv   top-20 layers by local rel RMSE across all components

Usage:
    # IPT numba functional model, all op scopes:
    OPENPI_DIR=/scratch/chloe.wong/openpi \\
    CUDA_VISIBLE_DEVICES=0 \\
    /scratch/chloe.wong/envs/pi0/bin/python experiments/run_eval.py \\
        --label ipt_numba_all \\
        --functional-model ipt_numba \\
        --ops linear,conv2d,attention \\
        --n-obs 4 --steps 10 \\
        --results-dir experiments/results/my_run

    # Software FP8 format flags, linear only:
    OPENPI_DIR=/scratch/chloe.wong/openpi \\
    /scratch/chloe.wong/envs/pi0/bin/python experiments/run_eval.py \\
        --label fp8_linear \\
        --mx-input-fmt float8_e4m3 --mx-output-fmt bfloat16 \\
        --ops linear
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
    patch_vector_ops, unpatch_vector_ops,
    get_functional_model_factory, list_functional_models,
    set_fp8_mode,
)
from pi0_inout.model_patcher import OpScope, ALL_SCOPES, patch_conv2d, unpatch_conv2d
from pi0_inout.reference_store import ReferenceStore
from pi0_inout.matmul_io_store import MatmulIOStore, _to_bf16_numpy

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

    # Load norm stats
    norm_stats = None
    _norm_dir = norm_stats_dir if norm_stats_dir is not None else checkpoint_dir
    try:
        norm_stats = _load_norm_stats(_norm_dir)
    except FileNotFoundError:
        print(f"[warn] norm_stats.json not found in {_norm_dir}; skipping normalization")

    # Load tokenizer (same search path as Pi0PyTorchPolicy)
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

        # Images: uint8 HWC (H,W,3) → float32 CHW (1,3,224,224) in [-1, 1]
        # Pre-resize to 224x224 here (matching the sim-evals WebSocket path which resizes
        # before sending) to avoid a batch-squeeze bug in resize_with_pad_torch.
        def _img(arr):
            t = torch.from_numpy(arr.copy()).to(device)
            t = t.permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
            t = t / 255.0 * 2.0 - 1.0
            if t.shape[2:] != (224, 224):
                t = torch.nn.functional.interpolate(
                    t, size=(224, 224), mode="bilinear", align_corners=False
                )
            return t

        base_img  = _img(data["right_image"])
        wrist_img = _img(data["wrist_image"])
        zero_img  = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device)

        # State: concat joint + gripper, normalize, pad to 32
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

        # Prompt: tokenize
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


def _write_worst_layers(path: Path, mx_tracker: StatsTracker, vec_tracker: StatsTracker, top_n: int = 10) -> None:
    """Write top-N worst layers by rel_rmse across all components and tags."""
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
     "mx_input", "mx_output", "vec_input", "vec_output",
     "functional_model", "active_groups", "ops"]
    + [f"mx_{c}_mean_rmse"                for c in _COMPONENTS]
    + [f"mx_{c}_mean_rel_rmse"            for c in _COMPONENTS]
    + [f"mx_{c}_std_rel_rmse"             for c in _COMPONENTS]
    + [f"mx_{c}_max_rel_rmse"             for c in _COMPONENTS]
    + [f"mx_{c}_max_rel_rmse_layer"       for c in _COMPONENTS]
    + [f"mx_{c}_mean_cumulative_rel_rmse" for c in _COMPONENTS]
    + [f"vec_{c}_mean_rmse"               for c in _COMPONENTS]
    + [f"vec_{c}_mean_rel_rmse"           for c in _COMPONENTS]
    + [f"vec_{c}_std_rel_rmse"            for c in _COMPONENTS]
    + [f"vec_{c}_max_rel_rmse"            for c in _COMPONENTS]
    + [f"vec_{c}_max_rel_rmse_layer"      for c in _COMPONENTS]
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
        "ops":             "|".join(config_record.get("ops", [])),
    }
    for c in _COMPONENTS:
        mx_row  = comp_lookup["mx"].get(c,  {})
        vec_row = comp_lookup["vec"].get(c, {})
        row[f"mx_{c}_mean_rmse"]                = mx_row.get("mean_rmse",                float("nan"))
        row[f"mx_{c}_mean_rel_rmse"]            = mx_row.get("mean_rel_rmse",            float("nan"))
        row[f"mx_{c}_std_rel_rmse"]             = mx_row.get("std_rel_rmse",             float("nan"))
        row[f"mx_{c}_max_rel_rmse"]             = mx_row.get("max_rel_rmse",             float("nan"))
        row[f"mx_{c}_max_rel_rmse_layer"]       = mx_row.get("max_rel_rmse_layer",       "")
        row[f"mx_{c}_mean_cumulative_rel_rmse"] = mx_row.get("mean_cumulative_rel_rmse", float("nan"))
        row[f"vec_{c}_mean_rmse"]               = vec_row.get("mean_rmse",               float("nan"))
        row[f"vec_{c}_mean_rel_rmse"]           = vec_row.get("mean_rel_rmse",           float("nan"))
        row[f"vec_{c}_std_rel_rmse"]            = vec_row.get("std_rel_rmse",            float("nan"))
        row[f"vec_{c}_max_rel_rmse"]            = vec_row.get("max_rel_rmse",            float("nan"))
        row[f"vec_{c}_max_rel_rmse_layer"]      = vec_row.get("max_rel_rmse_layer",      "")

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
    op_scopes: set[OpScope],
    mx_input_fmt: Optional[QuantFormat],
    mx_output_fmt: Optional[QuantFormat],
    vec_input_fmt: Optional[QuantFormat],
    vec_output_fmt: Optional[QuantFormat],
    functional_model_name: Optional[str],
    num_steps: int,
    t0: float,
    matmul_io_store: Optional[MatmulIOStore] = None,
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

    # ── Capture reference (unpatched) layer outputs for cumulative RMSE ──────
    ref_store = ReferenceStore()
    layer_names = {
        name for name, m in model.named_modules()
        if type(m) is nn.Linear or type(m) is nn.Conv2d
    }
    ref_hooks = ref_store.register_hooks(model, layer_names)

    # SANITY CHECKING FOR INT16->BF16 REINTERPRETATION
    # ── Real BF16 I/O capture for target layer (independent of MatmulIOStore) ──
    _TARGET_LAYER = (
        "paligemma_with_expert.paligemma.model.vision_tower"
        ".vision_model.encoder.layers.0.mlp.fc1"
    )
    _real_unpatched_x: list = []
    _real_unpatched_y: list = []
    _real_patched_x:   list = []
    _real_patched_y:   list = []
    _real_io_handles:  list = []

    if matmul_io_store is not None:
        _tgt = dict(model.named_modules()).get(_TARGET_LAYER)
        if _tgt is not None:
            def _unpatched_hook(module, inp, out):
                _real_unpatched_x.append(_to_bf16_numpy(inp[0]))
                _real_unpatched_y.append(_to_bf16_numpy(out))
            _real_io_handles.append(_tgt.register_forward_hook(_unpatched_hook))

    # ── Capture unpatched matmul I/O tensors and clean inputs during reference pass ──
    ref_input_store = ReferenceStore()
    io_hooks: list = []
    if matmul_io_store is not None:
        def _make_io_hook(store: MatmulIOStore, ri_store: ReferenceStore, name: str):
            def _hook(module, inp, out):
                store.record_unpatched(
                    name=name,
                    x=inp[0],
                    w=module.weight,
                    b=module.bias,
                    y=out,
                )
                ri_store.capture(name, inp[0])
            return _hook

        for name, module in model.named_modules():
            if type(module) is nn.Linear and name in layer_names:
                io_hooks.append(
                    module.register_forward_hook(_make_io_hook(matmul_io_store, ref_input_store, name))
                )

    # Also capture eager_attention_forward and SDPA outputs when attention is active.
    _ref_attn_handles = []
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

    with torch.no_grad():
        for i, obs in enumerate(observations):
            torch.manual_seed(i)
            ref_store.reset_counters()
            model.sample_actions(str(device), obs, num_steps=num_steps)
    for h in ref_hooks:
        h.remove()
    for h in io_hooks:
        h.remove()

    if OpScope.ATTENTION in op_scopes:
        _mg_ref.eager_attention_forward    = _ref_orig_eager
        _F_ref.scaled_dot_product_attention = _ref_orig_sdpa

    print(f"[reference_store] Captured {len(ref_store)} reference layer outputs.")

    # Remove unpatched hook before patch_model replaces the module
    for h in _real_io_handles:
        h.remove()
    _real_io_handles.clear()

    patch_model(
        model,
        mx_input_fmt=_mx_in,
        mx_output_fmt=_mx_out,
        tracker=mx_tracker,
        active_groups=active_groups,
        functional_model_factory=fm_factory,
        op_scopes=op_scopes,
        reference_store=ref_store,
        matmul_io_store=matmul_io_store,
        ref_input_store=ref_input_store if matmul_io_store is not None else None,
    )
    if OpScope.CONV2D in op_scopes:
        patch_conv2d(
            model,
            mx_input_fmt=_mx_in,
            mx_output_fmt=_mx_out,
            tracker=mx_tracker,
            active_groups=active_groups,
            functional_model_factory=fm_factory,
            reference_store=ref_store,
        )
    if OpScope.ATTENTION in op_scopes:
        attn_handles = patch_attn_sdpa(
            model,
            active_groups=active_groups,
            mx_input_fmt=_mx_in,
            mx_output_fmt=_mx_out,
            tracker=mx_tracker,
            functional_model_factory=fm_factory,
            reference_store=ref_store,
        )
        patch_attn_eager(
            model,
            active_groups=active_groups,
            mx_input_fmt=_mx_in,
            mx_output_fmt=_mx_out,
            tracker=mx_tracker,
            functional_model_factory=fm_factory,
            reference_store=ref_store,
        )
    else:
        attn_handles = []
    vec_handles, vec_ctx = patch_vector_ops(
        model,
        active_groups=active_groups,
        vec_input_fmt=_vi,
        vec_output_fmt=_vo,
        tracker=vec_tracker,
    )

    # Register patched hook on the (now QuantLinear) target layer
    if matmul_io_store is not None:
        _tgt = dict(model.named_modules()).get(_TARGET_LAYER)
        if _tgt is not None:
            def _patched_hook(module, inp, out):
                _real_patched_x.append(_to_bf16_numpy(inp[0]))
                _real_patched_y.append(_to_bf16_numpy(out))
            _real_io_handles.append(_tgt.register_forward_hook(_patched_hook))

    n_obs = len(observations)
    stop_heartbeat = threading.Event()
    _start_heartbeat(mx_tracker, vec_tracker, t0, stop_heartbeat)

    with torch.no_grad(), vec_ctx:
        for i, obs in enumerate(observations):
            src = getattr(obs, "_source", "dummy")
            print(f"\n[obs {i + 1}/{n_obs}] source={src}  running ({num_steps} diffusion steps)...", flush=True)
            obs_t0 = time.monotonic()
            torch.manual_seed(i)
            ref_store.reset_counters()
            ref_input_store.reset_counters()
            model.sample_actions(str(device), obs, num_steps=num_steps)
            obs_elapsed = time.monotonic() - obs_t0
            print(f"[obs {i + 1}/{n_obs}] done in {obs_elapsed:.1f}s", flush=True)
            _print_intermediate(
                f"obs {i + 1}/{n_obs}",
                mx_tracker, vec_tracker,
                elapsed_s=time.monotonic() - t0,
            )

    stop_heartbeat.set()
    for h in _real_io_handles:
        h.remove()
    unpatch_model(model)
    if OpScope.CONV2D in op_scopes:
        unpatch_conv2d(model)
    unpatch_attn_sdpa(attn_handles)
    unpatch_attn_eager()
    unpatch_vector_ops(vec_handles)

    if matmul_io_store is not None:
        matmul_io_store.save()

    if matmul_io_store is not None and (_real_unpatched_x or _real_patched_x):
        arrays: dict = {"n_calls": np.array(
            max(len(_real_unpatched_x), len(_real_patched_x)), dtype=np.int64
        )}
        for key, frames in [
            ("unpatched_x", _real_unpatched_x),
            ("unpatched_y", _real_unpatched_y),
            ("patched_x",   _real_patched_x),
            ("patched_y",   _real_patched_y),
        ]:
            if not frames:
                continue
            try:
                arrays[key] = np.stack(frames, axis=0)
            except ValueError:
                for i, arr in enumerate(frames):
                    arrays[f"{key}_call{i}"] = arr
        fname = _TARGET_LAYER.replace(".", "__") + "_real_io.npz"
        np.savez(matmul_io_store.save_dir / fname, **arrays)
        print(f"[real IO] Saved {fname}")

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
                        help="Number of random observations to run (ignored when --obs-dir is set)")
    parser.add_argument("--obs-dir", metavar="DIR", default=None,
                        help="Directory of obs_*.npz files saved from sim-evals. "
                             "When set, uses real observations instead of random dummies.")
    parser.add_argument("--obs-file", metavar="FILE", default=None,
                        help="Single obs_*.npz file to load (bare filename or full path). "
                             "Requires --obs-dir for norm_stats lookup. "
                             "When set, overrides globbing all files in --obs-dir.")
    parser.add_argument("--norm-stats-dir", metavar="DIR", default=None,
                        help="Directory containing norm_stats.json. "
                             "Defaults to --checkpoint-dir if not set.")
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

    # Op scope selection
    all_scope_names = [s.value for s in ALL_SCOPES]
    parser.add_argument("--ops", metavar="OP1,OP2,...",
                        default="linear",
                        help=f"Comma-separated op types to apply quantization to. "
                             f"Choices: {all_scope_names}  (default: linear)")

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
    parser.add_argument("--save-tensors", action="store_true",
                        help="Save per-layer matmul I/O tensors to "
                             "<results-dir>/<label>/tensors/ (one .npz per layer)")

    args = parser.parse_args()

    # ── Validate ────────────────────────────────────────────────────────────
    if args.functional_model is not None and args.mx_output_fmt != PASSTHROUGH:
        parser.error("--mx-output-fmt has no effect with --functional-model")

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

    # ── Build config record ──────────────────────────────────────────────────
    config_record = {
        "label":               args.label,
        "checkpoint_dir":      args.checkpoint_dir,
        "model_config":        args.config,
        "n_obs":               len(observations),
        "obs_dir":             args.obs_dir,
        "steps":               args.steps,
        "gpu":                 args.gpu,
        "fp8_mode":            args.fp8_mode,
        "active_groups":       [g.value for g in active_groups],
        "ops":                 [s.value for s in op_scopes],
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
    out_dir = Path(args.results_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    io_store = MatmulIOStore(out_dir / "tensors") if args.save_tensors else None

    print(f"\nRunning config: {args.label}")
    t0 = time.monotonic()
    mx_tracker, vec_tracker = run(
        model=model,
        observations=observations,
        device=device,
        active_groups=active_groups,
        op_scopes=op_scopes,
        mx_input_fmt=mx_input_fmt,
        mx_output_fmt=mx_output_fmt,
        vec_input_fmt=vec_input_fmt,
        vec_output_fmt=vec_output_fmt,
        functional_model_name=args.functional_model,
        num_steps=args.steps,
        t0=t0,
        matmul_io_store=io_store,
    )
    elapsed_s = time.monotonic() - t0
    config_record["elapsed_seconds"] = round(elapsed_s, 2)

    # ── Write outputs ────────────────────────────────────────────────────────
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
    _write_worst_layers(
        out_dir / "worst_layers.csv",
        mx_tracker, vec_tracker,
        top_n=20,
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
