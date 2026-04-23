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

Matrix path:
  --functional-model NAME   hardware-accurate simulation via functional model
                            available: ipt, ipt_numba, ipt_c, systolic_c
  (default: passthrough — no quantization)

Vector path (independent of matrix path):
  --vec-functional-model vector   route all vector ops through VectorRTLFunctions
                                  (default: passthrough — no interception)

Component selection:
  --active-groups vision,language,action_expert,action_head   (default: all)

Output — written to <results-dir>/<label>/:
  config.json        exact parameters used
  chronological.csv  one row per op call in execution order (local + cumulative RMSE)
  grouped.csv        same rows sorted by (component, layer_name)
  summary.csv        per-component aggregate stats (mx and vec separately)
  worst_layers.csv   top-20 layers by local rel RMSE across all components

Observation data:
  (default)     Random Gaussian noise — fast, no dataset required.
  --data-dir    Path to a droid_100 LeRobot dataset.  Loads real robot
                images, states, and task prompts.
  --frame-idx   Which frame within each episode to use (default: 0).
  --tokenizer-path  Path to paligemma_tokenizer.model. Auto-discovered
                from ~/.cache/openpi/big_vision/ or HF hub cache.

Usage:
    # IPT numba functional model, all op scopes, real data:
    OPENPI_DIR=/scratch/chloe.wong/openpi \\
    CUDA_VISIBLE_DEVICES=0 \\
    /scratch/chloe.wong/envs/pi0/bin/python experiments/run_eval.py \\
        --label ipt_numba_all_real \\
        --functional-model ipt_numba \\
        --ops linear,conv2d,attention \\
        --n-obs 4 --steps 10 \\
        --data-dir /scratch/chloe.wong/data/droid_100 \\
        --results-dir experiments/results/my_run
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
    get_functional_model_factory, list_functional_models,
    set_fp8_mode,
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


def _load_norm_stats(norm_stats_dir: str) -> dict:
    """Load norm_stats.json and return {key: SimpleNamespace(mean, std)}."""
    path = Path(norm_stats_dir) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats not found at: {path}")
    with open(path) as f:
        data = json.load(f)
    stats = {}
    for key, val in data["norm_stats"].items():
        stats[key] = SimpleNamespace(
            mean=np.array(val["mean"], dtype=np.float64),
            std=np.array(val["std"],  dtype=np.float64),
        )
    return stats


def _find_tokenizer(tokenizer_path: Optional[str]) -> str:
    """Resolve the SentencePiece tokenizer path, trying known locations."""
    if tokenizer_path is not None:
        return tokenizer_path
    candidates = [
        Path.home() / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model",
        Path.home() / "Desktop" / "paligemma_tokenizer.model",
    ]
    # Also search HF hub cache for paligemma
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    scratch_hf = Path("/scratch/chloe.wong/.huggingface/hub")
    for hub_root in [hf_cache, scratch_hf]:
        for snap_dir in sorted(hub_root.glob("models--google--paligemma*/snapshots/*/tokenizer.model")):
            candidates.append(snap_dir)
    for c in candidates:
        if Path(c).exists():
            return str(c)
    raise FileNotFoundError(
        "Cannot find paligemma_tokenizer.model. "
        "Pass --tokenizer-path or place it at ~/.cache/openpi/big_vision/paligemma_tokenizer.model"
    )


def _decode_video_frame(video_path: Path, global_frame_idx: int) -> np.ndarray:
    """Decode a single frame from a LeRobot mp4 (all episodes concatenated).

    Returns HWC uint8 RGB array.
    """
    import av
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        # pts = frame_idx * (duration / n_frames) — exact for constant-fps videos
        pts_per_frame = stream.duration / stream.frames
        target_pts = int(global_frame_idx * pts_per_frame)
        container.seek(target_pts, any_frame=False, backward=True, stream=stream)
        for frame in container.decode(stream):
            return frame.to_ndarray(format="rgb24")  # HWC uint8
    raise RuntimeError(f"Could not decode frame {global_frame_idx} from {video_path}")


def _load_droid_obs(
    data_dir: Path,
    n_obs: int,
    frame_idx: int,
    tokenizer_path: Optional[str],
    max_token_len: int,
    device: torch.device,
    norm_stats: Optional[dict] = None,
) -> list:
    """Load real observations from a droid_100 LeRobot dataset.

    Selects n_obs episodes (seeded shuffle for reproducibility), extracts
    frame `frame_idx` from each episode, and builds Pi0 SimpleNamespace obs.
    """
    import pandas as pd
    import sentencepiece
    import torchvision.transforms.functional as TF

    data_dir = Path(data_dir)
    df = pd.read_parquet(data_dir / "data" / "chunk-000" / "file-000.parquet")
    tasks_df = pd.read_parquet(data_dir / "meta" / "tasks.parquet")
    # tasks_df: row-label = task text string, column "task_index" = int
    task_text_map: dict[int, str] = dict(
        zip(tasks_df["task_index"].values.tolist(), tasks_df.index.tolist())
    )

    # Reproducible episode selection
    episodes = sorted(df["episode_index"].unique())
    rng = np.random.RandomState(0)
    rng.shuffle(episodes)
    selected_eps = episodes[:n_obs]

    tok_path = _find_tokenizer(tokenizer_path)
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(tok_path)
    print(f"[droid] tokenizer: {tok_path}  vocab={sp.GetPieceSize()}")

    cam_to_key = {
        "observation.images.exterior_image_1_left": "base_0_rgb",
        "observation.images.wrist_image_left":      "left_wrist_0_rgb",
    }
    video_root = data_dir / "videos"

    H, W = 224, 224
    observations = []

    for ep_idx in selected_eps:
        ep_rows = df[df["episode_index"] == ep_idx]
        # Get the row for the requested frame within this episode
        ep_row = ep_rows[ep_rows["frame_index"] == frame_idx]
        if ep_row.empty:
            # Fallback to first frame if frame_idx is out of range
            ep_row = ep_rows[ep_rows["frame_index"] == ep_rows["frame_index"].min()]
        ep_row = ep_row.iloc[0]
        global_idx = int(ep_row["index"])
        task_idx   = int(ep_row["task_index"])
        task_text  = task_text_map.get(task_idx, "")

        # ── Images ───────────────────────────────────────────────────────────
        images: dict[str, torch.Tensor] = {}
        for cam_name, key in cam_to_key.items():
            vid_path = video_root / cam_name / "chunk-000" / "file-000.mp4"
            raw_hwc = _decode_video_frame(vid_path, global_idx)  # HWC uint8
            t = torch.from_numpy(raw_hwc).permute(2, 0, 1)       # CHW uint8
            t = TF.resize(t, [H, W], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            t = t.to(torch.float32) / 255.0 * 2.0 - 1.0          # [-1, 1]
            images[key] = t.unsqueeze(0).to(device)               # [1,3,H,W]
        images["right_wrist_0_rgb"] = torch.zeros(1, 3, H, W, dtype=torch.float32, device=device)

        image_masks = {
            "base_0_rgb":        torch.ones(1,  dtype=torch.bool, device=device),
            "left_wrist_0_rgb":  torch.ones(1,  dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
        }

        # ── State: [7] → normalize → pad to [32] ────────────────────────────
        raw_state = np.array(ep_row["observation.state"], dtype=np.float32).flatten()
        if norm_stats and "state" in norm_stats:
            s = norm_stats["state"]
            dim = raw_state.shape[0]
            raw_state = (raw_state - s.mean[:dim].astype(np.float32)) / (s.std[:dim].astype(np.float32) + 1e-6)
        state_arr = np.zeros(32, dtype=np.float32)
        state_arr[:raw_state.shape[0]] = raw_state
        state = torch.from_numpy(state_arr).unsqueeze(0).to(device)  # [1,32]

        # ── Tokenise task prompt ──────────────────────────────────────────────
        prompt = task_text.strip().replace("_", " ").replace("\n", " ")
        if prompt:
            tokens = sp.Encode(prompt, add_bos=True) + sp.Encode("\n")
        else:
            tokens = []
        if len(tokens) > max_token_len:
            tokens = tokens[:max_token_len]
        pad_len = max_token_len - len(tokens)
        tokens_padded = tokens + [0] * pad_len
        mask_list = [True] * len(tokens) + [False] * pad_len

        tokenized_prompt      = torch.tensor([tokens_padded], dtype=torch.int64, device=device)
        tokenized_prompt_mask = torch.tensor([mask_list],     dtype=torch.bool,  device=device)
        token_ar_mask         = torch.zeros(1, max_token_len, dtype=torch.bool, device=device)
        token_loss_mask       = torch.zeros(1, max_token_len, dtype=torch.bool, device=device)

        print(f"  [droid] ep={ep_idx:3d}  global_frame={global_idx:5d}  "
              f"task='{task_text[:60]}'")

        observations.append(SimpleNamespace(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
        ))

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


def _mx_layer_tag(layer_name: str, comp_str: str) -> str:
    """Extract display layer tag from a QuantLinear module path and component string."""
    if comp_str == "vision":
        return "vision"
    if comp_str == "action_head":
        return "action_head"
    parts = layer_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            idx = parts[i + 1]
            if comp_str == "language":
                return f"language.{idx}"
            if comp_str == "action_expert":
                return f"expert.{idx}"
    return comp_str  # fallback (e.g. "unknown")


def _write_per_layer_csv(path: Path, mx_tracker: StatsTracker, vec_tracker: StatsTracker) -> None:
    """
    Write per-transformer-layer combined matrix + vector RMSE.

    layer_tag values: "vision", "language.0".."language.17",
                      "expert.0".."expert.17", "action_head", "unattributed".
    """
    from collections import defaultdict
    mx_by_tag:  dict[str, list[float]] = defaultdict(list)
    vec_by_tag: dict[str, list[float]] = defaultdict(list)

    for call in mx_tracker.calls:
        tag  = _mx_layer_tag(call["name"], call["component"])
        rmse = call["rmse"]
        if math.isfinite(rmse):
            mx_by_tag[tag].append(rmse)

    for call in vec_tracker.calls:
        parsed = _parse_vec_name(call["name"])
        if parsed is None:
            continue
        tag  = parsed[0]
        rmse = call["rmse"]
        if math.isfinite(rmse):
            vec_by_tag[tag].append(rmse)

    all_tags = sorted(set(list(mx_by_tag.keys()) + list(vec_by_tag.keys())))
    fields = ["layer_tag", "n_mx", "mx_mean_rmse", "n_vec", "vec_mean_rmse"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tag in all_tags:
            mx_vals  = mx_by_tag.get(tag,  [])
            vec_vals = vec_by_tag.get(tag, [])
            w.writerow({
                "layer_tag":    tag,
                "n_mx":         len(mx_vals),
                "mx_mean_rmse": sum(mx_vals) / len(mx_vals) if mx_vals else float("nan"),
                "n_vec":        len(vec_vals),
                "vec_mean_rmse": sum(vec_vals) / len(vec_vals) if vec_vals else float("nan"),
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
     "vec_functional_model",
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
        "timestamp":            datetime.datetime.now().isoformat(timespec="seconds"),
        "label":                config_record["label"],
        "elapsed_seconds":      elapsed_s,
        "elapsed_human":        elapsed_td,
        "vec_functional_model": vp.get("vec_functional_model") or "passthrough",
        "functional_model":     mp.get("functional_model") or "",
        "active_groups":        "|".join(config_record.get("active_groups", [])),
        "ops":                  "|".join(config_record.get("ops", [])),
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

def _parse_vec_name(name: str) -> tuple[str, str] | None:
    """
    Parse a vec record name into (layer_tag, op_short).

    New format: "vec.{layer_tag}.{aten::op}.{seq}"
    where layer_tag may be multi-part, e.g. "language.7".
    The aten op name is identified by containing "::".

    Examples:
        "vec.language.7.aten::add.42"         → ("language.7", "add")
        "vec.vision.aten::native_layer_norm.5" → ("vision", "native_layer_norm")
        "vec.unattributed.aten::mul.100"       → ("unattributed", "mul")
    """
    if not name.startswith("vec."):
        return None
    rest  = name[4:]   # strip "vec."
    parts = rest.split(".")
    for i, p in enumerate(parts):
        if "::" in p:
            layer_tag = ".".join(parts[:i])
            op_short  = p.split("::")[-1]
            return layer_tag, op_short
    return None


def _vec_op_breakdown(vec_tracker: StatsTracker) -> list[tuple[str, str, float, int]]:
    """
    Parse vec tracker call records into (layer_tag, op_short, mean_rel_rmse, n_calls).
    Layer names are formatted as 'vec.{layer_tag}.{aten::op}.{seq}'.
    """
    from collections import defaultdict
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts:  dict[tuple[str, str], int]         = defaultdict(int)
    for call in vec_tracker.calls:
        parsed = _parse_vec_name(call.get("name", ""))
        if parsed is None:
            continue
        layer_tag, op_short = parsed
        ref_rms = call.get("ref_rms", 0.0)
        rmse    = call.get("rmse",    0.0)
        rel = (rmse / ref_rms) if (ref_rms and not math.isnan(ref_rms) and ref_rms > 0) else float("nan")
        if not math.isnan(rel):
            buckets[(layer_tag, op_short)].append(rel)
        counts[(layer_tag, op_short)] += 1

    rows = []
    for (layer_tag, op), vals in sorted(buckets.items()):
        mean_rel = sum(vals) / len(vals) if vals else float("nan")
        rows.append((layer_tag, op, mean_rel, counts[(layer_tag, op)]))
    return rows


def _print_intermediate(
    label: str,
    mx_tracker: StatsTracker,
    vec_tracker: StatsTracker,
    elapsed_s: float,
    hook_fires: Optional[dict] = None,
) -> None:
    """Print a compact per-component RMSE table and vec op breakdown."""
    total_calls = mx_tracker._seq + vec_tracker._seq
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_s)))
    print(f"\n  elapsed={elapsed_str}  layer_calls={total_calls}", end="")
    if hook_fires:
        fires_str = "  hooks: " + "  ".join(
            f"{k}={v}" for k, v in sorted(hook_fires.items()) if v > 0
        )
        print(fires_str, end="")
    print()

    # ── Per-component summary ─────────────────────────────────────────────────
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

    # ── Vec op breakdown (only when vec FM is active) ─────────────────────────
    op_rows = _vec_op_breakdown(vec_tracker)
    if op_rows:
        print(f"\n  Vec op breakdown:")
        print(f"  {'layer_tag':<20} {'op':<24} {'mean_rel_rmse':>13}  {'calls':>6}")
        print(f"  {'-'*20} {'-'*24} {'-'*13}  {'-'*6}")
        cur_tag = None
        for tag, op, rel, n in op_rows:
            tag_label = tag if tag != cur_tag else ""
            cur_tag = tag
            print(f"  {tag_label:<20} {op:<24} {rel:>13.4e}  {n:>6}")


def _start_heartbeat(
    mx_tracker: StatsTracker,
    vec_tracker: StatsTracker,
    t0: float,
    stop_event: threading.Event,
    hook_fires: Optional[dict] = None,
    interval_s: int = 30,
) -> threading.Thread:
    """Background thread: prints a one-liner every `interval_s` seconds."""
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
    functional_model_name: Optional[str],
    vec_functional_model_name: Optional[str],
    num_steps: int,
    t0: float,
    trace: bool = False,
) -> tuple[StatsTracker, StatsTracker]:
    """
    Patch model, run observations, unpatch.  Returns (mx_tracker, vec_tracker).
    """
    mx_tracker  = StatsTracker()
    vec_tracker = StatsTracker()

    # Resolve matrix functional model factory
    fm_factory = None
    if functional_model_name is not None:
        fm_factory = get_functional_model_factory(functional_model_name)

    # Resolve vector functional model
    vec_fm = None
    if vec_functional_model_name is not None:
        from funct_models_vector.vector_rtl_forward import VectorRTLFunctions
        vec_fm = VectorRTLFunctions(num_lanes=16)

    # ── Capture reference (unpatched) layer outputs for cumulative RMSE ──────
    ref_store = ReferenceStore()
    layer_names = {
        name for name, m in model.named_modules()
        if type(m) is nn.Linear or type(m) is nn.Conv2d
    }
    ref_hooks = ref_store.register_hooks(model, layer_names)

    # Also capture eager_attention_forward and SDPA outputs when attention is active.
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

    if OpScope.ATTENTION in op_scopes:
        _mg_ref.eager_attention_forward    = _ref_orig_eager
        _F_ref.scaled_dot_product_attention = _ref_orig_sdpa

    print(f"[reference_store] Captured {len(ref_store)} reference layer outputs.")

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
    # Rough estimated_total for trace percentage: count patched linear layers,
    # multiply by (1 prefix pass + num_steps expert passes) × n_obs × ~2 (mx+vec).
    from pi0_inout.quant_linear import QuantLinear as _QL
    _n_mx = sum(1 for _, m in model.named_modules() if isinstance(m, _QL))
    _estimated_total = _n_mx * (1 + num_steps) * len(observations) * 2
    if trace:
        print(f"[trace] estimated_total={_estimated_total}  (n_mx={_n_mx})", flush=True)
    # Back-fill estimated_total into already-constructed QuantLinear layers
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
    )

    n_obs = len(observations)
    stop_heartbeat = threading.Event()
    _start_heartbeat(mx_tracker, vec_tracker, t0, stop_heartbeat, hook_fires=hook_fires)

    with torch.no_grad(), vec_ctx:
        for i, obs in enumerate(observations):
            print(f"\n[obs {i + 1}/{n_obs}] running ({num_steps} diffusion steps)...", flush=True)
            obs_t0 = time.monotonic()
            torch.manual_seed(i)
            model.sample_actions(str(device), obs, num_steps=num_steps)
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
    parser.add_argument("--checkpoint-dir", default="/scratch/chloe.wong/data/pi0_droid_jointpos_safetensors")
    parser.add_argument("--config", default="pi0_droid_jointpos_polaris")
    parser.add_argument("--gpu",    type=int, default=0)

    # Eval settings
    parser.add_argument("--n-obs",  type=int, default=4,
                        help="Number of observations to run "
                             "(random dummy obs, or episodes when --data-dir is set)")
    parser.add_argument("--steps",  type=int, default=10,
                        help="Diffusion steps per sample_actions call")

    # Real data (optional)
    parser.add_argument("--data-dir", default=None, metavar="PATH",
                        help="Path to a droid_100 LeRobot dataset directory. "
                             "When set, loads real robot observations instead of random noise.")
    parser.add_argument("--frame-idx", type=int, default=0, metavar="N",
                        help="Frame index within each episode to use (default: 0 = first frame)")
    parser.add_argument("--tokenizer-path", default=None, metavar="PATH",
                        help="Path to paligemma_tokenizer.model. Auto-discovered from "
                             "~/.cache/openpi/big_vision/ or HF hub cache if not given.")
    parser.add_argument("--norm-stats-dir", default=None, metavar="PATH",
                        help="Directory containing norm_stats.json for state z-score normalization. "
                             "When set, state is normalized before passing to the model. "
                             "Recommended: <checkpoint-dir>/assets/droid")

    # Matrix path
    parser.add_argument("--functional-model", metavar="NAME",
                        help=f"Hardware-accurate model for matmuls. "
                             f"Available: {list_functional_models()}")

    # Vector path
    parser.add_argument("--vec-functional-model", metavar="NAME",
                        choices=["vector"],
                        default=None,
                        help="Route all vector ops through VectorRTLFunctions. "
                             "Currently the only option is 'vector'.")

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
    parser.add_argument("--fp8-mode", default="po2", choices=["po2"],
                        help="FP8 scaling mode: po2=power-of-two (only supported mode)")
    parser.add_argument("--trace", action="store_true",
                        help="Print one line per op as it fires (MX and VEC) "
                             "with sequence number, layer tag, shape, and RMSE.")

    args = parser.parse_args()

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

    set_fp8_mode(args.fp8_mode)

    # ── Device / model ───────────────────────────────────────────────────────
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    from pi0_inout.serve_quant import load_pi0_pytorch, _get_model_config
    config_ns = _get_model_config(args.config)
    print(f"Loading model: {args.config}  checkpoint: {args.checkpoint_dir}")
    model = load_pi0_pytorch(args.config, args.checkpoint_dir, device)
    model.eval()
    # Remove torch.compile wrapping (applied in PI0Pytorch.__init__) so that
    # register_forward_hook fires correctly for all components during eval.
    if "sample_actions" in model.__dict__:
        del model.sample_actions
        print("[run_eval] Removed torch.compile wrapper from sample_actions (eager mode).")
    else:
        print(f"[run_eval] WARNING: sample_actions not in model.__dict__ — compile may still be active. Keys: {list(model.__dict__.keys())[:10]}")

    torch.manual_seed(0)
    norm_stats = None
    if args.norm_stats_dir:
        norm_stats = _load_norm_stats(args.norm_stats_dir)
        print(f"Loaded norm stats from: {args.norm_stats_dir}  (keys: {list(norm_stats.keys())})")

    if args.data_dir is not None:
        print(f"Loading real DROID observations from: {args.data_dir}")
        observations = _load_droid_obs(
            data_dir=Path(args.data_dir),
            n_obs=args.n_obs,
            frame_idx=args.frame_idx,
            tokenizer_path=args.tokenizer_path,
            max_token_len=config_ns.max_token_len,
            device=device,
            norm_stats=norm_stats,
        )
        print(f"Observations: {len(observations)} real  steps: {args.steps}")
    else:
        observations = [_make_dummy_obs(config_ns, device) for _ in range(args.n_obs)]
        print(f"Observations: {args.n_obs} random  steps: {args.steps}")

    # ── Build config record ──────────────────────────────────────────────────
    config_record = {
        "label":               args.label,
        "checkpoint_dir":      args.checkpoint_dir,
        "model_config":        args.config,
        "n_obs":               len(observations),
        "data_dir":            args.data_dir,
        "frame_idx":           args.frame_idx if args.data_dir else None,
        "steps":               args.steps,
        "gpu":                 args.gpu,
        "fp8_mode":            args.fp8_mode,
        "active_groups":       [g.value for g in active_groups],
        "ops":                 [s.value for s in op_scopes],
        "matrix_path": {
            "functional_model": args.functional_model,
        },
        "vector_path": {
            "vec_functional_model": args.vec_functional_model or "passthrough",
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
        op_scopes=op_scopes,
        functional_model_name=args.functional_model,
        vec_functional_model_name=args.vec_functional_model,
        num_steps=args.steps,
        t0=t0,
        trace=args.trace,
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
    _write_worst_layers(
        out_dir / "worst_layers.csv",
        mx_tracker, vec_tracker,
        top_n=20,
    )
    _write_per_layer_csv(
        out_dir / "per_layer.csv",
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
