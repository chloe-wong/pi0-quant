"""
serve_quant.py
--------------
Drop-in replacement for openpi/scripts/serve_policy.py that serves
Pi0Pytorch with hardware-accurate simulation via functional models.

How it works
------------
1. Adds openpi/src and openpi-client/src to sys.path.
2. Injects lightweight Python stubs for the three openpi files that import
   JAX at module level (gemma config, lora config, array_typing, image_tools).
   This lets Pi0Pytorch be imported in the pi0 conda env (torch-only).
3. Instantiates Pi0Pytorch from a hardcoded config dict for the known
   training configs (no JAX config system required).
4. Loads weights from a local safetensors checkpoint if available.
5. Patches every nn.Linear with QuantLinear routed through the chosen
   matrix functional model (IPT / systolic), or BF16 passthrough if omitted.
6. Patches all vector ops through VectorRTLFunctions if --vec-functional-model
   is set, or passthrough if omitted.
7. Serves via the openpi WebSocket protocol (msgpack + websockets).
8. On SIGTERM/SIGINT, writes per-layer RMSE stats to --stats-output.

Both paths are independent. Omitting both gives a zero-RMSE BF16 baseline.

Usage
-----
    python serve_quant.py \\
        --openpi-dir /path/to/openpi \\
        --checkpoint-dir /path/to/model.safetensors_dir \\
        --config pi0_droid_jointpos_polaris \\
        --functional-model ipt_numba \\
        --vec-functional-model vector \\
        --port 8003 --gpu 0
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import atexit
import contextlib
import json
import logging
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Make sure openpi source and openpi-client are importable ─────────────────
_THIS_DIR    = Path(__file__).resolve().parent
_OPENPI_DIR  = Path(os.environ.get("OPENPI_DIR", _THIS_DIR.parent / "openpi"))
_OPENPI_SRC  = _OPENPI_DIR / "src"
_CLIENT_SRC  = _OPENPI_DIR / "packages" / "openpi-client" / "src"
_PI0_INOUT   = _THIS_DIR          # for "from pi0_inout import ..."

for _p in [str(_PI0_INOUT.parent), str(_CLIENT_SRC), str(_OPENPI_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Inject JAX stubs BEFORE any openpi import ────────────────────────────────
from pi0_inout._jax_stubs import inject as _inject_jax_stubs   # noqa: E402
_inject_jax_stubs()

# ── Now it is safe to import the pytorch model ────────────────────────────────
import numpy as np
import torch
import torch.nn as nn

# ── pi0_inout imports (quantization layer) ────────────────────────────────────
from pi0_inout.quant_types import QuantFormat, set_fp8_mode
from pi0_inout.model_patcher import (
    patch_model, list_linear_layers,
    QuantGroup, ALL_GROUPS,
    patch_attn_sdpa, unpatch_attn_sdpa,
)
from pi0_inout.quant_linear import QuantLinear
from pi0_inout.rel_noise import NoiseMode
from pi0_inout.stats_tracker import StatsTracker
from pi0_inout.functional_models import get_functional_model_factory, list_functional_models
from pi0_inout.quant_vector import patch_vector_ops, unpatch_vector_ops


# ---------------------------------------------------------------------------
# Known training configs  (avoids importing openpi.training.config / JAX)
# ---------------------------------------------------------------------------

# Each entry: SimpleNamespace with the fields PI0Pytorch.__init__ reads.
# Add new configs here as needed.
_KNOWN_CONFIGS: dict[str, SimpleNamespace] = {
    # pi05 droid joint-position policy (Polaris)
    "pi05_droid_jointpos_polaris": SimpleNamespace(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        dtype="bfloat16",
        action_dim=32,
        action_horizon=15,
        max_token_len=200,
        pytorch_compile_mode="max-autotune",
    ),
    # pi05 droid (non-Polaris)
    "pi05_droid": SimpleNamespace(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        dtype="bfloat16",
        action_dim=32,
        action_horizon=15,
        max_token_len=200,
        pytorch_compile_mode="max-autotune",
    ),
    # pi0 droid joint-position Polaris (non-pi05)
    "pi0_droid_jointpos_polaris": SimpleNamespace(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
        dtype="bfloat16",
        action_dim=32,
        action_horizon=10,
        max_token_len=100,
        pytorch_compile_mode="max-autotune",
    ),
    # pi0 droid joint-position (non-pi05)
    "pi0_droid": SimpleNamespace(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
        dtype="bfloat16",
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
        pytorch_compile_mode="max-autotune",
    ),
    # aloha sim (non-pi05)
    "pi0_aloha_sim": SimpleNamespace(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
        dtype="bfloat16",
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
        pytorch_compile_mode="max-autotune",
    ),
}


def _get_model_config(config_name: str) -> SimpleNamespace:
    if config_name in _KNOWN_CONFIGS:
        return _KNOWN_CONFIGS[config_name]
    # Fallback: pi05 defaults
    logger.warning(
        f"Config '{config_name}' not in _KNOWN_CONFIGS; using pi05 defaults. "
        f"Known: {list(_KNOWN_CONFIGS)}"
    )
    return _KNOWN_CONFIGS["pi05_droid_jointpos_polaris"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pi0_pytorch(
    config_name: str,
    checkpoint_dir: str,
    device: torch.device,
) -> nn.Module:
    """
    Load PI0Pytorch without JAX.

    1. Constructs the model config from _KNOWN_CONFIGS.
    2. Instantiates PI0Pytorch (stubs handle JAX imports).
    3. Loads weights from a local safetensors file if available.
       Falls back to random init with a warning if no checkpoint found.
    """
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    cfg = _get_model_config(config_name)
    logger.info(
        f"Instantiating PI0Pytorch: pi05={cfg.pi05}, "
        f"paligemma={cfg.paligemma_variant}, expert={cfg.action_expert_variant}, "
        f"dtype={cfg.dtype}, action_horizon={cfg.action_horizon}"
    )

    model = PI0Pytorch(cfg)
    # torch.compile is applied in __init__; QuantLinear has Python-level conditional
    # logic that Dynamo cannot trace. Unwrap back to the raw Python method.

    model = model.to(device)
    model.eval()

    _load_checkpoint(model, checkpoint_dir, device)
    return model


def _load_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    device: torch.device,
) -> None:
    """
    Load weights into the model.  Tries (in order):
      1. <checkpoint_dir>/model.safetensors   (train_pytorch.py output)
      2. <checkpoint_dir>/pytorch_model.pt
      3. <checkpoint_dir>/pytorch_model.bin
    Falls back to a warning (random init) if none found.
    GCS paths (gs://) are skipped — download them first with gsutil.
    """
    if checkpoint_dir.startswith("gs://"):
        logger.warning(
            f"Checkpoint is a GCS path: {checkpoint_dir}\n"
            "  GCS checkpoints are JAX/Orbax format and require JAX to load.\n"
            "  Convert to safetensors first with:\n"
            "    python /path/to/openpi/examples/convert_jax_model_to_pytorch.py \\\n"
            f"        --checkpoint_dir <local_download_of_{checkpoint_dir}> \\\n"
            f"        --config_name pi05_droid_jointpos_polaris \\\n"
            "        --output_path /path/to/checkpoints/pi05_droid_pytorch\n"
            "  Then pass --checkpoint-dir /path/to/checkpoints/pi05_droid_pytorch\n"
            "  Running with RANDOM WEIGHTS (RMSE results will still be meaningful\n"
            "  for quantization analysis relative to the fp32 baseline)."
        )
        return

    ckpt_dir = Path(checkpoint_dir)

    # safetensors (preferred)
    sf_path = ckpt_dir / "model.safetensors"
    if sf_path.exists():
        try:
            import safetensors.torch
            safetensors.torch.load_model(model, str(sf_path), device=str(device))
            logger.info(f"Loaded weights from {sf_path}")
            return
        except Exception as e:
            logger.warning(f"safetensors load failed: {e}")

    # legacy PyTorch save
    for candidate in ["pytorch_model.pt", "pytorch_model.bin", "model.pt"]:
        pt_path = ckpt_dir / candidate
        if pt_path.exists():
            state = torch.load(str(pt_path), map_location=device)
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded weights from {pt_path}")
            return

    logger.warning(
        f"No checkpoint found in {checkpoint_dir}. "
        "Running with RANDOM WEIGHTS — RMSE will still correctly measure "
        "quantization error relative to the fp32 baseline, but the benchmark "
        "success rates and videos will not reflect the real model's behaviour."
    )


# ---------------------------------------------------------------------------
# Norm stats loading (self-contained — no pydantic/numpydantic dependency)
# ---------------------------------------------------------------------------

def _load_norm_stats(norm_stats_dir: str) -> dict:
    """Load norm_stats.json and return {key: SimpleNamespace(mean, std, q01, q99)}."""
    path = Path(norm_stats_dir) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats not found at: {path}")
    with open(path) as f:
        data = json.load(f)
    stats = {}
    for key, val in data["norm_stats"].items():
        stats[key] = SimpleNamespace(
            mean=np.array(val["mean"], dtype=np.float64),
            std=np.array(val["std"], dtype=np.float64),
            q01=np.array(val["q01"], dtype=np.float64) if val.get("q01") is not None else None,
            q99=np.array(val["q99"], dtype=np.float64) if val.get("q99") is not None else None,
        )
    return stats


# ---------------------------------------------------------------------------
# Post-patch diagnostics
# ---------------------------------------------------------------------------

def print_patch_diagnostics(
    model: nn.Module,
    functional_model: Optional[str],
    vec_functional_model: Optional[str],
) -> None:
    """Print a brief summary of the patched model configuration."""
    n_quant = sum(1 for _, m in model.named_modules() if isinstance(m, QuantLinear))
    n_plain = sum(1 for _, m in model.named_modules() if type(m) is nn.Linear)

    print("\n" + "=" * 60)
    print("PATCH DIAGNOSTICS")
    print("=" * 60)
    print(f"  QuantLinear layers : {n_quant}")
    print(f"  Plain nn.Linear    : {n_plain}")
    print(f"  Matrix path        : {functional_model or 'passthrough (BF16)'}")
    print(f"  Vector path        : {vec_functional_model or 'passthrough'}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Policy shim (Pi0PyTorchPolicy)
# ---------------------------------------------------------------------------

class Pi0PyTorchPolicy:
    """
    Adapts PI0Pytorch to the openpi policy interface expected by
    WebsocketPolicyServer:
        policy.infer(obs_dict)  → {"actions": np.ndarray}
        policy.metadata         → dict

    Replicates the openpi output transform pipeline:
        1. Normalize input state
        2. Model inference
        3. Unnormalize output actions (and state)
        4. AbsoluteActions: delta → absolute joint positions (joint-pos configs)
        5. Slice to first 8 dims (7 joints + 1 gripper)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        vec_ctx=None,
        norm_stats: dict | None = None,
        use_quantile_norm: bool = False,
        is_joint_position: bool = False,
        max_token_len: int = 48,
        tokenizer_path: str | None = None,
    ) -> None:
        self.model    = model
        self.device   = device
        self._vec_ctx = vec_ctx
        self.metadata: dict = {"model": "PI0Pytorch", "quantized": True}
        self.norm_stats = norm_stats
        self.use_quantile_norm = use_quantile_norm
        self.is_joint_position = is_joint_position
        self.max_token_len = max_token_len

        # Load SentencePiece tokenizer for prompt encoding
        import sentencepiece
        if tokenizer_path is None:
            # Try common locations
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
                "Pass --tokenizer-path or place it at ~/.cache/openpi/big_vision/paligemma_tokenizer.model"
            )
        with open(tokenizer_path, "rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        logger.info(f"Loaded tokenizer from {tokenizer_path} (vocab_size={self._tokenizer.vocab_size()})")

    # ── Normalization helpers (mirrors openpi/transforms.py) ──────────────

    def _normalize(self, x: np.ndarray, stats) -> np.ndarray:
        """Normalize x using norm_stats (truncates stats to x's last dim)."""
        if self.use_quantile_norm:
            q01 = stats.q01[..., :x.shape[-1]]
            q99 = stats.q99[..., :x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        mean = stats.mean[..., :x.shape[-1]]
        std  = stats.std[..., :x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _unnormalize(self, x: np.ndarray, stats) -> np.ndarray:
        """Unnormalize x using norm_stats (pads stats to x's last dim)."""
        if self.use_quantile_norm:
            q01, q99 = stats.q01, stats.q99
            dim = q01.shape[-1]
            if dim < x.shape[-1]:
                norm_part = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
                return np.concatenate([norm_part, x[..., dim:]], axis=-1)
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        # z-score: pad mean with 0, std with 1 for extra dims
        dim = stats.mean.shape[-1]
        extra = max(0, x.shape[-1] - dim)
        mean = np.concatenate([stats.mean, np.zeros(extra)])
        std  = np.concatenate([stats.std,  np.ones(extra)])
        return x * (std + 1e-6) + mean

    # ── Inference ─────────────────────────────────────────────────────────

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map DROID WebSocket obs dict → PI0Pytorch SimpleNamespace and run
        inference, then apply the full output transform pipeline.
        """
        dev = self.device
        dtype = torch.float32

        def _img_tensor(arr: np.ndarray) -> torch.Tensor:
            """uint8 HWC [H,W,3] → float tensor [1,3,H,W] in [-1, 1]."""
            t = torch.from_numpy(arr.copy()).to(dev)
            t = t.permute(2, 0, 1).unsqueeze(0).to(dtype)
            t = t / 255.0 * 2.0 - 1.0  # uint8 [0,255] → [-1, 1]
            return t

        H, W = 224, 224

        base_img  = _img_tensor(obs["observation/exterior_image_1_left"])
        wrist_img = _img_tensor(obs["observation/wrist_image_left"])
        zero_img  = torch.zeros(1, 3, H, W, dtype=dtype, device=dev)

        images = {
            "base_0_rgb":        base_img,
            "left_wrist_0_rgb":  wrist_img,
            "right_wrist_0_rgb": zero_img,
        }
        image_masks = {
            "base_0_rgb":        torch.ones(1,  dtype=torch.bool, device=dev),
            "left_wrist_0_rgb":  torch.ones(1,  dtype=torch.bool, device=dev),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=dev),
        }

        # ── State: concat joint_pos + gripper, normalize, pad to 32 ──────
        joint = np.array(obs["observation/joint_position"], dtype=np.float32).flatten()
        grip  = np.array(obs["observation/gripper_position"], dtype=np.float32).flatten()
        raw_state = np.concatenate([joint, grip])  # (8,)

        if self.norm_stats and "state" in self.norm_stats:
            norm_state = self._normalize(raw_state, self.norm_stats["state"])
        else:
            norm_state = raw_state

        state_padded = np.zeros(32, dtype=np.float32)
        state_padded[:norm_state.shape[0]] = norm_state
        state = torch.from_numpy(state_padded).unsqueeze(0).to(dev)  # (1, 32)

        # ── Tokenise prompt (matches openpi PaligemmaTokenizer) ────────────
        max_tok = self.max_token_len
        prompt_text = obs.get("prompt", "")
        if isinstance(prompt_text, bytes):
            prompt_text = prompt_text.decode("utf-8")
        prompt_text = prompt_text.strip().replace("_", " ").replace("\n", " ")

        if prompt_text:
            tokens = self._tokenizer.encode(prompt_text, add_bos=True) + self._tokenizer.encode("\n")
        else:
            tokens = []

        tok_len = len(tokens)
        if tok_len > max_tok:
            tokens = tokens[:max_tok]
            tok_len = max_tok
        # Pad to max_tok
        pad_len = max_tok - tok_len
        tokens_padded = tokens + [0] * pad_len
        mask_list = [True] * tok_len + [False] * pad_len

        tokenized_prompt      = torch.tensor([tokens_padded], dtype=torch.int64, device=dev)
        tokenized_prompt_mask = torch.tensor([mask_list],      dtype=torch.bool,  device=dev)
        token_ar_mask         = torch.zeros(1, max_tok, dtype=torch.bool, device=dev)
        token_loss_mask       = torch.zeros(1, max_tok, dtype=torch.bool, device=dev)

        obs_ns = SimpleNamespace(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
        )

        vec = self._vec_ctx if self._vec_ctx is not None else contextlib.nullcontext()
        with torch.no_grad(), vec:
            actions = self.model.sample_actions(str(dev), obs_ns, num_steps=10)
        # actions: [1, action_horizon, 32]  (normalized action space)
        actions = actions.squeeze(0).cpu().numpy()  # (horizon, 32)

        # ── Output transforms (mirrors openpi pipeline) ───────────────────
        # 1. Unnormalize actions (normalized delta → physical delta)
        if self.norm_stats and "actions" in self.norm_stats:
            actions = self._unnormalize(actions, self.norm_stats["actions"])

        # 2. AbsoluteActions for joint-position configs:
        #    model outputs delta joint positions → add current state to get absolute
        #    mask = make_bool_mask(7, -1) = first 7 dims True, last dim (gripper) False
        if self.is_joint_position:
            actions[..., :7] += raw_state[:7]

        # 3. Slice to 8 dims (7 joints + 1 gripper)
        return {"actions": actions[:, :8]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        force=True,
    )
    args = parse_args()

    # Propagate openpi dir to stubs (in case it was set via CLI)
    if args.openpi_dir:
        global _OPENPI_DIR, _OPENPI_SRC, _CLIENT_SRC
        _OPENPI_DIR = Path(args.openpi_dir)
        _OPENPI_SRC = _OPENPI_DIR / "src"
        _CLIENT_SRC = _OPENPI_DIR / "packages" / "openpi-client" / "src"
        for _p in [str(_CLIENT_SRC), str(_OPENPI_SRC)]:
            if _p not in sys.path:
                sys.path.insert(0, _p)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Load and optionally list layers ──────────────────────────────────
    logger.info(f"Loading model: config={args.config}  ckpt={args.checkpoint_dir}")
    model = load_pi0_pytorch(args.config, args.checkpoint_dir, device)

    if args.list_layers:
        rows = list_linear_layers(model)
        print(f"\n{'Component':16s}  {'In':6s}  {'Out':6s}  Name")
        print("-" * 80)
        for r in rows:
            print(f"{r['component']:16s}  {r['in_features']:6d}  {r['out_features']:6d}  {r['name']}")
        print(f"\nTotal linear layers: {len(rows)}")
        return

    # ── Resolve functional models and patch ───────────────────────────────
    set_fp8_mode("po2")

    active_groups = {QuantGroup(g) for g in args.quantize_components}
    noise_mode    = NoiseMode(args.noise_mode)

    fm_factory = None
    if args.functional_model:
        fm_factory = get_functional_model_factory(args.functional_model)

    vec_fm      = None
    vec_handles = []
    vec_ctx     = None
    if args.vec_functional_model is not None:
        from funct_models_vector.vector_rtl_forward import VectorRTLFunctions
        vec_fm = VectorRTLFunctions(num_lanes=16)

    tracker     = StatsTracker()
    vec_tracker = StatsTracker()

    patch_model(
        model=model,
        mx_input_fmt=QuantFormat.BFLOAT16,   # passthrough; FM handles actual compute
        mx_output_fmt=QuantFormat.BFLOAT16,
        tracker=tracker,
        active_groups=active_groups,
        functional_model_factory=fm_factory,
        noise_injection=args.rel_err,
        noise_mode=noise_mode,
        verbose=False,
    )
    attn_handles = patch_attn_sdpa(
        model=model,
        active_groups=active_groups,
        tracker=tracker,
        functional_model_factory=fm_factory,
    )
    vec_handles, vec_ctx, _hook_fires = patch_vector_ops(
        model,
        active_groups=active_groups,
        tracker=vec_tracker,
        functional_model=vec_fm,
    )

    fm_label  = args.functional_model     or "passthrough"
    vec_label = args.vec_functional_model or "passthrough"
    logger.info(
        f"Model patched: functional_model={fm_label}  vec={vec_label}  "
        f"rel_err={args.rel_err}  noise_mode={noise_mode.value}  "
        f"components={args.quantize_components}"
    )

    # ── Print patch diagnostics ───────────────────────────────────────────
    print_patch_diagnostics(model, args.functional_model, args.vec_functional_model)

    # ── Register stats dump on exit ───────────────────────────────────────
    def _dump_stats() -> None:
        unpatch_attn_sdpa(attn_handles)
        unpatch_vector_ops(vec_handles)
        logger.info("=== Matrix path RMSE ===")
        tracker.summary().print(show_layers=False)
        logger.info("=== Vector path RMSE ===")
        vec_tracker.summary().print(show_layers=False)
        if args.stats_output:
            Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
            combined = {
                "matrix": tracker.summary().to_dict(),
                "vector": vec_tracker.summary().to_dict(),
            }
            with open(args.stats_output, "w") as f:
                json.dump(combined, f, indent=2)
            logger.info(f"Stats saved to {args.stats_output}")

    atexit.register(_dump_stats)
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # ── Load norm stats ────────────────────────────────────────────────────
    norm_stats = None
    if args.norm_stats_dir:
        norm_stats = _load_norm_stats(args.norm_stats_dir)
        logger.info(f"Loaded norm stats from {args.norm_stats_dir} "
                     f"(keys: {list(norm_stats.keys())})")
    else:
        # Try checkpoint_dir/assets/droid/ as fallback
        fallback = Path(args.checkpoint_dir) / "assets" / "droid"
        if (fallback / "norm_stats.json").exists():
            norm_stats = _load_norm_stats(str(fallback))
            logger.info(f"Loaded norm stats from {fallback}")
        else:
            logger.warning(
                "No --norm-stats-dir provided and no norm_stats.json found in "
                f"{fallback}. Running WITHOUT normalization — actions will be wrong!"
            )

    cfg = _get_model_config(args.config)
    use_quantile_norm = getattr(cfg, "pi05", False)
    is_joint_position = "jointpos" in args.config
    logger.info(f"use_quantile_norm={use_quantile_norm}  "
                f"is_joint_position={is_joint_position}")

    # ── Build policy and start WebSocket server ───────────────────────────
    policy = Pi0PyTorchPolicy(
        model=model,
        device=device,
        vec_ctx=vec_ctx,
        norm_stats=norm_stats,
        use_quantile_norm=use_quantile_norm,
        is_joint_position=is_joint_position,
        max_token_len=cfg.max_token_len,
        tokenizer_path=args.tokenizer_path,
    )
    policy.metadata["quant"] = {
        "functional_model":     fm_label,
        "vec_functional_model": vec_label,
        "rel_err":              args.rel_err,
        "noise_mode":           noise_mode.value,
    }

    from openpi.serving import websocket_policy_server
    import socket
    logger.info(f"Starting server on {socket.gethostname()}:{args.port}  "
                f"(matrix={fm_label}, vector={vec_label}, rel_err={args.rel_err})")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve PI0Pytorch with matmul quantization over WebSocket."
    )
    p.add_argument("--openpi-dir", default=None,
                   help="Path to the openpi repository root (default: ../openpi relative to this file)")
    p.add_argument("--config", default="pi05_droid_jointpos_polaris",
                   help="Training config name (used to look up architecture params)")
    p.add_argument("--checkpoint-dir", default="",
                   help="Directory containing model.safetensors (or gs:// path with a warning)")
    p.add_argument("--norm-stats-dir", default=None,
                   help="Directory containing norm_stats.json "
                        "(default: tries <checkpoint-dir>/assets/droid/)")
    p.add_argument("--tokenizer-path", default=None,
                   help="Path to paligemma_tokenizer.model "
                        "(default: auto-detect from ~/Desktop or ~/.cache/openpi/)")
    p.add_argument("--port",  type=int, default=8003)
    p.add_argument("--gpu",   type=int, default=0,
                   help="CUDA device index (-1 for CPU)")

    # Simulation paths (both optional; omitting both gives BF16 passthrough)
    p.add_argument("--functional-model", default=None, metavar="NAME",
                   help=f"Hardware-accurate matrix functional model. "
                        f"Available: {list_functional_models()}")
    p.add_argument("--vec-functional-model", default=None,
                   choices=["vector"],
                   help="Route all vector ops through VectorRTLFunctions. "
                        "Currently the only option is 'vector'.")
    p.add_argument(
        "--quantize-components",
        nargs="+",
        default=[g.value for g in ALL_GROUPS],
        choices=[g.value for g in ALL_GROUPS],
        metavar="COMPONENT",
        help=(
            "Which model components to quantize (default: all three). "
            "Choices: vision  transformer  action_head. "
            "  vision      — SigLIP ViT encoder "
            "  transformer — PaliGemma LM + action expert (co-attention coupled) "
            "  action_head — action/state/time projection MLPs "
            "Example: --quantize-components transformer action_head"
        ),
    )

    # Noise injection
    p.add_argument("--rel-err", type=float, default=0.0,
                   help="Relative-error noise fraction injected into each matmul output "
                        "(e.g. 0.01 = 1%%). 0 = disabled.")
    p.add_argument("--noise-mode", default="uniform",
                   choices=[m.value for m in NoiseMode],
                   help="Noise distribution: "
                        "'uniform' = ±rel_err * |y| with random sign (two-point / Rademacher), "
                        "'laplace' = Laplace(0, rel_err) * |y| (better fit for FP8 quant error — "
                        "empirically heavy-tailed with excess kurtosis ~3-60, see "
                        "experiments/test_error_distribution.py)")

    # Output
    p.add_argument("--stats-output", default=None,
                   help="Write JSON RMSE stats here on exit")
    p.add_argument("--list-layers", action="store_true",
                   help="Print linear layer inventory and exit")
    return p.parse_args()


if __name__ == "__main__":
    main()
