"""
serve_quant.py
--------------
Drop-in replacement for openpi/scripts/serve_policy.py that serves
PI0Pytorch with configurable matmul input/output quantization.

How it works
------------
1. Adds openpi/src and openpi-client/src to sys.path.
2. Injects lightweight Python stubs for the three openpi files that import
   JAX at module level (gemma config, lora config, array_typing, image_tools).
   This lets PI0Pytorch be imported in the pi0 conda env (torch-only).
3. Instantiates PI0Pytorch from a hardcoded config dict for the known
   training configs (no JAX config system required).
4. Loads weights from a local safetensors checkpoint if available.
5. Patches every nn.Linear with QuantLinear (input_fmt / output_fmt).
6. Serves via the openpi WebSocket protocol (msgpack + websockets).
7. On SIGTERM/SIGINT, writes per-layer RMSE stats to --stats-output.

Quantization semantics (unchanged from quant_linear.py)
---------------------------------------------------------
For each nn.Linear:
    x_q  = quant(x,    input_fmt)
    W_q  = quant(W,    input_fmt)
    b_q  = quant(bias, input_fmt)          # bias loaded in input_fmt
    y    = F.linear(x_q, W_q, b_q)        # float32 accumulation
    out  = quant(y, output_fmt)            # single output quantization

FLOAT32/FLOAT32 is the identity — zero RMSE baseline.

Usage
-----
    python serve_quant.py \\
        --openpi-dir /path/to/openpi \\
        --checkpoint-dir /path/to/model.safetensors_dir \\
        --config pi05_droid_jointpos_polaris \\
        --input-fmt float8_e4m3 --output-fmt float16 \\
        --port 8003 --gpu 0
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import atexit
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
from pi0_inout.quant_types import QuantFormat, TORCH_DTYPE, FORMAT_BITS
from pi0_inout.model_patcher import patch_model, list_linear_layers
from pi0_inout.quant_linear import QuantLinear
from pi0_inout.stats_tracker import StatsTracker
from pi0_inout.ulp_noise import UlpNoiseConfig, ulp_step


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
    model.sample_actions = model.sample_actions.__wrapped__
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

def print_quant_diagnostics(
    model: nn.Module,
    input_fmt: QuantFormat,
    output_fmt: QuantFormat,
) -> None:
    """
    Print model structure and memory analysis after quantization patching.
    Shows that this is SIMULATED quantization (weights stay in original dtype).
    """
    print("\n" + "=" * 80)
    print("QUANTIZATION DIAGNOSTICS")
    print("=" * 80)

    # 1. Layer inventory: count QuantLinear vs plain nn.Linear
    n_quant = 0
    n_plain = 0
    total_params = 0
    quant_params = 0
    sample_layer = None

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            n_quant += 1
            n_params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            quant_params += n_params
            total_params += n_params
            if sample_layer is None:
                sample_layer = (name, module)
        elif type(module) is nn.Linear:
            n_plain += 1
            n_params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            total_params += n_params

    print(f"\n[1] Layer counts:")
    print(f"    QuantLinear layers: {n_quant}")
    print(f"    Plain nn.Linear:   {n_plain}")
    print(f"    Total parameters:  {total_params:,}")
    print(f"    Quantized params:  {quant_params:,}")

    # 2. Print a few representative QuantLinear layers
    print(f"\n[2] Sample QuantLinear layers (first 5):")
    print(f"    {'Name':<60s}  {'Weight dtype':<12s}  input_fmt    output_fmt")
    print("    " + "-" * 110)
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            print(f"    {name:<60s}  {str(module.weight.dtype):<12s}  "
                  f"{module.input_fmt.value:<12s} {module.output_fmt.value}")
            count += 1
            if count >= 5:
                print(f"    ... ({n_quant - 5} more)")
                break

    # 3. Memory analysis
    input_bits = FORMAT_BITS[input_fmt]["total"]
    bf16_bits = 16

    # Actual memory used (weights stay in original dtype)
    actual_bytes = 0
    for p in model.parameters():
        actual_bytes += p.numel() * p.element_size()

    # Hypothetical memory if weights were ACTUALLY stored in input_fmt
    hypothetical_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            n = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            hypothetical_bytes += n * (input_bits // 8)
        elif type(module) is nn.Linear:
            for p in module.parameters():
                hypothetical_bytes += p.numel() * p.element_size()
    # Add non-linear params at their actual size
    linear_param_ids = set()
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, nn.Linear)):
            for p in module.parameters():
                linear_param_ids.add(id(p))
    for p in model.parameters():
        if id(p) not in linear_param_ids:
            hypothetical_bytes += p.numel() * p.element_size()

    bf16_bytes = total_params * (bf16_bits // 8)

    print(f"\n[3] Memory analysis:")
    print(f"    Actual GPU memory (weights in original dtype): {actual_bytes / 1e9:.3f} GB")
    print(f"    Hypothetical if stored as {input_fmt.value}:   {hypothetical_bytes / 1e9:.3f} GB")
    print(f"    Reference bf16 size (linear params only):      {bf16_bytes / 1e9:.3f} GB")
    print(f"    Compression ratio (hypothetical vs bf16):      {bf16_bytes / max(hypothetical_bytes, 1):.2f}x")
    print(f"    NOTE: Actual memory is UNCHANGED — this is simulated quantization.")
    print(f"          Weights are stored in {sample_layer[1].weight.dtype if sample_layer else 'N/A'}, "
          f"cast through {input_fmt.value} at runtime.")

    # 4. Verify quantization actually changes values (pick one weight tensor)
    if sample_layer is not None:
        name, layer = sample_layer
        w = layer.weight.detach().float()
        target_dtype = TORCH_DTYPE[input_fmt]
        w_quant = w.to(target_dtype).to(w.dtype)
        diff = (w - w_quant).abs()
        n_changed = (diff > 0).sum().item()
        n_total = w.numel()

        print(f"\n[4] Quantization verification (layer: {name}):")
        print(f"    Weight shape:        {tuple(w.shape)}")
        print(f"    Weight dtype:        {layer.weight.dtype}")
        print(f"    Target quant dtype:  {target_dtype}")
        print(f"    Values changed:      {n_changed:,} / {n_total:,} "
              f"({100 * n_changed / n_total:.1f}%)")
        print(f"    Max abs difference:  {diff.max().item():.6e}")
        print(f"    Mean abs difference: {diff.mean().item():.6e}")
        if n_changed == 0 and input_fmt != QuantFormat.BFLOAT16:
            print(f"    WARNING: No values changed! Quantization may not be working.")
        elif input_fmt == QuantFormat.BFLOAT16:
            print(f"    OK: BFLOAT16 passthrough — zero difference expected for bf16 models.")
        else:
            print(f"    OK: Quantization is actively rounding values.")

    print("\n" + "=" * 80 + "\n")


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
        norm_stats: dict | None = None,
        use_quantile_norm: bool = False,
        is_joint_position: bool = False,
        max_token_len: int = 48,
        tokenizer_path: str | None = None,
    ) -> None:
        self.model    = model
        self.device   = device
        self.metadata = {"model": "PI0Pytorch", "quantized": True}
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

        with torch.no_grad():
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

    # ── Parse formats and patch model ────────────────────────────────────
    input_fmt  = QuantFormat(args.input_fmt)
    output_fmt = QuantFormat(args.output_fmt)

    # ULP noise injection into Linear matmul outputs
    ulp_noise = None
    if args.ulp_n and args.ulp_n > 0:
        ulp_fmt_val = args.ulp_fmt if args.ulp_fmt is not None else args.output_fmt
        ulp_noise = UlpNoiseConfig(
            n_ulp=args.ulp_n,
            ulp_fmt=QuantFormat(ulp_fmt_val),
        )
        _ulp_at_1 = float(ulp_step(torch.tensor(1.0), QuantFormat(ulp_fmt_val)).item())
        logger.info(f"ULP noise: input_fmt={input_fmt.value}  output_fmt={output_fmt.value}  ulp_n={args.ulp_n}  ulp_fmt={ulp_fmt_val}  ulp={_ulp_at_1:.4e}")

    tracker = StatsTracker()
    patch_model(
        model=model,
        input_fmt=input_fmt,
        output_fmt=output_fmt,
        tracker=tracker,
        ulp_noise=ulp_noise,
        verbose=False,
    )
    logger.info(f"Model patched: input_fmt={input_fmt.value}  output_fmt={output_fmt.value}")

    # ── Print quantization diagnostics ────────────────────────────────────
    print_quant_diagnostics(model, input_fmt, output_fmt)

    # ── Register stats dump on exit ───────────────────────────────────────
    def _dump_stats() -> None:
        logger.info("=== Quantization RMSE Report ===")
        report = tracker.summary()
        report.print(show_layers=False)
        if args.stats_output:
            Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.stats_output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
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
        norm_stats=norm_stats,
        use_quantile_norm=use_quantile_norm,
        is_joint_position=is_joint_position,
        max_token_len=cfg.max_token_len,
        tokenizer_path=args.tokenizer_path,
    )

    from openpi.serving import websocket_policy_server
    import socket
    logger.info(f"Starting server on {socket.gethostname()}:{args.port}  "
                f"(input={input_fmt.value}, output={output_fmt.value})")

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

    # Quantization
    p.add_argument("--input-fmt",  default="bfloat16",
                   choices=[f.value for f in QuantFormat])
    p.add_argument("--output-fmt", default="bfloat16",
                   choices=[f.value for f in QuantFormat])

    # Optional: matrix/vector separate formats (constraint: vec_in == mat_out)
    p.add_argument("--use-matvec", action="store_true",
                   help="Use matrix/vector IO formats for nn.Linear instead of simple input/output formats")
    p.add_argument("--mat-in-fmt", default="bfloat16",
                   choices=[f.value for f in QuantFormat],
                   help="Matrix input format (activation+weight)")
    p.add_argument("--mat-out-fmt", default="bfloat16",
                   choices=[f.value for f in QuantFormat],
                   help="Matrix output format (matmul output before bias add)")
    p.add_argument("--vec-out-fmt", default="bfloat16",
                   choices=[f.value for f in QuantFormat],
                   help="Vector output format (final output after bias add)")

    # Optional: ULP noise injection into matmul outputs
    p.add_argument("--ulp-n", type=int, default=0,
                   help="Inject +/- n ULP noise into each Linear matmul output (0 disables)")
    p.add_argument("--ulp-fmt", default=None,
                   choices=[f.value for f in QuantFormat],
                   help="Format whose ULP grid defines the step size (default: same as --output-fmt)")

    # Output
    p.add_argument("--stats-output", default=None,
                   help="Write JSON RMSE stats here on exit")
    p.add_argument("--list-layers", action="store_true",
                   help="Print linear layer inventory and exit")
    return p.parse_args()


if __name__ == "__main__":
    main()
