"""
experiments/extrabits_sweep.py
--------------------------------
In-process sweep of extraBits (0..N) using ipt_numba_exp.
Measures RMSE between base BF16 model output and ipt_numba_exp quantized output
for each extraBits value.

No server subprocesses — loads the model once, patches/unpatches per step.
The kernel (ipt_numba_exp/_numba_kernels.py) is parametric so different
extraBits values produce genuinely different accumulator precision.

Usage
-----
    python experiments/extrabits_sweep.py \\
        --checkpoint-dir /path/to/checkpoint \\
        --config pi05_droid_jointpos_polaris \\
        --gpu 0 \\
        --min-extra 0 --max-extra 17 \\
        --n-obs 4 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from pi0_inout.serve_quant import _get_model_config, load_pi0_pytorch
from pi0_inout.model_patcher import patch_model, unpatch_model
from pi0_inout.quant_types import QuantFormat
from pi0_inout.stats_tracker import StatsTracker
from funct_models_ipt.ipt_numba_exp.ipt_rtl_linear import IPTLinearRTLFunction

logger = logging.getLogger(__name__)


def _start_heartbeat(tracker: StatsTracker, t0: float, stop_event: threading.Event, interval_s: int = 30) -> threading.Thread:
    def _loop():
        while not stop_event.wait(timeout=interval_s):
            elapsed = time.monotonic() - t0
            print(
                f"  [heartbeat] elapsed={datetime.timedelta(seconds=int(elapsed))}  "
                f"layer_calls={tracker._seq}",
                flush=True,
            )
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Observation helper (matches run_eval.py)
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


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def _rmse(base: list[torch.Tensor], quantized: list[torch.Tensor]) -> float:
    b = torch.cat([a.reshape(-1) for a in base]).float()
    q = torch.cat([a.reshape(-1) for a in quantized]).float()
    return math.sqrt(float(torch.mean((b - q) ** 2).item()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        force=True,
    )

    p = argparse.ArgumentParser(
        description=(
            "Sweep extraBits (0..N) for ipt_numba_exp and measure RMSE "
            "vs in-process BF16 base model."
        )
    )
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--config", default="pi0_droid_jointpos_polaris")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--min-extra", type=int, default=0)
    p.add_argument("--max-extra", type=int, default=17)
    p.add_argument("--n-obs", type=int, default=4,
                   help="Number of random observations to average RMSE over")
    p.add_argument("--steps", type=int, default=10,
                   help="Diffusion steps per sample_actions call")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir",
                   default=str(_REPO / "experiments" / "results"))
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    cfg = _get_model_config(args.config)
    logger.info("Loading model (%s) on %s ...", args.config, device)
    model = load_pi0_pytorch(args.config, args.checkpoint_dir, device)
    model.eval()

    # ── Build observations ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    observations = [_make_dummy_obs(cfg, device) for _ in range(args.n_obs)]
    logger.info("Built %d observations (seed=%d)", args.n_obs, args.seed)

    # ── Base inference (unpatched BF16) ──────────────────────────────────────
    logger.info("Collecting base actions (unpatched BF16) ...")
    with torch.no_grad():
        base_actions: list[torch.Tensor] = [
            model.sample_actions(str(device), obs, num_steps=args.steps)
            for obs in observations
        ]
    logger.info("Base actions collected.")

    # ── Numba warmup (triggers JIT compilation before first real inference) ──
    logger.info("Warming up Numba kernel ...")
    from funct_models_ipt.ipt_numba_exp._numba_kernels import warmup as _numba_warmup
    _numba_warmup()
    logger.info("Numba kernel ready.")

    # ── Sweep ────────────────────────────────────────────────────────────────
    results: list[tuple[int, float]] = []
    n_steps = args.max_extra - args.min_extra + 1
    logger.info("Starting sweep: extra_bits %d..%d (%d steps)", args.min_extra, args.max_extra, n_steps)
    header = f"{'extra_bits':>11}  {'rmse':>12}  {'time':>8}"
    print(header)
    print("-" * len(header))

    for i, extra_bits in enumerate(range(args.min_extra, args.max_extra + 1)):
        t0 = time.perf_counter()
        logger.info("[%d/%d] extra_bits=%d — patching model ...", i + 1, n_steps, extra_bits)

        # Capture extra_bits in the factory closure via a default arg.
        def factory(in_f: int, out_f: int, eb: int = extra_bits) -> IPTLinearRTLFunction:
            return IPTLinearRTLFunction(extra_bits=eb)

        tracker = StatsTracker()
        patch_model(
            model,
            mx_input_fmt=QuantFormat.BFLOAT16,
            mx_output_fmt=QuantFormat.BFLOAT16,
            functional_model_factory=factory,
            tracker=tracker,
        )
        logger.info("[%d/%d] extra_bits=%d — running inference (%d obs) ...", i + 1, n_steps, extra_bits, args.n_obs)

        stop_heartbeat = threading.Event()
        _start_heartbeat(tracker, t0, stop_heartbeat)

        with torch.no_grad():
            quant_actions: list[torch.Tensor] = [
                model.sample_actions(str(device), obs, num_steps=args.steps)
                for obs in observations
            ]

        stop_heartbeat.set()
        unpatch_model(model)

        rmse = _rmse(base_actions, quant_actions)
        elapsed = time.perf_counter() - t0
        results.append((extra_bits, rmse))
        logger.info("[%d/%d] extra_bits=%d — RMSE=%.4e  (%.1fs)", i + 1, n_steps, extra_bits, rmse, elapsed)
        print(f"{extra_bits:>11}  {rmse:>12.4e}  ({elapsed:.1f}s)")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = results_dir / f"extrabits_sweep_{timestamp}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["extra_bits", "rmse"])
        writer.writerows(results)

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
