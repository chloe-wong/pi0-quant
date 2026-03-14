"""
run_ulp_server_experiment.py
----------------------------
RMSE runner on two live policy servers:

  - base server: baseline 
  - quantized server: same weights/config but with ULP injection

Metrics:
  - action RMSE

You typically run it by:
  1) Start base server on port P0 (e.g. 8000) with ulp_n=0
  2) Start quantized server on port P1 (e.g. 8001) with ulp_n>=1
  3) Run this script pointing at both ports
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

from openpi_client import websocket_client_policy as _ws

# Allow running as a script: ensure repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _random_observation_droid(rng: np.random.Generator) -> dict:
    return {
        "observation/exterior_image_1_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": rng.random(7, dtype=np.float32),
        "observation/gripper_position": rng.random(1, dtype=np.float32),
        "prompt": "do something",
    }


@dataclass(frozen=True)
class Metrics:
    rmse: float


def _to_actions_tensor(resp: dict) -> torch.Tensor:
    # serve_quant returns {"actions": np.ndarray} with shape [H, D] or [1,H,D]
    a = resp["actions"]
    # Copy to avoid "non-writable" numpy -> torch warning.
    t = torch.from_numpy(np.asarray(a).copy()).float()
    return t.reshape(-1)


def _metrics(base: list[torch.Tensor], quantized: list[torch.Tensor]) -> Metrics:
    b = torch.cat(base, dim=0)
    q = torch.cat(quantized, dim=0)
    diff = (b - q).abs()

    rmse = math.sqrt(float(diff.pow(2).mean().item()))
    return Metrics(rmse=rmse)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-host", "--exact-host", default="127.0.0.1")
    p.add_argument("--base-port", "--exact-port", type=int, default=8000)
    p.add_argument("--quantized-host", "--noisy-host", default="127.0.0.1")
    p.add_argument("--quantized-port", "--noisy-port", type=int, default=8001)
    p.add_argument("--n-obs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # These are client-side labels only (we don't control server injection here).
    # Use them to make the printed RMSE line self-describing.
    p.add_argument("--base-input-fmt", "--exact-input-fmt", default=None)
    p.add_argument("--base-output-fmt", "--exact-output-fmt", default=None)
    p.add_argument("--base-ulp-n", "--exact-ulp-n", type=int, default=None)
    p.add_argument("--base-ulp-fmt", "--exact-ulp-fmt", default=None)
    p.add_argument("--quantized-input-fmt", "--noisy-input-fmt", default=None)
    p.add_argument("--quantized-output-fmt", "--noisy-output-fmt", default=None)
    p.add_argument("--ulp-n", "--noisy-ulp-n", "--quantized-ulp-n", type=int, default=None)
    p.add_argument("--ulp-fmt", "--noisy-ulp-fmt", "--quantized-ulp-fmt", default=None)
    p.add_argument("--rmse-threshold", type=float, default=0.4)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    base = _ws.WebsocketClientPolicy(host=args.base_host, port=args.base_port)
    quantized = _ws.WebsocketClientPolicy(host=args.quantized_host, port=args.quantized_port)

    # Model load + determine action shape from server metadata.
    obs0 = _random_observation_droid(rng)
    warm = base.infer(obs0)
    quantized.infer(obs0)
    action_horizon = np.asarray(warm["actions"]).shape[0]
    base_md = base.get_server_metadata() or {}
    action_dim = int(base_md.get("action_dim", 32))

    base_actions = []
    quantized_actions = []

    for _ in range(args.n_obs):
        obs = _random_observation_droid(rng)
        # Deterministic diffusion noise so identical servers match.
        obs["pi0_noise"] = rng.standard_normal((action_horizon, action_dim)).astype(np.float32)

        b = _to_actions_tensor(base.infer(obs))
        q = _to_actions_tensor(quantized.infer(obs))
        base_actions.append(b)
        quantized_actions.append(q)

    m = _metrics(base_actions, quantized_actions)
    base_label = (
        f"in={args.base_input_fmt} out={args.base_output_fmt} ulp_n={args.base_ulp_n} ulp_fmt={args.base_ulp_fmt}"
    )
    quantized_label = (
        f"in={args.quantized_input_fmt} out={args.quantized_output_fmt} "
        f"ulp_n={args.ulp_n} ulp_fmt={args.ulp_fmt}"
    )
    print(
        f"rmse={m.rmse:.4e}  "
        f"base[{base_label}]  quantized[{quantized_label}]"
    )
    if m.rmse >= args.rmse_threshold:
        print("THRESHOLD VIOLATED")


if __name__ == "__main__":
    main()

