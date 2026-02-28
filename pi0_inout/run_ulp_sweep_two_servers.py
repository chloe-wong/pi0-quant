"""
run_ulp_sweep_two_servers.py
----------------------------
Automate an ULP-noise sweep using two policy servers:

  - base server, already running (default port 8000)
  - quantized server, optionally (re)started per sweep step with --ulp-n = 1,2,3,... (default port 8002)

For each n:
  - query both servers on the same set of observations
  - compute action RMSE
  - stop when RMSE >= threshold (default 0.4)
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Optional

import numpy as np
import torch

# Allow running as a script: ensure repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openpi_client import websocket_client_policy as _ws

from pi0_inout.quant_types import QuantFormat


def _random_observation_droid(rng: np.random.Generator) -> dict:
    return {
        "observation/exterior_image_1_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": rng.random(7, dtype=np.float32),
        "observation/gripper_position": rng.random(1, dtype=np.float32),
        "prompt": "Grab the object"
    }


def _with_fixed_pi0_noise(
    obs: dict,
    *,
    rng: np.random.Generator,
    action_shape: tuple[int, ...],
) -> dict:
    out = dict(obs)
    out["pi0_noise"] = rng.standard_normal(action_shape).astype(np.float32)
    return out


@dataclass(frozen=True)
class Metrics:
    rmse: float


def _to_actions_tensor(resp: dict) -> torch.Tensor:
    a = resp["actions"]
    t = torch.from_numpy(np.asarray(a).copy()).float()
    return t.reshape(-1)


def _metrics(base: list[torch.Tensor], quantized: list[torch.Tensor]) -> Metrics:
    r = torch.cat(base, dim=0)
    n = torch.cat(quantized, dim=0)
    diff = (r - n).abs()

    rmse = math.sqrt(float(diff.pow(2).mean().item()))
    return Metrics(rmse=rmse)


def _wait_until_ready(policy: _ws.WebsocketClientPolicy, obs: dict, *, timeout_s: float) -> None:
    t0 = time.time()
    last_err: Optional[BaseException] = None
    while True:
        try:
            policy.infer(obs)
            return
        except BaseException as e:
            last_err = e
            if time.time() - t0 >= timeout_s:
                raise RuntimeError(f"Server not ready after {timeout_s:.1f}s") from last_err
            time.sleep(0.25)


def _quant_cfg(policy: _ws.WebsocketClientPolicy) -> dict[str, Any]:
    """
    Best-effort extraction of quantization config from server metadata.
    `serve_quant.py` sends a top-level 'quant' dict on connect.
    """
    try:
        md = policy.get_server_metadata() or {}
    except Exception:
        return {}
    q = md.get("quant", {})
    return q if isinstance(q, dict) else {}


def _argv_has(argv: list[str], flag: str) -> bool:
    return flag in argv


def _argv_set_kv(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            argv[i + 1] = value
        else:
            argv.append(value)
        return
    argv.extend([flag, value])


def _replace_placeholders(template: str, values: dict[str, object]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _stop_proc_tree(proc: subprocess.Popen) -> None:
    """
    Stop a subprocess started with setsid() by killing its process group.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=15)


def _pids_listening_on_port(port: int) -> set[int]:
    """
    Best-effort: find PIDs that have a TCP LISTEN socket bound to `port`.

    Linux-only implementation without external deps (no psutil/lsof).
    """

    def _listen_inodes_from(path: str) -> set[str]:
        inodes: set[str] = set()
        with open(path, "r", encoding="utf-8") as f:
            next(f, None)  # header
            for line in f:
                parts = line.split()
                if len(parts) < 10:
                    continue
                local_address = parts[1]  # "0100007F:1F41"
                st = parts[3]  # "0A" is LISTEN
                inode = parts[9]
                if st != "0A":
                    continue
                try:
                    _ip_hex, port_hex = local_address.split(":")
                    p = int(port_hex, 16)
                except Exception:
                    continue
                if p == port:
                    inodes.add(inode)
        return inodes

    inodes: set[str] = set()
    with suppress(FileNotFoundError, PermissionError):
        inodes |= _listen_inodes_from("/proc/net/tcp")
    with suppress(FileNotFoundError, PermissionError):
        inodes |= _listen_inodes_from("/proc/net/tcp6")
    if not inodes:
        return set()

    pids: set[int] = set()
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        fd_dir = f"/proc/{pid_str}/fd"
        try:
            fds = os.listdir(fd_dir)
        except (FileNotFoundError, PermissionError):
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inode = target[len("socket:[") : -1]
                if inode in inodes:
                    pids.add(pid)
                    break
    return pids


def _kill_listeners_on_port(port: int, *, timeout_s: float = 10.0) -> None:
    """
    Best-effort: terminate any processes currently LISTENing on `port`.

    This is useful when you keep a quantized server running manually before a sweep:
    without freeing the port, the sweep may keep querying the old server while the
    newly launched one is still loading.
    """
    pids = _pids_listening_on_port(port)
    if not pids:
        return

    for pid in pids:
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not _pids_listening_on_port(port):
            return
        time.sleep(0.1)

    for pid in _pids_listening_on_port(port):
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def _start_quantized_server(
    quantized_server_cmd_template: str,
    *,
    ulp_n: int,
    quantized_port: int,
    defaults: dict[str, Any],
    stdout: Optional[IO[str]] = None,
) -> subprocess.Popen:
    """
    Start the quantized server for the given ulp_n.

    The template is a shell-ish command string. If it contains '{ulp_n}', it will be replaced.
    Otherwise '--ulp-n <n>' will be appended.
    """
    cmd_str = _replace_placeholders(
        quantized_server_cmd_template,
        {
            "ulp_n": ulp_n,
            "quantized_port": quantized_port,
            "input_fmt": defaults.get("input_fmt"),
            "output_fmt": defaults.get("output_fmt"),
            "ulp_fmt": defaults.get("ulp_fmt"),
            "mat_in_fmt": defaults.get("mat_in_fmt"),
            "mat_out_fmt": defaults.get("mat_out_fmt"),
            "vec_out_fmt": defaults.get("vec_out_fmt"),
        },
    )
    argv = shlex.split(cmd_str)

    # Always enforce the sweep value + port.
    _argv_set_kv(argv, "--ulp-n", str(ulp_n))
    _argv_set_kv(argv, "--port", str(quantized_port))

    # If the template omits these, auto-fill from defaults (usually base server).
    if defaults.get("use_matvec"):
        if not _argv_has(argv, "--use-matvec"):
            argv.append("--use-matvec")
        if defaults.get("mat_in_fmt") and not _argv_has(argv, "--mat-in-fmt"):
            _argv_set_kv(argv, "--mat-in-fmt", str(defaults["mat_in_fmt"]))
        if defaults.get("mat_out_fmt") and not _argv_has(argv, "--mat-out-fmt"):
            _argv_set_kv(argv, "--mat-out-fmt", str(defaults["mat_out_fmt"]))
        if defaults.get("vec_out_fmt") and not _argv_has(argv, "--vec-out-fmt"):
            _argv_set_kv(argv, "--vec-out-fmt", str(defaults["vec_out_fmt"]))
    else:
        if defaults.get("input_fmt") and not _argv_has(argv, "--input-fmt"):
            _argv_set_kv(argv, "--input-fmt", str(defaults["input_fmt"]))
        if defaults.get("output_fmt") and not _argv_has(argv, "--output-fmt"):
            _argv_set_kv(argv, "--output-fmt", str(defaults["output_fmt"]))

    if defaults.get("ulp_fmt") and not _argv_has(argv, "--ulp-fmt"):
        _argv_set_kv(argv, "--ulp-fmt", str(defaults["ulp_fmt"]))

    return subprocess.Popen(
        argv,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,  # new process group so we can kill the full tree
    )


def _timestamp_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _open_step_log(*, log_dir: Path, tag: str) -> IO[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{tag}.log"
    return path.open("w", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-host", default="127.0.0.1")
    p.add_argument("--base-port", type=int, default=8000)
    p.add_argument("--quantized-host", default="127.0.0.1")
    p.add_argument("--quantized-port", type=int, default=8002)

    p.add_argument("--n-obs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ulp-fmt",
        default=None,
        choices=[f.value for f in QuantFormat],
        help="Format used to compute ULP(base) for reporting. If unset, inferred from base server metadata (ulp_fmt → output_fmt → bfloat16).",
    )
    p.add_argument("--rmse-threshold", type=float, default=0.4)
    p.add_argument("--start-ulp-n", type=int, default=1,
                   help="Starting ULP level for the sweep (inclusive).")
    p.add_argument("--ulp-step", type=int, default=1,
                   help="Increment for ulp_n each sweep step (e.g. 1 → 1,2,3...; 5 → 5,10,15...).")
    p.add_argument("--max-ulp-n", type=int, default=32)
    p.add_argument("--ready-timeout-s", type=float, default=60.0)
    p.add_argument("--use-fixed-pi0-noise", action="store_true",
                   help="If set, inject deterministic obs['pi0_noise'] (requires server to consume it)")
    p.add_argument("--log-dir", default=str(_REPO_ROOT / "ulp_sweep_logs"),
                   help="Directory to write per-step log files (created if missing). Relative paths are rooted at repo root.")

    p.add_argument(
        "--quantized-server-cmd",
        default=None,
        help=(
            "If set, the script will (re)start the quantized server for each ulp_n. "
            "You may use placeholders like '{ulp_n}', '{quantized_port}', '{input_fmt}', '{output_fmt}', '{ulp_fmt}', "
            "'{mat_in_fmt}', '{mat_out_fmt}', '{vec_out_fmt}'. "
            "If you omit format/ulp flags entirely, they will be auto-filled from the base server's metadata (falling back to bfloat16)."
        ),
    )

    p.add_argument(
        "--dynamic-ulp",
        action="store_true",
        help="If set, do NOT restart the quantized server. Instead, send a control key '__quant_control__' to update ULP at runtime (requires updated serve_quant.py).",
    )
    p.add_argument(
        "--kill-existing-quantized-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), kill any existing process listening on --quantized-port "
            "before starting a new quantized server step. This is recommended if you "
            "start a quantized server manually before running the sweep."
        ),
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    base = _ws.WebsocketClientPolicy(host=args.base_host, port=args.base_port)
    base_q = _quant_cfg(base)
    eval_ulp_fmt = QuantFormat(args.ulp_fmt or str(base_q.get("ulp_fmt") or base_q.get("output_fmt") or "bfloat16"))

    start_ulp_n = max(0, int(args.start_ulp_n))
    ulp_step = max(1, int(args.ulp_step))
    max_ulp_n = max(0, int(args.max_ulp_n))

    log_root = Path(args.log_dir)
    if not log_root.is_absolute():
        log_root = _REPO_ROOT / log_root
    run_dir = (log_root / f"run-{_timestamp_tag()}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = (run_dir / "run.log").open("w", encoding="utf-8")
    run_log.write(f"# run_dir={run_dir}\n")
    run_log.write(f"# base={args.base_host}:{args.base_port}  quantized={args.quantized_host}:{args.quantized_port}\n")
    run_log.write(f"# ulp_fmt(eval)={eval_ulp_fmt.value}  n_obs={args.n_obs}  seed={args.seed}\n")
    run_log.write(f"# sweep: start_ulp_n={start_ulp_n}  ulp_step={ulp_step}  max_ulp_n={max_ulp_n}\n")
    run_log.write(f"# use_fixed_pi0_noise={bool(args.use_fixed_pi0_noise)}\n")
    run_log.write(f"# quantized_server_cmd={args.quantized_server_cmd}\n\n")
    if base_q:
        run_log.write(f"# base_quant={base_q}\n\n")
    run_log.flush()

    # Warmup reference and get action shape for optional deterministic pi0_noise.
    obs0 = _random_observation_droid(rng)
    warm = base.infer(obs0)
    action_shape = tuple(np.asarray(warm["actions"]).shape)

    # Pre-generate observations so each ulp_n sees identical inputs.
    observations: list[dict] = []
    for _ in range(args.n_obs):
        obs = _random_observation_droid(rng)
        if args.use_fixed_pi0_noise:
            obs = _with_fixed_pi0_noise(obs, rng=rng, action_shape=action_shape)
        observations.append(obs)

    quantized_proc: Optional[subprocess.Popen] = None
    quantized_log_fh: Optional[IO[str]] = None
    try:
        # If using dynamic updates and a command template is provided, start the quantized server once at ulp_n=0.
        if args.dynamic_ulp and args.quantized_server_cmd:
            if args.kill_existing_quantized_server:
                _kill_listeners_on_port(args.quantized_port)
            step_tag = "quantized_server_start"
            quantized_log_fh = _open_step_log(log_dir=run_dir, tag=step_tag)
            quantized_log_fh.write(f"# {step_tag}\n")
            quantized_log_fh.write(f"# base={args.base_host}:{args.base_port}  quantized={args.quantized_host}:{args.quantized_port}\n")
            quantized_log_fh.write(f"# ulp_fmt(eval)={eval_ulp_fmt.value}  n_obs={args.n_obs}  seed={args.seed}\n")
            quantized_log_fh.write(f"# quantized_server_cmd={args.quantized_server_cmd}\n\n")
            quantized_log_fh.flush()

            defaults = {
                "input_fmt": base_q.get("input_fmt", "bfloat16"),
                "output_fmt": base_q.get("output_fmt", "bfloat16"),
                "use_matvec": bool(base_q.get("use_matvec", False)),
                "mat_in_fmt": base_q.get("mat_in_fmt", "bfloat16"),
                "mat_out_fmt": base_q.get("mat_out_fmt", "bfloat16"),
                "vec_out_fmt": base_q.get("vec_out_fmt", "bfloat16"),
                "ulp_fmt": base_q.get("ulp_fmt") or base_q.get("output_fmt") or "bfloat16",
            }
            quantized_proc = _start_quantized_server(
                args.quantized_server_cmd,
                ulp_n=0,
                quantized_port=args.quantized_port,
                defaults=defaults,
                stdout=quantized_log_fh,
            )

        for n in range(start_ulp_n, max_ulp_n + 1, ulp_step):
            if args.quantized_server_cmd is None:
                # Evaluate whatever is already running on quantized-port (single step).
                n = -1
            elif not args.dynamic_ulp:
                # Restart-per-step mode.
                if quantized_proc is not None and quantized_proc.poll() is None:
                    _stop_proc_tree(quantized_proc)
                if args.kill_existing_quantized_server:
                    _kill_listeners_on_port(args.quantized_port)
                if quantized_log_fh is not None:
                    quantized_log_fh.close()
                    quantized_log_fh = None

                step_tag = f"ulp_n={n}"
                quantized_log_fh = _open_step_log(log_dir=run_dir, tag=step_tag)
                quantized_log_fh.write(f"# {step_tag}\n")
                quantized_log_fh.write(f"# base={args.base_host}:{args.base_port}  quantized={args.quantized_host}:{args.quantized_port}\n")
                quantized_log_fh.write(f"# ulp_fmt(eval)={eval_ulp_fmt.value}  n_obs={args.n_obs}  seed={args.seed}\n")
                quantized_log_fh.write(f"# use_fixed_pi0_noise={bool(args.use_fixed_pi0_noise)}\n")
                quantized_log_fh.write(f"# quantized_server_cmd={args.quantized_server_cmd}\n\n")
                if base_q:
                    quantized_log_fh.write(f"# base_quant={base_q}\n\n")
                quantized_log_fh.flush()

                defaults = {
                    "input_fmt": base_q.get("input_fmt", "bfloat16"),
                    "output_fmt": base_q.get("output_fmt", "bfloat16"),
                    "use_matvec": bool(base_q.get("use_matvec", False)),
                    "mat_in_fmt": base_q.get("mat_in_fmt", "bfloat16"),
                    "mat_out_fmt": base_q.get("mat_out_fmt", "bfloat16"),
                    "vec_out_fmt": base_q.get("vec_out_fmt", "bfloat16"),
                    "ulp_fmt": base_q.get("ulp_fmt") or base_q.get("output_fmt") or "bfloat16",
                }
                quantized_proc = _start_quantized_server(
                    args.quantized_server_cmd,
                    ulp_n=n,
                    quantized_port=args.quantized_port,
                    defaults=defaults,
                    stdout=quantized_log_fh,
                )

            if args.quantized_server_cmd is None and quantized_log_fh is None:
                step_tag = "ulp_n=as-is"
                quantized_log_fh = _open_step_log(log_dir=run_dir, tag=step_tag)
                quantized_log_fh.write(f"# {step_tag}\n")
                quantized_log_fh.write(f"# base={args.base_host}:{args.base_port}  quantized={args.quantized_host}:{args.quantized_port}\n")
                quantized_log_fh.write(f"# ulp_fmt(eval)={eval_ulp_fmt.value}  n_obs={args.n_obs}  seed={args.seed}\n")
                quantized_log_fh.write(f"# use_fixed_pi0_noise={bool(args.use_fixed_pi0_noise)}\n")
                quantized_log_fh.write(f"# quantized_server_cmd=None (evaluate existing server)\n\n")
                if base_q:
                    quantized_log_fh.write(f"# base_quant={base_q}\n\n")
                quantized_log_fh.flush()

            quantized = _ws.WebsocketClientPolicy(host=args.quantized_host, port=args.quantized_port)
            _wait_until_ready(quantized, obs0, timeout_s=args.ready_timeout_s)

            base_actions: list[torch.Tensor] = []
            quantized_actions: list[torch.Tensor] = []
            for i, obs in enumerate(observations):
                base_actions.append(_to_actions_tensor(base.infer(obs)))
                if args.dynamic_ulp and n >= 0:
                    # Update server-side ULP config (best-effort) on the first request of each step.
                    o = dict(obs)
                    if i == 0:
                        o["__quant_control__"] = {
                            "ulp_n": n,
                            "ulp_fmt": eval_ulp_fmt.value,
                        }
                    quantized_actions.append(_to_actions_tensor(quantized.infer(o)))
                else:
                    quantized_actions.append(_to_actions_tensor(quantized.infer(obs)))

            m = _metrics(base_actions, quantized_actions)
            tag = f"ulp_n={n}" if n >= 0 else "ulp_n=<as-is>"
            line = f"{tag:14s}  rmse={m.rmse:.4e}"
            print(line)
            run_log.write(line + "\n")
            run_log.flush()
            if quantized_log_fh is not None:
                quantized_log_fh.write(f"\n# result: {line}\n")
                quantized_log_fh.flush()

            if n >= 0 and m.rmse >= args.rmse_threshold:
                print(f"STOP: threshold violated (rmse >= {args.rmse_threshold})")
                break

            if args.quantized_server_cmd is None:
                break
    finally:
        run_log.close()
        if quantized_proc is not None and quantized_proc.poll() is None:
            _stop_proc_tree(quantized_proc)
        if quantized_log_fh is not None:
            quantized_log_fh.close()


if __name__ == "__main__":
    main()

