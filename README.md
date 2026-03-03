# pi0-inout

Matmul input/output quantization framework for [Pi0Pytorch](https://github.com/Physical-Intelligence/openpi).

Sweeps all 25 `(input_fmt × output_fmt)` combinations across
`{float32, float16, bfloat16, float8_e4m3, float8_e5m2}`, measures
per-layer and per-component RMSE, and optionally runs live sim-eval
benchmarks via the openpi WebSocket protocol.

## What it does

Every `nn.Linear` in the model is replaced in-place with a `QuantLinear`
that simulates reduced-precision matmuls:

```
x_q  = quant(x,    input_fmt)   # activation snapped to input_fmt grid
W_q  = quant(W,    input_fmt)   # weight snapped to input_fmt grid
b_q  = quant(b,    input_fmt)   # bias snapped to input_fmt grid
y    = x_q @ W_q^T + b_q        # accumulated in float32
out  = quant(y,    output_fmt)  # output snapped to output_fmt grid
```

This is **simulated** quantization — the weights are not stored in
reduced precision. The cast-and-back trick (`float32 → target → float32`)
replicates IEEE 754 round-to-nearest-even rounding, giving the same
numerical values that true hardware quantization would produce.

Attention score matmuls (`Q@K^T` and `weights@V`) are fused inside
`F.scaled_dot_product_attention` and are not `nn.Linear` layers.
`QuantAttnContext` handles these by temporarily monkey-patching
`F.scaled_dot_product_attention` for the duration of a forward pass.

RMSE is measured per layer and aggregated across four architectural
components:

| Component | Layers |
|---|---|
| `VISION` | SigLIP ViT encoder (inside PaliGemma) |
| `LANGUAGE` | Gemma language model (inside PaliGemma) |
| `ACTION_EXPERT` | Gemma action-expert transformer |
| `ACTION_HEAD` | Action projection MLPs at the Pi0 root |

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.1 (for `float8_e4m3fn` / `float8_e5m2` dtypes)
- The [openpi](https://github.com/Physical-Intelligence/openpi) repository
  on your Python path (provides `Pi0Pytorch` and the WebSocket server
  infrastructure)
- A Pi0 checkpoint converted to safetensors format (see
  [openpi conversion script](https://github.com/Physical-Intelligence/openpi/blob/main/examples/convert_jax_model_to_pytorch.py))
- `sim-evals` + Isaac Sim only if you want to run `run_benchmark.py`

### Why `_jax_stubs.py` exists

Several openpi source files (`gemma.py`, `lora.py`, `array_typing.py`,
`image_tools.py`) import JAX at module level even though Pi0Pytorch itself
is pure PyTorch. `_jax_stubs.py` injects lightweight pure-Python
replacements into `sys.modules` before those imports happen, so Pi0Pytorch
can be loaded in a PyTorch-only environment with no JAX installation.

Call `_jax_stubs.inject()` once, before any openpi import:

```python
from pi0_inout._jax_stubs import inject
inject()

from openpi.models.pi0_pytorch import Pi0Pytorch  # now works without JAX
```

`serve_quant.py` handles this automatically.

## Install

```bash
git clone https://github.com/chloe-wong/pi0-quant
cd pi0-quant
pip install -e .
```

## Quick start

### Patch and measure RMSE

```python
from pi0_inout import patch_model, unpatch_model, StatsTracker, QuantFormat

# Load Pi0Pytorch however you normally do it
model = ...  # Pi0Pytorch instance

tracker = StatsTracker()
patch_model(
    model,
    input_fmt=QuantFormat.FLOAT8_E4M3,
    output_fmt=QuantFormat.FLOAT16,
    tracker=tracker,
)

# Run inference
with torch.no_grad():
    actions = model.sample_actions(obs)

tracker.summary().print()

# Restore for the next sweep point
unpatch_model(model)
```

### Include attention score quantization

```python
from pi0_inout import patch_model, QuantAttnContext, StatsTracker, QuantFormat

tracker = StatsTracker()
patch_model(model, input_fmt=QuantFormat.FLOAT8_E4M3, output_fmt=QuantFormat.FLOAT16,
            tracker=tracker)

with QuantAttnContext(QuantFormat.FLOAT8_E4M3, QuantFormat.FLOAT16, tracker=tracker):
    actions = model.sample_actions(obs)

tracker.summary().print()
```

### Skip specific components

```python
from pi0_inout import patch_model, Component, QuantFormat

# Quantize everything except the vision tower
patch_model(
    model,
    input_fmt=QuantFormat.FLOAT8_E4M3,
    output_fmt=QuantFormat.FLOAT32,
    skip_components={Component.VISION},
)
```

### Sweep all 25 format pairs programmatically

```python
from pi0_inout import patch_model, unpatch_model, StatsTracker, QuantFormat
from pi0_inout.quant_types import sweep_pairs

for input_fmt, output_fmt in sweep_pairs():
    tracker = StatsTracker()
    patch_model(model, input_fmt=input_fmt, output_fmt=output_fmt, tracker=tracker)

    with torch.no_grad():
        actions = model.sample_actions(obs)

    report = tracker.summary()
    report.print()
    unpatch_model(model)
```

## Serving over WebSocket

`serve_quant.py` is a drop-in replacement for openpi's `serve_policy.py`
that loads Pi0Pytorch with configurable quantization and serves it over
the openpi WebSocket protocol (msgpack + websockets). On shutdown it writes
per-layer RMSE stats to `--stats-output`.

```bash
python pi0_inout/serve_quant.py \
    --openpi-dir /path/to/openpi \
    --checkpoint-dir /path/to/safetensors_checkpoint \
    --config pi05_droid_jointpos_polaris \
    --input-fmt float8_e4m3 \
    --output-fmt float16 \
    --port 8003 \
    --gpu 0
```

## ULP sweep (two servers)

`run_ulp_sweep_two_servers.py` compares a **base server** vs a **quantized server** over the same randomly generated observations, and sweeps `--ulp-n` on the quantized side. Base server and quantized server must be run on **different CUDA_VISIBLE_DEVICES** as each server is quite heavy to run.

### Typical usage

- Start a base server (no ULP noise) on some port
- Run the sweep, letting it (re)launch the quantized (with ULP noise) server each step:

```bash
python -m pi0_inout.run_ulp_sweep_two_servers \
    --base-port 8000 \
    --quantized-port 8001 \
    --start-ulp-n 0 --ulp-step 50 --max-ulp-n 5000 \
    --quantized-server-cmd 'env CUDA_VISIBLE_DEVICES=2 python pi0_inout/serve_quant.py --openpi-dir /path/to/openpi --checkpoint-dir /path/to/checkpoint --config pi0_droid --gpu 0 --input-fmt float8_e5m2 --output-fmt bfloat16 --ulp-fmt bfloat16'
```

If you omit `--quantized-server-cmd`, the script will **evaluate whatever is already running** on `--quantized-port` (single step).

### Options

- **server endpoints**
  - `--base-host` (default `127.0.0.1`)
  - `--base-port` (default `8000`)
  - `--quantized-host` (default `127.0.0.1`)
  - `--quantized-port` (default `8001`)
- **sweep controls**
  - `--start-ulp-n` (default `1`)
  - `--ulp-step` (default `1`)
  - `--max-ulp-n` (default `32`)
  - `--rmse-threshold` (default `0.4`, stops when `rmse >= threshold`)
  - `--ready-timeout-s` (default `60.0`)
  - `--n-obs` (default `16`)
  - `--seed` (default `0`)
- **quantized-server management**
  - `--quantized-server-cmd`: command template used to (re)start the quantized server for each step
  - `--kill-existing-quantized-server` / `--no-kill-existing-quantized-server` (default: kill)
- **misc**
  - `--ulp-fmt`: reporting format (defaults to inferred base-server metadata: `ulp_fmt → output_fmt → bfloat16`)
  - `--use-fixed-pi0-noise`: inject deterministic `obs["pi0_noise"]` (requires server to consume it)
  - `--log-dir`: where step logs go (default `./ulp_sweep_logs`)

### `--quantized-server-cmd` placeholders

The sweep performs simple string replacement on these placeholders before launching the quantized server:

- `{ulp_n}`
- `{quantized_port}`
- `{input_fmt}`, `{output_fmt}`
- `{ulp_fmt}`
- `{mat_in_fmt}`, `{mat_out_fmt}`, `{vec_out_fmt}`

## MatVec ULP sweep (`run_matvec_ulp_sweep.py`)

`run_matvec_ulp_sweep.py` automates iterations of run_ulp_sweep_two_servers.py.

**What it does:**

1. Starts a single **base server** (`serve_quant.py`, bfloat16, no noise) on `--gpu-base`
   and keeps it alive for the entire run.
2. Iterates over all **16 format combinations** `(input_fmt × output_fmt)` drawn
   from `{float8_e4m3, float8_e5m2, float16, bfloat16}`.
3. For each combo it calls `run_ulp_sweep_two_servers.py` as a subprocess.
   That script restarts the quantized server (`serve_quant.py`) on `--gpu-quant`
   for each `ulp_n` step, queries both servers with the same random DROID observations,
   and stops when `RMSE >= --rmse-threshold` and `nan repeated three times`.
4. Collects results and writes:
   - per-combo `results.json` and the inner server logs
   - `summary.csv` — one row per combo with `max_tol_ulp_n`
   - `ulp_rmse_grid.png` — 4×4 grid of RMSE-vs-ulp_n curves
   - `tolerance_hmap.png` — heatmap of max tolerable `ulp_n` per combo

> **Note:** This script currently uses the simple `QuantLinear` mode
> (`--input-fmt` / `--output-fmt`). `QuantLinearMatVec` (`--use-matvec`) is not
> used, so vector outputs are **not** yet separately quantized.

### Usage

```bash
# Standard run (base on GPU 0, quantized on GPU 1):
python pi0_inout/run_matvec_ulp_sweep.py \
    --checkpoint-dir /path/to/safetensors_checkpoint \
    --output-dir ./matvec_ulp_sweep \
    --gpu-base 0 --gpu-quant 1 \
    --ulp-step 10 --max-ulp-n 1000 # 10 iterations per combo

# Quick smoke test (3 ULP levels, 4 observations per combo):
python pi0_inout/run_matvec_ulp_sweep.py \
    --checkpoint-dir /path/to/ckpt \
    --max-ulp-n 3 --n-obs 4 --output-dir /tmp/test_sweep

# Resume an interrupted run:
python pi0_inout/run_matvec_ulp_sweep.py \
    --checkpoint-dir /path/to/ckpt --resume --output-dir ./matvec_ulp_sweep

# Run only specific combos (e.g. to find threshold crossing for out=fp16 combos):
python pi0_inout/run_matvec_ulp_sweep.py \
    --checkpoint-dir /path/to/ckpt \
    --only-combos e5m2:fp16,fp16:fp16,bf16:fp16 \
    --max-ulp-n 10000 --ulp-step 50 \
    --output-dir ./matvec_ulp_sweep
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint-dir` | *(required)* | Directory containing `model.safetensors` |
| `--config` | `pi05_droid_jointpos_polaris` | openpi training config name |
| `--output-dir` | `matvec_ulp_sweep` | Where run directories and plots are written |
| `--gpu-base` | `0` | CUDA device for the base (reference) server |
| `--gpu-quant` | `1` | CUDA device for the quantized server |
| `--base-port` | `8000` | WebSocket port for the base server |
| `--quantized-port` | `8002` | WebSocket port for the quantized server |
| `--max-ulp-n` | `5000` | Maximum `ulp_n` to sweep up to |
| `--ulp-step` | `1` | Increment per sweep step (e.g. `10` → 10, 20, 30 …) |
| `--n-obs` | `16` | Random DROID observations per combo |
| `--seed` | `0` | RNG seed |
| `--rmse-threshold` | `0.4` | Max rmse threshold default = 0.4 |
| `--ready-timeout` | `120.0` | Seconds to wait for each server to become ready |
| `--resume` | off | Skip combos whose `results.json` already exists |
| `--only-combos` | *(all 16)* | Comma-separated `INPUT:OUTPUT` pairs to run; others are skipped. Accepts short names (`e4m3`, `e5m2`, `fp16`, `bf16`) or full names. Example: `e5m2:fp16,fp16:fp16,bf16:fp16` |
| `--openpi-dir` | `./openpi` | Override path to openpi repo root |

---

## One-shot server comparison (`run_ulp_server_experiment.py`)

`run_ulp_server_experiment.py` is a **standalone** manual tool for a single
RMSE measurement between two already-running servers.

**How it differs from `run_ulp_sweep_two_servers.py`:**

| | `run_ulp_server_experiment.py` | `run_ulp_sweep_two_servers.py` |
|---|---|---|
| Server management | None — base and quantized servers both must already be running | Can (re)start quantized server per step |
| Sweep over `ulp_n` | No — single evaluation | Yes — sweeps start→max, stops at threshold |
| Log files | No — stdout only | Yes — per-step `.log` files |
| `pi0_noise` injection | Always (hardcoded) | Optional (`--use-fixed-pi0-noise`) |
| Typical use | Quick manual spot-check | Automated ULP tolerance characterization |

### Usage

```bash
# Start your servers manually first, then:
python pi0_inout/run_ulp_server_experiment.py \
    --base-port 8000 \
    --quantized-port 8001 \
    --n-obs 8 \
    --quantized-ulp-n 10 --quantized-ulp-fmt bfloat16
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--base-port` | `8000` | WebSocket port of the reference server |
| `--quantized-port` | `8001` | WebSocket port of the quantized server |
| `--n-obs` | `8` | Number of random observations to compare |
| `--seed` | `0` | RNG seed |
| `--quantized-ulp-n` | `None` | Label only — printed in the output line |
| `--quantized-ulp-fmt` | `None` | Label only — printed in the output line |
| `--rmse-threshold` | `0.4` | Print `THRESHOLD VIOLATED` if RMSE exceeds this |

---

## Full benchmark sweep

`run_benchmark.py` orchestrates a sweep of all 25 format pairs, spawning
a `serve_quant.py` server and running `sim-evals/run_eval.py` for each
combination. Results (RMSE stats + success rate + videos) are written to
`--output-dir`.

```bash
python pi0_inout/run_benchmark.py \
    --sim-evals-dir /path/to/sim-evals \
    --openpi-dir /path/to/openpi \
    --checkpoint-dir /path/to/safetensors_checkpoint \
    --config pi05_droid_jointpos_polaris \
    --episodes 5 --scenes 1 2 3 \
    --output-dir ./results
```

Run `python pi0_inout/run_benchmark.py --help` for all options, including
`--input-fmts` / `--output-fmts` to restrict the sweep and `--resume` to
continue an interrupted run.

## Package layout

```
pi0_inout/
├── quant_types.py              # QuantFormat enum, quant(), sweep_pairs()
├── quant_linear.py             # QuantLinear / QuantLinearMatVec: drop-in nn.Linear replacements
├── model_patcher.py            # patch_model(), unpatch_model(), QuantAttnContext
├── stats_tracker.py            # StatsTracker: per-layer Welford RMSE accumulator
├── eval_harness.py             # run_quantization_eval(), run_sweep() (no WebSocket needed)
├── serve_quant.py              # WebSocket server with quantized Pi0Pytorch
├── run_matvec_ulp_sweep.py     # Top-level orchestrator: sweeps 16 format combos × ULP levels
├── run_ulp_sweep_two_servers.py # Mid-level sweep: sweeps ulp_n on a live quantized server
├── run_ulp_server_experiment.py # One-shot RMSE comparison between two already-running servers
├── run_benchmark.py            # Full sweep orchestrator (spawns server + sim-evals)
└── _jax_stubs.py               # Stub modules so Pi0Pytorch loads without JAX
```
