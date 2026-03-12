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


## Automate relative-error sweep (`automate_rel_sweep.py`)

`automate_rel_sweep.py` automates iterations of different combinations of input / output format of `run_rel_sweep_two_servers.py`.

**What it does:**

1. Starts a single **base server** (`serve_quant.py`, bfloat16, no noise) on `--gpu-base`
   and keeps it alive for the entire run.
2. Iterates over all **16 format combinations** `(input_fmt × output_fmt)` drawn
   from `{float8_e4m3, float8_e5m2, float16, bfloat16}`.
3. For each combo it calls `run_rel_sweep_two_servers.py` as a subprocess.
   That script restarts the quantized server (`serve_quant.py`) on `--gpu-quant`
   for each `rel_err` step, queries both servers with the same random DROID observations,
   and stops when `RMSE >= --rmse-threshold` or nan repeats three times.
4. Collects results and writes:
   - per-combo `results.json` and the inner server logs
   - `summary.csv` — one row per combo with `max_tol_rel_err`
   - `ulp_rmse_grid.png` — 4×4 grid of RMSE-vs-rel_err curves
   - `tolerance_hmap.png` — heatmap of max tolerable `rel_err` per combo

### Usage

```bash
# Standard run w/ pi0_droid config, base on GPU 1, quantized on GPU 2:
uv run pi0_inout/automate_rel_sweep.py \
    --checkpoint-dir /path/to/pi0_droid_jointpos_safetensors \
    --config pi0_droid \
    --base-port 8001 --gpu-base 1 \
    --gpu-quant 2 --quantized-port 8002 \
    --max-rel-err 0.1 --rel-err-step 1e-4 \
    --output-dir automate_rel_sweep

# Run only specific combos
python pi0_inout/automate_rel_sweep.py \
    --checkpoint-dir /path/to/pi0_droid_jointpos_safetensors \
    --only-combos e5m2:fp16,fp16:fp16,bf16:fp16 \
    --config pi0_droid \
    --base-port 8001 --gpu-base 1 \
    --gpu-quant 2 --quantized-port 8002 \
    --max-rel-err 0.1 --rel-err-step 1e-4 \
    --output-dir automate_rel_sweep
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint-dir` | *(required)* | Directory containing `model.safetensors` |
| `--config` | `pi05_droid_jointpos_polaris` | openpi training config name |
| `--output-dir` | `automate_rel_sweep` | Where run directories and plots are written |
| `--gpu-base` | `0` | CUDA device for the base (reference) server |
| `--gpu-quant` | `1` | CUDA device for the quantized server |
| `--base-port` | `8000` | WebSocket port for the base server |
| `--quantized-port` | `8002` | WebSocket port for the quantized server |
| `--max-rel-err` | `0.1` | Maximum relative-error level to sweep up to |
| `--rel-err-step` | `1e-4` | Increment per sweep step |
| `--n-obs` | `16` | Random DROID observations per combo |
| `--seed` | `0` | RNG seed |
| `--rmse-threshold` | `0.4` | Max rmse threshold default = 0.4 |
| `--ready-timeout` | `120.0` | Seconds to wait for each server to become ready |
| `--resume` | off | Skip combos whose `results.json` already exists |
| `--only-combos` | *(all 16)* | Comma-separated `INPUT:OUTPUT` pairs to run; others are skipped. Accepts short names (`e4m3`, `e5m2`, `fp16`, `bf16`) or full names. Example: `e5m2:fp16,fp16:fp16,bf16:fp16` |
| `--openpi-dir` | `./openpi` | Override path to openpi repo root |
| `--no-fixed-pi0-noise` | off | Disable deterministic `obs["pi0_noise"]` injection (random diffusion noise per run) |

---

## Relative-error sweep (two servers)

`run_rel_sweep_two_servers.py` compares a **base server** vs a **quantized server** over the same randomly generated observations, and sweeps `--rel-err` on the quantized side. Base server and quantized server must be run on **different CUDA_VISIBLE_DEVICES** as each server is quite heavy to run.

### Typical usage

- Start a base server (no noise) on some port
- Run the sweep, letting it (re)launch the quantized (with relative-error noise) server each step:

```bash
python -m pi0_inout.run_rel_sweep_two_servers \
    --base-port 8000 \
    --quantized-port 8001 \
    --start-rel-err 1e-4 --rel-err-step 1e-4 --max-rel-err 0.1 \
    --quantized-server-cmd 'env CUDA_VISIBLE_DEVICES=2 python pi0_inout/serve_quant.py --openpi-dir /path/to/openpi --checkpoint-dir /path/to/checkpoint --config pi0_droid --gpu 0 --input-fmt float8_e5m2 --output-fmt bfloat16'
```