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

## Package layout

```
pi0_inout/
├── quant_types.py     # QuantFormat enum, quant(), sweep_pairs()
├── quant_linear.py    # QuantLinear: drop-in nn.Linear replacement
├── model_patcher.py   # patch_model(), unpatch_model(), QuantAttnContext
├── stats_tracker.py   # StatsTracker: per-layer Welford RMSE accumulator
├── eval_harness.py    # run_quantization_eval(), run_sweep() (no WebSocket needed)
├── serve_quant.py     # WebSocket server with quantized Pi0Pytorch
├── run_benchmark.py   # Full sweep orchestrator (spawns server + sim-evals)
└── _jax_stubs.py      # Stub modules so Pi0Pytorch loads without JAX
```
