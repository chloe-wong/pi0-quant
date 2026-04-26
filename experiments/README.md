# pi0-inout_noise

## Extra-bits accumulator sweep (`sweep_extra_bits.py`)

`sweep_extra_bits.py` sweeps the `extra_bits` parameter of `ipt_numba_exp`
(accumulator width = 15 + extra_bits) and measures action-output RMSE vs a
single shared baseline at each value. All sweep points use identical seeded
observations and diffusion noise for a fair comparison.
`extra_bits=17` → `int_width=32`, identical to `ipt_numba`.

Results are written to `experiments/results/sweep_extra_bits/<timestamp>_ipt_numba_exp_steps<N>/`:
- `action_rmse_eb<N>.csv` — per-observation RMSE for each extra_bits value
- `sweep_summary.csv` — one row per extra_bits value with overall RMSE
- `command.txt` — full command that produced this run
- `baseline_actions.txt` / `quant_actions_eb<N>.txt` — raw action arrays

A row per extra_bits value is appended to `results/sweep_extra_bits/all_runs_output_summary.csv`.

### Usage

```bash
OPENPI_DIR=/nscratch/juhyundo/pi0-quant/openpi/ \
CUDA_VISIBLE_DEVICES=1 \
uv run experiments/sweep_extra_bits.py \
    --label sweep_run \
    --checkpoint-dir /nscratch/juhyundo/pi0-quant/datasets/openpi/openpi-assets/checkpoints/pi0_droid_jointpos_safetensors \
    --config pi0_droid_jointpos_polaris \
    --norm-stats-dir /nscratch/juhyundo/pi0-quant/datasets/openpi/openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris/assets/droid \
    --gpu 0 --n-obs 1 --steps 5 \
    --extra-bits-min 0 --extra-bits-max 17 --extra-bits-step 1 --seed 0
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--label` | *(required)* | Run label, used in output summary CSV |
| `--checkpoint-dir` | *(required)* | Path to model checkpoint directory |
| `--config` | `pi05_droid_jointpos_polaris` | openpi training config name |
| `--norm-stats-dir` | — | Directory containing `norm_stats.json` for action unnormalization |
| `--gpu` | `0` | CUDA device index (use with `CUDA_VISIBLE_DEVICES`) |
| `--extra-bits-min` | `0` | Start of extra_bits sweep (inclusive) |
| `--extra-bits-max` | `17` | End of extra_bits sweep (inclusive); 17 = same as `ipt_numba` |
| `--extra-bits-step` | `1` | Step size for sweep |
| `--n-obs` | `1` | Number of random observations |
| `--steps` | `10` | Diffusion steps (label only — model uses 10 internally) |
| `--seed` | `0` | Seeds numpy RNG (observations) and torch RNG (diffusion noise) |
| `--active-groups` | all | Comma-separated components to quantize |
| `--results-dir` | `experiments/results` | Root directory for outputs |

---

## Per-layer quantization evaluation (`run_eval_mxu_io.py`)

`run_eval_mxu_io.py` patches Pi0's `nn.Linear` layers with quantization (either a hardware-accurate functional model or software format-flag quantization), runs two full inference passes on the same observations, and reports per-layer RMSE between the unpatched and patched outputs. With `--save-tensors`, it also writes per-layer matmul I/O tensors as `.npz` files — base model and functional model inputs/outputs — for use as golden data by verification teams.

### Two-pass flow

**Pass 1 — Reference (unpatched):** the model runs as-is, capturing each `nn.Linear`'s activation input, weight, bias, and output.

**Pass 2 — Patched:** each `nn.Linear` is replaced with `QuantLinear`. Crucially, each patched layer receives the **same clean activation from Pass 1** rather than the accumulated quantized activation from earlier layers. This isolates per-layer error from end-to-end propagated error, making RMSE measurements per-layer independent.

### Matrix path

Choose one mode; they are mutually exclusive:

| Mode | Flags | What it does |
|---|---|---|
| Hardware-accurate sim | `--functional-model NAME` | Delegates the matmul to a functional model (e.g. `ipt_numba`) that simulates hardware-accurate FP8 arithmetic tile by tile |
| Software format flags | `--mx-input-fmt` / `--mx-output-fmt` | Quantizes activations + weights to a software IEEE format (FP8, FP16, BF16) and computes in that dtype |
| Passthrough (default) | *(neither set)* | BF16 no-op; measures RMSE overhead of the patching infrastructure (~0) |

Available functional models: `ipt`, `ipt_numba`, `ipt_c`, `systolic_c`.

### Usage

```bash
# Hardware-accurate IPT simulation on vision linear layers, real sim observation, with tensor capture:
CUDA_VISIBLE_DEVICES=1 uv run python experiments/run_eval_mxu_io.py \
    --label ipt_numba_mx_io_vision_linear \
    --functional-model ipt_numba \
    --ops linear \
    --active-groups vision \
    --steps 3 \
    --save-tensors \
    --obs-dir sim-evals/runs/2026-04-07/19-57-01 \
    --obs-file obs_0000.npz \
    --checkpoint-dir datasets/openpi/openpi-assets/checkpoints/pi0_droid_jointpos_safetensors \
    --config pi0_droid_jointpos_polaris

# Systolic-array simulation on vision linear layers, real sim observation, with tensor capture:
CUDA_VISIBLE_DEVICES=1 uv run python experiments/run_eval_mxu_io.py \
    --label systolic__mx_io_vision_linear \
    --functional-model systolic_c \
    --ops linear \
    --active-groups vision \
    --steps 3 \
    --save-tensors \
    --obs-dir sim-evals/runs/2026-04-07/19-57-01 \
    --obs-file obs_0000.npz \
    --checkpoint-dir datasets/openpi/openpi-assets/checkpoints/pi0_droid_jointpos_safetensors \
    --config pi0_droid_jointpos_polaris

# Software FP8 format-flag quantization, all op types, random dummy observations:
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_eval_mxu_io.py \
    --label fp8_linear_all \
    --mx-input-fmt float8_e4m3 --mx-output-fmt bfloat16 \
    --ops linear,conv2d,attention \
    --n-obs 4 --steps 10 \
    --checkpoint-dir /path/to/checkpoint \
    --config pi0_droid_jointpos_polaris
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--label` | *(required)* | Run label — used as the output folder name under `--results-dir` |
| `--checkpoint-dir` | `/scratch/chloe.wong/data/pi05_base` | Path to model checkpoint directory |
| `--config` | `pi05_droid_jointpos_polaris` | openpi training config name |
| `--gpu` | `0` | CUDA device index (use with `CUDA_VISIBLE_DEVICES`) |
| `--n-obs` | `4` | Number of random dummy observations; ignored when `--obs-dir` is set |
| `--obs-dir` | — | Directory of `obs_*.npz` files from sim-evals; loads real observations instead of random dummies |
| `--obs-file` | — | Single `obs_*.npz` filename or path; requires `--obs-dir` (used for norm_stats lookup) |
| `--norm-stats-dir` | — | Directory containing `norm_stats.json`; defaults to `--checkpoint-dir` |
| `--steps` | `10` | Diffusion denoising steps per `sample_actions` call |
| `--functional-model` | — | Hardware-accurate matmul sim: `ipt`, `ipt_numba`, `ipt_c`, `systolic_c`; mutually exclusive with `--mx-input-fmt` |
| `--mx-input-fmt` | `passthrough` | Format-flag quantization for matmul inputs (activations + weights): `float8_e4m3`, `float8_e5m2`, `float16`, `bfloat16`, `passthrough` |
| `--mx-output-fmt` | `passthrough` | Format-flag quantization for matmul outputs |
| `--ops` | `linear` | Comma-separated op types to patch: `linear`, `conv2d`, `attention` |
| `--active-groups` | all | Comma-separated model components to quantize: `vision`, `language`, `action_expert`, `action_head` |
| `--fp8-mode` | `po2` | FP8 scaling mode: `po2` = power-of-two scale (hardware-friendly), `abs` = absmax scale |
| `--save-tensors` | off | Save per-layer matmul I/O tensors to `<results-dir>/<label>/tensors/` (one `.npz` per layer) |
| `--results-dir` | `experiments/results` | Root directory for all outputs |

### Output files

All outputs are written to `<results-dir>/<label>/`:

- `config.json` — exact parameters used (checkpoint, formats, groups, ops, elapsed time)
- `chronological.csv` — one row per patched op call in execution order; columns: `seq`, `tag` (`mx`/`vec`), `layer_name`, `component`, `rmse`, `ref_rms`, `rel_rmse`, `cumulative_rmse`, `cumulative_rel_rmse`
- `grouped.csv` — same rows sorted by `(component, layer_name, tag)` for per-layer analysis
- `summary.csv` — per-component aggregate stats (`n_layers`, `mean/std/max/min_rmse`, `mean/std/max_rel_rmse`, `max_rel_rmse_layer`, `total_calls`, `mean_cumulative_rel_rmse`); separate rows for `mx` and `vec`
- `worst_layers.csv` — top-20 layers by `rel_rmse` across all components and tags
- `tensors/` — one `.npz` per patched layer (only written when `--save-tensors` is set; see below)

A row is also appended to `<results-dir>/all_runs_summary.csv` after each run.

### NPZ tensor format (`--save-tensors`)

One file per patched layer. Layer names use dots as path separators; dots are replaced with `__` in filenames:
```
paligemma_with_expert__paligemma__language_model__model__layers__0__self_attn__q_proj.npz
```

Only layers belonging to `--active-groups` are saved; layers outside the active groups are not captured.

#### Unpatched pass (base model) — stored as int16 raw BF16 bits

Reconstruct with: `torch.from_numpy(arr).view(torch.bfloat16)`

| Key | Shape | Description |
|---|---|---|
| `unpatched_x` | `[N, *]` | Activation input to the layer |
| `unpatched_w` | `[out, in]` | Weight matrix (static; stored once) |
| `unpatched_b` | `[out]` | Bias (static; absent if layer has no bias) |
| `unpatched_y` | `[N, *]` | Output of `F.linear(x, w, b)` |

`N` = number of times the layer was called across all diffusion steps and observations. Vision ViT layers are typically called once per observation (image features are not recomputed per denoising step).

#### Patched pass (functional model) — FP8 fields as uint8 raw bits, output as int16 BF16 bits

Reconstruct FP8: `torch.from_numpy(arr).view(torch.float8_e4m3fn) * (2 ** scale_exp)`
Reconstruct output: `torch.from_numpy(arr).view(torch.bfloat16)`

| Key | Shape | Description |
|---|---|---|
| `patched_x_fp8` | `[N, *]` | Raw FP8 E4M3 bytes of activation input |
| `patched_x_fp8_scale` | `int32[N]` | Per-call scale exponent for activation; `scale = 2 ** exp` |
| `patched_w_fp8` | `[out, in]` | Raw FP8 E4M3 bytes of weight (static) |
| `patched_w_fp8_scale` | `int32` scalar | Scale exponent for weight (static) |
| `patched_b_fp8` | `[out]` | Raw FP8 E4M3 bytes of bias (static; absent if no bias) |
| `patched_b_fp8_scale` | `int32` scalar | Scale exponent for bias (static; absent if no bias) |
| `patched_y_quant` | `[N, *]` | Functional model output (BF16 bits) |

`patched_y_quant` is the output of the functional model (e.g. `ipt_numba`) given the clean reference activation from Pass 1. The FP8 fields always use power-of-two scaling (po2 mode) for storage — scales are stored as integer exponents (`int32`) since `scale = 2 ** exp` exactly. This is a separate quantization done for compact representation and may differ from the functional model's internal FP8 encoding.

If shapes are inconsistent across calls (rare), arrays fall back to per-call naming: `prefix_key_call0`, `prefix_key_call1`, etc.

### Caveats

- **Reusing `--label`**: the output directory is created with `exist_ok=True` — no warning is emitted. All per-run CSVs (`config.json`, `chronological.csv`, etc.) are silently overwritten. In `tensors/`, npz files for layers active in the new run are overwritten; files for layers **not** in the new run (e.g. from a prior wider `--active-groups`) are left untouched, silently mixing tensors from two different runs.
- **`--functional-model` and `--mx-input-fmt`** are mutually exclusive. Setting both raises a `ValueError` in `QuantLinear`.
- **`--obs-file` requires `--obs-dir`**: `--obs-dir` is used to locate `norm_stats.json` even when `--obs-file` points to a file outside that directory.
- **`--n-obs` is ignored** when `--obs-dir` is set; all matching `obs_*.npz` files in the directory are loaded (or the single file specified by `--obs-file`).

---

## Decode matmul I/O tensors (`decode_mxu_npz.py`)

`decode_mxu_npz.py` reads a `.npz` file produced by `run_eval_mxu_io.py --save-tensors`, reconstructs the original float values from their packed representations (BF16 raw int16 bits, FP8 E4M3 raw uint8 bits with per-tensor power-of-two scale), and prints them — or a per-tensor statistics summary — to stdout and optionally to a log file.

### Reconstruction equations

| Stored encoding | Reconstruction |
|---|---|
| `unpatched_*` / `patched_y_quant` (int16 BF16 bits) | `torch.from_numpy(arr).view(torch.bfloat16)` |
| `patched_{x,w,b}_fp8` (uint8 FP8 E4M3 bits) + `*_fp8_scale` (int32 exponent) | `view(torch.float8_e4m3fn).float() * (2 ** scale_exp)` |

### Usage

```bash
# Print all tensor values for call 0 (default), also save to a log file:
uv run decode_mxu_npz.py \
    experiments/results/ipt_numba_mx_io_vision_linear_exp/tensors/paligemma_with_expert__paligemma__model__vision_tower__vision_model__encoder__layers__0__mlp__fc1.npz \
    --log-dir experiments/results/ipt_numba_mx_io_vision_linear_exp/

# Print per-tensor statistics summary (min/max/mean/scale_exp) for call 1:
uv run decode_mxu_npz.py path/to/layer.npz --summary --call 1
```

### Options

| Flag | Default | Description |
|---|---|---|
| `npz` | *(required)* | Path to `.npz` file from `--save-tensors` |
| `--summary` | off | Print per-tensor stats (min/max/mean/scale_exp) instead of raw values |
| `--call` | `0` | Which inference call index to decode (vision layers: typically 0) |
| `--log-dir` | stdout only | Directory to write a `.log` file; filename is `<layer_stem>_call<N>.log` |

---

## Per-layer vector-op accuracy evaluation (`run_eval_vec_io.py`)

`run_eval_vec_io.py` patches Pi0's vector ops (elementwise adds, multiplications, softmax, RMSNorm, etc.) with a VPU functional model, runs two full inference passes on the same observations, and reports per-layer RMSE between unpatched and patched outputs. With `--save-tensors`, it also writes per-(layer, op) I/O tensors as `.npz` files for use as golden data.

### Two-pass flow

**Pass 1 — Reference (unpatched):** the model runs as-is. If a VPU functional model is active, clean vector-op arguments are captured so Pass 2 can replay each op with the same pristine inputs (isolating per-op error from accumulated upstream error). `nn.Linear` / conv2d / attention outputs are also captured for cumulative RMSE.

**Pass 2 — Patched:** each target vector op is intercepted by `VectorQuantMode.__torch_dispatch__`. When `--vec-functional-model vector` is set, ops are routed through `VectorRTLFunctions` and RMSE is measured against the Pass 1 reference output. Without a functional model the context manager is a no-op and RMSE is all zeros. `nn.Linear` layers are patched as BF16 passthrough (zero matmul error) so that vec-op RMSE is not masked by accumulated matmul error.

### Vector path

| Mode | Flag | What it does |
|---|---|---|
| Hardware-accurate sim | `--vec-functional-model vector` | Routes all vector ops through `VectorRTLFunctions` (16-lane VPU) |
| Passthrough (default) | *(not set)* | BF16 no-op; RMSE is all zeros |

### Usage

```bash
# VPU sim on all components, real sim observation, with tensor capture:
CUDA_VISIBLE_DEVICES=1 uv run python experiments/run_eval_vec_io.py \
    --label vec_io_vpu_vision \
    --vec-functional-model vector \
    --ops linear \
    --active-groups vision \
    --steps 3 \
    --save-tensors \
    --obs-dir sim-evals/runs/2026-04-07/19-57-01 \
    --obs-file obs_0000.npz \
    --checkpoint-dir datasets/openpi/openpi-assets/checkpoints/pi0_droid_jointpos_safetensors \
    --config pi0_droid_jointpos_polaris

# Passthrough (infrastructure overhead check), random dummy observations:
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_eval_vec_io.py \
    --label vec_passthrough \
    --n-obs 2 --steps 5 \
    --checkpoint-dir /path/to/checkpoint \
    --config pi0_droid_jointpos_polaris
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--label` | *(required)* | Run label — used as the output folder name under `--results-dir` |
| `--checkpoint-dir` | `/scratch/chloe.wong/data/pi05_base` | Path to model checkpoint directory |
| `--config` | `pi05_droid_jointpos_polaris` | openpi training config name |
| `--gpu` | `0` | CUDA device index (use with `CUDA_VISIBLE_DEVICES`) |
| `--n-obs` | `4` | Number of random dummy observations; ignored when `--obs-dir` is set |
| `--obs-dir` | — | Directory of `obs_*.npz` files from sim-evals; loads real observations instead of random dummies |
| `--obs-file` | — | Single `obs_*.npz` filename or path; requires `--obs-dir` |
| `--norm-stats-dir` | — | Directory containing `norm_stats.json`; defaults to `--checkpoint-dir` |
| `--steps` | `10` | Diffusion denoising steps per `sample_actions` call |
| `--vec-functional-model` | — | VPU sim: `vector` (routes all vector ops through `VectorRTLFunctions`); required when `--save-tensors` is set |
| `--ops` | `linear` | Comma-separated matrix op types to patch as BF16 passthrough: `linear`, `conv2d`, `attention` |
| `--active-groups` | all | Comma-separated model components: `vision`, `language`, `action_expert`, `action_head` |
| `--save-tensors` | off | Save per-(layer, op) vector I/O tensors to `<results-dir>/<label>/vec_tensors/`; requires `--vec-functional-model` |
| `--trace` | off | Print one line per op as it fires with shape and RMSE |
| `--results-dir` | `experiments/results` | Root directory for all outputs |

### Output files

All outputs are written to `<results-dir>/<label>/`:

- `config.json` — exact parameters used
- `chronological.csv` — one row per vector op call in execution order; columns: `seq`, `tag`, `layer_name`, `component`, `rmse`, `ref_rms`, `rel_rmse`, `cumulative_rmse`, `cumulative_rel_rmse`
- `grouped.csv` — same rows sorted by `(component, layer_name)` for per-layer analysis
- `summary.csv` — per-component aggregate stats; separate rows for `mx` (always zero) and `vec`
- `worst_layers.csv` — top-20 layers by `rel_rmse` across all components
- `vec_tensors/` — one `.npz` per `(layer_tag, op)` (only written when `--save-tensors` is set; see below)

A row is also appended to `<results-dir>/all_runs_summary.csv` after each run.

### NPZ tensor format (`--save-tensors`)

One file per `(layer_tag, op_short)` pair. Layer tags use dots as separators; dots are replaced with `__` in filenames:
```
vision__add.npz
action_expert__native_layer_norm.npz
```

All tensors are stored as **int16 raw BF16 bit patterns**.  
Reconstruct with: `torch.from_numpy(arr).view(torch.bfloat16)`

| Key | Description |
|---|---|
| `n_calls` | Total number of times this op fired |
| `call{N}_input_0` | First positional argument for call N |
| `call{N}_input_1` | Second positional argument for call N (if present) |
| `call{N}_reference_output` | Passthrough (aten) output for call N |
| `call{N}_fm_output` | VPU functional model output for call N |
| `call{N}_rmse` | `float32` scalar — RMSE between `reference_output` and `fm_output` for call N |

---

## Decode vector I/O tensors (`decode_vec_npz.py`)

`decode_vec_npz.py` reads a `.npz` file produced by `run_eval_vec_io.py --save-tensors`, reconstructs the original BF16 values from their int16 raw bit representations, and prints them — or a per-tensor statistics summary including RMSE between `reference_output` and `fm_output` — to stdout and optionally to a log file.

### Reconstruction

All fields are stored as int16 raw BF16 bits:
```python
torch.from_numpy(arr).view(torch.bfloat16)
```

### Usage

```bash
# Print all tensor values for call 0 (default), also save to a log file:
uv run python experiments/decode_vec_npz.py \
    experiments/results/vec_io_vpu_vision/vec_tensors/vision__add.npz \
    --log-dir experiments/results/vec_io_vpu_vision/

# Print per-tensor statistics summary (min/max/mean + RMSE) for call 1:
uv run python experiments/decode_vec_npz.py path/to/layer.npz --summary --call 1
```

### Options

| Flag | Default | Description |
|---|---|---|
| `npz` | *(required)* | Path to `.npz` file from `--save-tensors` |
| `--summary` | off | Print per-tensor stats (min/max/mean) and RMSE(reference\_output, fm\_output) instead of raw values |
| `--call` | `0` | Which inference call index to decode |
| `--log-dir` | stdout only | Directory to write a `.log` file; filename is `<stem>_call<N>.log` |