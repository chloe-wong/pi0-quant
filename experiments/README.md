# pi0-inout_noise

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