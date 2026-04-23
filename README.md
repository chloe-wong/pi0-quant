# pi0-quant

Hardware-accurate matmul and vector-op simulation framework for [Pi0Pytorch](https://github.com/Physical-Intelligence/openpi).

Routes every `nn.Linear` and attention score matmul through a hardware-accurate
Inner Product Tree (IPT) or Systolic Array functional model, and routes every
vector operation (layernorm, activations, elementwise ops) through
`VectorRTLFunctions` — a lane-exact simulation of the hardware vector unit.

Both simulation paths are independent and optional. Omitting both gives a
zero-RMSE BF16 passthrough for verifying that the patching infrastructure
itself introduces no error.

## What it simulates

### Matrix path (linear layers + attention)

Every `nn.Linear` is replaced with a `QuantLinear` that delegates the matmul to a
hardware-accurate functional model:

```
x_fp8, w_fp8 = quantize_to_fp8_e4m3(x, w)   # po2-scaled per-tensor
y             = functional_model(x_fp8, w_fp8, b)  # hardware-accurate BF16 accumulation
```

Attention score matmuls (`Q@K^T` and `weights@V`) use the same functional model
via `patch_attn_sdpa` / `patch_attn_eager` / `patch_attn_siglip_eager`.

### Vector path

`patch_vector_ops` installs a `TorchDispatchMode` that routes all vector operations
through `VectorRTLFunctions` (lane-box RTL-accurate simulation, BF16 in → BF16 out):

`add`, `sub`, `mul`, `pow`, `div`, `reciprocal`, `sqrt`, `rsqrt`, `sin`, `cos`,
`tanh`, `log2`, `exp`, `neg`, `amax`, `sum`, `mean` — plus fused composite ops
`native_layer_norm`, `_softmax`, `gelu`, `silu` which are decomposed into
VPU primitive sequences.

## Architecture components

RMSE is measured per layer and aggregated across four components:

| Component | Layers |
|---|---|
| `vision` | SigLIP ViT encoder (inside PaliGemma) |
| `language` | Gemma 2.6B language model (inside PaliGemma) |
| `action_expert` | Gemma 300M action-expert transformer |
| `action_head` | Action projection MLPs at the Pi0 root |

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.1
- [openpi](https://github.com/Physical-Intelligence/openpi) on your Python path
- A Pi0 checkpoint in safetensors format
- `numba` if using `ipt_numba`: `pip install numba`

## Quick start

```bash
OPENPI_DIR=/path/to/openpi \
CUDA_VISIBLE_DEVICES=0 \
/scratch/chloe.wong/envs/pi0/bin/python experiments/run_eval.py \
    --label full_hw \
    --functional-model ipt_numba \
    --vec-functional-model vector \
    --ops linear,conv2d,attention \
    --checkpoint-dir /path/to/pi0_droid_jointpos_safetensors \
    --config pi0_droid_jointpos_polaris \
    --n-obs 4 --steps 10
```

Results are written to `experiments/results/<label>/`:
- `config.json` — exact parameters used
- `chronological.csv` — one row per op call in execution order
- `grouped.csv` — same rows sorted by (component, layer_name)
- `summary.csv` — per-component aggregate RMSE stats (separate rows for `mx` and `vec`)
- `worst_layers.csv` — top-20 layers by relative RMSE

A top-level `experiments/results/all_runs_summary.csv` accumulates one row
per run across all configs.

## Common configs

```bash
# Matrix path only (hardware-accurate matmuls, passthrough vector ops)
python experiments/run_eval.py --label mx_only \
    --functional-model ipt_numba \
    --ops linear,conv2d,attention

# Vector path only (passthrough matmuls, hardware-accurate vector ops)
python experiments/run_eval.py --label vec_only \
    --vec-functional-model vector

# Full hardware path (both matrix and vector)
python experiments/run_eval.py --label full_hw \
    --functional-model ipt_numba \
    --vec-functional-model vector \
    --ops linear,conv2d,attention

# Baseline sanity check — both paths in BF16 passthrough, expect zero RMSE
python experiments/run_eval.py --label baseline

# Quantize only action components
python experiments/run_eval.py --label action_only \
    --functional-model ipt_numba \
    --active-groups action_expert,action_head
```

## Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--label` | *(required)* | Output folder name under `results/` |
| `--functional-model` | passthrough | Hardware sim for matmuls: `ipt`, `ipt_numba`, `ipt_c`, `systolic_c` |
| `--vec-functional-model` | passthrough | Hardware sim for vector ops: `vector` |
| `--ops` | `linear` | Op types to patch: `linear`, `conv2d`, `attention` |
| `--active-groups` | all | Components: `vision`, `language`, `action_expert`, `action_head` |
| `--n-obs` | 4 | Number of observations to run |
| `--steps` | 10 | Diffusion steps per observation |
| `--gpu` | 0 | CUDA device index |
| `--data-dir` | — | Path to droid_100 LeRobot dataset (uses random noise if omitted) |

## Passthrough (zero-RMSE baseline)

Omitting both `--functional-model` and `--vec-functional-model` gives BF16 passthrough:

- **Matrix path:** `patch_model` passes `QuantFormat.BFLOAT16` internally.
  `quant(x, BFLOAT16)` is an identity — `.bfloat16()` on a BF16 tensor is a no-op.
- **Vector path:** `VectorQuantMode` has `functional_model=None`, so
  `__torch_dispatch__` falls through to `return func(*args, **kwargs)` —
  the original PyTorch op runs unchanged.

Expected: RMSE = 0.0 for every layer. Use this to verify that the patching
infrastructure itself introduces no error before enabling hardware simulation.

## Functional models

### Matrix functional models

Each `nn.Linear` gets its own model instance created by a factory callable
`(in_features, out_features) → model`. During forward, the model receives
raw `(x, w, b)` tensors, quantizes them to FP8 E4M3 internally, and returns
a hardware-accurate BF16 accumulation.

| Name | Description |
|---|---|
| `ipt` | Pure Python reference (very slow) |
| `ipt_numba` | Parallel Numba JIT kernel |
| `ipt_c` | C/ctypes compiled kernel (~40 min for full eval) |
| `systolic_c` | Systolic array C kernel |

All simulate: E4M3 inputs with po2 scaling, BF16 partial sums, BF16 output.

### Adding a new matrix functional model

```python
from pi0_inout.functional_models import register_functional_model

def my_factory(in_features: int, out_features: int):
    return MyModel(in_features, out_features)

register_functional_model("my_model", my_factory)
```

Then pass `--functional-model my_model` to `run_eval.py`.

### Vector functional model

`--vec-functional-model vector` routes all vector ops through `VectorRTLFunctions`
from `funct_models_vector/vector_rtl_forward.py`. Each op is executed by the
corresponding hardware lane box:

| Op type | Lane box | HW-accurate? |
|---|---|---|
| add, sub | AddSubSumVec | Yes |
| mul | MulRec | Yes |
| rcp, sqrt, rsqrt | MulRec/Sqrt | Yes |
| sin, cos | SinCosVec | Yes |
| tanh | TanhRec | Yes |
| log2 | Log | Yes |
| exp | Exp (FPEX BF16) | Yes |
| exp2 | Exp | Math fallback (RTL port pending) |
| LayerNorm, Softmax, GELU, SiLU | Decomposed into primitives | Yes |
| rsum, rmax | AddSubSumVec/RowMax | Yes |

`exp` is now bit-exact with RTL (FPEX BF16 model). `exp2` is not used in Pi0
inference paths; the fallback never fires during a Pi0 forward pass.

## Programmatic API

```python
from pi0_inout import (
    QuantGroup, StatsTracker,
    patch_model, unpatch_model,
    patch_attn_sdpa, unpatch_attn_sdpa,
    patch_attn_eager, unpatch_attn_eager,
    patch_attn_siglip_eager, unpatch_attn_siglip_eager,
    patch_vector_ops, unpatch_vector_ops,
    get_functional_model_factory,
)

mx_tracker  = StatsTracker()
vec_tracker = StatsTracker()
active      = {QuantGroup.LANGUAGE, QuantGroup.ACTION_EXPERT}

factory = get_functional_model_factory("ipt_numba")

patch_model(model,
    mx_input_fmt=QuantFormat.BFLOAT16,   # passthrough; FM handles actual compute
    mx_output_fmt=QuantFormat.BFLOAT16,
    tracker=mx_tracker,
    active_groups=active,
    functional_model_factory=factory,
)
patch_attn_sdpa(model, active_groups=active, tracker=mx_tracker,
                functional_model_factory=factory)
patch_attn_eager(model, active_groups=active, tracker=mx_tracker,
                 functional_model_factory=factory)
patch_attn_siglip_eager(model, active_groups=active, tracker=mx_tracker,
                         functional_model_factory=factory)

from funct_models_vector.vector_rtl_forward import VectorRTLFunctions
vrf = VectorRTLFunctions(num_lanes=16)
vec_handles, vec_ctx = patch_vector_ops(model, active_groups=active,
                                         tracker=vec_tracker, functional_model=vrf)

with torch.no_grad(), vec_ctx:
    actions = model.sample_actions(device, obs, num_steps=10)

mx_tracker.summary().print()
vec_tracker.summary().print()

unpatch_model(model)
unpatch_attn_sdpa(attn_handles)
unpatch_attn_eager()
unpatch_attn_siglip_eager()
unpatch_vector_ops(vec_handles)
```

## Serving (real-time WebSocket)

`serve_quant.py` wraps the model in the openpi WebSocket protocol for
live robot serving:

```bash
OPENPI_DIR=/path/to/openpi CUDA_VISIBLE_DEVICES=0 \
python pi0_inout/serve_quant.py \
    --checkpoint-dir /path/to/pi0_droid_jointpos_safetensors \
    --config pi0_droid_jointpos_polaris \
    --functional-model ipt_numba \
    --vec-functional-model vector \
    --port 8003
```

On SIGTERM/SIGINT, RMSE stats are written to `--stats-output` (JSON).

## Package layout

```
pi0_inout/
├── quant_types.py         # QuantFormat enum, quant()
├── quant_linear.py        # QuantLinear: drop-in nn.Linear replacement
├── quant_vector.py        # VectorQuantMode: TorchDispatchMode for vector ops
├── vpu_decomp.py          # VPU decomposition functions + _VEC_FM_DISPATCH table
├── model_patcher.py       # patch_model(), unpatch_model(), patch_attn_*()
├── functional_models.py   # Registry for functional model factories
├── stats_tracker.py       # StatsTracker: per-layer Welford RMSE accumulator
├── _dispatch_guards.py    # Shared re-entrancy guard (quant_linear + quant_vector)
├── eval_harness.py        # Lower-level eval utilities
├── serve_quant.py         # WebSocket server with simulated Pi0Pytorch
├── run_benchmark.py       # Full sweep orchestrator
└── _jax_stubs.py          # Stub modules so Pi0Pytorch loads without JAX

funct_models_vector/
├── vector_rtl_forward.py  # VectorRTLFunctions torch adapter
├── vector_engine_model.py # VectorEngineModel lane-box routing
└── lane_boxes/            # Individual lane box implementations

funct_models_ipt/
├── python_ipt_base/       # Pure Python IPT ("ipt")
├── ipt_numba/             # Numba JIT IPT ("ipt_numba")
└── ipt_c/                 # C/ctypes IPT ("ipt_c")

experiments/
└── run_eval.py            # Main evaluation runner
```

### Why `_jax_stubs.py` exists

Several openpi source files import JAX at module level even though Pi0Pytorch
is pure PyTorch. `_jax_stubs.py` injects lightweight replacements into
`sys.modules` before those imports happen. `serve_quant.py` handles this
automatically; call `_jax_stubs.inject()` manually if loading Pi0Pytorch
directly.
