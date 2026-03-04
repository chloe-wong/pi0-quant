"""
pi0_inout
---------
Matmul input/output quantization framework for Pi0Pytorch.

Quantizes:
  - INPUTS  to each matmul (activation + weight) → input_fmt
  - OUTPUTS of each matmul                        → output_fmt

Distinguishes vision / language / action_expert / action_head components.

Quick start:

    from pi0_inout import patch_model, StatsTracker, QuantFormat

    tracker = StatsTracker()
    model   = load_pi0_pytorch(...)          # your Pi0Pytorch model

    patch_model(
        model,
        input_fmt=QuantFormat.FLOAT8_E4M3,
        output_fmt=QuantFormat.FLOAT16,
        tracker=tracker,
    )

    # run your evaluation / sim-evals episodes ...

    tracker.summary().print()
"""

from .quant_types   import QuantFormat, quant, TORCH_DTYPE, FORMAT_BITS, all_formats, sweep_pairs
from .quant_linear  import QuantLinear, QuantLinearMatVec
from .model_patcher import (
    patch_model,
    unpatch_model,
    count_layers,
    list_linear_layers,
    QuantAttnContext,
)
from .stats_tracker import StatsTracker, Component, StatsReport
from .ulp_noise import UlpNoiseConfig, ulp_step, inject_ulp_noise
from .eval_harness  import (
    QuantConfig,
    EvalResult,
    run_quantization_eval,
    run_sweep,
    default_sweep_configs,
    results_to_dataframe,
    save_results,
)
__all__ = [
    # Types
    "QuantFormat",
    "quant",
    "TORCH_DTYPE",
    "FORMAT_BITS",
    "all_formats",
    "sweep_pairs",
    # Core modules
    "QuantLinear",
    # Patching
    "patch_model",
    "unpatch_model",
    "count_layers",
    "list_linear_layers",
    "QuantAttnContext",
    # ULP utilities
    "UlpNoiseConfig",
    "ulp_step",
    "inject_ulp_noise",
    # Stats
    "StatsTracker",
    "Component",
    "StatsReport",
    # Evaluation
    "QuantConfig",
    "EvalResult",
    "run_quantization_eval",
    "run_sweep",
    "default_sweep_configs",
    "results_to_dataframe",
    "save_results",
]
