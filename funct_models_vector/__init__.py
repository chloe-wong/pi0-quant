"""Pure-Python functional model for the Atlas vector engine.

Mirrors `src/main/scala/atlas/vector` op-by-op. Bit-exact against RTL
for the arithmetic spine, compare family, row/col reductions, the VLI
family, phased FP8 pack/unpack, and the LUT-backed transcendentals
(rcp, sqrt, log, sin, cos). The Group D placeholders (tanh, exp, exp2)
route through a Python math fallback in `vector_engine_model.py`
(`_legacy_math_fallback`) pending the HardFloat-faithful lane boxes
tracked in `IMPLEMENTATION_PLAN.md §6`.

See `README.md` for usage and the list of test tiers. Entry points:
    from funct_models_vector.vpu_op import VPUOp
    from funct_models_vector.vector_params import VectorParams
    from funct_models_vector.vector_engine_model import VectorEngineModel
    from funct_models_vector.vector_rtl_forward import VectorRTLFunctions
"""

from .vpu_op import VPUOp
from .vector_params import VectorParams

__all__ = ["VPUOp", "VectorParams"]
