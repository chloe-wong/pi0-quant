"""
functional_models.py
--------------------
Registry of functional model factories for use with patch_model's
functional_model_factory parameter.

A functional model factory is a callable with signature:
    factory(in_features: int, out_features: int) -> callable(x, w, b) -> y_accum

The returned callable replaces the standard quant(x) @ quant(W) matmul inside
QuantLinear with a hardware-accurate simulation.  It must accept float32 tensors
(x, w, b) and return a float32 tensor of shape (..., out_features).

Adding a new functional model
-----------------------------
    from pi0_inout.functional_models import register_functional_model

    def my_factory(in_features: int, out_features: int):
        return MyModel(in_features, out_features)

    register_functional_model("my_model", my_factory)

Then pass --functional-model my_model to run_eval.py.

Built-in models
---------------
  "ipt"       — IPTLinearRTLFunction: bit-accurate IPT simulation (pure Python).
  "ipt_numba" — same, with a parallel Numba JIT kernel (faster).
  "ipt_c"     — same, with a C/ctypes kernel compiled at first import (fastest).
"""

from __future__ import annotations

from typing import Any, Callable

# Factory signature: (in_features, out_features) -> callable(x, w, b) -> y
FunctionalModelFactory = Callable[[int, int], Any]

_REGISTRY: dict[str, FunctionalModelFactory] = {}


def register_functional_model(name: str, factory: FunctionalModelFactory) -> None:
    """Register a functional model factory under the given name."""
    _REGISTRY[name] = factory


def get_functional_model_factory(name: str) -> FunctionalModelFactory:
    """Return the factory for the given name, or raise ValueError."""
    if name not in _REGISTRY:
        available = list(_REGISTRY)
        raise ValueError(
            f"Unknown functional model '{name}'. "
            f"Available: {available or ['(none registered)']}"
        )
    return _REGISTRY[name]


def list_functional_models() -> list[str]:
    """Return names of all registered functional models."""
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in: IPT (Inner Product Tree)
# ---------------------------------------------------------------------------

def _ipt_factory(in_features: int, out_features: int):
    from funct_models_ipt.python_ipt_base.ipt_rtl_linear import IPTLinearRTLFunction
    return IPTLinearRTLFunction()


register_functional_model("ipt", _ipt_factory)


# ---------------------------------------------------------------------------
# Built-in: IPT Numba (parallel Numba JIT kernel)
# ---------------------------------------------------------------------------

def _ipt_numba_factory(in_features: int, out_features: int):
    from funct_models_ipt.ipt_numba.ipt_rtl_linear import IPTLinearRTLFunction
    return IPTLinearRTLFunction()


register_functional_model("ipt_numba", _ipt_numba_factory)


# ---------------------------------------------------------------------------
# Built-in: IPT C (C-compiled ctypes kernel, fastest)
# ---------------------------------------------------------------------------

def _ipt_c_factory(in_features: int, out_features: int):
    from funct_models_ipt.ipt_c.ipt_rtl_linear_c import CIPTLinearRTLFunction
    return CIPTLinearRTLFunction()


register_functional_model("ipt_c", _ipt_c_factory)


# ---------------------------------------------------------------------------
# Built-in: Systolic Array C (C-compiled ctypes kernel)
# ---------------------------------------------------------------------------

def _systolic_c_factory(in_features: int, out_features: int):
    from func_models_sa.systolic_c.systolic_array_rtl_linear import SARTLLinearFunction
    return SARTLLinearFunction()


register_functional_model("systolic_c", _systolic_c_factory)
