"""
_dispatch_guards.py
-------------------
Shared thread-local guards for TorchDispatchMode interception.

Imported by both quant_vector.py (which checks the guard) and quant_linear.py
(which sets the guard around internal bookkeeping ops that should not be
intercepted by VectorQuantMode).
"""

import threading

# Re-entrant guard: while active, VectorQuantMode passes all ops through
# unchanged.  Set this whenever running internal bookkeeping (RMSE tracking,
# reference computations) that must not be intercepted.
_in_quant_guard: threading.local = threading.local()
