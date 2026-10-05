"""Shared operator inputs and correctness references for tests and benchmarks.

These helpers are independent of pytest, compiler IR, and backend kernel ABIs.
Torch is imported only when a helper is called; validation policy stays with callers.
"""

from .attn import make_attn_inputs, reference_attn
from .mamba import make_mamba_inputs, reference_mamba

__all__ = ["make_attn_inputs", "make_mamba_inputs", "reference_attn", "reference_mamba"]
