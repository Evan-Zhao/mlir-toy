"""Shared operator inputs, kernel ABI arguments and correctness references.

These helpers are independent of pytest and compiler IR. Operator-specific helpers
live in attn.py/mamba.py; this module also dispatches by export operator name.
Torch is imported only when a helper is called; validation policy stays with callers.
"""

from .attn import (
    document_lengths,
    make_attn_inputs,
    make_dense_arguments,
    make_varlen_arguments,
    reference_attn,
    reference_dense,
    reference_varlen,
)
from .mamba import make_mamba_arguments, make_mamba_inputs, reference_mamba

__all__ = [
    "document_lengths",
    "make_arguments",
    "make_attn_inputs",
    "make_dense_arguments",
    "make_mamba_arguments",
    "make_mamba_inputs",
    "make_varlen_arguments",
    "reference_attn",
    "reference_dense",
    "reference_mamba",
    "reference_output",
    "reference_varlen",
]


def make_arguments(operator, options, *, device="cuda", seed=0, scale=0.2, lengths=None):
    if operator == "dense":
        return make_dense_arguments(options, device=device, seed=seed, scale=scale)
    if operator == "varlen":
        return make_varlen_arguments(
            options, lengths=lengths, device=device, seed=seed, scale=scale
        )
    if operator == "mamba":
        return make_mamba_arguments(options, device=device, seed=seed, scale=scale)
    raise ValueError(f"unknown operator: {operator}")


def reference_output(operator, options, arguments, *, rows=None):
    if operator == "dense":
        return reference_dense(options, arguments, rows=rows)
    if operator == "varlen":
        return reference_varlen(arguments)
    if operator == "mamba":
        return reference_mamba(*arguments[:7]).float()
    raise ValueError(f"unknown operator: {operator}")
