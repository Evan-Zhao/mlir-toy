"""Dependency-free attention exporter option types."""

from enum import Enum


class AttentionVariant(str, Enum):
    GLOBAL_ATTN = "global-attn"
    CAUSAL_ATTN = "causal-attn"
    GLOBAL_GQA = "global-gqa"
    FLOAT8_INPUTS = "float8-inputs"
    FAKE_QUANT = "fake-quant"
    SPARSE_MM = "sparse-mm"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "AttentionVariant":
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"unknown attention variant: {value}") from None


VARIANTS = tuple(AttentionVariant)
