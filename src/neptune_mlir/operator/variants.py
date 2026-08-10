"""Dependency-free attention exporter option types."""

from enum import Enum


class AttentionVariant(str, Enum):
    GLOBAL_ATTN = "global"
    CAUSAL_ATTN = "causal"
    ALIBI_CAUSAL_ATTN = "alibi-causal"
    WINDOWED_CAUSAL_ATTN = "windowed-causal"
    KV_FP8_CAUSAL_ATTN = "kv-fp8-causal"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "AttentionVariant":
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"unknown attention variant: {value}") from None


VARIANTS = tuple(AttentionVariant)
