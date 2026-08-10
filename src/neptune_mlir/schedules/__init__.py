"""Packaged Transform dialect schedules."""

from dataclasses import dataclass
from enum import Enum
from importlib import resources
from string import Template


class AttentionSchedule(str, Enum):
    GLOBAL_ATTN = "global-attn"
    MASKED_ATTN = "masked-attn"
    ALIBI_CAUSAL_ATTN = "alibi-causal-attn"
    KV_FP8_CAUSAL_ATTN = "kv-fp8-causal-attn"
    GLOBAL_GQA = "global-gqa"
    VARLEN_ATTN = "varlen-attn"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class AttentionTileConfig:
    block_m: int = 128
    block_n: int = 64

    def validate(self) -> None:
        for name, value in (("block_m", self.block_m), ("block_n", self.block_n)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            if value % 16 != 0:
                raise ValueError(f"{name} must be a multiple of 16")

    def get_tile_sizes(self, n_batch_dims: int) -> str:
        tile_sizes = [1] * n_batch_dims + [self.block_m, self.block_n] + [0]
        return "[" + ", ".join(str(size) for size in tile_sizes) + "]"


# Schedule file name (relative to this __init__.py) and the number of batch dimensions in the attention tensor.
_SCHEDULE_TEMPLATES = {
    AttentionSchedule.GLOBAL_ATTN: ("attention.mlir.in", 2),
    AttentionSchedule.MASKED_ATTN: ("attention.mlir.in", 2),
    AttentionSchedule.ALIBI_CAUSAL_ATTN: ("attention.mlir.in", 2),
    AttentionSchedule.KV_FP8_CAUSAL_ATTN: ("attention_kv_fp8.mlir.in", 2),
    AttentionSchedule.GLOBAL_GQA: ("attention.mlir.in", 3),
    AttentionSchedule.VARLEN_ATTN: ("attention_varlen.mlir.in", None),
}
_PACKAGE = __name__


def _coerce_schedule(schedule: AttentionSchedule | str) -> AttentionSchedule:
    if isinstance(schedule, AttentionSchedule):
        return schedule
    try:
        return AttentionSchedule(schedule)
    except ValueError:
        raise ValueError(f"unknown attention schedule: {schedule}") from None


def materialize_attention_schedule(
    schedule: AttentionSchedule | str,
    tile_config: AttentionTileConfig | None = None,
) -> str:
    schedule = _coerce_schedule(schedule)
    tile_config = tile_config or AttentionTileConfig()
    tile_config.validate()
    template_name, n_batch_dims = _SCHEDULE_TEMPLATES[schedule]
    template_text = resources.files(_PACKAGE).joinpath(template_name).read_text()
    if schedule == AttentionSchedule.VARLEN_ATTN:
        tile_sizes = f"[1, {tile_config.block_m}, 1, {tile_config.block_n}, 0]"
    else:
        assert n_batch_dims is not None
        tile_sizes = tile_config.get_tile_sizes(n_batch_dims)
    return Template(template_text).substitute(tile_sizes=tile_sizes)


def read_attention_schedule(schedule: AttentionSchedule | str) -> str:
    return materialize_attention_schedule(schedule)
