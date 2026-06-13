"""Packaged Transform dialect schedules."""

from dataclasses import dataclass
from enum import Enum
from importlib import resources
from string import Template


class AttentionSchedule(str, Enum):
    GLOBAL_ATTN = "global-attn"
    MASKED_ATTN = "masked-attn"
    GLOBAL_GQA = "global-gqa"

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


_SCHEDULE_TEMPLATES = {
    AttentionSchedule.GLOBAL_ATTN: "global_attention.mlir.in",
    AttentionSchedule.MASKED_ATTN: "masked_attention.mlir.in",
    AttentionSchedule.GLOBAL_GQA: "global_gqa.mlir.in",
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
    template_name = _SCHEDULE_TEMPLATES[schedule]
    template_text = resources.files(_PACKAGE).joinpath(template_name).read_text()
    return Template(template_text).substitute(
        block_m=tile_config.block_m,
        block_n=tile_config.block_n,
    )


def read_attention_schedule(schedule: AttentionSchedule | str) -> str:
    return materialize_attention_schedule(schedule)
