import pytest

from neptune_mlir.schedules import (
    AttentionSchedule,
    AttentionTileConfig,
    materialize_attention_schedule,
    read_attention_schedule,
)


def test_default_attention_schedule_tile_sizes() -> None:
    text = materialize_attention_schedule(AttentionSchedule.GLOBAL_ATTN)
    assert "tile_sizes [1, 1, 128, 64, 0]" in text
    assert 'transform.apply_registered_pass "stablehlo-to-ta"' in text
    assert "${block_m}" not in text
    assert "${block_n}" not in text


def test_custom_attention_schedule_tile_sizes() -> None:
    text = materialize_attention_schedule(
        AttentionSchedule.MASKED_ATTN,
        AttentionTileConfig(block_m=64, block_n=32),
    )
    assert "tile_sizes [1, 1, 64, 32, 0]" in text
    assert "transform.loop.specialize_dead_tile" in text


def test_fp8_schedule_fuses_kv_dequantization() -> None:
    text = materialize_attention_schedule(AttentionSchedule.KV_FP8_CAUSAL_ATTN)
    assert "transform.fusion.greedy_input_producers_into_consumer %consumer_loops" in text
    assert "transform.linalg.greedy_inline_elementwise %bmm0" not in text
    assert "transform.apply_patterns.ta.sink_right_mul_after_matmul" in text


def test_gqa_keeps_batch_group_head_tiles_fixed() -> None:
    text = materialize_attention_schedule(
        AttentionSchedule.GLOBAL_GQA,
        AttentionTileConfig(block_m=64, block_n=32),
    )
    assert "tile_sizes [1, 1, 1, 64, 32, 0]" in text


def test_string_schedule_names_are_accepted() -> None:
    assert read_attention_schedule("global-attn") == materialize_attention_schedule(
        AttentionSchedule.GLOBAL_ATTN
    )


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (AttentionTileConfig(block_m=0), "block_m must be positive"),
        (AttentionTileConfig(block_n=7), "block_n must be a multiple of 16"),
    ],
)
def test_rejects_invalid_tile_sizes(config: AttentionTileConfig, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        materialize_attention_schedule(AttentionSchedule.GLOBAL_ATTN, config)


def test_rejects_unknown_schedule() -> None:
    with pytest.raises(ValueError, match="unknown attention schedule"):
        materialize_attention_schedule("unknown")
