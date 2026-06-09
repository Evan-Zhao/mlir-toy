import ast
from pathlib import Path

import pytest

from neptune_mlir.operator.export_attention_linalg import AttentionVariant
from neptune_mlir.pipeline import (
    attention_to_triton_input_pass_pipeline,
    export_attention_to_triton_input_mlir,
)
from neptune_mlir.plugin import find_neptune_plugins
from neptune_mlir.schedules import AttentionTileConfig


def require_plugins():
    plugins = find_neptune_plugins()
    assert plugins is not None, "Neptune failed to find its plugins (dynamic libs)"
    return plugins


@pytest.mark.parametrize(
    ("variant", "kwargs"),
    [
        (AttentionVariant.GLOBAL_ATTN, {"heads": 2}),
        (AttentionVariant.CAUSAL_ATTN, {"heads": 2}),
        (AttentionVariant.GLOBAL_GQA, {"heads": 4, "kv_heads": 2}),
    ],
)
def test_export_attention_to_triton_input_mlir(variant, kwargs) -> None:
    plugins = require_plugins()

    lowered = export_attention_to_triton_input_mlir(
        variant=variant,
        seq_len=128,
        dhead=64,
        plugins=plugins,
        **kwargs,
    )

    assert "func.func @attention" in lowered
    assert "gpu.launch" in lowered
    assert "htile.store" in lowered
    assert "transform.named_sequence" not in lowered
    assert "scf.forall" not in lowered
    assert "linalg.batch_matmul" not in lowered


def test_custom_tile_config_reaches_lowered_loop_bounds() -> None:
    plugins = require_plugins()

    lowered = export_attention_to_triton_input_mlir(
        variant=AttentionVariant.GLOBAL_ATTN,
        heads=2,
        seq_len=128,
        dhead=64,
        tile_config=AttentionTileConfig(block_m=64, block_n=32),
        plugins=plugins,
    )

    assert "arith.constant 64 : index" in lowered
    assert "arith.constant 32 : index" in lowered
    assert "htile.store" in lowered


def test_lowered_attention_mlir_is_triton_translator_input() -> None:
    plugins = require_plugins()
    lowered = export_attention_to_triton_input_mlir(
        variant=AttentionVariant.GLOBAL_ATTN,
        heads=2,
        seq_len=128,
        dhead=64,
        plugins=plugins,
    )
    from neptune_mlir.translators.triton import translate_mlir_text

    source = ast.unparse(translate_mlir_text(lowered))

    assert "@triton.jit" in source
    assert "def attention" in source


def test_attention_pass_pipeline_embeds_schedule_preload() -> None:
    pipeline = attention_to_triton_input_pass_pipeline(Path("/tmp/schedule.mlir"))

    assert pipeline.startswith("builtin.module(transform-preload-library")
    assert "transform-library-paths=/tmp/schedule.mlir" in pipeline
    assert "transform-interpreter" in pipeline
    assert "convert-parallel-loops-to-gpu" in pipeline
    assert "htile-dot-transpose-to-load-order" in pipeline


if __name__ == "__main__":
    test_lowered_attention_mlir_is_triton_translator_input()
