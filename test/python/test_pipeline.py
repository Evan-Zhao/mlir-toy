import ast
from itertools import product
from pathlib import Path

import pytest

from neptune_mlir.operator.variants import AttentionVariant
from neptune_mlir.pipeline import (
    attention_to_triton_input_pass_pipeline,
    export_attention_to_triton_input_mlir,
)
from neptune_mlir.plugin import find_neptune_plugins
from neptune_mlir.schedules import AttentionTileConfig


def make_attn_pytest_param(
    variant: AttentionVariant,
    batch: int,
    qh: int,
    kvh: int,
    seq_len: int,
    dhead: int,
    window_size: int | None = None,
):
    kwargs = {"batch": batch, "q_heads": qh, "kv_heads": kvh, "seq_len": seq_len, "head_dim": dhead}
    if window_size is not None:
        assert variant == AttentionVariant.WINDOWED_CAUSAL_ATTN
        kwargs["window_size"] = window_size
        variant_name = f"{variant.value}-w{window_size}"
    else:
        variant_name = variant.value
    case_id = f"{variant_name}-b{batch}-qh{qh}-kvh{kvh}-s{seq_len}-d{dhead}"
    return pytest.param(variant, kwargs, id=case_id)


BATCHES = (1, 2)
SEQ_LENS = (128, 1024, 16384)
HEAD_DIMS = (64, 128)
ATTN_HEADS = (2, 4)
ATTN_VARIANTS = (
    AttentionVariant.GLOBAL_ATTN,
    AttentionVariant.CAUSAL_ATTN,
    AttentionVariant.WINDOWED_CAUSAL_ATTN,  # Using default window size (128)
)
GQA_HEADS = ((4, 2), (4, 1))  # (4, 1) would be MQA
TRANSLATOR_INPUT_CASES = [
    make_attn_pytest_param(variant, batch, heads, heads, seq_len, hdim)
    for variant, batch, heads, seq_len, hdim in product(
        ATTN_VARIANTS, BATCHES, ATTN_HEADS, SEQ_LENS, HEAD_DIMS
    )
] + [
    make_attn_pytest_param(AttentionVariant.GLOBAL_GQA, batch, q_heads, kv_heads, seq_len, hd)
    for batch, (q_heads, kv_heads), seq_len, hd in product(BATCHES, GQA_HEADS, SEQ_LENS, HEAD_DIMS)
]


def require_plugins():
    plugins = find_neptune_plugins()
    assert plugins is not None, "Neptune failed to find its plugins (dynamic libs)"
    return plugins


def require_export_deps():
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is required for attention export tests")
    if importlib.util.find_spec("torch_mlir") is None:
        pytest.skip("Torch-MLIR is required for attention export tests")


@pytest.mark.parametrize(
    ("variant", "kwargs"),
    [
        (AttentionVariant.GLOBAL_ATTN, {"q_heads": 2}),
        (AttentionVariant.CAUSAL_ATTN, {"q_heads": 2}),
        (AttentionVariant.WINDOWED_CAUSAL_ATTN, {"q_heads": 2, "window_size": 128}),
        (AttentionVariant.GLOBAL_GQA, {"q_heads": 4, "kv_heads": 2}),
    ],
)
def test_export_attention_to_triton_input_mlir(variant, kwargs) -> None:
    require_export_deps()
    plugins = require_plugins()

    lowered = export_attention_to_triton_input_mlir(
        variant=variant, seq_len=128, head_dim=64, plugins=plugins, **kwargs
    )

    assert "func.func @attention" in lowered
    assert "gpu.launch" in lowered
    assert "htile.store" in lowered
    assert "transform.named_sequence" not in lowered
    assert "scf.forall" not in lowered
    assert "linalg.batch_matmul" not in lowered


def test_custom_tile_config_reaches_lowered_loop_bounds() -> None:
    require_export_deps()
    plugins = require_plugins()

    lowered = export_attention_to_triton_input_mlir(
        variant=AttentionVariant.GLOBAL_ATTN,
        q_heads=2,
        seq_len=128,
        head_dim=64,
        tile_config=AttentionTileConfig(block_m=64, block_n=32),
        plugins=plugins,
    )

    assert "arith.constant 64 : index" in lowered
    assert "arith.constant 32 : index" in lowered
    assert "htile.store" in lowered


@pytest.mark.parametrize(("variant", "kwargs"), TRANSLATOR_INPUT_CASES)
def test_lowered_attention_full_pipeline(variant, kwargs) -> None:
    require_export_deps()
    from neptune_mlir.translators.triton import translate_mlir_text

    lowered = export_attention_to_triton_input_mlir(variant=variant, **kwargs)
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
