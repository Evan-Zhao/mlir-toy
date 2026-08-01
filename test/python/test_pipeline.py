import ast
import tempfile
from itertools import product
from pathlib import Path

import pytest

from neptune_mlir.operator.variants import AttentionVariant
from neptune_mlir.pipeline import (
    attention_to_triton_input_pass_pipeline,
    export_attention_mlir,
    export_attention_to_triton_input_mlir,
)
from neptune_mlir.schedules import AttentionTileConfig


def make_attn_pytest_param(
    variant: AttentionVariant,
    batch: int,
    q_heads: int,
    seq_len: int,
    head_dim: int,
    kv_heads: int | None = None,
    kv_seq_len: int | None = None,
    window_size: int | None = None,
):
    kwargs = {"batch": batch, "q_heads": q_heads, "seq_len": seq_len, "head_dim": head_dim}
    if kv_heads is not None:
        kwargs["kv_heads"] = kv_heads
    if kv_seq_len is not None:
        kwargs["kv_seq_len"] = kv_seq_len
    if variant == AttentionVariant.ALIBI_CAUSAL_ATTN:
        input_dtypes = ("float16", "float16", "float16", "float32", "float16")
    else:
        input_dtypes = ("float16", "float16", "float16", "float16")
    if window_size is not None:
        assert variant == AttentionVariant.WINDOWED_CAUSAL_ATTN
        kwargs["window_size"] = window_size
        variant_name = f"{variant.value}-w{window_size}"
    else:
        variant_name = variant.value
    kvh_name = f"-kv{kv_heads}" if kv_heads is not None else ""
    kv_seq_name = f"-ks{kv_seq_len}" if kv_seq_len is not None else ""
    case_id = f"{variant_name}-b{batch}-qh{q_heads}{kvh_name}-qs{seq_len}{kv_seq_name}-d{head_dim}"
    return pytest.param((variant, kwargs, input_dtypes), id=case_id)


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
TRANSLATOR_INPUT_CASES = (
    [
        make_attn_pytest_param(variant, batch, heads, seq_len, hdim)
        for variant, batch, heads, seq_len, hdim in product(
            ATTN_VARIANTS, BATCHES, ATTN_HEADS, SEQ_LENS, HEAD_DIMS
        )
    ]
    + [
        make_attn_pytest_param(
            AttentionVariant.CAUSAL_ATTN, 1, 2, seq_len=s1, head_dim=64, kv_seq_len=s2
        )
        for s1, s2 in product(SEQ_LENS, SEQ_LENS)
    ]
    + [
        make_attn_pytest_param(AttentionVariant.ALIBI_CAUSAL_ATTN, batch, heads, seq_len, 64)
        for batch, heads, seq_len in product(BATCHES, ATTN_HEADS, SEQ_LENS)
    ]
    + [
        make_attn_pytest_param(
            AttentionVariant.GLOBAL_GQA, batch, q_heads, seq_len, hd, kv_heads=kv_heads
        )
        for batch, (q_heads, kv_heads), seq_len, hd in product(
            BATCHES, GQA_HEADS, SEQ_LENS, HEAD_DIMS
        )
    ]
)


def test_native_htile_dialect_typeids_match_mlir_runtime() -> None:
    from mlir import ir

    from neptune_mlir.dist import register_dialects

    context = ir.Context()
    register_dialects(context)
    with context:
        module = ir.Module.parse("module { htile.kernel @kernel() { htile.return } }")

    assert str(module).count("htile.return") == 1


def require_export_deps():
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is required for attention export tests")
    if importlib.util.find_spec("torch_mlir") is None:
        pytest.skip("Torch-MLIR is required for attention export tests")


def test_export_attention_uses_f16_dots_with_f32_accumulation() -> None:
    require_export_deps()

    exported = export_attention_mlir(
        variant=AttentionVariant.GLOBAL_ATTN,
        q_heads=2,
        seq_len=8,
        head_dim=4,
    )
    dots = [line for line in exported.splitlines() if "stablehlo.dot_general" in line]

    assert len(dots) == 2
    assert all(line.count("xf16>") >= 2 for line in dots)
    assert all(line.rstrip().endswith("xf32>") for line in dots)


def test_export_fp8_attention_preserves_quantized_kv_inputs() -> None:
    require_export_deps()

    exported = export_attention_mlir(
        variant=AttentionVariant.KV_FP8_CAUSAL_ATTN,
        q_heads=2,
        seq_len=8,
        head_dim=4,
    )
    signature = next(line for line in exported.splitlines() if "func.func @attention" in line)
    fp8_to_f32 = [
        line
        for line in exported.splitlines()
        if "stablehlo.convert" in line and "xf8E4M3FN>" in line and "xf32>" in line
    ]

    assert signature.count("xf8E4M3FN>") == 2
    assert len(fp8_to_f32) == 2


@pytest.mark.parametrize(
    ("variant", "kwargs"),
    [
        (AttentionVariant.GLOBAL_ATTN, {"q_heads": 2}),
        (AttentionVariant.CAUSAL_ATTN, {"q_heads": 2}),
        (AttentionVariant.ALIBI_CAUSAL_ATTN, {"q_heads": 2}),
        (AttentionVariant.WINDOWED_CAUSAL_ATTN, {"q_heads": 2, "window_size": 128}),
        (AttentionVariant.GLOBAL_GQA, {"q_heads": 4, "kv_heads": 2}),
    ],
)
def test_export_attention_to_triton_input_mlir(variant, kwargs) -> None:
    require_export_deps()

    lowered = export_attention_to_triton_input_mlir(
        variant=variant, seq_len=128, head_dim=64, **kwargs
    )

    assert "func.func @attention" in lowered
    assert "htile.launch_func" in lowered
    assert "htile.kernel" in lowered
    assert "htile.store" in lowered
    assert "transform.named_sequence" not in lowered
    assert "scf.forall" not in lowered
    assert "linalg.batch_matmul" not in lowered


def test_custom_tile_config_reaches_lowered_loop_bounds() -> None:
    require_export_deps()

    lowered = export_attention_to_triton_input_mlir(
        variant=AttentionVariant.GLOBAL_ATTN,
        q_heads=2,
        seq_len=128,
        head_dim=64,
        tile_config=AttentionTileConfig(block_m=64, block_n=32),
    )

    assert "arith.constant 64 : index" in lowered
    assert "arith.constant 32 : index" in lowered
    assert "htile.store" in lowered


@pytest.fixture(scope="module", params=TRANSLATOR_INPUT_CASES)
def lowered_triton_case(request):
    """Lower one attention case once for source checks and optional compilation."""
    from neptune_mlir.translators.triton import translate_mlir_text

    require_export_deps()
    variant, kwargs, input_dtypes = request.param
    lowered = export_attention_to_triton_input_mlir(variant=variant, **kwargs)
    source = ast.unparse(translate_mlir_text(lowered)) + "\n"
    return source, input_dtypes


def require_nvidia_triton():
    import importlib.util

    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton is required for Triton compilation tests")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is required for Triton compilation tests")

    import torch
    import triton

    if torch.version.cuda is None or not torch.cuda.is_available():
        pytest.skip("An Nvidia GPU is required for Triton compilation tests")
    try:
        torch.cuda.init()
        target = triton.runtime.driver.active.get_current_target()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Triton CUDA initialization failed: {exc}")
    if target.backend != "cuda":  # type: ignore
        pytest.skip(f"Triton compilation tests require the CUDA backend, got {target.backend}")  # type: ignore
    return torch


def compile_triton_source(source: str, input_dtypes: tuple[str, ...], torch) -> None:
    """Import generated source and force Triton to compile its attention kernel."""
    import importlib.util
    import inspect
    import sys

    module_name = f"compiled_attention_{abs(hash(source))}"
    with tempfile.TemporaryDirectory(prefix="neptune_triton_compile_") as temp_dir:
        module_path = Path(temp_dir) / f"{module_name}.py"
        module_path.write_text(source)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load generated Triton module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            kernel = module.attention_kernel
            parameter_count = len(inspect.signature(kernel.fn).parameters)
            assert parameter_count == len(input_dtypes)
            args = [
                torch.empty(1, dtype=getattr(torch, dtype), device="cuda") for dtype in input_dtypes
            ]
            kernel.warmup(*args, grid=(1, 1, 1))
        finally:
            sys.modules.pop(module_name, None)


def test_attention_lowering_pipeline(lowered_triton_case) -> None:
    source, _ = lowered_triton_case
    assert "@triton.jit" in source
    assert "def attention_kernel" in source


def test_attention_lowering_and_triton_compilation(lowered_triton_case) -> None:
    source, input_dtypes = lowered_triton_case
    torch = require_nvidia_triton()
    compile_triton_source(source, input_dtypes, torch)


def test_attention_pass_pipeline_embeds_schedule_preload() -> None:
    pipeline = attention_to_triton_input_pass_pipeline(Path("/tmp/schedule.mlir"))

    assert pipeline.startswith("builtin.module(transform-preload-library")
    assert "transform-library-paths=/tmp/schedule.mlir" in pipeline
    assert "transform-interpreter" in pipeline
    assert "lower-affine" in pipeline
    assert "gpu-map-parallel-loops" not in pipeline
    assert "convert-parallel-loops-to-gpu" not in pipeline
    assert "htile-dot-transpose-to-load-order" in pipeline
