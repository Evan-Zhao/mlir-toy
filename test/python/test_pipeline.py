import ast
import importlib.util
from itertools import product
from pathlib import Path
from typing import Literal

import pytest

from neptune_mlir.operator.variants import AttentionVariant
from neptune_mlir.pipeline import (
    attention_to_htile_pass_pipeline,
    compile_and_launch_cutile_source,
    compile_tilelang_source_to_cuda,
    compile_triton_source_to_ptx,
    export_attention_mlir,
    export_attention_to_htile_mlir,
    export_varlen_attention_mlir,
    export_varlen_attention_to_htile_mlir,
    get_htile_kernel_arguments,
    translate_htile_to_ast,
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
    elif variant == AttentionVariant.KV_FP8_CAUSAL_ATTN:
        input_dtypes = (
            "float16",
            "float8_e4m3fn",
            "float8_e4m3fn",
            "float32",
            "float32",
            "float16",
        )
    else:
        input_dtypes = ("float16", "float16", "float16", "float16")
    if window_size is not None:
        assert variant == AttentionVariant.WINDOWED_CAUSAL_ATTN
        kwargs["window_size"] = window_size
        variant_name = f"{variant.value}-w{window_size}"
    else:
        variant_name = variant.value
    kvh_name = f"-kh{kv_heads}" if kv_heads is not None else ""
    kv_seq_name = f"-ks{kv_seq_len}" if kv_seq_len is not None else ""
    case_id = f"{variant_name}-b{batch}-qh{q_heads}{kvh_name}-qs{seq_len}{kv_seq_name}-d{head_dim}"
    return pytest.param((variant, kwargs, input_dtypes), id=case_id)


ATTN_VARIANTS = (
    AttentionVariant.GLOBAL_ATTN,
    AttentionVariant.CAUSAL_ATTN,
    AttentionVariant.WINDOWED_CAUSAL_ATTN,  # Using default window size (128)
    AttentionVariant.ALIBI_CAUSAL_ATTN,
    AttentionVariant.KV_FP8_CAUSAL_ATTN,
)
Q_KV_HEADS = ((2, 2), (4, 2), (4, 1))  # (4, 1) would be MQA
HEAD_DIM = 64
SHORT_SEQ, LONG_SEQ = 512, 16384
TRANSLATOR_INPUT_CASES = [
    # Cover every variant and head layout at the standard shape.
    # Batch indexing has historically interacted with grouped-head indexing, so retain the full
    # variant/head-layout cross product for batch size two.
    *[
        make_attn_pytest_param(variant, batch, q_heads, SHORT_SEQ, HEAD_DIM, kv_heads)
        for variant, (q_heads, kv_heads), batch in product(ATTN_VARIANTS, Q_KV_HEADS, (1, 2))
    ],
    # Exercise alternate dot shapes and long reduction loops once per variant.
    *[
        make_attn_pytest_param(variant, 1, 2, SHORT_SEQ, 2 * HEAD_DIM, 2)
        for variant in ATTN_VARIANTS
    ],
    *[make_attn_pytest_param(variant, 1, 2, LONG_SEQ, HEAD_DIM, 1) for variant in ATTN_VARIANTS],
    # Rectangular causal attention needs one case in each direction: (1024, 128) and (128, 1024).
    make_attn_pytest_param(
        AttentionVariant.CAUSAL_ATTN, 1, 2, SHORT_SEQ, HEAD_DIM, kv_seq_len=LONG_SEQ
    ),
    make_attn_pytest_param(
        AttentionVariant.CAUSAL_ATTN, 1, 2, LONG_SEQ, HEAD_DIM, kv_seq_len=SHORT_SEQ
    ),
]

# Compile every variant and head layout on every backend, while distributing the expensive shape
# edges across head layouts instead of taking their Cartesian product.
BACKEND_COMPILATION_CASES = [
    make_attn_pytest_param(
        variant,
        batch=2 if (q_heads, kv_heads) == (4, 2) else 1,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seq_len=16384 if (q_heads, kv_heads) == (4, 1) else SHORT_SEQ,
        head_dim=128 if (q_heads, kv_heads) == (2, 2) else 64,
    )
    for variant, (q_heads, kv_heads) in product(ATTN_VARIANTS, Q_KV_HEADS)
]

VARLEN_BACKEND_COMPILATION_CASES = [
    pytest.param(
        {
            "num_docs": 2,
            "total_tokens": 512,
            "heads": 2,
            "max_doc_tokens": 512,
            "head_dim": 64,
            "index_dtype": "int32",
        },
        id="docs2-tokens512-h2-d64-i32",
    ),
    pytest.param(
        {
            "num_docs": 8,
            "total_tokens": 1024,
            "heads": 4,
            "max_doc_tokens": 512,
            "head_dim": 128,
            "index_dtype": "int32",
        },
        id="docs8-tokens1024-h4-d128-i32",
    ),
    pytest.param(
        {
            "num_docs": 4,
            "total_tokens": 512,
            "heads": 2,
            "max_doc_tokens": 256,
            "head_dim": 64,
            "index_dtype": "int64",
            "tile_config": AttentionTileConfig(block_m=64, block_n=32),
        },
        id="docs4-tokens512-h2-d64-i64-m64-n32",
    ),
]


def test_native_htile_dialect_typeids_match_mlir_runtime() -> None:
    from mlir import ir

    from neptune_mlir.dist import register_dialects

    context = ir.Context()
    register_dialects(context)
    with context:
        module = ir.Module.parse("module { htile.kernel @kernel() { htile.return } }")

    assert str(module).count("htile.return") == 1


def require_torch_mlir():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is required for attention export tests")
    if importlib.util.find_spec("torch_mlir") is None:
        pytest.skip("Torch-MLIR is required for attention export tests")


def require_jax():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is required for varlen attention export tests")


def test_export_varlen_attention_to_htile_mlir() -> None:
    require_jax()

    exported = export_varlen_attention_mlir()
    assert "func.func public @attention" in exported
    assert "stablehlo.custom_call @neptune.packed_window_extract" in exported
    assert "stablehlo.custom_call @neptune.packed_window_insert" in exported

    linalg = export_varlen_attention_mlir(output_type="linalg")
    assert "linalg.generic" in linalg
    assert "stablehlo.custom_call @neptune.packed_window_extract" in linalg

    lowered = export_varlen_attention_to_htile_mlir()
    assert "htile.kernel @attention_kernel" in lowered
    assert "mask(" in lowered
    assert "other(" in lowered
    assert "tensor.extract" not in lowered

    translated = ast.unparse(translate_htile_to_ast(lowered, "triton"))
    assert "tl.load(" in translated and "mask=" in translated and "other=" in translated
    assert "tl.store(" in translated and translated.count("mask=") > 1


def test_export_attention_uses_f16_dots_with_f32_accumulation() -> None:
    require_torch_mlir()

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
    require_torch_mlir()

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
        (AttentionVariant.KV_FP8_CAUSAL_ATTN, {"q_heads": 2}),
        (AttentionVariant.GLOBAL_ATTN, {"q_heads": 4, "kv_heads": 2}),
        (AttentionVariant.CAUSAL_ATTN, {"q_heads": 4, "kv_heads": 2}),
        (AttentionVariant.ALIBI_CAUSAL_ATTN, {"q_heads": 4, "kv_heads": 2}),
        (AttentionVariant.KV_FP8_CAUSAL_ATTN, {"q_heads": 4, "kv_heads": 2}),
        (AttentionVariant.KV_FP8_CAUSAL_ATTN, {"q_heads": 4, "kv_heads": 1}),
    ],
)
def test_export_attention_to_htile_mlir(variant, kwargs) -> None:
    require_torch_mlir()

    lowered = export_attention_to_htile_mlir(
        variant=variant, seq_len=SHORT_SEQ, head_dim=64, **kwargs
    )
    assert "func.func @attention" in lowered
    assert "htile.launch_func" in lowered
    assert "htile.kernel" in lowered
    assert "htile.store" in lowered
    assert "transform.named_sequence" not in lowered
    assert "scf.forall" not in lowered
    assert "linalg.batch_matmul" not in lowered


def test_custom_tile_config_reaches_lowered_loop_bounds() -> None:
    require_torch_mlir()

    lowered = export_attention_to_htile_mlir(
        variant=AttentionVariant.GLOBAL_ATTN,
        q_heads=2,
        seq_len=128,
        head_dim=64,
        tile_config=AttentionTileConfig(block_m=64, block_n=32),
    )

    assert "arith.constant 64 : index" in lowered
    assert "arith.constant 32 : index" in lowered
    assert "htile.store" in lowered


@pytest.fixture(scope="module", params=("triton", "cutile", "tilelang"))
def codegen_target(request):
    return request.param


@pytest.fixture(scope="module", params=TRANSLATOR_INPUT_CASES)
def lowered_attention_case(request):
    """Lower one attention case once before translating it to each backend."""
    require_torch_mlir()
    variant, kwargs, _ = request.param
    lowered = export_attention_to_htile_mlir(variant=variant, **kwargs)
    return lowered, get_htile_kernel_arguments(lowered)


@pytest.fixture(scope="module")
def translated_attention_case(codegen_target, lowered_attention_case):
    lowered, kernel_arguments = lowered_attention_case
    module = translate_htile_to_ast(lowered, codegen_target)
    return codegen_target, ast.unparse(module) + "\n", kernel_arguments


@pytest.fixture(scope="module", params=BACKEND_COMPILATION_CASES)
def backend_compilation_case(request, codegen_target):
    require_torch_mlir()
    variant, kwargs, _ = request.param
    lowered = export_attention_to_htile_mlir(variant=variant, **kwargs)
    kernel_arguments = get_htile_kernel_arguments(lowered)
    module = translate_htile_to_ast(lowered, codegen_target)
    return codegen_target, ast.unparse(module) + "\n", kernel_arguments


@pytest.fixture(scope="module", params=("triton", "cutile", "tilelang"))
def varlen_codegen_target(request):
    return request.param


@pytest.fixture(scope="module", params=VARLEN_BACKEND_COMPILATION_CASES)
def lowered_varlen_compilation_case(request):
    require_jax()
    kwargs = request.param
    lowered = export_varlen_attention_to_htile_mlir(**kwargs)
    kernel_arguments = get_htile_kernel_arguments(lowered)
    offsets = [
        document * kwargs["total_tokens"] // kwargs["num_docs"]
        for document in range(kwargs["num_docs"] + 1)
    ]
    return lowered, kernel_arguments, offsets


@pytest.fixture(scope="module")
def varlen_backend_compilation_case(varlen_codegen_target, lowered_varlen_compilation_case):
    require_nvidia_python_backend(varlen_codegen_target)
    lowered, kernel_arguments, offsets = lowered_varlen_compilation_case
    module = translate_htile_to_ast(lowered, varlen_codegen_target)
    return varlen_codegen_target, ast.unparse(module) + "\n", kernel_arguments, offsets


def require_torch_with_cuda():
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is required for backend compilation tests")

    import torch

    if torch.version.cuda is None or not torch.cuda.is_available():
        pytest.skip("An Nvidia GPU is required for backend compilation tests")
    try:
        torch.cuda.init()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Torch CUDA initialization failed: {exc}")
    return torch


def require_nvidia_python_backend(backend_name: Literal["cutile", "tilelang", "triton"]):
    import importlib

    module_name = "cuda.tile" if backend_name == "cutile" else backend_name
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module = None
    if module is None:
        pytest.skip(f"{module_name} is required for its compilation test")
    if module_name == "triton":
        try:
            target = module.runtime.driver.active.get_current_target()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Triton CUDA initialization failed: {exc}")
        if target.backend != "cuda":  # type: ignore
            pytest.skip(f"Triton compilation tests require the CUDA backend, got {target.backend}")  # type: ignore


def test_attention_lowering_pipeline(translated_attention_case) -> None:
    codegen_target, source, _ = translated_attention_case
    expected_decorators = {
        "triton": "@triton.jit",
        "cutile": "@ct.kernel",
        "tilelang": "@T.prim_func",
    }
    assert expected_decorators[codegen_target] in source
    assert "def attention_kernel" in source


def test_attention_lowering_and_backend_compilation(backend_compilation_case) -> None:
    codegen_target, source, kernel_arguments = backend_compilation_case
    torch = require_torch_with_cuda()
    require_nvidia_python_backend(codegen_target)
    if codegen_target == "triton":
        ptx = compile_triton_source_to_ptx(source, kernel_arguments)
        assert ".version" in ptx
    elif codegen_target == "cutile":
        if any(argument.dtype.startswith("f8") for argument in kernel_arguments):
            major, _ = torch.cuda.get_device_capability()
            if major < 10:
                pytest.skip("cuTile FP8 compilation requires an sm100 or newer GPU")
        compile_and_launch_cutile_source(source, kernel_arguments)
    else:
        cuda_source = compile_tilelang_source_to_cuda(
            source, output_index=len(kernel_arguments) - 1
        )
        assert "__global__" in cuda_source


def test_varlen_attention_backend_compilation(varlen_backend_compilation_case) -> None:
    codegen_target, source, kernel_arguments, offsets = varlen_backend_compilation_case

    if codegen_target == "triton":
        ptx = compile_triton_source_to_ptx(source, kernel_arguments)
        assert ".version" in ptx
        assert ".visible .entry attention_kernel" in ptx
    elif codegen_target == "cutile":
        assert "get_raw_memory()" in source
        assert ".load_offset(" in source
        assert ".store_offset(" in source
        compile_and_launch_cutile_source(
            source,
            kernel_arguments,
            argument_values={3: offsets},
        )
    else:
        assert "T.reshape(" in source
        assert "T.if_then_else(" in source
        assert "if view_" in source
        cuda_source = compile_tilelang_source_to_cuda(
            source, output_index=len(kernel_arguments) - 1
        )
        assert "__global__" in cuda_source


def test_attention_pass_pipeline_embeds_schedule_preload() -> None:
    pipeline = attention_to_htile_pass_pipeline(Path("/tmp/schedule.mlir"))

    assert "transform-library-paths=/tmp/schedule.mlir" in pipeline
    assert "transform-interpreter" in pipeline
    assert "lower-affine" in pipeline
    assert "gpu-map-parallel-loops" not in pipeline
    assert "convert-parallel-loops-to-gpu" not in pipeline
    assert "htile-dot-transpose-to-load-order" in pipeline
