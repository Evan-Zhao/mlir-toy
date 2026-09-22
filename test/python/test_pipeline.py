import ast
import importlib.util
from itertools import pairwise, product
from typing import Literal

import pytest

from neptune_mlir.benchmarks.runtime import kernel_info
from neptune_mlir.operator.variants import AttentionVariant
from neptune_mlir.pipeline import (
    _import_generated_source,
    compile_and_launch_cutile_source,
    compile_tilelang_source_to_cuda,
    compile_triton_source_to_ptx,
    export_attention_mlir,
    export_attention_to_htile_mlir,
    export_mamba_mlir,
    export_mamba_to_htile_mlir,
    export_varlen_attention_mlir,
    export_varlen_attention_to_htile_mlir,
    get_htile_kernel_arguments,
)
from neptune_mlir.schedules import AttentionTileConfig, MambaTileConfig
from neptune_mlir.testing import (
    make_attn_inputs,
    make_mamba_inputs,
    reference_attn,
    reference_mamba,
)


def translate_htile_to_ast(input_mlir: str, codegen_target: str) -> ast.Module:
    if codegen_target == "triton":
        from neptune_mlir.translators.triton import translate_mlir_text
    elif codegen_target == "tilelang":
        from neptune_mlir.translators.tilelang import translate_mlir_text
    elif codegen_target == "cutile":
        from neptune_mlir.translators.cutile import translate_mlir_text
    else:
        raise ValueError(f"unknown codegen target: {codegen_target}")
    return translate_mlir_text(input_mlir)


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


CODEGEN_TARGETS = ("triton", "cutile", "tilelang")

ATTN_VARIANTS = (
    AttentionVariant.GLOBAL_ATTN,
    AttentionVariant.CAUSAL_ATTN,
    AttentionVariant.WINDOWED_CAUSAL_ATTN,  # Using default window size (128)
    AttentionVariant.ALIBI_CAUSAL_ATTN,
    AttentionVariant.KV_FP8_CAUSAL_ATTN,
)
ATTN_HEAD_LAYOUTS = ((2, 2), (4, 2), (4, 1))  # (4, 1) is MQA.
ATTN_HEAD_DIM = 64
SHORT_SEQ_LEN, LONG_SEQ_LEN = 512, 16384
ATTN_TRANSLATOR_CASES = [
    # Cover every variant and head layout at the standard shape.
    # Batch indexing has historically interacted with grouped-head indexing, so retain the full
    # variant/head-layout cross product for batch size two.
    *[
        make_attn_pytest_param(variant, batch, q_heads, SHORT_SEQ_LEN, ATTN_HEAD_DIM, kv_heads)
        for variant, (q_heads, kv_heads), batch in product(ATTN_VARIANTS, ATTN_HEAD_LAYOUTS, (1, 2))
    ],
    # Exercise alternate dot shapes and long reduction loops once per variant.
    *[
        make_attn_pytest_param(variant, 1, 2, SHORT_SEQ_LEN, 2 * ATTN_HEAD_DIM, 2)
        for variant in ATTN_VARIANTS
    ],
    *[
        make_attn_pytest_param(variant, 1, 2, LONG_SEQ_LEN, ATTN_HEAD_DIM, 1)
        for variant in ATTN_VARIANTS
    ],
    # Cover rectangular causal attention in both long-query and long-K/V directions.
    make_attn_pytest_param(
        AttentionVariant.CAUSAL_ATTN, 1, 2, SHORT_SEQ_LEN, ATTN_HEAD_DIM, kv_seq_len=LONG_SEQ_LEN
    ),
    make_attn_pytest_param(
        AttentionVariant.CAUSAL_ATTN, 1, 2, LONG_SEQ_LEN, ATTN_HEAD_DIM, kv_seq_len=SHORT_SEQ_LEN
    ),
]

# Compile every attention variant and head layout on every backend, while distributing the
# expensive shape edges across head layouts instead of taking their Cartesian product.
ATTN_BACKEND_COMPILATION_CASES = [
    make_attn_pytest_param(
        variant,
        batch=2 if (q_heads, kv_heads) == (4, 2) else 1,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seq_len=(LONG_SEQ_LEN if (q_heads, kv_heads) == (4, 1) else SHORT_SEQ_LEN),
        head_dim=128 if (q_heads, kv_heads) == (2, 2) else 64,
    )
    for variant, (q_heads, kv_heads) in product(ATTN_VARIANTS, ATTN_HEAD_LAYOUTS)
]

# Small Mamba cases are shared by backend compilation and output-correctness tests.
MAMBA_KERNEL_NAME = "mamba_selective_scan_kernel"
MAMBA_TEST_KWARGS = {
    "batch": 2,
    "sequence_length": 8,
    "model_dim": 256,
    "expand": 1,
    "state_dim": 16,
}
MAMBA_TEST_TILE_CONFIG = MambaTileConfig(block_channels=128)
MAMBA_BACKEND_CASES = [
    pytest.param(
        (
            {
                "batch": batch,
                "sequence_length": seq_len,
                "model_dim": model_dim,
                "expand": expand,
                "state_dim": state_dim,
                "activation_dtype": dtype,
            },
            MambaTileConfig(block_channels=block),
        ),
        id=f"{dtype}-b{batch}-s{seq_len}-m{model_dim}-e{expand}-n{state_dim}-c{block}",
    )
    for dtype, (batch, seq_len, model_dim, expand, state_dim, block) in product(
        ("bfloat16", "float16", "float32"),
        ((3, 3, 128, 1, 8, 64), (2, 8, 256, 1, 16, 128), (2, 33, 256, 2, 32, 256)),
    )
]

# Every variant/layout pair sees multiple batches and query/reduction tiles. The
# local window is deliberately smaller than the sequence and crosses tile edges.
ATTN_RUNTIME_CASES = [
    make_attn_pytest_param(
        variant,
        batch=2,
        q_heads=qh,
        kv_heads=kh,
        seq_len=256,
        head_dim=64,
        window_size=73 if variant == AttentionVariant.WINDOWED_CAUSAL_ATTN else None,
    )
    for variant, (qh, kh) in product(ATTN_VARIANTS, ATTN_HEAD_LAYOUTS)
] + [
    make_attn_pytest_param(
        AttentionVariant.CAUSAL_ATTN,
        batch=1,
        q_heads=4,
        kv_heads=2,
        seq_len=qs,
        kv_seq_len=ks,
        head_dim=128,
    )
    for qs, ks in ((128, 256), (256, 128))
]

VARLEN_RUNTIME_CASES = [
    pytest.param(
        (lengths, index_dtype, head_dim, AttentionTileConfig(block_m=bm, block_n=bn)),
        id=f"{name}-{index_dtype}-d{head_dim}-m{bm}-n{bn}",
    )
    for index_dtype in ("int32", "int64")
    for name, lengths, head_dim, bm, bn in (
        ("uneven", (1, 63, 129, 256), 64, 128, 64),
        ("empty-docs", (0, 33, 0, 127, 1, 0), 128, 64, 32),
    )
]


# Packed variable-length attention cases compiled by each backend.
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


def test_export_mamba_to_triton() -> None:
    require_jax()

    exported = export_mamba_mlir(**MAMBA_TEST_KWARGS)
    assert "func.func public @selective_scan" in exported
    assert "stablehlo.while" in exported

    lowered = export_mamba_to_htile_mlir(**MAMBA_TEST_KWARGS, tile_config=MAMBA_TEST_TILE_CONFIG)
    assert "htile.kernel @mamba_selective_scan_kernel" in lowered
    assert "dimensions = [2]" in lowered
    assert "stablehlo." not in lowered
    assert "linalg." not in lowered

    translated = ast.unparse(translate_htile_to_ast(lowered, "triton"))
    assert "def mamba_selective_scan_kernel" in translated
    assert "tl.make_block_ptr" in translated
    assert "tl.sum(" in translated
    assert "tl.store(" in translated


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
        variant=variant, seq_len=SHORT_SEQ_LEN, head_dim=64, **kwargs
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


@pytest.fixture(scope="module", params=CODEGEN_TARGETS)
def attn_codegen_target(request):
    return request.param


@pytest.fixture(scope="module", params=ATTN_TRANSLATOR_CASES)
def lowered_attn_case(request):
    """Lower one attention case once before translating it to each backend."""
    require_torch_mlir()
    variant, kwargs, _ = request.param
    lowered = export_attention_to_htile_mlir(variant=variant, **kwargs)
    return lowered, get_htile_kernel_arguments(lowered)


@pytest.fixture(scope="module")
def translated_attn_case(attn_codegen_target, lowered_attn_case):
    lowered, kernel_arguments = lowered_attn_case
    module = translate_htile_to_ast(lowered, attn_codegen_target)
    return attn_codegen_target, ast.unparse(module) + "\n", kernel_arguments


@pytest.fixture(scope="module", params=ATTN_BACKEND_COMPILATION_CASES)
def attn_backend_compilation_case(request, attn_codegen_target):
    require_torch_mlir()
    variant, kwargs, _ = request.param
    lowered = export_attention_to_htile_mlir(variant=variant, **kwargs)
    kernel_arguments = get_htile_kernel_arguments(lowered)
    module = translate_htile_to_ast(lowered, attn_codegen_target)
    return attn_codegen_target, ast.unparse(module) + "\n", kernel_arguments


@pytest.fixture(scope="module", params=MAMBA_BACKEND_CASES)
def lowered_mamba_backend_case(request):
    """Lower each Mamba case once for all code generators."""
    require_jax()
    kwargs, tile_config = request.param
    lowered = export_mamba_to_htile_mlir(**kwargs, tile_config=tile_config)
    _, grid = kernel_info(lowered)
    return lowered, get_htile_kernel_arguments(lowered), kwargs, grid


@pytest.fixture(scope="module", params=CODEGEN_TARGETS)
def mamba_codegen_target(request):
    return request.param


@pytest.fixture(scope="module")
def mamba_backend_case(mamba_codegen_target, lowered_mamba_backend_case):
    lowered, kernel_arguments, kwargs, grid = lowered_mamba_backend_case
    module = translate_htile_to_ast(lowered, mamba_codegen_target)
    return mamba_codegen_target, ast.unparse(module) + "\n", kernel_arguments, kwargs, grid


@pytest.fixture(scope="module", params=CODEGEN_TARGETS)
def varlen_codegen_target(request):
    """Select varlen backends separately to avoid unrelated fixture products."""
    return request.param


@pytest.fixture(scope="module", params=VARLEN_BACKEND_COMPILATION_CASES)
def lowered_varlen_compilation_case(request):
    require_jax()
    kwargs = request.param
    lowered = export_varlen_attention_to_htile_mlir(**kwargs)
    kernel_arguments = get_htile_kernel_arguments(lowered)
    assert kernel_arguments[3].dtype == {"int32": "i32", "int64": "i64"}[kwargs["index_dtype"]]
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
        target_backend = target.backend  # type: ignore
        if target_backend != "cuda":
            pytest.skip(f"Triton compilation tests require the CUDA backend, got {target_backend}")


def _make_mamba_runtime_arguments(torch, kwargs):
    batch = kwargs["batch"]
    seq_len = kwargs["sequence_length"]
    channels = kwargs["model_dim"] * kwargs["expand"]
    state_dim = kwargs["state_dim"]

    inputs = make_mamba_inputs(
        batch,
        seq_len,
        channels,
        state_dim,
        getattr(torch, kwargs["activation_dtype"]),
        "cuda",
        seed=0,
        scale=0.2,
    )
    initial_state = torch.zeros((batch, channels, state_dim), dtype=torch.float32, device="cuda")
    scalar_zero = torch.zeros((1,), dtype=torch.float32, device="cuda")
    output = torch.full_like(inputs[0], float("nan"))
    return [*inputs, initial_state, scalar_zero, output]


def _launch_backend(torch, codegen_target, source, runtime_arguments, grid, kernel_name):
    with _import_generated_source(source, f"runtime_{codegen_target}") as module:
        kernel = getattr(module, kernel_name)
        if codegen_target == "triton":
            kernel[grid](*runtime_arguments)
            output = runtime_arguments[-1]
        elif codegen_target == "cutile":
            import cuda.tile as ct  # type: ignore

            ct.launch(
                torch.cuda.current_stream(),
                grid + (1,) * (3 - len(grid)),
                kernel,
                tuple(runtime_arguments),
            )
            output = runtime_arguments[-1]
        else:
            import tilelang

            compiled = tilelang.compile(
                kernel,
                out_idx=[],
                execution_backend="tvm_ffi",
                target="cuda",
            )
            # Pass the NaN-filled destination explicitly, so missing stores fail
            # on TileLang too rather than depending on allocator contents.
            compiled(*runtime_arguments)
            output = runtime_arguments[-1]
        torch.cuda.synchronize()
        return output


def test_attention_lowering_pipeline(translated_attn_case) -> None:
    codegen_target, source, _ = translated_attn_case
    expected_decorators = {
        "triton": "@triton.jit",
        "cutile": "@ct.kernel",
        "tilelang": "@T.prim_func",
    }
    assert expected_decorators[codegen_target] in source
    assert "def attention_kernel" in source


def test_mamba_backend_compilation(mamba_backend_case) -> None:
    codegen_target, source, kernel_arguments, _, grid = mamba_backend_case
    require_torch_with_cuda()
    require_nvidia_python_backend(codegen_target)
    if codegen_target == "triton":
        ptx = compile_triton_source_to_ptx(source, kernel_arguments, MAMBA_KERNEL_NAME)
        assert ".version" in ptx
        assert f".visible .entry {MAMBA_KERNEL_NAME}" in ptx
    elif codegen_target == "cutile":
        compile_and_launch_cutile_source(
            source,
            kernel_arguments,
            grid=(*grid, 1),
            kernel_name=MAMBA_KERNEL_NAME,
        )
    else:
        cuda_source = compile_tilelang_source_to_cuda(
            source,
            output_index=len(kernel_arguments) - 1,
            kernel_name=MAMBA_KERNEL_NAME,
        )
        assert "__global__" in cuda_source
        assert MAMBA_KERNEL_NAME in cuda_source


def test_mamba_backend_output_correctness(mamba_backend_case) -> None:
    codegen_target, source, _, kwargs, grid = mamba_backend_case
    torch = require_torch_with_cuda()
    require_nvidia_python_backend(codegen_target)
    runtime_arguments = _make_mamba_runtime_arguments(torch, kwargs)
    reference = reference_mamba(*runtime_arguments[:7])

    output = _launch_backend(
        torch, codegen_target, source, runtime_arguments, grid, MAMBA_KERNEL_NAME
    )

    assert tuple(output.shape) == tuple(reference.shape)
    assert output.dtype == reference.dtype
    assert bool(torch.isfinite(output).all())
    tolerance = {"bfloat16": 2e-3, "float16": 3e-4, "float32": 2e-5}[kwargs["activation_dtype"]]
    torch.testing.assert_close(output.float(), reference.float(), rtol=tolerance, atol=tolerance)


def test_attention_lowering_and_backend_compilation(attn_backend_compilation_case) -> None:
    codegen_target, source, kernel_arguments = attn_backend_compilation_case
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


@pytest.fixture(scope="module", params=ATTN_RUNTIME_CASES)
def lowered_attn_runtime_case(request):
    require_torch_mlir()
    variant, kwargs, _ = request.param
    # The rectangular cases also exercise non-default tile sizes.
    tile = (
        AttentionTileConfig(block_m=64, block_n=32)
        if "kv_seq_len" in kwargs
        else AttentionTileConfig()
    )
    lowered = export_attention_to_htile_mlir(variant=variant, **kwargs, tile_config=tile)
    return variant, kwargs, lowered


def test_attention_backend_output_correctness(attn_codegen_target, lowered_attn_runtime_case):
    torch = require_torch_with_cuda()
    require_nvidia_python_backend(attn_codegen_target)

    from neptune_mlir.operator.torch_mlir_export import (
        AlibiCausalAttentionModule,
        AttentionModule,
        KVOnlyQuantizedAttentionModule,
        causal_mask,
        windowed_causal_mask,
    )

    variant, kwargs, lowered = lowered_attn_runtime_case
    if (
        variant == AttentionVariant.KV_FP8_CAUSAL_ATTN
        and attn_codegen_target == "cutile"
        and torch.cuda.get_device_capability()[0] < 10
    ):
        pytest.skip("cuTile FP8 execution requires an sm100 or newer GPU")
    batch, qh, kh = kwargs["batch"], kwargs["q_heads"], kwargs["kv_heads"]
    qs, ks = kwargs["seq_len"], kwargs.get("kv_seq_len", kwargs["seq_len"])
    q = make_attn_inputs(batch, qh, qs, kwargs["head_dim"], device="cuda", seed=17)[0]
    _, k, v = make_attn_inputs(batch, kh, ks, kwargs["head_dim"], device="cuda", seed=29)
    inputs = [q, k, v]
    if variant == AttentionVariant.ALIBI_CAUSAL_ATTN:
        inputs.append(torch.linspace(0.01, 0.3, qh, device="cuda"))
        model = AlibiCausalAttentionModule()
    elif variant == AttentionVariant.KV_FP8_CAUSAL_ATTN:
        inputs[1:3] = [x.to(torch.float8_e4m3fn) for x in (k, v)]
        # Non-unit, per-KV-head scales catch omitted scaling and head indexing.
        shape = () if kh == 1 else (1, kh, 1, 1)
        inputs.extend(
            [
                torch.linspace(0.5, 1.25, kh, device="cuda").reshape(shape),
                torch.linspace(1.5, 0.75, kh, device="cuda").reshape(shape),
            ]
        )
        model = KVOnlyQuantizedAttentionModule()
    else:
        mask = {
            AttentionVariant.GLOBAL_ATTN: lambda _: None,
            AttentionVariant.CAUSAL_ATTN: causal_mask,
            AttentionVariant.WINDOWED_CAUSAL_ATTN: windowed_causal_mask(
                kwargs.get("window_size", 128)
            ),
        }[variant]
        model = AttentionModule(mask)
    with torch.no_grad():
        reference = model(*inputs)
    # Scalar memrefs use one-element runtime buffers in the backend ABI.
    arguments = [
        *(x.reshape(1) if x.ndim == 0 else x for x in inputs),
        torch.full_like(q, float("nan")),
    ]
    source = ast.unparse(translate_htile_to_ast(lowered, attn_codegen_target)) + "\n"
    # Use the exported launch bounds: singleton KV heads are folded differently
    # from GQA, and TileLang embeds these same bounds in its generated function.
    _, grid = kernel_info(lowered)
    output = _launch_backend(
        torch, attn_codegen_target, source, arguments, grid, "attention_kernel"
    )
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert bool(torch.isfinite(output).all())
    # Eager exporter modules round the QK dot to FP16, while the lowered dots
    # accumulate in FP32. Allow that difference and the tiled online softmax.
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=5e-3)


@pytest.fixture(scope="module", params=VARLEN_RUNTIME_CASES)
def lowered_varlen_runtime_case(request):
    require_jax()
    lengths, index_dtype, head_dim, tile = request.param
    kwargs = {
        "num_docs": len(lengths),
        "total_tokens": sum(lengths),
        "heads": 2,
        "max_doc_tokens": 256,
        "head_dim": head_dim,
        "index_dtype": index_dtype,
    }
    lowered = export_varlen_attention_to_htile_mlir(**kwargs, tile_config=tile)
    assert (
        get_htile_kernel_arguments(lowered)[3].dtype
        == {"int32": "i32", "int64": "i64"}[index_dtype]
    )
    return lengths, kwargs, lowered


def test_varlen_attention_backend_output_correctness(
    varlen_codegen_target, lowered_varlen_runtime_case
):
    torch = require_torch_with_cuda()
    require_nvidia_python_backend(varlen_codegen_target)
    lengths, kwargs, lowered = lowered_varlen_runtime_case
    # Reuse the dense FP32 reference independently for each document; the JAX
    # packed-window custom calls are compiler markers, not executable FFI ops.
    q, k, v = [
        x.squeeze(0).transpose(0, 1).contiguous()
        for x in make_attn_inputs(
            1, kwargs["heads"], kwargs["total_tokens"], kwargs["head_dim"], seed=41
        )
    ]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    reference = torch.empty_like(q, dtype=torch.float32)
    for start, end in pairwise(offsets):
        if start == end:
            continue
        doc = [x[start:end].transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
        reference[start:end] = reference_attn(*doc, causal=False).squeeze(0).transpose(0, 1)
    arguments = [
        q,
        k,
        v,
        torch.tensor(offsets, dtype=getattr(torch, kwargs["index_dtype"]), device="cuda"),
        torch.full_like(q, float("nan")),
    ]
    source = ast.unparse(translate_htile_to_ast(lowered, varlen_codegen_target)) + "\n"
    _, grid = kernel_info(lowered)
    output = _launch_backend(
        torch, varlen_codegen_target, source, arguments, grid, "attention_kernel"
    )
    assert output.shape == reference.shape
    assert output.dtype == q.dtype
    assert bool(torch.isfinite(output).all())
    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=2e-3)


def test_varlen_attention_backend_compilation(varlen_backend_compilation_case) -> None:
    codegen_target, source, kernel_arguments, offsets = varlen_backend_compilation_case
    require_torch_with_cuda()
    require_nvidia_python_backend(codegen_target)
    if codegen_target == "triton":
        ptx = compile_triton_source_to_ptx(source, kernel_arguments)
        assert ".version" in ptx
        assert ".visible .entry attention_kernel" in ptx
    elif codegen_target == "cutile":
        assert "get_raw_memory()" in source
        assert ".load_offset(" in source
        assert ".store_offset(" in source
        compile_and_launch_cutile_source(source, kernel_arguments, argument_values={3: offsets})
    else:
        assert "T.reshape(" in source
        assert "T.if_then_else(" in source
        assert "if view_" in source
        cuda_source = compile_tilelang_source_to_cuda(
            source, output_index=len(kernel_arguments) - 1
        )
        assert "__global__" in cuda_source
