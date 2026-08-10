"""Python orchestration for Neptune MLIR lowering pipelines."""

import ast
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .dist import find_neptune_opt
from .operator.variants import AttentionVariant
from .schedules import (
    AttentionSchedule,
    AttentionTileConfig,
    materialize_attention_schedule,
)

CodegenTarget = Literal["triton", "tilelang", "cutile"]


@dataclass(frozen=True)
class KernelArgument:
    shape: tuple[int, ...]
    dtype: str


_VARIANT_TO_SCHEDULE = {
    AttentionVariant.GLOBAL_ATTN: AttentionSchedule.GLOBAL_ATTN,
    AttentionVariant.CAUSAL_ATTN: AttentionSchedule.MASKED_ATTN,
    AttentionVariant.ALIBI_CAUSAL_ATTN: AttentionSchedule.ALIBI_CAUSAL_ATTN,
    AttentionVariant.WINDOWED_CAUSAL_ATTN: AttentionSchedule.MASKED_ATTN,
    AttentionVariant.KV_FP8_CAUSAL_ATTN: AttentionSchedule.KV_FP8_CAUSAL_ATTN,
}


def _run_neptune_opt_file(input_path: Path, pass_pipeline: str) -> str:
    executable = find_neptune_opt()
    if executable is None:
        raise RuntimeError(
            "failed to locate neptune-opt; set NEPTUNE_MLIR_OPT to the executable path"
        )
    cmd = [
        str(executable),
        str(input_path),
        f"--pass-pipeline={pass_pipeline}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "neptune-opt failed without output"
        raise RuntimeError(f"neptune-opt failed with exit code {result.returncode}: {details}")
    return result.stdout


def lower_stablehlo_to_linalg_mlir(input_mlir: str) -> str:
    """Lower StableHLO tensor operations to Linalg while preserving custom calls."""
    with tempfile.TemporaryDirectory(prefix="neptune_mlir_linalg_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.mlir"
        input_path.write_text(input_mlir)
        pass_pipeline = (
            "builtin.module(inline,canonicalize,cse,stablehlo-legalize-to-linalg,canonicalize,cse)"
        )
        return _run_neptune_opt_file(input_path, pass_pipeline)


def _run_export_worker(cmd: list[str], worker_name: str) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or f"{worker_name} failed without output"
        raise RuntimeError(f"{worker_name} failed with exit code {result.returncode}: {details}")
    return result.stdout


def export_attention_mlir(
    *,
    variant: AttentionVariant | str,
    batch: int = 1,
    q_heads: int = 4,
    kv_heads: int | None = None,
    seq_len: int = 128,
    kv_seq_len: int | None = None,
    head_dim: int = 64,
    window_size: int | None = 128,
    func_name: str = "attention",
    output_type: Literal["stablehlo", "linalg"] = "stablehlo",
) -> str:
    """Export dense attention through Torch-MLIR in an isolated process."""
    # Torch-MLIR and the standalone MLIR Python bindings ship separate native
    # runtimes that cannot be loaded into one Python process in arbitrary order.
    variant = _coerce_attention_variant(variant)
    if output_type not in {"stablehlo", "linalg"}:
        raise ValueError(f"unsupported Torch-MLIR output type: {output_type}")
    cmd = [sys.executable, "-m", "neptune_mlir.operator.torch_mlir_export"]
    cmd += ["--variant", variant.value, "--batch", str(batch), "--q-heads", str(q_heads)]
    cmd += ["--seq-len", str(seq_len), "--head-dim", str(head_dim), "--func-name", func_name]
    cmd += ["--output-type", output_type]
    if kv_heads is not None:
        cmd += ["--kv-heads", str(kv_heads)]
    if kv_seq_len is not None:
        cmd += ["--kv-seq-len", str(kv_seq_len)]
    if window_size is not None:
        cmd += ["--window-size", str(window_size)]
    return _run_export_worker(cmd, "Torch-MLIR export")


def export_varlen_attention_mlir(
    *,
    num_docs: int = 8,
    total_tokens: int = 1024,
    heads: int = 4,
    max_doc_tokens: int = 512,
    head_dim: int = 64,
    index_dtype: Literal["int32", "int64"] = "int32",
    func_name: str = "attention",
    output_type: Literal["stablehlo", "linalg"] = "stablehlo",
) -> str:
    """Export packed variable-length attention through JAX in an isolated process."""
    if output_type not in {"stablehlo", "linalg"}:
        raise ValueError(f"unsupported JAX export type: {output_type}")
    cmd = [sys.executable, "-m", "neptune_mlir.operator.jax_varlen_packed_attention"]
    cmd += ["--batch", str(num_docs), "--heads", str(heads)]
    cmd += ["--total-tokens", str(total_tokens), "--max-doc-tokens", str(max_doc_tokens)]
    cmd += ["--head-dim", str(head_dim), "--index-dtype", index_dtype]
    cmd += ["--func-name", func_name]
    stablehlo = _run_export_worker(cmd, "JAX varlen attention export")
    if output_type == "linalg":
        return lower_stablehlo_to_linalg_mlir(stablehlo)
    return stablehlo


def attention_to_htile_pass_pipeline(schedule_path: Path) -> str:
    return (
        "builtin.module("
        "inline,canonicalize,cse,"
        f"transform-preload-library{{transform-library-paths={schedule_path.as_posix()}}},"
        "transform-interpreter,lower-affine,htile-dot-transpose-to-load-order,cse,canonicalize"
        ")"
    )


def lower_attention_linalg_to_htile_mlir(
    input_mlir: str,
    schedule: AttentionSchedule | str,
    tile_config: AttentionTileConfig | None = None,
    *,
    n_batch_dims: int | None = None,
) -> str:
    schedule_mlir = materialize_attention_schedule(schedule, tile_config, n_batch_dims=n_batch_dims)
    with tempfile.TemporaryDirectory(prefix="neptune_mlir_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "input.mlir"
        schedule_path = tmp_path / "schedule.mlir"
        input_path.write_text(input_mlir)
        schedule_path.write_text(schedule_mlir)
        pass_pipeline = attention_to_htile_pass_pipeline(schedule_path)
        return _run_neptune_opt_file(input_path, pass_pipeline)


def export_attention_to_htile_mlir(
    *,
    variant: AttentionVariant | str,
    batch: int = 1,
    q_heads: int = 4,
    kv_heads: int | None = None,
    seq_len: int = 128,
    kv_seq_len: int | None = None,
    head_dim: int = 64,
    window_size: int = 128,
    func_name: str = "attention",
    tile_config: AttentionTileConfig | None = None,
) -> str:
    variant = _coerce_attention_variant(variant)
    schedule = _VARIANT_TO_SCHEDULE.get(variant)
    if schedule is None:
        raise ValueError(f"unsupported attention pipeline variant: {variant}")
    resolved_kv_heads = q_heads if kv_heads is None else kv_heads
    if q_heads <= 0 or resolved_kv_heads <= 0:
        raise ValueError("query and KV head counts must be positive")
    if q_heads % resolved_kv_heads != 0:
        raise ValueError(f"q heads ({q_heads}) must be divisible by kv heads ({resolved_kv_heads})")
    input_mlir = export_attention_mlir(
        variant=variant,
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seq_len=seq_len,
        kv_seq_len=kv_seq_len,
        head_dim=head_dim,
        window_size=window_size,
        func_name=func_name,
    )
    n_batch_dims = 2 if q_heads == resolved_kv_heads else 3
    return lower_attention_linalg_to_htile_mlir(
        input_mlir, schedule, tile_config, n_batch_dims=n_batch_dims
    )


def export_varlen_attention_to_htile_mlir(
    *,
    num_docs: int = 8,
    total_tokens: int = 1024,
    heads: int = 4,
    max_doc_tokens: int = 512,
    head_dim: int = 64,
    index_dtype: Literal["int32", "int64"] = "int32",
    func_name: str = "attention",
    tile_config: AttentionTileConfig | None = None,
) -> str:
    input_mlir = export_varlen_attention_mlir(
        num_docs=num_docs,
        total_tokens=total_tokens,
        heads=heads,
        max_doc_tokens=max_doc_tokens,
        head_dim=head_dim,
        index_dtype=index_dtype,
        func_name=func_name,
    )
    return lower_attention_linalg_to_htile_mlir(
        input_mlir, AttentionSchedule.VARLEN_ATTN, tile_config
    )


def translate_htile_to_ast(input_mlir: str, codegen_target: CodegenTarget) -> ast.Module:
    if codegen_target == "triton":
        from .translators.triton import translate_mlir_text
    elif codegen_target == "tilelang":
        from .translators.tilelang import translate_mlir_text
    elif codegen_target == "cutile":
        from .translators.cutile import translate_mlir_text
    else:
        raise ValueError(f"unknown codegen target: {codegen_target}")

    return translate_mlir_text(input_mlir)


def lower_attention_linalg_to_ast(
    input_mlir: str,
    schedule: AttentionSchedule | str,
    codegen_target: CodegenTarget,
    tile_config: AttentionTileConfig | None = None,
) -> ast.Module:
    lowered = lower_attention_linalg_to_htile_mlir(input_mlir, schedule, tile_config)
    return translate_htile_to_ast(lowered, codegen_target)


def get_htile_kernel_arguments(input_mlir: str) -> tuple[KernelArgument, ...]:
    """Return the static memref signature of the single outlined HTile kernel."""
    from mlir import ir

    from .translators.common import parse_mlir_module_from_text

    module = parse_mlir_module_from_text(input_mlir)
    kernels = [op for op in module.body.operations if op.operation.name == "htile.kernel"]
    if len(kernels) != 1:
        raise ValueError(f"expected one htile.kernel, found {len(kernels)}")

    arguments = []
    for argument in kernels[0].regions[0].blocks[0].arguments:
        memref_type = ir.MemRefType(argument.type)
        if any(extent < 0 for extent in memref_type.shape):
            raise ValueError("compiled backend targets require static kernel argument shapes")
        arguments.append(KernelArgument(tuple(memref_type.shape), str(memref_type.element_type)))
    return tuple(arguments)


@contextmanager
def _import_generated_source(source: str, module_prefix: str):
    module_name = f"{module_prefix}_{abs(hash(source))}"
    with tempfile.TemporaryDirectory(prefix=f"neptune_{module_prefix}_") as temp_dir:
        module_path = Path(temp_dir) / f"{module_name}.py"
        module_path.write_text(source)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load generated module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.modules.pop(module_name, None)


def _torch_dtype(torch, dtype: str):
    mapping = {
        "f8E4M3FN": "float8_e4m3fn",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    name = mapping.get(dtype)
    if name is None or not hasattr(torch, name):
        raise ValueError(f"unsupported Torch kernel argument dtype: {dtype}")
    return getattr(torch, name)


def compile_triton_source_to_ptx(source: str, kernel_arguments: tuple[KernelArgument, ...]) -> str:
    """Compile generated Triton source for the active CUDA target and return PTX."""
    try:
        import torch
        import triton  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Triton and PyTorch are required for Triton PTX compilation") from exc
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("Triton PTX compilation requires an Nvidia CUDA device")

    with _import_generated_source(source, "triton_compile") as module:
        kernel = module.attention_kernel
        parameter_count = len(inspect.signature(kernel.fn).parameters)
        if parameter_count != len(kernel_arguments):
            raise ValueError(
                f"kernel expects {parameter_count} arguments, got {len(kernel_arguments)}"
            )
        args = [
            torch.empty(1, dtype=_torch_dtype(torch, argument.dtype), device="cuda")
            for argument in kernel_arguments
        ]
        compiled = kernel.warmup(*args, grid=(1, 1, 1))
        ptx = compiled.asm.get("ptx")
        if not isinstance(ptx, str):
            raise RuntimeError("Triton compilation did not produce PTX")  # noqa: TRY004
        return ptx


def compile_cutile_source(
    source: str,
    kernel_arguments: tuple[KernelArgument, ...],
    grid: tuple[int, int, int] = (1, 1, 1),
) -> None:
    """Compile generated cuTile source by launching it once on the active CUDA device."""
    try:
        import cuda.tile as ct  # type: ignore
        import torch
    except ImportError as exc:
        raise RuntimeError("cuTile and PyTorch are required for cuTile compilation") from exc
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("cuTile compilation requires an Nvidia CUDA device")

    with _import_generated_source(source, "cutile_compile") as module:
        args = [
            torch.empty(
                argument.shape,
                dtype=_torch_dtype(torch, argument.dtype),
                device="cuda",
            )
            for argument in kernel_arguments
        ]
        stream = torch.cuda.current_stream()
        ct.launch(stream, grid, module.attention_kernel, tuple(args))
        torch.cuda.synchronize()


def compile_tilelang_source_to_cuda(source: str, output_index: int) -> str:
    """Compile generated TileLang source and return its lowered CUDA C++ kernel."""
    try:
        import tilelang
    except ImportError as exc:
        raise RuntimeError("TileLang is required for TileLang CUDA generation") from exc

    previous_print_setting = os.environ.get("TILELANG_PRINT_ON_COMPILATION")
    os.environ["TILELANG_PRINT_ON_COMPILATION"] = "0"
    try:
        with _import_generated_source(source, "tilelang_compile") as module:
            kernel = tilelang.compile(
                module.attention_kernel,
                out_idx=[output_index],
                execution_backend="tvm_ffi",
                target="cuda",
            )
            cuda_source = kernel.kernel_source
            if not isinstance(cuda_source, str) or not cuda_source:
                raise RuntimeError("TileLang compilation did not produce CUDA source")
            return cuda_source
    finally:
        if previous_print_setting is None:
            os.environ.pop("TILELANG_PRINT_ON_COMPILATION", None)
        else:
            os.environ["TILELANG_PRINT_ON_COMPILATION"] = previous_print_setting


def _coerce_attention_variant(variant):
    if isinstance(variant, AttentionVariant):
        return variant
    try:
        return AttentionVariant(variant)
    except ValueError:
        raise ValueError(f"unknown attention variant: {variant}") from None
