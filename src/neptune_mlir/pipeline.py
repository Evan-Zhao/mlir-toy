"""Python orchestration for Neptune MLIR lowering pipelines."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

from .operator.variants import AttentionVariant
from .plugin import NeptunePlugins, find_neptune_plugins
from .schedules import (
    AttentionSchedule,
    AttentionTileConfig,
    materialize_attention_schedule,
)

ATTENTION_TO_TRITON_INPUT_PIPELINE_BODY = (
    "transform-interpreter,"
    "func.func(scf-forall-to-parallel,gpu-map-parallel-loops,convert-parallel-loops-to-gpu),"
    "lower-affine,"
    "htile-dot-transpose-to-load-order,"
    "cse,"
    "canonicalize"
)

_VARIANT_TO_SCHEDULE = {
    AttentionVariant.GLOBAL_ATTN: AttentionSchedule.GLOBAL_ATTN,
    AttentionVariant.CAUSAL_ATTN: AttentionSchedule.CAUSAL_ATTN,
    AttentionVariant.GLOBAL_GQA: AttentionSchedule.GLOBAL_GQA,
}


def _require_plugins(plugins: NeptunePlugins | None = None) -> NeptunePlugins:
    if plugins is not None:
        return plugins
    resolved = find_neptune_plugins()
    if resolved is None:
        raise RuntimeError(
            "failed to locate Neptune MLIR native plugins; set NEPTUNE_MLIR_NATIVE_DIR "
            "to a directory containing LoopTransform, TADialect, and HTileDialect"
        )
    return resolved


def _run_mlir_opt_file(
    input_path: Path,
    pass_pipeline: str,
    plugins: NeptunePlugins,
) -> str:
    cmd = [
        "mlir-opt",
        *plugins.dialect_plugin_args(),
        *plugins.htile_pass_plugin_args(),
        str(input_path),
        f"--pass-pipeline={pass_pipeline}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "mlir-opt failed without output"
        raise RuntimeError(f"mlir-opt failed with exit code {result.returncode}: {details}")
    return result.stdout


def _export_attention_linalg_subprocess(
    *,
    variant: AttentionVariant,
    batch: int,
    q_heads: int,
    kv_heads: int | None,
    seq_len: int,
    head_dim: int,
    func_name: str,
) -> str:
    # Torch-MLIR and the standalone MLIR Python bindings ship separate native
    # runtimes that cannot be loaded into one Python process in arbitrary order.
    cmd = [sys.executable, "-m", "neptune_mlir.operator.export_attention_linalg"]
    cmd += ["--variant", variant.value, "--batch", str(batch), "--q-heads", str(q_heads)]
    cmd += ["--seq-len", str(seq_len), "--head-dim", str(head_dim), "--func-name", func_name]
    if kv_heads is not None:
        cmd.extend(["--kv-heads", str(kv_heads)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "Torch-MLIR export failed without output"
        raise RuntimeError(
            f"Torch-MLIR export failed with exit code {result.returncode}: {details}"
        )
    return result.stdout


def run_neptune_mlir_opt(
    input_mlir: str,
    pass_pipeline: str,
    plugins: NeptunePlugins | None = None,
) -> str:
    plugins = _require_plugins(plugins)
    with tempfile.TemporaryDirectory(prefix="neptune_mlir_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.mlir"
        input_path.write_text(input_mlir)
        return _run_mlir_opt_file(input_path, pass_pipeline, plugins)


def attention_to_triton_input_pass_pipeline(schedule_path: Path) -> str:
    return (
        "builtin.module("
        f"transform-preload-library{{transform-library-paths={schedule_path.as_posix()}}},"
        f"{ATTENTION_TO_TRITON_INPUT_PIPELINE_BODY}"
        ")"
    )


def lower_attention_linalg_to_triton_input_mlir(
    input_mlir: str,
    schedule: AttentionSchedule | str,
    tile_config: AttentionTileConfig | None = None,
    plugins: NeptunePlugins | None = None,
) -> str:
    plugins = _require_plugins(plugins)
    schedule_mlir = materialize_attention_schedule(schedule, tile_config)
    with tempfile.TemporaryDirectory(prefix="neptune_mlir_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "input.mlir"
        schedule_path = tmp_path / "schedule.mlir"
        input_path.write_text(input_mlir)
        schedule_path.write_text(schedule_mlir)
        pass_pipeline = attention_to_triton_input_pass_pipeline(schedule_path)
        return _run_mlir_opt_file(input_path, pass_pipeline, plugins)


def export_attention_to_triton_input_mlir(
    *,
    variant: AttentionVariant | str,
    batch: int = 1,
    q_heads: int = 4,
    kv_heads: int | None = None,
    seq_len: int = 128,
    head_dim: int = 64,
    func_name: str = "attention",
    tile_config: AttentionTileConfig | None = None,
    plugins: NeptunePlugins | None = None,
) -> str:
    variant = _coerce_attention_variant(variant)
    schedule = _VARIANT_TO_SCHEDULE.get(variant.value)
    if schedule is None:
        raise ValueError(f"unsupported attention pipeline variant: {variant}")
    input_mlir = _export_attention_linalg_subprocess(
        variant=variant,
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        func_name=func_name,
    )
    return lower_attention_linalg_to_triton_input_mlir(input_mlir, schedule, tile_config, plugins)


def lower_attention_linalg_to_triton_ast(
    input_mlir: str,
    schedule: AttentionSchedule | str,
    tile_config: AttentionTileConfig | None = None,
    plugins: NeptunePlugins | None = None,
) -> "ast.Module":
    from .translators.triton import translate_mlir_text

    lowered = lower_attention_linalg_to_triton_input_mlir(
        input_mlir, schedule, tile_config, plugins
    )
    return translate_mlir_text(lowered)


def _coerce_attention_variant(variant):
    if isinstance(variant, AttentionVariant):
        return variant
    try:
        return AttentionVariant(variant)
    except ValueError:
        raise ValueError(f"unknown attention variant: {variant}") from None
