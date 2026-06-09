"""Python orchestration for Neptune MLIR lowering pipelines."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .operator.export_attention_linalg import AttentionVariant, export_attention_linalg
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
    heads: int = 4,
    kv_heads: int | None = None,
    seq_len: int = 128,
    dhead: int = 64,
    func_name: str = "attention",
    tile_config: AttentionTileConfig | None = None,
    plugins: NeptunePlugins | None = None,
) -> str:
    variant = _coerce_attention_variant(variant)
    if variant not in _VARIANT_TO_SCHEDULE:
        raise ValueError(f"unsupported attention pipeline variant: {variant}")
    input_mlir = export_attention_linalg(
        variant=variant,
        batch=batch,
        heads=heads,
        kv_heads=kv_heads,
        seq_len=seq_len,
        dhead=dhead,
        func_name=func_name,
    )
    return lower_attention_linalg_to_triton_input_mlir(
        input_mlir,
        _VARIANT_TO_SCHEDULE[variant],
        tile_config,
        plugins,
    )


def _coerce_attention_variant(variant: AttentionVariant | str) -> AttentionVariant:
    if isinstance(variant, AttentionVariant):
        return variant
    try:
        return AttentionVariant(variant)
    except ValueError:
        raise ValueError(f"unknown attention variant: {variant}") from None
