"""Python orchestration for Neptune MLIR lowering pipelines."""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from .dist import find_neptune_opt
from .operator.variants import AttentionVariant
from .schedules import (
    AttentionSchedule,
    AttentionTileConfig,
    materialize_attention_schedule,
)

ATTENTION_TO_TRITON_INPUT_PIPELINE_BODY = (
    "transform-interpreter,"
    "lower-affine,"
    "htile-dot-transpose-to-load-order,"
    "cse,"
    "canonicalize"
)

_VARIANT_TO_SCHEDULE = {
    AttentionVariant.GLOBAL_ATTN: AttentionSchedule.GLOBAL_ATTN,
    AttentionVariant.CAUSAL_ATTN: AttentionSchedule.MASKED_ATTN,
    AttentionVariant.ALIBI_CAUSAL_ATTN: AttentionSchedule.ALIBI_CAUSAL_ATTN,
    AttentionVariant.WINDOWED_CAUSAL_ATTN: AttentionSchedule.MASKED_ATTN,
    AttentionVariant.KV_FP8_CAUSAL_ATTN: AttentionSchedule.KV_FP8_CAUSAL_ATTN,
    AttentionVariant.GLOBAL_GQA: AttentionSchedule.GLOBAL_GQA,
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
    """Export attention through Torch-MLIR in an isolated process."""
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "Torch-MLIR export failed without output"
        raise RuntimeError(
            f"Torch-MLIR export failed with exit code {result.returncode}: {details}"
        )
    return result.stdout


def run_neptune_mlir_opt(input_mlir: str, pass_pipeline: str) -> str:
    with tempfile.TemporaryDirectory(prefix="neptune_mlir_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.mlir"
        input_path.write_text(input_mlir)
        return _run_neptune_opt_file(input_path, pass_pipeline)


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
) -> str:
    schedule_mlir = materialize_attention_schedule(schedule, tile_config)
    with tempfile.TemporaryDirectory(prefix="neptune_mlir_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "input.mlir"
        schedule_path = tmp_path / "schedule.mlir"
        input_path.write_text(input_mlir)
        schedule_path.write_text(schedule_mlir)
        pass_pipeline = attention_to_triton_input_pass_pipeline(schedule_path)
        return _run_neptune_opt_file(input_path, pass_pipeline)


def export_attention_to_triton_input_mlir(
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
    schedule = _VARIANT_TO_SCHEDULE.get(variant.value)  # type: ignore
    if schedule is None:
        raise ValueError(f"unsupported attention pipeline variant: {variant}")
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
    return lower_attention_linalg_to_triton_input_mlir(input_mlir, schedule, tile_config)


def lower_attention_linalg_to_triton_ast(
    input_mlir: str,
    schedule: AttentionSchedule | str,
    tile_config: AttentionTileConfig | None = None,
) -> ast.Module:
    from .translators.triton import translate_mlir_text

    lowered = lower_attention_linalg_to_triton_input_mlir(input_mlir, schedule, tile_config)
    return translate_mlir_text(lowered)


def _coerce_attention_variant(variant):
    if isinstance(variant, AttentionVariant):
        return variant
    try:
        return AttentionVariant(variant)
    except ValueError:
        raise ValueError(f"unknown attention variant: {variant}") from None
