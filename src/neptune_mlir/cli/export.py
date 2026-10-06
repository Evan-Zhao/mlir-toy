"""Export attention and Mamba operators at different compiler pipeline stages."""

import argparse
import ast
from collections.abc import Callable

from neptune_mlir.cli.operators import OPERATORS, add_operator_arguments
from neptune_mlir.pipeline import (
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

STAGES = (
    "stablehlo",
    "linalg",
    "htile",
    "triton",
    "triton-ptx",
    "tilelang",
    "tilelang-cuda",
    "cutile",
)
MAMBA_STAGES = (
    "stablehlo",
    "htile",
    "triton",
    "triton-ptx",
    "tilelang",
    "tilelang-cuda",
    "cutile",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    operators = parser.add_subparsers(dest="operator", required=True)
    for operator in OPERATORS:
        child = operators.add_parser(operator)
        add_operator_arguments(child, operator, require_variant=True)
        child.add_argument(
            "--stage",
            choices=MAMBA_STAGES if operator == "mamba" else STAGES,
            default="stablehlo",
            help="pipeline stage to print (default: stablehlo)",
        )
    return parser.parse_args(argv)


def _emit_backend_stage(stage: str, lowered: str, kernel_name: str = "attention_kernel") -> str:
    if stage == "htile":
        return lowered
    if stage.startswith("triton"):
        from neptune_mlir.translators.triton import translate_mlir_text
    elif stage.startswith("tilelang"):
        from neptune_mlir.translators.tilelang import translate_mlir_text
    else:
        from neptune_mlir.translators.cutile import translate_mlir_text
    source = ast.unparse(translate_mlir_text(lowered)) + "\n"

    if stage in {"triton", "tilelang", "cutile"}:
        return source

    kernel_arguments = get_htile_kernel_arguments(lowered)
    if stage == "triton-ptx":
        return compile_triton_source_to_ptx(source, kernel_arguments, kernel_name)
    output_index = len(kernel_arguments) - 1
    return compile_tilelang_source_to_cuda(source, output_index, kernel_name)


def _export_operator_at_stage(
    args: argparse.Namespace,
    common_args: dict[str, object],
    export_mlir: Callable[..., str],
    export_htile: Callable[..., str],
) -> str:
    if args.stage in {"stablehlo", "linalg"}:
        return export_mlir(**common_args, output_type=args.stage)

    tile_config = AttentionTileConfig(block_m=args.block_m, block_n=args.block_n)
    lowered = export_htile(**common_args, tile_config=tile_config)
    return _emit_backend_stage(args.stage, lowered)


def _export_dense_at_stage(args: argparse.Namespace) -> str:
    common_args = {
        "variant": args.variant,
        "batch": args.batch,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "seq_len": args.seq_len,
        "kv_seq_len": args.kv_seq_len,
        "head_dim": args.head_dim,
        "window_size": args.window_size,
        "func_name": args.func_name,
    }
    return _export_operator_at_stage(
        args, common_args, export_attention_mlir, export_attention_to_htile_mlir
    )


def _export_varlen_at_stage(args: argparse.Namespace) -> str:
    common_args = {
        "num_docs": args.num_docs,
        "total_tokens": args.total_tokens,
        "heads": args.heads,
        "max_doc_tokens": args.max_doc_tokens,
        "head_dim": args.head_dim,
        "index_dtype": args.index_dtype,
        "func_name": args.func_name,
    }
    return _export_operator_at_stage(
        args, common_args, export_varlen_attention_mlir, export_varlen_attention_to_htile_mlir
    )


def _export_mamba_at_stage(args: argparse.Namespace) -> str:
    common_args = {
        "batch": args.batch,
        "sequence_length": args.sequence_length,
        "model_dim": args.model_dim,
        "expand": args.expand,
        "state_dim": args.state_dim,
        "activation_dtype": args.activation_dtype,
        "func_name": args.func_name,
    }
    if args.stage == "stablehlo":
        return export_mamba_mlir(**common_args)

    lowered = export_mamba_to_htile_mlir(
        **common_args,
        tile_config=MambaTileConfig(block_channels=args.block_channels),
    )
    return _emit_backend_stage(args.stage, lowered, "mamba_selective_scan_kernel")


def export_at_stage(args: argparse.Namespace) -> str:
    if args.operator == "dense":
        return _export_dense_at_stage(args)
    if args.operator == "varlen":
        return _export_varlen_at_stage(args)
    if args.operator == "mamba":
        return _export_mamba_at_stage(args)
    raise ValueError(f"unknown operator: {args.operator}")


def main() -> None:
    print(export_at_stage(parse_args()), end="")


if __name__ == "__main__":
    main()
