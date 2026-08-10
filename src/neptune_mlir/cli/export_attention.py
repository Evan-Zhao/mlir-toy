"""Export attention operators at different compiler pipeline stages."""

import argparse
import ast
from collections.abc import Callable

from neptune_mlir.operator.variants import VARIANTS, AttentionVariant
from neptune_mlir.pipeline import (
    compile_tilelang_source_to_cuda,
    compile_triton_source_to_ptx,
    export_attention_mlir,
    export_attention_to_htile_mlir,
    export_varlen_attention_mlir,
    export_varlen_attention_to_htile_mlir,
    get_htile_kernel_arguments,
)
from neptune_mlir.schedules import AttentionTileConfig

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


def _add_common_arguments(parser: argparse.ArgumentParser, stages: tuple[str, ...]) -> None:
    parser.add_argument(
        "--stage",
        choices=stages,
        default="stablehlo",
        help="pipeline stage to print (default: stablehlo)",
    )
    parser.add_argument("-d", "--head-dim", type=int, default=64, help="head dimension")
    parser.add_argument("--block-m", type=int, default=128, help="query tile size")
    parser.add_argument("--block-n", type=int, default=64, help="key/value tile size")
    parser.add_argument("--func-name", default="attention", help="exported function name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    operators = parser.add_subparsers(dest="operator", required=True)

    dense = operators.add_parser(
        "dense", help="dense attention variants such as global, causal, and alibi attention"
    )
    _add_common_arguments(dense, STAGES)
    dense.add_argument(
        "--variant",
        type=AttentionVariant,
        choices=VARIANTS,
        required=True,
        help="dense attention variant to export",
    )
    dense.add_argument("-b", "--batch", type=int, default=1, help="batch size")
    dense.add_argument("--q-heads", type=int, default=4, help="number of query heads")
    dense.add_argument(
        "--kv-heads", type=int, default=None, help="number of KV heads (defaults to --q-heads)"
    )
    dense.add_argument("-s", "--seq-len", type=int, default=128, help="query sequence length")
    dense.add_argument(
        "--kv-seq-len",
        type=int,
        default=None,
        help="K/V sequence length (defaults to --seq-len)",
    )
    dense.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="local history window for windowed causal attention",
    )

    varlen = operators.add_parser("varlen", help="packed variable-length attention")
    _add_common_arguments(varlen, STAGES)
    varlen.add_argument("--num-docs", type=int, default=8, help="number of packed documents")
    varlen.add_argument(
        "--total-tokens", type=int, default=1024, help="total number of packed tokens"
    )
    varlen.add_argument("--heads", type=int, default=4, help="number of attention heads")
    varlen.add_argument(
        "--max-doc-tokens", type=int, default=512, help="static maximum document length"
    )
    varlen.add_argument(
        "--index-dtype",
        choices=("int32", "int64"),
        default="int32",
        help="document-offset element type",
    )
    return parser.parse_args()


def _emit_backend_stage(stage: str, lowered: str) -> str:
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
        return compile_triton_source_to_ptx(source, kernel_arguments)
    output_index = len(kernel_arguments) - 1
    return compile_tilelang_source_to_cuda(source, output_index)


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


def export_at_stage(args: argparse.Namespace) -> str:
    if args.operator == "dense":
        return _export_dense_at_stage(args)
    if args.operator == "varlen":
        return _export_varlen_at_stage(args)
    raise ValueError(f"unknown attention operator: {args.operator}")


def main() -> None:
    print(export_at_stage(parse_args()), end="")


if __name__ == "__main__":
    main()
