#!/usr/bin/env python3
"""Export built-in attention variants at different compiler pipeline stages."""

import argparse
import ast

from neptune_mlir.operator.variants import VARIANTS, AttentionVariant
from neptune_mlir.pipeline import export_attention_mlir, export_attention_to_triton_input_mlir
from neptune_mlir.schedules import AttentionTileConfig

STAGES = ("stablehlo", "linalg", "htile", "triton", "tilelang", "cutile")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="stablehlo",
        help="pipeline stage to print (default: stablehlo)",
    )
    parser.add_argument(
        "--variant",
        type=AttentionVariant,
        choices=VARIANTS,
        required=True,
        help="attention variant to export",
    )
    parser.add_argument("-b", "--batch", type=int, default=1, help="batch size")
    parser.add_argument("--q-heads", type=int, default=4, help="number of query heads")
    parser.add_argument(
        "--kv-heads",
        type=int,
        default=None,
        help="number of KV heads (defaults to --q-heads)",
    )
    parser.add_argument("-s", "--seq-len", type=int, default=128, help="query sequence length")
    parser.add_argument(
        "--kv-seq-len",
        type=int,
        default=None,
        help="K/V sequence length (defaults to --seq-len)",
    )
    parser.add_argument("-d", "--head-dim", type=int, default=64, help="head dimension")
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="local history window for windowed causal attention",
    )
    parser.add_argument("--block-m", type=int, default=128, help="query tile size")
    parser.add_argument("--block-n", type=int, default=64, help="key/value tile size")
    parser.add_argument("--func-name", default="attention", help="exported function name")
    return parser.parse_args()


def export_at_stage(args: argparse.Namespace) -> str:
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
    if args.stage in {"stablehlo", "linalg"}:
        return export_attention_mlir(**common_args, output_type=args.stage)

    tile_config = AttentionTileConfig(block_m=args.block_m, block_n=args.block_n)
    lowered = export_attention_to_triton_input_mlir(**common_args, tile_config=tile_config)
    if args.stage == "htile":
        return lowered

    if args.stage == "triton":
        from neptune_mlir.translators.triton import translate_mlir_text
    elif args.stage == "tilelang":
        from neptune_mlir.translators.tilelang import translate_mlir_text
    else:
        from neptune_mlir.translators.cutile import translate_mlir_text
    return ast.unparse(translate_mlir_text(lowered)) + "\n"


def main() -> None:
    print(export_at_stage(parse_args()), end="")


if __name__ == "__main__":
    main()
