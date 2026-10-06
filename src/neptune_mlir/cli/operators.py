"""Shared operator options for export and benchmark CLIs (no GPU imports)."""

import argparse

from neptune_mlir.operator.variants import VARIANTS, AttentionVariant
from neptune_mlir.schedules import AttentionTileConfig, MambaTileConfig

OPERATORS = ("dense", "varlen", "mamba")


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def add_operator_arguments(parser, operator, *, require_variant=False):
    """Define shapes, variants and schedules once for both command-line tools."""
    if operator in {"dense", "varlen"}:
        parser.add_argument(
            "-d", "--head-dim", type=positive_int, default=64, help="head dimension"
        )
        parser.add_argument("--block-m", type=positive_int, default=128, help="query tile size")
        parser.add_argument("--block-n", type=positive_int, default=64, help="key/value tile size")
        parser.add_argument("--func-name", default="attention", help="exported function name")
    if operator == "dense":
        parser.add_argument(
            "--variant",
            type=AttentionVariant,
            choices=VARIANTS,
            default=AttentionVariant.CAUSAL_ATTN,
            required=require_variant,
            help="dense attention variant",
        )
        parser.add_argument("-b", "--batch", type=positive_int, default=1, help="batch size")
        parser.add_argument(
            "--q-heads", "--heads", type=positive_int, default=4, help="number of query heads"
        )
        parser.add_argument(
            "--kv-heads", type=positive_int, help="KV heads (defaults to --q-heads)"
        )
        parser.add_argument(
            "-s", "--seq-len", type=positive_int, default=128, help="query sequence length"
        )
        parser.add_argument(
            "--kv-seq-len", type=positive_int, help="K/V length (defaults to --seq-len)"
        )
        parser.add_argument(
            "--window-size",
            type=positive_int,
            default=128,
            help="local history window for windowed causal attention",
        )
    elif operator == "varlen":
        parser.add_argument(
            "--num-docs", type=positive_int, default=8, help="number of packed documents"
        )
        parser.add_argument(
            "--total-tokens", type=positive_int, default=1024, help="total number of packed tokens"
        )
        parser.add_argument(
            "--heads", type=positive_int, default=4, help="number of attention heads"
        )
        parser.add_argument(
            "--max-doc-tokens",
            type=positive_int,
            default=512,
            help="static maximum document length",
        )
        parser.add_argument(
            "--index-dtype",
            choices=("int32", "int64"),
            default="int32",
            help="document-offset element type",
        )
    elif operator == "mamba":
        parser.add_argument("-b", "--batch", type=positive_int, default=8, help="batch size")
        parser.add_argument(
            "-s",
            "--sequence-length",
            "--seq-len",
            type=positive_int,
            default=2048,
            help="sequence length",
        )
        parser.add_argument(
            "--model-dim", type=positive_int, default=768, help="base model dimension"
        )
        parser.add_argument(
            "--expand", type=positive_int, default=2, help="channel expansion factor"
        )
        parser.add_argument(
            "--state-dim", type=positive_int, default=16, help="selective state dimension"
        )
        parser.add_argument(
            "--activation-dtype",
            "--dtype",
            choices=("bfloat16", "float16", "float32"),
            default="bfloat16",
            help="activation storage type",
        )
        parser.add_argument(
            "--block-channels",
            type=positive_int,
            default=128,
            help="channels handled by each program",
        )
        parser.add_argument("--func-name", default="selective_scan", help="exported function name")
    else:
        raise ValueError(f"unknown operator: {operator}")


def operator_options(operator, values):
    """Select only operator fields from a CLI namespace, filling shared defaults."""
    parser = argparse.ArgumentParser(add_help=False)
    add_operator_arguments(parser, operator)
    defaults = vars(parser.parse_args([]))
    return {name: values.get(name, default) for name, default in defaults.items()}


def validate_runtime_options(operator, options):
    """Constraints of the current GPU schedules, not limits on StableHLO export."""
    o = argparse.Namespace(**options)
    for name, value in options.items():
        if isinstance(value, int) and value <= 0:
            raise ValueError(f"{name} must be positive")

    def power_of_two(value, name):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a power of two for the benchmark backends")

    if operator in {"dense", "varlen"}:
        AttentionTileConfig(o.block_m, o.block_n).validate()
        power_of_two(o.block_m, "block_m")
        power_of_two(o.block_n, "block_n")
        power_of_two(o.head_dim, "head_dim")
        if o.head_dim < 16:
            raise ValueError("head_dim must be >= 16")
        if operator == "dense":
            AttentionVariant(o.variant)
            kh = o.kv_heads or o.q_heads
            ks = o.kv_seq_len or o.seq_len
            if o.q_heads % kh:
                raise ValueError("q_heads must be divisible by kv_heads")
            if o.seq_len % o.block_m or ks % o.block_n:
                raise ValueError(
                    "query/KV lengths must be divisible by block_m/block_n respectively"
                )
        else:
            if o.index_dtype not in {"int32", "int64"}:
                raise ValueError("index_dtype must be int32 or int64")
            if o.total_tokens > o.num_docs * o.max_doc_tokens:
                raise ValueError("total_tokens exceeds num_docs * max_doc_tokens")
    elif operator == "mamba":
        MambaTileConfig(o.block_channels).validate()
        power_of_two(o.block_channels, "block_channels")
        power_of_two(o.state_dim, "state_dim")
        channels = o.model_dim * o.expand
        if channels % o.block_channels:
            raise ValueError("expanded channels must be divisible by block_channels")
        if o.batch == 1 or o.sequence_length == 1 or channels == o.block_channels:
            raise ValueError(
                "Mamba schedule requires batch > 1, sequence_length > 1 and > 1 channel block"
            )
        if o.activation_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("unsupported activation_dtype")
    else:
        raise ValueError(f"unknown operator: {operator}")
