#!/usr/bin/env python3
"""Export attention-like PyTorch modules to linalg-on-tensors MLIR.

Usage:
  python scripts/export_attention_linalg.py > attention.mlir
  python scripts/export_attention_linalg.py --variant global-gqa > attention_gqa.mlir
  python scripts/export_attention_linalg.py --variant sparse-mm > sparse_probe.mlir
"""

import argparse
import math

import torch
from torch_mlir import fx


class AttentionModule(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * scale
        probs = torch.softmax(scores, dim=-1)
        out_f32 = torch.matmul(probs, v.to(torch.float32))
        return out_f32.to(torch.float16)


class ManualGQAAttentionModule(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        if q_heads % kv_heads != 0:
            raise ValueError(f"q heads ({q_heads}) must be divisible by kv heads ({kv_heads})")
        groups = q_heads // kv_heads
        q = q.reshape(q.shape[0], groups, kv_heads, q.shape[2], q.shape[3])
        k = k[:, :, None, :, :].expand(k.shape[0], kv_heads, groups, k.shape[2], k.shape[3])
        v = v[:, :, None, :, :].expand(v.shape[0], kv_heads, groups, v.shape[2], v.shape[3])
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * scale
        probs = torch.softmax(scores, dim=-1)
        out_f32 = torch.matmul(probs, v.to(torch.float32))
        return out_f32.to(torch.float16).reshape(q.shape)


class CausalAttentionModule(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * scale
        q_len = scores.shape[-2]
        kv_len = scores.shape[-1]
        mask = torch.ones((q_len, kv_len), dtype=torch.bool, device=scores.device).tril()
        mask = mask.view(1, 1, q_len, kv_len)
        neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
        scores = torch.where(mask, scores, neg_inf)
        probs = torch.softmax(scores, dim=-1)
        out_f32 = torch.matmul(probs, v.to(torch.float32))
        return out_f32.to(torch.float16)


class Float8InputAttentionModule(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v.to(torch.float32))


class FakeQuantAttentionModule(torch.nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sq: torch.Tensor,
        sk: torch.Tensor,
        sv: torch.Tensor,
    ) -> torch.Tensor:
        q_q = torch.clamp(torch.round(q / sq), -448, 448).to(torch.int16)
        k_q = torch.clamp(torch.round(k / sk), -448, 448).to(torch.int16)
        v_q = torch.clamp(torch.round(v / sv), -448, 448).to(torch.int16)
        q_dq = q_q.to(torch.float32) * sq
        k_dq = k_q.to(torch.float32) * sk
        v_dq = v_q.to(torch.float32) * sv
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q_dq, k_dq.transpose(-1, -2))
        scores = scores * scale
        probs = torch.softmax(scores, dim=-1)
        out_f32 = torch.matmul(probs, v_dq)
        return out_f32.to(torch.float16)


class SparseMMModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(a, b)


def _module_to_text(module) -> str:
    op = getattr(module, "operation", None)
    if op is not None and hasattr(op, "get_asm"):
        return op.get_asm()
    return str(module)


VARIANTS = (
    "global-attn",
    "causal-attn",
    "global-gqa",
    "float8-inputs",
    "fake-quant",
    "sparse-mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        required=True,
        help="attention variant to export",
    )
    parser.add_argument("-b", "--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=4, help="number of heads")
    parser.add_argument(
        "--kv-heads",
        type=int,
        default=None,
        help="number of KV heads for GQA variants (defaults to half of --heads when possible)",
    )
    parser.add_argument("-s", "--seq-len", type=int, default=128, help="sequence length")
    parser.add_argument("-d", "--dhead", type=int, default=64, help="head dimension")
    parser.add_argument(
        "--func-name",
        default="attention",
        help="symbol name for the exported MLIR function",
    )
    return parser.parse_args()


def _kv_heads(args: argparse.Namespace) -> int:
    if args.kv_heads is not None:
        kv_heads = args.kv_heads
    elif args.heads % 2 == 0:
        kv_heads = args.heads // 2
    else:
        kv_heads = 1
    if kv_heads <= 0:
        raise ValueError("--kv-heads must be positive")
    if args.heads % kv_heads != 0:
        raise ValueError("--heads must be divisible by --kv-heads for GQA variants")
    return kv_heads


def _build_module_and_args(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    dense_shape = (args.batch, args.heads, args.seq_len, args.dhead)

    if args.variant == "global-attn":
        module = AttentionModule().eval()
        example_args = tuple(torch.randn(dense_shape, dtype=torch.float16) for _ in range(3))
        return module, example_args

    if args.variant == "causal-attn":
        q = torch.randn(dense_shape, dtype=torch.float16)
        k = torch.randn(dense_shape, dtype=torch.float16)
        v = torch.randn(dense_shape, dtype=torch.float16)
        return CausalAttentionModule().eval(), (q, k, v)

    if args.variant == "global-gqa":
        kv_heads = _kv_heads(args)
        q = torch.randn(dense_shape, dtype=torch.float16)
        kv_shape = (args.batch, kv_heads, args.seq_len, args.dhead)
        k = torch.randn(kv_shape, dtype=torch.float16)
        v = torch.randn(kv_shape, dtype=torch.float16)
        return ManualGQAAttentionModule().eval(), (q, k, v)

    if args.variant == "float8-inputs":
        q = torch.randn(dense_shape, dtype=torch.float32).to(torch.float8_e4m3fn)
        k = torch.randn(dense_shape, dtype=torch.float32).to(torch.float8_e4m3fn)
        v = torch.randn(dense_shape, dtype=torch.float32).to(torch.float8_e4m3fn)
        return Float8InputAttentionModule().eval(), (q, k, v)

    if args.variant == "fake-quant":
        q = torch.randn(dense_shape, dtype=torch.float32)
        k = torch.randn(dense_shape, dtype=torch.float32)
        v = torch.randn(dense_shape, dtype=torch.float32)
        sq = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float32)
        sk = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float32)
        sv = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float32)
        return FakeQuantAttentionModule().eval(), (q, k, v, sq, sk, sv)

    if args.variant == "sparse-mm":
        indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
        values = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
        a = torch.sparse_coo_tensor(indices, values, (2, 3))
        b = torch.randn(3, 4, dtype=torch.float32)
        return SparseMMModule().eval(), (a, b)

    raise ValueError(f"unknown variant: {args.variant}")


def main():
    args = parse_args()
    model, example_args = _build_module_and_args(args)

    exported_program = torch.export.export(model, example_args)
    # Possible to control decomposition behavior by passing this `decomp_table` to `run_decompositions`.
    # Now we don't run this decomposition step because we don't need it.
    # But it will be needed when we use SDPA.
    # decomp_table = torch.export.default_decompositions().materialize()
    # exported_program = exported_program.run_decompositions(decomp_table)
    module = fx.export_and_import(
        exported_program,
        output_type="linalg-on-tensors",
        func_name=args.func_name,
        import_symbolic_shape_expressions=True,
    )
    print(_module_to_text(module))


if __name__ == "__main__":
    main()
