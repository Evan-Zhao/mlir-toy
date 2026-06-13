#!/usr/bin/env python3
"""Export attention-like PyTorch modules to linalg-on-tensors MLIR.

Usage:
  python export_attention_linalg.py --variant global-attn > attention.mlir
  python export_attention_linalg.py --variant global-gqa > attention_gqa.mlir
  python export_attention_linalg.py --variant alibi-causal-attn > attention_alibi.mlir
  python export_attention_linalg.py --variant windowed-causal-attn > attention_sw.mlir
  python export_attention_linalg.py --variant sparse-mm > sparse_probe.mlir
"""

import argparse
import math

import torch
from torch_mlir import fx

from neptune_mlir.operator.variants import VARIANTS, AttentionVariant


class GlobalAttentionModule(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * scale
        probs = torch.softmax(scores, dim=-1)
        out_f32 = torch.matmul(probs, v.to(torch.float32))
        return out_f32.to(torch.float16)


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


class AlibiCausalAttentionModule(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slopes: torch.Tensor
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * scale
        q_len = scores.shape[-2]
        kv_len = scores.shape[-1]
        q_heads = scores.shape[1]
        query_pos = torch.arange(q_len, dtype=torch.float32, device=scores.device)
        key_pos = torch.arange(kv_len, dtype=torch.float32, device=scores.device)
        distance = key_pos.view(1, 1, 1, kv_len) - query_pos.view(1, 1, q_len, 1)
        scores = scores + distance * slopes.view(1, q_heads, 1, 1)
        mask = torch.ones((q_len, kv_len), dtype=torch.bool, device=scores.device).tril()
        mask = mask.view(1, 1, q_len, kv_len)
        neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
        scores = torch.where(mask, scores, neg_inf)
        probs = torch.softmax(scores, dim=-1)
        out_f32 = torch.matmul(probs, v.to(torch.float32))
        return out_f32.to(torch.float16)


class SlidingWindowCausalAttentionModule(torch.nn.Module):
    def __init__(self, window_size: int):
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * scale
        q_len = scores.shape[-2]
        kv_len = scores.shape[-1]
        mask = torch.ones((q_len, kv_len), dtype=torch.bool, device=scores.device)
        mask = mask.tril().triu(diagonal=1 - self.window_size)
        mask = mask.view(1, 1, q_len, kv_len)
        neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
        scores = torch.where(mask, scores, neg_inf)
        probs = torch.softmax(scores, dim=-1)
        out_f32 = torch.matmul(probs, v.to(torch.float32))
        return out_f32.to(torch.float16)


class GlobalGQAModule(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        if q_heads % kv_heads != 0:
            raise ValueError(f"q heads ({q_heads}) must be divisible by kv heads ({kv_heads})")
        groups = q_heads // kv_heads
        q = q.reshape(q.shape[0], groups, kv_heads, q.shape[2], q.shape[3])
        k = k[:, None, :, :, :].expand(k.shape[0], groups, kv_heads, k.shape[2], k.shape[3])
        v = v[:, None, :, :, :].expand(v.shape[0], groups, kv_heads, v.shape[2], v.shape[3])
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * scale
        probs = torch.softmax(scores, dim=-1)
        out_f32 = torch.matmul(probs, v.to(torch.float32))
        return out_f32.to(torch.float16).reshape(q.shape)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        type=AttentionVariant,
        choices=VARIANTS,
        required=True,
        help="attention variant to export",
    )
    parser.add_argument("-b", "--batch", type=int, default=1, help="batch size")
    parser.add_argument("--q-heads", type=int, default=4, help="number of heads")
    parser.add_argument(
        "--kv-heads",
        type=int,
        default=None,
        help="number of KV heads for GQA variants (defaults to --q-heads)",
    )
    parser.add_argument("-s", "--seq-len", type=int, default=128, help="sequence length")
    parser.add_argument("-d", "--head-dim", type=int, default=64, help="head dimension")
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="local history window for windowed causal attention",
    )
    parser.add_argument(
        "--func-name",
        default="attention",
        help="symbol name for the exported MLIR function",
    )
    return parser.parse_args()


def _kv_heads(heads: int, kv_heads: int | None) -> int:
    if kv_heads is not None:
        resolved_kv_heads = kv_heads
    elif heads % 2 == 0:
        resolved_kv_heads = heads // 2
    else:
        resolved_kv_heads = 1
    if resolved_kv_heads <= 0:
        raise ValueError("--kv-heads must be positive")
    if heads % resolved_kv_heads != 0:
        raise ValueError("--heads must be divisible by --kv-heads for GQA variants")
    return resolved_kv_heads


def _build_module_and_args(
    variant: AttentionVariant,
    batch: int,
    q_heads: int,
    kv_heads: int,
    seq_len: int,
    head_dim: int,
    window_size: int,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    q_shape = (batch, q_heads, seq_len, head_dim)

    if variant == AttentionVariant.GLOBAL_GQA:
        q = torch.randn(q_shape, dtype=torch.float16)
        kv_shape = (batch, kv_heads, seq_len, head_dim)
        k = torch.randn(kv_shape, dtype=torch.float16)
        v = torch.randn(kv_shape, dtype=torch.float16)
        return GlobalGQAModule().eval(), (q, k, v)

    if variant == AttentionVariant.GLOBAL_ATTN:
        example_args = tuple(torch.randn(q_shape, dtype=torch.float16) for _ in range(3))
        return GlobalAttentionModule(), example_args

    if variant == AttentionVariant.CAUSAL_ATTN:
        example_args = tuple(torch.randn(q_shape, dtype=torch.float16) for _ in range(3))
        return CausalAttentionModule(), example_args

    if variant == AttentionVariant.ALIBI_CAUSAL_ATTN:
        q, k, v = (torch.randn(q_shape, dtype=torch.float16) for _ in range(3))
        slopes = (torch.arange(q_heads, dtype=torch.float32) + 1.0) / q_heads
        return AlibiCausalAttentionModule().eval(), (q, k, v, slopes)

    if variant == AttentionVariant.WINDOWED_CAUSAL_ATTN:
        example_args = tuple(torch.randn(q_shape, dtype=torch.float16) for _ in range(3))
        return SlidingWindowCausalAttentionModule(window_size).eval(), example_args

    if variant == AttentionVariant.FLOAT8_INPUTS:
        q = torch.randn(q_shape, dtype=torch.float32).to(torch.float8_e4m3fn)
        k = torch.randn(q_shape, dtype=torch.float32).to(torch.float8_e4m3fn)
        v = torch.randn(q_shape, dtype=torch.float32).to(torch.float8_e4m3fn)
        return Float8InputAttentionModule().eval(), (q, k, v)

    if variant == AttentionVariant.FAKE_QUANT:
        q = torch.randn(q_shape, dtype=torch.float32)
        k = torch.randn(q_shape, dtype=torch.float32)
        v = torch.randn(q_shape, dtype=torch.float32)
        sq = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float32)
        sk = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float32)
        sv = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float32)
        return FakeQuantAttentionModule().eval(), (q, k, v, sq, sk, sv)

    if variant == AttentionVariant.SPARSE_MM:
        indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
        values = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
        a = torch.sparse_coo_tensor(indices, values, (2, 3))
        b = torch.randn(3, 4, dtype=torch.float32)
        return SparseMMModule().eval(), (a, b)

    raise ValueError(f"unknown variant: {variant}")


def export_attention_linalg(
    *,
    variant: AttentionVariant,
    batch: int = 1,
    q_heads: int = 4,
    kv_heads: int | None = None,
    seq_len: int = 128,
    head_dim: int = 64,
    window_size: int = 128,
    func_name: str = "attention",
) -> str:
    model, example_args = _build_module_and_args(
        variant, batch, q_heads, kv_heads or q_heads, seq_len, head_dim, window_size
    )
    exported_program = torch.export.export(model, example_args)
    # Possible to control decomposition behavior by passing this `decomp_table` to
    # `run_decompositions`. This will be needed when we use SDPA.
    # decomp_table = torch.export.default_decompositions().materialize()
    # exported_program = exported_program.run_decompositions(decomp_table)
    module = fx.export_and_import(
        exported_program,
        output_type="linalg-on-tensors",
        func_name=func_name,
        import_symbolic_shape_expressions=True,
    )
    return _module_to_text(module)


def main():
    args = parse_args()
    print(
        export_attention_linalg(
            variant=args.variant,
            batch=args.batch,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            window_size=args.window_size,
            func_name=args.func_name,
        )
    )


if __name__ == "__main__":
    main()
