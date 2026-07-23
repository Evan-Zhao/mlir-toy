#!/usr/bin/env python3
"""Export attention-like PyTorch modules to linalg-on-tensors MLIR.

Usage:
  python export_attention.py --variant global-attn > attention.mlir
  python export_attention.py --variant global-gqa > attention_gqa.mlir
  python export_attention.py --variant alibi-causal-attn > attention_alibi.mlir
  python export_attention.py --variant windowed-causal-attn > attention_sw.mlir
  python export_attention.py --variant kv-only-quantized > attention_kv_quant.mlir
  python export_attention.py --variant sparse-mm > sparse_probe.mlir
"""

import argparse
import math
from collections.abc import Callable

import torch
from torch_mlir import fx

from neptune_mlir.operator.variants import VARIANTS, AttentionVariant

MaskF = Callable[[torch.Tensor], torch.Tensor | None]


class Attention4DModule(torch.nn.Module):
    def __init__(self, mask_f: MaskF):
        super().__init__()
        self.mask_f = mask_f

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * scale
        mask = self.mask_f(scores)
        if mask is not None:
            neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
            scores = torch.where(mask, scores, neg_inf)
        probs = torch.softmax(scores, dim=-1)
        out_f32 = torch.matmul(probs, v.to(torch.float32))
        return out_f32.to(torch.float16)


def causal_mask(scores: torch.Tensor) -> torch.Tensor:
    *_, q_len, kv_len = scores.shape
    return torch.ones((q_len, kv_len), dtype=torch.bool, device=scores.device).tril()


def windowed_causal_mask(window_size: int) -> MaskF:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    def mask_f(scores: torch.Tensor) -> torch.Tensor:
        *_, q_len, kv_len = scores.shape
        mask = torch.ones((q_len, kv_len), dtype=torch.bool, device=scores.device)
        return mask.tril().triu(diagonal=1 - window_size)

    return mask_f


class AlibiCausalAttentionModule(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slopes: torch.Tensor
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2))
        scores = scores * scale
        *_, q_len, kv_len = scores.shape
        query_pos = torch.arange(q_len, dtype=torch.float32, device=scores.device)
        key_pos = torch.arange(kv_len, dtype=torch.float32, device=scores.device)
        distance = key_pos[None, :] - query_pos[:, None]
        scores = scores + distance * slopes[:, None, None]
        mask = causal_mask(scores)
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


class KVOnlyQuantizedAttentionModule(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, sv: torch.Tensor
    ) -> torch.Tensor:
        k_dq = k.to(torch.float32) * sk
        v_dq = v.to(torch.float32) * sv
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q.to(torch.float32), k_dq.transpose(-1, -2))
        scores = scores * scale
        mask = causal_mask(scores)
        neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
        scores = torch.where(mask, scores, neg_inf)
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
    parser.add_argument("-s", "--seq-len", type=int, default=128, help="query sequence length")
    parser.add_argument(
        "--kv-seq-len", type=int, default=None, help="K/V sequence length (defaults to --seq-len)"
    )
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
    kv_seq_len: int,
    head_dim: int,
    window_size: int,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    q_shape = (batch, q_heads, seq_len, head_dim)
    kv_shape = (batch, kv_heads, kv_seq_len, head_dim)
    fp16 = torch.float16
    fp32 = torch.float32

    def make_qkv():
        q = torch.randn(q_shape, dtype=fp16)
        k, v = [torch.randn(kv_shape, dtype=fp16) for _ in range(2)]
        return q, k, v

    if variant == AttentionVariant.GLOBAL_GQA:
        return GlobalGQAModule().eval(), make_qkv()

    if q_heads != kv_heads:
        raise ValueError(
            f"q_heads ({q_heads}) must equal kv_heads ({kv_heads}) for {variant.value}"
        )
    masked_variants = {
        AttentionVariant.GLOBAL_ATTN: lambda _: None,
        AttentionVariant.CAUSAL_ATTN: causal_mask,
        AttentionVariant.WINDOWED_CAUSAL_ATTN: windowed_causal_mask(window_size),
    }
    mask_f = masked_variants.get(variant, None)
    if mask_f is not None:
        return Attention4DModule(mask_f).eval(), make_qkv()

    if variant == AttentionVariant.ALIBI_CAUSAL_ATTN:
        q, k, v = make_qkv()
        slopes = (torch.arange(q_heads, dtype=fp32) + 1.0) / q_heads
        return AlibiCausalAttentionModule().eval(), (q, k, v, slopes)

    if variant == AttentionVariant.KV_FP8_CAUSAL_ATTN:
        q, k, v = make_qkv()
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        sk = torch.randn(kv_heads, dtype=fp32).reshape(1, q_heads, 1, 1)
        sv = torch.randn(kv_heads, dtype=fp32).reshape(1, q_heads, 1, 1)
        return KVOnlyQuantizedAttentionModule().eval(), (q, k, v, sk, sv)

    if variant == AttentionVariant.SPARSE_MM:
        indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
        values = torch.tensor([3.0, 4.0, 5.0], dtype=fp32)
        a = torch.sparse_coo_tensor(indices, values, (2, 3))
        b = torch.randn(3, 4, dtype=fp32)
        return SparseMMModule().eval(), (a, b)

    raise ValueError(f"unknown variant: {variant}")


_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def arange_default_iota_then_cast(
    end, *, dtype=None, layout=torch.strided, device=None, pin_memory=False
):
    import torch._prims as prims

    index_dtype = torch.int64
    if dtype not in _FLOAT_DTYPES and dtype is not None:
        index_dtype = dtype
    index = prims.iota(end, start=0, step=1, dtype=index_dtype, device=device, requires_grad=False)
    if index_dtype != dtype:
        index = torch.ops.aten._to_copy.default(index, dtype=dtype, layout=layout, device=device)
    return index


def export_attention(
    *,
    variant: AttentionVariant,
    batch: int = 1,
    q_heads: int = 4,
    kv_heads: int | None = None,
    seq_len: int = 128,
    kv_seq_len: int | None = None,
    head_dim: int = 64,
    window_size: int = 128,
    func_name: str = "attention",
) -> str:
    from torch_mlir.extras.fx_decomp_util import get_decomposition_table

    # Custom decomposition for aten.arange. The builtin one translates
    # `x + arange(N)` into `x[i] + 0.000 + i`, and we don't want that 0.000.
    decomposition_table = get_decomposition_table()
    decomposition_table[torch.ops.aten.arange.default] = arange_default_iota_then_cast
    kv_seq_len = kv_seq_len or seq_len
    model, example_args = _build_module_and_args(
        variant, batch, q_heads, kv_heads or q_heads, seq_len, kv_seq_len, head_dim, window_size
    )
    exported_program = torch.export.export(model, example_args)
    module = fx.export_and_import(
        exported_program,
        output_type="stablehlo",
        func_name=func_name,
        import_symbolic_shape_expressions=True,
        decomposition_table=decomposition_table,
    )
    return _module_to_text(module)


def main():
    args = parse_args()
    print(
        export_attention(
            variant=args.variant,
            batch=args.batch,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            seq_len=args.seq_len,
            kv_seq_len=args.kv_seq_len,
            head_dim=args.head_dim,
            window_size=args.window_size,
            func_name=args.func_name,
        )
    )


if __name__ == "__main__":
    main()
