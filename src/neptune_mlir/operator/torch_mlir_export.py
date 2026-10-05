"""Isolated Torch-MLIR worker for exporting built-in attention variants.

The public ``neptune-export`` CLI invokes this module in a subprocess so
Torch-MLIR and Neptune's standalone MLIR Python runtime are not loaded together.
Torch-MLIR imports stay local so the eager modules can also serve as test references.
"""

import argparse
import math
from collections.abc import Callable

import torch

from neptune_mlir.operator.variants import VARIANTS, AttentionVariant

MaskF = Callable[[torch.Tensor], torch.Tensor | None]


def _f16_matmul_f32(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    # Torch-MLIR exports this as f16 dot + f16-to-f32 convert. The StableHLO
    # fixup below folds the convert into the dot's accumulation type.
    return torch.matmul(lhs.to(torch.float16), rhs.to(torch.float16)).to(torch.float32)


class AttentionModule(torch.nn.Module):
    def __init__(self, mask_f: MaskF):
        super().__init__()
        self.mask_f = mask_f

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        output_shape = q.shape
        q_heads, kv_heads = q.shape[1], k.shape[1]
        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(f"q heads ({q_heads}) must be divisible by kv heads ({kv_heads})")
            groups = q_heads // kv_heads
            q = q.reshape(q.shape[0], groups, kv_heads, q.shape[2], q.shape[3])
            k = k[:, None, :, :, :].expand(k.shape[0], groups, kv_heads, k.shape[2], k.shape[3])
            v = v[:, None, :, :, :].expand(v.shape[0], groups, kv_heads, v.shape[2], v.shape[3])
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = _f16_matmul_f32(q, k.transpose(-1, -2))
        scores = scores * scale
        mask = self.mask_f(scores)
        if mask is not None:
            neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
            scores = torch.where(mask, scores, neg_inf)
        probs = torch.softmax(scores, dim=-1)
        output = _f16_matmul_f32(probs, v).to(torch.float16)
        if q_heads != kv_heads:
            output = output.reshape(output_shape)
        return output


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
        output_shape = q.shape
        q_heads, kv_heads = q.shape[1], k.shape[1]
        if q_heads != kv_heads:
            groups = q_heads // kv_heads
            q = q.reshape(q.shape[0], groups, kv_heads, q.shape[2], q.shape[3])
            k = k[:, None, :, :, :].expand(k.shape[0], groups, kv_heads, k.shape[2], k.shape[3])
            v = v[:, None, :, :, :].expand(v.shape[0], groups, kv_heads, v.shape[2], v.shape[3])
            slopes = slopes.reshape(groups, kv_heads)
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = _f16_matmul_f32(q, k.transpose(-1, -2))
        scores = scores * scale
        *_, q_len, kv_len = scores.shape
        query_pos = torch.arange(q_len, dtype=torch.float32, device=scores.device)
        key_pos = torch.arange(kv_len, dtype=torch.float32, device=scores.device)
        distance = key_pos[None, :] - query_pos[:, None]
        scores = scores + distance * slopes[..., None, None]
        mask = causal_mask(scores)
        neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
        scores = torch.where(mask, scores, neg_inf)
        probs = torch.softmax(scores, dim=-1)
        output = _f16_matmul_f32(probs, v).to(torch.float16)
        if q_heads != kv_heads:
            output = output.reshape(output_shape)
        return output


class KVOnlyQuantizedAttentionModule(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, sv: torch.Tensor
    ) -> torch.Tensor:
        output_shape = q.shape
        q_heads, kv_heads = q.shape[1], k.shape[1]
        k = (k.to(torch.float32) * sk).to(torch.float16)
        v = (v.to(torch.float32) * sv).to(torch.float16)
        if q_heads != kv_heads:
            groups = q_heads // kv_heads
            q = q.reshape(q.shape[0], groups, kv_heads, q.shape[2], q.shape[3])
            k = k[:, None, :, :, :].expand(k.shape[0], groups, kv_heads, k.shape[2], k.shape[3])
            v = v[:, None, :, :, :].expand(v.shape[0], groups, kv_heads, v.shape[2], v.shape[3])
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = _f16_matmul_f32(q, k.transpose(-1, -2))
        scores = scores * scale
        mask = causal_mask(scores)
        neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
        scores = torch.where(mask, scores, neg_inf)
        probs = torch.softmax(scores, dim=-1)
        output = _f16_matmul_f32(probs, v).to(torch.float16)
        if q_heads != kv_heads:
            output = output.reshape(output_shape)
        return output


def _use_f32_accumulation_for_f16_dots(module) -> int:
    """Make attention dots accumulate in f32 while retaining f16 operands.

    Torch-MLIR cannot currently lower ``aten.bmm.dtype``. Exporting an f16
    matmul followed by an f32 cast instead produces:

      %dot = stablehlo.dot_general %lhs, %rhs
          : (tensor<...xf16>, tensor<...xf16>) -> tensor<...xf16>
      %result = stablehlo.convert %dot
          : (tensor<...xf16>) -> tensor<...xf32>

    StableHLO supports expressing the intended accumulation directly:

      %result = stablehlo.dot_general %lhs, %rhs
          : (tensor<...xf16>, tensor<...xf16>) -> tensor<...xf32>

    Only fold a direct, single-use conversion so unrelated dots and conversions
    retain their original semantics.
    """
    from torch_mlir.ir import (
        F16Type,  # type: ignore[import]
        F32Type,  # type: ignore[import]
        Operation,  # type: ignore[import]
        RankedTensorType,  # type: ignore[import]
        WalkResult,  # type: ignore[import]
    )

    converts: list[Operation] = []

    def collect_f16_dot_convert(op: Operation) -> WalkResult:
        if op.name != "stablehlo.convert" or len(op.operands) != 1 or len(op.results) != 1:
            return WalkResult.ADVANCE

        dot_result = op.operands[0]
        dot = getattr(dot_result.owner, "operation", dot_result.owner)
        if not isinstance(dot, Operation) or dot.name != "stablehlo.dot_general":
            return WalkResult.ADVANCE
        if len(dot.operands) != 2:
            return WalkResult.ADVANCE

        values = [*dot.operands, dot_result, op.results[0]]
        if not all(isinstance(value.type, RankedTensorType) for value in values):
            return WalkResult.ADVANCE
        types = [RankedTensorType(value.type) for value in values]
        if not all(isinstance(type_.element_type, F16Type) for type_ in types[:3]):
            return WalkResult.ADVANCE
        if not isinstance(types[3].element_type, F32Type) or types[2].shape != types[3].shape:
            return WalkResult.ADVANCE
        if len(list(dot_result.uses)) != 1:
            return WalkResult.ADVANCE

        converts.append(op)
        return WalkResult.ADVANCE

    module.operation.walk(collect_f16_dot_convert)
    for convert in converts:
        dot_result = convert.operands[0]
        dot_result.set_type(convert.results[0].type)
        convert.results[0].replace_all_uses_with(dot_result)
        convert.erase()
    module.operation.verify()
    return len(converts)


def _use_finite_softmax_max_init(module) -> int:
    """Give built-in attention's f32 row max a finite rolling-softmax seed.

    This is an attention policy, not a general max canonicalization. Only change
    the reduction's -inf init operand: the same constant may also supply masked
    scores, which must remain -inf. All built-in variants have one row max.
    """
    from torch_mlir.ir import (
        DenseElementsAttr,  # type: ignore
        DenseFPElementsAttr,  # type: ignore
        F32Type,  # type: ignore
        FloatAttr,  # type: ignore
        InsertionPoint,  # type: ignore
        Operation,  # type: ignore
        RankedTensorType,  # type: ignore
        WalkResult,  # type: ignore
    )

    reductions: list[Operation] = []

    def collect_row_max(op: Operation) -> WalkResult:
        if op.name != "stablehlo.reduce" or len(op.operands) != 2 or len(op.results) != 1:
            return WalkResult.ADVANCE
        input_type = RankedTensorType(op.operands[0].type)
        init = op.operands[1]
        init_type = RankedTensorType(init.type)
        if (
            not isinstance(init_type.element_type, F32Type)
            or init_type.rank != 0
            or list(op.attributes["dimensions"]) != [input_type.rank - 1]
        ):
            return WalkResult.ADVANCE
        block = op.regions[0].blocks[0]
        body = list(block.operations)
        if (
            len(body) != 2
            or body[0].operation.name != "stablehlo.maximum"
            or list(body[0].operands) not in (list(block.arguments), list(block.arguments)[::-1])
            or body[1].operation.name != "stablehlo.return"
            or list(body[1].operands) != list(body[0].results)
        ):
            return WalkResult.ADVANCE
        constant = getattr(init.owner, "operation", init.owner)
        if not isinstance(constant, Operation) or constant.name != "stablehlo.constant":
            return WalkResult.ADVANCE
        value = constant.attributes["value"]
        if (
            isinstance(value, DenseFPElementsAttr)
            and value.is_splat
            and FloatAttr(value.get_splat_value()).value == -math.inf
        ):
            reductions.append(op)
        return WalkResult.ADVANCE

    module.operation.walk(collect_row_max)
    for reduction in reductions:
        init_type = RankedTensorType(reduction.operands[1].type)
        with module.context, reduction.location, InsertionPoint(reduction):
            value = DenseElementsAttr.get_splat(
                init_type, FloatAttr.get(init_type.element_type, torch.finfo(torch.float32).min)
            )
            seed = Operation.create(
                "stablehlo.constant",
                results=[init_type],
                attributes={"value": value},
                loc=reduction.location,
            )
            reduction.operands[1] = seed.results[0]
    module.operation.verify()
    return len(reductions)


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
        help="number of KV heads (defaults to --q-heads)",
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
    parser.add_argument(
        "--output-type",
        choices=("stablehlo", "linalg"),
        default="stablehlo",
        help="Torch-MLIR output type",
    )
    return parser.parse_args()


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

    masked_variants = {
        AttentionVariant.GLOBAL_ATTN: lambda _: None,
        AttentionVariant.CAUSAL_ATTN: causal_mask,
        AttentionVariant.WINDOWED_CAUSAL_ATTN: windowed_causal_mask(window_size),
    }
    mask_f = masked_variants.get(variant, None)
    if mask_f is not None:
        return AttentionModule(mask_f).eval(), make_qkv()

    if variant == AttentionVariant.ALIBI_CAUSAL_ATTN:
        q, k, v = make_qkv()
        slopes = (torch.arange(q_heads, dtype=fp32) + 1.0) / q_heads
        return AlibiCausalAttentionModule().eval(), (q, k, v, slopes)

    if variant == AttentionVariant.KV_FP8_CAUSAL_ATTN:
        q, k, v = make_qkv()
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        scale_shape = () if kv_heads == 1 else (1, kv_heads, 1, 1)
        sk = torch.randn(scale_shape, dtype=fp32)
        sv = torch.randn(scale_shape, dtype=fp32)
        return KVOnlyQuantizedAttentionModule().eval(), (q, k, v, sk, sv)

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
    output_type: str = "stablehlo",
) -> str:
    from torch_mlir import fx
    from torch_mlir.extras.fx_decomp_util import get_decomposition_table

    # Custom decomposition for aten.arange. The builtin one translates
    # `x + arange(N)` into `x[i] + 0.000 + i`, and we don't want that 0.000.
    decomposition_table = get_decomposition_table()
    decomposition_table[torch.ops.aten.arange.default] = arange_default_iota_then_cast
    kv_heads = q_heads if kv_heads is None else kv_heads
    if q_heads <= 0 or kv_heads <= 0:
        raise ValueError("query and KV head counts must be positive")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q heads ({q_heads}) must be divisible by kv heads ({kv_heads})")
    resolved_kv_seq_len = seq_len if kv_seq_len is None else kv_seq_len
    model, example_args = _build_module_and_args(
        variant, batch, q_heads, kv_heads, seq_len, resolved_kv_seq_len, head_dim, window_size
    )
    exported_program = torch.export.export(model, example_args)
    if output_type not in {"stablehlo", "linalg"}:
        raise ValueError(f"unsupported Torch-MLIR output type: {output_type}")
    torch_mlir_output_type = "linalg-on-tensors" if output_type == "linalg" else output_type
    module = fx.export_and_import(
        exported_program,
        output_type=torch_mlir_output_type,
        func_name=func_name,
        import_symbolic_shape_expressions=True,
        decomposition_table=decomposition_table,
    )
    if output_type == "stablehlo":
        promoted_dot_count = _use_f32_accumulation_for_f16_dots(module)
        if promoted_dot_count != 2:
            raise RuntimeError(
                f"expected to promote two f16 attention dots, promoted {promoted_dot_count}"
            )
        max_init_count = _use_finite_softmax_max_init(module)
        if max_init_count != 1:
            raise RuntimeError(
                f"expected to initialize one f32 attention row max, initialized {max_init_count}"
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
            output_type=args.output_type,
        )
    )


if __name__ == "__main__":
    main()
