#!/usr/bin/env python3
"""Export a minimal attention module from PyTorch to linalg-on-tensors MLIR.

Usage:
  python scripts/export_attention_linalg.py > attention.mlir
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


def _module_to_text(module) -> str:
    op = getattr(module, "operation", None)
    if op is not None and hasattr(op, "get_asm"):
        return op.get_asm()
    return str(module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-b", "--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=4, help="number of heads")
    parser.add_argument("-s", "--seq-len", type=int, default=128, help="sequence length")
    parser.add_argument("-d", "--dhead", type=int, default=64, help="head dimension")
    parser.add_argument(
        "--func-name",
        default="attention",
        help="symbol name for the exported MLIR function",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model = AttentionModule().eval()
    shape = (args.batch, args.heads, args.seq_len, args.dhead)
    example_args = tuple(torch.randn(shape, dtype=torch.float16) for _ in range(3))

    exported_program = torch.export.export(model, example_args)
    exported_program = exported_program.run_decompositions()
    module = fx.export_and_import(
        exported_program,
        output_type="linalg-on-tensors",
        func_name=args.func_name,
        import_symbolic_shape_expressions=True,
    )
    print(_module_to_text(module))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
