#!/usr/bin/env python3
"""Export a minimal attention module from PyTorch to linalg-on-tensors MLIR.

Usage:
  python scripts/export_attention_linalg.py > attention.mlir

Notes:
  - This uses an explicit attention spelling (`matmul`, `softmax`, `matmul`)
    instead of `scaled_dot_product_attention`.
  - The sample shape is specialized into the exported IR. Use small defaults for
    fast local export, or override `--b/--h/--n/--d` if you want a specific
    static shape in the emitted MLIR.
"""

import argparse
import math

import torch
from torch_mlir import fx


class AttentionModule(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        vf = v.to(torch.float32)

        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(qf, kf.transpose(-1, -2))
        scores = scores * scale

        # Spell stable softmax explicitly so export does not go through a
        # max-with-indices decomposition that leaves an unused argmax result.
        # row_max = torch.amax(scores, dim=-1, keepdim=True)
        # shifted = scores - row_max
        # exp_scores = torch.exp(shifted)
        # row_sum = torch.sum(exp_scores, dim=-1)
        # probs = exp_scores / row_sum[..., None]
        probs = torch.softmax(scores, dim=-1)

        # Keep the cast boundary explicit to stay closer to Neptune's L0 example.
        probs_f16 = probs.to(torch.float16)
        out_f32 = torch.matmul(probs_f16.to(torch.float32), vf)
        return out_f32.to(torch.float16)


def _module_to_text(module) -> str:
    op = getattr(module, "operation", None)
    if op is not None and hasattr(op, "get_asm"):
        return op.get_asm()
    return str(module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=int, default=1, help="batch size")
    parser.add_argument("--h", type=int, default=4, help="number of heads")
    parser.add_argument("--n", type=int, default=128, help="sequence length")
    parser.add_argument("--d", type=int, default=64, help="head dimension")
    parser.add_argument(
        "--func-name",
        default="attention",
        help="symbol name for the exported MLIR function",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed used for sample inputs during export",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    model = AttentionModule().eval()
    shape = (args.b, args.h, args.n, args.d)
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
