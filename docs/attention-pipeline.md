# Attention Lowering Pipeline

This document describes how an attention program is lowered from algorithmic attention to tile
languages like Triton. Each stage in the pipeline has its own design documents, which this
document links to and summarizes. The integrated example for attention lowering is at
[`test/Pipeline/tm_global_attention.mlir`](../test/Pipeline/tm_global_attention.mlir).
Potential next attention variants are tracked in the
[attention variant roadmap](attention-variants.md).

## Pipeline Overview

The pipeline uses these program levels:

- **L0** is the unscheduled algorithmic tensor/linalg program. Attention is written as
  `S = scale * (Q @ K^T)`, `P = softmax(S)`, and `O = P @ V`.
- **L1** is the scheduled, target-independent tile program. The L1 form of attention
  is FlashAttention2-like, with explicit tiling and loop-carried online-softmax state, but
  no GPU-specific details. Output tiling, the K/V streaming loop, and online-softmax loop state
  are explicit, but the program is still upstream `tensor` / `linalg` / `scf` / `arith` / `math` IR.
- **Semantic HTile** rewrites recognizable tile compute ops to HTile ops such as `htile.full`,
  `htile.dot`, and `htile.reduce`, while preserving the tensor ABI and loop structure.
- **Kernel HTile** is the backend-facing form with tile language-specific decisions, like
  bufferization, data placement, software pipelining, etc.

HTile design choices, including the scheduled L1 input contract, placement, view normalization,
and kernel ABI, are documented in [the HTile guide](htile-guide.md).

## L0 To L1 Schedule

We use an MLIR Transform dialect schedule to lower attention to a FlashAttention2-like L1 program.
The [attention scheduling doc](attention-scheduling.md) describes the full L0-to-L1 Transform
dialect schedule, and our custom transform operations that enable it.
The schedule is also given in the integrated example
[`test/Pipeline/tm_global_attention.mlir`](../test/Pipeline/tm_global_attention.mlir).

The scheduled L1 program exposes:

- an outer `scf.forall` over independent `(B, H, M-block)` output tiles,
- an inner sequential `scf.for` over K/V blocks,
- loop-carried online-softmax state `(m, acc, l)`,
- tile-local structured ops for matmul, row reductions, broadcasts, and elementwise arithmetic,
- no GPU hierarchy, memory placement, warp roles, layouts, or fragment details.

The schedule mirrors the FlashAttention online-softmax recurrence:

```text
m_next   = max(m_prev, row_max(score_tile))
p_tile   = exp2(score_tile - m_next)
l_next   = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)
acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile
out      = acc_final / l_final
```

Some key expression rewrites, such as `exp` to `exp2`, are powered by a custom **tensor algebra** (TA)
dialect that makes expression rewriting easier.

## L1 To HTile

HTile is the tile-language-facing IR after scheduled L1. It keeps the same high-level loop shape at
first, but replaces tile compute patterns with HTile operations that are closer to backend tile
language primitives. The [HTile guide](htile-guide.md) describes the dialect forms and lowering
steps in more detail.

The current `transform.htile.linalg_to_semantic` transform produces **Semantic HTile**:

- `linalg.fill` becomes `htile.full`,
- contraction-shaped `linalg.generic` becomes `htile.dot`,
- single-axis sum and max reductions become `htile.reduce`,
- all-parallel elementwise `linalg.generic` ops are opened into tensor `arith` / `math`,
- projected row-vector operands are materialized with `htile.broadcast`.

Semantic HTile deliberately keeps the tensor ABI, `scf.forall` / `scf.for` loop structure,
`tensor.extract_slice`, and `tensor.parallel_insert_slice`. It is a compute-level translation, not
yet a backend kernel ABI.

The later HTile/backend pipeline should:

- add placement decisions such as shared/local tile encodings,
- normalize tensor views into memory-rooted `htile.load` operations,
- convert tensor-return Semantic HTile into Kernel HTile with explicit output memory and
  `htile.store`,
- legalize Kernel HTile to each backend translator's launch form, types, layouts, and software
  pipelining constraints.
