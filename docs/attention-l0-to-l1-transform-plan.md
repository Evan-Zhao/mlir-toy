# L0 to L1 Program Transform Plan

## Goal

Implement an MLIR Transform dialect schedule that lowers the algorithm-only
attention form in `test/Loop/torch_mlir_attention.mlir` into the scheduled tile-level
FlashAttention form in `test/python/data/flash_attention_l1.mlir`.

The implementation should be a real structural transformation, not a
replacement pass that materializes a known output module.

Here, "L0" means a "math-like" algorithmic program:
the whole computation is spelled directly in terms of linalg operations,
(elementwise, reduction, etc.).
No tiling, no explicit streaming loop, and no online-softmax recurrence.
The payload in `test/Loop/torch_mlir_attention.mlir` is exactly this form.

"L1" means the scheduled, target-independent tile form used by the rest of the project:
output tiling is explicit, the outer parallel grid and inner sequential streaming loop are explicit,
and the online-softmax state is carried explicitly as loop state.
GPU hierarchy, memory placement, and other hardware-specific choices are still absent.
See also `docs/tile-ir-level1.md`.

## Scheduling Model

The source program describes attention as:

- `S = scale * (Q @ K^T)`
- `P = softmax(S, dim = j)`
- `O = P @ V`

The scheduled L1 program should expose:

- an outer `scf.forall` over independent `(B, H, M-block)` output tiles,
- an inner sequential `scf.for` over K/V blocks,
- loop-carried online-softmax state `(l, acc, m)`,
- tile-local structured ops for matmul, row reductions, broadcasts, and elementwise arithmetic,
- no GPU hierarchy, memory-placement, warp, or fragment information.

## TVM to MLIR Primitive Map

Note: `ts.` is short for `transform.structured.` (MLIR builtin transforms).

| TVM primitive                            | MLIR plan                                                                                                                           |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `get_block(name)`                        | Avoid relying on frontend block names. Match structurally using Transform dialect matchers and navigation ops [1].                  |
| `tile_loops([i, j])`                     | Works differently from TVM. `ts.tile_using_forall` applies tiling and produces `scf.forall`, which also implies parallel execution. |
| `bind_block_idx([*axes, i0])`            | Implied by `scf.forall` (parallel) or `scf.for` (serial). Correspondence to GPU block/thread is not recorded in L1.                 |
| `reverse_compute_at`                     | Two custom upward-fusion transforms for elementwise consumers and reduction consumers respectively [2].                             |
| `rolling_update`                         | Implement as a sequence of three custom operations, one analysis and two transformations [3].                                       |
| `split_scan_buffer`                      | Not needed in L1. MLIR rolling-update does not generate scan dependency (`x[t] = f(x[t-1], ...)`).                                  |
| `decompose_reduction`                    | _**TBD**_ Likely not needed in L1. Neptune TVM needed it to guide translation from a mem-based IR to value-based tile languages.    |
| `set_scope`, `cache_read`, `cache_write` | Not represented in L1. Defer memory placement to L1-to-HTile lowering.                                                              |
| `to_tile_expr_form`, `mem2reg`           | L1 is already value-based over tile tensors.                                                                                        |
| `cse`                                    | Use MLIR builtins for CSE and canonicalization.                                                                                     |
| `rewrite_expr`                           | Use MLIR pattern rewrites [4].                                                                                                      |

1. We may want custom match ops (to match einsum patterns, for example).
1. See the document on [upward fusion design](upward-fusion-design.md).
1. See the document on [rolling update design](rolling-update-design.md).
1. MLIR pattern rewriter allows expression rewrite to be rather easily implemented in a custom pass,
   but we may need something more powerful and available at the `transform` dialect level later.

## Structural Matching Strategy

Do not require `custom tags` or TVM-style block names in the algorithmic IR.

The useful constraints here are:

1. Match structurally, not by frontend names.
   Prefer use-def navigation plus Linalg indexing maps and iterator types over ad-hoc tags.

1. Let navigation do most of the work. In practice the schedule only needs to match
   a few key "anchor" ops. For attention: the QK contraction,
   the first reduction fused under `scf.forall`, etc.
   Once an anchor op is found, the rest of the chain is often easier to recover
   by following producers and consumers than by re-matching from scratch.
   - This also means transform operations should try to not invalidate handles.

1. Use custom match ops only where indexing-map structure really matters.
   The most plausible future example is an einsum-style matcher for contraction
   shapes such as `...ik,...jk->...ij`.

For an established example of this strategy, see the
[integrated attention transform test](../test/Loop/torch_mlir_attention.mlir) in the codebase.

## Upward Fusion

Upward fusion is the MLIR analogue of TVM `reverse_compute_at`: move a consumer
under the loop nest that already produces tiles of its input,
so the consumer runs directly on those tiles instead of on a larger tensor outside the loop.

We implement upward fusion as a few custom `transform`-dialect operations,
described in a separate [upward fusion design doc](upward-fusion-design.md).

Upstream MLIR does not provide transform-dialect ops for upward fusion.
While it does for downward fusion (producer into consumer),
Neptune cannot work with it because its core transformation (rolling update)
prefers upward fusion of reductions.
At the same time, the implementation is not built from scratch: it leans on upstream SCF
and Linalg helpers, such `scf::tileAndFuseConsumer` which is capable of elementwise fusion.

## Rolling-Update Semantics

Rolling update is the core novel transformation in Neptune
that fuses multiple reductions together.
Because it puts multiple reductions under the same streaming loop,
applying it to attention produces its online counter part (FlashAttention).

Rolling update is implemented as a set of three custom `transform`-dialect operations,
which is different from the implementation in Neptune TVM (a single monolithic pass).
The detailed rolling update design is documented in a
[separate design document](rolling-update-design.md).

## Current Status

The transform stack now has the main custom pieces needed for this schedule:

- custom upward fusion for pointwise and reduction consumers,
- rolling-update analysis plus repair for the softmax and `P @ V` frontiers,
- a working integrated attention transform test in
  [test/Loop/torch_mlir_attention.mlir](../test/Loop/torch_mlir_attention.mlir).

The remaining work is mostly cleanup and generalization to new computation patterns.

## Testing Strategy

Use layered tests:

1. Small synthetic tests for upward fusion and rolling update, especially on
   max, sum, and matmul-like reductions.
2. End-to-end structural tests from the payload in
   `test/Loop/torch_mlir_attention.mlir` to the scheduled L1 shape:
   - contains `scf.forall`,
   - contains `scf.for` with `iter_args`,
   - contains repaired row max and row sum recurrences,
   - contains `P @ V` accumulation,
   - contains the final normalization/cast structure.
3. Golden or structural comparison against `flash_attention_l1.mlir`.
4. Existing HTile/backend translator tests remain downstream checks once L1 is
   lowered further.
