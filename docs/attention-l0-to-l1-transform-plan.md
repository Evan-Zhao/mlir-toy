# L0 to L1 Program Transform Plan

## Goal

Implement an MLIR Transform dialect schedule that lowers an algorithm-only attention
into a scheduled, tile-level FlashAttention-like form.
The canonical handwritten reference for the tile-level program is
`test/python/data/flash_attention_l1.mlir`.

Currently the test cases under [`test/Pipeline`](../test/Pipeline) move toward that goal.
Each test contains a payload module in the algorithmic form, and a `transform` region with the schedule.

We will refer to this "math-like" algorithmic program as "L0", and the scheduled, tile-level program as "L1".
"L0" means the whole computation is spelled directly in terms of linalg operations (elementwise, reduction, etc.).
No tiling, no explicit streaming loop, and no online-softmax recurrence.
The payloads embedded in `test/Pipeline` test cases are in this form.

"L1" means the scheduled, target-independent tile form used by the rest of the project:
output tiling is explicit, the outer parallel grid and inner sequential streaming loop are explicit,
and the online-softmax state is carried explicitly as loop state.
GPU hierarchy, memory placement, and other hardware-specific choices are still absent.
See also [`docs/tile-ir-level1.md`](./tile-ir-level1.md).

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
| `get_block(name)`                        | Avoid frontend block names. Use linalg structural matchers and TA einsum matching [1].                                              |
| `tile_loops([i, j])`                     | Works differently from TVM. `ts.tile_using_forall` applies tiling and produces `scf.forall`, which also implies parallel execution. |
| `bind_block_idx([*axes, i0])`            | Implied by `scf.forall` (parallel) or `scf.for` (serial). GPU block/thread is not recorded in L1.                                   |
| `reverse_compute_at`                     | Two custom upward-fusion transforms for elementwise consumers and reduction consumers respectively [2].                             |
| `rolling_update`                         | Implement as a sequence of three custom operations, one analysis and two transformations [3].                                       |
| `split_scan_buffer`                      | Not needed in L1. MLIR rolling-update does not generate scan dependency (`x[t] = f(x[t-1], ...)`).                                  |
| `decompose_reduction`                    | Not a separate schedule step. Neptune TVM needed it to guide translation from a mem-based IR to value-based tile languages.         |
| `set_scope`, `cache_read`, `cache_write` | Not represented in L1. Defer memory placement to L1-to-HTile lowering.                                                              |
| `to_tile_expr_form`, `mem2reg`           | L1 is already value-based over tile tensors.                                                                                        |
| `cse`                                    | Use MLIR canonicalization and CSE, including `transform.apply_patterns ... canonicalization` and `transform.apply_cse`.             |
| `rewrite_expr`                           | Use our TA dialect [4] for rewrites: import linalg to TA, run TA expression rewrites, then lower back to linalg.                    |

1. The schedule generalizes named linalg ops, imports the computation into TA,
   and uses the TA expression view for `transform.ta.rewrite_exp_to_exp2`,
   `transform.apply_patterns.ta.exchange_div_and_matmul`, and
   `transform.match.ta.einsum`. See [the TA dialect design](ta-dialect-design.md).
2. See the document on [upward fusion design](upward-fusion-design.md).
3. See the document on [rolling update design](rolling-update-design.md).
4. See [the TA dialect design](ta-dialect-design.md).

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

1. Use custom match ops where indexing-map structure really matters.
   The current schedule uses `transform.match.ta.einsum` to identify both
   attention contractions before lowering TA back to linalg.

For an established example of this strategy, see the
[integrated attention transform test](../test/Pipeline/tm_global_attention.mlir) in the codebase.

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

- [x] TA prepass imports linalg attention into expression form, rewrites `exp`
      to `exp2`, exchanges division and matmul where needed, and lowers back to linalg.
- [x] TA einsum matching identifies the two attention contractions and keeps
      useful handles alive across TA-to-linalg lowering.
- [x] QK is tiled with `transform.structured.tile_using_forall`, producing the
      outer parallel `scf.forall` tile grid.
- [x] Pointwise score scaling is fused into the tiled producer loop.
- [x] Row-max is fused under the outer loop and creates the inner streaming
      `scf.for` over K/V blocks.
- [x] Rolling update repairs the softmax row-sum and `P @ V` frontiers into
      loop-carried online-softmax state.
- [x] Trailing normalization and FP32-to-FP16 cast are fused into the outer
      loop after the streaming loop.
- [x] The integrated global-attention test checks the FlashAttention-like
      structural shape in `test/Pipeline/tm_global_attention.mlir`.
- [ ] Automatic Linalg-to-HTile lowering is still separate work.
      See [Linalg to HTile Translation](translation-to-htile.md).
- [ ] HTile/backend integration should consume the scheduled L1 form rather
      than relying on handwritten HTile examples.
- [ ] The generated L1 is not normalized to the exact handwritten
      `test/python/data/flash_attention_l1.mlir` style; it still uses details such
      as singleton tile dimensions and generic linalg bodies.
- [ ] More general attention variants and shapes still need cleanup and
      generalization beyond the current fixed global-attention pipeline.

## Testing Strategy

Use layered tests:

1. Small synthetic tests for upward fusion and rolling update, especially on
   max, sum, and matmul-like reductions.
2. End-to-end structural tests from the payload in
   `test/Pipeline/tm_global_attention.mlir` to the scheduled L1 shape:
   - contains `scf.forall`,
   - contains `scf.for` with `iter_args`,
   - contains repaired row max and row sum recurrences,
   - contains `P @ V` accumulation,
   - contains the final normalization/cast structure.
3. Golden or structural comparison against `flash_attention_l1.mlir`.
4. Existing HTile/backend translator tests remain downstream checks once L1 is
   lowered further.
