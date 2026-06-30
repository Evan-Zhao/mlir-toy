# Attention Scheduling

This note records how the Transform dialect schedule lowers algorithmic attention from L0 into
the scheduled L1 form used by the downstream pipeline.

## Goal

Implement an MLIR Transform dialect schedule that lowers an algorithm-only attention program into
a scheduled, tile-level FlashAttention-like form.

The integrated schedule lives in the pipeline test
[`test/Pipeline/tm_global_attention.mlir`](../test/Pipeline/tm_global_attention.mlir).
The test contains a payload module in algorithmic form and a `transform` region with the schedule.

## Scheduling Model

The source program describes attention as:

- `S = scale * (Q @ K^T)`
- `P = softmax(S, dim = j)`
- `O = P @ V`

The scheduled L1 program should expose:

- an outer `scf.forall` over independent `(B, H, M-block)` output tiles,
- an inner sequential `scf.for` over K/V blocks,
- loop-carried online-softmax state `(m, acc, l)`,
- tile-local structured ops for matmul, row reductions, broadcasts, and elementwise arithmetic,
- no GPU hierarchy, memory-placement, warp, or fragment information.

## TVM To MLIR Primitive Map

Note: `ts` is short for `transform.structured`.

| TVM primitive                            | MLIR plan                                                                                                                           |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `get_block(name)`                        | Avoid frontend block names. Use linalg structural matchers and TA einsum matching [1].                                              |
| `tile_loops([i, j])`                     | Works differently from TVM. `ts.tile_using_forall` applies tiling and produces `scf.forall`, which also implies parallel execution. |
| `bind_block_idx([*axes, i0])`            | Implied by `scf.forall` (parallel) or `scf.for` (serial). GPU block/thread is not recorded in L1.                                   |
| `reverse_compute_at`                     | Two custom upward-fusion transforms for elementwise consumers and reduction consumers respectively [2].                             |
| `rolling_update`                         | Implement as a sequence of three custom operations, one analysis and two transformations [3].                                       |
| `split_scan_buffer`                      | Not needed in L1. MLIR rolling-update does not generate scan dependency (`x[t] = f(x[t-1], ...)`).                                  |
| `decompose_reduction`                    | Not a separate schedule step. Neptune TVM needed it to guide translation from a mem-based IR to value-based tile languages.         |
| `set_scope`, `cache_read`, `cache_write` | Not represented in L1. Defer memory placement to later HTile placement and kernel lowering.                                         |
| `to_tile_expr_form`, `mem2reg`           | L1 is already value-based over tile tensors.                                                                                        |
| `cse`                                    | Use MLIR canonicalization and CSE, including `transform.apply_patterns ... canonicalization` and `transform.apply_cse`.             |
| `rewrite_expr`                           | Use TA for rewrites: import linalg to TA, run TA expression rewrites, then lower back to linalg.                                    |

1. The schedule generalizes named linalg ops, imports the computation into TA, and uses the TA
   expression view for `transform.ta.rewrite_exp_to_exp2`,
   `transform.apply_patterns.ta.sink_div_after_matmul`, and `transform.match.ta.einsum`.
   See [the TA dialect design](ta-dialect-design.md).
1. See [the upward fusion design](upward-fusion-design.md).
1. See [the rolling update design](rolling-update-design.md).

## Structural Matching Strategy

Do not require `custom tags` or TVM-style block names in the algorithmic IR.

The useful constraints are:

1. Match structurally, not by frontend names.
   Prefer use-def navigation plus Linalg indexing maps and iterator types over ad-hoc tags.

1. Let navigation do most of the work.
   In practice, the schedule only needs to match a few key anchor ops: the QK contraction,
   the first reduction fused under `scf.forall`, and the later reduction frontiers. Once an
   anchor op is found, the rest of the chain is often easier to recover by following producers
   and consumers than by re-matching from scratch.

1. Use custom match ops where indexing-map structure really matters.
   The current schedule uses `transform.match.ta.einsum` to identify both attention
   contractions before lowering TA back to linalg.

The integrated schedule in `test/Pipeline/tm_global_attention.mlir` is the established example of
the structural matching strategy used here.

## Schedule Outline

The current integrated schedule proceeds as follows:

1. Generalize named linalg ops.
1. Translate the payload from linalg to TA.
1. Rewrite `exp(x)` to `exp2(x * log2(e))`.
1. Exchange division and matmul where needed so final normalization can move after `P @ V`.
1. Match the two attention contractions with `transform.match.ta.einsum`.
1. Lower TA back to linalg while preserving useful handles.
1. Inline elementwise producers into the contraction anchors.
1. Tile the QK contraction with `transform.structured.tile_using_forall`.
1. Fuse score scaling upward into the tiled QK loop.
1. Fuse row-max upward into the outer loop, creating the inner streaming `scf.for`.
1. Apply rolling update to repair the next reduction frontier.
1. Apply rolling update again for the remaining reduction frontier.
1. Fuse final normalization and FP32-to-FP16 truncation into the outer loop.
1. Run canonicalization, localize scratch tensors, fold unit extent dimensions, and CSE.

## Upward Fusion

Upward fusion is the MLIR analogue of TVM `reverse_compute_at`: move a consumer under the loop
nest that already produces tiles of its input, so the consumer runs directly on those tiles instead
of on a larger tensor outside the loop.

The schedule uses custom `transform`-dialect operations for this. See
[the upward fusion design](upward-fusion-design.md) for the detailed implementation contract.

## Rolling-Update Semantics

Rolling update is the transformation that fuses multiple reductions under the same streaming loop.
For attention, this creates the online-softmax recurrence.

Rolling update is implemented as three custom `transform`-dialect operations: one analysis and two
transformations. See [the rolling update design](rolling-update-design.md) for details.

## Current Status

The L0-to-L1 schedule is implemented for the current global attention shape.

- [x] TA prepass imports linalg attention into expression form, rewrites `exp` to `exp2`,
      exchanges division and matmul where needed, and lowers back to linalg.
- [x] TA einsum matching identifies the two attention contractions across TA-to-linalg lowering.
- [x] QK is tiled with `transform.structured.tile_using_forall` to produce `scf.forall`.
- [x] Pointwise score scaling is fused into the tiled producer loop.
- [x] Row-max is fused under the outer loop and creates the inner streaming `scf.for`.
- [x] Rolling update repairs row-sum and `P @ V` frontiers into online-softmax state.
- [x] Trailing normalization and FP32-to-FP16 cast are fused after the streaming loop.
- [x] The integrated pipeline test checks the FlashAttention-like structural shape.
