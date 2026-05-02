# Attention L0 to FlashAttention L1 Transform Plan

## Goal

Implement an MLIR Transform dialect schedule that lowers the algorithm-only
attention form in `tests/data/attention_l0.mlir` into the scheduled tile-level
FlashAttention form in `tests/data/flash_attention_l1.mlir`.

The implementation should be a real structural transformation, not a
replacement pass that materializes a known output module.

## Scheduling Model

The source program describes attention as:

- `S = scale * (Q @ K^T)`
- `P = softmax(S, dim = j)`
- `O = P @ V`

The scheduled L1 program should expose:

- an outer `scf.forall` over independent `(B, H, M-block)` output tiles,
- an inner sequential `affine.for` over K/V blocks,
- loop-carried online-softmax state `(l, acc, m)`,
- tile-local structured ops for matmul, row reductions, broadcasts, and elementwise arithmetic,
- no GPU hierarchy, memory-placement, warp, or fragment information.

## TVM to MLIR Primitive Map

| TVM primitive                            | MLIR plan                                                                                                                                                                                 |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_block(name)`                        | Avoid relying on frontend block names. Match structurally using Transform dialect matchers and custom match ops.                                                                          |
| `tile_loops([i, j])`                     | `transform.structured.tile_using_forall` for outer `(B, H, M)` tiling, then `transform.structured.tile_using_for` or custom tiling into an `affine.for` for the streaming K/V block loop. |
| `bind_block_idx([*axes, i0])`            | Represent as `scf.forall` at L1. GPU block mapping is deferred to later lowering.                                                                                                         |
| `reverse_compute_at`                     | Implement custom upward-fusion transforms. Start with `fuse_map_consumer_into_loop` for pointwise consumers, then add reduction-specific fusion for `scf.forall`. See `docs/attention-upward-fusion-design.md`. |
| `rolling_update`                         | Implement as a custom Transform dialect transform family for online-softmax-style repaired reductions. See `docs/attention-rolling-update-design.md`.                                   |
| `split_scan_buffer`                      | Model as explicit loop-carried tensors in `affine.for iter_args`.                                                                                                                         |
| `decompose_reduction`                    | Use Linalg reduction structure plus explicit initial tensors and loop-carried state.                                                                                                      |
| `set_scope`, `cache_read`, `cache_write` | Not represented in L1. Defer memory placement to L1-to-HTile lowering.                                                                                                                    |
| `to_tile_expr_form`, `mem2reg`           | L1 is already value-based over tile tensors.                                                                                                                                              |
| `rewrite_expr`, `cse`                    | Use canonicalization/CSE plus a targeted rewrite pattern for row-wise `exp2` hoisting if needed.                                                                                          |

## Structural Matching Strategy

Do not require `custom tags` or TVM-style block names in the algorithmic IR.

Use a layered matcher design:

1. **Find the enclosing function**
   Match the target `func.func` from the module root.

2. **Find the QK contraction**
   Match a `linalg.generic` or named contraction with:
    - rank-5 iterator space `(b, h, i, j, k)`,
    - four parallel dimensions and one reduction dimension,
    - two tensor inputs and one tensor output,
    - input maps equivalent to `Q[b,h,i,k]` and `K[b,h,j,k]`,
    - output map equivalent to `S[b,h,i,j]`,
    - body equivalent to f16 extension, multiply, f32 accumulation.

    Prefer upstream structured matchers where possible:
    - `transform.match.structured`
    - `transform.match.structured.rank`
    - `transform.match.structured.num_inputs`
    - `transform.match.structured.num_inits`
    - `transform.match.structured.input`
    - `transform.match.structured.init`
    - `transform.match.structured.body`
    - `transform.match.structured.classify_contraction_dims`

    Add a custom matcher if upstream matchers cannot express the required affine
    map relationship between `Q`, `K`, and `S`.

3. **Find score scaling**
   Navigate from the QK result to its consumers and match an elementwise scale:
    - single tensor input from QK,
    - same output shape as QK,
    - body computes `x * constant`,
    - scale constant is either `1/sqrt(D)` or can be converted to `log2(e)/sqrt(D)` when switching to `exp2`.

    The detailed design for upward fusion is intentionally kept out of this
    document. At this level, the schedule only assumes a custom
    `transform.linalg_ext.fuse_map_consumer_into_loop` primitive that can move
    score scaling under the tiled producer loop. See
    `docs/attention-upward-fusion-design.md`.

4. **Find softmax**
   Prefer decomposing `linalg.softmax` first into explicit max/exp/sum/div ops.

    After decomposition, structurally match:
    - row max reduction over the key dimension,
    - shifted score computation,
    - exp or exp2,
    - row sum reduction over the key dimension,
    - normalization divide.

    If upstream softmax decomposition does not preserve enough navigability,
    introduce a custom matcher:

    ```mlir
    %qk, %scale, %row_max, %exp, %row_sum, %norm =
      transform.match.linalg_ext.attention_softmax_chain %func
        : (!transform.any_op)
       -> (!transform.any_op, !transform.any_op, !transform.any_op,
           !transform.any_op, !transform.any_op, !transform.any_op)
    ```

5. **Find PV contraction**
   Match the second contraction structurally:
    - input 0 is the softmax probability tensor or its f16 cast,
    - input 1 is `V[b,h,j,d]`,
    - reduction dimension is key position `j`,
    - output map is `O[b,h,i,d]`,
    - body performs multiply and f32 accumulation.

6. **Find final cast**
   Match the final `arith.truncf` from f32 output accumulation to f16 result.

### Future Einsum / Einops Matcher

The current playground can match QK structurally with exact Linalg indexing
maps:

```text
Q[b,h,i,k], K[b,h,j,k] -> S[b,h,i,j]
```

This is shape-independent but not rank-polymorphic. It matches the rank-5
attention workload well, but it does not express the more general einsum
pattern:

```text
...ik,...jk->...ij
```

Add a custom matcher for rank-polymorphic einsum/einops structure:

```mlir
%qk = transform.match.linalg_ext.einsum %func
    {pattern = "...ik,...jk->...ij"}
  : (!transform.any_op) -> !transform.any_op
```

The matcher should inspect `linalg::LinalgOp` indexing maps and iterator types,
not tensor extents. For the `...ik,...jk->...ij` case it should verify:

- two inputs and one init/result,
- a shared variadic batch prefix represented by projected permutation dims,
- input 0 has batch dims followed by `i, k`,
- input 1 has batch dims followed by `j, k`,
- output has batch dims followed by `i, j`,
- `k` dims are reductions,
- batch, `i`, and `j` dims are parallel,
- no unexpected extra dimensions or non-projectable indexing expressions.

This matcher should compose with `transform.match.structured.body` or a custom
contraction-body predicate:

```mlir
%qk = transform.match.linalg_ext.einsum %func
    {pattern = "...ik,...jk->...ij"}
  : (!transform.any_op) -> !transform.any_op

%qk_checked = transform.match.structured %qk
    : (!transform.any_op) -> !transform.any_op {
^bb0(%candidate: !transform.any_op):
  transform.match.structured.body %candidate
      {contraction = ["arith.mulf", "arith.addf"]} : !transform.any_op
  transform.match.structured.yield %candidate : !transform.any_op
}
```

Separating indexing-pattern matching from compute-body matching is important:
the same einsum maps can describe integer reductions, max-plus semiring
contractions, boolean reductions, or ordinary floating-point matmul depending
on the body.

Longer term, generalize from einsum to einops-style operators by matching named
axis expressions:

```mlir
%pack = transform.match.linalg_ext.einops %func
    {pattern = "b h (m bm) d -> b h m bm d"}
  : (!transform.any_op) -> !transform.any_op
```

The einops matcher would cover reshape/expand/collapse/transpose/broadcast
families, while the einsum matcher covers reduction contractions. Together with
body predicates, they should be enough to recognize most structural tensor
operators without relying on frontend names or concrete shapes.

## Transform Dialect Schedule Shape

The intended schedule should look conceptually like:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %module: !transform.any_op {transform.consumed}) {
    %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op

    %qk, %scale, %row_max, %exp, %row_sum, %pv, %norm, %cast =
      transform.match.linalg_ext.attention_pattern %func
        : (!transform.any_op)
       -> (!transform.any_op, !transform.any_op, !transform.any_op,
           !transform.any_op, !transform.any_op, !transform.any_op,
           !transform.any_op, !transform.any_op)

    %qk_tiled, %forall =
      transform.structured.tile_using_forall %qk tile_sizes [1, 1, 128, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %qk_streamed, %j_loop =
      transform.structured.tile_using_for %qk_tiled tile_sizes [0, 0, 0, 64, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %scale_fused, %j_loop_1 =
      transform.linalg_ext.fuse_map_consumer_into_loop %scale into %j_loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %row_max_rf =
      transform.linalg_ext.rolling_update %row_max into %j_loop_1
        {factor_axis = 0 : i64}
        : (!transform.any_op, !transform.any_op) -> !transform.any_op

    %row_sum_rf =
      transform.linalg_ext.rolling_update %row_sum into %j_loop_1
        {factor_axis = 0 : i64}
        : (!transform.any_op, !transform.any_op) -> !transform.any_op

    %pv_rf =
      transform.linalg_ext.rolling_update %pv into %j_loop_1
        {factor_axis = 0 : i64}
        : (!transform.any_op, !transform.any_op) -> !transform.any_op

    %norm_fused, %forall_1 =
      transform.structured.fuse_into_containing_op %norm into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %cast_fused, %forall_2 =
      transform.structured.fuse_into_containing_op %cast into %forall_1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}
```

The exact op names and return handles can change during implementation, but the
schedule should remain readable as a structural schedule over matched handles.
Detailed design for the custom upward-fusion ops is in
`docs/attention-upward-fusion-design.md`.

## `rolling_update` Semantics

`rolling_update` is the core non-upstream reduction transform family for the
online-softmax part of FlashAttention. At a high level, it converts reductions
under the streaming loop into repaired loop-carried recurrences such as row
max, row sum, and `P @ V` accumulation.

The detailed decomposition, semantics, and FlashAttention-specific pipeline are
documented in `docs/attention-rolling-update-design.md`.

## Implementation Milestones

1. **Clean up playground state**
   Remove or quarantine any replacement-style L0-to-L1 pass so it remains only
   a test fixture, not the implementation path.

2. **Add structural matchers**
   Start with matchers for:
    - QK contraction,
    - score scaling,
    - softmax chain,
    - PV contraction,
    - final cast.

3. **Add transform extension plumbing**
   Register a LinalgExt/Neptune transform dialect extension that defines:
    - `transform.match.linalg_ext.attention_pattern`,
    - `transform.linalg_ext.fuse_map_consumer_into_loop`,
    - `transform.linalg_ext.rolling_update`,
    - optional cleanup pattern descriptors such as exp2-hoist-across-broadcast.

4. **Implement tiling and fusion without rolling update**
   Verify that the schedule can tile QK and fuse score computation structurally.

5. **Implement rolling update for row max**
   First support max recurrence:
   `m_next = max(m_prev, row_max(score_tile))`.

6. **Implement rolling update for row sum**
   Add the normalizer recurrence:
   `l_next = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)`.

7. **Implement rolling update for PV accumulation**
   Add accumulator repair:
   `acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile`.

8. **Integrate final normalization**
   Fuse final `acc / l` and cast into the `scf.forall` body after the streaming
   loop.

9. **Canonicalize output**
   Apply canonicalization, CSE, and targeted exp2/broadcast rewrites until the
   output shape is stable and close to `flash_attention_l1.mlir`.

## Testing Strategy

Use layered tests:

1. Parser tests for `attention_l0.mlir` and `flash_attention_l1.mlir`.
2. Matcher-only tests that verify the structural matchers find the intended
   handles without tags.
3. Small synthetic tests for `rolling_update` on max, sum, and matmul
   reductions.
4. End-to-end structural test from `attention_l0.mlir` to L1 shape:
    - contains `scf.forall`,
    - contains `affine.for` with `iter_args`,
    - contains row max and row sum reductions,
    - contains PV matmul,
    - contains final normalization.
5. Golden/canonicalized comparison against `flash_attention_l1.mlir`.
6. Existing HTile/backend translator tests remain downstream checks once L1 is
   lowered further.

## Design Guidance

Prefer structural matching over names or tags whenever the property is
recoverable from IR:

- Use op names only for broad filtering, such as `linalg.generic`,
  `linalg.matmul`, `linalg.softmax`, or `arith.truncf`.
- Use use-def navigation to establish that matched ops belong to the same
  attention chain.
- Use Linalg indexing maps and iterator types to distinguish QK from PV.
- Use custom `MatchOpInterface` ops for inferred properties that are too
  cumbersome in plain transform IR.
- Use marker attributes only as optional debugging aids or frontend contracts,
  not as the primary schedule interface.

## Rolling Update Redesign

The detailed rolling-update design has been moved to
`docs/attention-rolling-update-design.md`.

In this plan, `rolling_update` should be understood as a custom reduction
transform family that:

- analyzes frontier reductions under the streaming loop,
- builds a scheduled sidecar computation,
- converts incomplete reductions into repaired loop-carried recurrences,
- publishes the completed scheduled result only at the end.

That document covers the TVM comparison, MLIR decomposition, FlashAttention
pipeline, atomicity boundaries, and autoscheduler implications.
