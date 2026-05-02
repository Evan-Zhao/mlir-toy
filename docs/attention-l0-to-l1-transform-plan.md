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
| `rolling_update`                         | Implement as a custom Transform dialect op, e.g. `transform.linalg_ext.rolling_update`.                                                                                                   |
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

`rolling_update` is the core non-upstream primitive. It generalizes the
FlashAttention online-softmax rewrite.

Given a reduction-like consumer under a streaming loop, it should:

1. Walk the dataflow graph from the target block to find reduce producers that
   are incomplete under the target loop.
2. Identify the spatial producer chain between the target and those reduce
   producers.
3. Fuse that spatial chain under the streaming loop.
4. Convert the relevant reductions into scan-like loop-carried state.
5. R-factor the target reduction over the streaming loop.
6. Repair the writeback algebraically.

For attention, this creates:

```text
m_next   = max(m_prev, row_max(score_tile))
p_tile   = exp2(score_tile - m_next)
l_next   = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)
acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile
out      = acc_final / l_final
```

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

The TVM `rolling_update` primitive is intentionally large. It finds frontier
reductions, fuses a producer chain, converts reductions to scans, derives an
algebraic repair expression, performs rfactor, and patches the writeback. This
is atomic in TVM because each sub-step mutates the only live TIR buffer graph.
Stopping halfway can expose an invalid state where downstream blocks consume
incomplete producer values.

MLIR gives us a better decomposition point. Tensor SSA lets us build a new
scheduled sidecar computation while keeping the original L0 computation alive.
Until a final publish step rewires the function result, the payload IR remains
valid and semantically unchanged. This means many pieces that had to be one TVM
primitive can become separate MLIR transform ops.

### TVM Rolling Update Components

The current TVM implementation in
`/Users/evanzhao/workspace/nautilus/src/tir/schedule/primitive/reduction.cc`
bundles these phases:

1. **Frontier discovery**
   `FrontierBuildHelper` walks producers of the target block and returns:
    - incomplete reduce producers under the target loop,
    - post-loop/frontier reductions for split-K mode,
    - topologically ordered spatial blocks between the target and frontier.

2. **Unsafe fusion**
   `RollingUpdate` disables schedule checks and reverse-compute-ats every block
   in the topological chain under the target loop.

3. **Reduce-to-scan conversion**
   `RollingUpdateBlockFrontier::ApplyAllReduceToScan` applies `ReduceToScan` to
   frontier reductions, expanding buffers with a scan dimension and rewriting
   uses.

4. **Mock inlining for analysis**
   `MockInlineBlocks` inlines the spatial chain in a throwaway schedule state so
   the target block contains the full expression to analyze.

5. **Algebraic repair derivation**
   `ReduceRepairer::DeriveFromBlockExpr` abstracts incomplete buffer loads as
   variables, solves for the global update function `H`, and records how to
   substitute previous/current values.

6. **RFactor**
   `RFactorGeneralized` splits the target reduction into an rfactor block and a
   writeback block.

7. **Writeback repair**
   `ReduceRepairer::RewriteBlockExpr` rewrites the writeback RHS so the factored
   partial result is globally correct.

The risky phases are unsafe fusion, reduce-to-scan, and rfactor: they can change
what downstream blocks observe before the repair has been installed.

### MLIR Decomposition Principle

Use a two-track transformation:

- The original algorithmic tensor chain remains live.
- A new scheduled tensor chain is built side-by-side inside `scf.forall` /
  `affine.for`.
- Only the final `publish` step replaces the original result with the scheduled
  result.

This lets intermediate transform ops be small and verifier-valid. Some
intermediate sidecar values may be "not yet used by the final result", but they
are not semantically wrong because they do not replace the original result yet.

### Proposed Smaller Transform Ops

#### `transform.match.linalg_ext.rolling_frontier`

Pure matcher/analysis op.

Input:

- target reduction-like op,
- target streaming loop,
- optional mode: `scan` or `split_k`.

Output:

- frontier reduction handles,
- spatial producer-chain handles in topological order,
- metadata describing incomplete values and reduction dimensions.

This should not mutate payload IR. It replaces TVM's frontier discovery logic
and should fail with a silenceable failure if the graph is not a supported
rolling-update pattern.

#### `transform.linalg_ext.clone_chain_under_loop`

Clone or tile the matched spatial producer chain under the target loop.

Unlike TVM's unsafe reverse-compute-at, this should avoid destructively moving
the only producer. The original producer chain stays live until publication.

For attention, this creates loop-local score tiles and elementwise score
transforms inside the streaming loop.

#### `transform.linalg_ext.reduction_to_scan`

Convert one frontier reduction into an explicit scan or loop-carried tile state.

This should be atomic per reduction. For FlashAttention, useful instances are:

- row max recurrence,
- row sum recurrence.

In L1, the preferred representation is often not an expanded scan tensor, but an
`affine.for iter_args` value:

```text
m_next = max(m_prev, row_max(score_tile))
l_next = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)
```

The transform should be allowed to choose the `iter_args` representation for L1,
even if the TVM implementation used scan buffers internally.

#### `transform.linalg_ext.derive_rolling_repair`

Pure analysis op.

Given:

- reducer kind,
- cloned producer chain,
- incomplete value metadata,
- previous/current value mapping,

derive the repair expression needed to combine partial results into the global
result.

For the FlashAttention pattern this should derive or directly recognize:

```text
scale = exp2(m_prev - m_next)
l_next = scale * l_prev + row_sum
acc_next = broadcast(scale) * acc_prev + pv
```

This op should not mutate IR. It can return a parameter/attribute describing the
repair plan.

#### `transform.linalg_ext.rfactor_reduction`

Factor the target reduction over the streaming loop.

In MLIR, this should prefer constructing a sidecar partial computation rather
than immediately replacing original uses. For attention, this is the transition
from full `P @ V` to per-KV-block `p_tile @ v_tile`.

This op can be smaller than TVM's generalized rfactor because L1 can represent
the factored value directly as a tile tensor in the loop body.

#### `transform.linalg_ext.apply_rolling_repair`

Apply the repair expression to the factored/sidecar computation.

For attention, this creates the loop-carried accumulator:

```text
acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile
```

This is the point where the sidecar computation becomes mathematically complete,
but it still does not need to replace the original result.

#### `transform.linalg_ext.publish_scheduled_result`

Replace the original algorithmic result with the completed scheduled result.

This is the only step that changes the externally visible function result. After
this step, canonicalization and DCE can remove the old L0 chain.

### FlashAttention-Specific Pipeline

For the current attention example, the decomposed pipeline should be:

1. Match the attention chain structurally.
2. Create the outer `scf.forall` over `(B, H, M-block)`.
3. Extract `q_tile` once per output tile.
4. Create the inner `affine.for` over K/V blocks.
5. Clone/tile QK and score scaling inside the streaming loop.
6. Add row-max recurrence:

    ```text
    row_max = reduce_max(score_tile)
    m_next = max(m_prev, row_max)
    ```

7. Add probability tile:

    ```text
    p_tile = exp2(score_tile - broadcast(m_next))
    ```

8. Add row-sum recurrence:

    ```text
    row_sum = reduce_sum(p_tile)
    l_next = exp2(m_prev - m_next) * l_prev + row_sum
    ```

9. Add PV partial:

    ```text
    pv = p_tile @ v_tile
    ```

10. Add accumulator recurrence:

    ```text
    acc_next = broadcast(exp2(m_prev - m_next)) * acc_prev + pv
    ```

11. Normalize after the loop:

    ```text
    out_tile = acc_final / broadcast(l_final)
    ```

12. Insert the output tile into the `scf.forall` shared output.
13. Publish the scheduled result and DCE the original chain.

### Atomicity Boundaries in MLIR

The following operations can be independently atomic because they do not expose
incorrect values to the original users:

- matching frontier/attention structure,
- cloning/tile extraction for sidecar computation,
- deriving repair plans,
- adding row-max recurrence,
- adding row-sum recurrence,
- adding PV partial,
- adding accumulator recurrence.

The following should remain atomic internally:

- any op that replaces live uses with a value that is only correct after repair,
- any op that mutates an existing producer chain instead of cloning it,
- final publish plus required result type/shape checks.

The preferred design is to avoid the first two cases. If a transform must mutate
the existing chain in place, it should bundle the mutation and repair into a
single transform op.

### Autoscheduler Implications

The autoscheduler should schedule smaller, semantically meaningful steps instead
of invoking one monolithic `rolling_update`.

Useful knobs become:

- choose M tile size for `scf.forall`,
- choose KV block size for streaming loop,
- choose whether row-max and row-sum use scan tensors or loop-carried tile
  values,
- choose accumulator element type and truncation points,
- choose where to materialize or fuse probability tiles,
- choose whether to emit exp or exp2 form.

The scheduler can still expose a composite convenience transform:

```mlir
transform.linalg_ext.flash_attention_schedule %attention
  {m_tile = 128, kv_tile = 64}
```

But internally this should expand to the smaller transform ops above. That keeps
the autotuning search space easier to reason about while preserving a simple
high-level entry point.
