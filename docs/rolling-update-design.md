# Attention Rolling Update Design

This note describes the rolling-update operator fusion transformation in its current MLIR form.
Integrated attention examples using it live under `test/Pipeline`, starting with
`test/Pipeline/tm_global_attention.mlir`.

Rolling update is a generalized form of operator fusion that can put pairs of reductions together
under the same loop. Since that breaks the usual producer-consumer dependencies,
rolling update performs extra _repair_ steps to restore correctness.

## Pipeline

The current MLIR pipeline has three steps:

1. `transform.fusion.find_next_reduction`
   Starting from a loop-local producer, run a forward BFS on the def-use graph
   and return the nearest reduction plus the ordered elementwise chain between
   the producer and that reduction.

2. `transform.fusion.clone_fuse_elemwise`
   Clone that elementwise chain into a sidecar chain, fuse it under the
   streaming loop, and publish the sidecar tensors as extra loop results. The
   original out-of-loop chain is left untouched.

3. `transform.fusion.repair_reduction_frontier`
   Repair the chosen reduction frontier by deriving a repair term `H`,
   materializing it as a pointwise tensor update, and rebuilding the frontier
   reduction to use that repaired init tensor.

This keeps the payload IR verifier-valid throughout the schedule while still
matching the shape of TVM's rolling update.

## Transform Details

### `transform.fusion.find_next_reduction`

Signature:

```text
(producer_op) -> (next_reduce_op, elemwise_ops)
```

This is a pure analysis op using `NavigationTransformOpTrait` (so the input handle is not
consumed). It returns:

- `next_reduce_op`: the first reduction-like op reachable from `producer_op` by BFS,
- `elemwise_ops`: the elementwise ops strictly between the producer and that reduction,
  sorted in producer-to-consumer order.

The op fails if no reduction is reachable or if any op on the path is not a
supported single-result elementwise `linalg.generic`.
Tensor `arith` / `math` elementwise ops must be normalized first, e.g. with
`convert-elementwise-to-linalg`.

### `transform.fusion.clone_fuse_elemwise`

Signature:

```text
(elemwise_ops, parallel_loop, streaming_loop) -> (sidecar_ops)
```

This transform clones each elementwise op, fuses the clone under the loop nest,
and returns the cloned sidecar ops in the same order. It rebuilds both loops:

- the inner `scf.for` gets one extra iter_arg/result per sidecar tensor,
- the outer `scf.forall` gets matching extra shared_out/result slots.

The loop handles are preserved and remapped to the rebuilt loops. The original elementwise
chain remains unchanged until a later repair step rewires a specific reduction use.

### `transform.fusion.repair_reduction_frontier`

Signature:

```text
(producer_reduce_ops, reduce_op, elemwise_orig_ops, elemwise_sidecar_ops, outer_loop, inner_loop)
  -> (repaired_reduce_op)
```

This transform implements the _repair_ on the reduction frontier (`reduce_op`).
It performs the following steps:

1. Fuse `reduce_op` into `inner_loop` (via `outer_loop`) so that it stays next
   to the fused `sidecar_chain_ops`.
1. Inline the sidecars into `reduce_op` with repeated `linalg::fuseElementwiseOps`,
   until the frontier becomes a single `linalg.generic`.
1. Inspect that fused body and classify scalar inputs:
   - `r0`, `r1`, ... for values produced by earlier reductions `producer_reduce_ops`,
   - `c0`, `c1`, ... for ordinary values,
1. Extract:
   - `f_expr` from the `reduce_op`, which should be a simple reduction combiner, such as `a + b`
   - `g_expr` from the non-accumulator side of the reduction combiner,
     which should coincide with the combined computation of `elemwise_sidecar_ops`.
1. Call the Python/SymPy solver on these two expressions to derive a repair term `H`,
   and prove that `H` is valid for the combiner `f`.
1. Materialize `H` as an elementwise `linalg.generic` (call it `update_op`),
   which updates the DPS init argument of `reduce_op`;
   then clone `reduce_op` with its DPS init remapped to the output of `update_op`.

The result is a repaired reduction whose accumulator is no longer the original
partial state, but the repaired state computed from `H`.

## Current Assumptions

The current implementation assumes:

- a single-result, single-init, one-reduction-dim `linalg.generic` frontier,
- the frontier can be fused under the inner loop,
- the fused frontier still matches a self-reduction `out = f(out, g(...))`,
- producer reductions are destination-style ops whose previous value is their init operand,
- the solved repair term eliminates all `c*` variables before re-materialization.

## Attention Example

For the current attention schedule, the intended flow is:

1. build the outer `scf.forall` over output tiles and the inner streaming loop over K/V blocks,
2. fuse QK and score scaling under that streaming loop,
3. find the nearest reduction frontier from the loop-local score tile,
4. clone and fuse the elementwise chain under the loop as sidecar ops,
5. repair the frontier reduction by deriving and applying `H`,
6. repeat for later frontiers such as `P @ V` accumulation and row sum.

For row sum and output accumulation this yields the usual FlashAttention
recurrences:

```text
m_next   = max(m_prev, row_max(score_tile))
p_tile   = exp2(score_tile - m_next)
l_next   = exp2(m_prev - m_next) * l_prev + row_sum(p_tile)
acc_next = exp2(m_prev - m_next) * acc_prev + p_tile @ v_tile
out      = acc_final / l_final
```

## SplitKUpdate Status

SplitKUpdate is an in-progress extension for decode-input attention, where the query
length is one and the K/V dimension should remain parallel across splits.
Unlike rolling update, SplitKUpdate should not create a scan dependency over K/V blocks.
Instead, each split computes partial reduction results, and a write-back reduction merges
those partials after the split loop.

At the IR level this should be treated as a split-k rfactor/write-back plan:

```text
parallel for k_split:
  m_rf[k_split]   = row_max(score_tile)
  p_tile          = exp2(score_tile - m_rf[k_split])
  l_rf[k_split]   = row_sum(p_tile)
  acc_rf[k_split] = p_tile @ v_tile

m   = reduce_max(m_rf over k_split)
l   = reduce_sum(exp2(m_rf - m) * l_rf over k_split)
acc = reduce_sum(exp2(m_rf - m) * acc_rf over k_split)
out = acc / l
```

### `transform.scf.fuse_partial_reduction_into_forall`

This is the partial-reduction counterpart to `transform.scf.fuse_reduction_into_forall`:

```text
transform.scf.fuse_reduction_into_forall
  -> scan/rolling form: the existing forall dimension becomes an inner scf.for

transform.scf.fuse_partial_reduction_into_forall
  -> split/partial form: the existing forall dimension stays parallel, and a
     final merge reduction is emitted after the forall
```

Signature:

```text
(reduce_op, forall_loop) -> (rf_reduce_op, wb_reduce_op)
```

Internally, it should reuse MLIR's partial-reduction interfaces instead of
implementing the raw rfactor/write-back primitive from scratch. In particular,
`PartialReductionOpInterface` already provides the identity-tensor creation,
partial-reduction tiling, and merge-reduction construction used by upstream
`transform.structured.tile_reduction_using_forall`. The wrapper differs from
that upstream transform because it must fuse into an existing attention
`scf.forall` rather than creating a new one.

Use this to create the first rfactor/write-back pair, starting with the row-max
frontier. The related builtin transforms are less directly useful here:
`tile_reduction_using_for` creates a serial `scf.for`, and `split_reduction`
does not by itself create the parallel split-k loop shape.

The transform takes a reduction frontier whose input is produced by an existing `scf.forall`.
It keeps the split dimension parallel, creates a split-local rfactor reduction under that forall,
and emits the final write-back reduction after the forall.
This is enough to form the first SplitK pair, starting with the row-max frontier.

It reuses MLIR's `PartialReductionOpInterface` for identity-tensor creation and
merge-reduction construction. This wrapper exists because the upstream
partial-reduction transforms do not fuse into an existing attention
`scf.forall`: `tile_reduction_using_for` creates a serial `scf.for`, and
`split_reduction` does not create the parallel split-k loop shape by itself.

The transform performs the following steps:

1. Identify the producer `tensor.parallel_insert_slice` and the reduced tensor
   dimension controlled by one forall induction variable.
1. Create an identity-filled rfactor tensor with result shape plus one split dimension.
1. Rebuild the forall with the rfactor tensor as an extra `shared_out`, clone the original body,
   and compute the split-local reduction on the loop-local producer tile.
1. Insert the split-local result into the rfactor tensor, then call
   `PartialReductionOpInterface::mergeReductions` to create the write-back reduction.

The `forall_loop` handle is preserved and remapped to the rebuilt loop.
The `reduce_op` handle is consumed. The returned handles are the split-local
rfactor reduction and the final write-back reduction.

For decode attention row max, the resulting shape is (note `%m_rf` vs. `%m`):

```text
%score, %m_rf = scf.forall (...) shared_outs(...) {
  %score_tile = ...
  %m_part = linalg.generic ... row max over local K tile ...

  tensor.parallel_insert_slice %score_tile into %score[..., k_tile]
  tensor.parallel_insert_slice %m_part into %m_rf[..., k_split]
}

%m = linalg.reduce ins(%m_rf) ... dimensions = [k_split_dim]
```

### Planned: SplitK Sidecar Fusion

Clone and fuse the elementwise chain under the split-k loop, like
`clone_fuse_elemwise`, but substitute reads of prior write-back results with
matching rfactor reads. For example, a split-local sidecar that originally reads
the final row max `m` should read `m_rf[..., k_split]` instead.

### Planned: SplitK Repair

Reuse the rolling-update solver to derive `H`, but apply the result to rfactor
partials before the write-back reduction. For row sum and output accumulation,
this materializes the familiar repair factors:

```text
l_repaired   = exp2(m_rf - m) * l_rf
acc_repaired = exp2(m_rf - m) * acc_rf
```

Those repaired tensors then feed the final split-dimension merge reductions.

### Planned SplitK Output Shape

The target tensor-level MLIR shape before lower-level lowering is:

```text
%m_rf, %l_rf, %acc_rf = scf.forall (...) shared_outs(...) {
  %score_tile = ...
  %m_part = linalg.generic ... row max over local K tile ...
  %p_tile = linalg.generic ... exp2(score_tile - m_part) ...
  %l_part = linalg.generic ... row sum over local K tile ...
  %acc_part = linalg.matmul ins(%p_tile, %v_tile) ...

  tensor.parallel_insert_slice %m_part into %m_rf[..., k_split]
  tensor.parallel_insert_slice %l_part into %l_rf[..., k_split]
  tensor.parallel_insert_slice %acc_part into %acc_rf[..., k_split]
}

%m = linalg.reduce ins(%m_rf) ... dimensions = [k_split_dim]

%l_repaired = linalg.generic ins(%l_rf, %m_rf, %m) {
  exp2(m_rf - m) * l_rf
}
%l = linalg.reduce ins(%l_repaired) ... dimensions = [k_split_dim]

%acc_repaired = linalg.generic ins(%acc_rf, %m_rf, %m) {
  exp2(m_rf - m) * acc_rf
}
%acc = linalg.reduce ins(%acc_repaired) ... dimensions = [k_split_dim]

%out = linalg.generic ins(%acc, %l) {
  acc / l
}
```

For decode attention, canonicalization and unit-dimension folding should still
be able to remove the query-length-one dimension after the split-k structure is
formed.

## Transform-Dialect Notes

- `transform.fusion.find_next_reduction` is analysis-only and does not consume its input handle.
- `clone_fuse_elemwise` and `repair_reduction_frontier` rewrite explicit ops in functional style,
  while loop handles are read-only inputs remapped to rebuilt loops.
- SplitKUpdate should use explicit rfactor/write-back handle pairs for values
  produced by partial-reduction tiling. Later SplitK steps need both sides of
  the pair to rewrite reads correctly.
- Ordered multi-op handles are part of the contract: the elementwise chain is
  not treated as an unordered set.
- The fusion helpers depend on loop results being traceable through
  `tensor.insert_slice` / `tensor.parallel_insert_slice` relays. The integrated
  schedules run canonicalization at points where those relays are still
  recoverable by later fusion steps.

## Testing

Useful coverage is:

1. analysis-only tests for `transform.fusion.find_next_reduction`,
2. synthetic tests for cloned and fused elementwise sidecar chains,
3. reduction-frontier repair tests for supported single-result reduction cases,
4. end-to-end attention tests that check the repaired loop-carried recurrence shape.

For the planned SplitKUpdate work, useful additional coverage is:

1. direct tests that `transform.scf.fuse_partial_reduction_into_forall`
   produces the expected partial-reduction and merge-reduction handles for
   attention-like max and sum reductions,
2. tests for write-back-to-rfactor read substitution in split-k sidecar chains,
3. repair tests where `H` is materialized on rfactor partials before the
   write-back reduction,
4. end-to-end decode attention tests that check for split-k partial tensors and
   no scan-style `scf.for` dependency over K/V splits.
