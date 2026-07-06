# Split-K Update Design

This note describes the SplitK update extension for decode-input attention, where the query length
is one and the K/V dimension should remain parallel across splits.

SplitK update is related to [rolling update](rolling-update-design.md): it reuses the same
reduction-frontier repair solver, but it does not create a scan dependency over K/V blocks.
Instead, each split computes partial reduction results, and a write-back reduction merges those
partials after the split loop.

At the IR level this is a split-k rfactor/write-back plan:

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

## Pipeline

The SplitK update pipeline has three split-k-specific pieces:

1. `transform.scf.fuse_partial_reduction_into_forall`
   Split the first reduction frontier into a split-local rfactor reduction under the existing
   `scf.forall`, plus a final write-back reduction after the forall.

1. `transform.fusion.clone_fuse_rfactor_elemwise`
   Clone the elementwise chain under the split-k forall and substitute write-back reduction reads
   with split-local rfactor tiles.

1. `transform.fusion.repair_rfactor_reduction_frontier`
   Repair later rfactor partials before their write-back reductions by deriving and materializing
   a repair term `H`.

## Transform Details

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

Internally, it reuses MLIR's partial-reduction interfaces instead of implementing the raw
rfactor/write-back primitive from scratch. In particular, `PartialReductionOpInterface` already
provides the identity-tensor creation, partial-reduction tiling, and merge-reduction construction
used by upstream `transform.structured.tile_reduction_using_forall`.

The wrapper differs from that upstream transform because it must fuse into an existing attention
`scf.forall` rather than creating a new one. The related builtin transforms are less directly
useful here: `tile_reduction_using_for` creates a serial `scf.for`, and `split_reduction` does not
create the parallel split-k loop shape by itself.

The transform takes a reduction frontier whose input is produced by an existing `scf.forall`. It
keeps the split dimension parallel, creates a split-local rfactor reduction under that forall, and
emits the final write-back reduction after the forall.

The transform performs the following steps:

1. Identify the producer `tensor.parallel_insert_slice` and the reduced tensor dimension controlled
   by one forall induction variable.
1. Create an identity-filled rfactor tensor with result shape plus one split dimension.
1. Rebuild the forall with the rfactor tensor as an extra `shared_out`, clone the original body, and
   compute the split-local reduction on the loop-local producer tile.
1. Insert the split-local result into the rfactor tensor, then call
   `PartialReductionOpInterface::mergeReductions` to create the write-back reduction.

The `forall_loop` handle is preserved and remapped to the rebuilt loop. The `reduce_op` handle is
consumed. The returned handles are the split-local rfactor reduction and the final write-back
reduction.

For decode attention row max, the resulting shape is:

```text
%score, %m_rf = scf.forall (...) shared_outs(...) {
  %score_tile = ...
  %m_part = linalg.generic ... row max over local K tile ...

  tensor.parallel_insert_slice %score_tile into %score[..., k_tile]
  tensor.parallel_insert_slice %m_part into %m_rf[..., k_split]
}

%m = linalg.reduce ins(%m_rf) ... dimensions = [k_split_dim]
```

### `transform.fusion.clone_fuse_rfactor_elemwise`

This is the SplitK sidecar-fusion counterpart to `transform.fusion.clone_fuse_elemwise`.

Signature:

```text
(elemwise_ops, forall_loop, writeback_reduce_ops, rfactor_reduce_ops) -> (sidecar_ops)
```

The transform clones the ordered elementwise chain under the split-k `scf.forall`, publishes each
sidecar tensor as an extra `shared_out` / loop result, and substitutes write-back reduction reads
with split-local rfactor tiles. For example, a sidecar that reads the final row max `%m` outside the
loop reads the local `%m_part` tile inside the loop.

The `writeback_reduce_ops` and `rfactor_reduce_ops` handles are paired by handle order. For each
pair, the transform follows the rfactor result through its `tensor.parallel_insert_slice`, drops
the split dimension from the recorded tile, and maps the write-back result to that in-loop tile.

The transform:

1. Rebuilds the `scf.forall` once with extra `shared_out` operands for all sidecar DPS init tensors,
   then clones the original body and `scf.forall.in_parallel` region.
1. Builds a relay map from program values to in-loop tiles, including write-back results mapped to
   paired rfactor tiles.
1. Tiles each elementwise op in producer-to-consumer order, patches generated inputs to known
   in-loop tiles, and patches DPS inits to slices of the new forall block args.
1. Publishes each sidecar tile with `tensor.parallel_insert_slice`, records it for later sidecars,
   and erases temporary operand slices produced by the tiling interface.

The `forall_loop` handle is preserved and remapped to the rebuilt loop. The original out-of-loop
elementwise chain is intentionally left unchanged; the returned sidecar ops are used by later
SplitK repair steps.

For decode attention after row-max rfactoring, the resulting shape is:

```text
%score, %m_rf, %p = scf.forall (...) shared_outs(...) {
  %score_tile = ...
  %m_part = linalg.generic ... row max over local K tile ...
  %p_tile = linalg.generic ins(%score_tile, %m_part) {
    exp2(score_tile - m_part)
  }

  tensor.parallel_insert_slice %score_tile into %score[..., k_tile]
  tensor.parallel_insert_slice %m_part into %m_rf[..., k_split]
  tensor.parallel_insert_slice %p_tile into %p[..., k_tile]
}

%m = linalg.reduce ins(%m_rf) ... dimensions = [k_split_dim]
%p_original = linalg.generic ins(%score, %m) {
  exp2(score - m)
}
```

The duplicated `%p_original` remains valid payload IR and preserves the original program value. The
sidecar `%p_tile` is the split-local value that later repair steps should use to build repaired
rfactor partials.

### `transform.fusion.repair_rfactor_reduction_frontier`

This is the SplitK repair counterpart to `transform.fusion.repair_reduction_frontier`. It reuses
the rolling-update expression solver to derive `H`, but materializes the repair over rfactor
partial tensors before the final write-back reduction.

Signature:

```text
(reduce_op, writeback_reduce_ops, rfactor_reduce_ops,
 elemwise_orig_ops, elemwise_sidecar_ops, forall_loop)
  -> (repaired_rfactor_reduce_op, repaired_writeback_reduce_op)
```

The producer write-back reductions and rfactor reductions are paired by handle order. Each pair
describes one reduction value that appears as a final write-back tensor outside the loop, but whose
split-local rfactor tensor must be used when building the repair term.

The transform:

1. Matches the original and sidecar elementwise chains and builds a substitution map from original
   elementwise results to the sidecar tensors published by `forall_loop`.
1. Clones `reduce_op` with that substitution, so the frontier consumes the fused sidecar chain.
1. Detects the split dimension and uses the same rfactor/write-back machinery as
   `transform.scf.fuse_partial_reduction_into_forall` to split the staged frontier under the
   existing `scf.forall`.
1. Remaps producer rfactor reductions and sidecar ops to their cloned counterparts in the rebuilt
   forall, preserving transform handles.
1. Calls the shared repair solver on the split-local rfactor frontier and sidecar chain to derive a
   scalar repair term `H`.
1. Builds `H` in `SplitKUpdate` mode using bindings of the form `r_i = rfactor_tensor_i`,
   `r_i' = writeback_tensor_i`, plus the current rfactor accumulator tensor.
1. Clones the new write-back reduction with its DPS init remapped to the repaired rfactor tensor,
   and replaces the staged frontier with that repaired write-back reduction.

For row sum and output accumulation, this materializes the familiar repair factors:

```text
l_repaired   = exp2(m_rf - m) * l_rf
acc_repaired = exp2(m_rf - m) * acc_rf
```

Those repaired tensors feed the final split-dimension write-back reductions. The returned handles
are the repaired split-local rfactor reduction and the repaired write-back reduction.

## SplitK Output Shape

After the row-max rfactor, sidecar fusion, and one or more
`repair_rfactor_reduction_frontier` applications, the target tensor-level MLIR shape before
lower-level lowering is:

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

For decode attention, canonicalization and unit-dimension folding remove many of the
query-length-one dimensions after the split-k structure is formed.

## Transform-Dialect Notes

- SplitKUpdate uses explicit rfactor/write-back handle pairs for values produced by
  partial-reduction tiling. Later SplitK steps need both sides of the pair to rewrite reads
  correctly.
- Ordered multi-op handles are part of the contract: the elementwise chain is not treated as an
  unordered set.
- The fusion helpers depend on loop results being traceable through `tensor.parallel_insert_slice`
  relays. The integrated schedules run canonicalization at points where those relays are still
  recoverable by later fusion steps.
- The repair step reuses the rolling-update solver and `FusionRepairTerm`, but builds the result in
  `SplitKUpdate` mode so indexing maps are patched for rfactor/write-back tensors.

## Testing

Useful SplitKUpdate coverage is:

1. Direct tests that `transform.scf.fuse_partial_reduction_into_forall` produces the expected
   partial-reduction and merge-reduction handles for attention-like max and sum reductions.
1. Tests for write-back-to-rfactor read substitution in split-k sidecar chains
   (`test/Loop/clone_fuse_rfactor_elemwise.mlir`).
1. Repair tests where `H` is materialized on rfactor partials before the write-back reduction
   (`transform.fusion.repair_rfactor_reduction_frontier`).
1. End-to-end decode attention tests that check for split-k partial tensors and no scan-style
   `scf.for` dependency over K/V splits.
