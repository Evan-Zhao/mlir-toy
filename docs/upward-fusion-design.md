# Attention Upward Fusion Design

## Goal

This note documents the custom upward-fusion primitives used by the attention
L0-to-L1 schedules under `test/Pipeline`, starting with
`test/Pipeline/tm_global_attention.mlir`.

The intent is the same as TVM `reverse_compute_at`: move a consumer under the
loop nest that already materializes tiles of its input. Neptune uses two custom
transform ops for the pointwise and reduction cases:

- `transform.fusion.greedy_consumers_into_producer`
- `transform.scf.fuse_reduction_into_forall`

Both fuse Linalg consumer operations into an SCF loop nest.

## `fusion.greedy_consumers_into_producer`

`transform.fusion.greedy_consumers_into_producer` repeatedly fuses direct consumers into an
`scf.forall`. For attention, it first fuses the elementwise score prefix and stops before the row
reduction:

```mlir
%prefix = transform.fusion.greedy_consumers_into_producer %forall_loop until %row_max
    { inline_elementwise } : (!transform.any_op, !transform.any_op) -> !transform.any_op
```

The operation delegates pointwise fusion to upstream `scf::tileAndFuseConsumerOfSlice`. The
`inline_elementwise` option first folds eligible elementwise producers into each selected consumer.

## `scf.fuse_reduction_into_forall`

`transform.scf.fuse_reduction_into_forall` handles fusing a
reduction consumer into a loop nest.

It takes:

- a single-result, single-dimension `linalg.generic` reduction consumer
  whose reduced input is produced by the target `scf.forall`,
- a containing `scf.forall` whose result is the reduction input.

More precisely, the current implementation requires:

- the reduction input to be produced by the target `scf.forall`,
- that `scf.forall` to have exactly one result,
- that result to be published through exactly one
  `tensor.parallel_insert_slice`,
- the reduced dimension of the input tensor to correspond to one dynamic
  `scf.forall` induction variable.

It performs a TVM-like `reverse_compute_at`, but on reduction,
with one extra structural change required by MLIR's parallel semantics:

1. rebuild the outer `scf.forall` without the induction variable that controls
   the reduced tiled dimension,
2. add the reduction result tensor as a new `shared_out` / result of that new
   `scf.forall`,
3. materialize an outer-thread producer panel slice and an outer-thread
   reduction tile slice,
4. create an inner sequential `scf.for` over the removed tiled dimension,
5. clone the original producer tile computation into that `scf.for`,
6. thread both the producer panel and the reduction tile as `iter_args` of the
   inner `scf.for`,
7. update the producer panel with `tensor.insert_slice`,
8. run the fused reduction update on the just-computed producer tile,
9. publish the completed producer panel and completed reduced tile once per
   outer `scf.forall` instance.

The transform does not keep the reduced tiled dimension parallel. It
sequentializes that one dimension and computes the reduction completely before
publishing the reduced result.
This has the side effect of producing a local partial-reduction buffer plus a
final published tile, which is structurally close to an **r-factor** style
rewrite, even though the pass is phrased as upward consumer fusion.

This is the MLIR analogue of TVM `reverse_compute_at` for a reduction consumer
when the original outer schedule placed the reduced tiled dimension in a
parallel loop.

## Why The Reduction Case Needs Loop Restructuring

For a row-wise reduction such as:

```text
row_max[i] = max_j score[i, j]
```

publishing one partial result per `j` tile directly from `scf.forall` would
cause overlapping `tensor.parallel_insert_slice` writes into the same row-max
slice. That is not a legal reduction combine in `scf.forall`.

The implemented transform avoids that problem by converting the tiled reduction
dimension into an inner sequential `scf.for`. The row-max tile is then fully
computed inside that loop and inserted exactly once by each outer `forall`
instance.
