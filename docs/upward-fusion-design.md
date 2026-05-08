# Attention Upward Fusion Design

## Goal

This note documents the custom upward-fusion primitives used by the attention
L0-to-L1 schedule in
`test/python/data/attention_l0_to_l1.transform.mlir`.

The intent is the same as TVM `reverse_compute_at`: move a consumer under the
loop nest that already materializes tiles of its input. In MLIR, this is not a
single general upstream primitive, so Neptune uses two narrow transform ops:

- `transform.loop.fuse_into_producer_op`
- `transform.loop.fuse_reduction_consumer_into_forall`

Both are specialized for fusing a Linalg consumer operation into an SCF loop nest.

## `fuse_elemwise_into_producer`

`transform.loop.fuse_into_producer_op` applies pointwise,
"upward" (consumer into producer) fusion. It takes:

- an elementwise consumer, currently either:
    - a single-result `linalg.map`, or
    - a single-result, single-init `linalg.generic` with all-parallel loops and
      projected-permutation indexing maps,
- a containing tiled loop nest rooted at `scf.for` or `scf.forall`.

The implementation is intentionally thin. It delegates most of the real work to
upstream `scf::tileAndFuseConsumer` after checking that the consumer is
elementwise and that the supplied loop is the root of a supported tiled loop nest.

For attention, this is used to fused the score-scaling step into the main loop nest:

```text
qk = Q @ K^T
score = scale(qk)
```

After tiling `qk`, the scale map is fused directly onto each `qk` tile before
that tile is written back.

## `fuse_reduction_consumer_into_forall`

`transform.loop.fuse_reduction_consumer_into_forall` handles fusing a
reduction consumer into a loop nest.

It takes:

- a unary single-reduction `linalg.generic` consumer,
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

## Attention Example

For the attention softmax prefix:

```text
qk = Q @ K^T
score = scale(qk)
row_max = reduce_max(score, dim = j)
```

the schedule uses the two ops in order:

1. tile the `qk` producer,
2. fuse `scale` upward with `fuse_elemwise_into_producer`,
3. fuse `row_max` upward with `fuse_reduction_consumer_into_forall`.

After the reduction fusion, the relevant loop shape is:

```text
scf.forall over outer non-reduction tiles
  scf.for over the tiled reduction dimension
    compute qk tile
    scale tile
    update row-max tile
  publish full score panel
  publish completed row-max tile
```

In the current attention configuration, that means:

- outer `scf.forall` over head and row tiles,
- inner `scf.for` over column tiles,
- loop-carried score-panel and row-max tile state inside the inner `scf.for`,
- one final `tensor.parallel_insert_slice` of the completed row-max tile per
  outer parallel instance.
