# Attention Upward Fusion Design

## Goal

This note isolates the upward-fusion part of the L0-to-L1 attention schedule.
It has two purposes:

1. Record the intended semantics of the existing
   `transform.linalg_ext.fuse_map_consumer_into_loop` primitive.
2. Define the next reduction-fusion primitive needed to fuse row-wise
   reductions into the tiled attention loop nest without violating MLIR's
   parallel semantics.

The main transform schedule lives in
`test/python/data/attention_l0_to_l1.transform.mlir`. This document is the
design reference for the custom fusion ops used there.

## Problem Shape

For the QK / softmax portion of attention, the relevant producer-consumer chain
is:

```text
qk = Q @ K^T
score = scale(qk)
row_max = reduce_max(score, dim = j)
```

After tiling, MLIR naturally gives producer tiles inside a loop and consumer
ops outside the loop:

```text
loop -> full tensor result -> downstream consumer
```

Upstream `transform.structured.fuse_into_containing_op` is producer-to-consumer
fusion. It is not the right primitive for this direction. We need a narrow
consumer-to-loop reverse-fusion operation.

## Existing Primitive: `fuse_map_consumer_into_loop`

### Intent

`transform.linalg_ext.fuse_map_consumer_into_loop` is the minimal valid
`reverse_compute_at`-style primitive for unary elementwise maps.

It is deliberately narrow:

- target consumer is a single `linalg.map`,
- the map has one tensor input and one tensor output,
- the map input is the result of the containing tiled loop,
- the loop materializes that result through:
  - `tensor.insert_slice` for `scf.for`, or
  - `tensor.parallel_insert_slice` for `scf.forall`.

### Effect

Given:

```text
producer_tile = ...
publish producer_tile into loop result
full_tensor = loop(...)
mapped = map(full_tensor)
```

rewrite to:

```text
producer_tile = ...
mapped_tile = map(producer_tile)
publish mapped_tile into loop result
full_tensor = loop(...)
```

and replace uses of the original full-tensor map result with the loop result.

### Why It Is Sound

This works because unary elementwise maps are pointwise with respect to the
published tile. No cross-thread or cross-iteration combine is needed. The loop
still publishes disjoint tiles of the same full tensor shape.

### Relation to the Main Plan

This is the focused version of the original `reverse_compute_at` entry in
`docs/attention-l0-to-l1-transform-plan.md`. The larger plan described the op
as:

- a custom upward-fusion primitive,
- intentionally narrower than general TVM `reverse_compute_at`,
- sufficient for score scaling in FlashAttention.

That design remains correct. This document just isolates it and makes the loop
semantics explicit for both `scf.for` and `scf.forall`.

## Why Row-Max Is Different

Fusing row-max upward is not just "the same thing for reductions".

For a row-wise max:

```text
row_max[i] = max_j score[i, j]
```

if the `j` dimension is distributed across `scf.forall` threads, then each
thread only sees a tile-local partial reduction:

```text
partial_row_max[i, j_tile] = max_{j in tile} score[i, j]
```

Publishing that partial result directly into the final `row_max` tensor is not
legal in `scf.forall` unless each output element is written exactly once.
Otherwise the `tensor.parallel_insert_slice` writes overlap, and MLIR gives
undefined semantics for those conflicting writes.

For FlashAttention L1, the simpler and more TVM-like answer is not to keep the
reduced tile dimension parallel. Instead, the upward-fusion transform should
turn that one tiled dimension into a sequential inner `scf.for` and compute the
reduction completely inside it.

## New Primitive: Reduction Fusion By Sequentializing The Reduced Tile Loop

### Intent

Add a new custom transform op that fuses a reduction consumer into an existing
tiled producer loop by rebuilding the loop nest:

1. remove the reduced tile dimension from the outer `scf.forall`,
2. introduce a new inner sequential `scf.for` over that tile dimension,
3. fuse the producer chain and the reduction into that `scf.for`,
4. publish the final reduced tile once per outer `scf.forall` instance.

This is conceptually "reverse_compute_at + sequentialize reduced tile loop" as
one atomic transformation.

### Proposed Name

The exact name can change, but the semantics should be explicit in the name.
Examples:

- `transform.linalg_ext.fuse_reduction_consumer_into_forall`
- `transform.linalg_ext.fuse_reduction_consumer_by_sequentializing_forall_dim`

The key semantic point is that the pass starts from an `scf.forall` producer
loop, but the reduced tiled dimension stops being part of the `forall`.

### Expected Inputs

- a handle to the reduction consumer,
- a handle to the containing `scf.forall` loop,
- optional attributes describing:
  - which tiled producer dimension corresponds to the consumer reduction,
  - the sequential step for the new inner `scf.for` if it cannot be derived
    from the existing tile.

### Expected Outputs

- the fused reduction op inside the new `scf.for`,
- the rebuilt `scf.forall`,
- the new inner `scf.for`.

Returning fresh loop handles is important. The transform rewrites the outer
loop, so descendant handles should be treated as invalid unless they are
returned explicitly.

## IR Shape

### Before

```text
%producer = scf.forall ... shared_outs(%out = %init) -> tensor<1x32x4096x4096xf32> {
  %tile = ... : tensor<1x1x128x128xf32>
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %tile into %out[0, %h, %i, %j]
  }
}

%row_max = linalg.generic
  ins(%producer : tensor<1x32x4096x4096xf32>)
  outs(%row_max_init : tensor<1x32x4096xf32>)
```

### After

```text
%producer_new, %row_max_new =
  scf.forall (%h, %i) in (32, 32)
      shared_outs(%scores = %scores_init, %rows = %row_max_init)
      -> (tensor<1x32x4096x4096xf32>, tensor<1x32x4096xf32>) {
    %i_offset = affine.apply (d0) -> (d0 * 128) (%i)
    %row_init = tensor.extract_slice %rows[0, %h, %i_offset]
        [1, 1, 128] [1, 1, 1]
        : tensor<1x32x4096xf32> to tensor<1x1x128xf32>

    %row_final = scf.for %j = %c0 to %c32 step %c1
        iter_args(%acc = %row_init) -> (tensor<1x1x128xf32>) {
      %j_offset = affine.apply (d0) -> (d0 * 128) (%j)
      %score_tile = ... : tensor<1x1x128x128xf32>
      %scaled_tile = ... : tensor<1x1x128x128xf32>
      %row_tile = linalg.generic
          ins(%scaled_tile : tensor<1x1x128x128xf32>)
          outs(%acc : tensor<1x1x128xf32>)
          -> tensor<1x1x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %scaled_tile into
          %scores[0, %h, %i_offset, %j_offset] [1, 1, 128, 128] [1, 1, 1, 1]
      }
      scf.yield %row_tile : tensor<1x1x128xf32>
    }

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %row_final into
        %rows[0, %h, %i_offset] [1, 1, 128] [1, 1, 1]
    }
  }
```

The critical property is that the row-wise reduction is complete before the
`tensor.parallel_insert_slice` into `%rows`. Each `(h, i)` `forall` instance
writes exactly one disjoint row-max tile `tensor<1x1x128xf32>`.

## Feasibility And Upstream Reuse

This transform is doable, but it is primarily loop surgery. The fused reduction
itself is ordinary once the reduced tile dimension has become sequential.

### What We Can Reuse

- existing matching and cloning logic from `fuse_map_consumer_into_loop`,
- reduction-shape verification and combiner-body cloning,
- the earlier loop-rebuild code for adding results / region arguments to SCF
  loops.

### What Existing Upstream `rfactor` Does Not Solve

Upstream `split_reduction` and `tile_reduction_using_forall` operate on a
standalone reduction op. They are useful for a design that keeps the reduced
tile dimension parallel and introduces partial-reduction tensors.

That is not this design. After this transform, the reduction lives as a
loop-carried update inside an inner `scf.for`. There is no separate
partial-reduction tensor and no final combine after the outer `forall`.
So upstream `rfactor` is not the main primitive here.

### New Code We Still Need

- rebuild the `scf.forall` with one fewer induction variable,
- create the inner sequential `scf.for` over the removed tile dimension,
- thread the fused reduction tile through that `scf.for` as an iter_arg,
- keep score-tile publication inside the loop nest,
- publish the final reduced tile once at the end of the `forall` body,
- return fresh handles for the rebuilt `forall` and inner `for`.

## Relationship Between the Two Fusion Ops

These two custom ops should be understood as one family:

- `fuse_map_consumer_into_loop`
  - pointwise upward fusion,
  - no extra state,
  - valid for both `scf.for` and `scf.forall`.

- reduction fusion by sequentializing one tiled `scf.forall` dimension
  - upward fusion plus loop restructuring,
  - loop-carried reduction state in an inner `scf.for`,
  - no partial-result tensor and no post-loop combine,
  - valid because the final reduced tile is published once per outer parallel
    instance.

The attention schedule can then stay conceptually simple:

1. Tile QK.
2. Fuse score scaling upward with the map-fusion op.
3. Fuse row-max upward with the reduction-fusion op, which turns the `j` tile
   loop from parallel to sequential.
4. Continue with exp / row-sum / normalization using similarly explicit
   schedule primitives.

## Schedule Guidance

The transform script should keep only the high-level scheduling intent inline.
Detailed semantics of the custom fusion ops should live here, not in the
schedule file.

That keeps the transform readable while preserving one stable design reference
for implementation and testing.
