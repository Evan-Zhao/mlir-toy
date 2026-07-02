# Dead Tile Specialization

## Goal

`transform.loop.specialize_dead_tile` specializes a scheduled streaming
`scf.for` when a loop-local tile producer becomes trivial on contiguous regions
of the loop IV space.

The motivating case is causal attention: after tiling, a mask producer may
return a full `-inf` tile for K/V blocks strictly to the right of the current
query tile, and it may return the unmasked score tile for an initial fully-live
prefix. The transform exploits those two regions to simplify the loop.

## Transform

It takes as input:

- a schedule-provided producer op,
- a surrounding `scf.for`,
- and a schedule-provided dead value.

```mlir
%live_loop, %mixed_loop = transform.loop.specialize_dead_tile %producer in %loop
    {dead_value = 0xFF800000 : f32} : (!any, !any) -> !any, !any
```

Contract:

- `%producer` is a loop-local tensor producer whose result may become a full
  dead-value tile.
- `%loop` is the streaming `scf.for` whose IV appears in the producer
  predicate.
- `dead_value` is an explicit hint. The pass does not try to rediscover it.
- The transform may fail if it cannot prove the rewrite preserves loop state.
- When the producer handle is omitted or empty, the transform infers a unique
  matching producer. If no producer matches, it succeeds as a no-op and returns
  empty handles. Multiple inferred matches remain an ambiguity and fail.

The op returns handles to the new fully-live prefix loop and mixed loop. The
producer handle remains valid when tracking succeeds.

## Current Implementation

The pass is implemented in four pieces.

### 1. Classify the Producer

The pass matches a producer of the form:

```mlir
%selected = arith.select %pred, %live, %dead : f32
linalg.yield %selected : f32
```

with `%dead` equal to the transform attribute.

It then rebuilds `%pred` as affine expressions over:

- loop IVs,
- `linalg.index`,
- and simple `affine.apply` compositions.

Using affine/Presburger reasoning, it classifies the producer tile as:

- fully live: `%pred` is true everywhere in the tile,
- fully dead: `%pred` is false everywhere,
- mixed: otherwise.

For tiled causal attention:

```text
q_abs = q_block * BQ + row
k_abs = j * BK + col
live iff k_abs <= q_abs
```

This yields two useful boundaries:

```text
fully-live prefix: j * BK + (BK - 1) <= q_block * BQ
fully-dead suffix: j * BK > q_block * BQ + (BQ - 1)
```

The implementation handles general affine expressions, not just hand-written
linear forms.

### 2. Propagate Dead-Tile Facts

When the producer is fully dead, the pass runs a small loop-local abstract
interpretation over downstream ops to answer one question:

> does this iteration yield the incoming loop-carried state unchanged?

The analysis tracks values such as:

- constants,
- equivalence to another tensor or loop-carried value,
- and a few operation-specific identities.

The useful attention-style rules are things like:

- `maximumf(dead, x) -> x`,
- `exp(-inf) -> 0`,
- reduction with zero input returns the init/accumulator,
- DPS elementwise ops may forward an input or init tensor unchanged.

This is intentionally narrow. The pass does not rely on generic tensor-level
constant propagation through arbitrary `linalg.generic` bodies.

### 3. Derive Loop Regions

The fully-live and fully-dead conditions are projected onto the streaming loop
IV and converted into half-open affine intervals.

For the common causal-attention case with `BQ = 128`, `BK = 64`:

```text
live_prefix_ub = min(num_j_tiles, 2 * q_block + 1)
dead_lb        = min(num_j_tiles, 2 * q_block + 2)
```

Those boundaries are then materialized as SSA values for rewriting.

### 4. Rewrite the Loop

The current rewrite specializes the fully-live prefix and keeps the remaining
possibly-live region as a mixed loop:

```mlir
%state_live = scf.for %j = %lb to %live_prefix_ub step %step ...
%state_mixed = scf.for %j = %live_prefix_ub to %mixed_ub step %step ...
```

In the live prefix, the matched mask producer is removed and its result is
rewired directly to the live input tile when that replacement is pointwise
aligned with the producer output.

The mixed loop keeps the original masked body. When dead-tile propagation proves
that fully-dead iterations preserve every loop-carried value, `%mixed_ub` is
tightened to the first fully-dead iteration; otherwise it remains the original
loop upper bound.

## Constant Propagation Note

This transformation drops iterations from a loop by reasoning
if the loop would return the same value as its carried inputs in this iteration
(basically a no-op).

Stock MLIR constant propagation and folding is not enough for this transformation.
Current MLIR can fold `arith` and `math` operations over dense constants (tensors),
but it does not apply to `linalg.generic` operations.
Furthermore, it does not track if the output of an operation would be _identical_
to another value, which is crucial for this transformation.

This transformation builds its own abstract interpretation-based analysis
over Linalg and SCF to track value propagation.
The implementation is limited to a small set of patterns that are common in attention masking,
but it could be extended in the future if needed.

## Tests

Current tests cover:

- small producer-only dead-tile classification,
- windowed and causal boundary derivation,
- dead-tile-driven live/mixed loop splitting,
- causal-attention end-to-end scheduling,
- and handle preservation across loop-rewriting transforms.
