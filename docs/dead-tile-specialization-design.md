# Dead Tile Specialization Design

## Goal

This note sketches a transform-dialect optimization for scheduled tile programs
where a schedule writer can identify a tile-producing op and a "dead value".
The transform should prove when that op produces a full tensor of the dead
value, propagate the consequence through nearby loop-local computation, and
specialize the streaming loop so fully-dead iterations do no work.

The motivating case is causal attention after the L0-to-L1 attention schedule:
a loop-local mask op may produce a full tile of `-inf` for K/V blocks that are
strictly to the right of the current query tile. Those blocks should be removed
from the streaming loop rather than computed and masked elementwise.

The design is intentionally not attention-only. It relies on a schedule-provided
anchor op plus a dead-value hint, and then applies structured-analysis rules to
Linalg and SCF.

## Proposed Transform

Use a dedicated transform op that targets one loop-local producer and one
surrounding streaming loop:

```mlir
transform.loop.specialize_dead_tile %producer in %loop
  { dead_value = 0xFF800000 : f32 }
```

The exact name is open. The important contract is:

- `%producer` is the op whose result may become a full dead-value tensor.
- `%loop` is the `scf.for` whose induction variable appears in the producer's
  tile predicate.
- `dead_value` is an explicit schedule hint, not rediscovered by the compiler.
- The transform is allowed to fail unless it can prove the rewrite preserves
  the loop-carried results.

The first implementation should be conservative and pattern-driven. General
dataflow can be added later, but the useful optimization comes from a few
well-chosen rules.

## Step 1: Prove a Full Dead Tile

Given a Linalg producer, match a scalar yield of this shape:

```mlir
%selected = arith.select %pred, %live, %dead : f32
linalg.yield %selected : f32
```

where `%dead` equals the schedule-provided dead value.

Then prove `%pred` is false for every point in the producer iteration domain.
For causal attention, after tiling:

```text
q_abs = q_block * BQ + row
k_abs = j * BK + col
live iff k_abs <= q_abs
dead iff k_abs > q_abs
```

A full dead tile is:

```text
forall row in [0, BQ), col in [0, BK):
  j * BK + col > q_block * BQ + row
```

which simplifies to:

```text
j * BK > q_block * BQ + (BQ - 1)
```

Existing MLIR tools to use:

- Linalg op interfaces expose iterator types, indexing maps, `linalg.index`,
  inputs, outputs, and scalar body operations.
- Affine composition utilities can recover expressions such as
  `j * BK + col` from `affine.apply`, `linalg.index`, and loop IVs.
- Presburger / affine constraint utilities, such as
  `FlatLinearValueConstraints`, can prove that the conjunction of loop/tile
  bounds and `pred == true` is empty.

What not to rely on:

- `sccp` and scalar integer range optimizations do not prove "this entire
  tensor tile is a constant". The fact quantifies over the Linalg iteration
  domain, so it needs structured tensor/tile reasoning.

## Step 2: Propagate Dead-Tile Consequences

After proving the producer result is a full dead-value tensor for some loop
iterations, run a small rule-based analysis over loop-local consumers.

Use a compact lattice, for example:

```text
Unknown
AllConstant(value)
AllDead(value)
SameAsLoopIterArg(index)
AllZero
```

The exact states can evolve. The important point is that the analysis should
answer whether each yielded loop-carried value is unchanged in a fully-dead
iteration.

For the attention recurrence, the useful proof is not "everything downstream is
also `-inf`". It is:

```text
if score_tile is all -inf,
then row_max update is the previous row max,
then row_sum update is the previous row sum,
then accumulator update is the previous accumulator,
therefore the scf.for iteration yields its incoming iter_args unchanged.
```

That requires semantic rules for reductions and common elementwise operations:

- `maximumf(dead_value, x) -> x` for `dead_value = -inf`, subject to the chosen
  NaN semantics and assumptions about `x`.
- `exp(-inf) -> 0`.
- `sum(zeros, init) -> init`.
- `matmul(zeros, rhs, init) -> init`.
- Pointwise identity rules such as `x * 1 -> x`, `x + 0 -> x`, and `x / x -> 1`
  only when the required nonzero / non-NaN preconditions are available.

The first version should avoid aggressive floating-point reasoning. For
example, only use rules that directly prove a loop yield equals the incoming
iter_arg. If a downstream value becomes `NaN` in a corner case, the transform
should fail unless the value is proven unused or the NaN path is unreachable.

Existing MLIR tools to use:

- Def-use walking and dominance to restrict analysis to loop-local consumers.
- `linalg::LinalgOp` and `DestinationStyleOpInterface` to distinguish inputs,
  DPS init operands, reductions, and yielded values.
- Existing canonicalization, CSE, and dead-value cleanup after the transform
  rewrites the loop.

What not to rely on:

- MLIR does not currently run an inter-op tensor constant propagation pass
  through arbitrary `linalg.generic` bodies.
- Generic scalar constant folding applies inside scalar/tensor `arith` and
  `math` operations, but it does not evaluate a whole Linalg op over a dense
  constant input.

## Step 3: Derive a Loop Boundary

Once the full-dead condition is represented as an affine inequality in the
streaming loop IV, derive the first dead iteration.

For causal attention:

```text
dead iff j * BK > q_block * BQ + (BQ - 1)
live_ub = min(num_j_tiles, floordiv(q_block * BQ + BQ - 1, BK) + 1)
```

For the common `BQ = 128`, `BK = 64` case:

```text
live_ub = min(num_j_tiles, 2 * q_block + 2)
```

Existing MLIR tools to use:

- Affine maps and `affine.min` can materialize clipped static expressions.
- `scf-for-loop-range-folding` and SCF canonicalization can simplify loop
  bounds after the rewrite.

The transform should not try to solve arbitrary nonlinear arithmetic. Start
with affine expressions and constant positive tile sizes.

## Step 4: Split or Erase the Dead Suffix

The core rewrite is loop splitting:

```mlir
%state_live = scf.for %j = %lb to %live_ub step %step
    iter_args(...) -> (...) {
  // original or simplified live body
}

%state_dead = scf.for %j = %live_ub to %ub step %step
    iter_args(%state_live...) -> (...) {
  // simplified dead body
}
```

If Step 2 proves the dead body yields every incoming iter_arg unchanged, erase
the dead suffix and replace `%state_dead` with `%state_live`.

Existing MLIR tools to use:

- `scf::ForOp` builders and ordinary IR cloning/remapping for the split.
- `scf-for-loop-peeling` or Transform dialect loop peeling only after the split,
  for cleanup or boundary specialization.
- Canonicalization, CSE, and remove-dead-values to erase now-unused tile ops.

Loop peeling is not the discovery mechanism. It can peel static first/last
iterations, but it does not lift a tensor mask predicate into a loop-level
condition. The custom transform must derive and materialize the split point
first.

## Constant Propagation Reality Check

The local toolchain used for this note is Homebrew LLVM/MLIR 20.1.8.

Current MLIR can fold tensor-level `arith` and `math` operations over dense
constants. For example, this is folded by `--sccp --canonicalize`:

```mlir
%a = arith.constant dense<0xFF800000> : tensor<2x3xf32>
%b = arith.maximumf %a, %a : tensor<2x3xf32>
%c = arith.subf %a, %b : tensor<2x3xf32>
%d = math.exp %c : tensor<2x3xf32>
```

The result becomes a dense NaN tensor because `-inf - -inf` is NaN and
`exp(NaN)` is NaN.

Current MLIR does not, however, fold equivalent computation when it is wrapped
in `linalg.generic`. These remain as Linalg ops after
`--sccp --canonicalize --cse`:

```mlir
%r = linalg.generic ... ins(%dense_constant) outs(%empty) {
^bb0(%in: f32, %out: f32):
  %e = math.exp %in : f32
  linalg.yield %e : f32
} -> tensor<...xf32>
```

The same is true for reductions over dense constant tensors. A row-wise
`maximumf` reduction over an all-`-inf` tensor is not folded to an all-`-inf`
constant by the standard cleanup pipeline.

Therefore dead-tile specialization should not be phrased as "let constant
propagation compute the rest". It should be a structured analysis that proves
tile-level facts from Linalg domains and then applies explicit semantic rules
for the small set of operations in the scheduled loop.

## Testing Plan

Add tests in layers:

1. A small `linalg.generic select(live, dead)` test where the transform proves
   all-dead from affine loop/tile inequalities.
2. A test where all-dead propagation proves an `scf.for` iteration yields
   unchanged iter_args and erases the dead suffix.
3. A causal-attention scheduled test where the inner K/V loop upper bound
   becomes `min(num_k_tiles, floor((q_end) / BK) + 1)`.
4. Negative tests where the predicate is non-affine, the dead value does not
   match, NaN-sensitive rules would be required, or a loop yield cannot be
   proven unchanged.

