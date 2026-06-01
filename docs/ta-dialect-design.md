# The `ta` Dialect: A Tensor Algebra IR for Whole-Program Indexed Rewrites

## Status

This is a design note for a proposed MLIR dialect tentatively named `ta`, short for **tensor algebra**. The purpose of the dialect is to support whole-program algebraic rewrites over tensor computations by presenting a tensor program as a graph of scalar indexed expressions over a shared set of named axes.

The motivating example is scaled dot-product attention. We want rewrites such as:

```text
exp(x)  =>  exp2(log2(e) * x)
```

and then we want to move the inserted factor `log2(e)` across elementwise operations, broadcasts, and reductions until it folds into an existing multiplicative constant, such as the attention score scale.

This kind of rewrite is difficult if attention is already split into many `linalg.generic` operations, because each `linalg.generic` has its own local iteration space and scalar region. The proposed `ta` dialect keeps the scalar-program view across those op boundaries while retaining enough scheduling metadata to lower back to `linalg`.

---

## Problem

Tensor programs are often represented as graphs of high-level tensor operations or structured loop nests. That is good for lowering, but it is awkward for whole-program algebraic rewriting.

Consider materialized attention:

```text
Dot[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d]
S[b,h,i,j]   = scale * Dot[b,h,i,j]
M[b,h,i]     = max_j S[b,h,i,j]
P[b,h,i,j]   = exp(S[b,h,i,j] - M[b,h,i])
L[b,h,i]     = sum_j P[b,h,i,j]
Num[b,h,i,e] = sum_j P[b,h,i,j] * V[b,h,j,e]
O[b,h,i,e]   = Num[b,h,i,e] / L[b,h,i]
```

A local rewrite of `exp` gives:

```text
P = exp2(log2(e) * (S - M))
```

To fold `log2(e)` into `scale`, the rewriter must use facts that span the whole attention subgraph:

```text
M = max_j S
log2(e) > 0
log2(e) * max_j S = max_j (log2(e) * S)
S = scale * Dot
log2(e) * scale * Dot = (log2(e) * scale) * Dot
```

This is not an attention-specific rewrite. It is a general scalar and reduction rewrite. The issue is that the program representation must make the relevant scalar expressions and reduction binders visible at once.

---

## Design Goal

`ta` should let a compiler temporarily view a pure tensor subgraph as an indexed scalar expression graph over a shared coordinate system.

The core idea is:

```text
tensor program
  -> ta.scope over named axes
  -> indexed scalar expressions with axis sets
  -> whole-program algebraic rewrites
  -> stage placement
  -> linalg/scf/vector lowering
```

The dialect should support:

1. Whole-program scalar-looking rewrites.
2. Named logical axes such as `b`, `h`, `i`, `j`, `d`, `e`.
3. First-class mathematical reductions, not merely loops.
4. Axis dependency tracking in the type system.
5. Preservation of original `linalg` op boundaries as scheduling hints.
6. Lowering back to `linalg` when a stage has structured-loop form.

---

## Non-Goals

The first version of `ta` does not need to solve every tensor programming problem.

Initial non-goals:

- It does not need to be a final scheduling IR.
- It does not need to model memory layout directly.
- It does not need to represent mutation or side effects.
- It does not need to cover scans, sorting, top-k, or scatter in v1.
- It does not need to guarantee that every rewritten stage lowers to one `linalg.generic`.

The dialect is primarily an algebraic rewrite view. Lowering and scheduling are separate passes.

---

## Core Semantic Model

A `ta` program lives inside a `ta.scope`.

A scope declares an ambient set of named axes:

```mlir
ta.scope axes(%b : Batch,
              %h : Heads,
              %i : QuerySeq,
              %j : KeySeq,
              %d : QKHeadDim,
              %e : ValueDim) {
  ...
}
```

These axes are not loops. They are symbolic coordinates.

Inside the scope, most values are indexed scalar expressions. A value has an element type and an axis support set:

```mlir
!ta.expr<f32, [b,h,i,j]>
```

This means:

```text
an f32 scalar expression that may vary over axes b,h,i,j
```

It is not a materialized tensor. It is a scalar-valued function over those axes.

Examples:

```text
Q[b,h,i,d]       : !ta.expr<f32, [b,h,i,d]>
K[b,h,j,d]       : !ta.expr<f32, [b,h,j,d]>
Q[b,h,i,d]*K[...] : !ta.expr<f32, [b,h,i,j,d]>
sum_d (...)      : !ta.expr<f32, [b,h,i,j]>
max_j (...)      : !ta.expr<f32, [b,h,i]>
```

Broadcasting is implicit: combining values takes the union of axis supports.

```text
x : !ta.expr<f32, [b,h,i,j]>
y : !ta.expr<f32, [b,h,i]>
x - y : !ta.expr<f32, [b,h,i,j]>
```

`y` is constant along `j`.

---

## Core Operations

### `ta.scope`

Owns an algebraic rewrite region and binds named axes to extents.

Sketch:

```mlir
%result = ta.scope axes(%b : %B, %h : %H, %i : %I, %j : %J) {
  ...
  ta.yield %out
}
```

A scope should usually correspond to a maximal pure tensor subgraph.

---

### `ta.at`

Observes an external tensor at indexed coordinates.

```mlir
%q = ta.at %Q[%b, %h, %i, %d]
     : tensor<?x?x?x?xf32> -> !ta.expr<f32, [b,h,i,d]>
```

`ta.at` is the analog of scalar tensor element access, but it produces a `ta.expr` value rather than an ordinary scalar.

---

### `ta.eval`

Observes a staged indexed expression at particular axes.

```mlir
%s = ta.eval %S[%b, %h, %i, %j]
     : !ta.stage<f32, [b,h,i,j]> -> !ta.expr<f32, [b,h,i,j]>
```

`ta.eval` should be rewrite-transparent. Conceptually:

```text
ta.eval(ta.stage axes(...) { body }, indices)  =>  body[axes := indices]
```

but implementations should preserve let-sharing rather than eagerly inlining everything.

---

### `ta.map`

Applies ordinary scalar computation pointwise over the union of operand axis sets.

Sketch:

```mlir
%centered = ta.map (%s, %m)
  : (!ta.expr<f32, [b,h,i,j]>, !ta.expr<f32, [b,h,i]>)
 -> !ta.expr<f32, [b,h,i,j]> {
^bb0(%s0: f32, %m0: f32):
  %r = arith.subf %s0, %m0 : f32
  ta.yield %r : f32
}
```

This avoids needing custom `ta.addf`, `ta.mulf`, `ta.exp`, etc. for every scalar op. A practical implementation can add sugar later:

```mlir
%centered = ta.subf %s, %m
%p = ta.exp %centered
```

but the semantic primitive can be `ta.map`.

---

### `ta.reduce`

A mathematical reduction binder over one or more axes.

```mlir
%dot = ta.reduce add over(%d) identity(%zero) %qk
       : !ta.expr<f32, [b,h,i,j,d]> -> !ta.expr<f32, [b,h,i,j]>
```

`ta.reduce` is not a loop. It is a mathematical expression.

It should carry reducer metadata:

```text
reducer kind: add, mul, max, min, custom
identity value
associative / commutative / idempotent flags
ordered or unordered semantics
fastmath flags
NaN policy
empty-domain semantics
```

This is what makes rewrites such as the following possible:

```text
c * reduce_max_j f(j)
  => reduce_max_j (c * f(j))
```

with guards:

```text
c > 0
j not in axes(c)
max semantics permit the transform
```

---

### `ta.stage`

Names an indexed expression and optionally records a scheduling/materialization boundary.

```mlir
%S = ta.stage @score
     origin = @linalg_score
     kind = "reduction"
     axes(%b, %h, %i, %j) -> f32 {
  ...
  ta.yield %s : f32
}
```

A stage is not necessarily a tensor. It is a named expression with a preferred lowering boundary.

A stage should carry scheduling/provenance metadata:

```text
origin op
original result value
original result shape
original indexing maps
original iterator types
preferred lowering kind
materialization policy
source location
users outside the ta.scope
```

The materialization policy can be:

```text
required    // observable boundary; must materialize or be preserved
preferred   // imported linalg boundary; good default schedule
inlineable  // may be inlined freely
forbidden   // do not materialize here
```

The important rule is:

```text
stages are rewrite-transparent unless marked required
```

---

### `ta.materialize`

Commits an indexed expression to a tensor value.

```mlir
%O = ta.materialize %out over(%b, %h, %i, %e)
     : !ta.expr<f32, [b,h,i,e]> -> tensor<?x?x?x?xf32>
```

Verifier rule:

```text
axes(%out) ⊆ materialized axes
```

Materializing over a superset is a broadcast. Materializing over a missing dependent axis is illegal.

Example illegal materialization:

```text
%p : !ta.expr<f32, [b,h,i,j]>
ta.materialize %p over(%b,%h,%i)   // illegal: j not eliminated
```

---

## Type System

The key type is:

```mlir
!ta.expr<element_type, axis_set>
```

Examples:

```mlir
!ta.expr<f32, []>             // true scalar
!ta.expr<f32, [b,h,i]>        // row-wise scalar expression
!ta.expr<f32, [b,h,i,j]>      // score-like expression
!ta.expr<f32, [b,h,i,e]>      // output-like expression
```

Axis sets should be semantic sets, printed in canonical scope order. The following should not be distinct types:

```text
[b,h,i,j]
[j,i,h,b]
```

The axis support set is best treated as an upper bound on dependency, not necessarily a proven-minimal dependency set. For example:

```text
x : !ta.expr<f32, [i]>
x - x : !ta.expr<f32, [i]>
```

After simplification, this may become:

```text
0 : !ta.expr<f32, []>
```

Core typing rules:

```text
axes(constant) = {}
axes(ta.at T[index_exprs...]) = axes used by index expressions
axes(ta.eval Stage[index_exprs...]) = axes used by index expressions
axes(ta.map f(x1,...,xn)) = union_i axes(xi)
axes(ta.reduce over R of x) = axes(x) - R
axes(ta.select c x y) = axes(c) ∪ axes(x) ∪ axes(y)
```

---

## Attention in `ta`

This section shows the target imported representation for plain materialized attention.

Symbolic axes:

```text
b : Batch
h : Heads
i : QuerySeq
j : KeySeq
d : QKHeadDim
e : ValueDim
```

Inputs:

```text
Q : tensor<Batch x Heads x QuerySeq x QKHeadDim x f32>
K : tensor<Batch x Heads x KeySeq   x QKHeadDim x f32>
V : tensor<Batch x Heads x KeySeq   x ValueDim  x f32>
scale : f32
```

Program:

```mlir
ta.scope axes(%b : Batch,
              %h : Heads,
              %i : QuerySeq,
              %j : KeySeq,
              %d : QKHeadDim,
              %e : ValueDim) {

  %Dot = ta.stage @dot
         kind = "reduction"
         axes(%b, %h, %i, %j) -> f32 {
    %q = ta.at %Q[%b, %h, %i, %d]
         : tensor<?x?x?x?xf32> -> !ta.expr<f32, [b,h,i,d]>
    %k = ta.at %K[%b, %h, %j, %d]
         : tensor<?x?x?x?xf32> -> !ta.expr<f32, [b,h,j,d]>
    %qk = ta.mulf %q, %k
          : !ta.expr<f32, [b,h,i,j,d]>

    %dot = ta.reduce add over(%d) identity(%zero) %qk
           : !ta.expr<f32, [b,h,i,j,d]> -> !ta.expr<f32, [b,h,i,j]>
    ta.yield %dot : !ta.expr<f32, [b,h,i,j]>
  }

  %S = ta.stage @scale
       kind = "elemwise"
       axes(%b, %h, %i, %j) -> f32 {
    %dot = ta.eval %Dot[%b, %h, %i, %j]
           : !ta.expr<f32, [b,h,i,j]>
    %s = ta.mulf %scale, %dot
         : !ta.expr<f32, [b,h,i,j]>
    ta.yield %s : !ta.expr<f32, [b,h,i,j]>
  }

  %M = ta.stage @rowmax
       kind = "reduction"
       axes(%b, %h, %i) -> f32 {
    %s = ta.eval %S[%b, %h, %i, %j]
         : !ta.expr<f32, [b,h,i,j]>
    %m = ta.reduce max over(%j) identity(%neg_inf) %s
         : !ta.expr<f32, [b,h,i,j]> -> !ta.expr<f32, [b,h,i]>
    ta.yield %m : !ta.expr<f32, [b,h,i]>
  }

  %P = ta.stage @exp
       kind = "elemwise"
       axes(%b, %h, %i, %j) -> f32 {
    %s = ta.eval %S[%b, %h, %i, %j]
         : !ta.expr<f32, [b,h,i,j]>
    %m = ta.eval %M[%b, %h, %i]
         : !ta.expr<f32, [b,h,i]>
    %centered = ta.subf %s, %m
                : !ta.expr<f32, [b,h,i,j]>
    %p = ta.exp %centered
         : !ta.expr<f32, [b,h,i,j]>
    ta.yield %p : !ta.expr<f32, [b,h,i,j]>
  }

  %L = ta.stage @rowsum
       kind = "reduction"
       axes(%b, %h, %i) -> f32 {
    %p = ta.eval %P[%b, %h, %i, %j]
         : !ta.expr<f32, [b,h,i,j]>
    %l = ta.reduce add over(%j) identity(%zero) %p
         : !ta.expr<f32, [b,h,i,j]> -> !ta.expr<f32, [b,h,i]>
    ta.yield %l : !ta.expr<f32, [b,h,i]>
  }

  %Num = ta.stage @weighted_sum
         kind = "reduction"
         axes(%b, %h, %i, %e) -> f32 {
    %p = ta.eval %P[%b, %h, %i, %j]
         : !ta.expr<f32, [b,h,i,j]>
    %v = ta.at %V[%b, %h, %j, %e]
         : tensor<?x?x?x?xf32> -> !ta.expr<f32, [b,h,j,e]>
    %pv = ta.mulf %p, %v
          : !ta.expr<f32, [b,h,i,j,e]>
    %num = ta.reduce add over(%j) identity(%zero) %pv
           : !ta.expr<f32, [b,h,i,j,e]> -> !ta.expr<f32, [b,h,i,e]>
    ta.yield %num : !ta.expr<f32, [b,h,i,e]>
  }

  %Oexpr = ta.stage @normalize
           kind = "elemwise"
           axes(%b, %h, %i, %e) -> f32 {
    %num = ta.eval %Num[%b, %h, %i, %e]
           : !ta.expr<f32, [b,h,i,e]>
    %l = ta.eval %L[%b, %h, %i]
         : !ta.expr<f32, [b,h,i]>
    %o = ta.divf %num, %l
         : !ta.expr<f32, [b,h,i,e]>
    ta.yield %o : !ta.expr<f32, [b,h,i,e]>
  }

  %O = ta.materialize %Oexpr over(%b, %h, %i, %e)
       : tensor<Batch x Heads x QuerySeq x ValueDim x f32>

  ta.yield %O
}
```

This representation preserves the original materialized stages, but all stages can be looked through during whole-program rewriting.

---

## Example Rewrite: `exp` to `exp2` in Attention

Let:

```text
c = log2(e)
```

The local rule is:

```text
exp(x) => exp2(c * x)
```

Inside `%P`:

```text
P = exp(S - M)
```

becomes:

```text
P = exp2(c * (S - M))
```

Distribute:

```text
c * (S - M) => c*S - c*M
```

Since:

```text
M = reduce_max_j S
```

and `c > 0` and `j ∉ axes(c)`, apply:

```text
c * reduce_max_j S => reduce_max_j (c*S)
```

Introduce or CSE the rebased score:

```text
S2 = c * S
M2 = reduce_max_j S2
P2 = exp2(S2 - M2)
```

Since:

```text
S = scale * Dot
```

fold:

```text
S2 = c * scale * Dot
   = (c * scale) * Dot
```

The final staged structure can remain almost identical:

```text
Dot  = sum_d Q*K
S2   = (scale * log2(e)) * Dot
M2   = max_j S2
P2   = exp2(S2 - M2)
L2   = sum_j P2
Num2 = sum_j P2*V
O2   = Num2 / L2
```

No attention-specific rule is required. The necessary general rules are scalar distributivity, constant folding, CSE, and reduction movement with side conditions.

---

## Translating from `linalg` to `ta`

The importer should take a pure tensor subgraph of `linalg` operations and translate it into one `ta.scope` with one `ta.stage` per original op.

High-level pipeline:

```text
1. Identify a pure tensor subgraph.
2. Discover a shared set of logical axes.
3. Create one ta.scope that binds those axes.
4. Translate each linalg op into one ta.stage.
5. Preserve original linalg boundaries as stage metadata.
6. Run whole-program ta rewrites.
7. Place/split/fuse stages.
8. Lower stages back to linalg/scf/vector/etc.
```

---

## Axis Discovery

Axis discovery is the most important part of import.

The goal is to avoid giving each `linalg` op unrelated local loop names. Instead, discover that many local loops refer to the same logical coordinate.

For attention, the desired global axes are:

```text
b : Batch
h : Heads
i : QuerySeq
j : KeySeq
d : QKHeadDim
e : ValueDim
```

### Axis Occurrences

Create axis occurrences for:

```text
1. Each linalg iterator of each op.
2. Each dimension of each tensor value in the subgraph.
3. Each result dimension of each translated stage.
```

Example occurrences:

```text
@dot.loop0
@dot.loop1
@dot.loop2
@dot.loop3
@dot.loop4
Q.dim0
Q.dim1
Q.dim2
Q.dim3
K.dim0
K.dim1
K.dim2
K.dim3
Dot.dim0
Dot.dim1
Dot.dim2
Dot.dim3
```

Then use union-find to unify occurrences that indexing maps prove equal.

### Projection Maps

For the dot-product stage, a typical linalg op has local loops:

```text
(b, h, i, j, d)
```

and maps:

```text
Q   : (b,h,i,j,d) -> (b,h,i,d)
K   : (b,h,i,j,d) -> (b,h,j,d)
Dot : (b,h,i,j,d) -> (b,h,i,j)
```

Create equivalences:

```text
@dot.loop.b == Q.dim0 == K.dim0 == Dot.dim0
@dot.loop.h == Q.dim1 == K.dim1 == Dot.dim1
@dot.loop.i == Q.dim2 == Dot.dim2
@dot.loop.j == K.dim2 == Dot.dim3
@dot.loop.d == Q.dim3 == K.dim3
```

Repeat for every linalg op. When `Dot` is consumed by `S`, the dimensions of `Dot` connect the producer and consumer axes.

### Do Not Unify by Size Alone

If `QuerySeq` and `KeySeq` both have extent `L`, they are still distinct axes:

```text
i : QuerySeq
j : KeySeq
```

Unify axes only when indexing maps prove they are the same coordinate.

### Broadcasts

Broadcasts appear as missing axes.

For:

```text
P[b,h,i,j] = exp(S[b,h,i,j] - M[b,h,i])
```

`M` is evaluated with no `j` coordinate:

```mlir
%m = ta.eval %M[%b, %h, %i]
```

Thus:

```text
axes(%m) = [b,h,i]
```

When combined with `S : [b,h,i,j]`, it broadcasts along `j` by type union.

### Non-Projection Maps

For v1, it is reasonable to support projected permutations and broadcasts first.

For affine maps such as convolution indexing:

```text
X[n, oh + r, ow + s, c]
```

do not over-unify all involved axes. Preserve the affine index expression:

```mlir
%x = ta.at %X[%n, %oh + %r, %ow + %s, %c]
```

The dependency set of this access includes all axes used in the index expressions:

```text
axes(%x) = {n, oh, r, ow, s, c}
```

but the input-height dimension is not simply the same axis as `oh` or `r`.

---

## Translating One `linalg.generic`

For each `linalg.generic` op:

1. Read its iterator domain.
2. Map each local iterator to a global `ta` axis using the union-find result.
3. Determine output axes from the output operand indexing map.
4. Determine reduction axes from iterators marked `reduction` that do not appear in the output axes.
5. Translate input element accesses into `ta.at` or `ta.eval`.
6. Copy the scalar payload computation into the stage body.
7. Wrap reduction payloads in `ta.reduce`.
8. Attach provenance and schedule metadata.

A linalg op with one reduction usually becomes:

```mlir
%Y = ta.stage @origin
     kind = "reduction"
     axes(output_axes...) -> element_type {
  %body_value = ... expression over output axes and reduction axes ...
  %r = ta.reduce reducer over(reduction_axes...) identity(%init) %body_value
  ta.yield %r
}
```

An elementwise linalg op becomes:

```mlir
%Y = ta.stage @origin
     kind = "elemwise"
     axes(output_axes...) -> element_type {
  %body_value = ... expression over output axes ...
  ta.yield %body_value
}
```

A contraction becomes the same as a reduction stage, just with multiply/add structure in the scalar body.

---

## Stage Placement and Lowering Back to `linalg`

After rewriting, `ta` must choose a schedule again.

The original linalg boundaries provide an initial placement:

```text
one original linalg op -> one imported ta.stage
```

This is useful, but it must not be mandatory. Rewrites may delete, fuse, split, or create stages.

A stage can lower to one `linalg.generic` when it has structured form:

```text
stage axes       -> parallel iterators
reduce axes      -> reduction iterators
at/eval accesses -> affine indexing maps
scalar body      -> linalg region
```

A stage may not lower cleanly to one `linalg.generic` if it contains:

```text
nested dependent reductions
scan/recurrence
sort/top-k
scatter or data-dependent writes
non-affine indexing
multiple incompatible reduction structures
```

In those cases, the lowering pass can:

```text
1. split the stage,
2. introduce materialization,
3. lower to scf loops,
4. lower to a custom/fused op,
5. or reject the transformation if no legal lowering is available.
```

The important separation is:

```text
ta.scope      = algebraic meaning
ta.stage      = preferred schedule boundary
ta.materialize = actual tensor boundary
```

---

## Implementation Notes

### Types and Attributes

Define:

```mlir
!ta.expr<element_type, axis_set>
```

and likely:

```mlir
#ta.axis<"b">
#ta.axis<"h">
#ta.axis<"i">
```

Types should refer to axis identities as attributes, not directly to SSA values.

The enclosing `ta.scope` binds axis identities to dynamic or static extents.

### Sugar vs Primitive Ops

Initial primitive:

```text
ta.map with scalar region
```

Useful sugar:

```text
ta.addf
ta.subf
ta.mulf
ta.divf
ta.exp
ta.exp2
ta.maximumf
ta.select
```

The sugar can canonicalize to `ta.map`, or `ta.map` can be used as the only semantic primitive.

### Rewriter Requirements

The rewrite engine needs:

```text
axis-support queries
axis independence checks
reduction algebra metadata
positivity/nonnegativity facts
fastmath/NaN policy checks
CSE across stages
stage-transparent eval/build beta-reduction
```

Useful side-condition query:

```text
isIndependent(value, axis) := axis not in axes(value)
```

Useful fact query:

```text
isPositive(value)
```

For constants like `log2(e)`, positivity should be known immediately.

### Floating-Point Legality

Many desirable rewrites are not bitwise-preserving over IEEE floating point.

Examples:

```text
sum_j (c*x_j) = c * sum_j x_j
exp(x) = exp2(log2(e)*x)
```

The dialect should carry an equivalence mode or fastmath flags. Rewrites should be guarded by those flags.

For max movement:

```text
c * max_j x_j = max_j (c*x_j)
```

require:

```text
c > 0
c independent of j
no-NaN semantics or compatible fastmath policy
empty-domain behavior is compatible
```

---

## Minimal Implementation Plan

A practical v1 could be:

1. Implement `ta.scope`, `ta.stage`, `ta.eval`, `ta.at`, `ta.reduce`, `ta.materialize`, `ta.yield`.
2. Implement `!ta.expr<type, axes>`.
3. Implement axis attributes and scope verification.
4. Implement dependency/axis-set inference.
5. Import simple `linalg.generic` ops with projected permutation maps.
6. Import attention-like programs into one `ta.scope` with one stage per linalg op.
7. Implement stage-transparent CSE and beta-reduction for `ta.eval`.
8. Implement scalar rewrites and the reduction movement rule needed for `exp -> exp2`.
9. Lower unchanged or simply rewritten stages back to `linalg.generic`.
10. Add support for affine access expressions later.

A useful first demo is exactly:

```text
plain attention in linalg
  -> ta
  -> exp-to-exp2 rewrite
  -> fold log2(e) into score scale
  -> lower back to linalg with same stage structure
```

Expected before/after:

```text
Before:
  S = scale * Dot
  M = max_j S
  P = exp(S - M)
  L = sum_j P
  Num = sum_j P*V
  O = Num / L

After:
  S2 = (scale * log2(e)) * Dot
  M2 = max_j S2
  P2 = exp2(S2 - M2)
  L2 = sum_j P2
  Num2 = sum_j P2*V
  O2 = Num2 / L2
```

---

## Open Questions

1. Should `ta.stage` be multi-result, or should multi-output linalg ops be split into separate stages?
2. Should `ta.reduce` support custom reducer regions in v1, or only built-in reducers?
3. How much fastmath policy should live on `ta.scope` versus individual ops?
4. Should `ta.expr` axis sets be exact dependencies or conservative supports?
5. How should stage placement be represented: attributes on `ta.stage`, a separate schedule dialect, or transform annotations?
6. How should materialization costs be estimated after rewrites?
7. How should the dialect represent masks: as ordinary selects, or as semantic extended-real masked logits?
8. Should `ta.scan` be part of v1, or added later for recurrence-like models such as Mamba?

---

## Summary

The `ta` dialect is a proposed tensor algebra rewrite IR.

Its main abstraction is:

```text
indexed scalar expressions over a shared set of named axes
```

instead of isolated tensor ops with local loop nests.

The key operations are:

```text
ta.scope       ambient axis universe
ta.stage       named expression plus scheduling hint
ta.at          external tensor element access
ta.eval        stage expression access
ta.map         scalar computation lifted over axes
ta.reduce      mathematical reduction binder
ta.materialize tensor boundary
```

The key type is:

```text
!ta.expr<element_type, axis_set>
```

This type carries the dependency information needed for whole-program rewrites. Original `linalg` ops translate naturally into `ta.stage`s, preserving their scheduling boundaries while making the algebra visible across them.

The design is especially useful for rewrites like attention's `exp` to `exp2` base conversion, where a local scalar identity must be propagated through broadcasts, max reductions, and earlier scalar producers before it becomes profitable.
