# The `ta` Dialect: A Tensor Algebra IR for Whole-Program Indexed Rewrites

## Motivation

`ta`, short for **tensor algebra**, is a proposed MLIR dialect for whole-program
algebraic rewrites over tensor computations. It presents a pure tensor subgraph
as scalar indexed expressions over a shared set of named axes, so rewrites can
see across individual `linalg.generic` boundaries while retaining enough
structure to lower back to `linalg`.

The motivating case is scaled dot-product attention:

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

To fold `log2(e)` into `scale`, the rewriter must use facts that span the whole
attention subgraph:

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
  -> ta.scope boundaries over named axes
  -> scope-local indexed scalar expressions with axis sets
  -> whole-program algebraic rewrites
  -> scope placement
  -> linalg/scf/vector lowering
```

The dialect should support:

1. Whole-program scalar-looking rewrites.
2. Named logical axes such as `b`, `h`, `i`, `j`, `d`, `e`.
3. First-class mathematical reductions, not merely loops.
4. Axis dependency tracking in the type system.
5. Preservation of original `linalg` op boundaries as rewrite scopes.
6. Lowering back to `linalg` when a scope has structured-loop form.

---

## Core Semantic Model

A `ta` program lives inside a `ta.scope`.

A scope declares an ambient set of named axes:

```mlir
ta.scope axes(
  %b "b" : index, %h "h" : index,
  %i "i" : index, %j "j" : index,
  %d "d" : index, %e "e" : index) {
  ...
}
```

These axes are not loops. They are symbolic coordinates available to the scalar
indexed expressions inside the scope body. In the current syntax the block
arguments are `index`-typed SSA values, while the quoted strings are the stable
semantic axis identities used by attributes and `!ta.expr` types. The SSA name
is only a local handle and must not define the axis identity.

Using ordinary `index` for symbolic coordinates is a pragmatic v1 choice. It
allows affine-style index expressions such as convolution input coordinates,
but it also means generic `index`/`arith` operations could accidentally treat
axis coordinates like runtime loop values. A future version may introduce a
dedicated coordinate type, such as `!ta.index<[axes]>`, or restrict which ops
may consume scope axis block arguments.

All expression-level `ta` operations, such as `ta.at`, `ta.map`, and
`ta.reduce`, must be nested inside a `ta.scope`. They may only use or define
axes declared by the enclosing scope. This makes every scope a closed indexed
expression over a known coordinate system.

Inside a scope, most values are indexed scalar expressions. A value has an element type and an axis support set:

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

Owns one indexed expression region, declares the allowed body axes, and
materializes the yielded expression as a tensor result.

```mlir
%O = ta.scope axes(%b "b" : index, %h "h" : index,
                   %i "i" : index, %j "j" : index,
                   %d "d" : index, %e "e" : index) {
  ...
  ta.yield %o : !ta.expr<f32, [b,h,i,e]>
} : () -> tensor<?x?x?x?xf32>
```

A scope result is the tensor version of the expression yielded by its terminator.
The scope axis list is the ambient coordinate universe for the body,
not necessarily the result shape.
The yielded expression may depend on a subset of the scope axes;
axes used only inside reductions or intermediate expressions
do not appear in the result expression.

Scopes are the only place where `ta` indexed expression ops may appear.

### `ta.at`

Observes an external tensor at indexed coordinates.

```mlir
%q = ta.at %Q[%b, %h, %i, %d]
     : tensor<?x?x?x?xf32> -> !ta.expr<f32, [b,h,i,d]>
```

`ta.at` is the analog of scalar tensor element access, but it produces a `ta.expr` value rather than an ordinary scalar.

---

### `ta.map`

Applies ordinary scalar computation pointwise over the union of operand axis sets.

Sketch:

```mlir
%centered = ta.map %s, %m {
^bb0(%s0: f32, %m0: f32):
  %r = arith.subf %s0, %m0 : f32
  ta.yield %r : f32
} : (!ta.expr<f32, [b,h,i,j]>, !ta.expr<f32, [b,h,i]>)
 -> !ta.expr<f32, [b,h,i,j]>
```

The body computes on ordinary scalar values. The `ta.map` result has the
yielded scalar element type and the union of all operand axes, ordered by the
enclosing `ta.scope`.

For rewrite friendliness, common scalar operations also have first-class `ta`
ops. These are not merely pretty syntax; they are the preferred canonical form
for algebraic rewrites:

```mlir
%centered = ta.subf %s, %m
  : (!ta.expr<f32, [b,h,i,j]>, !ta.expr<f32, [b,h,i]>)
 -> !ta.expr<f32, [b,h,i,j]>
%p = ta.exp %centered
  : (!ta.expr<f32, [b,h,i,j]>) -> !ta.expr<f32, [b,h,i,j]>
```

`ta.map` remains the escape hatch for scalar code without a dedicated `ta` op.

---

### `ta.reduce` and `ta.map_reduce`

Mathematical reduction binders over one or more axes. `ta.reduce` reduces an
existing expression value. `ta.map_reduce` computes a local payload expression
region and reduces the yielded value.

```mlir
%dot = ta.map_reduce #ta.reduce_kind<add> {
  %q = ta.at %Q[%b, %h, %i, %d]
       : tensor<?x?x?x?xf32> -> !ta.expr<f32, [b,h,i,d]>
  %k = ta.at %K[%b, %h, %j, %d]
       : tensor<?x?x?x?xf32> -> !ta.expr<f32, [b,h,j,d]>
  %qk = ta.mulf %q, %k
        : (!ta.expr<f32, [b,h,i,d]>, !ta.expr<f32, [b,h,j,d]>)
       -> !ta.expr<f32, [b,h,i,j,d]>
  ta.yield %qk : !ta.expr<f32, [b,h,i,j,d]>
} {axes = #ta.axes<d>} : !ta.expr<f32, [b,h,i,j]>
```

These ops are not loops. They are mathematical expressions.

The reducer kind is a structured enum attribute, not an arbitrary string.
Built-in reducers include `add`, `mul`, `max`, and `min`.

`ta.map_reduce` preserves a single structured contraction-like unit.
`ta.reduce` is equivalent when the payload already exists:

```mlir
%qk = ta.mulf %q, %k
  : (!ta.expr<f32, [b,h,i,d]>, !ta.expr<f32, [b,h,j,d]>)
 -> !ta.expr<f32, [b,h,i,j,d]>
%dot = ta.reduce #ta.reduce_kind<add> %qk {axes = #ta.axes<d>}
  : !ta.expr<f32, [b,h,i,j,d]> -> !ta.expr<f32, [b,h,i,j]>
```

Both ops represent the same mathematical binder and share the same verifier rule
through a reduce-like interface. `ta.map_reduce` maps more directly to and from
one `linalg.generic` reduction. `ta.reduce` is a more normalized expression DAG
node and is convenient when the payload is shared or already exists as a value.
A `ta.map_reduce` body that only yields an existing value should canonicalize to
`ta.reduce`.

Reduction ops should carry reducer metadata:

```text
reducer kind: add, mul, max, min, later custom
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

Axis sets should be semantic sets, printed in the enclosing scope's canonical
axis order. The following should not be distinct types:

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
axes(ta.map f(x1,...,xn)) = union_i axes(xi)
axes(ta.elementwise_op(x1,...,xn)) = union_i axes(xi)
axes(ta.map_reduce over R { yield x }) = axes(x) - R
axes(ta.reduce over R x) = axes(x) - R
axes(ta.select c x y) = axes(c) ∪ axes(x) ∪ axes(y)
```

The same elementwise rule is used by `ta.map` and by sugar ops such as
`ta.addf`, `ta.mulf`, `ta.exp`, `ta.cmpf`, and `ta.select`. Binary and ternary
ops implicitly broadcast operands over missing axes:

```text
!ta.expr<f32, [i]> + !ta.expr<f32, [i,j]> -> !ta.expr<f32, [i,j]>
!ta.expr<f32, [i]> + !ta.expr<f32, [j]>   -> !ta.expr<f32, [i,j]>
```

The result axis order is the enclosing `ta.scope` order, not operand order.

---

## Attention in `ta`

This section sketches the target representation for plain attention inside one
ambient `ta.scope`. The full executable MLIR example lives in
`test/TA/attention.mlir`.

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
scale : f32 constant
```

Program shape:

```mlir
%O = ta.scope axes(%b "b" : index, %h "h" : index,
                   %i "i" : index, %j "j" : index,
                   %d "d" : index, %e "e" : index) {
  %dot = ta.map_reduce #ta.reduce_kind<add> {
    ...
    ta.yield %qk : !ta.expr<f32, [b,h,i,j,d]>
  } {axes = #ta.axes<d>} : !ta.expr<f32, [b,h,i,j]>

  %scale_expr = ta.constant 1.250000e-01 : f32 : !ta.expr<f32, []>
  %s = ta.mulf %scale_expr, %dot
       : (!ta.expr<f32, []>, !ta.expr<f32, [b,h,i,j]>)
      -> !ta.expr<f32, [b,h,i,j]>

  %m = ta.reduce #ta.reduce_kind<max> %s {axes = #ta.axes<j>}
       : !ta.expr<f32, [b,h,i,j]> -> !ta.expr<f32, [b,h,i]>

  %centered = ta.subf %s, %m
       : (!ta.expr<f32, [b,h,i,j]>, !ta.expr<f32, [b,h,i]>)
      -> !ta.expr<f32, [b,h,i,j]>
  %p = ta.exp %centered
       : (!ta.expr<f32, [b,h,i,j]>) -> !ta.expr<f32, [b,h,i,j]>

  %l = ta.reduce #ta.reduce_kind<add> %p {axes = #ta.axes<j>}
       : !ta.expr<f32, [b,h,i,j]> -> !ta.expr<f32, [b,h,i]>

  %num = ta.map_reduce #ta.reduce_kind<add> {
    ...
    ta.yield %pv : !ta.expr<f32, [b,h,i,j,e]>
  } {axes = #ta.axes<j>} : !ta.expr<f32, [b,h,i,e]>

  %o = ta.divf %num, %l
       : (!ta.expr<f32, [b,h,i,e]>, !ta.expr<f32, [b,h,i]>)
      -> !ta.expr<f32, [b,h,i,e]>
  ta.yield %o : !ta.expr<f32, [b,h,i,e]>
} : () -> tensor<Batch x Heads x QuerySeq x ValueDim x f32>
```

One `ta.scope` can expose the whole attention computation as a scalar indexed
expression graph. Intermediate reductions over `d` and `j` are internal binders;
only the final yielded expression axes `[b,h,i,e]` determine the materialized
tensor result.

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

The final scoped structure can remain almost identical:

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

The importer should take a pure tensor subgraph of `linalg` operations and
translate it into a graph of `ta.scope` operations, usually one scope per
original op.

High-level pipeline:

```text
1. Identify a pure tensor subgraph.
2. Discover a shared set of logical axes.
3. Translate each linalg op into one ta.scope with the axes its body may use.
4. Run whole-program ta rewrites.
5. Place/split/fuse scopes.
6. Lower scopes back to linalg/scf/vector/etc.
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
3. Each result dimension of each translated scope.
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

For the dot-product scope, a typical linalg op has local loops:

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
5. Translate input element accesses into `ta.at`.
6. Copy the scalar payload computation into the scope body.
7. Wrap reduction payloads in `ta.map_reduce`, or use `ta.reduce` when the
   payload already exists as a value.
8. Preserve the original op boundary as a `ta.scope`.

A linalg op with one reduction usually becomes:

```mlir
%Y = ta.scope axes(%i "i" : index, %j "j" : index, %red "r" : index) {
  %sum = ta.map_reduce #ta.reduce_kind<add> {
    %body_value = ... expression over output axes and reduction axes ...
    ta.yield %body_value : !ta.expr<element_type, [i,j,r]>
  } {axes = #ta.axes<r>} : !ta.expr<element_type, [i,j]>
  ta.yield %sum : !ta.expr<element_type, [i,j]>
} : () -> tensor<...xelement_type>
```

An elementwise linalg op becomes:

```mlir
%Y = ta.scope axes(%i "i" : index, %j "j" : index) {
  %body_value = ... expression over output axes ...
  ta.yield %body_value
} : () -> tensor<...xelement_type>
```

A contraction becomes the same as a reduction scope, just with multiply/add structure in the scalar body.

---

## Scope Placement and Lowering Back to `linalg`

After rewriting, `ta` must choose a schedule again.

The original linalg boundaries provide an initial placement:

```text
one original linalg op -> one imported ta.scope
```

This is useful, but it must not be mandatory. Rewrites may delete, fuse, split, or create scopes.

A scope can lower to one `linalg.generic` when it has structured form:

```text
scope axes       -> parallel iterators
reduce axes      -> reduction iterators
at/eval accesses -> affine indexing maps
scalar body      -> linalg region
```

A scope may not lower cleanly to one `linalg.generic` if it contains:

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
1. split the scope,
2. create additional ta.scope materialization boundaries,
3. lower to scf loops,
4. lower to a custom/fused op,
5. or reject the transformation if no legal lowering is available.
```

The important invariant is:

```text
ta.scope body = algebraic scalar expression over declared axes
ta.scope op   = tensor materialization boundary
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
#ta.axes<b, h, i>
```

Types should refer to axis identities as attributes, not directly to SSA values.

The enclosing `ta.scope` declares the axis identities its body may use with
axis block arguments. The quoted axis name is stored as an attribute; the SSA
name is a printable handle for the coordinate in the body. The scope result
tensor type carries the materialized shape.

### Sugar vs Primitive Ops

Escape hatch:

```text
ta.map with scalar region
```

Canonical scalar rewrite surface:

```text
ta.constant

ta.negf
ta.addf
ta.subf
ta.mulf
ta.divf
ta.maximumf
ta.minimumf
ta.maxnumf
ta.minnumf

ta.absf
ta.ceil
ta.exp
ta.exp2
ta.floor
ta.log
ta.log2
ta.rsqrt
ta.sqrt
ta.tanh
ta.extf
ta.truncf

ta.powf
ta.fma

ta.cmpf
ta.select
```

These ops share the same type rule as `ta.map`: result axes are the union of
operand axes in the enclosing scope order. Rewriters should primarily match
these first-class ops. `ta.map` is available for imported scalar regions that
have not been canonicalized to a known operation.

### Rewriter Requirements

The rewrite engine needs:

```text
axis-support queries
axis independence checks
elementwise op queries
reduction algebra metadata
positivity/nonnegativity facts
fastmath/NaN policy checks
CSE across scopes
scope-transparent eval/build beta-reduction
```

Elementwise `ta` ops should expose a common interface with the scalar op kind,
operand axes, and result axes, so rewrites can choose between matching specific
ops such as `ta.mulf` and reasoning generically about axis-broadcasted
elementwise computation.

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

1. Implement `ta.scope`, `ta.at`, `ta.reduce`, `ta.map_reduce`, `ta.yield`.
2. Implement `!ta.expr<type, axes>`.
3. Implement axis attributes and scope-local axis verification.
4. Implement dependency/axis-set inference.
5. Import simple `linalg.generic` ops with projected permutation maps.
6. Import attention-like programs into one scope graph that can span multiple linalg ops.
7. Implement scalar rewrites and the reduction movement rule needed for `exp -> exp2`.
8. Lower unchanged or simply rewritten scopes back to `linalg.generic`.
9. Add support for affine access expressions later.

A useful first demo is exactly:

```text
plain attention in linalg
  -> ta
  -> exp-to-exp2 rewrite
  -> fold log2(e) into score scale
  -> choose scope placement
  -> lower back to linalg
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

1. Should `ta.scope` be multi-result, or should multi-output linalg ops be split into separate scopes?
2. Should reduction ops support custom reducer definitions in v1, or only built-in reducers?
3. How much fastmath policy should live on `ta.scope` versus individual ops?
4. Should `ta.expr` axis sets be exact dependencies or conservative supports?
5. How should scope placement be represented: a separate schedule dialect, or transform annotations?
6. How should materialization costs be estimated after rewrites?
7. How should the dialect represent masks: as ordinary selects, or as semantic extended-real masked logits?
8. Should `ta.scan` be part of v1, or added later for recurrence-like models such as Mamba?
