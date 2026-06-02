# The `ta` Dialect: A Tensor Algebra IR for Whole-Program Indexed Rewrites

## Motivation

`ta`, short for **tensor algebra**, represents pure tensor computations as
scalar indexed expressions over named axes. It gives the compiler a temporary
view of a tensor dataflow graph where scalar algebra, broadcasts, and reductions
are visible across individual `linalg.generic` boundaries.

The motivating case is scaled dot-product attention, whose usual tensor
pipeline is:

```text
Dot[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d]
S[b,h,i,j]   = scale * Dot[b,h,i,j]
M[b,h,i]     = max_j S[b,h,i,j]
P[b,h,i,j]   = exp(S[b,h,i,j] - M[b,h,i])
L[b,h,i]     = sum_j P[b,h,i,j]
Num[b,h,i,e] = sum_j P[b,h,i,j] * V[b,h,j,e]
O[b,h,i,e]   = Num[b,h,i,e] / L[b,h,i]
```

A local rewrite of `exp` gives `P = exp2(log2(e) * (S - M))`. Folding
`log2(e)` into `scale` then requires facts that span the attention subgraph:

```text
log2(e) > 0,  j not in axes(log2(e))
log2(e) * max_j S = max_j (log2(e) * S)
log2(e) * scale * Dot = (log2(e) * scale) * Dot
```

This is not an attention-specific optimization. It is scalar algebra plus
reduction movement with side conditions. `ta` provides the representation that
makes those expressions and binders visible together.

The implemented core is:

```text
tensor program
  -> ta.scope over named axes
  -> scope-local ta.expr values with axis support sets
  -> first-class elementwise and reduction ops
```

There is also a `ta-import-linalg` pass for supported `linalg.generic` tensor
dataflow, and a small PDLL-backed algebraic rewrite driver for the `exp` to
`exp2` transformation. Scope placement, lowering back to `linalg`, and a full
floating-point legality policy remain future work.

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

### `ta.reduce`

Mathematical reduction binder over one or more axes. `ta.reduce` reduces an
existing expression value.

```mlir
%q = ta.at %Q[%b, %h, %i, %d]
     : tensor<?x?x?x?xf32> -> !ta.expr<f32, [b,h,i,d]>
%k = ta.at %K[%b, %h, %j, %d]
     : tensor<?x?x?x?xf32> -> !ta.expr<f32, [b,h,j,d]>
%qk = ta.mulf %q, %k
      : (!ta.expr<f32, [b,h,i,d]>, !ta.expr<f32, [b,h,j,d]>)
     -> !ta.expr<f32, [b,h,i,j,d]>
%dot = ta.reduce #ta.reduce_kind<add> %qk {axes = #ta.axes<d>}
     : !ta.expr<f32, [b,h,i,j,d]> -> !ta.expr<f32, [b,h,i,j]>
```

This op is not a loop. It is a mathematical expression.

The reducer kind is a structured enum attribute, not an arbitrary string.
Built-in reducers include `add`, `mul`, `max`, and `min`.

The importer may attach `ta.import_group` attributes to expression ops to record
that a payload and reduction came from the same source `linalg.generic`. This is
provenance metadata, not a semantic boundary.

`ta.reduce` intentionally has no payload body. Rewrites over `ta` are expected
to be expressible in MLIR pattern languages such as PDLL or DRR, which are much
better at matching and creating ordinary SSA op DAGs than constructing region
bodies. A contraction therefore appears as elementwise expression ops followed
by a bodyless `ta.reduce`. This keeps the mathematical payload visible to local
pattern matching, avoids custom region-building helpers for common rewrites, and
uses `ta.import_group` when we still need to remember that several ops came from
one source operation.

Future reduction metadata will likely include:

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

## Example Rewrite: `exp` to `exp2`

The implemented rewrite target replaces `exp` with `exp2` and uses small
algebraic rules to move the new `log2(e)` factor through the expression graph.
The current implementation is in `lib/TA/ExpToExp2.pdll` and is exposed through:

```mlir
transform.ta.rewrite_exp_to_exp2 %target : !transform.any_op
```

The same pattern set is also available to `transform.apply_patterns` as:

```mlir
transform.apply_patterns.ta.exp_to_exp2
```

The motivating attention fragment inside one `ta.scope` looks like:

```mlir
%dot = ta.reduce #ta.reduce_kind<add> %qk {axes = #ta.axes<d>}
     : !ta.expr<f32, [b,h,i,j,d]> -> !ta.expr<f32, [b,h,i,j]>
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
```

With `c = log2(e)`, the rewrite proceeds as ordinary algebra over the
score expression:

| Step | Rule | Result |
| --- | --- | --- |
| Original softmax numerator | definition | `P = exp(S - M)` |
| Change exponential base | `exp(x) => exp2(c * x)` | `P = exp2(c * (S - M))` |
| Distribute scale | `c * (x - y) => c*x - c*y` | `P = exp2(c*S - c*M)` |
| Move through max | `c * max_j(S) => max_j(c*S)`, for `c > 0` and `j notin axes(c)` | `P = exp2(S2 - M2)` |
| Rebase score | CSE `S2 = c*S`, `M2 = max_j S2` | `P = exp2(S2 - M2)` |
| Fold score scale | `S = scale * Dot` | `S2 = (c * scale) * Dot` |

The current rewrite driver reaches the desired form by repeatedly applying
these separate PDLL rules and running CSE between greedy iterations.

No attention-specific rule is required. Attention only supplies one graph where
these general facts compose into:

```text
S2 = (scale * log2(e)) * Dot
M2 = max_j S2
P2 = exp2(S2 - M2)
```

The current legality checks are still deliberately minimal. The `max` movement
rule checks that the scaling factor is a finite positive axisless constant, but
the dialect still needs a broader fastmath / floating-point equivalence policy
before this should be treated as generally valid for production lowering.

---

## Importing from `linalg`

The `ta-import-linalg` pass imports supported pure tensor dataflow rooted at a
function return value and materializes it as one `ta.scope`.

Example invocation:

```bash
mlir-opt \
  --load-dialect-plugin=libTADialect.so \
  --load-pass-plugin=libTADialect.so \
  --pass-pipeline='builtin.module(func.func(ta-import-linalg))' \
  input.mlir
```

The importer is demand-driven: starting from the returned tensor, it walks
producers backward and asks each producer for an expression over the axes
required by its users.

Supported producer forms include:

```text
function tensor arguments
arith constants
tensor.collapse_shape
tensor.expand_shape
linalg.generic with projected-permutation and broadcast indexing maps
```

The importer currently emits one `ta.scope` for the returned expression graph.
This makes cross-op algebra visible immediately, but it is not yet a full
whole-program scope-placement system.

### Axis Propagation

The current importer uses demand-driven axis propagation. Result tensor
dimensions get fresh axes. Each visited `linalg.generic` uses its output
indexing map to assign those axes to loop dimensions. Input maps then project
loop axes onto operand dimensions and produce `ta.at` expressions.

For attention-like programs, this recovers axes corresponding to:

```text
b : Batch
h : Heads
i : QuerySeq
j : KeySeq
d : QKHeadDim
e : ValueDim
```

For a dot product, a typical `linalg.generic` has local loops:

```text
(b, h, i, j, d)
```

and maps:

```text
Q   : (b,h,i,j,d) -> (b,h,i,d)
K   : (b,h,i,j,d) -> (b,h,j,d)
Dot : (b,h,i,j,d) -> (b,h,i,j)
```

When translating this op, the output map assigns user-requested axes to loop
dimensions:

```text
loop.b := Dot.axis0
loop.h := Dot.axis1
loop.i := Dot.axis2
loop.j := Dot.axis3
```

Reduction loop dimensions that do not appear in the output map get fresh
reduction axes. Input maps then project those loop axes onto operand tensor
dimensions, producing `ta.at` expressions.

### Broadcasts

Broadcasts appear as missing axes or constant affine-map results.

For:

```text
P[b,h,i,j] = exp(S[b,h,i,j] - M[b,h,i])
```

`M` is evaluated with no `j` coordinate, so its expression support is:

```text
axes(%m) = [b,h,i]
```

When combined with `S : [b,h,i,j]`, elementwise type inference broadcasts it
along `j` by taking the union of operand axes in scope order.

### Imported `linalg.generic` Bodies

For each imported `linalg.generic`, the importer translates tensor operands by
projecting loop axes through input maps, then translates the scalar body to
first-class `ta` scalar ops.

Recognized scalar ops currently include floating-point constants,
`arith.extf`, `arith.truncf`, `arith.addf`, `arith.subf`, `arith.mulf`,
`arith.divf`, `arith.maximumf`, `arith.minimumf`, and `math.exp`. Recognized
reduction combiners are add, multiply, maximum, and minimum. Reduction bodies
with those accumulator forms are imported as elementwise payload ops followed by
`ta.reduce`. The importer annotates ops created from each source
`linalg.generic` with `ta.import_group = N : i64`.

### Unsupported Maps

The current importer supports projected permutations and broadcasts. Affine
index expressions are a remaining goal.

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

## Not Implemented Yet: Scope Placement and Lowering

After rewriting, `ta` will need to choose a schedule again.

The importer currently creates one scope for the returned expression graph.
Longer term, original `linalg` boundaries can provide useful initial placement
hints:

```text
one original linalg op -> one imported ta.scope
```

This is useful, but it must not be mandatory. Rewrites may delete, fuse, split,
or create scopes.

A future lowering pass can lower a scope to one `linalg.generic` when it has
structured form:

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

In those cases, the lowering pass could:

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

## Developer Notes

### Types and Attributes

The core expression type is:

```mlir
!ta.expr<element_type, axis_set>
```

Axis identity is stored in attributes:

```mlir
#ta.axis<"b">
#ta.axis<"h">
#ta.axis<"i">
#ta.axes<b, h, i>
```

Types refer to axis identities as attributes, not directly to SSA values.

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

The current rewrite support is intentionally small: PDLL pattern sets are
compiled into the plugin, and transform ops populate those patterns into MLIR's
greedy rewrite driver. `transform.ta.rewrite_exp_to_exp2` applies the
exp-to-exp2 algebra rules repeatedly and runs CSE between iterations.

A more general rewrite engine will need:

```text
axis-support queries
axis independence checks
elementwise op queries
reduction algebra metadata
positivity/nonnegativity facts
fastmath/NaN policy checks
scope-transparent eval/build beta-reduction
```

Elementwise `ta` ops expose a common interface with operand axes and result
axes. Future rewrites should use that interface to choose between matching
specific ops such as `ta.mulf` and reasoning generically about
axis-broadcasted elementwise computation.

Useful side-condition query:

```text
isIndependent(value, axis) := axis not in axes(value)
```

Useful fact query:

```text
isPositive(value)
```

For constants like `log2(e)`, positivity should eventually be known
immediately.

### Not Implemented Yet: Floating-Point Legality

Many desirable rewrites are not bitwise-preserving over IEEE floating point.

Examples:

```text
sum_j (c*x_j) = c * sum_j x_j
exp(x) = exp2(log2(e)*x)
```

The dialect will need an equivalence mode or fastmath flags. Rewrites should be
guarded by those flags.

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

## Current Limitations

1. Global axis discovery with union-find across an entire tensor subgraph.
   The current importer propagates axes backward from each use; it can assign
   distinct fresh reduction axes to equivalent producer computations.
2. Shared expression DAG import and CSE. Reused tensor producers may currently
   be cloned when demanded under different axis contexts.
3. Multi-result import, including multi-output reductions such as max+argmax.
4. Affine access expressions for non-projection maps, needed for direct
   convolution-style indexing.
5. Additional algebraic rewrite rules beyond the current exp-to-exp2 pattern
   set.
6. Scope placement after rewrites: splitting, fusing, or reusing original
   `linalg` boundaries.
7. Lowering `ta.scope` back to `linalg.generic` / `scf` / vector form.
8. Fastmath and floating-point legality policy.
9. Transform-interpreted rewrite patterns, so users can supply rewrite rules
   from transform IR instead of precompiling every PDLL pattern into the plugin.
10. A compact custom rewrite syntax, for example:

    ```text
    match reduce($x{$axes_x} / $d{$axes_d} * $y{$axes_y},
                 axes=$axes_k, reducer="add")
      if intersect($axes_k, $axes_d).empty()
    ```

    This syntax would be closer to tensor-algebra notation than PDLL, but it
    requires a custom parser and a lowering into PDL/PDLL or native rewrite
    patterns.

A useful end-to-end demo remains:

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
