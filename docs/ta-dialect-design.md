# The `ta` Dialect: Tensor Algebra IR for Indexed Rewrites

`ta` represents pure tensor computations as scalar indexed expressions over
named axes. It is meant to be a temporary compiler view of a tensor dataflow
graph: import from `linalg`, expose algebra and reductions across original
operation boundaries, rewrite, then lower back to structured tensor code.

The core shape is:

```text
linalg tensor dataflow
  -> one ta.scope over named axes with extents
  -> scope-local ta.expr values
  -> elementwise scalar ops plus bodyless reductions
  -> rewritten ta
  -> linalg.generic materializations
```

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

Folding `log2(e)` into the score scale then needs facts that span several
tensor operations:

```text
log2(e) * max_j S = max_j (log2(e) * S), when log2(e) > 0
log2(e) * scale * Dot = (log2(e) * scale) * Dot
```

`ta` is not attention-specific. It gives these expressions a common indexed
form so ordinary scalar algebra and reduction movement can apply across the
whole subgraph.

## A Small TA Program

A `ta` program lives inside a `ta.scope`. The scope declares the symbolic axes
available in the body, including the extent of each axis:

```mlir
%O = ta.scope axes(%b "b" extent 2, %h "h" extent 3,
                   %i "i" extent 4, %j "j" extent 5,
                   %d "d" extent 6, %e "e" extent 7) {
  ...
  ta.yield %o : !ta.expr<f32, [b, h, i, e]>
} : () -> tensor<2x3x4x7xf32>
```

Dynamic extents are also allowed in the syntax:

```mlir
ta.scope axes(%i "i" extent %I, %j "j" extent %J) { ... }
```

The scope block arguments are `index`-typed coordinate handles. The quoted
strings are the semantic axis identities used by attributes and `!ta.expr`
types; the SSA names are local handles only.

Inside the scope, tensor accesses produce scalar indexed expressions:

```mlir
%q = ta.at %Q[%b, %h, %i, %d] {axes = #ta.axes<b, h, i, d>}
    : tensor<2x3x4x6xf32> -> !ta.expr<f32, [b, h, i, d]>
%k = ta.at %K[%b, %h, %j, %d] {axes = #ta.axes<b, h, j, d>}
    : tensor<2x3x5x6xf32> -> !ta.expr<f32, [b, h, j, d]>
%qk = ta.mulf %q, %k
    : (!ta.expr<f32, [b, h, i, d]>, !ta.expr<f32, [b, h, j, d]>)
   -> !ta.expr<f32, [b, h, i, j, d]>
%dot = ta.reduce #ta.reduce_kind<add> %qk {axes = #ta.axes<d>}
    : !ta.expr<f32, [b, h, i, j, d]> -> !ta.expr<f32, [b, h, i, j]>
```

`ta.scope` materializes the yielded expression as a tensor. The yielded
expression may use a subset of the declared axes; axes used only by reductions
or intermediates do not appear in the result expression.

## Expressions And Axes

The central type is:

```mlir
!ta.expr<element_type, axis_set>
```

Examples:

```mlir
!ta.expr<f32, []>             // axisless scalar expression
!ta.expr<f32, [b, h, i]>      // may vary over b, h, i
!ta.expr<f32, [b, h, i, j]>   // score-like expression
```

An expression is not a materialized tensor. It is a scalar-valued function over
its axis support set. Axis sets are semantic sets printed in the enclosing
scope order.

Broadcasting is implicit. Elementwise operations take the union of operand
axes:

```text
!ta.expr<f32, [i]> + !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i, j]>
!ta.expr<f32, [i]> + !ta.expr<f32, [j]>    -> !ta.expr<f32, [i, j]>
```

This makes standard tensor broadcasts visible as scalar algebra. For example,
in softmax:

```text
S : [b, h, i, j]
M : [b, h, i]
S - M : [b, h, i, j]
```

`M` is independent of `j`, so it broadcasts along `j`.

## Rewriting TA

The preferred rewrite surface is an ordinary SSA DAG of first-class `ta`
elementwise ops and bodyless reductions. This is deliberate: MLIR pattern
languages such as PDLL and DRR can match and create ordinary ops directly, but
are awkward for constructing new region bodies. A contraction is therefore
represented as payload expression ops followed by `ta.reduce`, not as a
reduction with an embedded payload region.

Implemented scalar ops include:

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

These ops share the same axis rule: result axes are the union of operand axes
in scope order. `ta.map` remains available as an escape hatch for scalar code
without a dedicated `ta` op.

The current rewrite support is compiled into the plugin:

```mlir
transform.apply_patterns to %target {
  transform.apply_patterns.ta.exp_to_exp2
  transform.apply_patterns.ta.exchange_div_and_matmul
} : !transform.any_op
```

There is also a greedy driver for the `exp` to `exp2` algebra rules:

```mlir
transform.ta.rewrite_exp_to_exp2 %target : !transform.any_op
```

### `exp` To `exp2`

The `exp` rewrite is expressed as several small PDLL algebra rules. In an
attention fragment:

```text
S = scale * Dot
M = max_j S
P = exp(S - M)
```

the rules compose into:

| Step | Rule | Result |
| --- | --- | --- |
| Change base | `exp(x) => exp2(c * x)`, where `c = log2(e)` | `P = exp2(c * (S - M))` |
| Distribute | `c * (x - y) => c*x - c*y` | `P = exp2(c*S - c*M)` |
| Move through max | `c * max_j(S) => max_j(c*S)`, for finite `c > 0` and `j notin axes(c)` | `P = exp2(S2 - M2)` |
| Fold constants | `c * (scale * Dot) => (c * scale) * Dot` | `S2 = (c * scale) * Dot` |

The driver repeatedly applies the rules and runs CSE between greedy iterations.
No single rule needs to match the full attention graph.

### Division And Matmul

The PDLL pattern set also includes a focused rewrite for:

```text
reduce_add_k((x / d) * y) => reduce_add_k(x * y) / d
```

when `d` is independent of the reduced axes. This is representative of the
intended TA rewrite style: match a small expression DAG, query axis
side-conditions, create new ordinary `ta` ops, and let later lowering decide
materialization boundaries.

## Importing From Linalg

The `linalg-to-ta` pass imports supported pure tensor dataflow rooted at a
function return value and materializes it as one `ta.scope`.

```bash
mlir-opt \
  --load-dialect-plugin=libTADialect.so \
  --load-pass-plugin=libTADialect.so \
  --pass-pipeline='builtin.module(func.func(linalg-to-ta))' \
  input.mlir
```

The importer walks the tensor dataflow graph, assigns canonical axes to tensor
dimensions and `linalg.generic` loops, then emits each source operation once in
dominance order using a value-to-value map. Shared producers stay shared in the
TA program.

Supported producer forms:

```text
function tensor arguments
arith constants
tensor.collapse_shape
tensor.expand_shape
linalg.generic with projected-permutation and broadcast indexing maps
```

Recognized scalar body ops:

```text
arith.extf
arith.truncf
arith.addf
arith.subf
arith.mulf
arith.divf
arith.maximumf
arith.minimumf
math.exp
```

Recognized reduction combiners are add, multiply, maximum, and minimum.
Reduction bodies with those accumulator forms are imported as elementwise
payload ops followed by `ta.reduce`.

The importer annotates ops created from each source `linalg.generic` with:

```mlir
{ta.import_group = N : i64}
```

This is provenance metadata. It is useful for lowering and debugging, but it is
not part of the mathematical semantics.

### Axis Discovery

Result tensor dimensions receive axes first. The importer then propagates those
axes backward through output indexing maps to loop dimensions, and through input
indexing maps to operand tensor dimensions. If a tensor value is reached from
multiple users, equivalent dimensions are unified so the producer is not
re-imported with fresh axes.

For a dot product with local loops:

```text
(b, h, i, j, d)
```

and maps:

```text
Q   : (b,h,i,j,d) -> (b,h,i,d)
K   : (b,h,i,j,d) -> (b,h,j,d)
Dot : (b,h,i,j,d) -> (b,h,i,j)
```

the output map assigns axes to `b`, `h`, `i`, and `j`; the missing reduction
loop gets a fresh axis `d`; input maps project those loop axes onto `Q` and
`K`.

Broadcasts appear as missing axes or constant affine-map results. For:

```text
P[b,h,i,j] = exp(S[b,h,i,j] - M[b,h,i])
```

`M` imports as an expression over `[b, h, i]`; combining it with `S` broadcasts
it by unioning axis sets.

## Lowering Back To Linalg

The `ta-to-linalg` pass lowers a `ta.scope` back to `linalg.generic`
materializations.

```bash
mlir-opt \
  --load-dialect-plugin=libTADialect.so \
  --load-pass-plugin=libTADialect.so \
  --pass-pipeline='builtin.module(func.func(ta-to-linalg))' \
  input.mlir
```

Lowering uses `ta.import_group` as an initial partitioning hint. A grouped
payload and reduction can often become one `linalg.generic`, preserving the
shape of the imported program. Ungrouped rewrite-created ops are materialized
as their own structured ops when needed.

Axis sizes come from `ta.scope` extents. The lowering does not infer or
cross-check them from downstream tensor shapes.

Each lowered partition has this structure:

```text
scope axes       -> parallel iterators
reduce axes      -> reduction iterators
ta.at accesses   -> affine indexing maps
scalar TA ops     -> linalg region scalar ops
```

The current lowering is conservative. It handles the attention demo after
`linalg-to-ta`, `exp` to `exp2`, and division/matmul exchange, but more complex
partitions may need to be split or lowered through a more general path such as
`scf`.

Transform schedules that need to keep handles across the TA-to-linalg boundary
can use a TA-side einsum matcher and the transform lowering op:

```mlir
transform.named_sequence @match_ta_matmul(%candidate: !transform.any_op {transform.readonly})
    -> !transform.any_op {
  %matched = transform.match.ta.einsum %candidate
      {equation = "i k, k j -> i j"}
      : (!transform.any_op) -> !transform.any_op
  transform.yield %matched : !transform.any_op
}

%ta_matmuls = transform.collect_matching @match_ta_matmul in %ta_func
    : (!transform.any_op) -> !transform.any_op
transform.ta.to_linalg %ta_func : !transform.any_op

// `%ta_matmuls` now points at the lowered `linalg.generic` ops when there is a
// clear TA-root-to-linalg-op mapping.
transform.structured.tile_using_forall %ta_matmuls tile_sizes [64, 64, 0]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

The matcher currently recognizes two-input add reductions of a multiply. Axis
names in the equation are pattern symbols, so the equation above can match TA
axes named `%i0`, `%r0`, or anything else as long as the reuse structure is the
same. This is enough to select matmul-like contractions in TA without
re-matching the lowered `linalg.generic` region body. Handle preservation is
best effort: handles to expression roots that materialize as linalg ops
survive; handles to internal TA ops that lower into indexing maps or
linalg-region scalar ops may be dropped.

## End-To-End Demo Shape

`test/TA/attention.mlir` demonstrates the implemented flow:

```mlir
transform.named_sequence @__transform_main(%module: !transform.any_op) {
  %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %ta_func = transform.apply_registered_pass "linalg-to-ta" to %func
      : (!transform.any_op) -> !transform.any_op
  transform.ta.rewrite_exp_to_exp2 %ta_func : !transform.any_op
  transform.apply_patterns to %ta_func {
    transform.apply_patterns.ta.exchange_div_and_matmul
  } : !transform.any_op
  %linalg_func = transform.apply_registered_pass "ta-to-linalg" to %ta_func
      : (!transform.any_op) -> !transform.any_op
  transform.yield
}
```

The resulting program has the same high-level tensor computation, but the
softmax numerator uses `exp2`, and the score scale has absorbed the `log2(e)`
factor.

## Operation Reference

### `ta.scope`

Declares axes and materializes the yielded expression as a tensor.

```mlir
%out = ta.scope axes(%i "i" extent 16, %j "j" extent 32) {
  ...
  ta.yield %expr : !ta.expr<f32, [i, j]>
} : () -> tensor<16x32xf32>
```

All expression-level `ta` ops must be nested inside a scope, and may only use
axes declared by that scope.

### `ta.at`

Reads a tensor at symbolic coordinates and returns a `ta.expr`.

```mlir
%x = ta.at %tensor[%i, %j] {axes = #ta.axes<i, j>}
    : tensor<16x32xf32> -> !ta.expr<f32, [i, j]>
```

### `ta.map`

Runs a scalar region pointwise over the union of operand axes.

```mlir
%diff = ta.map %x, %y {
^bb0(%sx : f32, %sy : f32):
  %r = arith.subf %sx, %sy : f32
  ta.yield %r : f32
} : (!ta.expr<f32, [i]>, !ta.expr<f32, [j]>)
 -> !ta.expr<f32, [i, j]>
```

Prefer first-class scalar TA ops when one exists; use `ta.map` for scalar code
that has not been canonicalized.

### `ta.reduce`

Reduces an expression over one or more axes.

```mlir
%sum = ta.reduce #ta.reduce_kind<add> %payload {axes = #ta.axes<k>}
    : !ta.expr<f32, [i, j, k]> -> !ta.expr<f32, [i, j]>
```

Built-in reducer kinds are `add`, `mul`, `max`, and `min`. The reducer kind is
a structured enum attribute, not a string.

`ta.reduce` has no payload body. Its input is an ordinary expression value, so
the payload remains visible to standard op-DAG pattern matching.

## Types And Attributes

Axis identities are attributes:

```mlir
#ta.axis<"b">
#ta.axes<b, h, i>
```

Expression types refer to these identities:

```mlir
!ta.expr<f32, [b, h, i]>
```

The enclosing `ta.scope` declares the allowed axis identities and provides
block arguments that can be used as coordinates. The scope result tensor type
is the materialized type of its yielded expression.

Core typing rules:

```text
axes(ta.constant) = {}
axes(ta.at T[index_exprs...]) = axes named by its axes attribute
axes(ta.map f(x1,...,xn)) = union_i axes(xi)
axes(ta.elementwise_op(x1,...,xn)) = union_i axes(xi)
axes(ta.reduce over R x) = axes(x) - R
axes(ta.select c x y) = axes(c) union axes(x) union axes(y)
```

The result axis order is the enclosing `ta.scope` order.

The axis support set is a conservative dependency support, not necessarily a
minimal dependency set. For example, `x - x` may initially keep `axes(x)` even
though simplification can later produce an axisless zero.

## Current Limitations

The implemented dialect and passes cover the current attention rewrite demo,
but several areas remain intentionally narrow:

1. Import is limited to projected-permutation and broadcast indexing maps.
   Affine access expressions such as convolution indices are not imported yet.
1. Import and lowering are single-result oriented. Multi-output reductions such
   as max+argmax still need a representation strategy.
1. Lowering is conservative and targets `linalg.generic`; complex partitions
   may need splitting or non-linalg lowering.
1. Dynamic scope extents parse, but `ta-to-linalg` currently requires static
   extents.
1. Floating-point legality is minimal. Rewrites such as `exp -> exp2` and
   moving positive factors through `max` need a fuller fastmath / NaN policy
   before they are generally legal.
1. PDLL patterns are compiled into the plugin. Transform-interpreted rewrite
   patterns would let users provide rules without rebuilding.
1. A more compact custom rewrite syntax could sit above PDLL, for example:

   ```text
   match reduce($x{$axes_x} / $d{$axes_d} * $y{$axes_y},
                axes=$axes_k, reducer="add")
     if intersect($axes_k, $axes_d).empty()
   ```

   That syntax is closer to tensor-algebra notation, but would require a
   custom parser and a lowering into PDL/PDLL or native rewrite patterns.
