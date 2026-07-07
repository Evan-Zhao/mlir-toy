# HTile Guide

This guide collects the current HTile dialect design decisions in Neptune's lowering pipeline.

HTile sits after scheduled L1 -- see [attention pipeline](attention-pipeline.md) for what L1 means.

## L1 To Semantic HTile Translation

`transform.htile.linalg_to_semantic` is the first HTile lowering step.
It is a local structural rewrite over an already scheduled L1 program; it does not rediscover
attention from L0 and does not make memory placement or backend ABI decisions.

The input is assumed to already be scheduled:

- output tiles are explicit through `scf.forall`,
- the K/V streaming loop is explicit through `scf.for`,
- online-softmax state is carried as loop `iter_args`,
- tile math is still expressed with `tensor`, `linalg`, `arith`, and `math`.

The output should preserve that schedule but replace the tile-level structured Linalg compute ops
with semantic HTile operations where HTile has a direct equivalent. Placement, memory access, and
backend ABI choices are separate HTile transforms.

The transform preserves the surrounding program shape:

- It keeps `scf.forall`, `scf.for`, loop-carried tensor state, function arguments, tensor
  returns, `tensor.extract_slice`, and `tensor.parallel_insert_slice` in tensor form.
- It assumes scheduling has made the output tile grid, K/V streaming loop, and recurrence explicit.
- It rewrites only tile-level Linalg compute operations with direct semantic HTile equivalents.

Before this transform, the scheduled program should be normalized so tile-body Linalg ops have
useful ranks. The current integrated pipeline uses:

```mlir
transform.apply_patterns to %func {
  transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
  transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
  transform.apply_patterns.canonicalization
} : !any
transform.apply_cse to %func : !any
```

This removes unit dimensions from the Linalg compute ops by inserting `tensor.collapse_shape` /
`tensor.expand_shape` around them, and rewrites SCF loop-carried tensor state to carry the
collapsed forms where possible.

The transform rewrites these operations:

- `linalg.fill` becomes `htile.full`.
- Contraction-shaped `linalg.generic` becomes `htile.dot`.
  - A contraction whose RHS indexing map is transposed emits `transpose_b`; same for LHS.
  - The DPS init operand is preserved as the accumulating input to `htile.dot`.
- Single-axis add reductions become `htile.reduce ... kind "sum"` followed by an add with the
  original DPS init; same for max.
- All-parallel elementwise `linalg.generic` ops are opened into tensor elementwise ops.
- Elementwise scalar ops are rebuilt generically when they are pure, single-result, regionless,
  successorless `OpTrait::Elementwise` ops.
- `affine.apply` ops inside elementwise bodies are first expanded with MLIR affine utilities, so
  the rewritten tensor body uses ordinary tensor `arith` ops.
- `linalg.index` inside elementwise bodies becomes an `htile.arange` along that dimension,
  broadcast to the elementwise result shape when needed.
- Projected row-vector operands in elementwise ops are materialized with `htile.broadcast`.
- Captured scalar operands in elementwise ops are materialized as tile-shaped `htile.full` values;
  scalar `arith.constant` ops inside the body become tensor `arith.constant` splats.

The reduction translation uses the shared binary-reduction combiner matcher from `LoopTr/Utils`.
HTile-specific code only maps supported combiner ops to HTile reduce kinds.

The transform fails loudly if any original Linalg operation under the target is not converted.
Unsupported payload ops emit an error at the payload op location.

Current limitations:

- Elementwise conversion is generic over pure single-result elementwise ops, but still rejects ops
  with regions, successors, multiple results, or unsupported result typing.
- Reduction conversion only recognizes the currently supported binary reduction combiners.
- It expects static tile shapes.

## Kernel HTile And Backend ABI

The main ABI distinction is between Semantic HTile and Kernel HTile:

- **Semantic HTile** preserves the scheduled L1 tensor ABI. It can still return
  a tensor and publish the final tile with `tensor.parallel_insert_slice`. Input tiles may remain
  as `tensor.extract_slice` values. A separate placement transform may insert `htile.copy`
  operations around these values. This form is convenient for structural comparison against L1
  and for keeping the L1-to-HTile rewrite mostly local.
- **Kernel HTile** matches the existing backend translators. It takes input and output memory
  arguments, uses `htile.load` / `htile.store` at memory boundaries, and has no tensor return.

The current Triton, TileLang, and cuTile translators are backend translators, not general
tensor-return HTile interpreters. They should consume Kernel HTile.
Therefore a semantic HTile program should use `transform.htile.outline_kernels`
before invoking those translators.

`transform.htile.outline_kernels` performs the minimal kernel-boundary conversion used by the
current pipeline:

1. selected top-level `scf.forall` loop nests become `htile.kernel` definitions,
1. the host function gets result-free `htile.launch_func` operations,
1. tensor values crossing selected loop boundaries are materialized as explicit memrefs,
1. kernel-body tensor reads become `htile.load`,
1. kernel-body result publications become `htile.store`,
1. `htile.program_id` and `program_bounds` describe the logical launch domain.

## Placement Transforms

Placement is not part of `transform.htile.linalg_to_semantic`. A later HTile placement transform
can start with this simple policy:

- input tiles copied from Q/K/V extracts use `#htile.encoding<placement = shared>`,
- temporary compute tiles and loop-carried online-softmax state use
  `#htile.encoding<placement = local>`,
- values crossing back to plain tensor L1 use `htile.copy`.

This is intentionally coarse. More precise placement, cache staging, async copies, warp roles, and
layout choices belong to later HTile/backend lowering.

`htile.copy` is a placement/value conversion, not a global memory access. If a placement transform
inserts semantic copies, kernel-ABI legalization still decides which copies become loads/stores.

## Backend Lowering Notes

### Broadcast Semantics

MLIR tensor arithmetic requires equal operand types, so L1 materializes row broadcasts explicitly.
Semantic HTile represents those broadcasts as non-DPS `htile.broadcast` ops. Backend translators can
lower them to unsqueeze / expand-dims forms appropriate for their target.

### Tensor-Return ABI Versus Kernel ABI

The scheduled L1 examples return tensors. Triton kernels naturally take pointer arguments and
store output tiles. A placement/backend pipeline that preserves tensor-return IR may use
`htile.copy` plus `tensor.parallel_insert_slice`; a backend-facing pipeline should instead
introduce or target an explicit output argument and emit `htile.store`.

### Numerical Policy

The current scheduled attention keeps softmax probabilities in `f32` for the `P @ V` dot. Some
backend examples truncate probabilities to `f16` before the dot. That is a numerical policy
choice, not a mechanical HTile translation requirement. The translator should preserve the L1
element types unless a separate lowering policy explicitly changes them.

### Backend Constraints

HTile IR may allow combinations that a backend cannot lower efficiently, such as `f32 x f16` dot
inputs. The semantic compute translator should preserve source element types; backend-specific
legalization can later insert casts, layout conversions, or choose different dot lowerings.
