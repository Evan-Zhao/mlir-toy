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
  transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
} : !any
transform.apply_cse to %func : !any
```

This removes unit dimensions from the Linalg compute ops by inserting `tensor.collapse_shape` /
`tensor.expand_shape` around them. It does not rewrite `scf.for` iter_arg or result types, so
loop-carried state may still have the original higher-rank tensor type.

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
- Projected row-vector operands in elementwise ops are materialized with `linalg.broadcast`.
- Captured scalar operands in elementwise ops are materialized as tile-shaped `htile.full` values;
  scalar `arith.constant` ops inside the body become tensor `arith.constant` splats.

The reduction translation uses the shared binary-reduction combiner matcher from `LoopTr/Utils`.
HTile-specific code only maps supported combiner ops to HTile reduce kinds.

The transform fails loudly if any original non-broadcast Linalg operation under the target is not
converted. Unsupported payload ops emit an error at the payload op location.

Current limitations:

- Elementwise conversion is generic over pure single-result elementwise ops, but still rejects ops
  with regions, successors, multiple results, or unsupported result typing.
- Reduction conversion only recognizes the currently supported binary reduction combiners.
- It expects static tile shapes.

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
tensor-return HTile interpreters. They should consume Kernel HTile. Therefore
`attention_htile.mlir` should either be treated as a semantic intermediate, or rewritten by a
small ABI-conversion pass before invoking those translators.

For the current attention shape, kernel ABI conversion is mechanical:

1. Change the function type from `(q, k, v) -> out_tensor` to
   `(q_mem, k_mem, v_mem, out_mem) -> ()`.
1. Ensure `htile.load` sources are memory arguments, not returned tensor values.
1. Replace the final semantic output boundary with:

   ```mlir
   htile.store %norm_f16_local, %out[%c0, %h, %c0, %c0]
       : tensor<128x64xf16, #local>, memref<1x4x128x64xf16>
   ```

1. Lower the outer parallel grid to the backend translator's expected kernel launch form.
   Handwritten backend examples use `gpu.launch`, while scheduled L1 uses `scf.forall`.

Boundary-only bufferization does not by itself complete this conversion. It may leave a memref
return bridged from a tensor result, and the final `tensor.parallel_insert_slice` remains in the
tensor body. The output argument and `htile.store` rewrite are still HTile/backend ABI work.

## View Normalization And Memory Loads

Do not interpret semantic input copies as backend memory loads. If placement has introduced
`htile.copy`, the HTile dialect still defines it as a placement change for an already materialized
tile, and the Python backend translators lower it as an SSA alias with no emitted memory
operation. A backend legalization pass must rewrite:

```mlir
%tile = tensor.extract_slice %q[...] : tensor<...> to tensor<128x64xf16>
%shared = htile.copy %tile
    : tensor<128x64xf16> -> tensor<128x64xf16, #shared>
```

to:

```mlir
%shared = htile.load %q_memref[...]
    : memref<...> -> tensor<128x64xf16, #shared>
```

Input argument conversion can reuse partial One-Shot Bufferize. Restricting bufferization to the
`func` dialect rewrites tensor function arguments to memrefs and inserts `bufferization.to_tensor`
bridges at the top of the function, while leaving the scheduled tensor body intact:

```shell
mlir-opt input.mlir --canonicalize \
  --one-shot-bufferize='bufferize-function-boundaries allow-unknown-ops dialect-filter=func function-boundary-type-conversion=identity-layout-map'
```

This is useful for L1-to-HTile lowering: view normalization can treat
`bufferization.to_tensor %arg_memref` as a memory-boundary root, emit `htile.load` from
`%arg_memref`, and keep the remaining tensor body value-based.

The input IR may contain intermediate `tensor.extract_slice` values that are not real memory
operations. HTile should usually load the final tile directly from the original source or a
canonical single view of that source.

MLIR bufferization can help expose these relationships because `tensor.extract_slice` bufferizes
to `memref.subview`, and memref alias folding can compose nested subviews. It does not remove the
problem by itself: the downstream HTile/Triton path must either see a base memory object plus
explicit offsets, or explicitly understand subview layout metadata. Feeding arbitrary subview
results to `htile.load` without honoring their offsets and strides would be incorrect.

For the tensor path, implement view normalization as a narrow `tensor.extract_slice` chain resolver:

1. Start from the tile value that will become an `htile.load`.
1. Walk through defining `tensor.extract_slice` ops until reaching a supported root, usually a
   function argument or output tensor.
1. For each slice, use `OffsetSizeAndStrideOpInterface` to read mixed offsets/sizes/strides,
   and use `ExtractSliceOp::computeRankReductionMask()` to map rank-reduced result dimensions.
1. Compose each child offset into the corresponding root dimension:
   `new_offset = parent_offset + child_offset * parent_stride`.
   For the first implementation, require unit strides; then this reduces to simple affine
   addition. Keep sizes from the final tile.
1. Materialize composed dynamic offsets with folded affine/index arithmetic such as
   `affine::makeComposedFoldedAffineApply`, then create one `htile.load` from the root.
1. Fail if the chain contains non-unit strides, dynamic rank ambiguity, non-slice producers, or
   a root that is not a valid memory boundary.

This mirrors the memref subview composition algorithm conceptually, but should operate before full
bufferization so that the rest of the scheduled tile body stays in value-based tensor form.

## Backend Lowering Notes

### Broadcast Semantics

MLIR tensor arithmetic requires equal operand types, so L1 materializes row broadcasts explicitly.
Triton treats row vectors and panels more flexibly. For now, keeping `linalg.broadcast` is the
lowest-risk representation; a later HTile-level broadcast convention may clean up backends.

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
