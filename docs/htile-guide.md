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
Therefore a semantic HTile program should run `transform.htile.semantic_to_kernel_abi`
before invoking those translators.

The `semantic_to_kernel_abi` transform currently implements:

1. ranked tensor function arguments are rewritten to memref arguments,
1. ranked tensor function results are appended as trailing memref output arguments,
1. converted input arguments get `bufferization.to_tensor` bridges so the existing tensor body
   remains verifier-valid,
1. direct `tensor.extract_slice` reads from converted function input arguments are rewritten to
   `htile.load`,
1. returned `scf.forall` tensor results are rewritten into side-effecting `htile.store` operations
   by converting the corresponding `tensor.parallel_insert_slice` publications,
1. returned `scf.forall` operations are rebuilt without `shared_outs` and without tensor results,
1. `func.return` operations are rewritten to return no operands.

### Emitting HTile Loads and Stores

After function ABI rewriting inserts `bufferization.to_tensor` bridges,
`semantic_to_kernel_abi` looks at each `extract_slice` and expects this pattern:

```mlir
%tensor = bufferization.to_tensor %q_memref restrict writable : memref<...> to tensor<...>
%tile = tensor.extract_slice %tensor[...] [...] [1, 1, ...] : tensor<...> to tensor<128x64xf16>
%use = htile.dot %tile, ...
```

and rewrites it into:

```mlir
%tile = htile.load %q_memref[...] : memref<...> -> tensor<128x64xf16>
%use = htile.dot %tile, ...
```

Similarly, to produce stores, `semantic_to_kernel_abi` expects each return value
of the function is produced by an `scf.forall`
that publishes it with a `parallel_insert_slice`, like this:

```mlir
%result = scf.forall (...) shared_outs(%out = %init) -> (tensor<...>) {
  ...
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %tile into %out[...] [...] [1, 1, ...]
      : tensor<...> into tensor<...>
  }
}
return %result : tensor<...>
```

and rewrites it into a side-effecting loop with an explicit store:

```mlir
scf.forall (...) {
  ...
  htile.store %tile, %out_memref[...] : tensor<...>, memref<...>
}
return
```

This transform is complete for the current global, causal, and GQA attention pipelines:
the output function has memref input/output arguments, direct `htile.load` / `htile.store`
memory boundaries, and no tensor return.
It intentionally leaves the outer `scf.forall` schedule in place and does not make placement,
launch, or backend-specific layout decisions.

Current ABI conversion assumptions:

- the target is a non-external `func.func` with exactly one `func.return`,
- tensor argument/result ABI types are ranked, unencoded tensors,
- each input memory read is a unit-stride `tensor.extract_slice` directly from a
  `bufferization.to_tensor` bridge over a function memref argument,
- each converted `tensor.extract_slice` result feeds only HTile ops,
- each returned tensor is produced by an `scf.forall`,
- returned `scf.forall` results are published through unit-stride
  `tensor.parallel_insert_slice` ops,
- after those publications are converted, the `scf.forall` shared output block arguments and
  tensor results have no remaining uses.

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
