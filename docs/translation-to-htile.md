# Linalg to HTile Translation

This note describes the intended structural translation from scheduled L1
attention IR, such as `attention.mlir`, into value-based HTile IR, such as the
handwritten `attention_htile.mlir`.

The input is assumed to already be scheduled:

- output tiles are explicit through `scf.forall`,
- the K/V streaming loop is explicit through `scf.for`,
- online-softmax state is carried as loop `iter_args`,
- tile math is still expressed with `tensor`, `linalg`, `arith`, and `math`.

The output should preserve that schedule but replace the tile-level memory,
contraction, reduction, and placement decisions with HTile operations.

## Translation Shape

The first pass should produce **Semantic HTile**. It should be a local
structural rewrite over the scheduled L1 loop nest; it should not rediscover
attention from the original L0 algorithm.

### Linalg To Semantic HTile

- [ ] Require normalized tile ranks.
      Run `mlir-opt --linalg-fold-unit-extent-dims --canonicalize` before this
      pass, or reject unnormalized Linalg tile bodies. The pass may still see
      boundary `collapse_shape` / `expand_shape` around loop-carried state.

- [ ] Preserve `scf.forall` and `scf.for`.
      Keep the existing scheduled loop structure and rewrite loop-carried tile
      tensor types to HTile-encoded tensor types.

- [ ] Rewrite input tile extracts to semantic placement copies.
      Convert a rank-reduced `tensor.extract_slice` from Q/K/V into
      `tensor.extract_slice` followed by `htile.copy` to `#shared`. Do not compose
      slice chains or create `htile.load` in this pass.

- [ ] Rewrite tile fills to `htile.full`.
      Use `#local` for online-softmax state, accumulator tiles, and scalar tile
      constants such as the score scale.

- [ ] Rewrite attention contractions to `htile.dot`.
      Match the contraction indexing maps, not op names. Emit `transpose_b` for
      `Q @ K^T`, and use the accumulator operand for `acc_scaled + P @ V`.

- [ ] Rewrite row reductions to `htile.reduce`.
      Map add reductions to `kind "sum"` and max reductions to `kind "max"`.
      Remap axes after unit dimensions have been removed.

- [ ] Preserve pointwise tensor ops.
      Keep tensor `arith`, `math`, and `linalg.broadcast` for elementwise math and
      row-vector broadcasts.

- [ ] Preserve the tensor-return boundary.
      Keep the final `htile.copy` from local HTile tensor to plain tensor and keep
      `tensor.parallel_insert_slice` / `return`.

### Semantic HTile To Kernel HTile

- [ ] Convert function boundaries to backend memory arguments.
      Use memref/pointer-style Q/K/V arguments, introduce an explicit output
      argument, and remove the tensor return.

- [ ] Legalize semantic input copies to `htile.load`.
      Normalize `tensor.extract_slice` chains, compose offsets, and rewrite
      `tensor.extract_slice` + `htile.copy` into `htile.load` from the memory root.

- [ ] Legalize the output boundary to `htile.store`.
      Replace final `htile.copy` + `tensor.parallel_insert_slice` with
      `htile.store`.

- [ ] Lower the outer parallel grid to the backend launch form.
      The current backend examples use `gpu.launch`; scheduled L1 uses
      `scf.forall`.

## Placement Policy

The first implementation can use a simple placement policy:

- input tiles copied from Q/K/V extracts use `#htile.encoding<placement = shared>`,
- temporary compute tiles and loop-carried online-softmax state use
  `#htile.encoding<placement = local>`,
- values crossing back to plain tensor L1 use `htile.copy`.

This is intentionally coarse. More precise placement, cache staging, async
copies, warp roles, and layout choices belong to later HTile/backend lowering.

## Backend Translator ABI

There are two useful HTile forms:

- **Semantic HTile** preserves the scheduled L1 tensor ABI. It can still return
  a tensor and publish the final tile with `tensor.parallel_insert_slice`.
  Input tiles may also remain as `tensor.extract_slice` followed by
  `htile.copy` to an HTile placement. This form is convenient for structural
  comparison against L1 and for keeping the L1-to-HTile rewrite mostly local.
- **Kernel HTile** matches the existing backend translators. It takes input and
  output memory arguments, uses `htile.load` / `htile.store` at memory
  boundaries, and has no tensor return value.

The current Triton, TileLang, and cuTile translators are backend translators,
not general tensor-return HTile interpreters. They should consume Kernel HTile.
Therefore `attention_htile.mlir` should either be treated as a semantic
intermediate, or rewritten by a small ABI-conversion pass before invoking those
translators.

Do not interpret semantic input copies as backend memory loads. The HTile
dialect currently defines `htile.copy` as a placement change for an already
materialized tile, and the Python backend translators lower it as an SSA alias
with no emitted memory operation. A backend legalization pass must rewrite:

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

Input argument conversion can reuse partial One-Shot Bufferize. Restricting
bufferization to the `func` dialect rewrites tensor function arguments to
memrefs and inserts `bufferization.to_tensor` bridges at the top of the
function, while leaving the scheduled tensor body intact:

```shell
mlir-opt input.mlir \
  --one-shot-bufferize='bufferize-function-boundaries allow-unknown-ops dialect-filter=func function-boundary-type-conversion=identity-layout-map' \
  --canonicalize
```

This is useful for L1-to-HTile lowering: view normalization can treat
`bufferization.to_tensor %arg_memref` as a memory-boundary root and emit
`htile.load` from `%arg_memref` directly. The remaining tensor body still
preserves value-based tile recurrences.

For the current attention shape, that conversion is mechanical:

1. Change the function type from `(q, k, v) -> out_tensor` to
   `(q_mem, k_mem, v_mem, out_mem) -> ()`.
1. Ensure `htile.load` sources are memory arguments, not returned tensor values.
1. Replace the final `htile.copy` plus `tensor.parallel_insert_slice` with:

   ```mlir
   htile.store %norm_f16_local, %out[%c0, %h, %c0, %c0]
       : tensor<128x64xf16, #local>, memref<1x4x128x64xf16>
   ```

1. Lower the outer parallel grid to the backend translator's expected kernel
   launch form. The current handwritten backend examples use `gpu.launch`,
   while the scheduled L1 form uses `scf.forall`.

Boundary-only bufferization does not by itself complete this conversion. It may
leave a memref return bridged from a tensor result, and the final
`tensor.parallel_insert_slice` remains in the tensor body. The output argument
and `htile.store` rewrite are still HTile/backend ABI work.

## Pattern Preconditions

The initial translator can be narrow and structural. It should require:

- static tile sizes,
- unit strides on extracted slices,
- no boundary masks in the global-attention case,
- projected-permutation indexing maps for elementwise and reduction ops,
- contraction maps matching the two attention dot patterns,
- one reduction dimension for `htile.reduce`,
- loop-carried state that is already explicit in the L1 IR.

Failing loudly on unsupported shapes is better than silently producing a
backend-looking program with changed semantics.

## Challenges

### Rank And Offset Mapping

L1 tile tensors may include logical singleton dimensions that are useful for
generic linalg transformations but not useful for HTile backend code. The
translator must keep a consistent map from source tensor dimensions to HTile
tile dimensions so that load offsets, reduction axes, broadcast dimensions, and
result insertion all agree.

### View Normalization

The input IR may contain intermediate `tensor.extract_slice` values that are
not real memory operations. HTile should usually load the final tile directly
from the original source or from a canonical single view of that source.

MLIR bufferization can help expose these relationships because
`tensor.extract_slice` bufferizes to `memref.subview`, and memref alias folding
can compose nested subviews. It does not remove the problem by itself:
the downstream HTile/Triton path must either see a base memory object plus
explicit offsets, or explicitly understand subview layout metadata. Feeding
arbitrary subview results to `htile.load` without honoring their offsets and
strides would be incorrect.

For the tensor path, implement view normalization as a narrow
`tensor.extract_slice` chain resolver:

1. Start from the tile value that will become an `htile.load`.
1. Walk through defining `tensor.extract_slice` ops until reaching a supported
   root, usually a function argument or output tensor.
1. For each slice, use `OffsetSizeAndStrideOpInterface` to read mixed
   offsets/sizes/strides, and use `ExtractSliceOp::computeRankReductionMask()`
   to map rank-reduced result dimensions back to source dimensions.
1. Compose each child offset into the corresponding root dimension:
   `new_offset = parent_offset + child_offset * parent_stride`.
   For the first implementation, require unit strides; then this reduces to
   simple affine addition. Keep sizes from the final tile.
1. Materialize composed dynamic offsets with folded affine/index arithmetic
   such as `affine::makeComposedFoldedAffineApply`, and create one `htile.load`
   from the root with those offsets.
1. Fail if the chain contains non-unit strides, dynamic rank ambiguity,
   non-slice producers, or a root that is not a valid memory boundary.

This mirrors the memref subview composition algorithm conceptually, but should
operate before full bufferization so that the rest of the scheduled tile body
stays in value-based tensor form.

### Recognizing Accumulating Dot

The output accumulation in online softmax is often expressed as:

```text
acc_next = acc_scaled + p_tile @ v_tile
```

HTile can represent this more directly as an accumulating `htile.dot`. The
translator must prove that the add combines exactly the dot result and the
scaled accumulator, and that no other user requires the standalone dot result.

### Broadcast Semantics

MLIR tensor arithmetic requires equal operand types, so L1 materializes row
broadcasts explicitly. Triton treats row vectors and panels more flexibly. For
now, keeping `linalg.broadcast` is the lowest-risk representation, but a later
HTile-level broadcast/unsqueeze convention may make backend translation cleaner.

### Tensor-Return ABI Versus Kernel ABI

The scheduled L1 examples return tensors. Triton kernels naturally take pointer
arguments and store output tiles. A translator that preserves tensor-return IR
will need `htile.copy` plus `tensor.parallel_insert_slice`; a backend-facing
translator should instead introduce or target an explicit output argument and
emit `htile.store`.

### Numerical Policy

The current scheduled attention keeps softmax probabilities in `f32` for the
`P @ V` dot. Some backend examples truncate probabilities to `f16` before the
dot. That is a numerical policy choice, not a mechanical HTile translation
requirement. The translator should preserve the L1 element types unless a
separate lowering policy explicitly changes them.

### Backend Constraints

The HTile IR may allow combinations that a backend cannot lower efficiently or
at all, such as `f32 x f16` dot inputs. The L1-to-HTile translator should
preserve semantics first; backend-specific legalization can later insert casts,
layout conversions, or choose different dot lowering strategies.
