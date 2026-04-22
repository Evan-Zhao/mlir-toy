// Neptune HTile IR operators.
// Some operators may have semantic alikes in other dialects, but may be redefined here anyways
// for clarity.
// Tensor shape, placement (e.g. SMEM vs. registers), and layout are represented on the result tensor,
// not as arguments to the op.

include "HTile/HTileDialect.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def HTile_HelloOp : Op<HTile_Dialect, "hello"> {
  let summary = "trivial hello operation";
  let description = [{
    A zero-operand, zero-result operation used to demonstrate a custom op.
  }];

  let assemblyFormat = "attr-dict";
}

def HTile_DotOp : Op<HTile_Dialect, "dot"> {
  let summary = "matrix multiply over tiles";
  let description = [{
    Computes a dot product / matrix multiply between two tile values,
    optionally accumulating into an existing accumulator tile.
    The result element type encodes the output dtype.

    `transpose_a` and `transpose_b` logically transpose the corresponding
    operand before the multiply, without requiring a separate `htile.permute`.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    Optional<AnyRankedTensor>:$acc,
    UnitAttr:$transpose_a,
    UnitAttr:$transpose_b
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
    "$lhs `,` $rhs (`,` $acc^)? attr-dict `:` "
    "type($lhs) `,` type($rhs) (`,` type($acc)^)? `->` type($result)";
}

def HTile_ReduceOp : Op<HTile_Dialect, "reduce"> {
  let summary = "reduce one dimension of a tile";
  let description = [{
    Reduces one dimension of a tile using a named reduction combiner such as
    `sum`, `max`, or `min`.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$axis,
    StrAttr:$kind
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
    "$input `axis` $axis `kind` $kind attr-dict `:` type($input) `->` type($result)";
}

def HTile_FullOp : Op<HTile_Dialect, "full"> {
  let summary = "create a tile filled with a scalar value";
  let description = [{
    Creates a ranked tensor tile where every element is initialized to the scalar operand.
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def HTile_ArangeOp : Op<HTile_Dialect, "arange"> {
  let summary = "create a one-dimensional integer tile range";
  let description = [{
    Creates a one-dimensional tile containing values in the half-open interval
    [`start`, `end`) with step size one.
  }];

  let arguments = (ins Index:$start, Index:$end);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$start `to` $end attr-dict `:` type($result)";
}

def HTile_UnsqueezeOp : Op<HTile_Dialect, "unsqueeze"> {
  let summary = "insert unit dimensions into a tile";
  let description = [{
    Inserts (multiple) dimensions of size one into a tile.
    The `mask` attribute is a boolean array that describes where to insert the dimensions:
    each entry describes one output dimension, where `true` means the dimension is inserted
    and `false` means the dimension is backed by the next input dimension.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    DenseBoolArrayAttr:$mask
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
    "$input `mask` $mask attr-dict `:` type($input) `->` type($result)";
}

def HTile_PermuteOp : Op<HTile_Dialect, "permute"> {
  let summary = "permute tile dimensions";
  let description = [{
    Permutes tile dimensions according to the `permutation` attribute.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    DenseI64ArrayAttr:$permutation
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
    "$input `permutation` $permutation attr-dict `:` type($input) `->` type($result)";
}

def HTile_GatherNdOp : Op<HTile_Dialect, "gather_nd"> {
  let summary = "gather a tile using tile-shaped index tensors";
  let description = [{
    Gathers elements from a source using tile-shaped index tensors.

    The source may be a memory object, a tensor tile, or a lower-level pointer
    value depending on the lowering stage. All index tensors are expected to
    have the result tile shape.
  }];

  let arguments = (ins
    AnyType:$source,
    Variadic<AnyRankedTensor>:$indices
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
    "$source `[` $indices `]` attr-dict `:` type($source) `,` "
    "type($indices) `->` type($result)";
}

def HTile_LoadOp : Op<HTile_Dialect, "load"> {
  let summary = "load a tile from memory";
  let description = [{
    Loads a tile from a memory-like source at the given scalar offsets. This
    makes explicit the tile read semantics that Neptune currently represents
    with TIR BufferRegion reads.

    Attributes can carry backend lowering hints such as `boundary_check`,
    `cache_modifier`, `eviction_policy`, or TMA/block-pointer metadata.
  }];

  let arguments = (ins
    AnyType:$source,
    Variadic<Index>:$offsets
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
    "$source `[` $offsets `]` attr-dict `:` type($source) `->` type($result)";
}

def HTile_CopyOp : Op<HTile_Dialect, "copy"> {
  let summary = "copy a tile value to a different tile placement";
  let description = [{
    Materializes a tile value as another tile value, usually with a different
    HTile tensor encoding. This models synchronous movement between tile
    placements such as shared memory and local/register fragments.

    This op is not a global memory access. Use `htile.load` and `htile.store`
    for memory-like sources and destinations.
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
    "$input attr-dict `:` type($input) `->` type($result)";
}

def HTile_StoreOp : Op<HTile_Dialect, "store"> {
  let summary = "store a tile to memory";
  let description = [{
    Stores a tile to a memory-like destination at the given scalar offsets. This
    makes explicit the tile write semantics that Neptune currently represents
    with TIR BufferRegion stores.

    Attributes can carry backend lowering hints such as `boundary_check`,
    `cache_modifier`, `eviction_policy`, or TMA/block-pointer metadata.
  }];

  let arguments = (ins
    AnyRankedTensor:$value,
    AnyType:$dest,
    Variadic<Index>:$offsets
  );

  let assemblyFormat =
    "$value `,` $dest `[` $offsets `]` attr-dict `:` type($value) `,` type($dest)";
}
