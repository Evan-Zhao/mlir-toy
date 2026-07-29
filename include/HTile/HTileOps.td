// Neptune HTile IR operators.
// Some operators may have semantic alikes in other dialects, but may be redefined here anyways
// for clarity.
// Tensor shape, placement (e.g. SMEM vs. registers), and layout are represented on the result
// tensor, not as arguments to the op.

include "HTile/HTileAttrs.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def HTile_WarpPolicyAttr
    : Attr<And<[StrAttr.predicate,
                CPred<"([](::mlir::Attribute attr) { auto value = "
                      "::llvm::cast<::mlir::StringAttr>(attr).getValue(); return value == "
                      "\"square\" || value == \"full_row\" || value == \"full_col\"; })($_self)">]>,
           "TileLang GEMM warp policy string attribute"> {
  let storageType = "::mlir::StringAttr";
  let returnType = "::mlir::StringAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def HTile_HelloOp : Op<HTile_Dialect, "hello"> {
  let summary = "trivial hello operation";
  let description = [{
    A zero-operand, zero-result operation used to demonstrate a custom op.
  }];

  let assemblyFormat = "attr-dict";
}

def HTile_KernelOp
    : Op<HTile_Dialect, "kernel",
         [IsolatedFromAbove, SingleBlock, SingleBlockImplicitTerminator<"ReturnOp">, Symbol]> {
  let summary = "HTile device kernel";
  let description = [{
    `htile.kernel` is a symbol-like device-code container. Values used by the
    kernel body must be explicit entry block arguments; the region is isolated
    from above and terminates with `htile.return`.

    `program_bounds` records the generic static launch-domain extent for each
    `htile.program_id` dimension used by the kernel body.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, OptionalAttr<DenseI64ArrayAttr>:$program_bounds);
  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<(ins "::llvm::StringRef":$symName), [{
      build($_builder, $_state, $_builder.getStringAttr(symName),
            /*programBounds=*/nullptr);
    }]>];

  let hasCustomAssemblyFormat = 1;
}

def HTile_ProgramIdOp : Op<HTile_Dialect, "program_id", [Pure, HasParent<"KernelOp">]> {
  let summary = "get the current HTile program id";
  let description = [{
    Returns the id of the current logical program instance along a generic
    launch-domain dimension. The dimension is target-independent and may later
    be mapped to GPU blocks, CPU parallel workers, or another backend schedule.
  }];

  let arguments = (ins I64Attr:$dimension);
  let results = (outs Index:$result);

  let assemblyFormat = "$dimension attr-dict";
}

def HTile_LaunchFuncOp
    : Op<HTile_Dialect, "launch_func", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "launch an HTile device kernel";
  let description = [{
    `htile.launch_func` is a host-side launch operation that references an
    `htile.kernel` symbol and passes explicit launch operands. Thrust 1 keeps
    launches result-free; inter-kernel values must flow through explicit buffer
    operands.
  }];

  let arguments = (ins FlatSymbolRefAttr:$kernel, Variadic<AnyType>:$operands);

  let assemblyFormat = "$kernel `(` $operands `)` attr-dict `:` type($operands)";
}

def HTile_ReturnOp
    : Op<HTile_Dialect, "return", [Pure, HasParent<"KernelOp">, ReturnLike, Terminator]> {
  let summary = "return from an HTile kernel";
  let description = [{
    Terminates an `htile.kernel` body. Kernel results are intentionally not
    supported in the minimal launch ABI.
  }];

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
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

    `warp_policy` is an optional backend lowering hint for partitioning GEMM
    work across warps. TileLang currently recognizes `square`, `full_row`, and
    `full_col`.
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs, Optional<AnyRankedTensor>:$acc,
      UnitAttr:$transpose_a, UnitAttr:$transpose_b,
      OptionalAttr<HTile_WarpPolicyAttr>:$warp_policy);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs (`,` $acc^)? attr-dict `:` "
                       "type($lhs) `,` type($rhs) (`,` type($acc)^)? `->` type($result)";
}

def HTile_ReduceOp : Op<HTile_Dialect, "reduce"> {
  let summary = "reduce one dimension of a tile";
  let description = [{
    Reduces one dimension of a tile using a named reduction combiner such as
    `sum`, `max`, or `min`.
  }];

  let arguments = (ins AnyRankedTensor:$input, I64Attr:$axis, StrAttr:$kind);
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

def HTile_BroadcastOp : Op<HTile_Dialect, "broadcast"> {
  let summary = "broadcast a tile by inserting dimensions";
  let description = [{
    Broadcasts the input tile to the result shape by inserting the dimensions
    listed in the `dimensions` attribute.
  }];

  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$dimensions);
  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let assemblyFormat =
      "$input `dimensions` `=` $dimensions attr-dict `:` type($input) `->` type($result)";
}

def HTile_UnsqueezeOp : Op<HTile_Dialect, "unsqueeze"> {
  let summary = "insert unit dimensions into a tile";
  let description = [{
    Inserts (multiple) dimensions of size one into a tile.
    The `mask` attribute is a boolean array that describes where to insert the dimensions:
    each entry describes one output dimension, where `true` means the dimension is inserted
    and `false` means the dimension is backed by the next input dimension.
  }];

  let arguments = (ins AnyRankedTensor:$input, DenseBoolArrayAttr:$mask);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$input `mask` $mask attr-dict `:` type($input) `->` type($result)";
}

def HTile_PermuteOp : Op<HTile_Dialect, "permute"> {
  let summary = "permute tile dimensions";
  let description = [{
    Permutes tile dimensions according to the `permutation` attribute.
  }];

  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$permutation);
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

  let arguments = (ins AnyType:$source, Variadic<AnyRankedTensor>:$indices);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$source `[` $indices `]` attr-dict `:` type($source) `,` "
                       "type($indices) `->` type($result)";
}

def HTile_LoadOp : Op<HTile_Dialect, "load", [AttrSizedOperandSegments]> {
  let summary = "load a tile from memory";
  let description = [{
    Loads a tile from a memory-like source at the given scalar offsets. This
    makes explicit the tile read semantics that Neptune currently represents
    with TIR BufferRegion reads.

    An optional `mask` and `other` pair gives the load predicated semantics.
    For every result coordinate, a true mask element loads from the source and
    a false mask element produces `other` without accessing memory. An address
    selected by a true mask element must be in bounds. The mask must have the
    result shape and an `i1` element type; `other` must be a scalar with the
    result element type. Supplying only one of the pair is invalid.

    Attributes can carry backend lowering hints such as `boundary_check`,
    `cache_modifier`, `eviction_policy`, or TMA/block-pointer metadata.
  }];

  let arguments = (ins AnyType:$source, Variadic<Index>:$offsets,
                       Optional<AnyRankedTensor>:$mask, Optional<AnyType>:$other);
  let results = (outs AnyRankedTensor:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::Value":$source,
                   "::mlir::ValueRange":$offsets), [{
      build($_builder, $_state, resultType, source, offsets,
            ::mlir::Value(), ::mlir::Value());
    }]>
  ];

  let assemblyFormat = [{
    $source `[` $offsets `]`
      (`mask` `(` $mask^ `:` type($mask) `)`
       `other` `(` $other `:` type($other) `)`)?
      attr-dict `:` type($source) `->` type($result)
  }];

  let hasVerifier = 1;
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

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def HTile_MaskedParallelInsertSliceOp
    : Op<HTile_Dialect, "masked_parallel_insert_slice", [
        AttrSizedOperandSegments,
        OffsetSizeAndStrideOpInterface,
        DeclareOpInterfaceMethods<ParallelCombiningOpInterface,
                                  ["getUpdatedDestinations", "getIteratingParent"]>
      ]> {
  let summary = "publish a masked tensor slice to an scf.forall result";
  let description = [{
    Contributes `source` to a slice of the shared `dest` tensor of an enclosing
    `scf.forall`. The operation must be directly nested in the forall's
    `scf.forall.in_parallel` terminator.

    The slice descriptor and rank-reduction behavior match
    `tensor.parallel_insert_slice`. The `mask` has the source shape and an `i1`
    element type. A true mask element publishes the corresponding source
    element. A false mask element leaves the destination unchanged and does not
    access the corresponding destination address, even when that address would
    be out of bounds. An address selected by a true mask element must be in
    bounds.

    As with `tensor.parallel_insert_slice`, conflicting parallel publications
    have undefined behavior.
  }];

  let arguments = (ins AnyRankedTensor:$source,
                       AnyRankedTensor:$dest,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       AnyRankedTensor:$mask,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);

  let assemblyFormat = [{
    $source `into` $dest ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    `mask` `(` $mask `:` type($mask) `)`
    attr-dict `:` type($source) `into` type($dest)
  }];

  let extraClassDeclaration = [{
    ::mlir::RankedTensorType getSourceType() {
      return ::llvm::cast<::mlir::RankedTensorType>(getSource().getType());
    }

    ::mlir::RankedTensorType getDestType() {
      return ::llvm::cast<::mlir::RankedTensorType>(getDest().getType());
    }

    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getDestType().getRank();
      return {rank, rank, rank};
    }

    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$source, "::mlir::Value":$dest,
      "::llvm::ArrayRef<::mlir::OpFoldResult>":$offsets,
      "::llvm::ArrayRef<::mlir::OpFoldResult>":$sizes,
      "::llvm::ArrayRef<::mlir::OpFoldResult>":$strides, "::mlir::Value":$mask,
      CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>
  ];

  let hasVerifier = 1;
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

  let arguments = (ins AnyRankedTensor:$value, AnyType:$dest, Variadic<Index>:$offsets);

  let assemblyFormat =
      "$value `,` $dest `[` $offsets `]` attr-dict `:` type($value) `,` type($dest)";
}
