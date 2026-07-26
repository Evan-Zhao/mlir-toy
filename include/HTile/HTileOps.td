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

def HTile_LoadOp : Op<HTile_Dialect, "load"> {
  let summary = "load a tile from memory";
  let description = [{
    Loads a tile from a memory-like source at the given scalar offsets. This
    makes explicit the tile read semantics that Neptune currently represents
    with TIR BufferRegion reads.

    Attributes can carry backend lowering hints such as `boundary_check`,
    `cache_modifier`, `eviction_policy`, or TMA/block-pointer metadata.
  }];

  let arguments = (ins AnyType:$source, Variadic<Index>:$offsets);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$source `[` $offsets `]` attr-dict `:` type($source) `->` type($result)";
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

def HTile_ParallelScatterOp
    : Op<HTile_Dialect, "parallel_scatter", [
        DeclareOpInterfaceMethods<ParallelCombiningOpInterface,
                                  ["getUpdatedDestinations", "getIteratingParent"]>
      ]> {
  let summary = "scatter a tensor contribution into an scf.forall result";
  let description = [{
    Contributes `source` to the shared `dest` tensor of an enclosing
    `scf.forall`. The operation must be directly nested in the forall's
    `scf.forall.in_parallel` terminator.

    There is one entry in `indices` per destination dimension. A ranked tensor
    entry supplies indirect absolute coordinates and broadcasts over the
    leading batch dimensions of `source`. A scalar entry supplies one absolute
    coordinate. For dimensions listed in `broadcast_dims`, the corresponding
    scalar is instead a base offset and the matching trailing source coordinate
    is added to it. For example, indexing `tensor<NxHxF>` with
    `[%rows, %head, %c0]` and `broadcast_dims([2])` has the semantics of the
    NumPy expression `dest[rows, head, 0:F]`.

    `unique` promises that the destination elements written by parallel
    contributions do not overlap. Violating this promise is undefined
    behavior. `out_of_bounds <discard>` specifies that a source element whose
    destination coordinate is out of bounds does not update the destination.
  }];

  let arguments = (ins AnyRankedTensor:$source,
                       AnyRankedTensor:$dest,
                       Variadic<AnyType>:$indices,
                       DenseI64ArrayAttr:$broadcast_dims,
                       UnitAttr:$unique,
                       HTile_ScatterOutOfBoundsAttr:$out_of_bounds);

  let assemblyFormat = [{
    $source `into` $dest `[` $indices `]`
      `broadcast_dims` `(` $broadcast_dims `)`
      (`unique` $unique^)?
      `out_of_bounds` $out_of_bounds
      attr-dict `:` `(` type($source) `,` type($indices) `)`
      `->` `(` type($dest) `)`
  }];

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
