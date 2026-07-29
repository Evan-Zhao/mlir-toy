include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HTileLinalgToSemanticOp : Op<Transform_Dialect, "htile.linalg_to_semantic",
                                 [TransformOpInterface, TransformEachOpTrait,
                                  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                  ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Translate supported tensor Linalg ops to semantic HTile ops";

  let description = [{
    Rewrites supported `linalg` operations nested in the target payload into
    target-independent semantic HTile operations:

    - contraction-shaped `linalg.generic` ops become `htile.dot`,
    - simple row reductions become `htile.reduce` plus the original init combiner,
    - all-parallel elementwise `linalg.generic` ops are opened into tensor
      `arith`/`math` operations and semantic broadcasts,
    - `linalg.fill` becomes `htile.full`.

    The transform emits a silenceable failure if any Linalg op originally
    present in the target cannot be converted.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def HTileFuseOOBSinkScatterAsMaskedInsertSliceOp
    : Op<Transform_Dialect, "htile.fuse_oob_sink_scatter_as_masked_insert_slice",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Fuse an OOB-sink scatter as a masked forall publication";

  let description = [{
    Fuses a specific overwrite `stablehlo.scatter` into the `scf.forall` that
    produces its updates. This transform recognizes the convention where a
    predicate selects either a contiguous destination row or an index that is
    guaranteed to be out of bounds:

    ```mlir
    %valid_rows = linalg.generic ... {
      %i = linalg.index RANGE_DIM
      %i_int = arith.index_cast %i
      %row = arith.addi %base, %i_int
      linalg.yield %row
    }
    %selected_rows = linalg.generic
        ins(%mask, %valid_rows, %sink_rows : ...) ... {
    ^bb0(%valid: i1, %row: i32, %sink: i32, ...):
      %selected = arith.select %valid, %row, %sink
      linalg.yield %selected
    }
    %indices = linalg.broadcast ins(%selected_rows : tensor<BxTxi32>) ...
        dimensions = [2]  // tensor<BxTxi32> -> tensor<BxTx1xi32>
    %result = "stablehlo.scatter"(%init, %indices, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
          update_window_dims = [2, 3], inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0], index_vector_dim = 2>,
      unique_indices = true
    }> ...
    ```

    The final extent-one dimension is StableHLO's one-component index vector.
    The transform strips that shape adaptation and matches the elementwise
    `select` exactly. The true branch must be `base + linalg.index RANGE_DIM`;
    the indexing map for `base` must be invariant in `RANGE_DIM`.

    The false branch is not assumed to be out of bounds merely because it is
    named a sink. The transform conservatively infers the signed element range
    of `%sink_rows` and requires its lower bound to be at least the static
    extent of destination dimension zero. For example, with a destination
    extent of 1024, sink rows computed as
    `1024 + document_id * 512 + token_id` have lower bound 1024. StableHLO drops
    these updates, so the original `%mask` can replace the sink convention.
    If the range cannot be proven, the transform fails without treating the
    false branch as a mask.

    The selected `forall` must directly produce `%updates` through one
    `tensor.parallel_insert_slice`. Its publication descriptor tiles
    `%valid_rows` and `%mask`. The first tiled valid row becomes the destination
    row offset, non-range scatter dimensions must be unit tile dimensions, and
    the mask is reshaped and broadcast to the published source shape. The
    resulting publication is equivalent to:

    ```mlir
    %result = scf.forall ... shared_outs(..., %dest = %init) {
      %source = ... : tensor<128x1x64xf16>
      %mask_tile = ... : tensor<128x1x64xi1>
      %row = ... : index
      scf.forall.in_parallel {
        htile.masked_parallel_insert_slice
            %source into %dest[%row, %head, 0] [128, 1, 64] [1, 1, 1]
            mask(%mask_tile : tensor<128x1x64xi1>)
            : tensor<128x1x64xf16> into tensor<1024x4x64xf16>
      }
    }
    ```

    The current implementation supports one indirect destination dimension,
    mapped to dimension zero, and does not support scatter batching or a
    permutation between the surviving update dimensions and destination
    dimensions. The scatter must have one input, update, and result; an
    overwrite update computation; and `unique_indices = true`.

    The `forall` operand handle is updated to refer to the rebuilt loop, and the
    transform returns a handle to the masked insertion.
  }];

  let arguments = (ins TransformHandleTypeInterface:$scatter,
                       TransformHandleTypeInterface:$forall);
  let results = (outs TransformHandleTypeInterface:$masked_insert_slice);

  let assemblyFormat = "$scatter `into` $forall attr-dict `:` "
                       "functional-type(operands, results)";
}

def HTileFuseRangedGatherIntoLoopsOp
    : Op<Transform_Dialect, "htile.fuse_ranged_gather_into_loops",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Fuse ranged StableHLO gathers into selected loop nests as masked HTile loads";

  let description = [{
    Convert a ranged stablehlo.gather into a masked htile.load rectangular load
    and fuse under the given loop nest. Before:
    ```mlir
    %padded = stablehlo.pad %source, %zero,
        low = [0], high = [128], interior = [0]
        : (tensor<1024xf16>, tensor<f16>) -> tensor<1152xf16>
    %gather = "stablehlo.gather"(%padded, %starts) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1], start_index_map = [0], index_vector_dim = 1>,
      slice_sizes = array<i64: 128>
    }> : (tensor<1152xf16>, tensor<8x1xi32>) -> tensor<8x128xf16>

    scf.for %j = ... {
      %tile = tensor.extract_slice %gather[%batch, %j] [1, 64] [1, 1]
          : tensor<8x128xf16> to tensor<1x64xf16>
      // use %tile
    }
    ```

    After:
    ```mlir
    scf.for %j = ... {
      %c0 = arith.constant 0 : index
      %start_i32 = tensor.extract %starts[%batch, %c0] : tensor<8x1xi32>
      %start = arith.index_cast %start_i32 : i32 to index
      %offset = arith.addi %start, %j : index
      %mask = linalg.generic ...  // 0 <= %offset + linalg.index 0 < dim(%source, 0)
      %other = tensor.extract %zero[] : tensor<f16>
      %loaded = htile.load %source[%offset]
          mask(%mask : tensor<64xi1>) other(%other : f16)
          : tensor<1024xf16> -> tensor<64xf16>
      %tile = tensor.expand_shape %loaded [[0, 1]] output_shape [1, 64]
          : tensor<64xf16> into tensor<1x64xf16>
      // use %tile
    }
    ```

    The padding is an anti-clamping contract. For each indexed operand
    dimension `d`, the input must satisfy:

    ```text
    start_indices[d] >= 0
    edge_padding_low[d] == 0
    edge_padding_high[d] >= slice_sizes[d]
    interior_padding[*] == 0
    ```

    For example, a source extent of 1024 with a slice size and high padding of
    128 has a StableHLO maximum start of 1024. A start above 1024 clamps to an
    all-padding window; the masked source load also produces only `other`.

    Since stablehlo.gather can change the layout of the tensor (but htile.load cannot),
    this transform may also create `tensor.expand_shape` and `tensor.collapse_shape`.
    stablehlo.gather ops that imply permutation are unsupported.

    The selected loops are not rebuilt or moved. The gather handle is consumed,
    the loop handle remains valid, and the result identifies the created loads.
  }];

  let arguments = (ins TransformHandleTypeInterface:$gathers,
                       TransformHandleTypeInterface:$loops);
  let results = (outs TransformHandleTypeInterface:$loads);

  let assemblyFormat = "$gathers `into` $loops attr-dict `:` "
                       "functional-type(operands, results)";
}

def HTileOutlineKernelsOp : Op<Transform_Dialect, "htile.outline_kernels",
                               [FunctionalStyleTransformOpTrait,
                                DeclareOpInterfaceMethods<TransformOpInterface>,
                                DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Outline selected HTile loop nests to kernels";

  let description = [{
    Outlines an ordered payload handle of top-level `scf.forall` loop nests into
    `htile.kernel` definitions and replaces the original loops with result-free
    `htile.launch_func` operations.

    Because `htile.launch_func` has no data results, this transform is also
    responsible for materializing explicit boundary buffers for tensor values
    that cross selected loop boundaries.
  }];

  let arguments = (ins TransformHandleTypeInterface:$foralls,
                       OptionalAttr<StrArrayAttr>:$kernel_names);
  let results = (outs TransformHandleTypeInterface:$launches,
                      TransformHandleTypeInterface:$kernels);

  let assemblyFormat = "$foralls attr-dict `:` functional-type(operands, results)";
}
