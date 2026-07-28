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

def HTileFuseScatterIntoForallOp
    : Op<Transform_Dialect, "htile.fuse_scatter_into_forall",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Fuse a unique StableHLO scatter into its update-producing forall";

  let description = [{
    Fuses the `stablehlo.scatter` identified by `scatter` into the `scf.forall`
    identified by `forall`. The forall must produce the scatter's updates.

    The transform rebuilds the forall over the scatter destination and replaces
    the update-tensor publication with an `htile.parallel_scatter` contribution.
    The `forall` operand handle is updated to refer to the rebuilt loop, and the
    transform returns a handle to the parallel scatter.
  }];

  let arguments = (ins TransformHandleTypeInterface:$scatter,
                       TransformHandleTypeInterface:$forall);
  let results = (outs TransformHandleTypeInterface:$parallel_scatter);

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
