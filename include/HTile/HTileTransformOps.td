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

def HTileFusePackedWindowInsertOp
    : Op<Transform_Dialect, "htile.fuse_packed_window_insert",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Fuse packed-window insertion as a masked forall publication";

  let description = [{
    Fuses a `stablehlo.custom_call @neptune.packed_window_insert` into the
    `scf.forall` that produces its window tensor. For a window tile beginning at
    document `d` and token `t`, the destination row and mask are:

    ```text
    row = starts[d] + t
    mask[i] = t + i < lengths[d]
    ```

    The transform appends the custom call's destination tensor to the forall
    outputs and publishes the tiled source with
    `htile.masked_parallel_insert_slice`. Unwritten destination elements remain
    unchanged.

    The `forall` operand handle is updated to refer to the rebuilt loop, and the
    result identifies the masked insertion.
  }];

  let arguments = (ins TransformHandleTypeInterface:$insert,
                       TransformHandleTypeInterface:$forall);
  let results = (outs TransformHandleTypeInterface:$masked_insert_slice);

  let assemblyFormat = "$insert `into` $forall attr-dict `:` "
                       "functional-type(operands, results)";
}

def HTileFusePackedWindowExtractOp
    : Op<Transform_Dialect, "htile.fuse_packed_window_extract",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Fuse packed-window extraction into loops as masked HTile loads";

  let description = [{
    Replaces in-loop tiles of
    `stablehlo.custom_call @neptune.packed_window_extract` with masked
    `htile.load` operations. For a result tile beginning at document `d` and
    token `t`, the packed source row and mask are:

    ```text
    row = starts[d] + t
    mask[i] = t + i < lengths[d]
    ```

    Invalid lanes produce the custom call's scalar `other` value without
    accessing the packed source. The selected loops are not rebuilt or moved.
    The extract handle is consumed, the loop handle remains valid, and the
    result identifies the created loads.
  }];

  let arguments = (ins TransformHandleTypeInterface:$extracts,
                       TransformHandleTypeInterface:$loops);
  let results = (outs TransformHandleTypeInterface:$loads);

  let assemblyFormat = "$extracts `into` $loops attr-dict `:` "
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
