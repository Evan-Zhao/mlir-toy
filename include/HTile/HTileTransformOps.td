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
