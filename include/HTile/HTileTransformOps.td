include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
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
