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

def HTileSemanticToKernelAbiOp : Op<Transform_Dialect, "htile.semantic_to_kernel_abi",
                                    [TransformOpInterface, TransformEachOpTrait,
                                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Rewrite Semantic HTile function ABI toward Kernel HTile ABI";

  let description = [{
    Rewrites the targeted `func.func` boundary from tensor-return Semantic HTile
    form toward the memref-facing Kernel HTile ABI.

    This first step only rewrites the function ABI:

    - ranked tensor function arguments become memref arguments,
    - ranked tensor function results become trailing memref output arguments,
    - tensor uses of converted input arguments are preserved through
      `bufferization.to_tensor` bridge operations,
    - `func.return` operations are rewritten to return no operands.
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
