include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def StablehloLegalizeControlFlowOp
    : Op<Transform_Dialect, "stablehlo.legalize_control_flow",
         [TransformOpInterface, TransformEachOpTrait,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Legalize StableHLO control flow to SCF control flow";

  let description = [{
    Rewrites `stablehlo.while`, `stablehlo.if`, and `stablehlo.case` nested
    under the target to SCF control flow. Counted while loops matching the JAX
    `fori_loop` form become `scf.for`; other while loops become `scf.while`.

    ```mlir
    transform.stablehlo.legalize_control_flow %func : !transform.any_op
    ```
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

def StablehloSliceToTensorConversionPatternsOp
    : Op<Transform_Dialect, "apply_conversion_patterns.stablehlo.slice_to_tensor",
         [DeclareOpInterfaceMethods<ConversionPatternDescriptorOpInterface,
                                    ["getTypeConverter"]>]> {
  let summary = "Collect the StableHLO slice-to-tensor conversion pattern";

  let description = [{
    Makes StableHLO's static slice conversion available to
    `transform.apply_conversion_patterns`. The conversion replaces
    `stablehlo.slice` with `tensor.extract_slice` without running the complete
    StableHLO-to-Linalg conversion pipeline.

    ```
    transform.apply_conversion_patterns to %func {
      transform.apply_conversion_patterns.stablehlo.slice_to_tensor
    } {illegal_ops = ["stablehlo.slice"], partial_conversion, preserve_handles}
      : !transform.any_op
    ```
  }];

  let assemblyFormat = "attr-dict";
}
