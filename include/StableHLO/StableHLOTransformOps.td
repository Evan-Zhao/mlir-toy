include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"

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
