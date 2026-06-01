include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TAExchangeDivAndMatmulOp
    : Op<Transform_Dialect, "ta.exchange_div_and_matmul",
         [FunctionalStyleTransformOpTrait, TransformOpInterface, TransformEachOpTrait,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Exchange an invariant division with a ta.map_reduce matmul";

  let description = [{
    Rewrites an add-reduction whose payload multiplies one operand by a
    reduction-invariant division:

    ```
      sum((num / den) * rhs)  ->  sum(num * rhs) / den
    ```

    The target must be a `ta.map_reduce <add>` whose yielded value is a
    `ta.mulf`. The selected multiply operand must be produced by `ta.divf`.
    The divisor expression must not depend on any axis reduced by the
    `ta.map_reduce`.

    `operand_number` selects which multiply operand is expected to be the
    division. It defaults to 0, matching the usual `(softmax_probs @ V)` shape.
    The original map-reduction is replaced by the trailing `ta.divf`. The op
    returns handles to the rebuilt reduction and the trailing division.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
      DefaultValuedOptionalAttr<I64Attr, "0">:$operand_number);
  let results = (outs TransformHandleTypeInterface:$reduction,
      TransformHandleTypeInterface:$division);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::ta::MapReduceOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
