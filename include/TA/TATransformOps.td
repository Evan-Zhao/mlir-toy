include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TAExchangeDivAndMatmulPatternsOp
    : Op<Transform_Dialect, "apply_patterns.ta.exchange_div_and_matmul",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect TA exchange-div-and-matmul rewrite patterns";

  let description = [{
    Collects PDLL-authored TA rewrite patterns that exchange a division with a
    reduction:

    ```
      sum((num / den) * rhs)  ->  sum(num * rhs) / den
    ```

    This is intentionally narrow: it matches a `ta.reduce` marked with a
    `target` unit attribute whose input is a `ta.mulf` with a `ta.divf` on its
    left-hand side. The pattern preserves the matched reducer kind, axes, and
    result type.
  }];

  let assemblyFormat = "attr-dict";
}

def TAExpToExp2PatternsOp
    : Op<Transform_Dialect, "apply_patterns.ta.exp_to_exp2",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect TA algebraic rewrite patterns for exp-to-exp2";

  let description = [{
    Collects PDLL-authored TA patterns for the rewrite:

    ```
      exp(x) -> exp2(log2(e) * x)
    ```

    The pattern set also contains the minimal scalar algebra needed to push the
    new positive scalar factor through a softmax-like expression graph:
    distribution through `ta.subf`, movement through `ta.reduce <max>`,
    reassociation through `ta.mulf`, and multiplication folding for scalar
    `ta.constant` values.
  }];

  let assemblyFormat = "attr-dict";
}

def TARewriteExpToExp2Op
    : Op<Transform_Dialect, "ta.rewrite_exp_to_exp2",
         [TransformOpInterface, TransformEachOpTrait,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Greedily apply TA exp-to-exp2 algebraic rewrites";

  let description = [{
    Applies the `apply_patterns.ta.exp_to_exp2` pattern set greedily to the
    targeted payload operation until convergence.
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
