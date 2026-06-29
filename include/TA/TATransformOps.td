include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TAMatchEinsumOp
    : Op<Transform_Dialect,
         "match.ta.einsum", [SingleOpMatcher, MatchOpInterface, MemoryEffectsOpInterface]> {
  let summary = "Match a TA expression against a small einsum equation";

  let description = [{
    Matches a `ta.reduce <add>` whose payload is a `ta.mulf` against a
    whitespace-separated two-input einsum equation. For example:

    ```
      transform.match.ta.einsum %candidate
          {equation = "b h i d, b h j d -> b h i j"}
          : (!transform.any_op) -> !transform.any_op
    ```

    The operation succeeds on the matched `ta.reduce` and returns the same
    payload handle. It emits a silenceable failure when the target is not an
    add-reduced multiply with operand and result axis structure matching the equation.

    Axis names in the equation are pattern symbols: they do not have
    to match the concrete TA axis names, but repeated symbols must map to the same concrete axis.
  }];

  let arguments = (ins TransformHandleTypeInterface:$operand_handle, StrAttr:$equation);
  let results = (outs TransformHandleTypeInterface:$matched);

  let assemblyFormat = "$operand_handle attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;
}

def TAToLinalgOp : Op<Transform_Dialect, "ta.to_linalg",
                      [TransformOpInterface, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                       ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Lower TA scopes to linalg while best-effort preserving handles";

  let description = [{
    Lowers `ta.scope` operations nested in the target payload to `linalg.generic`
    materializations. Any existing transform handles to TA expression roots that
    have a clear lowered linalg op are updated in-place. Handles to internal TA
    ops that lower only to linalg region scalar ops or indexing maps are not
    guaranteed to survive.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TAExchangeDivAndMatmulPatternsOp
    : Op<Transform_Dialect, "apply_patterns.ta.exchange_div_and_matmul",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect TA div-matmul exchange rewrite patterns";

  let description = [{
    Collects PDLL-authored TA rewrite patterns that exchange a division with
    an add reduction:

    ```
      sum((num / den) * rhs)  ->  sum(num * rhs) / den
    ```

    The divisor must have axes disjoint from the reduction axes. The pattern
    preserves the matched add reducer kind, axes, and import-group attributes.
  }];

  let assemblyFormat = "attr-dict";
}

def TAExpToExp2PatternsOp : Op<Transform_Dialect, "apply_patterns.ta.exp_to_exp2",
                               [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect TA algebraic rewrite patterns for exp-to-exp2";

  let description = [{
    Collects PDLL-authored TA patterns for the rewrite:

    ```
      exp(x) -> exp2(log2(e) * x)
    ```

    The pattern set also contains the minimal scalar algebra needed to push the
    new positive scalar factor through a softmax-like expression graph:
    distribution through `ta.subf` and `ta.select`, movement through
    `ta.reduce <max>`, reassociation through `ta.mulf`, and the `ta.mulf`
    canonicalization that folds scalar `ta.constant` values.
  }];

  let assemblyFormat = "attr-dict";
}

def TARewriteExpToExp2Op : Op<Transform_Dialect, "ta.rewrite_exp_to_exp2",
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
