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

    The special token `...` matches a variadic axis sequence and may appear at
    most once per input/result axis list. All `...` occurrences in an equation
    must match the same number of axes and the same concrete axes. For example,
    `b ... h i d, b h j d -> b ... h i j` matches both
    `b h i d, b h j d -> b h i j` and
    `b g h i d, b h j d -> b g h i j`.
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

def TASinkDivAfterMatmulPatternsOp
    : Op<Transform_Dialect, "apply_patterns.ta.sink_div_after_matmul",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect TA div-after-matmul sinking rewrite patterns";

  let description = [{
    Collects PDLL-authored TA rewrite patterns that sink a division after a
    multiplication and add reduction:

    ```
      sum((num / denom) * rhs) -> sum(num * rhs) / denom
      sum(f32(f16(num / denom)) * rhs)
        -> sum(f32(f16(num)) * rhs) / denom
    ```

    The cast-aware form supports f16 matmul operands with f32 accumulation. The
    `denom` divisor must have axes disjoint from the reduction axes. The pattern
    preserves the matched add reducer kind, axes, and import-group attributes.
  }];

  let assemblyFormat = "attr-dict";
}

def TASinkRightMulAfterMatmulPatternsOp
    : Op<Transform_Dialect, "apply_patterns.ta.sink_right_mul_after_matmul",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect rewrite patterns to sink an RHS multiplicative factor after a matmul";

  let description = [{
    Collects PDLL-authored TA rewrite patterns that sink a multiplicative factor
    after an add-reduced multiply. The pattern set may first reassociate a
    multiply when the right-associated multiply has a smaller axis footprint:

    ```
      c * (a * b) -> a * (b * c)
      sum(lhs * (rhs * factor)) -> sum(lhs * rhs) * factor
    ```

    The `factor` must have axes disjoint from the reduction axes.
    The pattern preserves the matched add reducer kind, axes, and import-group attributes.
  }];

  let assemblyFormat = "attr-dict";
}

def TAReassociateRightMulfPatternsOp
    : Op<Transform_Dialect, "apply_patterns.ta.reassociate_right_mulf",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect rewrite patterns to reassociate TA multiplications to the right";

  let description = [{
    Collects PDLL-authored TA rewrite patterns that reassociate a nested multiplication
    ```
      c * (a * b) -> a * (b * c)
    ```
    if the right-associated multiply involves fewer computed elements.
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
    distribution through `ta.sub` and `ta.select`, movement through
    `ta.reduce <max>`, reassociation through `ta.mul`, and canonicalizations
    that fold scalar products and floating-point infinity identities.
  }];

  let assemblyFormat = "attr-dict";
}
