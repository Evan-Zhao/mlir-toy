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
    Collects PDLL-authored TA rewrite patterns that sink a division through
    f16 or bf16 operand rounding and after a multiplication and add reduction:

    ```
      sum(f32(bf16(num / denom)) * rhs)
        -> sum(f32(bf16(num)) * rhs) / denom
    ```

    The patterns support the left- and right-operand forms of f16 and bf16
    matmuls with f32 accumulation. Moving the divisor across low-precision
    rounding is intentionally approximate. The `denom` divisor must have axes
    disjoint from the reduction axes. The patterns preserve the matched add
    reducer kind, axes, and import-group attributes.
  }];

  let assemblyFormat = "attr-dict";
}

def TASinkRightMulAfterMatmulPatternsOp
    : Op<Transform_Dialect, "apply_patterns.ta.sink_right_mul_after_matmul",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect rewrite patterns to sink an RHS multiplicative factor after a matmul";

  let description = [{
    Collects a PDLL-authored TA rewrite pattern that sinks a multiplicative
    factor through f16 or bf16 operand rounding and after an add-reduced
    multiply:

    ```
      sum(lhs * f32(bf16(rhs * factor)))
        -> sum(lhs * f32(bf16(rhs))) * factor
    ```

    The pattern supports f16 and bf16 matmul operands with f32 accumulation.
    Moving the factor across low-precision rounding is intentionally
    approximate. The `factor` must have axes disjoint from the reduction axes.
    The pattern preserves the matched add reducer kind, axes, and import-group
    attributes.
  }];

  let assemblyFormat = "attr-dict";
}

def TASinkScaleAfterMaxPatternsOp
    : Op<Transform_Dialect, "apply_patterns.ta.sink_scale_after_max",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect TA rewrites that move a positive scale after a max reduction";

  let description = [{
    Collects PDLL-authored TA patterns for the rewrite:

    ```
      select(p, c * x, -inf)           -> c * select(p, x, -inf)
      max_axes(c * x)                  -> c * max_axes(x)
      max_axes(select(p, c * x, -inf)) -> c * max_axes(select(p, x, -inf))
    ```

    The masked form is obtained by composing the first two rewrites. The axisless scale `c`
    must be a finite positive floating-point constant.
    Both operand orders of `ta.mul` are supported. The patterns preserve the
    matched max reducer kind, axes, and import-group attributes. Apply this
    descriptor separately from pattern sets that move scales into max reductions.
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

    The pattern set intentionally does not distribute through `ta.add`. It preserves
    `log2(e) * (scores + bias)` rather than producing separately scaled terms. This keeps the
    common scale visible for a later `apply_patterns.ta.sink_scale_after_max`, including for
    biased logits such as ALiBi.
  }];

  let assemblyFormat = "attr-dict";
}
