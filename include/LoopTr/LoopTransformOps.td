include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LinalgEraseUnusedOperandsAndResultsOp
    : Op<Transform_Dialect, "linalg.erase_unused_operands_and_results",
         [TransformOpInterface, TransformEachOpTrait,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Remove unused operands and dead results from linalg.generic";

  let description = [{
    Removes dead outputs and redundant operands of `linalg.generic` ops,
    by applying the MLIR built-in rewrite patterns
    `linalg::populateEraseUnusedOperandsAndResultsPatterns`, including:
    - `RemoveUnusedCycleInGenericOp`
    - `DeduplicateAndRemoveDeadOperandsAndResults`

    This pass uses greedy rewrites with the patterns above.
    If rewrite happens, the target handle is preserved remains valid and points to the rewritten op.
    This pass fails if the rewriter doesn't converge, or the rewrite listener fails to
    find the op after the rewrite.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::linalg::GenericOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ApplyFoldExpandingReshapePatternsOp
    : Op<Transform_Dialect, "apply_patterns.linalg.fold_expanding_reshape",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect patterns that expand linalg iteration spaces to fold reshapes into the op";

  let description = [{
    Collects patterns that create one or more size-one dimensions in the
    iteration space of a `linalg` op when that makes it possible to fold
    reshapes into the op, including:
    - `tensor.expand_shape` on the producer (operand) side
    - `tensor.collapse_shape` on the consumer (result) side
  }];

  let assemblyFormat = "attr-dict";
}

def LinalgGreedyInlineElementwiseOp : Op<Transform_Dialect, "linalg.greedy_inline_elementwise",
                                         [TransformOpInterface, TransformEachOpTrait,
                                          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Greedily inline elementwise producers into a linalg.generic consumer";

  let description = [{
    Thin transform-level wrapper around upstream `linalg::fuseElementwiseOps`.

    For the targeted `linalg.generic`, this op repeatedly inlines one producer
    of an input operand into the consumer, until no supported operations are left to inline.

    If `operand_number` is present, only that operand of the generic op is considered for inlining.
    Otherwise, operands are tried from left to right and the first eligible operand is inlined.
    The transform keeps iterating until no inlining can make further progress.
    After the inlining loop, it also applies
    `linalg::populateEraseUnusedOperandsAndResultsPatterns(...)` to remove
    dead relay results and redundant operands exposed by the fusion.

    This transform reads the target handle and modifies the payload. If payload
    tracking succeeds, the target handle remains valid and is updated to point
    to the rewritten `linalg.generic`.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target, OptionalAttr<I64Attr>:$operand_number);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::linalg::GenericOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LinalgFoldZeroIndexedUnitDimsOp : Op<Transform_Dialect, "linalg.fold_zero_indexed_unit_dims",
                                         [TransformOpInterface, TransformEachOpTrait,
                                          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Drop unit tensor dims implicated by zero-valued indexing-map results";

  let description = [{
    Removes unit-extent dimensions from the iteration space of a `linalg.generic` op.
    Compared to the upstream `linalg::dropUnitDims(...)`, this transform only removes dims
    that are implicated by `0`-valued indexing map results,
    e.g. maps like `(d0, d1, d2, d3) -> (d0, d1, d2, 0)`.

    This transform is designed to convert keepdim-style reductions to non-keepdim ones,
    and remove `(d0, d1, d2, 0)` indexing maps, which many MLIR upstream transforms
    don't recognize or like.

    This transform recursively visits the target op and ops inside it.
    While it preserves the target handle and remaps it to the rewritten op,
    it can invalidate handles that point to operations inside it.
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

def ScfLocalizeScratchTensorsOp : Op<Transform_Dialect, "scf.localize_scratch_tensors",
                                     [TransformOpInterface, TransformEachOpTrait,
                                      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                      ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Localize overwrite-only scratch tensors in loops and run cleanup";

  let description = [{
    Apply a function-level cleanup aimed at fused loop nests that carry large
    tensor values only as destination scaffolding.

    The transform first rewrites overwrite-only `tensor.extract_slice` uses of
    `scf.for` iter_args and `scf.forall` shared_out block arguments to local
    `tensor.empty` values. This is only done when:
    - the corresponding loop result has no uses outside the loop, and
    - the extracted slice is used purely as a DPS destination whose value is
      not read by the payload.

    After that rewrite, the transform reuses upstream cleanup utilities on the
    enclosing isolated op:
    - `linalg::populateSwapExtractSliceWithFillPatterns(...)`
    - `tensor::populateFoldTensorEmptyPatterns(...)`
    - `tensor::populateReassociativeReshapeFoldingPatterns(...)`
    - `scf::populateSCFForLoopCanonicalizationPatterns(...)`
    - the `remove-dead-values` pass
    - canonicalization

    In practice, this pushes scratch `tensor.empty` values down into loop
    bodies when possible and removes loop-carried full-tensor assembly state
    that is not observed outside the loop nest.
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

def ScfFuseReductionIntoForallOp
    : Op<Transform_Dialect, "scf.fuse_reduction_into_forall",
         [FunctionalStyleTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary =
      "Fuse a reduction consumer into a forall producer, sequentializing one forall dimension";

  let description = [{
    Fuse a one-dimension reduction `linalg.generic` op into an `scf.forall`.

    Prerequisites:
    - The forall must have exactly one result (the producer tile).
    - The reduction consumer must take exactly one input, produced by the forall.

    The op removes that reduction-tiled dimension from the forall, adds the
    reduction result as a new shared_out, introduces an inner sequential
    `scf.for`, and computes both the producer tile and the reduction update
    inside that loop.  The outer forall issues only the final values.

    The `forall_loop` handle is preserved and remapped to the rewritten loop.
    Existing handles to operations nested under that loop are also expected to
    remain valid when payload tracking succeeds.
    The `consumer_op` handle is consumed. The new inner `scf.for` is returned
    explicitly because it did not exist before the transformation.
  }];

  let arguments = (ins TransformHandleTypeInterface:$consumer_op,
      TransformHandleTypeInterface:$forall_loop);
  let results = (outs TransformHandleTypeInterface:$fused_op,
      TransformHandleTypeInterface:$new_inner_loop);

  let assemblyFormat = "$consumer_op `into` $forall_loop attr-dict "
                       " `:` functional-type(operands, results)";
}

def FusionIntoProducerOp
    : Op<Transform_Dialect, "fusion.into_producer",
         [FunctionalStyleTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Fuse a consumer into a producer that is also a loop";

  let description = [{
    Fuse a single consumer operation into a producer that implements LoopLikeOpInterface.

    Delegates to upstream `scf::tileAndFuseConsumer`, which:
    - identifies operands of `consumer` defined by the containing loop,
    - locates the `tensor.insert_slice`/`tensor.parallel_insert_slice` that
      produces each such operand from within the loop body,
    - tiles the consumer to operate on those same slices in-place, and
    - adds the tiled result as an additional loop-carried value.

    The `producer_loop` handle is preserved and remapped to the rewritten loop.
    The `consumer_op` handle is consumed.
  }];

  let arguments = (ins TransformHandleTypeInterface:$consumer_op,
      TransformHandleTypeInterface:$producer_loop);
  let results = (outs TransformHandleTypeInterface:$fused_op);

  let assemblyFormat = "$consumer_op `into` $producer_loop attr-dict "
                       " `:` functional-type(operands, results)";
}

def FusionFindNextReductionOp
    : Op<Transform_Dialect, "fusion.find_next_reduction",
         [NavigationTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          MemoryEffectsOpInterface]> {
  let summary = "Find the nearest reduction consumer from a producer op via forward BFS";

  let description = [{
    Pure analysis op. Forward BFS from `producer_op` along def-use edges to find the nearest
    `linalg.generic` with at least one reduction iterator. Once found, walk
    backward from the reduction (bounded by the BFS visited set) to collect
    the elementwise ops on the path.

    Returns the reduction op and the elementwise chain (def-to-use order).
    Fails if no reduction is reachable.

    This analysis is a part of a larger "rolling-update" fusion transformation.
  }];

  let arguments = (ins TransformHandleTypeInterface:$producer_op);
  let results = (outs TransformHandleTypeInterface:$next_reduce_op,
      TransformHandleTypeInterface:$elemwise_ops);

  let assemblyFormat = "$producer_op attr-dict `:` functional-type(operands, results)";
}

def FusionCloneFuseElemwiseOp
    : Op<Transform_Dialect, "fusion.clone_fuse_elemwise",
         [FunctionalStyleTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Clone elementwise ops and force-fuse them into forall + inner for";

  let description = [{
    Create a duplicated version of each elementwise op (a "sidecar")
    and fuse it under an `scf.forall` outer loop, and then under an `scf.for` inner loop.
    Fusing into `scf.for` can change the semantics of the sidecar
    (when the elemwise reads from a reduction), while the original op remains unchanged.

    The transform also rebuilds both loops to carry and publish the full-tensor
    sidecar results as additional loop outputs. Those new loop results are not
    wired into the original users yet; they are only made available for a later
    repair step at the reduction frontier.

    All the three input handles are read-only (not consumed). The loops are changed (actually
    cloned and replaces the old ones) during fusion, and these handles are remapped to point
    to the new loops. The `elemwise_chain_ops` handle is preserved because the elemwise ops
    before fusion are just unchanged.
  }];

  let arguments = (ins TransformHandleTypeInterface:$elemwise_chain_ops,
      TransformHandleTypeInterface:$outer_loop, TransformHandleTypeInterface:$inner_loop);
  let results = (outs TransformHandleTypeInterface:$sidecar_chain_ops);

  let assemblyFormat = "$elemwise_chain_ops `into` $outer_loop `,` $inner_loop attr-dict "
                       " `:` functional-type(operands, results)";
}

def FusionRepairReductionFrontierOp
    : Op<Transform_Dialect, "fusion.repair_reduction_frontier",
         [FunctionalStyleTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Repair a reduction frontier using fused sidecar ops under a streaming loop";

  let description = [{
    First reduction-frontier repair step for rolling update.

    Takes the producer reduction frontier, the frontier reduction, the ordered
    sidecar chain already fused under the streaming loop, the enclosing
    `scf.forall`, and the refreshed inner `scf.for`. The current
    implementation supports unary single-result `linalg.generic` reductions
    that consume exactly one value from the original elementwise chain.

    It rewires that reduction use to the matching relayed sidecar result,
    rebuilds the outer `scf.forall` with one extra shared_out/result,
    rebuilds the inner `scf.for` with one extra iter_arg/result, clones the
    reduction under the inner loop on the sidecar tile, and relays the new
    reduction state through the rebuilt outer loop.
  }];

  let arguments = (ins TransformHandleTypeInterface:$producer_reduces,
      TransformHandleTypeInterface:$this_reduce, TransformHandleTypeInterface:$elemwise_orig,
      TransformHandleTypeInterface:$elemwise_sidecars, TransformHandleTypeInterface:$outer_loop,
      TransformHandleTypeInterface:$inner_loop);
  let results = (outs TransformHandleTypeInterface:$repaired_reduce);

  let assemblyFormat = "`(` $producer_reduces `,` $this_reduce `)` `and` "
                       "`(` $elemwise_orig `,` $elemwise_sidecars `)` `into` "
                       "$outer_loop `,` $inner_loop attr-dict "
                       " `:` functional-type(operands, results)";
}
