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

def ScfFoldUnitExtentDimsViaReshapesPatternsOp
    : Op<Transform_Dialect, "apply_patterns.scf.fold_unit_extent_dims_via_reshapes",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Collect SCF loop-carried unit-extent dimension folding patterns";

  let description = [{
    Collects rewrite patterns that rebuild `scf.for` and `scf.forall` ops whose
    loop-carried tensor values have statically-known unit-extent dimensions.
    The replacement loop carries collapsed tensors, using the same reassociation
    helper as `transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes`.

    Bridge `tensor.collapse_shape` and `tensor.expand_shape` ops are inserted
    around the rebuilt loop as needed. For `scf.forall`,
    `tensor.parallel_insert_slice` combining ops are rewritten against the
    collapsed shared_out block arguments; non-combining uses of shared_out block
    arguments are bridged with `tensor.expand_shape`.
  }];

  let assemblyFormat = "attr-dict";
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

def ScfFusePartialReductionIntoForallOp
    : Op<Transform_Dialect, "scf.fuse_partial_reduction_into_forall",
         [FunctionalStyleTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Fuse a reduction consumer into a forall producer as a partial reduction";

  let description = [{
    Fuse a one-dimension reduction `linalg.generic` op into an existing
    `scf.forall`, keeping the reduction-tiled forall dimension parallel.

    This is the partial-reduction counterpart to
    `transform.scf.fuse_reduction_into_forall`:
    - `fuse_reduction_into_forall` rewrites the existing forall dimension into
      an inner sequential `scf.for` and produces a scan/rolling form,
    - `fuse_partial_reduction_into_forall` keeps the existing forall dimension
      parallel, publishes split-local partial reductions through an extra
      `shared_out`, and creates a final write-back reduction after the forall.

    Prerequisites:
    - The reduction consumer must take one reduced input produced by the target
      `scf.forall`.
    - The reduced dimension of that input tensor must correspond to one dynamic
      `scf.forall` induction variable.
    - The reduction must implement `PartialReductionOpInterface`.

    The `forall_loop` handle is preserved and remapped to the rewritten loop.
    The `consumer_op` handle is consumed. The returned handles are the fused
    split-local partial reduction op and the final write-back reduction op.
  }];

  let arguments = (ins TransformHandleTypeInterface:$consumer_op,
      TransformHandleTypeInterface:$forall_loop);
  let results = (outs TransformHandleTypeInterface:$partial_reduce_op,
      TransformHandleTypeInterface:$writeback_reduce_op);

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

def FusionGreedyConsumersIntoProducerOp
    : Op<Transform_Dialect, "fusion.greedy_consumers_into_producer",
         [FunctionalStyleTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Greedily fuse consumers into a producer loop";

  let description = [{
    Starting from `producer_loop[result_number]`, repeatedly fuse the unique
    direct consumer of the current loop result into `producer_loop`.

    This transform follows a single def-use chain. The `result_number`
    identifies only the initial loop result. After each successful consumer
    fusion, the implementation is expected to advance the frontier to the loop
    result produced by the fused consumer, rather than continuing to query the
    original result number.

    If `stop_op` is provided and the next direct consumer is one of its payload
    operations, fusion stops before that consumer and succeeds. This is useful
    for fusing the elementwise prefix of a chain and stopping at a reduction
    frontier.

    If `inline_elementwise` is present, the transform should first try to
    greedily inline elementwise producers into each selected consumer before
    fusing that consumer into the loop. This inlining is best-effort: a
    consumer with no eligible elementwise producers should still be considered
    for loop fusion.

    The `producer_loop` handle is preserved and remapped to the rewritten loop.
    The optional `stop_op` handle is read-only. The result handle contains the
    fused consumer operations in fusion order.
  }];

  let arguments = (ins TransformHandleTypeInterface:$producer_loop, I64Attr:$result_number,
      Optional<TransformHandleTypeInterface>:$stop_op, UnitAttr:$inline_elementwise);
  let results = (outs TransformHandleTypeInterface:$fused_ops);

  let assemblyFormat = "$producer_loop `[` $result_number `]` "
                       "(`until` $stop_op^)? "
                       "(`inline_elementwise` $inline_elementwise^)? "
                       "attr-dict `:` functional-type(operands, results)";
}

def FusionGreedyProducersIntoConsumerOp
    : Op<Transform_Dialect, "fusion.greedy_producers_into_consumer",
         [FunctionalStyleTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Greedily fuse producers into a consumer loop";

  let description = [{
    Greedily fuse producers into `consumer_op` along def-use paths that start
    at one of the operations in the required `producer_ops` handle and end at
    a tensor.extract_slice nested in the consumer loop.
    Producers outside these paths are not fused.

    `consumer_op` may identify either one loop or multiple strictly nested
    loops. The outermost loop must be an `scf.forall`. When multiple loops are
    provided, producers are fused into the innermost loop. Loop nesting, rather
    than handle order, determines the outer-to-inner order passed to MLIR's
    producer fusion utility.

    Producers are processed in a topological order whenever possible, so a
    producer exposed by fusing another producer can be fused in a later step.
    Function arguments and non-tileable operations are not fused.

    The `producer_ops` and `consumer_op` handles are read-only. The
    `new_consumer` result contains the updated loop nest in outer-to-inner
    order. The `fused_ops` result contains the producer operations fused in
    fusion order.
  }];

  let arguments = (ins TransformHandleTypeInterface:$producer_ops,
      TransformHandleTypeInterface:$consumer_op);
  let results = (outs TransformHandleTypeInterface:$fused_ops,
      TransformHandleTypeInterface:$new_consumer);

  let assemblyFormat = "$producer_ops `into` $consumer_op attr-dict "
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

    If `result_number` is present, the BFS starts only from that result of
    `producer_op`. Otherwise, it starts from all results of `producer_op`.

    This analysis is a part of a larger "rolling-update" fusion transformation.
  }];

  let arguments = (ins TransformHandleTypeInterface:$producer_op,
      OptionalAttr<I64Attr>:$result_number);
  let results = (outs TransformHandleTypeInterface:$next_reduce_op,
      TransformHandleTypeInterface:$elemwise_ops);

  let assemblyFormat = "$producer_op (`[` $result_number^ `]`)? attr-dict "
                       " `:` functional-type(operands, results)";
}

def LoopSpecializeDeadTileOp
    : Op<Transform_Dialect, "loop.specialize_dead_tile",
         [FunctionalStyleTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
          ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Split a streaming loop into a fully-live prefix and a mixed suffix";

  let description = [{
    The op takes a loop-local `linalg.generic` producer, an enclosing `scf.for`,
    and a schedule-provided `dead_value` attribute. It recognizes a yielded
    `arith.select` that chooses between a live value and the dead value,
    derives live-interval information over the selected loop IV by eliminating
    the producer's `linalg.index` dimensions, and rewrites the `scf.for` into:
    - a fully-live prefix loop, where the producer is bypassed and replaced by
      its live input, and
    - a mixed suffix loop, where the original producer remains.

    The `loop` handle is consumed because the original loop is replaced by two
    new loops. The transform returns handles to those new loops explicitly:
    `live_loop` and `mixed_loop`.

    The optional `producer_op` handle is read-only and remains valid after the rewrite.
    If provided, it is remapped to the cloned producer operation in the mixed suffix loop.
    If omitted or empty, the producer is inferred as the unique dead-select
    `linalg.generic` nested in `loop`. If no such producer is found, the op is
    a no-op and returns empty handles. If more than one is found, the op fails.
  }];

  let arguments = (ins Optional<TransformHandleTypeInterface>:$producer_op,
      TransformHandleTypeInterface:$loop, AnyAttr:$dead_value);
  let results = (outs TransformHandleTypeInterface:$live_loop,
      TransformHandleTypeInterface:$mixed_loop);

  let assemblyFormat = "($producer_op^)? `in` $loop attr-dict "
                       " `:` functional-type(operands, results)";
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

def FusionCloneFuseRfactorElemwiseOp
    : Op<Transform_Dialect, "fusion.clone_fuse_rfactor_elemwise",
         [FunctionalStyleTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Clone elementwise ops and fuse them into a forall with rfactor substitution";

  let description = [{
    Create a duplicated version of each elementwise op (a "sidecar") and fuse
    it under an existing `scf.forall`.

    This is the rfactor/write-back counterpart to
    `transform.fusion.clone_fuse_elemwise`. Instead of fusing through an
    `scf.forall` + inner `scf.for` loop nest, it keeps the split dimension
    parallel and rewrites reads of write-back reductions to the corresponding
    split-local rfactor reductions.

    The `writeback_reduce_ops` and `rfactor_reduce_ops` handles are paired in
    handle order. During cloning, any sidecar read of a write-back reduction
    result is substituted with the matching rfactor reduction result inside the
    forall body. The original out-of-loop elementwise chain is left unchanged;
    the fused sidecar chain is returned for later repair.

    The `forall_loop` handle is preserved and remapped to the rebuilt loop.
    The `elemwise_chain_ops`, `writeback_reduce_ops`, and `rfactor_reduce_ops`
    handles are read-only.
  }];

  let arguments = (ins TransformHandleTypeInterface:$elemwise_chain_ops,
      TransformHandleTypeInterface:$forall_loop, TransformHandleTypeInterface:$writeback_reduce_ops,
      TransformHandleTypeInterface:$rfactor_reduce_ops);
  let results = (outs TransformHandleTypeInterface:$sidecar_chain_ops);

  let assemblyFormat = "$elemwise_chain_ops `into` $forall_loop "
                       "`substituting` `(` $writeback_reduce_ops `->` "
                       "$rfactor_reduce_ops `)` attr-dict "
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

  let arguments = (ins TransformHandleTypeInterface:$this_reduce,
      TransformHandleTypeInterface:$producer_reduces, TransformHandleTypeInterface:$elemwise_orig,
      TransformHandleTypeInterface:$elemwise_sidecars, TransformHandleTypeInterface:$outer_loop,
      TransformHandleTypeInterface:$inner_loop);
  let results = (outs TransformHandleTypeInterface:$repaired_reduce);

  let assemblyFormat =
      "$this_reduce `reduce_producer` $producer_reduces "
      "`substituting` `elemwise` $elemwise_orig `->` $elemwise_sidecars "
      "`into` $outer_loop `,` $inner_loop attr-dict `:` functional-type(operands, results)";
}

def FusionRepairRfactorReductionFrontierOp
    : Op<Transform_Dialect, "fusion.repair_rfactor_reduction_frontier",
         [FunctionalStyleTransformOpTrait, DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Repair a SplitK rfactor reduction frontier before write-back";

  let description = [{
    SplitK reduction-frontier repair.

    This is the rfactor/write-back counterpart to
    `transform.fusion.repair_reduction_frontier`. Instead of repairing a
    loop-carried accumulator under an `scf.forall` + inner `scf.for` rolling
    loop nest, this transform repairs split-local rfactor partials before the
    final write-back reduction.

    The producer write-back and rfactor reductions are paired in handle order.
    They describe values that the sidecar chain reads as final write-back
    tensors outside the loop, but which should be interpreted as split-local
    rfactor tensors when deriving and materializing the repair term.

    The transform is expected to:
    - stage the frontier reduction to consume the fused rfactor sidecar chain,
    - split that frontier into a new rfactor/write-back pair under the existing
      `scf.forall`,
    - derive a repair term `H` using the shared fusion expression solver, and
    - materialize `H` over rfactor tensors before rebuilding the write-back
      reduction to consume the repaired rfactor tensor.

    The `forall_loop`, producer reduction handles, and elementwise-chain
    handles are read-only. The `this_reduce` handle is consumed and replaced by
    the repaired write-back result. The returned handles are the split-local
    rfactor reduction for this frontier and the repaired write-back reduction.
  }];

  let arguments = (ins TransformHandleTypeInterface:$this_reduce,
      TransformHandleTypeInterface:$reduces_wb, TransformHandleTypeInterface:$reduces_rf,
      TransformHandleTypeInterface:$elemwise_orig, TransformHandleTypeInterface:$elemwise_sidecars,
      TransformHandleTypeInterface:$forall_loop);
  let results = (outs TransformHandleTypeInterface:$repaired_rfactor_reduce,
      TransformHandleTypeInterface:$repaired_writeback_reduce);

  let assemblyFormat = "$this_reduce `substituting` `reduce` $reduces_wb `->` $reduces_rf "
                       "`elemwise` $elemwise_orig `->` $elemwise_sidecars "
                       "`into` $forall_loop attr-dict `:` functional-type(operands, results)";
}
