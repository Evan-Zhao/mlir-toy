#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include <deque>

namespace mlir::transform {

#define BAIL(message) return emitSilenceableFailure(transform, message);

void FusionIntoProducerOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getConsumerOpMutable(), effects);
  onlyReadsHandle(getProducerLoopMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure FusionIntoProducerOp::apply(transform::TransformRewriter &rewriter,
                                                        TransformResults &transformResults,
                                                        TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());

  // Step 1. Resolve the payload ops and check that the producer is loop-like.
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getConsumerOp, "consumer", consumer);
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getProducerLoop, "producer loop", loop);

  auto loopI = dyn_cast<LoopLikeOpInterface>(loop);
  if (!loopI)
    BAIL("expected the producer loop to implement the LoopLikeOpInterface");

  // Step 2. Delegate the actual tile-and-fuse rewrite to the upstream SCF utility.
  FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult =
      tileAndFuseConsumerWithDebug(rewriter, *consumer, {loopI});
  if (failed(fuseResult))
    BAIL("failed to tile and fuse elementwise consumer into loop");
  if (fuseResult->tiledOps.empty())
    BAIL("consumer had no operands defined by the containing loop");

  // Step 3. Clean up the old consumer if it became dead and publish the new handles.
  if (isOpTriviallyDead(consumer))
    rewriter.eraseOp(consumer);

  transformResults.set(getOperation()->getResult(0), fuseResult->tiledOps);
  return DiagnosedSilenceableFailure::success();
}

void FusionGreedyConsumersIntoProducerOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getProducerLoopMutable(), effects);
  onlyReadsHandle(getStopOpMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

static FailureOr<scf::ForallOp> canonicalizeForLoop(RewriterBase &rewriter, scf::ForallOp loop) {
  RewritePatternSet patterns(rewriter.getContext());
  scf::ForallOp::getCanonicalizationPatterns(patterns, rewriter.getContext());
  SmallVector<Operation *> trackedLoops{loop};
  TrackedOperationsListener listener(trackedLoops, rewriter.getListener());
  GreedyRewriteConfig config;
  config.setListener(&listener);
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  if (failed(applyOpPatternsGreedily({loop}, FrozenRewritePatternSet(std::move(patterns)), config)))
    return failure();

  if (!llvm::hasSingleElement(trackedLoops))
    return failure();
  auto rewrittenLoop = dyn_cast<scf::ForallOp>(trackedLoops.front());
  if (!rewrittenLoop)
    return failure();
  return rewrittenLoop;
}

DiagnosedSilenceableFailure
FusionGreedyConsumersIntoProducerOp::apply(transform::TransformRewriter &rewriter,
                                           TransformResults &transformResults,
                                           TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  scf::ForallOp loop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getProducerLoop, "producer loop", loop,
                               scf::ForallOp);
  DenseSet<Operation *> stopOps;
  if (Value stopOpHandle = getStopOp()) {
    auto stopOpRange = state.getPayloadOps(stopOpHandle);
    stopOps.insert(stopOpRange.begin(), stopOpRange.end());
  }
  const size_t resultNumber = getResultNumber();
  if (resultNumber >= loop->getNumResults())
    BAIL("result number is out of range for producer loop");
  const bool inlineElemwise = getInlineElementwise();

  SmallVector<Operation *> fusedOps;
  DenseSet<Operation *> failedConsumers;
  OpBuilder::Listener *previousListener = rewriter.getListener();
  TrackedOperationsListener fusedOpsListener(fusedOps, previousListener);
  rewriter.setListener(&fusedOpsListener);
  auto restoreListener = llvm::scope_exit([&]() { rewriter.setListener(previousListener); });

  while (true) {
    // Run CSE on the loop body because fusion may fail without it (fusion compares indices by
    // operation equality of affine ops, so we want to make sure that we don't have duplicate affine
    // ops in the loop body).
    eliminateLocalCommonSubexpressions(rewriter, loop);

    SmallVector<Operation *> consumers;
    for (Operation *consumer : loop->getResult(resultNumber).getUsers()) {
      if (failedConsumers.contains(consumer))
        continue;
      if (stopOps.contains(consumer)) {
        // Stop all fusing when we reach the stop op.
        transformResults.set(getOperation()->getResult(0), fusedOps);
        if (!failedConsumers.empty())
          transform.emitRemark("did not fuse all discovered consumers into the producer loop");
        return DiagnosedSilenceableFailure::success();
      }
      consumers.push_back(consumer);
    }
    if (consumers.empty()) {
      // No consumer found -- we are done.
      transformResults.set(getOperation()->getResult(0), fusedOps);
      if (!failedConsumers.empty())
        transform.emitRemark("did not fuse all discovered consumers into the producer loop");
      return DiagnosedSilenceableFailure::success();
    }
    // Sort by their position in the block so that we fuse consumers in program order.
    llvm::sort(consumers, [](Operation *lhs, Operation *rhs) {
      return lhs->getBlock() == rhs->getBlock() && lhs->isBeforeInBlock(rhs);
    });

    for (Operation *consumer : consumers) {
      auto genericConsumer = dyn_cast<linalg::GenericOp>(consumer);
      if (inlineElemwise && genericConsumer) {
        FailureOr<ElementwiseInlineResult> inlineResult =
            greedyInlineElementwiseProducers(rewriter, genericConsumer);
        if (failed(inlineResult))
          BAIL("failed to inline elementwise producer into consumer");
        consumer = inlineResult->fusedOp;
      }

      SmallVector<LoopLikeOpInterface> loops{loop};
      FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult =
          tileAndFuseConsumerWithDebug(rewriter, *consumer, loops);
      if (failed(fuseResult) || fuseResult->tiledOps.empty()) {
        consumer->emitRemark("failed to fuse this consumer into the producer loop");
        failedConsumers.insert(consumer);
        continue;
      }
      loop = cast<scf::ForallOp>(loops.front());
      fusedOps.append(fuseResult->tiledOps);
      if (isOpTriviallyDead(consumer))
        rewriter.eraseOp(consumer);
    }

    FailureOr<scf::ForallOp> canonicalizedLoop = canonicalizeForLoop(rewriter, loop);
    if (failed(canonicalizedLoop))
      BAIL("failed to canonicalize the producer loop after consumer fusion");
    loop = *canonicalizedLoop;
    if (resultNumber >= loop->getNumResults())
      BAIL("result number is out of range for producer loop after canonicalization");
  }
}

void FusionGreedyProducersIntoConsumerOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getProducerOpsMutable(), effects);
  onlyReadsHandle(getConsumerOpMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
FusionGreedyProducersIntoConsumerOp::apply(transform::TransformRewriter &rewriter,
                                           TransformResults &transformResults,
                                           TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_NON_EMPTY_OPS(state, transform, getProducerOps, "producer", producers);
  CHECK_NON_EMPTY_OPS(state, transform, getConsumerOp, "consumer loop", consumerLoops);

  // The consumer handle denotes a loop nest. Infer its outer-to-inner order
  // from payload ancestry rather than relying on the order in the handle.
  llvm::sort(consumerLoops, [](Operation *lhs, Operation *rhs) {
    auto getDepth = [](Operation *op) {
      unsigned depth = 0;
      for (Operation *parent = op->getParentOp(); parent; parent = parent->getParentOp())
        ++depth;
      return depth;
    };
    return getDepth(lhs) < getDepth(rhs);
  });

  SmallVector<LoopLikeOpInterface> loops;
  loops.reserve(consumerLoops.size());
  for (auto [index, loop] : llvm::enumerate(consumerLoops)) {
    auto loopLike = dyn_cast<LoopLikeOpInterface>(loop);
    if (!loopLike)
      BAIL("expected every consumer loop to implement LoopLikeOpInterface");
    if (index > 0 && !consumerLoops[index - 1]->isProperAncestor(loop))
      BAIL("expected consumer loops to form a strictly nested loop nest");
    loops.push_back(loopLike);
  }
  if (!isa<scf::ForallOp>(loops.front().getOperation()))
    BAIL("expected the outermost consumer loop to be an scf.forall");

  // Expose an inner tile directly on the original producer. In particular,
  // this composes an inner extract_slice with an outer extract_slice before
  // producer fusion chooses the innermost loop as its placement boundary.
  if (loops.size() > 1) {
    RewritePatternSet patterns(rewriter.getContext());
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    GreedyRewriteConfig config;
    config.setListener(static_cast<RewriterBase::Listener *>(rewriter.getListener()));
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    SmallVector<Operation *> slices;
    loops.back()->walk(
        [&](tensor::ExtractSliceOp slice) { slices.push_back(slice.getOperation()); });
    if (!slices.empty() && failed(applyOpPatternsGreedily(
                               slices, FrozenRewritePatternSet(std::move(patterns)), config)))
      BAIL("failed to merge consecutive tensor slices in the consumer loop nest");
  }

  Operation *fusionBoundary = loops.back().getOperation();
  SmallVector<Value> producerResults;
  for (Operation *producer : producers)
    llvm::append_range(producerResults, producer->getResults());
  DefUsePathCollection paths = collectOpsOnDefUsePaths(producerResults, [&](Operation *op) {
    return fusionBoundary->isAncestor(op) && isa<tensor::ExtractSliceOp>(op);
  });
  DenseSet<Operation *> allowedOps;
  allowedOps.insert(paths.operations.begin(), paths.operations.end());

  std::deque<tensor::ExtractSliceOp> worklist;
  auto enqueue = [&](tensor::ExtractSliceOp slice) {
    auto source = dyn_cast<OpResult>(slice.getSource());
    if (!source || fusionBoundary->isAncestor(source.getOwner()) ||
        !allowedOps.contains(source.getOwner()))
      return;
    worklist.push_back(slice);
  };
  for (Operation *descendant : paths.descendants)
    enqueue(cast<tensor::ExtractSliceOp>(descendant));

  SmallVector<Operation *> fusedOps;
  bool hasFailure = false;
  while (!worklist.empty()) {
    tensor::ExtractSliceOp slice = worklist.front();
    worklist.pop_front();

    std::optional<scf::SCFFuseProducerOfSliceResult> fused =
        scf::tileAndFuseProducerOfSlice(rewriter, slice, loops);
    if (!fused) {
      if (Operation *source = slice.getSource().getDefiningOp())
        source->emitRemark("failed to fuse this op into the consumer loop nest");
      else
        slice.emitRemark("failed to fuse the producer of this slice into the consumer loop nest");
      hasFailure = true;
      continue;
    }
    fusedOps.append(fused->tiledOps);

    fusionBoundary = loops.back().getOperation();
    for (Operation *generated : fused->generatedSlices)
      if (auto generatedSlice = dyn_cast<tensor::ExtractSliceOp>(generated))
        enqueue(generatedSlice);

    // tileAndFuseProducerOfSlice replaces the uses of the slice but
    // intentionally leaves the now-dead operation behind.
    if (slice->use_empty())
      rewriter.eraseOp(slice);
  }

  SmallVector<Operation *> updatedLoops =
      llvm::map_to_vector(loops, [](LoopLikeOpInterface loop) { return loop.getOperation(); });
  transformResults.set(getOperation()->getResult(0), fusedOps);
  transformResults.set(getOperation()->getResult(1), updatedLoops);
  if (hasFailure)
    transform.emitRemark("failed to fuse some producers into the consumer loop nest");
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
