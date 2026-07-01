#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

static LogicalResult canonicalizeForLoop(RewriterBase &rewriter, scf::ForallOp loop) {
  RewritePatternSet patterns(rewriter.getContext());
  scf::ForallOp::getCanonicalizationPatterns(patterns, rewriter.getContext());
  GreedyRewriteConfig config;
  config.setListener(static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  return applyOpPatternsGreedily({loop}, FrozenRewritePatternSet(std::move(patterns)), config);
}

DiagnosedSilenceableFailure
FusionGreedyConsumersIntoProducerOp::apply(transform::TransformRewriter &rewriter,
                                           TransformResults &transformResults,
                                           TransformState &state) {
  const auto transform = cast<TransformOpInterface>(getOperation());
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

  SmallVector<Operation *> fusedOps;
  while (true) {
    SmallVector<Operation *> consumers;
    for (Operation *consumer : loop->getResult(resultNumber).getUsers()) {
      if (stopOps.contains(consumer)) {
        // Stop all fusing when we reach the stop op.
        transformResults.set(getOperation()->getResult(0), fusedOps);
        return DiagnosedSilenceableFailure::success();
      }
      consumers.push_back(consumer);
    }
    if (consumers.empty()) {
      // No consumer found -- we are done.
      transformResults.set(getOperation()->getResult(0), fusedOps);
      return DiagnosedSilenceableFailure::success();
    }
    // Sort by their position in the block so that we fuse consumers in program order.
    llvm::sort(consumers, [](Operation *lhs, Operation *rhs) {
      return lhs->getBlock() == rhs->getBlock() && lhs->isBeforeInBlock(rhs);
    });

    for (Operation *consumer : consumers) {
      SmallVector<LoopLikeOpInterface> loops{loop};
      FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult =
          tileAndFuseConsumerWithDebug(rewriter, *consumer, loops);
      if (failed(fuseResult))
        BAIL("failed to tile and fuse elementwise consumer into loop");
      if (fuseResult->tiledOps.empty())
        BAIL("consumer had no operands defined by the containing loop");
      loop = cast<scf::ForallOp>(loops.front());
      fusedOps.append(fuseResult->tiledOps);
      if (isOpTriviallyDead(consumer))
        rewriter.eraseOp(consumer);
    }

    if (failed(canonicalizeForLoop(rewriter, loop)))
      BAIL("failed to canonicalize the producer loop after consumer fusion");
    // Update `loop` from the handle because canonicalization may have replaced the forall.
    CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getProducerLoop, "producer loop", loop,
                                 scf::ForallOp);
  }
}

LogicalResult FusionGreedyConsumersIntoProducerOp::verify() { return success(); }

} // namespace mlir::transform
