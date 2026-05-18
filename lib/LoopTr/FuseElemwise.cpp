#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::transform {

#define BAIL(message) return emitSilenceableFailure(transform, message);

using scf::ForallOp;
using scf::ForOp;

void LoopFuseIntoProducerOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getConsumerOpMutable(), effects);
  onlyReadsHandle(getProducerLoopMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure LoopFuseIntoProducerOp::apply(transform::TransformRewriter &rewriter,
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

void LoopRUCloneFuseElemwise::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getElemwiseChainOpsMutable(), effects);
  onlyReadsHandle(getOuterLoopMutable(), effects);
  onlyReadsHandle(getInnerLoopMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure LoopRUCloneFuseElemwise::apply(transform::TransformRewriter &rewriter,
                                                           TransformResults &transformResults,
                                                           TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseChainOps, "elementwise", elemwiseOps)
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getOuterLoop, "outer loop", outerLoop, ForallOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getInnerLoop, "inner loop", innerLoop, ForOp);

  // Keep a value mapping so that later elemwise ops can be remapped to read from in-loop values.
  IRMapping mapping;
  for (Operation *&elemwiseOp : elemwiseOps) {
#define BAIL_AND_POINT(message)                                                                    \
  {                                                                                                \
    elemwiseOp->emitError() << "failed on this elementwise op";                                    \
    return emitSilenceableFailure(transform, message);                                             \
  }
    // Check if the op is an elemwise op.
    if (failed(isSingleOutputElemwiseLinalgOp(elemwiseOp)))
      BAIL_AND_POINT(
          "expected every op to be an elementwise linalg.map or linalg.generic with one result");

    // Clone the elemwise op so we don't affect the original one at all.
    rewriter.setInsertionPoint(elemwiseOp);
    auto newElemwiseOp = rewriter.clone(*elemwiseOp, mapping);
    // This cloning may have inserted some operations after the outer loop, which prevents the
    // fusion from working. We'll try and move them before the inner loop.
    if (failed(recursiveMoveOperandsBeforeOp(*newElemwiseOp, rewriter, *outerLoop)))
      BAIL_AND_POINT("failed to move operands before the outer loop");

    // Run fusion with our helper.
    auto fuseResult =
        tileAndFuseConsumerIntoDoubleLoops(rewriter, outerLoop, innerLoop, *newElemwiseOp);
    if (failed(fuseResult))
      BAIL_AND_POINT("failed to fuse consumer into double loops");
    auto [outerFusedOp, innerFusedOp] = *fuseResult;

    // Map the old (before clone) elemwise op results to the new fused op results, which is the new
    // return values of the outer loop.
    auto newLoopResults = outerLoop->getResults().take_back(elemwiseOp->getNumResults());
    for (auto [oldResult, newLoopResult] :
         llvm::zip_equal(elemwiseOp->getResults(), newLoopResults)) {
      mapping.map(oldResult, newLoopResult);
    }

    // Remove the cloned op and the outer fused op.
    rewriter.eraseOp(newElemwiseOp);
    rewriter.eraseOp(outerFusedOp);
    // Update elemwiseOps inplace.
    elemwiseOp = innerFusedOp;

    // Upstream tile-and-fuse requires structurally identical offsets/sizes
    // across operand slices. Deduplicate the affine.apply scaffolding we just
    // introduced before attempting to fuse the next elemwise consumer.
    eliminateLocalCommonSubexpressions(rewriter, outerLoop.getOperation());
  }

  // Just return elemwiseOps because we've updated that vector inplace.
  transformResults.set(getOperation()->getResult(0), elemwiseOps);
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
