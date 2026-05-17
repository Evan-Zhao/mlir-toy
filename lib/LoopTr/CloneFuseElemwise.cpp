#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"

#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

namespace mlir::transform {

using scf::ForallOp;
using scf::ForOp;

namespace {

struct MatchFailureCaptureListener : public mlir::RewriterBase::ForwardingListener {
  using Base = mlir::RewriterBase::ForwardingListener;

  explicit MatchFailureCaptureListener(mlir::OpBuilder::Listener *previous) : Base(previous) {}

  void notifyMatchFailure(mlir::Location loc,
                          llvm::function_ref<void(mlir::Diagnostic &)> reasonCallback) override {
    // Preserve any existing listener behavior.
    Base::notifyMatchFailure(loc, reasonCallback);

    mlir::Diagnostic diag(loc, mlir::DiagnosticSeverity::Remark);
    reasonCallback(diag);

    std::string msg;
    llvm::raw_string_ostream os(msg);
    diag.print(os);
    os.flush();

    messages.push_back(std::move(msg));
  }

  llvm::SmallVector<std::string> messages;
};

FailureOr<scf::SCFFuseConsumerOfSliceResult>
tryTileAndFuseConsumerWithDebug(mlir::RewriterBase &rewriter, mlir::Operation *consumer,
                                mlir::MutableArrayRef<mlir::LoopLikeOpInterface> loops) {
  mlir::OpBuilder::Listener *previousListener = rewriter.getListener();
  MatchFailureCaptureListener capture(previousListener);
  rewriter.setListener(&capture);
  auto restoreListener = llvm::scope_exit([&]() { rewriter.setListener(previousListener); });
  FailureOr<scf::SCFFuseConsumerOfSliceResult> result =
      scf::tileAndFuseConsumer(rewriter, consumer, loops);
  if (failed(result)) {
    llvm::errs() << "\nCaptured match failures:\n";
    for (StringRef msg : capture.messages)
      llvm::errs() << "  - " << msg << "\n";
  }
  return result;
}

LogicalResult recursiveMoveOperandsBeforeOp(Operation *toMoveOperands, RewriterBase &rewriter,
                                            Operation *moveBefore) {
  IRMapping mapping;
  rewriter.setInsertionPoint(moveBefore);
  for (auto value : toMoveOperands->getOperands()) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      continue;
    // This function does not clone if the value is already defined before the insertion point of
    // the rewriter.
    auto newValue = cloneValueDefChainAtInsertionPoint(rewriter, result, mapping);
    if (failed(newValue)) {
      result.getDefiningOp()->emitRemark("when cloning this operation");
      return failure();
    }
    if (*newValue == value)
      continue;
    mapping.map(result, *newValue);
    rewriter.replaceAllUsesWith(result, *newValue);
    rewriter.eraseOp(result.getDefiningOp());
  }
  return success();
}

/// Detects `tensor.insert_slice` operations that feed into the yield of a loop,
/// and moves them right before the yield. This reduces the chance of scf::tileAndFuseConsumer
/// getting confused.
LogicalResult sinkYieldInsertSlices(RewriterBase &rewriter, ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (Value operand : yield.getOperands()) {
    auto insertSlice = operand.getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSlice)
      continue;
    if (insertSlice->getBlock() != yield->getBlock() ||
        !llvm::hasSingleElement(insertSlice->getUses()))
      return failure();
    rewriter.moveOpBefore(insertSlice, yield);
  }
  return success();
}

} // namespace

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
    if (failed(recursiveMoveOperandsBeforeOp(newElemwiseOp, rewriter, outerLoop)))
      BAIL_AND_POINT("failed to move operands before the outer loop");

    // We are going to use scf::tileAndFuseConsumer twice. While it takes a vector of loops, it can
    // only work with one scf.forall loop at a time.
    SmallVector<LoopLikeOpInterface> outerLoops{outerLoop};
    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedIntoForall =
        tryTileAndFuseConsumerWithDebug(rewriter, newElemwiseOp, outerLoops);
    if (failed(fusedIntoForall))
      BAIL_AND_POINT("failed to use scf::tileAndFuseConsumer on the outer loop");
    outerLoop = cast<ForallOp>(outerLoops[0]);
    auto outerFusedOp = fusedIntoForall->tiledOps[0];

    // Map the old (before clone) elemwise op results to the new fused op results, which is the new
    // return values of the outer loop.
    auto newLoopResults = outerLoop->getResults().take_back(newElemwiseOp->getNumResults());
    for (auto [oldResult, newLoopResult] :
         llvm::zip_equal(elemwiseOp->getResults(), newLoopResults)) {
      mapping.map(oldResult, newLoopResult);
    }

    // Similarly, this first fusion may have inserted some operations after the inner loop, and we
    // move them before the inner loop.
    if (failed(recursiveMoveOperandsBeforeOp(outerFusedOp, rewriter, innerLoop)))
      BAIL_AND_POINT("failed to move operands before the inner loop");
    if (failed(sinkYieldInsertSlices(rewriter, innerLoop)))
      BAIL_AND_POINT("failed to sink inner-loop insert_slice yield operands");

    // Apply the same fusion on the inner loop.
    SmallVector<LoopLikeOpInterface> innerLoops{innerLoop};
    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedIntoFor =
        tryTileAndFuseConsumerWithDebug(rewriter, outerFusedOp, innerLoops);
    if (failed(fusedIntoFor))
      BAIL_AND_POINT("failed to use scf::tileAndFuseConsumer on inner loop");
    innerLoop = cast<ForOp>(innerLoops[0]);
    auto innerFusedOp = fusedIntoFor->tiledOps[0];

    // Remove the cloned op and the outer fused op.
    rewriter.eraseOp(newElemwiseOp);
    rewriter.eraseOp(outerFusedOp);
    // Update elemwiseOps inplace.
    elemwiseOp = innerFusedOp;
  }

  // Fold some tensor.{insert|extract}_slice operations because scf::tileAndFuseConsumer
  // can introduce a lot of them.
  RewritePatternSet patterns(getContext());
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  SmallVector<Operation *> ops;
  outerLoop->walk([&](Operation *op) {
    if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op))
      ops.push_back(op);
  });
  if (failed(applyOpPatternsGreedily(ops, std::move(patterns))))
    return emitSilenceableFailure(
        transform, "failed to apply merge consecutive insert/extract_slice patterns");

  // Just return elemwiseOps because we've updated that vector inplace.
  transformResults.set(getOperation()->getResult(0), elemwiseOps);
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
