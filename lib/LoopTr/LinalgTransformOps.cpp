#include "LoopTr/LoopTransformOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mlir/Support/LLVM.h>

namespace mlir::transform {

namespace {

#define BAIL(message) return emitSilenceableFailure(transform, message);
LogicalResult rewriteGreedilyWithPatternSet(MLIRContext *context, OpBuilder &rewriter,
                                            RewritePatternSet &patterns, Operation *target) {
  GreedyRewriteConfig config;
  config.setListener(static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  bool changed = false;
  return applyOpPatternsGreedily({target}, FrozenRewritePatternSet(std::move(patterns)), config,
                                 &changed);
}

} // namespace

void LinalgEraseUnusedOperandsAndResultsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure LinalgEraseUnusedOperandsAndResultsOp::applyToOne(
    TransformRewriter &rewriter, linalg::GenericOp target, ApplyToEachResultList &results,
    TransformState &state) {
  (void)results;
  (void)state;
  auto transform = cast<TransformOpInterface>(getOperation());

  RewritePatternSet patterns(getContext());
  linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
  if (failed(rewriteGreedilyWithPatternSet(getContext(), rewriter, patterns, target))) {
    BAIL("erase_unused_operands_and_results did not converge");
  }
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
