#include "LoopTr/LoopTransformOps.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::transform {

void LoopEraseUnusedOperandsAndResultsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure LoopEraseUnusedOperandsAndResultsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results, transform::TransformState &state) {
  (void)results;
  (void)state;

  RewritePatternSet patterns(getContext());
  linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);

  GreedyRewriteConfig config;
  config.setListener(static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);

  bool changed = false;
  if (failed(applyOpPatternsGreedily({target}, FrozenRewritePatternSet(std::move(patterns)), config,
                                     &changed))) {
    return emitDefiniteFailure() << "erase-unused-operands-and-results did not converge";
  }
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
