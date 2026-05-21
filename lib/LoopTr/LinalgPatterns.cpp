#include "LoopTr/LoopTransformOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::transform {

namespace {

static SmallVector<unsigned> getLoopDimsWithZeroIndexedMaps(linalg::GenericOp genericOp) {
  SmallVector<unsigned> allowedDims;
  llvm::SmallBitVector seen(genericOp.getNumLoops(), false);

  for (AffineMap map : genericOp.getIndexingMapsArray()) {
    bool hasZeroResult = llvm::any_of(map.getResults(), [](AffineExpr expr) {
      auto constantExpr = dyn_cast<AffineConstantExpr>(expr);
      return constantExpr && constantExpr.getValue() == 0;
    });
    if (!hasZeroResult)
      continue;

    for (unsigned loopDim = 0, e = genericOp.getNumLoops(); loopDim != e; ++loopDim) {
      if (seen.test(loopDim) || map.isFunctionOfDim(loopDim))
        continue;
      seen.set(loopDim);
      allowedDims.push_back(loopDim);
    }
  }

  return allowedDims;
}

static linalg::ControlDropUnitDims makeZeroIndexedUnitDimsOptions() {
  linalg::ControlDropUnitDims options;
  options.controlFn = [](Operation *op) -> SmallVector<unsigned> {
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp)
      return {};
    return getLoopDimsWithZeroIndexedMaps(genericOp);
  };
  return options;
}

} // namespace

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

void LoopFoldZeroIndexedUnitDimsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
LoopFoldZeroIndexedUnitDimsOp::applyToOne(TransformRewriter &rewriter, Operation *target,
                                          ApplyToEachResultList &results, TransformState &state) {
  (void)results;
  (void)state;

  linalg::ControlDropUnitDims options = makeZeroIndexedUnitDimsOptions();

  {
    RewritePatternSet patterns(getContext());
    linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    walkAndApplyPatterns(target, FrozenRewritePatternSet(std::move(patterns)),
                         static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  }
  {
    RewritePatternSet patterns(getContext());
    linalg::populateMoveInitOperandsToInputPattern(patterns);
    linalg::populateFoldUnitExtentDimsCanonicalizationPatterns(patterns, options);
    if (failed(applyPatternsGreedily(target, std::move(patterns)))) {
      return emitDefiniteFailure()
             << "fold-zero-indexed-unit-dims canonicalization did not converge";
    }
  }
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
