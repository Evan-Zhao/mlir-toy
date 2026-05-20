#include "LoopTr/LoopTransformOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::transform {

namespace {

#define BAIL(message) return emitSilenceableFailure(transform, message);
using linalg::GenericOp;

void replaceUsesAfterElementwiseFusion(RewriterBase &rewriter,
                                       const linalg::ElementwiseOpFusionResult &fusionResult,
                                       Operation *producer) {
  for (auto &[origVal, replacement] : fusionResult.replacements) {
    if (origVal.getDefiningOp() != producer)
      rewriter.replaceUsesWithIf(origVal, replacement,
                                 [&](OpOperand &use) { return use.getOwner() != producer; });
  }
}

FailureOr<Operation *> tryDirectElementwiseFusion(RewriterBase &rewriter,
                                                  linalg::LinalgOp linalgTarget,
                                                  size_t operandNumber) {
  OpOperand &fusedOperand = linalgTarget->getOpOperand(operandNumber);
  if (!linalg::areElementwiseOpsFusable(&fusedOperand))
    return failure();

  Operation *producer = fusedOperand.get().getDefiningOp();
  rewriter.setInsertionPoint(linalgTarget);
  FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
      linalg::fuseElementwiseOps(rewriter, &fusedOperand);
  if (failed(fusionResult))
    return failure();

  replaceUsesAfterElementwiseFusion(rewriter, *fusionResult, producer);
  rewriter.eraseOp(linalgTarget);
  return fusionResult->fusedOp;
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

void LoopFoldExpandingReshapeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
LoopFoldExpandingReshapeOp::applyToOne(transform::TransformRewriter &rewriter, Operation *target,
                                       transform::ApplyToEachResultList &results,
                                       transform::TransformState &state) {
  (void)results;
  (void)state;

  RewritePatternSet patterns(getContext());
  linalg::populateFoldReshapeOpsByExpansionPatterns(patterns, [](OpOperand *) { return true; });

  GreedyRewriteConfig config;
  config.setListener(static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);

  bool changed = false;
  if (failed(applyOpPatternsGreedily({target}, FrozenRewritePatternSet(std::move(patterns)), config,
                                     &changed))) {
    return emitDefiniteFailure() << "fold-reshape did not converge";
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure LoopInlineElementwiseOp::applyToOne(TransformRewriter &rewriter,
                                                                Operation *target,
                                                                ApplyToEachResultList &results,
                                                                TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  auto genericOp = dyn_cast<linalg::GenericOp>(target);
  if (!genericOp) {
    target->emitError() << "expected a linalg.generic op as the target";
    BAIL("expected target to be a linalg.generic");
  }

  std::optional<int64_t> operandNumber = getOperandNumber();
  if (operandNumber && (*operandNumber < 0 || *operandNumber >= genericOp->getNumOperands())) {
    genericOp->emitError() << "this operation has " << genericOp->getNumOperands()
                           << " operands, but operand_number is " << *operandNumber;
    BAIL("operand_number is out of range");
  }

  GenericOp currentOp = genericOp;
  auto scanAllOperands = [&]() {
    size_t beginOprndNum = operandNumber ? *operandNumber : 0,
           endOprndNum = operandNumber ? beginOprndNum + 1 : currentOp.getNumOperands();
    bool changed = false;
    for (size_t i = beginOprndNum; i < endOprndNum; ++i) {
      auto folded = tryDirectElementwiseFusion(rewriter, currentOp, i);
      if (succeeded(folded)) {
        currentOp = cast<GenericOp>(*folded);
        changed = true;
      }
    }
    return changed;
  };

  bool changed = false;
  while (scanAllOperands()) {
    changed = true;
  }
  if (!changed)
    BAIL("no eligible elementwise inlining or reshape folding");
  results.push_back(currentOp);
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
