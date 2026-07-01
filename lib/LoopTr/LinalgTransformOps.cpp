#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mlir/Support/LLVM.h>

namespace mlir::transform {

namespace {

#define BAIL(message) return emitSilenceableFailure(transform, message);
using linalg::GenericOp;

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

void LinalgGreedyInlineElementwiseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
LinalgGreedyInlineElementwiseOp::applyToOne(TransformRewriter &rewriter, linalg::GenericOp target,
                                            ApplyToEachResultList &results, TransformState &state) {
  (void)results;
  (void)state;
  auto transform = cast<TransformOpInterface>(getOperation());
  std::optional<int64_t> operandNumber = getOperandNumber();
  if (operandNumber && (*operandNumber < 0 || *operandNumber >= target.getNumDpsInputs())) {
    target.emitError() << "this operation has " << target.getNumDpsInputs()
                       << " DPS input operands, but operand_number is " << *operandNumber;
    BAIL("operand_number is out of range");
  }

  FailureOr<ElementwiseInlineResult> inlineResult =
      greedyInlineElementwiseProducers(rewriter, target, operandNumber);
  if (failed(inlineResult))
    return emitDefiniteFailure() << "failed to update payload tracking after elementwise fusion";
  if (!inlineResult->applied)
    BAIL("no eligible elementwise inlining or reshape folding");
  GenericOp currentOp = cast<GenericOp>(inlineResult->fusedOp);

  RewritePatternSet cleanupPatterns(getContext());
  linalg::populateEraseUnusedOperandsAndResultsPatterns(cleanupPatterns);
  if (failed(rewriteGreedilyWithPatternSet(getContext(), rewriter, cleanupPatterns, currentOp))) {
    BAIL("cleanup patterns did not converge after elementwise fusion");
  }
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
