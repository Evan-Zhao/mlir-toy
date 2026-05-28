#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include <mlir/Support/LLVM.h>

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

FailureOr<Operation *> tryDirectElementwiseFusion(TransformRewriter &rewriter,
                                                  linalg::LinalgOp linalgTarget,
                                                  size_t operandNumber) {
  OpOperand &fusedOperand = linalgTarget->getOpOperand(operandNumber);
  if (!linalg::areElementwiseOpsFusable(&fusedOperand))
    return static_cast<Operation *>(nullptr);

  Operation *producer = fusedOperand.get().getDefiningOp();
  rewriter.setInsertionPoint(linalgTarget);
  FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
      linalg::fuseElementwiseOps(rewriter, &fusedOperand);
  if (failed(fusionResult))
    return static_cast<Operation *>(nullptr);

  replaceUsesAfterElementwiseFusion(rewriter, *fusionResult, producer);
  if (failed(rewriter.notifyPayloadOperationReplaced(linalgTarget, fusionResult->fusedOp)))
    return failure();
  rewriter.eraseOp(linalgTarget);
  return fusionResult->fusedOp;
}

SmallVector<unsigned> getLoopDimsWithZeroIndexedMaps(linalg::GenericOp genericOp) {
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

bool isZeroFill(Value value) {
  auto fill = value.getDefiningOp<linalg::FillOp>();
  if (!fill || fill.getInputs().empty())
    return false;

  Attribute attr;
  if (!matchPattern(fill.getInputs().front(), m_Constant(&attr)))
    return false;
  auto floatAttr = dyn_cast<FloatAttr>(attr);
  return floatAttr && floatAttr.getValue().isZero();
}

FailureOr<uint64_t> matchMatmulLikeReduction(linalg::GenericOp target, unsigned operandIndex) {
  FailureOr<uint64_t> reductionDim = getReductionIteratorIndex(target);
  if (failed(reductionDim))
    return failure();

  FailureOr<BinaryReductionCombinerMatch> combiner =
      matchBinaryReductionCombiner(target, /*resultNumber=*/0, /*emitDiagnostics=*/true);
  if (failed(combiner) || !isa<arith::AddFOp>(combiner->combiner))
    return failure();

  BlockArgument reductionOperandArg = target.getBlock()->getArgument(operandIndex);
  auto mul = combiner->nonAccumulator.getDefiningOp<arith::MulFOp>();
  if (!mul || (mul.getLhs() != reductionOperandArg && mul.getRhs() != reductionOperandArg))
    return failure();

  return *reductionDim;
}

FailureOr<arith::DivFOp> matchElementwiseDivProducer(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1 ||
      generic->getNumResults() != 1 || !generic.isAllParallelLoops() ||
      !generic.hasPureTensorSemantics())
    return failure();
  if (!generic.getIndexingMapsArray().back().isIdentity())
    return failure();

  auto yield = cast<linalg::YieldOp>(generic.getBody()->getTerminator());
  if (yield.getNumOperands() != 1)
    return failure();
  auto div = yield.getOperand(0).getDefiningOp<arith::DivFOp>();
  if (!div)
    return failure();
  if (div.getLhs() != generic.getBlock()->getArgument(0) ||
      div.getRhs() != generic.getBlock()->getArgument(1))
    return failure();
  return div;
}

std::optional<unsigned> findResultPosition(AffineMap map, AffineExpr expr) {
  for (auto [pos, resultExpr] : llvm::enumerate(map.getResults())) {
    if (resultExpr == expr)
      return pos;
  }
  return std::nullopt;
}

FailureOr<AffineMap> buildExchangedDivisorMap(AffineMap reductionOperandMap,
                                              AffineMap reductionOutputMap, AffineMap divisorMap,
                                              unsigned reductionDim) {
  MLIRContext *context = reductionOperandMap.getContext();
  SmallVector<AffineExpr> exchangedResults;
  exchangedResults.reserve(divisorMap.getNumResults());

  for (AffineExpr divisorExpr : divisorMap.getResults()) {
    auto divisorDim = dyn_cast<AffineDimExpr>(divisorExpr);
    if (!divisorDim || divisorDim.getPosition() >= reductionOperandMap.getNumResults())
      return failure();

    AffineExpr reductionDomainExpr = reductionOperandMap.getResult(divisorDim.getPosition());
    if (reductionDomainExpr.isFunctionOfDim(reductionDim))
      return failure();

    std::optional<unsigned> outputPos = findResultPosition(reductionOutputMap, reductionDomainExpr);
    if (!outputPos)
      return failure();
    exchangedResults.push_back(getAffineDimExpr(*outputPos, context));
  }

  return AffineMap::get(reductionOutputMap.getNumResults(), 0, exchangedResults, context);
}

linalg::ControlDropUnitDims makeZeroIndexedUnitDimsOptions() {
  linalg::ControlDropUnitDims options;
  options.controlFn = [](Operation *op) -> SmallVector<unsigned> {
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp)
      return {};
    return getLoopDimsWithZeroIndexedMaps(genericOp);
  };
  return options;
}

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

void ApplyFoldExpandingReshapePatternsOp::populatePatterns(RewritePatternSet &patterns) {
  linalg::populateFoldReshapeOpsByExpansionPatterns(patterns, [](OpOperand *) { return true; });
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
  if (operandNumber && (*operandNumber < 0 || *operandNumber >= target.getNumOperands())) {
    target.emitError() << "this operation has " << target.getNumOperands()
                       << " operands, but operand_number is " << *operandNumber;
    BAIL("operand_number is out of range");
  }

  GenericOp currentOp = target;
  bool applied = false;
  while (true) {
    size_t beginOprndNum = operandNumber ? *operandNumber : 0,
           endOprndNum = operandNumber ? beginOprndNum + 1 : currentOp.getNumOperands();
    bool changed = false;
    for (size_t i = beginOprndNum; i < endOprndNum; ++i) {
      auto folded = tryDirectElementwiseFusion(rewriter, currentOp, i);
      if (failed(folded))
        return emitDefiniteFailure()
               << "failed to update payload tracking after elementwise fusion";
      if (*folded) {
        currentOp = cast<GenericOp>(*folded);
        changed = applied = true;
      }
    }
    if (!changed)
      break;
  }
  if (!applied)
    BAIL("no eligible elementwise inlining or reshape folding");

  RewritePatternSet cleanupPatterns(getContext());
  linalg::populateEraseUnusedOperandsAndResultsPatterns(cleanupPatterns);
  if (failed(rewriteGreedilyWithPatternSet(getContext(), rewriter, cleanupPatterns, currentOp))) {
    BAIL("cleanup patterns did not converge after elementwise fusion");
  }
  return DiagnosedSilenceableFailure::success();
}

void LinalgExchangeDivAndMatmulOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure LinalgExchangeDivAndMatmulOp::applyToOne(TransformRewriter &rewriter,
                                                                     linalg::GenericOp target,
                                                                     ApplyToEachResultList &results,
                                                                     TransformState &state) {
  (void)state;
  auto transform = cast<TransformOpInterface>(getOperation());
  uint64_t operandNumber = getOperandNumber();

  if (target->getNumResults() != 1 || !target.hasPureTensorSemantics())
    BAIL("expected a single-result tensor linalg.generic reduction");
  if (operandNumber >= static_cast<uint64_t>(target.getNumDpsInputs()))
    BAIL("operand_number is out of range for target inputs");
  unsigned operandIndex = static_cast<unsigned>(operandNumber);
  if (!isZeroFill(target.getDpsInitOperand(0)->get()))
    BAIL("expected the reduction init to be a zero linalg.fill");

  FailureOr<uint64_t> reductionDim = matchMatmulLikeReduction(target, operandIndex);
  if (failed(reductionDim))
    BAIL("expected an add/mul matmul-like reduction using the selected operand");

  auto opResult = dyn_cast<OpResult>(target.getDpsInputOperand(operandIndex)->get());
  if (!opResult)
    BAIL("expected the selected operand to be produced by an operation");
  auto producerOp = dyn_cast<linalg::GenericOp>(opResult.getDefiningOp());
  if (!producerOp || failed(matchElementwiseDivProducer(producerOp))) {
    producerOp.emitError() << "when analyzing this producer of operand #" << operandIndex;
    BAIL("expected the selected operand producer to be an elementwise arith.divf generic");
  }

  SmallVector<AffineMap> normalizeMaps = producerOp.getIndexingMapsArray();
  AffineMap numeratorMap = normalizeMaps[0], divisorMap = normalizeMaps[1],
            outputMap = normalizeMaps.back();
  if (!numeratorMap.isIdentity() || !outputMap.isIdentity())
    BAIL("expected numerator and output maps of the division producer to be identity");

  SmallVector<AffineMap> reductionMaps = target.getIndexingMapsArray();
  FailureOr<AffineMap> exchangedDivisorMap = buildExchangedDivisorMap(
      reductionMaps[operandIndex], reductionMaps.back(), divisorMap, *reductionDim);
  if (failed(exchangedDivisorMap))
    BAIL("failed to derive a reduction-output divisor map; divisor may not be invariant");

  auto producerInputs = producerOp.getDpsInputs();
  auto numerator = producerInputs[0], divisor = producerInputs[1];
  SmallVector<Value> newReductionInputs = target.getDpsInputs();
  newReductionInputs[operandIndex] = numerator;

  Location loc = target.getLoc();
  rewriter.setInsertionPoint(target);
  auto newReduction = linalg::GenericOp::create(
      rewriter, loc, target->getResultTypes(), newReductionInputs, target.getDpsInits(),
      reductionMaps, target.getIteratorTypesArray(),
      [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
        IRMapping mapping;
        for (auto [oldArg, newArg] : llvm::zip_equal(target.getBlock()->getArguments(), newArgs))
          mapping.map(oldArg, newArg);
        for (Operation &op : target.getBlock()->without_terminator())
          builder.clone(op, mapping);
        auto oldYield = cast<linalg::YieldOp>(target.getBlock()->getTerminator());
        linalg::YieldOp::create(builder, nestedLoc, mapping.lookup(oldYield.getOperand(0)));
      });

  auto resultType = cast<RankedTensorType>(target->getResult(0).getType());
  MLIRContext *context = target.getContext();
  AffineMap identityMap = AffineMap::getMultiDimIdentityMap(resultType.getRank(), context);
  SmallVector<AffineMap> divisionMaps{identityMap, *exchangedDivisorMap, identityMap};
  SmallVector<utils::IteratorType> divisionIterators(resultType.getRank(),
                                                     utils::IteratorType::parallel);
  auto division = linalg::GenericOp::create(
      rewriter, loc, target->getResultTypes(), ValueRange{newReduction->getResult(0), divisor},
      target.getDpsInits(), divisionMaps, divisionIterators,
      [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
        Value quotient = arith::DivFOp::create(builder, nestedLoc, newArgs[0], newArgs[1]);
        linalg::YieldOp::create(builder, nestedLoc, quotient);
      });

  rewriter.replaceOp(target, division->getResults());
  if (producerOp->use_empty())
    rewriter.eraseOp(producerOp);

  results.push_back(newReduction.getOperation());
  results.push_back(division.getOperation());
  return DiagnosedSilenceableFailure::success();
}

void LinalgFoldZeroIndexedUnitDimsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
LinalgFoldZeroIndexedUnitDimsOp::applyToOne(TransformRewriter &rewriter, Operation *target,
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
