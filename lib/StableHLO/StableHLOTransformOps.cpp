#include "StableHLO/StableHLOTransformOps.h"

#include "StableHLO/StableHLOLegalizeControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "stablehlo/conversions/linalg/transforms/Rewriters.h"
#include "stablehlo/conversions/linalg/transforms/TypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace mlir::transform {

void StablehloLegalizeControlFlowOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
StablehloLegalizeControlFlowOp::applyToOne(TransformRewriter &rewriter, Operation *target,
                                           ApplyToEachResultList &results, TransformState &state) {
  (void)results;
  (void)state;
  auto *listener = static_cast<RewriterBase::Listener *>(rewriter.getListener());
  if (failed(neptune::stablehlo::legalizeControlFlow(target, listener)))
    return emitSilenceableError() << "failed to legalize StableHLO control flow";
  return DiagnosedSilenceableFailure::success();
}

namespace {

FailureOr<std::pair<Value, int64_t>> getValueAndConstant(Value lhs, Value rhs) {
  APInt constant;
  Value value;
  if (matchPattern(rhs, m_ConstantInt(&constant))) {
    value = lhs;
  } else if (matchPattern(lhs, m_ConstantInt(&constant))) {
    value = rhs;
  } else {
    return failure();
  }

  if (matchPattern(value, m_Constant()) || !constant.isSignedIntN(64))
    return failure();
  return std::make_pair(value, constant.getSExtValue());
}

FailureOr<int64_t> computeConstantBound(Value value, presburger::BoundType boundType) {
  // StableHLO-to-Linalg casts scalar integer slice indices to index. A
  // widening index cast preserves the signed range of its input.
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>();
      cast && value.getType().isIndex() && cast.getIn().getType().isInteger() &&
      DataLayout::closest(cast).getTypeSizeInBits(value.getType()).getFixedValue() >=
          cast.getIn().getType().getIntOrFloatBitWidth())
    value = cast.getIn();

  bool allowIntegerType = value.getType().isInteger();
  if (allowIntegerType) {
    auto blockArg = dyn_cast<BlockArgument>(value);
    auto forOp = blockArg ? dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp()) : nullptr;
    if (!forOp || forOp.getInductionVar() != value)
      return failure();
  }

  ValueBoundsOptions options;
  options.closedUB = true;
  options.allowIntegerType = allowIntegerType;
  return ValueBoundsConstraintSet::computeConstantBound(boundType,
                                                        ValueBoundsConstraintSet::Variable(value),
                                                        /*stopCondition=*/nullptr, options);
}

struct SimplifyBoundedIndexCastRoundTripPattern : OpRewritePattern<arith::IndexCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp op, PatternRewriter &rewriter) const override {
    if (!op.getType().isIndex())
      return failure();
    auto narrow = op.getIn().getDefiningOp<arith::IndexCastOp>();
    if (!narrow || !narrow.getIn().getType().isIndex())
      return failure();
    auto narrowType = dyn_cast<IntegerType>(narrow.getType());
    if (!narrowType || narrowType.getWidth() > 64)
      return failure();

    FailureOr<int64_t> lowerBound = computeConstantBound(narrow.getIn(), presburger::BoundType::LB);
    FailureOr<int64_t> upperBound = computeConstantBound(narrow.getIn(), presburger::BoundType::UB);
    if (failed(lowerBound) || failed(upperBound))
      return failure();

    APInt signedMin = APInt::getSignedMinValue(narrowType.getWidth()).sextOrTrunc(64);
    APInt signedMax = APInt::getSignedMaxValue(narrowType.getWidth()).sextOrTrunc(64);
    if (*lowerBound < signedMin.getSExtValue() || *upperBound > signedMax.getSExtValue())
      return failure();

    rewriter.replaceOp(op, narrow.getIn());
    return success();
  }
};

struct SimplifyBoundedMaxSIPattern : OpRewritePattern<arith::MaxSIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MaxSIOp op, PatternRewriter &rewriter) const override {
    FailureOr<std::pair<Value, int64_t>> match = getValueAndConstant(op.getLhs(), op.getRhs());
    if (failed(match))
      return failure();

    FailureOr<int64_t> lowerBound = computeConstantBound(match->first, presburger::BoundType::LB);
    if (failed(lowerBound) || *lowerBound < match->second)
      return failure();

    rewriter.replaceOp(op, match->first);
    return success();
  }
};

struct SimplifyBoundedMinSIPattern : OpRewritePattern<arith::MinSIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MinSIOp op, PatternRewriter &rewriter) const override {
    FailureOr<std::pair<Value, int64_t>> match = getValueAndConstant(op.getLhs(), op.getRhs());
    if (failed(match))
      return failure();

    FailureOr<int64_t> upperBound = computeConstantBound(match->first, presburger::BoundType::UB);
    if (failed(upperBound) || *upperBound > match->second)
      return failure();

    rewriter.replaceOp(op, match->first);
    return success();
  }
};

void appendStablehloConversionPatternsForRoot(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns, StringRef rootName) {
  RewritePatternSet stablehloPatterns(patterns.getContext());
  stablehlo::populateStablehloToLinalgConversionPatterns(
      patterns.getContext(), typeConverter, &stablehloPatterns,
      /*enablePrimitiveOps=*/false, /*enableSparseOps=*/false,
      /*captureScalarInputs=*/true);
  for (std::unique_ptr<RewritePattern> &pattern : stablehloPatterns.getNativePatterns()) {
    std::optional<OperationName> root = pattern->getRootKind();
    if (root && root->getStringRef() == rootName)
      patterns.getNativePatterns().push_back(std::move(pattern));
  }
}

} // namespace

void StablehloSimplifyInBoundsClampsPatternsOp::populatePatterns(RewritePatternSet &patterns) {
  patterns.add<SimplifyBoundedIndexCastRoundTripPattern, SimplifyBoundedMaxSIPattern,
               SimplifyBoundedMinSIPattern>(patterns.getContext());
}

void StablehloSliceToTensorConversionPatternsOp::populatePatterns(TypeConverter &typeConverter,
                                                                  RewritePatternSet &patterns) {
  appendStablehloConversionPatternsForRoot(typeConverter, patterns,
                                           stablehlo::SliceOp::getOperationName());
}

std::unique_ptr<TypeConverter> StablehloSliceToTensorConversionPatternsOp::getTypeConverter() {
  return std::make_unique<stablehlo::LinalgTypeConverter>();
}

} // namespace mlir::transform

#define GET_OP_CLASSES
#include "StableHLOTransformOps.cpp.inc"
