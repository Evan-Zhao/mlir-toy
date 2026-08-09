#include "LoopTr/LoopTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::transform {
namespace {

bool canGetScalarFromRankZeroValue(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type ? type.getRank() == 0 : !isa<ShapedType>(value.getType());
}

FailureOr<Value> getScalarFromRankZeroValue(PatternRewriter &rewriter, Location loc, Value value) {
  if (!canGetScalarFromRankZeroValue(value))
    return failure();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return value;
  if (auto fromElements = value.getDefiningOp<tensor::FromElementsOp>();
      fromElements && fromElements.getElements().size() == 1)
    return fromElements.getElements().front();
  return tensor::ExtractOp::create(rewriter, loc, value, ValueRange{}).getResult();
}

struct RewriteRankZeroExtractSliceAsScalar : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    if (extractOp.getResultType().getRank() != 0 ||
        extractOp.getSource().getDefiningOp<tensor::ExpandShapeOp>() ||
        !llvm::all_of(extractOp.getMixedSizes(), isOneInteger))
      return failure();

    SmallVector<Value> indices =
        getValueOrCreateConstantIndexOp(rewriter, extractOp.getLoc(), extractOp.getMixedOffsets());
    Value scalar =
        tensor::ExtractOp::create(rewriter, extractOp.getLoc(), extractOp.getSource(), indices);
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(extractOp, extractOp.getResultType(),
                                                        scalar);
    return success();
  }
};

struct FoldRankReducingExtractOfExpandShape : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = extractOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp || !extractOp.hasUnitStride())
      return failure();
    if (extractOp.getResultType().getRank() >= expandOp.getResultType().getRank() ||
        extractOp.getResultType().getRank() > expandOp.getSrcType().getRank())
      return failure();

    ArrayRef<int64_t> expandedShape = expandOp.getResultType().getShape();
    SmallVector<OpFoldResult> oldOffsets = extractOp.getMixedOffsets();
    SmallVector<OpFoldResult> oldSizes = extractOp.getMixedSizes();
    SmallVector<OpFoldResult> oldStrides = extractOp.getMixedStrides();
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;

    for (ArrayRef<int64_t> group : expandOp.getReassociationIndices()) {
      SmallVector<int64_t> carriedDims;
      for (int64_t expandedDim : group) {
        if (expandedShape[expandedDim] == 1) {
          if (!isZeroInteger(oldOffsets[expandedDim]) || !isOneInteger(oldSizes[expandedDim]) ||
              !isOneInteger(oldStrides[expandedDim]))
            return failure();
          continue;
        }
        carriedDims.push_back(expandedDim);
      }
      if (carriedDims.size() > 1)
        return failure();
      std::optional<int64_t> carriedDim =
          carriedDims.empty() ? std::optional<int64_t>() : carriedDims.front();
      newOffsets.push_back(carriedDim ? oldOffsets[*carriedDim] : rewriter.getIndexAttr(0));
      newSizes.push_back(carriedDim ? oldSizes[*carriedDim] : rewriter.getIndexAttr(1));
      newStrides.push_back(carriedDim ? oldStrides[*carriedDim] : rewriter.getIndexAttr(1));
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        extractOp, extractOp.getResultType(), expandOp.getSrc(), newOffsets, newSizes, newStrides);
    return success();
  }
};

struct ScalarizeRankZeroLinalgInputs : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpOperand *> inputs;
    for (OpOperand *input : genericOp.getDpsInputOperands()) {
      auto type = dyn_cast<RankedTensorType>(input->get().getType());
      if (!type || type.getRank() != 0)
        continue;
      if (!genericOp.getMatchingIndexingMap(input).getResults().empty() ||
          genericOp.getRegion().front().getArgument(input->getOperandNumber()).use_empty())
        continue;
      inputs.push_back(input);
    }
    if (inputs.empty())
      return failure();

    rewriter.setInsertionPoint(genericOp);
    SmallVector<Value> scalars;
    scalars.reserve(inputs.size());
    for (OpOperand *input : inputs) {
      FailureOr<Value> scalar =
          getScalarFromRankZeroValue(rewriter, genericOp.getLoc(), input->get());
      if (failed(scalar))
        return failure();
      scalars.push_back(*scalar);
    }
    rewriter.modifyOpInPlace(genericOp, [&]() {
      Block &body = genericOp.getRegion().front();
      for (auto [input, scalar] : llvm::zip_equal(inputs, scalars))
        body.getArgument(input->getOperandNumber()).replaceAllUsesWith(scalar);
    });
    return success();
  }
};

struct ScalarizeRankZeroLinalgGeneric : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (genericOp.getNumLoops() != 0 || genericOp->getNumResults() != 1)
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(genericOp->getResult(0).getType());
    if (!resultType || resultType.getRank() != 0)
      return failure();

    Block &body = genericOp.getRegion().front();
    if (body.getNumArguments() != genericOp->getNumOperands())
      return failure();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getValues().size() != 1 ||
        !llvm::all_of(genericOp->getOperands(), canGetScalarFromRankZeroValue))
      return failure();

    rewriter.setInsertionPoint(genericOp);
    IRMapping mapping;
    for (auto [argument, operand] : llvm::zip_equal(body.getArguments(), genericOp->getOperands())) {
      FailureOr<Value> scalar = getScalarFromRankZeroValue(rewriter, genericOp.getLoc(), operand);
      if (failed(scalar))
        return failure();
      mapping.map(argument, *scalar);
    }
    for (Operation &bodyOp : body.without_terminator())
      rewriter.clone(bodyOp, mapping);

    Value scalarResult = mapping.lookupOrDefault(yield.getValues().front());
    auto tensorResult = tensor::FromElementsOp::create(rewriter, genericOp.getLoc(), resultType,
                                                       scalarResult);
    rewriter.replaceOp(genericOp, tensorResult.getResult());
    return success();
  }
};

struct ScalarizeRankZeroArithOp : public RewritePattern {
  explicit ScalarizeRankZeroArithOp(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getDialectNamespace() != arith::ArithDialect::getDialectNamespace() ||
        !op->hasTrait<OpTrait::Elementwise>() || op->getNumResults() != 1 ||
        op->getNumOperands() == 0 || op->getNumRegions() != 0 || op->getNumSuccessors() != 0 ||
        !isMemoryEffectFree(op))
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.getRank() != 0 ||
        !llvm::all_of(op->getOperands(), canGetScalarFromRankZeroValue))
      return failure();

    SmallVector<Value> scalarOperands;
    scalarOperands.reserve(op->getNumOperands());
    bool foundTensorOperand = false;
    for (Value operand : op->getOperands()) {
      FailureOr<Value> scalar = getScalarFromRankZeroValue(rewriter, op->getLoc(), operand);
      if (failed(scalar))
        return failure();
      scalarOperands.push_back(*scalar);
      foundTensorOperand |= isa<RankedTensorType>(operand.getType());
    }
    if (!foundTensorOperand)
      return failure();

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(scalarOperands);
    state.addTypes(resultType.getElementType());
    state.addAttributes(op->getAttrs());
    Operation *scalarOp = rewriter.create(state);
    auto tensorResult = tensor::FromElementsOp::create(
        rewriter, op->getLoc(), resultType, scalarOp->getResult(0));
    rewriter.replaceOp(op, tensorResult.getResult());
    return success();
  }
};

} // namespace

void LinalgDetensorizeRankZeroPatternsOp::populatePatterns(RewritePatternSet &patterns) {
  patterns.add<ScalarizeRankZeroLinalgInputs, ScalarizeRankZeroLinalgGeneric,
               ScalarizeRankZeroArithOp, RewriteRankZeroExtractSliceAsScalar,
               FoldRankReducingExtractOfExpandShape>(patterns.getContext());
  linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
  tensor::ExtractOp::getCanonicalizationPatterns(patterns, patterns.getContext());
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
}

} // namespace mlir::transform
