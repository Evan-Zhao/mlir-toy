#include "LoopTr/LoopTransformOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

struct FoldExtractAfterInsertSlicePattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto insertOp = extractOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!insertOp)
      return failure();

    auto isSame = [](OpFoldResult lhs, OpFoldResult rhs) { return lhs == rhs; };
    if (insertOp.getSource().getType() != extractOp.getType() ||
        !insertOp.isSameAs(extractOp, isSame)) {
      return failure();
    }

    rewriter.replaceOp(extractOp, insertOp.getSource());
    return success();
  }
};

struct FoldInsertAfterExtractSlicePattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto extractOp = insertOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp)
      return failure();

    auto isSame = [](OpFoldResult lhs, OpFoldResult rhs) { return lhs == rhs; };
    if (extractOp.getSource() != insertOp.getDest() ||
        !extractOp.isSameAs(insertOp, isSame)) {
      return failure();
    }

    rewriter.replaceOp(insertOp, extractOp.getSource());
    return success();
  }
};

} // namespace

void transform::ApplyLoopTensorSubsetCanonicalizationPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldExtractAfterInsertSlicePattern,
               FoldInsertAfterExtractSlicePattern>(patterns.getContext());
}
