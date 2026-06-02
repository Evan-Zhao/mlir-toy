#include "TA/TATransformOps.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace ta_exchange_div_and_matmul_pdl {
using namespace mlir;
#include "ExchangeDivAndMatmul.cpp.inc"
} // namespace ta_exchange_div_and_matmul_pdl

namespace ta_exp_to_exp2_pdl {
using namespace mlir;
using llvm::APFloat;
#include "ExpToExp2.cpp.inc"
} // namespace ta_exp_to_exp2_pdl

namespace mlir::transform {

namespace {

LogicalResult rewriteGreedily(TransformRewriter &rewriter, RewritePatternSet patterns,
                              Operation *target) {
  GreedyRewriteConfig config;
  config.setListener(static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  bool cseChanged = false;
  constexpr int64_t maxIterations = 50;
  int64_t iteration = 0;
  do {
    LogicalResult result = failure();
    if (target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      result = applyPatternsGreedily(target, frozenPatterns, config);
    } else {
      SmallVector<Operation *> ops;
      target->walk([&](Operation *nestedOp) {
        if (target != nestedOp)
          ops.push_back(nestedOp);
      });
      result = applyOpPatternsGreedily(ops, frozenPatterns, config);
    }
    if (failed(result))
      return failure();

    DominanceInfo domInfo;
    cseChanged = false;
    eliminateCommonSubExpressions(rewriter, domInfo, target, &cseChanged);
  } while (cseChanged && ++iteration < maxIterations);

  return success(iteration < maxIterations);
}

} // namespace

void TAExchangeDivAndMatmulPatternsOp::populatePatterns(RewritePatternSet &patterns) {
  ta_exchange_div_and_matmul_pdl::populateGeneratedPDLLPatterns(patterns);
}

void TAExpToExp2PatternsOp::populatePatterns(RewritePatternSet &patterns) {
  ta_exp_to_exp2_pdl::populateGeneratedPDLLPatterns(patterns);
}

void TARewriteExpToExp2Op::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure TARewriteExpToExp2Op::applyToOne(
    TransformRewriter &rewriter, Operation *target, ApplyToEachResultList &results,
    TransformState &state) {
  (void)results;
  (void)state;
  RewritePatternSet patterns(getContext());
  ta_exp_to_exp2_pdl::populateGeneratedPDLLPatterns(patterns);

  if (failed(rewriteGreedily(rewriter, std::move(patterns), target))) {
    auto transform = cast<TransformOpInterface>(getOperation());
    return emitSilenceableFailure(transform, "exp-to-exp2 rewrite did not converge");
  }

  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform

namespace ta {

void registerTATransformExtension(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *, mlir::transform::TransformDialect *dialect) {
    struct TransformDialectAccess : public mlir::transform::TransformDialect {
      using mlir::Dialect::addOperations;
    };
    static_cast<TransformDialectAccess *>(dialect)
        ->addOperations<mlir::transform::TAExchangeDivAndMatmulPatternsOp,
                        mlir::transform::TAExpToExp2PatternsOp,
                        mlir::transform::TARewriteExpToExp2Op>();
  });
}

} // namespace ta

#define GET_OP_CLASSES
#include "TATransformOps.cpp.inc"
