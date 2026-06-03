#include "TA/TATransformOps.h"

#include "TA/TAPasses.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"

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

struct ParsedEinsum {
  SmallVector<SmallVector<std::string>, 2> inputs;
  SmallVector<std::string> result;
};

class TAHandleUpdater : public TransformState::Extension {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TAHandleUpdater)

  explicit TAHandleUpdater(TransformState &state) : TransformState::Extension(state) {}

  LogicalResult replace(Operation *op, Operation *replacement) {
    return replacePayloadOp(op, replacement);
  }
};

SmallVector<std::string> parseAxisList(StringRef text) {
  SmallVector<StringRef> tokens;
  llvm::SplitString(text.trim(), tokens);
  SmallVector<std::string> axes;
  for (StringRef token : tokens)
    axes.push_back(token.str());
  return axes;
}

FailureOr<ParsedEinsum> parseEinsumEquation(StringRef equation) {
  SmallVector<StringRef> sides;
  equation.split(sides, "->");
  if (sides.size() != 2)
    return failure();

  SmallVector<StringRef> inputs;
  sides[0].split(inputs, ",");
  if (inputs.size() != 2)
    return failure();

  ParsedEinsum parsed;
  parsed.inputs.push_back(parseAxisList(inputs[0]));
  parsed.inputs.push_back(parseAxisList(inputs[1]));
  parsed.result = parseAxisList(sides[1]);
  if (parsed.inputs[0].empty() || parsed.inputs[1].empty())
    return failure();
  return parsed;
}

SmallVector<std::string> exprAxisNames(ta::ExprType expr) {
  SmallVector<std::string> names;
  for (Attribute attr : expr.getAxes().getAxes())
    names.push_back(cast<ta::AxisAttr>(attr).getName().getValue().str());
  return names;
}

SmallVector<std::string> axisAttrNames(ta::AxesAttr axes) {
  SmallVector<std::string> names;
  for (Attribute attr : axes.getAxes())
    names.push_back(cast<ta::AxisAttr>(attr).getName().getValue().str());
  return names;
}

bool sameAxisSet(ArrayRef<std::string> lhs, ArrayRef<std::string> rhs) {
  llvm::SmallSetVector<StringRef, 8> lhsSet;
  llvm::SmallSetVector<StringRef, 8> rhsSet;
  for (StringRef axis : lhs)
    lhsSet.insert(axis);
  for (StringRef axis : rhs)
    rhsSet.insert(axis);
  if (lhsSet.size() != rhsSet.size())
    return false;
  for (StringRef axis : lhsSet) {
    if (!rhsSet.contains(axis))
      return false;
  }
  return true;
}

bool matchPatternAxes(ArrayRef<std::string> patternAxes, ArrayRef<std::string> actualAxes,
                      llvm::StringMap<std::string> &axisMapping,
                      SmallVectorImpl<std::string> &usedActualAxes) {
  if (patternAxes.size() != actualAxes.size())
    return false;

  for (auto [patternAxis, actualAxis] : llvm::zip_equal(patternAxes, actualAxes)) {
    auto mapped = axisMapping.find(patternAxis);
    if (mapped != axisMapping.end())
      return mapped->second == actualAxis;

    if (llvm::is_contained(usedActualAxes, actualAxis))
      return false;

    axisMapping[patternAxis] = actualAxis;
    usedActualAxes.push_back(actualAxis);
  }

  return true;
}

SmallVector<std::string> reductionAxesForEquation(const ParsedEinsum &equation) {
  llvm::SmallSetVector<StringRef, 8> resultAxes(equation.result.begin(), equation.result.end());
  llvm::SmallSetVector<StringRef, 8> reductionAxes;
  for (ArrayRef<std::string> input : equation.inputs) {
    for (StringRef axis : input) {
      if (!resultAxes.contains(axis))
        reductionAxes.insert(axis);
    }
  }

  SmallVector<std::string> axes;
  for (StringRef axis : reductionAxes)
    axes.push_back(axis.str());
  return axes;
}

bool matchReductionAxes(ArrayRef<std::string> patternReductionAxes,
                        ArrayRef<std::string> actualReductionAxes,
                        const llvm::StringMap<std::string> &axisMapping) {
  SmallVector<std::string> mappedReductionAxes;
  for (StringRef patternAxis : patternReductionAxes) {
    auto mapped = axisMapping.find(patternAxis);
    if (mapped == axisMapping.end())
      return false;
    mappedReductionAxes.push_back(mapped->second);
  }
  return sameAxisSet(mappedReductionAxes, actualReductionAxes);
}

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

DiagnosedSilenceableFailure TAMatchEinsumOp::matchOperation(Operation *target,
                                                            TransformResults &results,
                                                            TransformState &state) {
  (void)state;
  auto transform = cast<TransformOpInterface>(getOperation());

  auto parsed = parseEinsumEquation(getEquation());
  if (failed(parsed))
    return emitSilenceableFailure(transform, "expected equation like 'a b k, a k c -> a b c'");

  auto reduce = dyn_cast<ta::ReduceOp>(target);
  if (!reduce)
    return emitSilenceableFailure(transform, "expected target to be ta.reduce");
  if (reduce.getKind() != ta::ReduceKind::Add)
    return emitSilenceableFailure(transform, "expected ta.reduce <add>");

  auto mul = reduce.getInput().getDefiningOp<ta::MulFOp>();
  if (!mul)
    return emitSilenceableFailure(transform, "expected reduce payload to be ta.mulf");

  auto lhsType = cast<ta::ExprType>(mul.getLhs().getType());
  auto rhsType = cast<ta::ExprType>(mul.getRhs().getType());
  auto resultType = cast<ta::ExprType>(reduce.getResult().getType());

  llvm::StringMap<std::string> axisMapping;
  SmallVector<std::string> usedActualAxes;
  if (!matchPatternAxes((*parsed).inputs[0], exprAxisNames(lhsType), axisMapping,
                        usedActualAxes))
    return emitSilenceableFailure(transform, "lhs axes do not match einsum structure");
  if (!matchPatternAxes((*parsed).inputs[1], exprAxisNames(rhsType), axisMapping,
                        usedActualAxes))
    return emitSilenceableFailure(transform, "rhs axes do not match einsum structure");
  if (!matchPatternAxes((*parsed).result, exprAxisNames(resultType), axisMapping,
                        usedActualAxes))
    return emitSilenceableFailure(transform, "result axes do not match einsum structure");
  if (!matchReductionAxes(reductionAxesForEquation(*parsed), axisAttrNames(reduce.getAxes()),
                          axisMapping))
    return emitSilenceableFailure(transform, "reduction axes do not match einsum equation");

  results.set(getOperation()->getResult(0), {target});
  return DiagnosedSilenceableFailure::success();
}

void TAToLinalgOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure TAToLinalgOp::apply(TransformRewriter &rewriter,
                                                TransformResults &transformResults,
                                                TransformState &state) {
  (void)transformResults;
  auto transform = cast<TransformOpInterface>(getOperation());

  SmallVector<Operation *> targets = llvm::to_vector(state.getPayloadOps(getTarget()));
  if (targets.empty())
    return emitSilenceableFailure(transform, "expected at least one target op");

  TAHandleUpdater *handleUpdater = state.getExtension<TAHandleUpdater>();
  if (!handleUpdater)
    handleUpdater = &state.addExtension<TAHandleUpdater>();

  for (Operation *target : targets) {
    DenseMap<Operation *, Operation *> loweredOps;
    auto updateTransformHandles = [&](Operation *scope,
                                      const DenseMap<Operation *, Operation *> &scopeLoweredOps) {
      for (auto [taOp, linalgOp] : scopeLoweredOps)
        (void)handleUpdater->replace(taOp, linalgOp);

      scope->walk([&](Operation *nested) {
        if (nested == scope || scopeLoweredOps.contains(nested))
          return;
        if (nested->getName().getDialectNamespace() != "ta")
          return;
        (void)handleUpdater->replace(nested, nullptr);
      });

      if (isa<ta::ScopeOp>(scope))
        (void)handleUpdater->replace(scope, nullptr);
    };

    if (failed(ta::lowerTAToLinalg(target, rewriter, &loweredOps, updateTransformHandles)))
      return emitSilenceableFailure(transform, "failed to lower ta to linalg");
  }

  return DiagnosedSilenceableFailure::success();
}

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

DiagnosedSilenceableFailure TARewriteExpToExp2Op::applyToOne(TransformRewriter &rewriter,
                                                             Operation *target,
                                                             ApplyToEachResultList &results,
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
  registry.addExtension(+[](mlir::MLIRContext *ctx, mlir::transform::TransformDialect *dialect) {
    ctx->loadDialect<ta::TADialect>();
    struct TransformDialectAccess : public mlir::transform::TransformDialect {
      using mlir::Dialect::addOperations;
    };
    static_cast<TransformDialectAccess *>(dialect)
        ->addOperations<mlir::transform::TAMatchEinsumOp, mlir::transform::TAToLinalgOp,
                        mlir::transform::TAExchangeDivAndMatmulPatternsOp,
                        mlir::transform::TAExpToExp2PatternsOp,
                        mlir::transform::TARewriteExpToExp2Op>();
  });
}

} // namespace ta

#define GET_OP_CLASSES
#include "TATransformOps.cpp.inc"
