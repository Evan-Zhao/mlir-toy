#include "TA/TATransformOps.h"

#include "TA/TAPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace ta_mul_scale_motion_pdl {
using namespace mlir;

#include "MulScaleMotion.cpp.inc"
} // namespace ta_mul_scale_motion_pdl

namespace ta_exp_to_exp2_pdl {
using namespace mlir;
using llvm::APFloat;
#include "ExpToExp2.cpp.inc"
} // namespace ta_exp_to_exp2_pdl

namespace mlir::transform {

namespace {

class TAHandleUpdater : public TransformState::Extension {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TAHandleUpdater)

  explicit TAHandleUpdater(TransformState &state) : TransformState::Extension(state) {}

  LogicalResult replace(Operation *op, Operation *replacement) {
    return replacePayloadOp(op, replacement);
  }
};

using AxisNames = SmallVector<std::string>;
using AxisNamesRef = ArrayRef<std::string>;

struct ParsedEinsum {
  AxisNames lhs, rhs, result;
};

AxisNames parseAxisList(StringRef text) {
  SmallVector<StringRef> tokens;
  llvm::SplitString(text.trim(), tokens);
  return llvm::map_to_vector(tokens, [](StringRef token) { return token.str(); });
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

  auto lhs = parseAxisList(inputs[0]), rhs = parseAxisList(inputs[1]),
       result = parseAxisList(sides[1]);
  if (lhs.empty() || rhs.empty() || result.empty())
    return failure();
  return ParsedEinsum{std::move(lhs), std::move(rhs), std::move(result)};
}

FailureOr<AxisNames> expandAxisList(AxisNamesRef patternAxes, AxisNamesRef actualAxes,
                                    std::optional<size_t> &ellipsisRank) {
  size_t ellipsisCount = llvm::count(patternAxes, "...");
  if (ellipsisCount > 1)
    return failure();
  if (ellipsisCount == 0) {
    if (patternAxes.size() != actualAxes.size())
      return failure();
    return AxisNames(patternAxes.begin(), patternAxes.end());
  }

  size_t fixedRank = patternAxes.size() - 1;
  if (actualAxes.size() < fixedRank)
    return failure();
  size_t rank = actualAxes.size() - fixedRank;
  if (ellipsisRank && *ellipsisRank != rank)
    return failure();
  ellipsisRank = rank;

  AxisNames expanded;
  expanded.reserve(actualAxes.size());
  for (StringRef axis : patternAxes) {
    if (axis != "...") {
      expanded.push_back(axis.str());
      continue;
    }
    for (size_t i = 0; i < rank; ++i)
      expanded.push_back((Twine("\1ta_einsum_ellipsis_") + Twine(i)).str());
  }
  return expanded;
}

FailureOr<ParsedEinsum> expandEinsumEquation(const ParsedEinsum &equation, AxisNamesRef lhsAxes,
                                             AxisNamesRef rhsAxes, AxisNamesRef resultAxes) {
  std::optional<size_t> ellipsisRank;
  auto lhsPattern = expandAxisList(equation.lhs, lhsAxes, ellipsisRank);
  if (failed(lhsPattern))
    return failure();
  auto rhsPattern = expandAxisList(equation.rhs, rhsAxes, ellipsisRank);
  if (failed(rhsPattern))
    return failure();
  auto resultPattern = expandAxisList(equation.result, resultAxes, ellipsisRank);
  if (failed(resultPattern))
    return failure();
  return ParsedEinsum{std::move(*lhsPattern), std::move(*rhsPattern), std::move(*resultPattern)};
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

bool sameAxisEqualityPattern(ArrayRef<std::string> patternAxes, ArrayRef<std::string> actualAxes) {
  if (patternAxes.size() != actualAxes.size())
    return false;

  for (size_t i = 0, e = patternAxes.size(); i < e; ++i)
    for (size_t j = i + 1; j < e; ++j)
      if ((patternAxes[i] == patternAxes[j]) != (actualAxes[i] == actualAxes[j]))
        return false;
  return true;
}

SmallVector<std::string> reductionAxesForEquation(const ParsedEinsum &equation) {
  llvm::SmallSetVector<StringRef, 8> resultAxes(equation.result.begin(), equation.result.end());
  llvm::SmallSetVector<StringRef, 8> reductionAxes;
  for (const AxisNames *input : {&equation.lhs, &equation.rhs}) {
    for (StringRef axis : *input) {
      if (!resultAxes.contains(axis))
        reductionAxes.insert(axis);
    }
  }

  SmallVector<std::string> axes;
  for (StringRef axis : reductionAxes)
    axes.push_back(axis.str());
  return axes;
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

  auto mul = reduce.getInput().getDefiningOp<ta::MulOp>();
  if (!mul)
    return emitSilenceableFailure(transform, "expected reduce payload to be ta.mulf");

  auto lhsType = cast<ta::ExprType>(mul.getLhs().getType());
  auto rhsType = cast<ta::ExprType>(mul.getRhs().getType());
  auto resultType = cast<ta::ExprType>(reduce.getResult().getType());
  auto lhsAxes = exprAxisNames(lhsType);
  auto rhsAxes = exprAxisNames(rhsType);
  auto resultAxes = exprAxisNames(resultType);

  auto expanded = expandEinsumEquation(*parsed, lhsAxes, rhsAxes, resultAxes);
  if (failed(expanded))
    return emitSilenceableFailure(transform, "axes do not match einsum rank structure");

  AxisNames patternAxes;
  patternAxes.append(expanded->lhs);
  patternAxes.append(expanded->rhs);
  patternAxes.append(expanded->result);
  patternAxes.append(reductionAxesForEquation(*expanded));

  AxisNames actualAxes;
  actualAxes.append(lhsAxes);
  actualAxes.append(rhsAxes);
  actualAxes.append(resultAxes);
  actualAxes.append(axisAttrNames(reduce.getAxes()));

  if (!sameAxisEqualityPattern(patternAxes, actualAxes))
    return emitSilenceableFailure(transform, "axes do not match einsum structure");

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

void TASinkDivAfterMatmulPatternsOp::populatePatterns(RewritePatternSet &patterns) {
  patterns.add<ta_mul_scale_motion_pdl::SinkLeftDivThroughF16AfterMatmul>(patterns.getContext());
  patterns.add<ta_mul_scale_motion_pdl::SinkRightDivThroughF16AfterMatmul>(patterns.getContext());
}

void TASinkRightMulAfterMatmulPatternsOp::populatePatterns(RewritePatternSet &patterns) {
  patterns.add<ta_mul_scale_motion_pdl::SinkRightMulWithPreWidenThroughF16AfterMatmul>(
      patterns.getContext());
}

void TAExpToExp2PatternsOp::populatePatterns(RewritePatternSet &patterns) {
  ta_exp_to_exp2_pdl::populateGeneratedPDLLPatterns(patterns);
  ta::MulOp::getCanonicalizationPatterns(patterns, patterns.getContext());
  ta::MaximumOp::getCanonicalizationPatterns(patterns, patterns.getContext());
  ta::MinimumOp::getCanonicalizationPatterns(patterns, patterns.getContext());
}

} // namespace mlir::transform

#define GET_OP_CLASSES
#include "TATransformOps.cpp.inc"
