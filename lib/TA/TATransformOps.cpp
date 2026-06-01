#include "TA/TATransformOps.h"

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace mlir::transform {
namespace {

#define BAIL(message) return emitSilenceableFailure(transform, message)

bool containsAxis(ta::AxesAttr axes, Attribute axis) {
  return llvm::is_contained(axes.getAxes(), axis);
}

bool intersects(ta::AxesAttr lhs, ta::AxesAttr rhs) {
  return llvm::any_of(lhs.getAxes(), [&](Attribute axis) { return containsAxis(rhs, axis); });
}

ta::AxesAttr unionAxesInScopeOrder(MLIRContext *context, ta::ScopeOp scope, ValueRange values) {
  SmallVector<Attribute> axes;
  for (Attribute scopeAxis : scope.getAxes().getAxes()) {
    bool used = llvm::any_of(values, [&](Value value) {
      auto expr = dyn_cast<ta::ExprType>(value.getType());
      return expr && containsAxis(expr.getAxes(), scopeAxis);
    });
    if (used)
      axes.push_back(scopeAxis);
  }
  return ta::AxesAttr::get(context, ArrayAttr::get(context, axes));
}

ta::AxesAttr subtractAxes(MLIRContext *context, ta::AxesAttr axes, ta::AxesAttr remove) {
  SmallVector<Attribute> kept;
  for (Attribute axis : axes.getAxes()) {
    if (!containsAxis(remove, axis))
      kept.push_back(axis);
  }
  return ta::AxesAttr::get(context, ArrayAttr::get(context, kept));
}

ta::ExprType exprType(Type elementType, ta::AxesAttr axes) {
  return ta::ExprType::get(elementType.getContext(), elementType, axes);
}

} // namespace

void TAExchangeDivAndMatmulOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure TAExchangeDivAndMatmulOp::applyToOne(
    TransformRewriter &rewriter, ta::MapReduceOp target, ApplyToEachResultList &results,
    TransformState &state) {
  (void)state;
  auto transform = cast<TransformOpInterface>(getOperation());

  if (target.getKind() != ta::ReduceKind::Add)
    BAIL("expected a ta.map_reduce <add>");

  uint64_t operandNumber = getOperandNumber();
  if (operandNumber >= 2)
    BAIL("operand_number must select operand 0 or 1 of the payload ta.mulf");

  Block &body = target.getBody().front();
  auto yield = dyn_cast<ta::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    BAIL("expected ta.map_reduce to yield one value");

  auto mul = yield.getValues().front().getDefiningOp<ta::MulFOp>();
  if (!mul)
    BAIL("expected ta.map_reduce payload to yield ta.mulf");
  if (mul->getBlock() != &body)
    BAIL("expected yielded ta.mulf to be inside the ta.map_reduce body");
  if (mul->getNextNode() != body.getTerminator())
    BAIL("expected yielded ta.mulf to be immediately before ta.yield");

  Value selected = mul->getOperand(static_cast<unsigned>(operandNumber));
  auto div = selected.getDefiningOp<ta::DivFOp>();
  if (!div)
    BAIL("expected selected ta.mulf operand to be produced by ta.divf");

  Value numerator = div.getLhs();
  Value divisor = div.getRhs();
  auto numeratorType = dyn_cast<ta::ExprType>(numerator.getType());
  auto divisorType = dyn_cast<ta::ExprType>(divisor.getType());
  if (!numeratorType || !divisorType)
    BAIL("expected ta.divf operands to be ta.expr values");
  if (intersects(divisorType.getAxes(), target.getAxes()))
    BAIL("expected divisor not to depend on reduced axes");

  Value other = mul->getOperand(operandNumber == 0 ? 1 : 0);
  auto otherType = dyn_cast<ta::ExprType>(other.getType());
  if (!otherType)
    BAIL("expected non-division ta.mulf operand to be a ta.expr value");

  auto scope = target->getParentOfType<ta::ScopeOp>();
  if (!scope)
    BAIL("expected ta.map_reduce to be nested in ta.scope");

  MLIRContext *context = target.getContext();
  ta::AxesAttr newPayloadAxes = unionAxesInScopeOrder(context, scope, ValueRange{numerator, other});
  ta::AxesAttr newReductionAxes = subtractAxes(context, newPayloadAxes, target.getAxes());
  auto originalResultType = cast<ta::ExprType>(target.getResult().getType());

  Location loc = target.getLoc();
  rewriter.setInsertionPoint(target);
  auto newReduction = ta::MapReduceOp::create(
      rewriter, loc, exprType(originalResultType.getElementType(), newReductionAxes),
      target.getKindAttr(), target.getIdentity(), target.getAxes());

  Block *newBody = new Block();
  newReduction.getBody().push_back(newBody);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(newBody);
    IRMapping mapping;

    for (Operation &op : body.without_terminator()) {
      if (&op == div.getOperation()) {
        mapping.map(div.getResult(), mapping.lookupOrDefault(numerator));
        continue;
      }
      if (&op == mul.getOperation())
        continue;
      rewriter.clone(op, mapping);
    }

    Value mappedNumerator = mapping.lookupOrDefault(numerator);
    Value mappedOther = mapping.lookupOrDefault(other);
    Value lhs = operandNumber == 0 ? mappedNumerator : mappedOther;
    Value rhs = operandNumber == 0 ? mappedOther : mappedNumerator;
    auto product = ta::MulFOp::create(rewriter, loc, lhs, rhs);
    ta::YieldOp::create(rewriter, loc, product.getResult());
  }

  rewriter.setInsertionPointAfter(newReduction);
  auto division = ta::DivFOp::create(rewriter, loc, newReduction.getResult(), divisor);

  bool divIsInsideTarget = div->getBlock() == &body;
  rewriter.replaceOp(target, division.getResult());
  if (!divIsInsideTarget && div->use_empty())
    rewriter.eraseOp(div);

  results.push_back(newReduction.getOperation());
  results.push_back(division.getOperation());
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
        ->addOperations<mlir::transform::TAExchangeDivAndMatmulOp>();
  });
}

} // namespace ta

#define GET_OP_CLASSES
#include "TATransformOps.cpp.inc"
