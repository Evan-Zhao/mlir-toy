#include "TA/TADialect.h"
#include "TA/TAAttrs.h"
#include "TA/TAOps.h"
#include "TA/TAPasses.h"
#include "TA/TATransformOps.h"
#include "TA/TATypes.h"
#include "TA/TAUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringSet.h"

#define GET_DIALECT_DEFS
#include "TAOpsDialect.cpp.inc"

namespace ta {

using namespace mlir;

void TADialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TAAttrs.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "TATypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "TAOps.cpp.inc"
      >();
}

static LogicalResult verifyAxisArray(function_ref<InFlightDiagnostic()> emitError, ArrayAttr axes) {
  if (!axes)
    return emitError() << "expected an array attribute of #ta.axis attributes";

  StringSet<> seen;
  for (Attribute attr : axes) {
    auto axis = dyn_cast<AxisAttr>(attr);
    if (!axis)
      return emitError() << "expected axis list element to be a #ta.axis attribute";

    StringRef name = axis.getName().getValue();
    if (!seen.insert(name).second)
      return emitError() << "duplicate axis '" << name << "'";
  }

  return success();
}

static void printAxisNames(AsmPrinter &printer, ArrayAttr axes, StringRef open, StringRef close) {
  printer << open;
  llvm::interleaveComma(
      axes, printer, [&](Attribute attr) { printer << cast<AxisAttr>(attr).getName().getValue(); });
  printer << close;
}

static FailureOr<AxesAttr> parseAxisList(AsmParser &parser, AsmParser::Delimiter delimiter) {
  SmallVector<Attribute> axes;
  if (parser.parseCommaSeparatedList(delimiter, [&]() -> ParseResult {
        StringRef name;
        if (parser.parseKeyword(&name))
          return failure();
        axes.push_back(AxisAttr::get(parser.getContext(), name));
        return success();
      }))
    return failure();

  return AxesAttr::get(parser.getContext(), ArrayAttr::get(parser.getContext(), axes));
}

LogicalResult AxisAttr::verify(function_ref<InFlightDiagnostic()> emitError, StringAttr name) {
  if (!name || name.getValue().empty())
    return emitError() << "axis name must be non-empty";

  return success();
}

Attribute AxesAttr::parse(AsmParser &parser, Type type) {
  (void)type;
  FailureOr<AxesAttr> axes = parseAxisList(parser, AsmParser::Delimiter::LessGreater);
  if (failed(axes))
    return {};
  return *axes;
}

void AxesAttr::print(AsmPrinter &printer) const { printAxisNames(printer, getAxes(), "<", ">"); }

LogicalResult AxesAttr::verify(function_ref<InFlightDiagnostic()> emitError, ArrayAttr axes) {
  return verifyAxisArray(emitError, axes);
}

Type ExprType::parse(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  Type elementType;

  if (parser.parseLess() || parser.parseType(elementType) || parser.parseComma())
    return {};

  FailureOr<AxesAttr> axes = parseAxisList(parser, AsmParser::Delimiter::Square);
  if (failed(axes) || parser.parseGreater())
    return {};

  return parser.getChecked<ExprType>(loc, parser.getContext(), elementType, *axes);
}

void ExprType::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getElementType());
  printer << ", ";
  printAxisNames(printer, getAxes().getAxes(), "[", "]");
  printer << ">";
}

LogicalResult ExprType::verify(function_ref<InFlightDiagnostic()> emitError, Type elementType,
                               AxesAttr axes) {
  if (!elementType)
    return emitError() << "expression element type must be present";
  if (!axes)
    return emitError() << "expression axes must be present";

  return success();
}

static bool axisContains(ArrayAttr axes, AxisAttr axis) {
  return llvm::any_of(
      axes, [&](Attribute attr) { return cast<AxisAttr>(attr).getName() == axis.getName(); });
}

static LogicalResult verifyAxesSubset(Operation *op, AxesAttr scopeAxes, AxesAttr usedAxes,
                                      StringRef what) {
  if (!usedAxes)
    return success();

  ArrayAttr allowed = scopeAxes.getAxes();
  for (Attribute attr : usedAxes.getAxes()) {
    AxisAttr axis = cast<AxisAttr>(attr);
    if (!axisContains(allowed, axis)) {
      return op->emitOpError() << what << " uses axis '" << axis.getName().getValue()
                               << "' outside enclosing ta.scope axes";
    }
  }
  return success();
}

static FailureOr<ScopeOp> verifyInsideScope(Operation *op) {
  ScopeOp scope = op->getParentOfType<ScopeOp>();
  if (!scope)
    return op->emitOpError("must be nested inside a ta.scope");
  return scope;
}

static LogicalResult verifyExprAxes(Operation *op, ScopeOp scope, Type type, StringRef what) {
  if (auto expr = dyn_cast<ExprType>(type))
    return verifyAxesSubset(op, scope.getAxes(), expr.getAxes(), what);
  return success();
}

static bool sameAxes(AxesAttr lhs, AxesAttr rhs) {
  ArrayAttr lhsAxes = lhs.getAxes();
  ArrayAttr rhsAxes = rhs.getAxes();
  if (lhsAxes.size() != rhsAxes.size())
    return false;

  for (auto [lhsAttr, rhsAttr] : llvm::zip_equal(lhsAxes, rhsAxes)) {
    AxisAttr lhsAxis = cast<AxisAttr>(lhsAttr);
    AxisAttr rhsAxis = cast<AxisAttr>(rhsAttr);
    if (lhsAxis.getName() != rhsAxis.getName())
      return false;
  }

  return true;
}

static AxesAttr inferOrderedUnionAxes(MLIRContext *context, ValueRange operands) {
  llvm::StringSet<> seen;
  SmallVector<Attribute> inferred;
  for (Value operand : operands) {
    auto expr = cast<ExprType>(operand.getType());
    for (Attribute attr : expr.getAxes().getAxes()) {
      AxisAttr axis = cast<AxisAttr>(attr);
      if (!seen.insert(axis.getName().getValue()).second)
        continue;
      inferred.push_back(attr);
    }
  }

  return AxesAttr::get(context, ArrayAttr::get(context, inferred));
}

static LogicalResult emitInferError(std::optional<Location> location, StringRef message) {
  if (location)
    emitError(*location) << message;
  return failure();
}

static FailureOr<AxesAttr> inferUnionAxesFromOperands(MLIRContext *context,
                                                      std::optional<Location> location,
                                                      ValueRange operands) {
  if (operands.empty())
    return emitInferError(location, "cannot infer expression axes without operands");

  for (Value operand : operands) {
    auto expr = dyn_cast<ExprType>(operand.getType());
    if (!expr)
      return emitInferError(location, "expected ta.expr operands for type inference");
  }

  return inferOrderedUnionAxes(context, operands);
}

static FailureOr<AxesAttr> inferSelectAxes(MLIRContext *context, std::optional<Location> location,
                                           Value condition, Value trueValue, Value falseValue) {
  if (!isa<ExprType>(condition.getType()) || !isa<ExprType>(trueValue.getType()) ||
      !isa<ExprType>(falseValue.getType()))
    return emitInferError(location, "expected ta.expr operands for select type inference");

  // The selected value determines the result layout. A condition may be broadcast over
  // extra axes, but it should not reorder value axes.
  return inferOrderedUnionAxes(context, ValueRange{trueValue, falseValue, condition});
}

static LogicalResult inferSameElementwiseReturnTypes(MLIRContext *context,
                                                     std::optional<Location> location,
                                                     ValueRange operands,
                                                     SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty())
    return emitInferError(location, "expected at least one operand for elementwise type inference");

  auto first = dyn_cast<ExprType>(operands.front().getType());
  if (!first)
    return emitInferError(location, "expected ta.expr operands for elementwise type inference");

  for (Value operand : operands) {
    auto expr = dyn_cast<ExprType>(operand.getType());
    if (!expr)
      return emitInferError(location, "expected ta.expr operands for elementwise type inference");
    if (expr.getElementType() != first.getElementType())
      return emitInferError(location, "expected matching operand element types for type inference");
  }

  FailureOr<AxesAttr> axes = inferUnionAxesFromOperands(context, location, operands);
  if (failed(axes))
    return failure();
  inferredReturnTypes.push_back(ExprType::get(context, first.getElementType(), *axes));
  return success();
}

static AxesAttr subtractAxes(MLIRContext *context, AxesAttr source, AxesAttr removed) {
  StringSet<> removedNames;
  for (Attribute attr : removed.getAxes()) {
    AxisAttr axis = cast<AxisAttr>(attr);
    removedNames.insert(axis.getName().getValue());
  }

  SmallVector<Attribute> kept;
  for (Attribute attr : source.getAxes()) {
    AxisAttr axis = cast<AxisAttr>(attr);
    if (!removedNames.contains(axis.getName().getValue()))
      kept.push_back(attr);
  }

  return AxesAttr::get(context, ArrayAttr::get(context, kept));
}

struct ScopeAxisExtent {
  int64_t staticExtent = ShapedType::kDynamic;
  Value dynamicExtent;
};

static std::optional<ScopeAxisExtent> getScopeAxisExtent(ScopeOp scope, AxisAttr axis) {
  unsigned dynamicIndex = 0;
  for (auto [scopeAxisAttr, staticExtent] :
       llvm::zip_equal(scope.getAxes().getAxes(), scope.getStaticExtents())) {
    Value dynamicExtent;
    if (staticExtent == ShapedType::kDynamic)
      dynamicExtent = scope.getDynamicExtents()[dynamicIndex++];

    AxisAttr scopeAxis = cast<AxisAttr>(scopeAxisAttr);
    if (scopeAxis.getName() == axis.getName())
      return ScopeAxisExtent{.staticExtent = staticExtent, .dynamicExtent = dynamicExtent};
  }
  return std::nullopt;
}

static FailureOr<AxesAttr> substAxes(MLIRContext *context, AxesAttr source, AxesAttr from,
                                     AxesAttr to, function_ref<InFlightDiagnostic()> emitError) {
  ArrayAttr fromArray = from.getAxes();
  ArrayAttr toArray = to.getAxes();
  if (fromArray.size() != toArray.size())
    return emitError() << "expected the same number of source and target axes";

  DenseMap<StringAttr, Attribute> substitutions;
  for (auto [fromAttr, toAttr] : llvm::zip_equal(fromArray, toArray)) {
    AxisAttr fromAxis = cast<AxisAttr>(fromAttr);
    substitutions[fromAxis.getName()] = toAttr;
  }

  StringSet<> seen;
  SmallVector<Attribute> substituted;
  for (Attribute attr : source.getAxes()) {
    AxisAttr axis = cast<AxisAttr>(attr);
    Attribute replacement = substitutions.lookup(axis.getName());
    Attribute resultAttr = replacement ? replacement : attr;
    StringRef resultName = cast<AxisAttr>(resultAttr).getName().getValue();
    if (!seen.insert(resultName).second)
      return emitError() << "substitution produces duplicate axis '" << resultName << "'";
    substituted.push_back(resultAttr);
  }

  return AxesAttr::get(context, ArrayAttr::get(context, substituted));
}

static LogicalResult verifyElementwiseAxes(Operation *op, ScopeOp scope) {
  for (Value operand : op->getOperands()) {
    if (failed(verifyExprAxes(op, scope, operand.getType(), "operand")))
      return failure();
  }

  if (failed(verifyExprAxes(op, scope, op->getResult(0).getType(), "result")))
    return failure();

  auto result = cast<ExprType>(op->getResult(0).getType());
  AxesAttr expected = inferOrderedUnionAxes(op->getContext(), op->getOperands());
  if (!sameAxes(result.getAxes(), expected))
    return op->emitOpError()
           << "result axes must be the ordered union of operand axes; expected " << expected;

  return success();
}

static LogicalResult verifyFloatElementwiseOp(Operation *op) {
  auto scopeOr = verifyInsideScope(op);
  if (failed(scopeOr))
    return failure();

  if (failed(verifyElementwiseAxes(op, *scopeOr)))
    return failure();

  auto result = cast<ExprType>(op->getResult(0).getType());
  Type elementType = result.getElementType();
  if (!isa<FloatType>(elementType))
    return op->emitOpError("requires a floating-point expression result");

  for (Value operand : op->getOperands()) {
    auto expr = cast<ExprType>(operand.getType());
    if (expr.getElementType() != elementType)
      return op->emitOpError("requires all operand and result element types to match");
  }

  return success();
}

static LogicalResult verifyUnaryFloatElementwiseOp(Operation *op) {
  if (op->getNumOperands() != 1)
    return op->emitOpError("expected one operand");
  return verifyFloatElementwiseOp(op);
}

static LogicalResult verifyBinaryFloatElementwiseOp(Operation *op) {
  if (op->getNumOperands() != 2)
    return op->emitOpError("expected two operands");
  return verifyFloatElementwiseOp(op);
}

static LogicalResult verifyTernaryFloatElementwiseOp(Operation *op) {
  if (op->getNumOperands() != 3)
    return op->emitOpError("expected three operands");
  return verifyFloatElementwiseOp(op);
}

static LogicalResult verifyFloatCastElementwiseOp(Operation *op, bool widening) {
  if (op->getNumOperands() != 1)
    return op->emitOpError("expected one operand");

  auto scopeOr = verifyInsideScope(op);
  if (failed(scopeOr))
    return failure();
  if (failed(verifyElementwiseAxes(op, *scopeOr)))
    return failure();

  auto operand = cast<ExprType>(op->getOperand(0).getType());
  auto result = cast<ExprType>(op->getResult(0).getType());
  auto operandElement = dyn_cast<FloatType>(operand.getElementType());
  auto resultElement = dyn_cast<FloatType>(result.getElementType());
  if (!operandElement || !resultElement)
    return op->emitOpError("requires floating-point operand and result element types");

  unsigned operandWidth = operandElement.getWidth();
  unsigned resultWidth = resultElement.getWidth();
  if (widening && resultWidth <= operandWidth)
    return op->emitOpError("result element type must be wider than operand element type");
  if (!widening && resultWidth >= operandWidth)
    return op->emitOpError("result element type must be narrower than operand element type");

  return success();
}

LogicalResult MapOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                      Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  FailureOr<AxesAttr> axes = inferUnionAxesFromOperands(context, location, adaptor.getInputs());
  if (failed(axes))
    return failure();

  if (adaptor.getBody().empty())
    return emitInferError(location, "expected ta.map body for type inference");
  Block &block = adaptor.getBody().front();
  auto yield = dyn_cast_or_null<YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return emitInferError(location, "expected ta.map body to yield one value for type inference");

  inferredReturnTypes.push_back(ExprType::get(context, yield.getValues().front().getType(), *axes));
  return success();
}

LogicalResult ConstantOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                           Adaptor adaptor,
                                           SmallVectorImpl<Type> &inferredReturnTypes) {
  Attribute value = adaptor.getValue();
  if (!value)
    return emitInferError(location, "expected constant value for type inference");
  auto typedValue = dyn_cast<TypedAttr>(value);
  if (!typedValue)
    return emitInferError(location, "expected typed constant value for type inference");
  Type elementType = typedValue.getType();
  if (!elementType)
    return emitInferError(location, "expected typed constant value for type inference");

  inferredReturnTypes.push_back(
      ExprType::get(context, elementType, AxesAttr::get(context, ArrayAttr::get(context, {}))));
  return success();
}

#define DEFINE_TA_SAME_ELEMENTWISE_INFER(OP)                                                       \
  LogicalResult OP::inferReturnTypes(MLIRContext *context, std::optional<Location> location,       \
                                     Adaptor adaptor,                                              \
                                     SmallVectorImpl<Type> &inferredReturnTypes) {                 \
    return inferSameElementwiseReturnTypes(context, location, adaptor.getOperands(),               \
                                           inferredReturnTypes);                                   \
  }

DEFINE_TA_SAME_ELEMENTWISE_INFER(NegFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(AddFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(SubFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(MulFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(DivFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(MaximumFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(MinimumFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(MaxNumFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(MinNumFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(AbsFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(CeilOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(ExpOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(Exp2Op)
DEFINE_TA_SAME_ELEMENTWISE_INFER(FloorOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(LogOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(Log2Op)
DEFINE_TA_SAME_ELEMENTWISE_INFER(RsqrtOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(SqrtOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(TanhOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(PowFOp)
DEFINE_TA_SAME_ELEMENTWISE_INFER(FmaOp)

#undef DEFINE_TA_SAME_ELEMENTWISE_INFER

LogicalResult CmpFOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                       Adaptor adaptor,
                                       SmallVectorImpl<Type> &inferredReturnTypes) {
  FailureOr<AxesAttr> axes = inferUnionAxesFromOperands(context, location, adaptor.getOperands());
  if (failed(axes))
    return failure();
  inferredReturnTypes.push_back(ExprType::get(context, IntegerType::get(context, 1), *axes));
  return success();
}

LogicalResult CmpIOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                       Adaptor adaptor,
                                       SmallVectorImpl<Type> &inferredReturnTypes) {
  FailureOr<AxesAttr> axes = inferUnionAxesFromOperands(context, location, adaptor.getOperands());
  if (failed(axes))
    return failure();
  inferredReturnTypes.push_back(ExprType::get(context, IntegerType::get(context, 1), *axes));
  return success();
}

LogicalResult SelectOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type> &inferredReturnTypes) {
  auto trueValue = dyn_cast<ExprType>(adaptor.getTrueValue().getType());
  auto falseValue = dyn_cast<ExprType>(adaptor.getFalseValue().getType());
  if (!trueValue || !falseValue)
    return emitInferError(location, "expected ta.expr select values for type inference");
  if (trueValue.getElementType() != falseValue.getElementType())
    return emitInferError(location, "expected matching select value element types");

  FailureOr<AxesAttr> axes =
      inferSelectAxes(context, location, adaptor.getCondition(), adaptor.getTrueValue(),
                      adaptor.getFalseValue());
  if (failed(axes))
    return failure();
  inferredReturnTypes.push_back(ExprType::get(context, trueValue.getElementType(), *axes));
  return success();
}

LogicalResult ReduceOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type> &inferredReturnTypes) {
  auto payload = dyn_cast<ExprType>(adaptor.getInput().getType());
  if (!payload)
    return emitInferError(location, "expected ta.reduce input to be a ta.expr value");

  inferredReturnTypes.push_back(
      ExprType::get(context, payload.getElementType(),
                    subtractAxes(context, payload.getAxes(), adaptor.getAxes())));
  return success();
}

LogicalResult SubstOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                        Adaptor adaptor,
                                        SmallVectorImpl<Type> &inferredReturnTypes) {
  auto input = dyn_cast<ExprType>(adaptor.getInput().getType());
  if (!input)
    return emitInferError(location, "expected ta.subst input to be a ta.expr value");
  if (!adaptor.getFromAxes() || !adaptor.getToAxes())
    return emitInferError(location, "expected ta.subst source and target axes");

  auto emitError = [&]() -> InFlightDiagnostic {
    if (location)
      return mlir::emitError(*location);
    return mlir::emitError(UnknownLoc::get(context));
  };
  FailureOr<AxesAttr> axes =
      substAxes(context, input.getAxes(), adaptor.getFromAxes(), adaptor.getToAxes(), emitError);
  if (failed(axes))
    return failure();

  inferredReturnTypes.push_back(ExprType::get(context, input.getElementType(), *axes));
  return success();
}

LogicalResult YieldOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (failed(scopeOr))
    return failure();

  Operation *parent = getOperation()->getParentOp();
  if (auto map = dyn_cast<MapOp>(parent)) {
    if (getValues().size() != 1)
      return emitOpError("terminating ta.map must yield exactly one value");

    auto result = cast<ExprType>(map.getResult().getType());
    if (getValues().front().getType() != result.getElementType())
      return emitOpError("terminating ta.map must yield the map result element type");
  } else if (auto scope = dyn_cast<ScopeOp>(parent)) {
    if (getValues().size() != 1)
      return emitOpError("terminating ta.scope must yield exactly one value");

    auto expr = dyn_cast<ExprType>(getValues().front().getType());
    if (!expr)
      return emitOpError("terminating ta.scope must yield a ta.expr value");

    if (failed(verifyAxesSubset(getOperation(), scope.getAxes(), expr.getAxes(),
                                "yielded expression")))
      return failure();

    auto resultType = cast<RankedTensorType>(scope.getResult().getType());
    if (resultType.getElementType() != expr.getElementType())
      return emitOpError("yielded expression element type must match ta.scope result tensor "
                         "element type");
  }

  return success();
}

LogicalResult AtOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (failed(scopeOr))
    return failure();

  ScopeOp scope = *scopeOr;
  auto result = cast<ExprType>(getResult().getType());
  if (failed(verifyExprAxes(getOperation(), scope, result, "result")))
    return failure();

  FailureOr<AxesAttr> expected =
      inferAxesFromScopeIndexOperands(getOperation(), scope, getIndices());
  if (failed(expected))
    return failure();
  if (!sameAxes(result.getAxes(), *expected))
    return emitOpError() << "result axes must match scope-axis indices; expected " << *expected;

  return success();
}

LogicalResult MapOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (failed(scopeOr))
    return failure();

  ScopeOp scope = *scopeOr;
  if (failed(verifyElementwiseAxes(getOperation(), scope)))
    return failure();

  auto result = cast<ExprType>(getResult().getType());
  Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size())
    return emitOpError("expected one body argument per input");

  for (auto [input, arg] : zip_equal(getInputs(), block.getArguments())) {
    auto expr = cast<ExprType>(input.getType());
    if (arg.getType() != expr.getElementType())
      return emitOpError("body argument types must match input expression element types");
  }

  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("body must terminate with ta.yield");
  if (yield.getValues().size() != 1)
    return emitOpError("body must yield exactly one value");
  if (yield.getValues().front().getType() != result.getElementType())
    return emitOpError("body yield type must match result expression element type");

  return success();
}

LogicalResult ConstantOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (failed(scopeOr))
    return failure();

  auto result = cast<ExprType>(getResult().getType());
  if (!result.getAxes().getAxes().empty())
    return emitOpError("result axes must be empty");
  if (failed(verifyExprAxes(getOperation(), *scopeOr, getResult().getType(), "result")))
    return failure();
  if (getValue().getType() != result.getElementType())
    return emitOpError("value type must match result expression element type");

  return success();
}

#define DEFINE_TA_UNARY_FLOAT_VERIFY(OP)                                                           \
  LogicalResult OP::verify() { return verifyUnaryFloatElementwiseOp(getOperation()); }

#define DEFINE_TA_BINARY_FLOAT_VERIFY(OP)                                                          \
  LogicalResult OP::verify() { return verifyBinaryFloatElementwiseOp(getOperation()); }

#define DEFINE_TA_TERNARY_FLOAT_VERIFY(OP)                                                         \
  LogicalResult OP::verify() { return verifyTernaryFloatElementwiseOp(getOperation()); }

DEFINE_TA_UNARY_FLOAT_VERIFY(NegFOp)
DEFINE_TA_BINARY_FLOAT_VERIFY(AddFOp)
DEFINE_TA_BINARY_FLOAT_VERIFY(SubFOp)
DEFINE_TA_BINARY_FLOAT_VERIFY(MulFOp)
DEFINE_TA_BINARY_FLOAT_VERIFY(DivFOp)
DEFINE_TA_BINARY_FLOAT_VERIFY(MaximumFOp)
DEFINE_TA_BINARY_FLOAT_VERIFY(MinimumFOp)
DEFINE_TA_BINARY_FLOAT_VERIFY(MaxNumFOp)
DEFINE_TA_BINARY_FLOAT_VERIFY(MinNumFOp)
DEFINE_TA_UNARY_FLOAT_VERIFY(AbsFOp)
DEFINE_TA_UNARY_FLOAT_VERIFY(CeilOp)
DEFINE_TA_UNARY_FLOAT_VERIFY(ExpOp)
DEFINE_TA_UNARY_FLOAT_VERIFY(Exp2Op)
DEFINE_TA_UNARY_FLOAT_VERIFY(FloorOp)
DEFINE_TA_UNARY_FLOAT_VERIFY(LogOp)
DEFINE_TA_UNARY_FLOAT_VERIFY(Log2Op)
DEFINE_TA_UNARY_FLOAT_VERIFY(RsqrtOp)
DEFINE_TA_UNARY_FLOAT_VERIFY(SqrtOp)
DEFINE_TA_UNARY_FLOAT_VERIFY(TanhOp)
DEFINE_TA_BINARY_FLOAT_VERIFY(PowFOp)
DEFINE_TA_TERNARY_FLOAT_VERIFY(FmaOp)

#undef DEFINE_TA_UNARY_FLOAT_VERIFY
#undef DEFINE_TA_BINARY_FLOAT_VERIFY
#undef DEFINE_TA_TERNARY_FLOAT_VERIFY

LogicalResult ExtFOp::verify() {
  return verifyFloatCastElementwiseOp(getOperation(), /*widening=*/true);
}

LogicalResult TruncFOp::verify() {
  return verifyFloatCastElementwiseOp(getOperation(), /*widening=*/false);
}

namespace {

struct FoldTruncFOfConstant : OpRewritePattern<TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TruncFOp op, PatternRewriter &rewriter) const override {
    auto constant = op.getOperand().getDefiningOp<ConstantOp>();
    if (!constant)
      return failure();

    auto value = dyn_cast<FloatAttr>(constant.getValue());
    if (!value)
      return failure();

    auto result = cast<ExprType>(op.getResult().getType());
    auto resultElement = dyn_cast<FloatType>(result.getElementType());
    if (!resultElement)
      return failure();

    APFloat rounded = value.getValue();
    bool losesInfo = false;
    APFloat::opStatus status =
        rounded.convert(resultElement.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
    if (status == APFloat::opInvalidOp)
      return failure();

    auto replacement =
        ConstantOp::create(rewriter, op.getLoc(), op.getResult().getType(),
                           FloatAttr::get(resultElement, rounded));
    for (NamedAttribute attr : op->getDiscardableAttrs())
      replacement->setAttr(attr.getName(), attr.getValue());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

} // namespace

void TruncFOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldTruncFOfConstant>(context);
}

LogicalResult CmpFOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (failed(scopeOr))
    return failure();
  if (failed(verifyElementwiseAxes(getOperation(), *scopeOr)))
    return failure();

  auto lhs = cast<ExprType>(getLhs().getType());
  auto rhs = cast<ExprType>(getRhs().getType());
  auto result = cast<ExprType>(getResult().getType());
  if (!isa<FloatType>(lhs.getElementType()))
    return emitOpError("requires floating-point operand element types");
  if (lhs.getElementType() != rhs.getElementType())
    return emitOpError("requires matching operand element types");
  if (!result.getElementType().isInteger(1))
    return emitOpError("result element type must be i1");

  return success();
}

LogicalResult IndexOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (failed(scopeOr))
    return failure();

  auto result = cast<ExprType>(getResult().getType());
  Type elementType = result.getElementType();
  if (!(elementType.isIndex() || elementType.isSignlessInteger()))
    return emitOpError("result element type must be index or signless integer");

  FailureOr<AxesAttr> expected =
      inferAxesFromScopeIndexOperands(getOperation(), *scopeOr, ValueRange{getAxis()});
  if (failed(expected))
    return failure();
  if (!sameAxes(result.getAxes(), *expected))
    return emitOpError() << "result axes must match indexed scope axis; expected " << *expected;

  return success();
}

LogicalResult CmpIOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (failed(scopeOr))
    return failure();
  if (failed(verifyElementwiseAxes(getOperation(), *scopeOr)))
    return failure();

  auto lhs = cast<ExprType>(getLhs().getType());
  auto rhs = cast<ExprType>(getRhs().getType());
  auto result = cast<ExprType>(getResult().getType());
  Type lhsElement = lhs.getElementType();
  Type rhsElement = rhs.getElementType();
  if (!(lhsElement.isIndex() || lhsElement.isSignlessInteger()))
    return emitOpError("requires index or signless integer operand element types");
  if (lhsElement != rhsElement)
    return emitOpError("requires matching operand element types");
  if (!result.getElementType().isInteger(1))
    return emitOpError("result element type must be i1");

  return success();
}

LogicalResult SelectOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (failed(scopeOr))
    return failure();

  auto condition = cast<ExprType>(getCondition().getType());
  auto trueValue = cast<ExprType>(getTrueValue().getType());
  auto falseValue = cast<ExprType>(getFalseValue().getType());
  auto result = cast<ExprType>(getResult().getType());

  for (Value operand : getOperation()->getOperands()) {
    if (failed(verifyExprAxes(getOperation(), *scopeOr, operand.getType(), "operand")))
      return failure();
  }
  if (failed(verifyExprAxes(getOperation(), *scopeOr, getResult().getType(), "result")))
    return failure();

  FailureOr<AxesAttr> expected = inferSelectAxes(
      getContext(), getOperation()->getLoc(), getCondition(), getTrueValue(), getFalseValue());
  if (failed(expected))
    return failure();
  if (!sameAxes(result.getAxes(), *expected))
    return emitOpError()
           << "result axes must be the selected-value ordered union; expected " << *expected;

  if (!condition.getElementType().isInteger(1))
    return emitOpError("condition element type must be i1");
  if (trueValue.getElementType() != falseValue.getElementType() ||
      trueValue.getElementType() != result.getElementType())
    return emitOpError("true, false, and result element types must match");

  return success();
}

static LogicalResult verifyReducePayload(Operation *op, ScopeOp scope, AxesAttr reductionAxes,
                                         ExprType payload, ExprType result, Value identity) {
  if (failed(verifyAxesSubset(op, scope.getAxes(), reductionAxes, "reduction")))
    return failure();
  if (failed(verifyAxesSubset(op, scope.getAxes(), payload.getAxes(), "payload")))
    return failure();
  if (failed(verifyAxesSubset(op, scope.getAxes(), result.getAxes(), "result")))
    return failure();

  if (result.getElementType() != payload.getElementType())
    return op->emitOpError("result element type must match payload element type");

  AxesAttr expected = subtractAxes(op->getContext(), payload.getAxes(), reductionAxes);
  if (!sameAxes(result.getAxes(), expected))
    return op->emitOpError() << "result axes must be payload axes minus reduction axes; expected "
                             << expected;

  if (identity && identity.getType() != result.getElementType())
    return op->emitOpError("identity type must match result expression element type");

  return success();
}

LogicalResult ReduceOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (failed(scopeOr))
    return failure();

  auto payload = cast<ExprType>(getInput().getType());
  auto result = cast<ExprType>(getResult().getType());
  return verifyReducePayload(getOperation(), *scopeOr, getAxes(), payload, result, getIdentity());
}

LogicalResult SubstOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (failed(scopeOr))
    return failure();

  ScopeOp scope = *scopeOr;
  auto input = cast<ExprType>(getInput().getType());
  auto result = cast<ExprType>(getResult().getType());

  if (failed(verifyAxesSubset(getOperation(), scope.getAxes(), input.getAxes(), "input")))
    return failure();
  if (failed(verifyAxesSubset(getOperation(), scope.getAxes(), getFromAxes(), "source")))
    return failure();
  if (failed(verifyAxesSubset(getOperation(), scope.getAxes(), getToAxes(), "target")))
    return failure();
  if (failed(verifyAxesSubset(getOperation(), scope.getAxes(), result.getAxes(), "result")))
    return failure();

  if (input.getElementType() != result.getElementType())
    return emitOpError("result element type must match input element type");

  ArrayAttr inputAxes = input.getAxes().getAxes();
  for (Attribute attr : getFromAxes().getAxes()) {
    AxisAttr axis = cast<AxisAttr>(attr);
    if (!axisContains(inputAxes, axis))
      return emitOpError() << "source axis '" << axis.getName().getValue()
                           << "' is not present in the input axes";
  }

  FailureOr<AxesAttr> expected = substAxes(getContext(), input.getAxes(), getFromAxes(),
                                           getToAxes(), [&]() { return emitOpError(); });
  if (failed(expected))
    return failure();
  if (!sameAxes(result.getAxes(), *expected))
    return emitOpError() << "result axes must be input axes after substitution; expected "
                         << *expected;

  for (auto [fromAttr, toAttr] : llvm::zip_equal(getFromAxes().getAxes(), getToAxes().getAxes())) {
    AxisAttr fromAxis = cast<AxisAttr>(fromAttr);
    AxisAttr toAxis = cast<AxisAttr>(toAttr);
    std::optional<ScopeAxisExtent> fromExtent = getScopeAxisExtent(scope, fromAxis);
    std::optional<ScopeAxisExtent> toExtent = getScopeAxisExtent(scope, toAxis);
    if (!fromExtent || !toExtent)
      return emitOpError("substituted axes must be present in the enclosing scope");

    if (fromExtent->staticExtent != toExtent->staticExtent)
      return emitOpError() << "substituted axes must have equal extents, but axis '"
                           << fromAxis.getName().getValue() << "' has extent "
                           << fromExtent->staticExtent << " and axis '"
                           << toAxis.getName().getValue() << "' has extent "
                           << toExtent->staticExtent;
    if (fromExtent->staticExtent == ShapedType::kDynamic &&
        fromExtent->dynamicExtent != toExtent->dynamicExtent)
      return emitOpError("dynamic substituted axes must use the same extent operand");
  }

  return success();
}

ParseResult ScopeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> axisArgs;
  SmallVector<Attribute> axes;
  SmallVector<int64_t> staticExtents;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicExtents;
  if (parser.parseKeyword("axes") || parser.parseLParen())
    return failure();

  if (parser.parseOptionalRParen()) {
    do {
      OpAsmParser::Argument arg;
      std::string axisName;
      if (parser.parseArgument(arg) || parser.parseString(&axisName) ||
          parser.parseKeyword("extent"))
        return failure();
      arg.type = parser.getBuilder().getIndexType();

      int64_t staticExtent = ShapedType::kDynamic;
      OptionalParseResult parsedInteger = parser.parseOptionalInteger(staticExtent);
      if (parsedInteger.has_value()) {
        if (failed(*parsedInteger))
          return failure();
        if (staticExtent < 0)
          return parser.emitError(parser.getCurrentLocation(),
                                  "expected non-negative static axis extent");
      } else {
        OpAsmParser::UnresolvedOperand dynamicExtent;
        if (parser.parseOperand(dynamicExtent))
          return failure();
        dynamicExtents.push_back(dynamicExtent);
      }

      axisArgs.push_back(arg);
      axes.push_back(AxisAttr::get(parser.getContext(), axisName));
      staticExtents.push_back(staticExtent);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen())
      return failure();
  }

  result.addAttribute(
      getAxesAttrName(result.name),
      AxesAttr::get(parser.getContext(), ArrayAttr::get(parser.getContext(), axes)));
  result.addAttribute(getStaticExtentsAttrName(result.name),
                      parser.getBuilder().getDenseI64ArrayAttr(staticExtents));

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, axisArgs))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<Type> resultTypes;
  if (parser.parseColon() || parser.parseLParen() || parser.parseRParen() ||
      parser.parseArrowTypeList(resultTypes))
    return failure();
  if (resultTypes.size() != 1)
    return parser.emitError(parser.getCurrentLocation(), "expected one result type");
  if (parser.resolveOperands(dynamicExtents, parser.getBuilder().getIndexType(),
                             parser.getCurrentLocation(), result.operands))
    return failure();

  result.addTypes(resultTypes);
  return success();
}

void ScopeOp::print(OpAsmPrinter &printer) {
  printer << " axes(";
  Block &block = getBody().front();
  ArrayAttr axes = getAxes().getAxes();
  ArrayRef<int64_t> staticExtents = getStaticExtents();
  OperandRange dynamicExtents = getDynamicExtents();
  unsigned dynamicIndex = 0;
  interleaveComma(llvm::seq<unsigned>(0, block.getNumArguments()), printer, [&](unsigned i) {
    BlockArgument arg = block.getArgument(i);
    printer.printOperand(arg);
    printer << " \"";
    printer << cast<AxisAttr>(axes[i]).getName().getValue();
    printer << "\" extent ";
    if (staticExtents[i] == ShapedType::kDynamic)
      printer.printOperand(dynamicExtents[dynamicIndex++]);
    else
      printer << staticExtents[i];
  });
  printer << ") ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDict((*this)->getAttrs(), {"axes", "static_extents"});
  printer << " : () -> ";
  printer.printType(getResult().getType());
}

LogicalResult ScopeOp::verify() {
  Block &block = getBody().front();
  if (block.getNumArguments() != getAxes().getAxes().size())
    return emitOpError("expected one region argument per axis");
  if (getStaticExtents().size() != getAxes().getAxes().size())
    return emitOpError("expected one static extent entry per axis");

  unsigned dynamicCount = 0;
  for (int64_t extent : getStaticExtents()) {
    if (extent == ShapedType::kDynamic) {
      ++dynamicCount;
      continue;
    }
    if (extent < 0)
      return emitOpError("static axis extents must be non-negative or dynamic");
  }
  if (dynamicCount != getDynamicExtents().size())
    return emitOpError("expected one dynamic extent operand per dynamic static extent entry");

  for (BlockArgument arg : block.getArguments()) {
    if (!arg.getType().isIndex())
      return emitOpError("expected axis region arguments to have index type");
  }

  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("body must terminate with ta.yield");

  if (yield.getValues().size() != 1)
    return emitOpError("body must yield exactly one value");

  auto expr = dyn_cast<ExprType>(yield.getValues().front().getType());
  if (!expr)
    return emitOpError("body must yield a ta.expr value");

  auto resultType = cast<RankedTensorType>(getResult().getType());
  if (resultType.getElementType() != expr.getElementType())
    return emitOpError("result tensor element type must match yielded expression element type");

  if (failed(verifyAxesSubset(getOperation(), getAxes(), expr.getAxes(), "yielded expression")))
    return failure();

  if (resultType.getRank() == static_cast<int64_t>(expr.getAxes().getAxes().size())) {
    for (auto [dim, axisAttr] : llvm::zip_equal(resultType.getShape(), expr.getAxes().getAxes())) {
      if (dim == ShapedType::kDynamic)
        continue;
      AxisAttr axis = cast<AxisAttr>(axisAttr);
      std::optional<int64_t> extent;
      for (auto [scopeAxis, scopeExtent] :
           llvm::zip_equal(getAxes().getAxes(), getStaticExtents())) {
        if (cast<AxisAttr>(scopeAxis).getName() == axis.getName()) {
          extent = scopeExtent;
          break;
        }
      }
      if (extent && *extent != ShapedType::kDynamic && dim != *extent)
        return emitOpError("result tensor dimension does not match yielded axis extent");
    }
  }

  return success();
}

void ScopeOp::getAsmBlockArgumentNames(Region &region, OpAsmSetValueNameFn setNameFn) {
  if (&region != &getBody())
    return;

  Block &block = region.front();
  ArrayAttr axes = getAxes().getAxes();
  for (auto [arg, axis] : zip_equal(block.getArguments(), axes))
    setNameFn(arg, cast<AxisAttr>(axis).getName().getValue());
}
} // namespace ta

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TADialectPlugin", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<ta::TADialect>();
            ta::registerTATransformExtension(*registry);
            ta::registerLinalgToTAPass();
            ta::registerTAToLinalgPass();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TAPassPlugin", LLVM_VERSION_STRING, []() {
            ta::registerLinalgToTAPass();
            ta::registerTAToLinalgPass();
          }};
}

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TAEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TAAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TATypes.cpp.inc"

#define GET_OP_CLASSES
#include "TAOps.cpp.inc"
