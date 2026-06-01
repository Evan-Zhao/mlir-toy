#include "TA/TADialect.h"
#include "TA/TAAttrs.h"
#include "TA/TAInterfaces.h"
#include "TA/TAOps.h"
#include "TA/TATypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#define GET_DIALECT_DEFS
#include "TAOpsDialect.cpp.inc"

namespace ta {
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

static mlir::LogicalResult verifyAxisArray(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                           mlir::ArrayAttr axes) {
  if (!axes)
    return emitError() << "expected an array attribute of #ta.axis attributes";

  llvm::StringSet<> seen;
  for (mlir::Attribute attr : axes) {
    auto axis = llvm::dyn_cast<AxisAttr>(attr);
    if (!axis)
      return emitError() << "expected axis list element to be a #ta.axis attribute";

    llvm::StringRef name = axis.getName().getValue();
    if (!seen.insert(name).second)
      return emitError() << "duplicate axis '" << name << "'";
  }

  return mlir::success();
}

static void printAxisNames(mlir::AsmPrinter &printer, mlir::ArrayAttr axes, llvm::StringRef open,
                           llvm::StringRef close) {
  printer << open;
  llvm::interleaveComma(axes, printer, [&](mlir::Attribute attr) {
    printer << llvm::cast<AxisAttr>(attr).getName().getValue();
  });
  printer << close;
}

static mlir::FailureOr<AxesAttr> parseAxisList(mlir::AsmParser &parser,
                                               mlir::AsmParser::Delimiter delimiter) {
  llvm::SmallVector<mlir::Attribute> axes;
  if (parser.parseCommaSeparatedList(delimiter, [&]() -> mlir::ParseResult {
        llvm::StringRef name;
        if (parser.parseKeyword(&name))
          return mlir::failure();
        axes.push_back(AxisAttr::get(parser.getContext(), name));
        return mlir::success();
      }))
    return mlir::failure();

  return AxesAttr::get(parser.getContext(), mlir::ArrayAttr::get(parser.getContext(), axes));
}

mlir::LogicalResult AxisAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                     mlir::StringAttr name) {
  if (!name || name.getValue().empty())
    return emitError() << "axis name must be non-empty";

  return mlir::success();
}

mlir::Attribute AxesAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  (void)type;
  mlir::FailureOr<AxesAttr> axes = parseAxisList(parser, mlir::AsmParser::Delimiter::LessGreater);
  if (mlir::failed(axes))
    return {};
  return *axes;
}

void AxesAttr::print(mlir::AsmPrinter &printer) const {
  printAxisNames(printer, getAxes(), "<", ">");
}

mlir::LogicalResult AxesAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                     mlir::ArrayAttr axes) {
  return verifyAxisArray(emitError, axes);
}

mlir::Type ExprType::parse(mlir::AsmParser &parser) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  mlir::Type elementType;

  if (parser.parseLess() || parser.parseType(elementType) || parser.parseComma())
    return {};

  mlir::FailureOr<AxesAttr> axes = parseAxisList(parser, mlir::AsmParser::Delimiter::Square);
  if (mlir::failed(axes) || parser.parseGreater())
    return {};

  return parser.getChecked<ExprType>(loc, parser.getContext(), elementType, *axes);
}

void ExprType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getElementType());
  printer << ", ";
  printAxisNames(printer, getAxes().getAxes(), "[", "]");
  printer << ">";
}

mlir::LogicalResult ExprType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                     mlir::Type elementType, AxesAttr axes) {
  if (!elementType)
    return emitError() << "expression element type must be present";
  if (!axes)
    return emitError() << "expression axes must be present";

  return mlir::success();
}

static bool axisContains(mlir::ArrayAttr axes, AxisAttr axis) {
  return llvm::any_of(axes, [&](mlir::Attribute attr) {
    return llvm::cast<AxisAttr>(attr).getName() == axis.getName();
  });
}

static mlir::LogicalResult verifyAxesSubset(mlir::Operation *op, AxesAttr scopeAxes,
                                            AxesAttr usedAxes, llvm::StringRef what) {
  if (!usedAxes)
    return mlir::success();

  mlir::ArrayAttr allowed = scopeAxes.getAxes();
  for (mlir::Attribute attr : usedAxes.getAxes()) {
    AxisAttr axis = llvm::cast<AxisAttr>(attr);
    if (!axisContains(allowed, axis)) {
      return op->emitOpError() << what << " uses axis '" << axis.getName().getValue()
                               << "' outside enclosing ta.scope axes";
    }
  }
  return mlir::success();
}

static mlir::FailureOr<ScopeOp> verifyInsideScope(mlir::Operation *op) {
  ScopeOp scope = op->getParentOfType<ScopeOp>();
  if (!scope)
    return op->emitOpError("must be nested inside a ta.scope");
  return scope;
}

static mlir::LogicalResult verifyExprAxes(mlir::Operation *op, ScopeOp scope, mlir::Type type,
                                          llvm::StringRef what) {
  if (auto expr = llvm::dyn_cast<ExprType>(type))
    return verifyAxesSubset(op, scope.getAxes(), expr.getAxes(), what);
  return mlir::success();
}

static bool sameAxes(AxesAttr lhs, AxesAttr rhs) {
  mlir::ArrayAttr lhsAxes = lhs.getAxes();
  mlir::ArrayAttr rhsAxes = rhs.getAxes();
  if (lhsAxes.size() != rhsAxes.size())
    return false;

  for (auto [lhsAttr, rhsAttr] : llvm::zip_equal(lhsAxes, rhsAxes)) {
    AxisAttr lhsAxis = llvm::cast<AxisAttr>(lhsAttr);
    AxisAttr rhsAxis = llvm::cast<AxisAttr>(rhsAttr);
    if (lhsAxis.getName() != rhsAxis.getName())
      return false;
  }

  return true;
}

static AxesAttr inferUnionAxes(mlir::MLIRContext *context, AxesAttr scopeAxes,
                               mlir::ValueRange operands) {
  llvm::StringSet<> used;
  for (mlir::Value operand : operands) {
    auto expr = llvm::cast<ExprType>(operand.getType());
    for (mlir::Attribute attr : expr.getAxes().getAxes()) {
      AxisAttr axis = llvm::cast<AxisAttr>(attr);
      used.insert(axis.getName().getValue());
    }
  }

  llvm::SmallVector<mlir::Attribute> inferred;
  for (mlir::Attribute attr : scopeAxes.getAxes()) {
    AxisAttr axis = llvm::cast<AxisAttr>(attr);
    if (used.contains(axis.getName().getValue()))
      inferred.push_back(attr);
  }

  return AxesAttr::get(context, mlir::ArrayAttr::get(context, inferred));
}

static mlir::LogicalResult verifyElementwiseAxes(mlir::Operation *op, ScopeOp scope) {
  for (mlir::Value operand : op->getOperands()) {
    if (mlir::failed(verifyExprAxes(op, scope, operand.getType(), "operand")))
      return mlir::failure();
  }

  if (mlir::failed(verifyExprAxes(op, scope, op->getResult(0).getType(), "result")))
    return mlir::failure();

  auto result = llvm::cast<ExprType>(op->getResult(0).getType());
  AxesAttr expected = inferUnionAxes(op->getContext(), scope.getAxes(), op->getOperands());
  if (!sameAxes(result.getAxes(), expected))
    return op->emitOpError()
           << "result axes must be the union of operand axes in enclosing ta.scope order; "
           << "expected " << expected;

  return mlir::success();
}

static mlir::LogicalResult verifyFloatElementwiseOp(mlir::Operation *op) {
  auto scopeOr = verifyInsideScope(op);
  if (mlir::failed(scopeOr))
    return mlir::failure();

  if (mlir::failed(verifyElementwiseAxes(op, *scopeOr)))
    return mlir::failure();

  auto result = llvm::cast<ExprType>(op->getResult(0).getType());
  mlir::Type elementType = result.getElementType();
  if (!llvm::isa<mlir::FloatType>(elementType))
    return op->emitOpError("requires a floating-point expression result");

  for (mlir::Value operand : op->getOperands()) {
    auto expr = llvm::cast<ExprType>(operand.getType());
    if (expr.getElementType() != elementType)
      return op->emitOpError("requires all operand and result element types to match");
  }

  return mlir::success();
}

static mlir::LogicalResult verifyUnaryFloatElementwiseOp(mlir::Operation *op) {
  if (op->getNumOperands() != 1)
    return op->emitOpError("expected one operand");
  return verifyFloatElementwiseOp(op);
}

static mlir::LogicalResult verifyBinaryFloatElementwiseOp(mlir::Operation *op) {
  if (op->getNumOperands() != 2)
    return op->emitOpError("expected two operands");
  return verifyFloatElementwiseOp(op);
}

static mlir::LogicalResult verifyTernaryFloatElementwiseOp(mlir::Operation *op) {
  if (op->getNumOperands() != 3)
    return op->emitOpError("expected three operands");
  return verifyFloatElementwiseOp(op);
}

mlir::LogicalResult YieldOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (mlir::failed(scopeOr))
    return mlir::failure();

  mlir::Operation *parent = getOperation()->getParentOp();
  if (auto map = llvm::dyn_cast<MapOp>(parent)) {
    if (getValues().size() != 1)
      return emitOpError("terminating ta.map must yield exactly one value");

    auto result = llvm::cast<ExprType>(map.getResult().getType());
    if (getValues().front().getType() != result.getElementType())
      return emitOpError("terminating ta.map must yield the map result element type");
  } else if (auto scope = llvm::dyn_cast<ScopeOp>(parent)) {
    if (getValues().size() != 1)
      return emitOpError("terminating ta.scope must yield exactly one value");

    auto expr = llvm::dyn_cast<ExprType>(getValues().front().getType());
    if (!expr)
      return emitOpError("terminating ta.scope must yield a ta.expr value");

    if (mlir::failed(verifyAxesSubset(getOperation(), scope.getAxes(), expr.getAxes(),
                                      "yielded expression")))
      return mlir::failure();

    auto resultType = llvm::cast<mlir::RankedTensorType>(scope.getResult().getType());
    if (resultType.getElementType() != expr.getElementType())
      return emitOpError("yielded expression element type must match ta.scope result tensor "
                         "element type");
  }

  return mlir::success();
}

mlir::LogicalResult AtOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (mlir::failed(scopeOr))
    return mlir::failure();

  ScopeOp scope = *scopeOr;
  if (auto axes = getAxes()) {
    if (mlir::failed(verifyAxesSubset(getOperation(), scope.getAxes(), *axes, "access")))
      return mlir::failure();
  }

  return verifyExprAxes(getOperation(), scope, getResult().getType(), "result");
}

mlir::LogicalResult EvalOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (mlir::failed(scopeOr))
    return mlir::failure();

  ScopeOp scope = *scopeOr;
  if (auto axes = getAxes()) {
    if (mlir::failed(verifyAxesSubset(getOperation(), scope.getAxes(), *axes, "eval")))
      return mlir::failure();
  }

  return verifyExprAxes(getOperation(), scope, getResult().getType(), "result");
}

mlir::LogicalResult MapOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (mlir::failed(scopeOr))
    return mlir::failure();

  ScopeOp scope = *scopeOr;
  if (mlir::failed(verifyElementwiseAxes(getOperation(), scope)))
    return mlir::failure();

  auto result = llvm::cast<ExprType>(getResult().getType());
  mlir::Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size())
    return emitOpError("expected one body argument per input");

  for (auto [input, arg] : llvm::zip_equal(getInputs(), block.getArguments())) {
    auto expr = llvm::cast<ExprType>(input.getType());
    if (arg.getType() != expr.getElementType())
      return emitOpError("body argument types must match input expression element types");
  }

  auto yield = llvm::dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("body must terminate with ta.yield");
  if (yield.getValues().size() != 1)
    return emitOpError("body must yield exactly one value");
  if (yield.getValues().front().getType() != result.getElementType())
    return emitOpError("body yield type must match result expression element type");

  return mlir::success();
}

mlir::LogicalResult ConstantOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (mlir::failed(scopeOr))
    return mlir::failure();

  auto result = llvm::cast<ExprType>(getResult().getType());
  if (!result.getAxes().getAxes().empty())
    return emitOpError("result axes must be empty");
  if (mlir::failed(verifyExprAxes(getOperation(), *scopeOr, getResult().getType(), "result")))
    return mlir::failure();
  if (getValue().getType() != result.getElementType())
    return emitOpError("value type must match result expression element type");

  return mlir::success();
}

#define DEFINE_TA_UNARY_FLOAT_VERIFY(OP)                                                          \
  mlir::LogicalResult OP::verify() { return verifyUnaryFloatElementwiseOp(getOperation()); }

#define DEFINE_TA_BINARY_FLOAT_VERIFY(OP)                                                         \
  mlir::LogicalResult OP::verify() { return verifyBinaryFloatElementwiseOp(getOperation()); }

#define DEFINE_TA_TERNARY_FLOAT_VERIFY(OP)                                                        \
  mlir::LogicalResult OP::verify() { return verifyTernaryFloatElementwiseOp(getOperation()); }

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

mlir::LogicalResult CmpFOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (mlir::failed(scopeOr))
    return mlir::failure();
  if (mlir::failed(verifyElementwiseAxes(getOperation(), *scopeOr)))
    return mlir::failure();

  auto lhs = llvm::cast<ExprType>(getLhs().getType());
  auto rhs = llvm::cast<ExprType>(getRhs().getType());
  auto result = llvm::cast<ExprType>(getResult().getType());
  if (!llvm::isa<mlir::FloatType>(lhs.getElementType()))
    return emitOpError("requires floating-point operand element types");
  if (lhs.getElementType() != rhs.getElementType())
    return emitOpError("requires matching operand element types");
  if (!result.getElementType().isInteger(1))
    return emitOpError("result element type must be i1");

  return mlir::success();
}

mlir::LogicalResult SelectOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (mlir::failed(scopeOr))
    return mlir::failure();
  if (mlir::failed(verifyElementwiseAxes(getOperation(), *scopeOr)))
    return mlir::failure();

  auto condition = llvm::cast<ExprType>(getCondition().getType());
  auto trueValue = llvm::cast<ExprType>(getTrueValue().getType());
  auto falseValue = llvm::cast<ExprType>(getFalseValue().getType());
  auto result = llvm::cast<ExprType>(getResult().getType());

  if (!condition.getElementType().isInteger(1))
    return emitOpError("condition element type must be i1");
  if (trueValue.getElementType() != falseValue.getElementType() ||
      trueValue.getElementType() != result.getElementType())
    return emitOpError("true, false, and result element types must match");

  return mlir::success();
}

mlir::LogicalResult ReduceOp::verify() {
  auto scopeOr = verifyInsideScope(getOperation());
  if (mlir::failed(scopeOr))
    return mlir::failure();

  ScopeOp scope = *scopeOr;
  if (mlir::failed(verifyAxesSubset(getOperation(), scope.getAxes(), getAxes(), "reduction")))
    return mlir::failure();
  if (mlir::failed(verifyExprAxes(getOperation(), scope, getInput().getType(), "input")))
    return mlir::failure();

  return verifyExprAxes(getOperation(), scope, getResult().getType(), "result");
}

mlir::ParseResult ScopeOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::Argument> axisArgs;
  llvm::SmallVector<mlir::Attribute> axes;
  if (parser.parseKeyword("axes") || parser.parseLParen())
    return mlir::failure();

  if (parser.parseOptionalRParen()) {
    do {
      mlir::OpAsmParser::Argument arg;
      std::string axisName;
      if (parser.parseArgument(arg) || parser.parseString(&axisName) ||
          parser.parseColonType(arg.type))
        return mlir::failure();
      if (!arg.type.isIndex())
        return parser.emitError(arg.ssaName.location, "expected axis to have index type");

      axisArgs.push_back(arg);
      axes.push_back(AxisAttr::get(parser.getContext(), axisName));
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen())
      return mlir::failure();
  }

  result.addAttribute(
      getAxesAttrName(result.name),
      AxesAttr::get(parser.getContext(), mlir::ArrayAttr::get(parser.getContext(), axes)));

  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, axisArgs))
    return mlir::failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  llvm::SmallVector<mlir::Type> resultTypes;
  if (parser.parseColon() || parser.parseLParen() || parser.parseRParen() ||
      parser.parseArrowTypeList(resultTypes))
    return mlir::failure();
  if (resultTypes.size() != 1)
    return parser.emitError(parser.getCurrentLocation(), "expected one result type");

  result.addTypes(resultTypes);
  return mlir::success();
}

void ScopeOp::print(mlir::OpAsmPrinter &printer) {
  printer << " axes(";
  mlir::Block &block = getBody().front();
  mlir::ArrayAttr axes = getAxes().getAxes();
  llvm::interleaveComma(llvm::seq<unsigned>(0, block.getNumArguments()), printer, [&](unsigned i) {
    mlir::BlockArgument arg = block.getArgument(i);
    printer.printOperand(arg);
    printer << " \"";
    printer << llvm::cast<AxisAttr>(axes[i]).getName().getValue();
    printer << "\" : ";
    printer.printType(arg.getType());
  });
  printer << ") ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDict((*this)->getAttrs(), {"axes"});
  printer << " : () -> ";
  printer.printType(getResult().getType());
}

mlir::LogicalResult ScopeOp::verify() {
  mlir::Block &block = getBody().front();
  if (block.getNumArguments() != getAxes().getAxes().size())
    return emitOpError("expected one region argument per axis");
  for (mlir::BlockArgument arg : block.getArguments()) {
    if (!arg.getType().isIndex())
      return emitOpError("expected axis region arguments to have index type");
  }

  auto yield = llvm::dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("body must terminate with ta.yield");

  if (yield.getValues().size() != 1)
    return emitOpError("body must yield exactly one value");

  auto expr = llvm::dyn_cast<ExprType>(yield.getValues().front().getType());
  if (!expr)
    return emitOpError("body must yield a ta.expr value");

  auto resultType = llvm::cast<mlir::RankedTensorType>(getResult().getType());
  if (resultType.getElementType() != expr.getElementType())
    return emitOpError("result tensor element type must match yielded expression element type");

  if (mlir::failed(
          verifyAxesSubset(getOperation(), getAxes(), expr.getAxes(), "yielded expression")))
    return mlir::failure();

  return mlir::success();
}

void ScopeOp::getAsmBlockArgumentNames(mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  if (&region != &getBody())
    return;

  mlir::Block &block = region.front();
  mlir::ArrayAttr axes = getAxes().getAxes();
  for (auto [arg, axis] : llvm::zip_equal(block.getArguments(), axes))
    setNameFn(arg, llvm::cast<AxisAttr>(axis).getName().getValue());
}
} // namespace ta

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TADialectPlugin", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) { registry->insert<ta::TADialect>(); }};
}

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TAInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TAAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TATypes.cpp.inc"

#define GET_OP_CLASSES
#include "TAOps.cpp.inc"
