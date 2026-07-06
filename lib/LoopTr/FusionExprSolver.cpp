#include "LoopTr/FusionExprSolver.h"
#include "LoopTr/Utils.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/embed.h>

#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace mlir {
namespace json = llvm::json;
namespace py = pybind11;

namespace {

/* Serialization: MLIR program to JSON */

struct SerializationState {
  const DenseMap<Value, std::string> &variableNames;
  Operation *scope;
  DenseSet<Value> activeValues;
};

bool isNestedUnderScope(Operation *operation, Operation *scope) {
  return operation && scope && (operation == scope || scope->isAncestor(operation));
}

std::string stringifyType(Type type) {
  std::string typeString;
  llvm::raw_string_ostream os(typeString);
  type.print(os);
  return typeString;
}

std::string stringifyInteger(const APInt &value) {
  SmallString<32> storage;
  value.toStringSigned(storage);
  return std::string(storage);
}

std::string stringifyFloat(const APFloat &value) {
  SmallString<32> storage;
  value.toString(storage);
  return std::string(storage);
}

std::string stringifyJSON(const json::Value &value) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << value;
  return storage;
}

FailureOr<json::Value> serializeMLIRExprValueToJSON(Value value, SerializationState &state);

template <size_t N>
FailureOr<json::Value> buildNAryExpr(llvm::StringRef opName, Type type, std::array<Value, N> values,
                                     SerializationState &state) {
  json::Array args;
  args.reserve(N);
  for (Value value : values) {
    auto expr = serializeMLIRExprValueToJSON(value, state);
    if (failed(expr))
      return failure();
    args.push_back(std::move(*expr));
  }
  return json::Value(json::Object{
      {"op", opName.str()},
      {"type", stringifyType(type)},
      {"args", std::move(args)},
  });
}

template <typename Op>
FailureOr<json::Value> buildUnaryExpr(llvm::StringRef opName, Op op, SerializationState &state) {
  return buildNAryExpr<1>(opName, op.getType(), {op.getOperand()}, state);
}

template <typename Op>
FailureOr<json::Value> buildBinaryExpr(llvm::StringRef opName, Op op, SerializationState &state) {
  return buildNAryExpr<2>(opName, op.getType(), {op.getLhs(), op.getRhs()}, state);
}

FailureOr<json::Value> buildConstantExpr(arith::ConstantOp constantOp) {
  Type type = constantOp.getType();
  Attribute attr = constantOp.getValue();
  std::string valueString;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    valueString = stringifyInteger(intAttr.getValue());
  else if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    valueString = stringifyFloat(floatAttr.getValue());
  else {
    constantOp.emitError("unsupported constant attribute");
    return failure();
  }

  return json::Value(json::Object{
      {"op", "const"},
      {"type", stringifyType(type)},
      {"value", valueString},
  });
}

FailureOr<Value> mapBlockArgumentToDef(Value value, SerializationState &state) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return failure();

  Block *block = blockArg.getOwner();
  Operation *parentOp = block ? block->getParentOp() : nullptr;
  if (!isNestedUnderScope(parentOp, state.scope)) {
    if (parentOp)
      parentOp->emitError("expression walk escaped the requested scope through a block argument");
    return failure();
  }

  if (!block || !parentOp || !block->isEntryBlock() || parentOp->getNumRegions() != 1) {
    if (parentOp)
      parentOp->emitError("unsupported block argument shape while serializing expression");
    return failure();
  }

  unsigned argNumber = blockArg.getArgNumber();
  if (argNumber >= parentOp->getNumOperands()) {
    parentOp->emitError("block argument does not map to a parent operand");
    return failure();
  }
  return parentOp->getOperand(argNumber);
}

FailureOr<json::Value> serializeMLIRExprValueToJSON(Value value, SerializationState &state) {
  if (auto it = state.variableNames.find(value); it != state.variableNames.end()) {
    return json::Value(json::Object{
        {"op", "var"},
        {"name", it->second},
        {"type", stringifyType(value.getType())},
    });
  }

  if (!state.activeValues.insert(value).second) {
    if (auto *def = value.getDefiningOp())
      def->emitError("cyclic value dependency while serializing expression");
    return failure();
  }
  auto eraseGuard = llvm::scope_exit([&] { state.activeValues.erase(value); });

  auto mapped = mapBlockArgumentToDef(value, state);
  if (succeeded(mapped))
    return serializeMLIRExprValueToJSON(*mapped, state);

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return failure();

  Operation *def = result.getDefiningOp();
  if (!isNestedUnderScope(def, state.scope)) {
    if (def)
      def->emitError("expression walk escaped the requested scope");
    return failure();
  }

  if (auto constantOp = dyn_cast<arith::ConstantOp>(def))
    return buildConstantExpr(constantOp);
  if (auto addf = dyn_cast<arith::AddFOp>(def))
    return buildBinaryExpr("add", addf, state);
  if (auto subf = dyn_cast<arith::SubFOp>(def))
    return buildBinaryExpr("sub", subf, state);
  if (auto mulf = dyn_cast<arith::MulFOp>(def))
    return buildBinaryExpr("mul", mulf, state);
  if (auto divf = dyn_cast<arith::DivFOp>(def))
    return buildBinaryExpr("div", divf, state);
  if (auto maxnumf = dyn_cast<arith::MaxNumFOp>(def))
    return buildBinaryExpr("maxnum", maxnumf, state);
  if (auto maximumf = dyn_cast<arith::MaximumFOp>(def))
    return buildBinaryExpr("max", maximumf, state);
  if (auto exp = dyn_cast<math::ExpOp>(def))
    return buildUnaryExpr("exp", exp, state);
  if (auto exp2 = dyn_cast<math::Exp2Op>(def))
    return buildUnaryExpr("exp2", exp2, state);
  if (auto log = dyn_cast<math::LogOp>(def))
    return buildUnaryExpr("log", log, state);
  if (auto sqrt = dyn_cast<math::SqrtOp>(def))
    return buildUnaryExpr("sqrt", sqrt, state);
  if (auto rsqrt = dyn_cast<math::RsqrtOp>(def))
    return buildUnaryExpr("rsqrt", rsqrt, state);
  // The Python solver works over real-valued expressions. For now, ignore
  // precision-changing casts during extraction and treat them as transparent.
  if (auto truncf = dyn_cast<arith::TruncFOp>(def))
    return serializeMLIRExprValueToJSON(truncf.getIn(), state);
  if (auto extf = dyn_cast<arith::ExtFOp>(def))
    return serializeMLIRExprValueToJSON(extf.getIn(), state);

  def->emitError("unsupported operation while serializing expression");
  return failure();
}

FailureOr<json::Value> serializeMLIRExprToJSON(Value output,
                                               const DenseMap<Value, std::string> &variableNames,
                                               Operation *scope) {
  Operation *def = output.getDefiningOp();
  if (!scope || (def && !isNestedUnderScope(def, scope))) {
    if (def)
      def->emitError("output is not nested under the requested scope");
    return failure();
  }
  SerializationState state{
      .variableNames = variableNames,
      .scope = scope,
      .activeValues = {},
  };
  return serializeMLIRExprValueToJSON(output, state);
}

/* Deserialization: JSON to MLIR program */

const json::Array *getArgsField(const json::Object &object) { return object.getArray("args"); }

FailureOr<Type> parseJSONExprType(const json::Object &object, MLIRContext *context) {
  auto typeString = object.getString("type");
  if (!typeString)
    return failure();
  Type type = parseType(*typeString, context);
  if (!type)
    return failure();
  return type;
}

FailureOr<TypedAttr> parseJSONConstAttr(const json::Object &object, Type type,
                                        MLIRContext *context) {
  auto *value = object.get("value");
  if (!value)
    return failure();

  Attribute attr;
  if (auto str = value->getAsString())
    attr = parseAttribute(*str, context, type);
  else if (auto intValue = value->getAsInteger()) {
    std::string printed = std::to_string(*intValue);
    Type elementType = type;
    if (auto shapedType = dyn_cast<ShapedType>(type))
      elementType = shapedType.getElementType();
    if (isa<FloatType>(elementType))
      printed += ".0";
    attr = parseAttribute(printed, context, type);
  } else if (auto doubleValue = value->getAsNumber()) {
    std::string printed;
    llvm::raw_string_ostream os(printed);
    os << *doubleValue;
    os.flush();
    attr = parseAttribute(printed, context, type);
  }
  if (!attr)
    return failure();

  auto typedAttr = dyn_cast<TypedAttr>(attr);
  if (!typedAttr || typedAttr.getType() != type) {
    llvm::errs() << "deserializeMLIRExprFromJSON: parsed constant has unexpected type "
                 << (typedAttr ? typedAttr.getType() : Type()) << ", expected " << type << "\n";
    return failure();
  }
  return typedAttr;
}

struct DeserializationState {
  RewriterBase &rewriter;
  Location loc;
  llvm::StringMap<Value> variablesByName;
};

FailureOr<Value> deserializeJSONExprToMLIR(const json::Value &expr, DeserializationState &state);

FailureOr<Value> getOrCreateVariable(const json::Object &object, DeserializationState &state) {
  auto name = object.getString("name");
  if (!name)
    return failure();
  auto type = parseJSONExprType(object, state.rewriter.getContext());
  if (failed(type))
    return failure();

  if (auto it = state.variablesByName.find(*name); it != state.variablesByName.end()) {
    if (it->second.getType() != *type) {
      llvm::errs() << "deserializeMLIRExprFromJSON: variable `" << *name
                   << "` is prebound to value " << it->second << " : " << it->second.getType()
                   << " but JSON expects type " << *type << "\n";
      return failure();
    }
    return it->second;
  }

  Block *block = state.rewriter.getInsertionBlock();
  if (!block) {
    llvm::errs() << "deserializeMLIRExprFromJSON: no insertion block available for fresh variable `"
                 << *name << "`\n";
    return failure();
  }
  Value variable = block->addArgument(*type, state.loc);
  state.variablesByName.try_emplace(*name, variable);
  return variable;
}

template <typename Op>
FailureOr<Value> buildUnaryValueExpr(const json::Object &object, DeserializationState &state) {
  auto args = getArgsField(object);
  if (!args || args->size() != 1)
    return failure();
  auto operand = deserializeJSONExprToMLIR(args->front(), state);
  if (failed(operand))
    return failure();
  return Op::create(state.rewriter, state.loc, *operand)->getResult(0);
}

template <typename Op>
FailureOr<Value> buildBinaryValueExpr(const json::Object &object, DeserializationState &state) {
  auto args = getArgsField(object);
  if (!args || args->size() != 2)
    return failure();
  auto lhs = deserializeJSONExprToMLIR(args->front(), state);
  if (failed(lhs))
    return failure();
  auto rhs = deserializeJSONExprToMLIR(args->back(), state);
  if (failed(rhs))
    return failure();
  return Op::create(state.rewriter, state.loc, *lhs, *rhs)->getResult(0);
}

/// Constructs a sequence of binary MLIR operations (such as `arith::AddFOp`)
/// from an N-ary JSON expression (such as `add` with N arguments).
template <typename Op>
FailureOr<Value> buildLeftFoldExpr(const json::Object &object, DeserializationState &state) {
  auto args = getArgsField(object);
  if (!args || args->empty())
    return failure();

  auto current = deserializeJSONExprToMLIR(args->front(), state);
  if (failed(current))
    return failure();
  for (const json::Value &arg : llvm::drop_begin(*args)) {
    auto next = deserializeJSONExprToMLIR(arg, state);
    if (failed(next))
      return failure();
    current = Op::create(state.rewriter, state.loc, *current, *next)->getResult(0);
  }
  return current;
}

FailureOr<Value> deserializeJSONExprToMLIR(const json::Value &expr, DeserializationState &state) {
  auto object = expr.getAsObject();
  if (!object)
    return failure();
  auto opNameOpt = object->getString("op");
  if (!opNameOpt)
    return failure();
  StringRef opName = *opNameOpt;

  auto *context = state.rewriter.getContext();
  if (opName == "const") {
    auto parsedType = parseJSONExprType(*object, context);
    if (failed(parsedType))
      return failure();
    auto attr = parseJSONConstAttr(*object, *parsedType, context);
    if (failed(attr))
      return failure();
    return arith::ConstantOp::create(state.rewriter, state.loc, *parsedType, *attr)->getResult(0);
  }
  if (opName == "var")
    return getOrCreateVariable(*object, state);
  if (opName == "add")
    return buildLeftFoldExpr<arith::AddFOp>(*object, state);
  if (opName == "sub")
    return buildBinaryValueExpr<arith::SubFOp>(*object, state);
  if (opName == "mul")
    return buildLeftFoldExpr<arith::MulFOp>(*object, state);
  if (opName == "div")
    return buildBinaryValueExpr<arith::DivFOp>(*object, state);
  if (opName == "maxnum")
    return buildLeftFoldExpr<arith::MaxNumFOp>(*object, state);
  if (opName == "max")
    return buildLeftFoldExpr<arith::MaximumFOp>(*object, state);
  if (opName == "exp")
    return buildUnaryValueExpr<math::ExpOp>(*object, state);
  if (opName == "exp2")
    return buildUnaryValueExpr<math::Exp2Op>(*object, state);
  if (opName == "log")
    return buildUnaryValueExpr<math::LogOp>(*object, state);
  if (opName == "sqrt")
    return buildUnaryValueExpr<math::SqrtOp>(*object, state);
  if (opName == "rsqrt")
    return buildUnaryValueExpr<math::RsqrtOp>(*object, state);
  if (opName == "abs")
    return buildUnaryValueExpr<math::AbsFOp>(*object, state);
  if (opName == "pow")
    return buildBinaryValueExpr<math::PowFOp>(*object, state);
  llvm::errs() << "deserializeMLIRExprFromJSON: unsupported op `" << opName << "`\n";
  return failure();
}

struct DeserializedValueExpr {
  std::unique_ptr<Block> block;
  Value result;
};

FailureOr<DeserializedValueExpr> deserializeMLIRExprFromJSON(const json::Value &expr,
                                                             ArrayRef<std::string> variableNames,
                                                             RewriterBase &rewriter, Location loc) {
  auto parsedBlock = std::make_unique<Block>();
  DeserializationState state{.rewriter = rewriter, .loc = loc, .variablesByName = {}};
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(parsedBlock.get());
  auto result = deserializeJSONExprToMLIR(expr, state);
  if (failed(result))
    return failure();

  // Rebuild the block so its arguments exactly match `variableNames`. This
  // gives FusionRepairTerm::build a stable scalar ABI: r0, r0', ..., acc.
  auto orderedBlock = std::make_unique<Block>();
  IRMapping mapping;
  for (auto &varName : variableNames) {
    auto it = state.variablesByName.find(varName);
    if (it == state.variablesByName.end())
      return emitError(loc) << "deserializeMLIRExprFromJSON: missing expected variable `" << varName
                            << "`\n";
    Value orderedArg = orderedBlock->addArgument(it->second.getType(), loc);
    mapping.map(it->second, orderedArg);
    state.variablesByName.erase(it);
  }
  if (!state.variablesByName.empty())
    return emitError(loc) << "deserializeMLIRExprFromJSON: " << state.variablesByName.size()
                          << " unexpected variable(s)\n";
  rewriter.setInsertionPointToStart(orderedBlock.get());
  for (Operation &op : *parsedBlock)
    rewriter.clone(op, mapping);

  return DeserializedValueExpr{.block = std::move(orderedBlock), .result = mapping.lookup(*result)};
}

/* Solver logic begins */

llvm::Expected<json::Value> solveRollingUpdaterWithPython(const json::Value &fExpr,
                                                          const json::Value &gExpr,
                                                          ArrayRef<std::string> rVariables,
                                                          StringRef accVar) {
  static std::once_flag initOnce;
  static py::object solverModule;
  static py::object jsonModule;
  static std::string initError;

  std::call_once(initOnce, []() {
    try {
      // Intentionally leak the interpreter. If the d'tor of the interpreter throws an exception, it
      // can cause a crash (likely during program shutdown, but still ugly).
      auto interpreter = std::make_unique<py::scoped_interpreter>();
      auto interpPtr = interpreter.release();
      (void)interpPtr;
      solverModule = py::module_::import("neptune_mlir.rolling_solver");
      jsonModule = py::module_::import("json");
    } catch (const py::error_already_set &e) {
      initError = e.what();
    }
  });
  if (!initError.empty()) {
    return llvm::make_error<llvm::StringError>("failed to initialize embedded Python: " + initError,
                                               llvm::inconvertibleErrorCode());
  }

  try {
    py::gil_scoped_acquire gil;
    py::object fExprPy = jsonModule.attr("loads")(stringifyJSON(fExpr));
    py::object gExprPy = jsonModule.attr("loads")(stringifyJSON(gExpr));
    py::list rVariablesPy;
    for (const std::string &name : rVariables)
      rVariablesPy.append(name);
    py::object hExprPy = solverModule.attr("solve_rolling_updater_json")(
        fExprPy, gExprPy, rVariablesPy, accVar.str());
    std::string hExprText = py::str(jsonModule.attr("dumps")(hExprPy, py::arg("sort_keys") = true));
    return json::parse(hExprText);
  } catch (const py::error_already_set &e) {
    return llvm::make_error<llvm::StringError>("embedded solver raised Python exception: " +
                                                   std::string(e.what()),
                                               llvm::inconvertibleErrorCode());
  }
}

FailureOr<AffineMap> dropDomainDim(AffineMap map, unsigned droppedDim) {
  MLIRContext *ctx = map.getContext();
  unsigned oldNumDims = map.getNumDims();
  if (droppedDim >= oldNumDims)
    return failure();

  for (AffineExpr expr : map.getResults()) {
    if (expr.isFunctionOfDim(droppedDim))
      return failure();
  }
  SmallVector<AffineExpr> dimRepls = llvm::to_vector(llvm::map_range(
      llvm::index_range(0, oldNumDims), [&](size_t i) { return getAffineDimExpr(i, ctx); }));
  dimRepls.insert(dimRepls.begin() + droppedDim, getAffineConstantExpr(0, ctx));

  return map.replaceDimsAndSymbols(dimRepls, map.getResults(), oldNumDims - 1, map.getNumSymbols());
}

FailureOr<AffineMap> appendDomainDimAsResult(AffineMap map, unsigned dim) {
  if (dim >= map.getNumDims())
    return failure();
  SmallVector<AffineExpr> results(map.getResults());
  results.push_back(getAffineDimExpr(dim, map.getContext()));
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), results, map.getContext());
}

} // namespace

FailureOr<FusionRepairTerm> solveFusionRepairExpr(RewriterBase &rewriter,
                                                  ArrayRef<Operation *> producingReds,
                                                  linalg::GenericOp thisRed,
                                                  ArrayRef<Operation *> elemwiseSidecars) {
  // Step 1. Fuse the frontier reduction with its sidecar elementwise producers until we get a
  // single linalg.generic op that contains all the computation.
  rewriter.setInsertionPointAfter(thisRed);
  SmallPtrSet<Operation *, 4> sidecarSet(elemwiseSidecars.begin(), elemwiseSidecars.end());
  auto findFusableOperand =
      [&sidecarSet](Operation *consumer) -> std::optional<std::pair<Operation *, unsigned>> {
    for (auto &operand : consumer->getOpOperands()) {
      auto producer = operand.get().getDefiningOp();
      if (producer && sidecarSet.count(producer))
        return std::make_pair(producer, operand.getOperandNumber());
    }
    return std::nullopt;
  };

  OpResult reductionResult = cast<OpResult>(thisRed->getResult(0));
  linalg::GenericOp currentOp = thisRed;
  while (auto nextFusionTarget = findFusableOperand(currentOp)) {
    auto [producer, consumerOpndNum] = *nextFusionTarget;
    FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
        linalg::fuseElementwiseOps(rewriter, &currentOp->getOpOperand(consumerOpndNum));
    if (failed(fusionResult)) {
      producer->emitError("failed to fuse this op...");
      return currentOp->emitError("into this op");
    }
    auto it = fusionResult->replacements.find(reductionResult);
    if (it == fusionResult->replacements.end())
      return failure();
    if (currentOp != thisRed)
      rewriter.eraseOp(currentOp);
    reductionResult = cast<OpResult>(it->second);
    currentOp = cast<linalg::GenericOp>(fusionResult->fusedOp);
  }
  auto scopeGuard = llvm::scope_exit([&]() {
    if (currentOp != thisRed)
      rewriter.eraseOp(currentOp);
  });

  // Step 2. We have the single generic op as `currentOp`. Check it is still a reduction, and get a
  // list of its inputs. Distinguish inputs from reductions (`producingReds`), which we give
  // variable names "r0", "r1", etc., from inputs that are not reductions, which we give variable
  // names "c0", "c1", etc. The
  auto match = matchBinaryReductionCombiner(currentOp, reductionResult.getResultNumber(),
                                            /*emitDiagnostics=*/true);
  if (failed(match))
    return currentOp.emitError() << "this reduction doesn't have a binary reduce op in body";

  SmallPtrSet<Value, 4> prodRedResults;
  for (Operation *prodRed : producingReds)
    prodRedResults.insert(prodRed->result_begin(), prodRed->result_end());

  auto genericBodyBlk = currentOp.getBlock();
  size_t rCounter = 0, cCounter = 0;
  SmallVector<std::string> redVarNames;
  SmallVector<OpResult> redVarResults;
  SmallVector<AffineMap> redVarIndexingMaps;
  DenseMap<Value, std::string> gExprVarNames;
  for (int64_t i = 0; i < currentOp.getNumDpsInputs(); ++i) {
    Value operand = currentOp.getOperand(i);
    AffineMap indexMap = currentOp.getIndexingMapsArray()[i];
    BlockArgument blkArg = genericBodyBlk->getArgument(i);
    if (prodRedResults.contains(operand)) {
      auto rName = "r" + std::to_string(rCounter++);
      gExprVarNames[blkArg] = rName;
      redVarNames.push_back(rName);
      redVarResults.push_back(cast<OpResult>(operand));
      redVarIndexingMaps.push_back(indexMap);
    } else {
      gExprVarNames[blkArg] = "c" + std::to_string(cCounter++);
    }
  }

  // Step 3. Serialize reduction RHS into a JSON expression, which we call the g expression.
  auto gExpr = serializeMLIRExprToJSON(match->nonAccumulator, gExprVarNames, currentOp);
  if (failed(gExpr)) {
    currentOp->emitRemark("this is the compute operation we're extracting from");
    return failure();
  }

  // Step 4. Similarly serialize the reduction op itself into a JSON expression (f expression).
  static const std::string accVarName = "acc";
  DenseMap<Value, std::string> fExprVarNames{
      {match->accumulatorArg, accVarName},
      {match->nonAccumulator, "x"},
  };
  auto fExpr = serializeMLIRExprToJSON(match->yieldedValue, fExprVarNames, currentOp);
  if (failed(fExpr))
    return failure();

  // Step 5. Call the Python solver to derive the repair term expression.
  auto hExpr = solveRollingUpdaterWithPython(fExpr, gExpr, redVarNames, accVarName);
  if (!hExpr)
    return thisRed.emitError("failed to solve for the rolling update expression: ")
           << llvm::toString(hExpr.takeError());

  // Step 6. Deserialize the repair term expression into MLIR using the solver ABI: r0, r0', r1,
  // r1', ..., acc. The prime in variable names is a convension of the solver (that you can see in
  // the Python solver code).
  SmallVector<std::string> hVarNames;
  hVarNames.reserve(2 * rCounter + 1);
  for (size_t i = 0; i < rCounter; ++i) {
    std::string varName = "r" + std::to_string(i);
    hVarNames.push_back(varName);
    hVarNames.push_back(varName + "'");
  }
  hVarNames.push_back(accVarName);
  auto deserialized = deserializeMLIRExprFromJSON(*hExpr, hVarNames, rewriter, thisRed.getLoc());
  if (failed(deserialized))
    return thisRed.emitError("failed to deserialize the repair term expression into MLIR");

  size_t accIndex = currentOp.getNumDpsInputs() + reductionResult.getResultNumber();
  AffineMap accIndexingMap = currentOp.getIndexingMapsArray()[accIndex];
  return FusionRepairTerm(std::move(redVarResults), std::move(redVarIndexingMaps), accIndexingMap,
                          std::move(deserialized->block), deserialized->result);
}

FailureOr<linalg::GenericOp>
FusionRepairTerm::build(RewriterBase &rewriter, Location loc,
                        ArrayRef<FusionRepairReductionBinding> reduceArgs, Value accArg,
                        FusionRepairTermMode mode, unsigned reduceDim) const {
  DenseMap<OpResult, FusionRepairReductionBinding> bindingByReduction;
  for (const FusionRepairReductionBinding &binding : reduceArgs) {
    if (!bindingByReduction.try_emplace(binding.reductionResult, binding).second)
      return binding.reductionResult.getOwner()->emitError()
             << "duplicate repair binding for this reduction result";
  }

  // Build input arguments and their indexing maps in scalarBlock ABI order.
  size_t numReductions = reductionOrder.size();
  SmallVector<Value> inputTensors;
  inputTensors.reserve(2 * numReductions + 1);
  SmallVector<AffineMap> indexingMaps;
  indexingMaps.reserve(2 * numReductions + 2);
  for (size_t i = 0; i < numReductions; ++i) {
    auto redResult = reductionOrder[i];
    auto it = bindingByReduction.find(redResult);
    if (it == bindingByReduction.end())
      return redResult.getOwner()->emitError()
             << "missing repair binding for this reduction result";
    inputTensors.push_back(it->second.current);
    inputTensors.push_back(it->second.next);

    AffineMap currentMap, nextMap;
    switch (mode) {
    case FusionRepairTermMode::RollingUpdate: {
      auto trimmedMap = dropDomainDim(argsIndexingMaps[i], reduceDim);
      if (failed(trimmedMap))
        return failure();
      currentMap = nextMap = *trimmedMap;
      break;
    }
    case FusionRepairTermMode::SplitKUpdate: {
      auto fullMap = appendDomainDimAsResult(argsIndexingMaps[i], reduceDim);
      if (failed(fullMap))
        return failure();
      currentMap = *fullMap;
      nextMap = argsIndexingMaps[i];
      break;
    }
    }
    indexingMaps.push_back(currentMap);
    indexingMaps.push_back(nextMap);
  }
  // The `acc` scalar argument and DPS result use the same patched accumulator map.
  auto resultMap = mode == FusionRepairTermMode::RollingUpdate
                       ? dropDomainDim(accIndexingMap, reduceDim)
                       : appendDomainDimAsResult(accIndexingMap, reduceDim);
  if (failed(resultMap))
    return failure();
  inputTensors.push_back(accArg);
  indexingMaps.push_back(*resultMap);
  indexingMaps.push_back(*resultMap);

  SmallVector<utils::IteratorType> iteratorTypes(resultMap->getNumDims(),
                                                 utils::IteratorType::parallel);
  return linalg::GenericOp::create(
      rewriter, loc, TypeRange{accArg.getType()}, inputTensors, ValueRange{accArg}, indexingMaps,
      iteratorTypes, [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
        IRMapping mapping;
        for (auto [oldArg, newArg] :
             llvm::zip_equal(scalarBlock->getArguments(), newArgs.take_front(inputTensors.size())))
          mapping.map(oldArg, newArg);
        for (Operation &op : *scalarBlock)
          builder.clone(op, mapping);
        Value mappedResult = mapping.lookupOrDefault(scalarResult);
        linalg::YieldOp::create(builder, nestedLoc, mappedResult);
      });
}

} // namespace mlir
