#include "TA/TAAttrs.h"
#include "TA/TAOps.h"
#include "TA/TAPasses.h"
#include "TA/TATypes.h"
#include "TA/TAUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#include <optional>

namespace ta {

using namespace mlir;

namespace {

static std::optional<int64_t> getImportGroup(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>("ta.import_group");
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static bool sameImportGroup(Operation *lhs, Operation *rhs) {
  std::optional<int64_t> lhsGroup = getImportGroup(lhs);
  std::optional<int64_t> rhsGroup = getImportGroup(rhs);
  return lhsGroup && rhsGroup && *lhsGroup == *rhsGroup;
}

static SmallVector<StringRef> axisNames(AxesAttr axes) {
  SmallVector<StringRef> result;
  for (Attribute attr : axes.getAxes())
    result.push_back(cast<AxisAttr>(attr).getName().getValue());
  return result;
}

static SmallVector<StringRef> axisNames(ExprType expr) { return axisNames(expr.getAxes()); }

static bool containsAxis(ArrayRef<StringRef> axes, StringRef axis) {
  return llvm::is_contained(axes, axis);
}

static FailureOr<unsigned> findAxis(Operation *op, ArrayRef<StringRef> axes, StringRef axis) {
  for (auto [index, candidate] : llvm::enumerate(axes)) {
    if (candidate == axis)
      return index;
  }
  return op->emitOpError("lowering could not find axis '") << axis << "' in loop axes";
}

class ScopeLowering {
public:
  ScopeLowering(ScopeOp scope, OpBuilder &builder,
                DenseMap<Operation *, Operation *> *loweredOps = nullptr,
                llvm::function_ref<void(Operation *, const DenseMap<Operation *, Operation *> &)>
                    beforeErase = nullptr)
      : scope(scope), builder(builder), context(builder.getContext()), loc(scope.getLoc()),
        loweredOps(loweredOps), beforeErase(beforeErase) {}

  LogicalResult run() {
    if (failed(discoverAxisSizes()))
      return failure();

    Block &body = scope.getBody().front();
    for (Operation &op : body.without_terminator()) {
      if (!hasExprResult(&op))
        continue;
      if (!shouldMaterialize(&op))
        continue;

      FailureOr<Value> materialized = materializeRoot(&op);
      if (failed(materialized))
        return failure();
      valueToTensor[op.getResult(0)] = *materialized;
    }

    auto yield = cast<YieldOp>(body.getTerminator());
    Value yielded = yield.getValues().front();
    FailureOr<Value> result = getTensorForExpr(yielded);
    if (failed(result))
      return failure();

    scope.replaceAllUsesWith(*result);
    if (beforeErase) {
      if (loweredOps) {
        beforeErase(scope.getOperation(), *loweredOps);
      } else {
        DenseMap<Operation *, Operation *> empty;
        beforeErase(scope.getOperation(), empty);
      }
    }
    scope.erase();
    return success();
  }

private:
  struct InputDescriptor {
    Value tensor;
    AffineMap map;
    Value exprValue;
    AtOp at;
  };

  bool hasExprResult(Operation *op) const {
    return op->getNumResults() == 1 && isa<ExprType>(op->getResult(0).getType());
  }

  bool isYielded(Value value) {
    auto yield = cast<YieldOp>(scope.getBody().front().getTerminator());
    return llvm::is_contained(yield.getValues(), value);
  }

  bool shouldMaterialize(Operation *op) {
    if (isa<AtOp>(op))
      return false;

    if (isa<ConstantOp>(op))
      return isYielded(op->getResult(0));

    if (!getImportGroup(op))
      return true;

    Value result = op->getResult(0);
    if (isYielded(result))
      return true;

    return llvm::any_of(result.getUsers(), [&](Operation *user) {
      if (isa<YieldOp>(user))
        return true;
      return !sameImportGroup(op, user);
    });
  }

  LogicalResult discoverAxisSizes() {
    for (auto [axisAttr, extent] :
         llvm::zip_equal(scope.getAxes().getAxes(), scope.getStaticExtents())) {
      StringRef axis = cast<AxisAttr>(axisAttr).getName().getValue();
      if (extent == ShapedType::kDynamic)
        return scope.emitOpError(
                   "ta-to-linalg lowering does not yet support dynamic extent for axis '")
               << axis << "'";
      axisSizes[axis] = extent;
    }

    return success();
  }

  FailureOr<int64_t> getAxisSize(Operation *op, StringRef axis) {
    auto it = axisSizes.find(axis);
    if (it == axisSizes.end())
      return op->emitOpError("lowering does not know the size of axis '") << axis << "'";
    return it->second;
  }

  FailureOr<RankedTensorType> tensorTypeFor(Operation *op, ExprType expr) {
    SmallVector<int64_t> shape;
    for (StringRef axis : axisNames(expr)) {
      FailureOr<int64_t> size = getAxisSize(op, axis);
      if (failed(size))
        return failure();
      shape.push_back(*size);
    }
    return RankedTensorType::get(shape, expr.getElementType());
  }

  FailureOr<Value> getTensorForExpr(Value expr) {
    auto it = valueToTensor.find(expr);
    if (it != valueToTensor.end())
      return it->second;

    Operation *def = expr.getDefiningOp();
    if (!def)
      return emitError(expr.getLoc()) << "cannot lower unmaterialized block argument expr";
    if (!hasExprResult(def))
      return def->emitOpError("expected expression-producing op");

    FailureOr<Value> materialized = materializeRoot(def);
    if (failed(materialized))
      return failure();
    valueToTensor[expr] = *materialized;
    return *materialized;
  }

  bool canInline(Operation *op, Operation *root) const {
    if (isa<ConstantOp>(op))
      return true;
    return sameImportGroup(op, root);
  }

  FailureOr<AffineMap> mapForExprAxes(Operation *op, ExprType expr, ArrayRef<StringRef> loopAxes) {
    SmallVector<AffineExpr> results;
    for (StringRef axis : axisNames(expr)) {
      FailureOr<unsigned> position = findAxis(op, loopAxes, axis);
      if (failed(position))
        return failure();
      results.push_back(builder.getAffineDimExpr(*position));
    }
    return AffineMap::get(loopAxes.size(), 0, results, context);
  }

  FailureOr<AffineMap> mapForAt(AtOp at, ArrayRef<StringRef> loopAxes) {
    SmallVector<AffineExpr> results;
    FailureOr<SmallVector<ScopeIndexOperand>> indices =
        decodeScopeIndexOperands(at.getOperation(), scope, at.getIndices(),
                                 "lowering only supports scope-axis or constant indices");
    if (failed(indices))
      return failure();

    for (const ScopeIndexOperand &index : *indices) {
      if (index.isAxis()) {
        FailureOr<unsigned> position =
            findAxis(at.getOperation(), loopAxes, index.axes.front().getName().getValue());
        if (failed(position))
          return failure();
        results.push_back(builder.getAffineDimExpr(*position));
        continue;
      }

      if (index.isLinearized()) {
        AffineExpr linearized = builder.getAffineConstantExpr(0);
        for (auto [axisIndex, axis] : llvm::enumerate(index.axes)) {
          FailureOr<unsigned> position =
              findAxis(at.getOperation(), loopAxes, axis.getName().getValue());
          if (failed(position))
            return failure();

          int64_t stride = 1;
          for (int64_t basis : ArrayRef(index.staticBasis).drop_front(axisIndex))
            stride *= basis;
          linearized = linearized + builder.getAffineDimExpr(*position) * stride;
        }
        results.push_back(linearized);
        continue;
      }

      results.push_back(builder.getAffineConstantExpr(*index.constant));
    }
    return AffineMap::get(loopAxes.size(), 0, results, context);
  }

  FailureOr<unsigned> addTensorInput(Value expr, Value tensor, AffineMap map) {
    for (auto [index, descriptor] : llvm::enumerate(inputs)) {
      if (descriptor.exprValue == expr && descriptor.tensor == tensor && descriptor.map == map)
        return index;
    }
    inputs.push_back({tensor, map, expr, AtOp()});
    return inputs.size() - 1;
  }

  FailureOr<unsigned> addAtInput(AtOp at, ArrayRef<StringRef> loopAxes) {
    FailureOr<AffineMap> map = mapForAt(at, loopAxes);
    if (failed(map))
      return failure();
    for (auto [index, descriptor] : llvm::enumerate(inputs)) {
      if (descriptor.at == at && descriptor.map == *map)
        return index;
    }
    inputs.push_back({at.getSource(), *map, Value(), at});
    return inputs.size() - 1;
  }

  LogicalResult collectInputs(Value value, Operation *root, ArrayRef<StringRef> loopAxes) {
    Operation *def = value.getDefiningOp();
    if (!def)
      return emitError(value.getLoc()) << "cannot lower block argument expr";

    if (auto constant = dyn_cast<ConstantOp>(def))
      return success();
    if (auto at = dyn_cast<AtOp>(def))
      return success(addAtInput(at, loopAxes));

    if (canInline(def, root)) {
      for (Value operand : def->getOperands()) {
        if (isa<ExprType>(operand.getType()) && failed(collectInputs(operand, root, loopAxes)))
          return failure();
      }
      return success();
    }

    FailureOr<Value> tensor = getTensorForExpr(value);
    if (failed(tensor))
      return failure();
    FailureOr<AffineMap> map = mapForExprAxes(def, cast<ExprType>(value.getType()), loopAxes);
    if (failed(map))
      return failure();
    return success(addTensorInput(value, *tensor, *map));
  }

  FailureOr<Value> materializeRoot(Operation *root) {
    inputs.clear();

    auto resultExpr = cast<ExprType>(root->getResult(0).getType());
    SmallVector<StringRef> loopAxes;
    SmallVector<utils::IteratorType> iterators;
    Value payloadValue;
    ReduceOp reduce = dyn_cast<ReduceOp>(root);

    if (reduce) {
      payloadValue = reduce.getInput();
      SmallVector<StringRef> reductionAxes = axisNames(reduce.getAxes());
      loopAxes = axisNames(resultExpr);
      iterators.assign(loopAxes.size(), utils::IteratorType::parallel);
      for (StringRef axis : reductionAxes) {
        if (containsAxis(loopAxes, axis))
          continue;
        loopAxes.push_back(axis);
        iterators.push_back(utils::IteratorType::reduction);
      }
      if (failed(collectInputs(payloadValue, root, loopAxes)))
        return failure();
    } else {
      loopAxes = axisNames(resultExpr);
      iterators.assign(loopAxes.size(), utils::IteratorType::parallel);
      for (Value operand : root->getOperands()) {
        if (isa<ExprType>(operand.getType()) && failed(collectInputs(operand, root, loopAxes)))
          return failure();
      }
    }

    FailureOr<RankedTensorType> resultType = tensorTypeFor(root, resultExpr);
    if (failed(resultType))
      return failure();

    SmallVector<Value> inputTensors;
    SmallVector<AffineMap> maps;
    for (const InputDescriptor &input : inputs) {
      inputTensors.push_back(input.tensor);
      maps.push_back(input.map);
    }

    FailureOr<AffineMap> outputMap = mapForExprAxes(root, resultExpr, loopAxes);
    if (failed(outputMap))
      return failure();
    maps.push_back(*outputMap);

    Value init = createInitTensor(root, *resultType, reduce);
    auto generic = linalg::GenericOp::create(
        builder, root->getLoc(), TypeRange{*resultType}, inputTensors, ValueRange{init}, maps,
        iterators, [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          buildLinalgBody(nestedBuilder, nestedLoc, args, root, reduce, loopAxes);
        });
    if (loweredOps)
      (*loweredOps)[root] = generic.getOperation();
    return generic->getResult(0);
  }

  Value createInitTensor(Operation *root, RankedTensorType resultType, ReduceOp reduce) {
    auto empty = tensor::EmptyOp::create(builder, root->getLoc(), resultType.getShape(),
                                         resultType.getElementType());
    if (!reduce)
      return empty.getResult();

    Value identity = createIdentity(root->getLoc(), resultType.getElementType(), reduce.getKind());
    auto fill = linalg::FillOp::create(builder, root->getLoc(), TypeRange{resultType},
                                       ValueRange{identity}, ValueRange{empty.getResult()});
    return fill.getResult(0);
  }

  Value createIdentity(Location identityLoc, Type type, ReduceKind kind) {
    auto floatType = cast<FloatType>(type);
    const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
    APFloat value = APFloat::getZero(semantics);
    switch (kind) {
    case ReduceKind::Add:
      value = APFloat::getZero(semantics);
      break;
    case ReduceKind::Mul:
      value = APFloat(semantics, "1.0");
      break;
    case ReduceKind::Max:
      value = APFloat::getInf(semantics, /*Negative=*/true);
      break;
    case ReduceKind::Min:
      value = APFloat::getInf(semantics, /*Negative=*/false);
      break;
    }
    return arith::ConstantOp::create(builder, identityLoc, FloatAttr::get(floatType, value));
  }

  void buildLinalgBody(OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args,
                       Operation *root, ReduceOp reduce, ArrayRef<StringRef> loopAxes) {
    scalarValues.clear();
    for (auto [index, descriptor] : llvm::enumerate(inputs)) {
      if (descriptor.at)
        scalarValues[descriptor.at.getResult()] = args[index];
      if (descriptor.exprValue)
        scalarValues[descriptor.exprValue] = args[index];
    }

    Value yielded;
    if (reduce) {
      Value payload = buildScalar(nestedBuilder, nestedLoc, reduce.getInput(), root, loopAxes);
      Value accumulator = args[inputs.size()];
      yielded = buildCombiner(nestedBuilder, nestedLoc, reduce.getKind(), accumulator, payload);
    } else {
      yielded = buildScalar(nestedBuilder, nestedLoc, root->getResult(0), root, loopAxes);
    }
    linalg::YieldOp::create(nestedBuilder, nestedLoc, yielded);
  }

  Value buildScalar(OpBuilder &nestedBuilder, Location nestedLoc, Value value, Operation *root,
                    ArrayRef<StringRef> loopAxes) {
    auto it = scalarValues.find(value);
    if (it != scalarValues.end())
      return it->second;

    Operation *def = value.getDefiningOp();
    if (auto constant = dyn_cast<ConstantOp>(def)) {
      Value scalar = arith::ConstantOp::create(nestedBuilder, nestedLoc, constant.getValue());
      scalarValues[value] = scalar;
      return scalar;
    }

    if (auto index = dyn_cast<IndexOp>(def)) {
      StringRef axis = axisNames(cast<ExprType>(index.getResult().getType())).front();
      FailureOr<unsigned> position = findAxis(index.getOperation(), loopAxes, axis);
      if (failed(position))
        llvm_unreachable("verified ta.index axis was not present in lowering loop axes");
      Value scalar = linalg::IndexOp::create(nestedBuilder, nestedLoc, *position);
      Type elementType = cast<ExprType>(index.getResult().getType()).getElementType();
      if (!elementType.isIndex())
        scalar = arith::IndexCastOp::create(nestedBuilder, nestedLoc, elementType, scalar);
      scalarValues[value] = scalar;
      return scalar;
    }

    SmallVector<Value> operands;
    operands.reserve(def->getNumOperands());
    for (Value operand : def->getOperands()) {
      if (isa<ExprType>(operand.getType()))
        operands.push_back(buildScalar(nestedBuilder, nestedLoc, operand, root, loopAxes));
    }

    Value scalar;
    if (isa<ExtFOp>(def)) {
      auto resultType = cast<ExprType>(def->getResult(0).getType()).getElementType();
      scalar = arith::ExtFOp::create(nestedBuilder, nestedLoc, resultType, operands[0]);
    } else if (isa<TruncFOp>(def)) {
      auto resultType = cast<ExprType>(def->getResult(0).getType()).getElementType();
      scalar = arith::TruncFOp::create(nestedBuilder, nestedLoc, resultType, operands[0]);
    } else if (auto cmpi = dyn_cast<CmpIOp>(def)) {
      scalar = arith::CmpIOp::create(nestedBuilder, nestedLoc, cmpi.getPredicate(), operands[0],
                                     operands[1]);
    } else if (isa<SelectOp>(def)) {
      scalar =
          arith::SelectOp::create(nestedBuilder, nestedLoc, operands[0], operands[1], operands[2]);
    } else if (isa<AddFOp>(def)) {
      scalar = arith::AddFOp::create(nestedBuilder, nestedLoc, operands[0], operands[1]);
    } else if (isa<SubFOp>(def)) {
      scalar = arith::SubFOp::create(nestedBuilder, nestedLoc, operands[0], operands[1]);
    } else if (isa<MulFOp>(def)) {
      scalar = arith::MulFOp::create(nestedBuilder, nestedLoc, operands[0], operands[1]);
    } else if (isa<DivFOp>(def)) {
      scalar = arith::DivFOp::create(nestedBuilder, nestedLoc, operands[0], operands[1]);
    } else if (isa<MaximumFOp>(def)) {
      scalar = arith::MaximumFOp::create(nestedBuilder, nestedLoc, operands[0], operands[1]);
    } else if (isa<MinimumFOp>(def)) {
      scalar = arith::MinimumFOp::create(nestedBuilder, nestedLoc, operands[0], operands[1]);
    } else if (isa<ExpOp>(def)) {
      scalar = math::ExpOp::create(nestedBuilder, nestedLoc, operands[0]);
    } else if (isa<Exp2Op>(def)) {
      scalar = math::Exp2Op::create(nestedBuilder, nestedLoc, operands[0]);
    } else {
      llvm_unreachable("unsupported scalar op reached after legality checks");
    }

    scalarValues[value] = scalar;
    return scalar;
  }

  Value buildCombiner(OpBuilder &nestedBuilder, Location nestedLoc, ReduceKind kind, Value lhs,
                      Value rhs) {
    switch (kind) {
    case ReduceKind::Add:
      return arith::AddFOp::create(nestedBuilder, nestedLoc, lhs, rhs);
    case ReduceKind::Mul:
      return arith::MulFOp::create(nestedBuilder, nestedLoc, lhs, rhs);
    case ReduceKind::Max:
      return arith::MaximumFOp::create(nestedBuilder, nestedLoc, lhs, rhs);
    case ReduceKind::Min:
      return arith::MinimumFOp::create(nestedBuilder, nestedLoc, lhs, rhs);
    }
    llvm_unreachable("unknown reduce kind");
  }

  ScopeOp scope;
  OpBuilder &builder;
  MLIRContext *context;
  Location loc;
  llvm::StringMap<int64_t> axisSizes;
  DenseMap<Value, Value> valueToTensor;
  SmallVector<InputDescriptor> inputs;
  DenseMap<Value, Value> scalarValues;
  DenseMap<Operation *, Operation *> *loweredOps;
  llvm::function_ref<void(Operation *, const DenseMap<Operation *, Operation *> &)> beforeErase;
};

struct LowerTAToLinalgPass : public PassWrapper<LowerTAToLinalgPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTAToLinalgPass)

  StringRef getArgument() const final { return "ta-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower supported ta.scope expression graphs to linalg.generic";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    OpBuilder builder(getOperation());
    if (failed(lowerTAToLinalg(getOperation(), builder))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

LogicalResult lowerTAToLinalg(
    Operation *target, OpBuilder &builder, DenseMap<Operation *, Operation *> *loweredOps,
    llvm::function_ref<void(Operation *, const DenseMap<Operation *, Operation *> &)> beforeErase) {
  SmallVector<ScopeOp> scopes;
  if (auto scope = dyn_cast<ScopeOp>(target)) {
    scopes.push_back(scope);
  } else {
    target->walk([&](ScopeOp scope) { scopes.push_back(scope); });
  }

  for (ScopeOp scope : scopes) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(scope);
    DenseMap<Operation *, Operation *> scopeLoweredOps;
    DenseMap<Operation *, Operation *> *activeLoweredOps =
        (loweredOps || beforeErase) ? &scopeLoweredOps : nullptr;
    ScopeLowering lowering(scope, builder, activeLoweredOps, beforeErase);
    if (failed(lowering.run()))
      return failure();
    if (loweredOps) {
      for (auto [taOp, linalgOp] : scopeLoweredOps)
        (*loweredOps)[taOp] = linalgOp;
    }
  }
  return success();
}

void registerTAToLinalgPass() { PassRegistration<LowerTAToLinalgPass>(); }

} // namespace ta
