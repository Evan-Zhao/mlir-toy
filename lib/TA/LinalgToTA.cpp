#include "TA/TAAttrs.h"
#include "TA/TAOps.h"
#include "TA/TAPasses.h"
#include "TA/TATypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

namespace ta {

using namespace mlir;

namespace {

struct Axis {
  std::string name;
  int64_t extent = ShapedType::kDynamic;
};

using AxisPack = SmallVector<Axis, 2>;
using TensorAxes = SmallVector<AxisPack, 4>;

template <typename IdT> class AxisUnionFind {
public:
  IdT add() {
    IdT id = parents.size();
    parents.push_back(id);
    return id;
  }

  IdT find(IdT id) {
    IdT parent = parents[id];
    if (parent == id)
      return id;
    parents[id] = find(parent);
    return parents[id];
  }

  // Returns true if a union was performed, or false if the two sets were already unified.
  bool unite(IdT lhs, IdT rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return false;
    if (lhs > rhs)
      std::swap(lhs, rhs);
    parents[rhs] = lhs;
    return true;
  }

  SmallVector<IdT> getUniqueRoots() {
    DenseSet<IdT> seen;
    SmallVector<IdT> roots;
    for (IdT id = 0, e = parents.size(); id < e; ++id) {
      IdT root = find(id);
      if (seen.insert(root).second)
        roots.push_back(root);
    }
    return roots;
  }

  size_t size() const { return parents.size(); }

private:
  SmallVector<IdT> parents;
};

struct FunctionAxisInfo {
  DenseMap<Value, TensorAxes> valueAxes;
  DenseMap<linalg::GenericOp, TensorAxes> loopAxisMap;
  DenseMap<OpOperand *, TensorAxes> operandAxes;
  SmallVector<Axis> scopeAxes;
};

class FunctionAxisDiscovery {
  using AxisId = unsigned;
  using AxisIdPack = SmallVector<AxisId, 2>;
  using TensorAxisIds = SmallVector<AxisIdPack, 4>;

public:
  FunctionAxisDiscovery(func::FuncOp func, func::ReturnOp returnOp)
      : func(func), returnOp(returnOp) {}

  FailureOr<FunctionAxisInfo> run(RankedTensorType resultType) {
    if (failed(discoverAxes(resultType)))
      return failure();
    if (failed(discoverAxisExtents()))
      return failure();
    return canonicalizeAndBuildInfo();
  }

private:
  AxisId newAxis(char prefix) {
    AxisId id = axisUnions.add();
    axes.push_back(Axis{.name = prefix + std::to_string(id)});
    return id;
  }

  TensorAxisIds makeResultAxes(RankedTensorType type) {
    TensorAxisIds axes;
    for (int64_t i = 0; i < type.getRank(); ++i)
      axes.push_back(AxisIdPack{newAxis('i')});
    return axes;
  }

  LogicalResult unifyAxisPacks(Location loc, const AxisIdPack &existing, const AxisIdPack &desired,
                               bool &changed) {
    if (existing.empty() || desired.empty())
      return success();
    if (existing.size() != desired.size()) {
      emitError(loc) << "incompatible logical axis packs: existing pack has " << existing.size()
                     << " axes, but newly required pack has " << desired.size() << " axes";
      return failure();
    }
    for (auto [lhs, rhs] : zip_equal(existing, desired))
      changed |= axisUnions.unite(lhs, rhs);
    return success();
  }

  LogicalResult mergeValueAxes(Value value, const TensorAxisIds &desired, bool &changed) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return emitError(value.getLoc()) << "expected ranked tensor value";
    if (static_cast<int64_t>(desired.size()) != type.getRank())
      return emitError(value.getLoc()) << "axis rank does not match tensor rank";

    // If the value is not yet in the map, just map it to `desired`.
    auto [it, inserted] = valueAxes.try_emplace(value, desired);
    if (inserted) {
      changed = true;
      return success();
    }
    // Otherwise, run a unification process to merge the existing axes with `desired`.
    for (auto [axis, desiredAxis] : zip_equal(it->second, desired)) {
      if (failed(unifyAxisPacks(value.getLoc(), axis, desiredAxis, changed)))
        return failure();
    }
    return success();
  }

  FailureOr<TensorAxisIds> sourceAxesForCollapse(tensor::CollapseShapeOp op,
                                                 const TensorAxisIds &resultAxes) {
    TensorAxisIds sourceAxes(op.getSrcType().getRank());
    for (auto [resultDim, group] : enumerate(op.getReassociationIndices())) {
      const AxisIdPack &collapsed = resultAxes[resultDim];
      if (group.size() == 1) {
        sourceAxes[group.front()] = collapsed;
        continue;
      }
      if (collapsed.size() != group.size())
        return op.emitOpError("cannot split collapsed logical axis pack");
      for (auto [axisIndex, sourceDim] : enumerate(group))
        sourceAxes[sourceDim] = AxisIdPack{collapsed[axisIndex]};
    }
    return sourceAxes;
  }

  FailureOr<TensorAxisIds> sourceAxesForExpand(tensor::ExpandShapeOp op,
                                               const TensorAxisIds &resultAxes) {
    TensorAxisIds sourceAxes(op.getSrcType().getRank());
    for (auto [sourceDim, group] : enumerate(op.getReassociationIndices())) {
      AxisIdPack pack;
      for (int64_t resultDim : group)
        pack.append(resultAxes[resultDim].begin(), resultAxes[resultDim].end());
      sourceAxes[sourceDim] = pack;
    }
    return sourceAxes;
  }

  FailureOr<bool> discoverGenericAxes(linalg::GenericOp op) {
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
    unsigned numInputs = op.getInputs().size();

    auto &loopAxes = loopAxisMap[op];
    if (loopAxes.empty())
      loopAxes.resize(iterators.size());

    bool changed = false;
    for (auto [resultNumber, result] : llvm::enumerate(op->getResults())) {
      auto resultType = dyn_cast<RankedTensorType>(result.getType());
      if (!resultType)
        continue;

      auto it = valueAxes.find(result);
      if (it == valueAxes.end())
        continue;

      unsigned outputMapIndex = numInputs + resultNumber;
      if (outputMapIndex >= maps.size())
        return op.emitOpError("missing output indexing map");
      if (failed(assignLoopAxesFromOutputMap(op, maps[outputMapIndex], it->second, loopAxes)))
        return failure();
    }

    for (auto [index, iterator] : enumerate(iterators)) {
      if (iterator == utils::IteratorType::reduction && loopAxes[index].empty()) {
        loopAxes[index] = AxisIdPack{newAxis('r')};
        changed = true;
      }
    }

    for (auto [index, input] : llvm::enumerate(op.getInputs())) {
      if (dyn_cast<RankedTensorType>(input.getType())) {
        FailureOr<TensorAxisIds> axes = projectMap(op, maps[index], loopAxes);
        if (failed(axes))
          return failure();
        operandAxes[&op->getOpOperand(index)] = *axes;
        // Intentionally ignoring the result of mergeValueAxes because we continue anyways.
        auto _ = mergeValueAxes(input, *axes, changed);
      }
    }

    return changed;
  }

  LogicalResult discoverAxes(RankedTensorType resultType) {
    TensorAxisIds resultAxes = makeResultAxes(resultType);
    bool changed = false;
    if (failed(mergeValueAxes(returnOp.getOperand(0), resultAxes, changed)))
      return failure();

    bool keepGoing = true;
    while (keepGoing) {
      keepGoing = false;
      for (Operation &op : llvm::reverse(func.front().without_terminator())) {
        if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
          auto it = valueAxes.find(collapse.getResult());
          if (it == valueAxes.end())
            continue;
          FailureOr<TensorAxisIds> sourceAxes = sourceAxesForCollapse(collapse, it->second);
          if (failed(sourceAxes) ||
              failed(mergeValueAxes(collapse.getSrc(), *sourceAxes, keepGoing)))
            return failure();
          continue;
        }
        // Note the difference from the collapse case: we don't enforce that the axes be merged. If
        // there is a layout conflict, it will go downstream and the FunctionEmitter will produce a
        // `ta.subst` operation.
        if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
          auto it = valueAxes.find(expand.getResult());
          if (it == valueAxes.end())
            continue;
          FailureOr<TensorAxisIds> sourceAxes = sourceAxesForExpand(expand, it->second);
          if (failed(sourceAxes))
            return failure();
          // ExpandShapeOp directly admits the axes of its source as the axes of its result. It
          // doesn't do mergeValueAxes.
          auto [_, inserted] = valueAxes.try_emplace(expand.getSrc(), *sourceAxes);
          changed |= inserted;
          continue;
        }

        if (auto generic = dyn_cast<linalg::GenericOp>(&op)) {
          FailureOr<bool> genericChanged = discoverGenericAxes(generic);
          if (failed(genericChanged))
            return failure();
          keepGoing |= *genericChanged;
        }
      }
    }

    return success();
  }

  LogicalResult assignLoopAxesFromOutputMap(Operation *op, AffineMap map,
                                            const TensorAxisIds &resultAxes,
                                            MutableArrayRef<AxisIdPack> loopAxes) {
    if (map.getNumResults() != resultAxes.size())
      return op->emitOpError("output indexing map rank does not match result axis rank");

    for (auto [resultIndex, expr] : enumerate(map.getResults())) {
      if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
        bool changed = false;
        AxisIdPack &assigned = loopAxes[dim.getPosition()];
        if (assigned.empty())
          assigned = resultAxes[resultIndex];
        else if (failed(unifyAxisPacks(op->getLoc(), assigned, resultAxes[resultIndex], changed)))
          return failure();
      } else if (!isa<AffineConstantExpr>(expr))
        return op->emitOpError("non-projected output indexing maps are not supported");
    }
    return success();
  }

  FailureOr<TensorAxisIds> projectMap(Operation *op, AffineMap map, ArrayRef<AxisIdPack> loopAxes) {
    TensorAxisIds axes;
    for (AffineExpr expr : map.getResults()) {
      if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
        axes.push_back(loopAxes[dim.getPosition()]);
        continue;
      }
      if (isa<AffineConstantExpr>(expr)) {
        axes.push_back(AxisIdPack{});
        continue;
      }
      return op->emitOpError("non-projected input indexing maps are not supported");
    }
    return axes;
  }

  FunctionAxisInfo canonicalizeAndBuildInfo() {
    unsigned nextElementwise = 0;
    unsigned nextReduction = 0;
    unsigned nextOther = 0;
    SmallVector<Axis> scopeAxes;
    for (AxisId root : axisUnions.getUniqueRoots()) {
      auto &axisName = axes[root].name;
      switch (axisName.front()) {
      case 'i':
        axisName = "i" + std::to_string(nextElementwise++);
        break;
      case 'r':
        axisName = "r" + std::to_string(nextReduction++);
        break;
      default:
        axisName = "x" + std::to_string(nextOther++);
        break;
      }
      scopeAxes.push_back(axes[root]);
    }

    auto mapIdsToAxes = [&](const auto &inMap, auto &outMap) {
      for (auto &[value, tensorAxisIds] : inMap) {
        outMap[value] = llvm::map_to_vector<4>(tensorAxisIds, [&](const AxisIdPack &pack) {
          return llvm::map_to_vector<2>(pack, [&](AxisId id) { return axes[axisUnions.find(id)]; });
        });
      }
    };
    DenseMap<Value, TensorAxes> valueAxes;
    mapIdsToAxes(this->valueAxes, valueAxes);
    DenseMap<linalg::GenericOp, TensorAxes> loopAxisMap;
    mapIdsToAxes(this->loopAxisMap, loopAxisMap);
    DenseMap<OpOperand *, TensorAxes> operandAxes;
    mapIdsToAxes(this->operandAxes, operandAxes);
    return FunctionAxisInfo{.valueAxes = std::move(valueAxes),
                            .loopAxisMap = std::move(loopAxisMap),
                            .operandAxes = std::move(operandAxes),
                            .scopeAxes = scopeAxes};
  }

  LogicalResult discoverAxisExtents() {
    for (auto &[value, axes_] : valueAxes) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type)
        continue;
      for (auto [pack, extent] : llvm::zip_equal(axes_, type.getShape())) {
        if (pack.size() != 1 || extent == ShapedType::kDynamic)
          continue;
        auto &axis = axes[axisUnions.find(pack.front())];
        if (axis.extent == ShapedType::kDynamic)
          axis.extent = extent;
        else if (axis.extent != extent)
          return emitError(func.getLoc())
                 << "conflicting imported extents for axis '" << axis.name << "'";
      }
    }
    return success();
  }

  func::FuncOp func;
  func::ReturnOp returnOp;
  AxisUnionFind<AxisId> axisUnions;
  DenseMap<Value, TensorAxisIds> valueAxes;
  DenseMap<linalg::GenericOp, TensorAxisIds> loopAxisMap;
  DenseMap<OpOperand *, TensorAxisIds> operandAxes;
  SmallVector<Axis> axes;
};

static TypedAttr splatScalarConstant(arith::ConstantOp constant) {
  auto elements = dyn_cast<DenseElementsAttr>(constant.getValue());
  if (!elements || !elements.isSplat())
    return {};
  return dyn_cast<TypedAttr>(elements.getSplatValue<Attribute>());
}

static SmallVector<std::string> flattenAxes(const TensorAxes &axes) {
  SmallVector<std::string> flat;
  for (const AxisPack &pack : axes)
    for (const auto &[name, _] : pack)
      flat.push_back(name);
  return flat;
}

class ScopedTABuilder {
public:
  ScopedTABuilder(Operation *insertBefore, Location loc, RankedTensorType resultType,
                  ArrayRef<Axis> scopeAxes)
      : context(insertBefore->getContext()), loc(loc), builder(insertBefore) {
    Block *body = new Block();
    SmallVector<std::string> axisNames;
    SmallVector<int64_t> staticExtents;
    for (const auto &[axisName, extent] : scopeAxes) {
      if (axes.contains(axisName))
        continue;
      auto arg = body->addArgument(builder.getIndexType(), loc);
      axisNames.push_back(axisName);
      staticExtents.push_back(extent);
      axes.try_emplace(axisName, arg);
    }
    scope = ScopeOp::create(builder, loc, resultType, ValueRange{}, getAxesAttr(axisNames),
                            DenseI64ArrayAttr::get(context, staticExtents));
    scope.getBody().push_back(body);
    builder.setInsertionPointToStart(body);
  }

  class ImportGroupGuard {
  public:
    ImportGroupGuard(ScopedTABuilder &ta, int64_t group) : ta(ta), oldGroup(ta.importGroup) {
      ta.importGroup = group;
    }

    ~ImportGroupGuard() { ta.importGroup = oldGroup; }

  private:
    ScopedTABuilder &ta;
    std::optional<int64_t> oldGroup;
  };

  ScopeOp getScope() const { return scope; }

  AxesAttr getAxesAttr(ArrayRef<std::string> names) const {
    SmallVector<Attribute> axes;
    DenseSet<StringRef> seen;
    for (StringRef name : names) {
      if (!seen.insert(name).second)
        continue;
      axes.push_back(AxisAttr::get(context, name));
    }
    return AxesAttr::get(context, ArrayAttr::get(context, axes));
  }

  ExprType expr(Type elementType, ArrayRef<std::string> axes) const {
    return ExprType::get(context, elementType, getAxesAttr(axes));
  }

  SmallVector<std::string> unionAxes(ValueRange operands) const {
    DenseSet<StringRef> seen;
    SmallVector<std::string> result;
    for (Value operand : operands) {
      auto exprType = cast<ExprType>(operand.getType());
      for (Attribute attr : exprType.getAxes().getAxes()) {
        auto axis = cast<AxisAttr>(attr);
        StringRef name = axis.getName().getValue();
        if (!seen.insert(name).second)
          continue;
        result.push_back(name.str());
      }
    }
    return result;
  }

  FailureOr<Value> at(Value source, const TensorAxes &dimAxes, Type elementType) {
    SmallVector<Value> indices;
    for (const AxisPack &pack : dimAxes) {
      if (pack.empty()) {
        indices.push_back(indexZero());
        continue;
      }
      if (pack.size() == 1) {
        auto &axisName = pack.front().name;
        auto it = axes.find(axisName);
        if (it == axes.end())
          return emitError(loc) << "missing materialized ta axis '" << axisName << "'";
        indices.push_back(it->second);
        continue;
      }

      SmallVector<Value> multiIndex;
      SmallVector<int64_t> basis;
      for (auto &[name, extent] : pack) {
        auto it = axes.find(name);
        if (it == axes.end())
          return emitError(loc) << "missing materialized ta axis '" << name << "'";
        if (extent == ShapedType::kDynamic)
          return emitError(loc) << "cannot linearize multiple dynamic logical axes into "
                                << "one tensor dimension";
        multiIndex.push_back(it->second);
        basis.push_back(extent);
      }
      indices.push_back(affine::AffineLinearizeIndexOp::create(builder, loc, multiIndex, basis,
                                                               /*disjoint=*/true));
    }

    SmallVector<std::string> resultAxes = flattenAxes(dimAxes);
    auto op = AtOp::create(builder, loc, expr(elementType, resultAxes), source, indices);
    return annotate(op);
  }

  Value constant(TypedAttr value) {
    auto op = ConstantOp::create(builder, loc, expr(value.getType(), {}), value);
    return annotate(op);
  }

  FailureOr<Value> index(const std::string &axisName, Type elementType) {
    if (!elementType)
      elementType = builder.getIndexType();
    auto it = axes.find(axisName);
    if (it == axes.end())
      return emitError(loc) << "missing materialized ta axis '" << axisName << "'";
    auto op = IndexOp::create(builder, loc, expr(elementType, {axisName}), it->second);
    return annotate(op);
  }

  Value cmpi(arith::CmpIPredicate predicate, Value lhs, Value rhs) {
    SmallVector<std::string> axes = unionAxes({lhs, rhs});
    auto op =
        CmpIOp::create(builder, loc, expr(IntegerType::get(context, 1), axes), predicate, lhs, rhs);
    return annotate(op);
  }

  Value select(Value condition, Value trueValue, Value falseValue) {
    SmallVector<std::string> axes = unionAxes({trueValue, falseValue, condition});
    auto trueType = cast<ExprType>(trueValue.getType());
    auto op = SelectOp::create(builder, loc, expr(trueType.getElementType(), axes), condition,
                               trueValue, falseValue);
    return annotate(op);
  }

  template <typename OpTy> Value unary(Type elementType, Value input) {
    SmallVector<std::string> axes = unionAxes({input});
    auto op = OpTy::create(builder, loc, expr(elementType, axes), input);
    return annotate(op);
  }

  template <typename OpTy> Value binary(Type elementType, Value lhs, Value rhs) {
    SmallVector<std::string> axes = unionAxes({lhs, rhs});
    auto op = OpTy::create(builder, loc, expr(elementType, axes), lhs, rhs);
    return annotate(op);
  }

  Value reduce(ReduceKind kind, Value input, ArrayRef<std::string> reductionAxes, Type elementType,
               ArrayRef<std::string> resultAxes) {
    auto op = ReduceOp::create(builder, loc, expr(elementType, resultAxes), kind, input, Value(),
                               getAxesAttr(reductionAxes));
    return annotate(op);
  }

  Value subst(Value input, ArrayRef<std::string> fromAxes, ArrayRef<std::string> toAxes,
              Type elementType) {
    auto op = SubstOp::create(builder, loc, expr(elementType, toAxes), input, getAxesAttr(fromAxes),
                              getAxesAttr(toAxes));
    return annotate(op);
  }

  void yield(Value value) { YieldOp::create(builder, loc, value); }

private:
  template <typename OpTy> Value annotate(OpTy op) const {
    if (!importGroup)
      return op.getResult();
    op->setAttr("ta.import_group", IntegerAttr::get(IntegerType::get(context, 64), *importGroup));
    return op.getResult();
  }

  Value indexZero() {
    if (zero)
      return zero;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&scope.getBody().front());
    zero = arith::ConstantIndexOp::create(builder, loc, 0);
    return zero;
  }

  MLIRContext *context;
  const Location loc;
  OpBuilder builder;

  Value zero;
  ScopeOp scope;
  std::optional<int64_t> importGroup;
  llvm::MapVector<std::string, Value, llvm::StringMap<unsigned>> axes;
};

class FunctionEmitter {
public:
  FunctionEmitter(func::FuncOp func, func::ReturnOp returnOp, FunctionAxisInfo axisInfo_)
      : func(func), returnOp(returnOp), axisInfo(std::move(axisInfo_)),
        ta(returnOp, func.getLoc(), cast<RankedTensorType>(func.getResultTypes().front()),
           axisInfo.scopeAxes) {}

  LogicalResult run() {
    if (failed(emitForward()))
      return failure();

    auto result = valueMap.lookupOrNull(returnOp.getOperand(0));
    if (!result)
      return emitError(returnOp.getLoc()) << "failed to translate returned tensor";
    ta.yield(result);

    returnOp.getOperation()->setOperands(ta.getScope().getResult());
    eraseUnusedScopeOps();
    return success();
  }

private:
  FailureOr<Value> emitGeneric(linalg::GenericOp op, unsigned resultNumber) {
    ScopedTABuilder::ImportGroupGuard guard(ta, nextImportGroup++);

    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    SmallVector<utils::IteratorType> iterators = linalgOp.getIteratorTypesArray();
    unsigned numInputs = op.getInputs().size();

    auto it = axisInfo.valueAxes.find(op->getResult(resultNumber));
    if (it == axisInfo.valueAxes.end())
      return op.emitOpError("missing axis info for the result of this op");
    SmallVector<std::string> flatResultAxes = flattenAxes(it->second);
    TensorAxes loopAxes = axisInfo.loopAxisMap.lookup(op);

    SmallVector<Value> inputExprs;
    inputExprs.reserve(numInputs);
    for (auto [index, input] : llvm::enumerate(op.getInputs())) {
      if (dyn_cast<RankedTensorType>(input.getType())) {
        FailureOr<Value> expr = getTensorExpr(input);
        if (failed(expr))
          return failure();

        auto axesIt = axisInfo.operandAxes.find(&op->getOpOperand(index));
        if (axesIt == axisInfo.operandAxes.end())
          return op.emitOpError("missing use-site axis info for tensor input");

        auto exprType = cast<ExprType>((*expr).getType());
        SmallVector<std::string> currentAxes;
        for (Attribute attr : exprType.getAxes().getAxes())
          currentAxes.push_back(cast<AxisAttr>(attr).getName().getValue().str());
        SmallVector<std::string> useAxes = flattenAxes(axesIt->second);
        bool sameAxes = currentAxes.size() == useAxes.size();
        if (sameAxes) {
          for (auto [currentAxis, useAxis] : llvm::zip_equal(currentAxes, useAxes)) {
            if (currentAxis != useAxis) {
              sameAxes = false;
              break;
            }
          }
        }
        if (!sameAxes && currentAxes.size() == 1 && useAxes.size() == 1)
          expr = ta.subst(*expr, currentAxes, useAxes, exprType.getElementType());

        inputExprs.push_back(*expr);
      } else {
        DenseMap<Value, Value> emptyEnv;
        FailureOr<Value> expr = translateScalar(input, emptyEnv, loopAxes);
        if (failed(expr))
          return failure();
        inputExprs.push_back(*expr);
      }
    }

    Block &block = op.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getNumOperands() <= resultNumber)
      return op.emitOpError("expected linalg.yield for result");

    DenseMap<Value, Value> env;
    for (auto [arg, expr] : zip_equal(block.getArguments().take_front(numInputs), inputExprs)) {
      env[arg] = expr;
    }

    AxisPack reductionAxes;
    for (auto [index, iterator] : enumerate(iterators)) {
      if (iterator == utils::IteratorType::reduction)
        reductionAxes.append(loopAxes[index].begin(), loopAxes[index].end());
    }
    auto redAxisNames =
        llvm::map_to_vector(reductionAxes, [&](const Axis &axis) { return axis.name; });

    Value yielded = yield.getOperand(resultNumber);
    if (reductionAxes.empty())
      return translateScalar(yielded, env, loopAxes);

    FailureOr<std::pair<ReduceKind, Value>> combiner =
        peelReductionCombiner(op, yielded, block.getArguments().drop_front(numInputs));
    if (failed(combiner))
      return failure();

    FailureOr<Value> payload = translateScalar(combiner->second, env, loopAxes);
    if (failed(payload))
      return failure();
    Type elementType =
        dyn_cast<RankedTensorType>(op->getResult(resultNumber).getType()).getElementType();
    return ta.reduce(combiner->first, *payload, redAxisNames, elementType, flatResultAxes);
  }

  LogicalResult emitForward() {
    for (Operation &op : func.front().without_terminator()) {
      if (isa<arith::ConstantOp, tensor::EmptyOp, ScopeOp>(&op))
        continue;

      if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(&op)) {
        FailureOr<Value> expr = getTensorExpr(collapse.getSrc());
        if (failed(expr))
          return failure();
        valueMap.map(collapse.getResult(), *expr);
        continue;
      }

      if (auto expand = dyn_cast<tensor::ExpandShapeOp>(&op)) {
        FailureOr<Value> expr = getTensorExpr(expand.getSrc());
        if (failed(expr))
          return failure();
        valueMap.map(expand.getResult(), *expr);
        continue;
      }

      if (auto generic = dyn_cast<linalg::GenericOp>(&op)) {
        for (auto [resultNumber, result] : llvm::enumerate(generic->getResults())) {
          if (!dyn_cast<RankedTensorType>(result.getType()))
            continue;
          if (!axisInfo.valueAxes.contains(result))
            continue;
          FailureOr<Value> expr = emitGeneric(generic, resultNumber);
          if (failed(expr))
            return failure();
          valueMap.map(result, *expr);
        }
        continue;
      }

      if (llvm::any_of(op.getResults(),
                       [](Value value) { return dyn_cast<RankedTensorType>(value.getType()); }))
        return op.emitOpError("unsupported tensor producer for ta import");
    }

    return success();
  }

  FailureOr<Value> translateLinalgIndex(linalg::IndexOp index, ArrayRef<AxisPack> loopAxes,
                                        Type elementType = Type()) {
    if (!elementType)
      elementType = index.getResult().getType();

    unsigned dim = index.getDim();
    if (dim >= loopAxes.size())
      return index.emitOpError("linalg.index dimension is outside the loop rank");

    const AxisPack &packedAxes = loopAxes[dim];
    if (packedAxes.empty())
      return index.emitOpError("cannot import linalg.index for a constant-indexed dimension");

    if (packedAxes.size() == 1)
      return ta.index(packedAxes.front().name, elementType);

    SmallVector<StringRef> nonUnitAxes;
    for (auto &axis : packedAxes)
      if (axis.extent != 1)
        nonUnitAxes.push_back(axis.name);
    if (nonUnitAxes.empty()) {
      return ta.constant(IntegerAttr::get(elementType, 0));
    }
    if (nonUnitAxes.size() == 1)
      return ta.index(nonUnitAxes.front().str(), elementType);

    return index.emitOpError("cannot import linalg.index for a packed tensor dimension with "
                             "multiple non-unit axes");
  }

  FailureOr<Value> getTensorExpr(Value value) {
    auto existing = valueMap.lookupOrNull(value);
    if (existing)
      return existing;

    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return emitError(value.getLoc()) << "expected ranked tensor value";

    if (auto constant = dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
      if (TypedAttr scalar = splatScalarConstant(constant)) {
        Value expr = ta.constant(scalar);
        valueMap.map(value, expr);
        return expr;
      }
      return constant.emitOpError("only splat tensor constants are supported by ta import");
    }

    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner()->getParentOp() != func)
        return emitError(value.getLoc()) << "unsupported tensor block argument";
      auto it = axisInfo.valueAxes.find(value);
      if (it == axisInfo.valueAxes.end())
        return emitError(value.getLoc()) << "missing axis info for this tensor value";
      FailureOr<Value> expr = ta.at(value, it->second, type.getElementType());
      if (failed(expr))
        return failure();
      valueMap.map(value, *expr);
      return *expr;
    }

    Operation *def = value.getDefiningOp();
    if (auto collapse = dyn_cast_or_null<tensor::CollapseShapeOp>(def))
      return getTensorExpr(collapse.getSrc());
    if (auto expand = dyn_cast_or_null<tensor::ExpandShapeOp>(def))
      return getTensorExpr(expand.getSrc());

    return emitError(value.getLoc()) << "tensor value was not translated: " << value;
  }

  FailureOr<std::pair<ReduceKind, Value>>
  peelReductionCombiner(linalg::GenericOp op, Value yielded, Block::BlockArgListType outputArgs) {
    Operation *def = yielded.getDefiningOp();
    if (!def || def->getNumResults() != 1 || def->getNumOperands() != 2)
      return op.emitOpError("unsupported reduction combiner");

    std::optional<ReduceKind> kind;
    if (isa<arith::AddFOp>(def))
      kind = ReduceKind::Add;
    else if (isa<arith::MulFOp>(def))
      kind = ReduceKind::Mul;
    else if (isa<arith::MaximumFOp>(def))
      kind = ReduceKind::Max;
    else if (isa<arith::MinimumFOp>(def))
      kind = ReduceKind::Min;
    if (!kind)
      return op.emitOpError("unsupported reduction combiner op: ") << def->getName();

    auto isOutputArg = [&](Value value) { return llvm::is_contained(outputArgs, value); };

    Value lhs = def->getOperand(0);
    Value rhs = def->getOperand(1);
    if (isOutputArg(lhs) && !isOutputArg(rhs))
      return std::make_pair(*kind, rhs);
    if (isOutputArg(rhs) && !isOutputArg(lhs))
      return std::make_pair(*kind, lhs);

    return op.emitOpError("reduction combiner must combine one output argument with one payload");
  }

  template <typename OpTy>
  FailureOr<Value> translateUnaryScalarOp(Operation *def, const DenseMap<Value, Value> &env,
                                          ArrayRef<AxisPack> loopAxes) {
    FailureOr<Value> input = translateScalar(def->getOperand(0), env, loopAxes);
    if (failed(input))
      return failure();
    return ta.unary<OpTy>(def->getResult(0).getType(), *input);
  }

  template <typename OpTy>
  FailureOr<Value> translateBinaryScalarOp(Operation *def, const DenseMap<Value, Value> &env,
                                           ArrayRef<AxisPack> loopAxes) {
    FailureOr<Value> lhs = translateScalar(def->getOperand(0), env, loopAxes);
    FailureOr<Value> rhs = translateScalar(def->getOperand(1), env, loopAxes);
    if (failed(lhs) || failed(rhs))
      return failure();
    return ta.binary<OpTy>(def->getResult(0).getType(), *lhs, *rhs);
  }

  FailureOr<Value> translateScalar(Value value, const DenseMap<Value, Value> &env,
                                   ArrayRef<AxisPack> loopAxes) {
    auto it = env.find(value);
    if (it != env.end())
      return it->second;

    Operation *def = value.getDefiningOp();
    if (!def)
      return emitError(value.getLoc()) << "unsupported scalar block argument";

    if (auto constant = dyn_cast<arith::ConstantOp>(def)) {
      auto typed = dyn_cast<TypedAttr>(constant.getValue());
      if (!typed)
        return def->emitOpError("expected typed constant attribute");
      return ta.constant(typed);
    }

    if (auto index = dyn_cast<linalg::IndexOp>(def)) {
      return translateLinalgIndex(index, loopAxes);
    }

    if (def->getNumResults() != 1)
      return def->emitOpError("unsupported scalar op with multiple results");

    if (isa<arith::ExtFOp, arith::TruncFOp, arith::SIToFPOp>(def))
      return translateUnaryScalarOp<CastOp>(def, env, loopAxes);
    if (auto indexCast = dyn_cast<arith::IndexCastOp>(def)) {
      if (auto index = indexCast.getIn().getDefiningOp<linalg::IndexOp>())
        return translateLinalgIndex(index, loopAxes, indexCast.getResult().getType());
      return def->emitOpError("only index_cast of linalg.index is supported by ta import");
    }
    if (isa<math::ExpOp>(def))
      return translateUnaryScalarOp<ExpOp>(def, env, loopAxes);
    if (isa<arith::AddFOp>(def))
      return translateBinaryScalarOp<AddFOp>(def, env, loopAxes);
    if (isa<arith::SubFOp>(def))
      return translateBinaryScalarOp<SubFOp>(def, env, loopAxes);
    if (isa<arith::SubIOp>(def))
      return translateBinaryScalarOp<SubIOp>(def, env, loopAxes);
    if (isa<arith::AndIOp>(def))
      return translateBinaryScalarOp<AndIOp>(def, env, loopAxes);
    if (isa<arith::MulFOp>(def))
      return translateBinaryScalarOp<MulFOp>(def, env, loopAxes);
    if (isa<arith::DivFOp>(def))
      return translateBinaryScalarOp<DivFOp>(def, env, loopAxes);
    if (isa<arith::MaximumFOp>(def))
      return translateBinaryScalarOp<MaximumFOp>(def, env, loopAxes);
    if (isa<arith::MinimumFOp>(def))
      return translateBinaryScalarOp<MinimumFOp>(def, env, loopAxes);
    if (auto cmpi = dyn_cast<arith::CmpIOp>(def)) {
      FailureOr<Value> lhs = translateScalar(def->getOperand(0), env, loopAxes);
      FailureOr<Value> rhs = translateScalar(def->getOperand(1), env, loopAxes);
      if (failed(lhs) || failed(rhs))
        return failure();
      return ta.cmpi(cmpi.getPredicate(), *lhs, *rhs);
    }
    if (isa<arith::SelectOp>(def)) {
      FailureOr<Value> condition = translateScalar(def->getOperand(0), env, loopAxes);
      FailureOr<Value> trueValue = translateScalar(def->getOperand(1), env, loopAxes);
      FailureOr<Value> falseValue = translateScalar(def->getOperand(2), env, loopAxes);
      if (failed(condition) || failed(trueValue) || failed(falseValue))
        return failure();
      return ta.select(*condition, *trueValue, *falseValue);
    }

    return def->emitOpError("unsupported scalar op for ta import: ") << def->getName();
  }

  void eraseUnusedScopeOps() {
    Block &body = ta.getScope().getBody().front();
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(body.without_terminator()))) {
      if (!op.use_empty())
        continue;
      if (isa<AtOp, ConstantOp, IndexOp, CmpIOp, SelectOp, CastOp, ExpOp, Exp2Op, AddFOp, SubFOp,
              SubIOp, AndIOp, MulFOp, DivFOp, MaximumFOp, MinimumFOp, ReduceOp>(&op))
        op.erase();
    }
  }

  func::FuncOp func;
  func::ReturnOp returnOp;
  FunctionAxisInfo axisInfo;
  ScopedTABuilder ta;
  IRMapping valueMap;
  int64_t nextImportGroup = 0;
};

static SmallVector<Operation *> collectOldBodyOps(func::FuncOp func) {
  Block &entry = func.front();
  SmallVector<Operation *> toErase;
  for (Operation &op : entry.without_terminator())
    toErase.push_back(&op);
  return toErase;
}

struct ImportLinalgToTAPass
    : public PassWrapper<ImportLinalgToTAPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportLinalgToTAPass)

  StringRef getArgument() const final { return "linalg-to-ta"; }
  StringRef getDescription() const final {
    return "Import supported linalg.generic tensor dataflow into the ta dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TADialect, affine::AffineDialect, arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (func.empty())
      return;

    bool hasGeneric = false;
    func.walk([&](linalg::GenericOp) { hasGeneric = true; });
    if (!hasGeneric)
      return;

    auto returnOp = dyn_cast<func::ReturnOp>(func.front().getTerminator());
    if (!returnOp || returnOp.getNumOperands() == 0)
      return;
    if (func.getNumResults() != 1)
      return;
    if (returnOp.getNumOperands() != 1) {
      func.emitOpError("ta importer currently expects one function result");
      signalPassFailure();
      return;
    }

    auto resultType = dyn_cast<RankedTensorType>(func.getResultTypes().front());
    if (!resultType)
      return;

    SmallVector<Operation *> oldOps = collectOldBodyOps(func);
    FunctionAxisDiscovery discovery(func, returnOp);
    FailureOr<FunctionAxisInfo> axisInfo = discovery.run(resultType);
    if (failed(axisInfo)) {
      signalPassFailure();
      return;
    }

    FunctionEmitter emitter(func, returnOp, std::move(*axisInfo));
    if (failed(emitter.run())) {
      signalPassFailure();
      return;
    }
    for (Operation *op : llvm::reverse(oldOps))
      op->erase();
  }
};

} // namespace

void registerLinalgToTAPass() { PassRegistration<ImportLinalgToTAPass>(); }

} // namespace ta
