#include "TA/TAAttrs.h"
#include "TA/TAOps.h"
#include "TA/TAPasses.h"
#include "TA/TATypes.h"

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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <memory>
#include <optional>
#include <string>

namespace ta {

using namespace mlir;

namespace {

using AxisPack = SmallVector<std::string, 2>;
using TensorAxes = SmallVector<AxisPack, 4>;
using AxisId = unsigned;
using AxisIdPack = SmallVector<AxisId, 2>;
using TensorAxisIds = SmallVector<AxisIdPack, 4>;

static RankedTensorType rankedTensor(Type type) { return dyn_cast<RankedTensorType>(type); }

static SmallVector<std::string> flattenAxes(const TensorAxes &axes) {
  SmallVector<std::string> flat;
  for (const AxisPack &pack : axes)
    flat.append(pack.begin(), pack.end());
  return flat;
}

class ScopedTABuilder {
public:
  ScopedTABuilder(OpBuilder &builder, Location loc, RankedTensorType resultType)
      : context(builder.getContext()), loc(loc), indexType(builder.getIndexType()) {
    scope = ScopeOp::create(builder, loc, resultType, getAxes({}));
    Block *body = new Block();
    scope.getBody().push_back(body);
    bodyBuilder = std::make_unique<OpBuilder>(context);
    bodyBuilder->setInsertionPointToStart(body);
  }

  ScopeOp getScope() const { return scope; }

  OpBuilder &builder() { return *bodyBuilder; }

  ArrayRef<std::string> getScopeAxisNames() const { return axisNames; }

  void setImportGroup(std::optional<int64_t> group) { importGroup = group; }

  std::optional<int64_t> getImportGroup() const { return importGroup; }

  std::string createAxis(StringRef prefix) {
    std::string name;
    do {
      name = (prefix + std::to_string(nextAxis++)).str();
    } while (axisValues.contains(name));
    addAxis(name);
    return name;
  }

  Value axis(StringRef name) {
    auto it = axisValues.find(name);
    if (it != axisValues.end())
      return it->second;
    addAxis(name.str());
    return axisValues.lookup(name);
  }

  AxesAttr getAxes(ArrayRef<std::string> names) const {
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
    return ExprType::get(context, elementType, getAxes(axes));
  }

  SmallVector<std::string> unionAxes(ValueRange operands) const {
    DenseSet<StringRef> used;
    for (Value operand : operands) {
      auto exprType = cast<ExprType>(operand.getType());
      for (Attribute attr : exprType.getAxes().getAxes()) {
        auto axis = cast<AxisAttr>(attr);
        used.insert(axis.getName().getValue());
      }
    }

    SmallVector<std::string> result;
    for (StringRef name : axisNames) {
      if (used.contains(name))
        result.push_back(name.str());
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
      if (pack.size() != 1)
        return emitError(loc) << "cannot index one tensor dimension with multiple "
                              << "logical axes";
      indices.push_back(axis(pack.front()));
    }

    SmallVector<std::string> resultAxes = flattenAxes(dimAxes);
    auto op =
        AtOp::create(builder(), loc, expr(elementType, resultAxes), source, indices,
                     getAxes(resultAxes));
    annotate(op);
    return op.getResult();
  }

  Value constant(TypedAttr value) {
    auto op = ConstantOp::create(builder(), loc, expr(value.getType(), {}), value);
    annotate(op);
    return op.getResult();
  }

  template <typename OpTy> Value unary(Type elementType, Value input) {
    SmallVector<std::string> axes = unionAxes({input});
    auto op = OpTy::create(builder(), loc, expr(elementType, axes), input);
    annotate(op);
    return op.getResult();
  }

  template <typename OpTy> Value binary(Type elementType, Value lhs, Value rhs) {
    SmallVector<std::string> axes = unionAxes({lhs, rhs});
    auto op = OpTy::create(builder(), loc, expr(elementType, axes), lhs, rhs);
    annotate(op);
    return op.getResult();
  }

  Value reduce(ReduceKind kind, Value input, ArrayRef<std::string> reductionAxes,
               Type elementType, ArrayRef<std::string> resultAxes) {
    auto op = ReduceOp::create(builder(), loc, expr(elementType, resultAxes), kind, input, Value(),
                               getAxes(reductionAxes));
    annotate(op);
    return op.getResult();
  }

  void yield(Value value) { YieldOp::create(builder(), loc, value); }

private:
  void annotate(Operation *op) const {
    if (!importGroup)
      return;
    op->setAttr("ta.import_group", IntegerAttr::get(IntegerType::get(context, 64),
                                                    *importGroup));
  }

  void addAxis(const std::string &name) {
    if (axisValues.contains(name))
      return;

    Block &body = scope.getBody().front();
    BlockArgument arg = body.addArgument(indexType, loc);
    axisNames.push_back(name);
    axisValues.try_emplace(axisNames.back(), arg);
    scope.setAxesAttr(getAxes(axisNames));
  }

  Value indexZero() {
    if (zero)
      return zero;
    OpBuilder::InsertionGuard guard(builder());
    builder().setInsertionPointToStart(&scope.getBody().front());
    zero = arith::ConstantIndexOp::create(builder(), loc, 0);
    return zero;
  }

  MLIRContext *context;
  Location loc;
  Type indexType;
  ScopeOp scope;
  SmallVector<std::string> axisNames;
  llvm::StringMap<Value> axisValues;
  std::unique_ptr<OpBuilder> bodyBuilder;
  Value zero;
  std::optional<int64_t> importGroup;
  unsigned nextAxis = 0;
};

class FunctionImporter {
public:
  FunctionImporter(func::FuncOp func, func::ReturnOp returnOp)
      : func(func), returnOp(returnOp), insertionBuilder(returnOp) {
    ta = std::make_unique<ScopedTABuilder>(insertionBuilder, func.getLoc(),
                                           rankedTensor(func.getResultTypes().front()));
  }

  LogicalResult run() {
    if (func.getNumResults() != 1 || returnOp.getNumOperands() != 1)
      return func.emitOpError("ta importer currently expects one function result");

    auto resultType = rankedTensor(func.getResultTypes().front());
    if (!resultType)
      return func.emitOpError("ta importer expects a ranked tensor result");

    if (failed(discoverAxes(resultType)))
      return failure();

    materializeScopeAxes();

    if (failed(emitForward()))
      return failure();

    Value expr = lookupTensorExpr(returnOp.getOperand(0));
    if (!expr)
      return emitError(returnOp.getLoc()) << "failed to translate returned tensor";

    ta->yield(expr);
    returnOp.getOperation()->setOperands(ta->getScope().getResult());
    eraseUnusedScopeOps();
    return success();
  }

  ScopeOp getScope() const { return ta->getScope(); }

private:
  class ImportGroupGuard {
  public:
    ImportGroupGuard(ScopedTABuilder &ta, std::optional<int64_t> group)
        : ta(ta), oldGroup(ta.getImportGroup()) {
      ta.setImportGroup(group);
    }
    ~ImportGroupGuard() { ta.setImportGroup(oldGroup); }

  private:
    ScopedTABuilder &ta;
    std::optional<int64_t> oldGroup;
  };

  AxisId newAxis(StringRef prefix) {
    AxisId id = parents.size();
    parents.push_back(id);
    axisNames.push_back((prefix + std::to_string(id)).str());
    return id;
  }

  AxisId find(AxisId id) {
    AxisId parent = parents[id];
    if (parent == id)
      return id;
    parents[id] = find(parent);
    return parents[id];
  }

  void unite(AxisId lhs, AxisId rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (lhs > rhs)
      std::swap(lhs, rhs);
    parents[rhs] = lhs;
  }

  TensorAxisIds makeResultAxes(RankedTensorType type) {
    TensorAxisIds axes;
    for (int64_t i = 0; i < type.getRank(); ++i)
      axes.push_back(AxisIdPack{newAxis("a")});
    return axes;
  }

  bool mergeAxisPack(AxisIdPack &existing, const AxisIdPack &desired) {
    if (desired.empty())
      return false;
    if (existing.empty()) {
      existing = desired;
      return true;
    }
    if (existing.size() != desired.size())
      return false;
    for (auto [lhs, rhs] : zip_equal(existing, desired))
      unite(lhs, rhs);
    return false;
  }

  FailureOr<bool> mergeValueAxes(Value value, const TensorAxisIds &desired) {
    auto type = rankedTensor(value.getType());
    if (!type)
      return emitError(value.getLoc()) << "expected ranked tensor value";
    if (static_cast<int64_t>(desired.size()) != type.getRank())
      return emitError(value.getLoc()) << "axis rank does not match tensor rank";

    auto [it, inserted] = valueAxes.try_emplace(value, TensorAxisIds(type.getRank()));
    TensorAxisIds &existing = it->second;
    bool changed = inserted;
    for (auto [axis, desiredAxis] : zip_equal(existing, desired))
      changed |= mergeAxisPack(axis, desiredAxis);
    return changed;
  }

  FailureOr<TensorAxisIds> getKnownAxes(Value value) {
    auto it = valueAxes.find(value);
    if (it == valueAxes.end())
      return emitError(value.getLoc()) << "missing discovered axes for tensor value";
    return it->second;
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
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
    SmallVector<utils::IteratorType> iterators = linalgOp.getIteratorTypesArray();
    unsigned numInputs = op.getInputs().size();

    auto &loopAxes = loopAxisMap[op.getOperation()];
    if (loopAxes.empty())
      loopAxes.resize(iterators.size());

    bool changed = false;
    for (auto [resultNumber, result] : llvm::enumerate(op->getResults())) {
      auto resultType = rankedTensor(result.getType());
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
        loopAxes[index] = AxisIdPack{newAxis("r")};
        changed = true;
      }
    }

    for (auto [index, input] : llvm::enumerate(op.getInputs())) {
      if (rankedTensor(input.getType())) {
        FailureOr<TensorAxisIds> axes = projectMap(op, maps[index], loopAxes);
        if (failed(axes))
          return failure();
        FailureOr<bool> inputChanged = mergeValueAxes(input, *axes);
        if (failed(inputChanged))
          return failure();
        changed |= *inputChanged;
      }
    }

    return changed;
  }

  LogicalResult discoverAxes(RankedTensorType resultType) {
    TensorAxisIds resultAxes = makeResultAxes(resultType);
    FailureOr<bool> changed = mergeValueAxes(returnOp.getOperand(0), resultAxes);
    if (failed(changed))
      return failure();

    bool keepGoing = true;
    while (keepGoing) {
      keepGoing = false;
      for (Operation &op : llvm::reverse(func.front().without_terminator())) {
        if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(&op)) {
          auto it = valueAxes.find(collapse.getResult());
          if (it == valueAxes.end())
            continue;
          FailureOr<TensorAxisIds> sourceAxes = sourceAxesForCollapse(collapse, it->second);
          if (failed(sourceAxes))
            return failure();
          FailureOr<bool> sourceChanged = mergeValueAxes(collapse.getSrc(), *sourceAxes);
          if (failed(sourceChanged))
            return failure();
          keepGoing |= *sourceChanged;
          continue;
        }

        if (auto expand = dyn_cast<tensor::ExpandShapeOp>(&op)) {
          auto it = valueAxes.find(expand.getResult());
          if (it == valueAxes.end())
            continue;
          FailureOr<TensorAxisIds> sourceAxes = sourceAxesForExpand(expand, it->second);
          if (failed(sourceAxes))
            return failure();
          FailureOr<bool> sourceChanged = mergeValueAxes(expand.getSrc(), *sourceAxes);
          if (failed(sourceChanged))
            return failure();
          keepGoing |= *sourceChanged;
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

  FailureOr<Value> emitGeneric(linalg::GenericOp op, unsigned resultNumber) {
    ImportGroupGuard guard(*ta, getImportGroup(op));

    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    SmallVector<utils::IteratorType> iterators = linalgOp.getIteratorTypesArray();
    unsigned numInputs = op.getInputs().size();
    TensorAxes resultAxes = canonicalize(valueAxes.lookup(op->getResult(resultNumber)));
    TensorAxes loopAxes = canonicalize(loopAxisMap.lookup(op.getOperation()));

    SmallVector<Value> inputExprs;
    inputExprs.reserve(numInputs);
    for (Value input : op.getInputs()) {
      if (rankedTensor(input.getType())) {
        FailureOr<Value> expr = getTensorExpr(input);
        if (failed(expr))
          return failure();
        inputExprs.push_back(*expr);
      } else {
        DenseMap<Value, Value> emptyEnv;
        FailureOr<Value> expr = translateScalar(input, emptyEnv);
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

    SmallVector<std::string> reductionAxes;
    for (auto [index, iterator] : enumerate(iterators)) {
      if (iterator == utils::IteratorType::reduction)
        reductionAxes.append(loopAxes[index].begin(), loopAxes[index].end());
    }

    SmallVector<std::string> flatResultAxes = flattenAxes(resultAxes);
    Value yielded = yield.getOperand(resultNumber);
    if (reductionAxes.empty())
      return translateScalar(yielded, env);

    FailureOr<std::pair<ReduceKind, Value>> combiner =
        peelReductionCombiner(op, yielded, block.getArguments().drop_front(numInputs));
    if (failed(combiner))
      return failure();

    FailureOr<Value> payload = translateScalar(combiner->second, env);
    if (failed(payload))
      return failure();

    Type elementType = rankedTensor(op->getResult(resultNumber).getType()).getElementType();
    return ta->reduce(combiner->first, *payload, reductionAxes, elementType, flatResultAxes);
  }

  LogicalResult emitForward() {
    for (Operation &op : func.front().without_terminator()) {
      if (isa<arith::ConstantOp, tensor::EmptyOp, ScopeOp>(&op))
        continue;

      if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(&op)) {
        FailureOr<Value> expr = getTensorExpr(collapse.getSrc());
        if (failed(expr))
          return failure();
        valueMap[collapse.getResult()] = *expr;
        continue;
      }

      if (auto expand = dyn_cast<tensor::ExpandShapeOp>(&op)) {
        FailureOr<Value> expr = getTensorExpr(expand.getSrc());
        if (failed(expr))
          return failure();
        valueMap[expand.getResult()] = *expr;
        continue;
      }

      if (auto generic = dyn_cast<linalg::GenericOp>(&op)) {
        for (auto [resultNumber, result] : llvm::enumerate(generic->getResults())) {
          if (!rankedTensor(result.getType()))
            continue;
          if (!valueAxes.contains(result))
            continue;
          FailureOr<Value> expr = emitGeneric(generic, resultNumber);
          if (failed(expr))
            return failure();
          valueMap[result] = *expr;
        }
        continue;
      }

      if (llvm::any_of(op.getResults(), [](Value value) { return rankedTensor(value.getType()); }))
        return op.emitOpError("unsupported tensor producer for ta import");
    }

    return success();
  }

  int64_t getImportGroup(linalg::GenericOp op) {
    auto [it, inserted] = importGroups.try_emplace(op.getOperation(), nextImportGroup);
    if (inserted)
      ++nextImportGroup;
    return it->second;
  }

  LogicalResult assignLoopAxesFromOutputMap(Operation *op, AffineMap map,
                                            const TensorAxisIds &resultAxes,
                                            MutableArrayRef<AxisIdPack> loopAxes) {
    if (map.getNumResults() != resultAxes.size())
      return op->emitOpError("output indexing map rank does not match result axis rank");

    for (auto [resultIndex, expr] : enumerate(map.getResults())) {
      if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
        AxisIdPack &assigned = loopAxes[dim.getPosition()];
        mergeAxisPack(assigned, resultAxes[resultIndex]);
        continue;
      }
      if (isa<AffineConstantExpr>(expr))
        continue;
      return op->emitOpError("non-projected output indexing maps are not supported");
    }
    return success();
  }

  FailureOr<TensorAxisIds> projectMap(Operation *op, AffineMap map,
                                      ArrayRef<AxisIdPack> loopAxes) {
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

  TensorAxes canonicalize(const TensorAxisIds &ids) {
    TensorAxes axes;
    for (const AxisIdPack &pack : ids) {
      AxisPack canonicalPack;
      for (AxisId id : pack)
        canonicalPack.push_back(axisNames[find(id)]);
      axes.push_back(canonicalPack);
    }
    return axes;
  }

  void materializeScopeAxes() {
    DenseSet<AxisId> seen;
    for (AxisId id = 0, e = parents.size(); id < e; ++id) {
      AxisId root = find(id);
      if (!seen.insert(root).second)
        continue;
      ta->axis(axisNames[root]);
    }
  }

  Value lookupTensorExpr(Value value) const {
    auto it = valueMap.find(value);
    if (it == valueMap.end())
      return Value();
    return it->second;
  }

  FailureOr<Value> getTensorExpr(Value value) {
    if (Value expr = lookupTensorExpr(value))
      return expr;

    auto type = rankedTensor(value.getType());
    if (!type)
      return emitError(value.getLoc()) << "expected ranked tensor value";

    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner()->getParentOp() != func)
        return emitError(value.getLoc()) << "unsupported tensor block argument";
      FailureOr<TensorAxisIds> axes = getKnownAxes(value);
      if (failed(axes))
        return failure();
      FailureOr<Value> expr = ta->at(value, canonicalize(*axes), type.getElementType());
      if (failed(expr))
        return failure();
      valueMap[value] = *expr;
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
  FailureOr<Value> translateUnaryScalarOp(Operation *def, const DenseMap<Value, Value> &env) {
    FailureOr<Value> input = translateScalar(def->getOperand(0), env);
    if (failed(input))
      return failure();
    return ta->unary<OpTy>(def->getResult(0).getType(), *input);
  }

  template <typename OpTy>
  FailureOr<Value> translateBinaryScalarOp(Operation *def, const DenseMap<Value, Value> &env) {
    FailureOr<Value> lhs = translateScalar(def->getOperand(0), env);
    FailureOr<Value> rhs = translateScalar(def->getOperand(1), env);
    if (failed(lhs) || failed(rhs))
      return failure();
    return ta->binary<OpTy>(def->getResult(0).getType(), *lhs, *rhs);
  }

  FailureOr<Value> translateScalar(Value value, const DenseMap<Value, Value> &env) {
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
      return ta->constant(typed);
    }

    if (def->getNumResults() != 1)
      return def->emitOpError("unsupported scalar op with multiple results");

    if (isa<arith::ExtFOp>(def))
      return translateUnaryScalarOp<ExtFOp>(def, env);
    if (isa<arith::TruncFOp>(def))
      return translateUnaryScalarOp<TruncFOp>(def, env);
    if (isa<math::ExpOp>(def))
      return translateUnaryScalarOp<ExpOp>(def, env);
    if (isa<arith::AddFOp>(def))
      return translateBinaryScalarOp<AddFOp>(def, env);
    if (isa<arith::SubFOp>(def))
      return translateBinaryScalarOp<SubFOp>(def, env);
    if (isa<arith::MulFOp>(def))
      return translateBinaryScalarOp<MulFOp>(def, env);
    if (isa<arith::DivFOp>(def))
      return translateBinaryScalarOp<DivFOp>(def, env);
    if (isa<arith::MaximumFOp>(def))
      return translateBinaryScalarOp<MaximumFOp>(def, env);
    if (isa<arith::MinimumFOp>(def))
      return translateBinaryScalarOp<MinimumFOp>(def, env);

    return def->emitOpError("unsupported scalar op for ta import: ") << def->getName();
  }

  void eraseUnusedScopeOps() {
    Block &body = ta->getScope().getBody().front();
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(body.without_terminator()))) {
      if (!op.use_empty())
        continue;
      if (isa<AtOp, ConstantOp, ExtFOp, TruncFOp, ExpOp, Exp2Op, AddFOp, SubFOp, MulFOp, DivFOp,
              MaximumFOp, MinimumFOp, ReduceOp>(&op))
        op.erase();
    }
  }

  func::FuncOp func;
  func::ReturnOp returnOp;
  OpBuilder insertionBuilder;
  std::unique_ptr<ScopedTABuilder> ta;
  SmallVector<AxisId> parents;
  SmallVector<std::string> axisNames;
  DenseMap<Value, TensorAxisIds> valueAxes;
  DenseMap<Operation *, SmallVector<AxisIdPack>> loopAxisMap;
  DenseMap<Value, Value> valueMap;
  DenseMap<Operation *, int64_t> importGroups;
  int64_t nextImportGroup = 0;
};

static SmallVector<Operation *> collectOldBodyOps(func::FuncOp func) {
  Block &entry = func.front();
  SmallVector<Operation *> toErase;
  for (Operation &op : entry.without_terminator())
    toErase.push_back(&op);
  return toErase;
}

static void eraseOps(ArrayRef<Operation *> toErase) {
  for (Operation *op : reverse(toErase))
    op->erase();
}

struct ImportLinalgToTAPass
    : public PassWrapper<ImportLinalgToTAPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportLinalgToTAPass)

  StringRef getArgument() const final { return "ta-import-linalg"; }
  StringRef getDescription() const final {
    return "Import supported linalg.generic tensor dataflow into the ta dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TADialect, arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
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
    if (func.getNumResults() != 1 || !rankedTensor(func.getResultTypes().front()))
      return;

    SmallVector<Operation *> oldOps = collectOldBodyOps(func);
    FunctionImporter importer(func, returnOp);
    if (failed(importer.run())) {
      signalPassFailure();
      return;
    }
    eraseOps(oldOps);
  }
};

} // namespace

void registerTAPasses() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;
  PassRegistration<ImportLinalgToTAPass>();
  registerTAToLinalgPass();
}

} // namespace ta
