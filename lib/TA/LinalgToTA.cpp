#include "LoopTr/Utils.h"
#include "TA/TAOps.h"
#include "TA/TAPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-to-ta"

namespace ta {

using namespace mlir;

namespace {

// Normalize linalg.generic ops of the form:
//   collapse_shape(inputs) -> linalg.generic -> expand_shape(results)
// into a higher-rank linalg.generic over the original input/result shapes.
// This keeps axis discovery forward-only by eliminating matched reshape
// sandwiches before the TA importer reasons about tensor axes.
static LogicalResult expandFullyWrappedGenericOps(func::FuncOp func);

static LogicalResult importFunctionAsTA(func::FuncOp func, func::ReturnOp returnOp,
                                        RankedTensorType resultType);

struct ImportLinalgToTAPass
    : public PassWrapper<ImportLinalgToTAPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportLinalgToTAPass)

  StringRef getArgument() const final { return "linalg-to-ta"; }
  StringRef getDescription() const final {
    return "Import supported linalg.generic tensor dataflow into the ta dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TADialect, affine::AffineDialect, func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (func.empty())
      return;
    auto returnOp = dyn_cast<func::ReturnOp>(func.front().getTerminator());
    if (!returnOp || returnOp.getNumOperands() == 0)
      return;
    if (func.getNumResults() != 1 || returnOp.getNumOperands() != 1) {
      func.emitOpError("ta importer currently expects one function result");
      return signalPassFailure();
    }
    auto resultType = dyn_cast<RankedTensorType>(func.getResultTypes().front());
    if (!resultType)
      return;
    if (failed(expandFullyWrappedGenericOps(func)))
      return signalPassFailure();
    if (failed(importFunctionAsTA(func, returnOp, resultType)))
      return signalPassFailure();
  }
};

struct Axis {
  std::string name;
  int64_t extent = ShapedType::kDynamic;
};

using TensorAxes = SmallVector<std::optional<Axis>>;

struct DiscoveredAxisInfo {
  DenseMap<Value, TensorAxes> valueAxes;
  DenseMap<Operation *, SmallVector<Axis>> genericLoopAxes;
};

class ForwardAxisDiscovery {
public:
  FailureOr<DiscoveredAxisInfo> run(func::FuncOp func) {
    unsigned genericIndex = 0;
    for (Operation &op : func.front().without_terminator()) {
      if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(&op)) {
        if (failed(discoverCollapseShape(collapse)))
          return failure();
        continue;
      }
      if (auto expand = dyn_cast<tensor::ExpandShapeOp>(&op)) {
        if (failed(discoverExpandShape(expand)))
          return failure();
        continue;
      }
      if (auto generic = dyn_cast<linalg::GenericOp>(&op);
          generic && failed(discoverGeneric(generic, genericIndex++)))
        return failure();
    }

    DiscoveredAxisInfo info;
    for (auto &[value, axisIds] : valueAxisIds)
      info.valueAxes[value] = llvm::map_to_vector(axisIds, [&](std::optional<AxisId> id) {
        return id ? axes[find(*id)] : std::optional<Axis>{};
      });
    for (auto &[generic, axisIds] : genericLoopAxisIds)
      info.genericLoopAxes[generic] =
          llvm::map_to_vector(axisIds, [&](AxisId id) { return axes[find(id)]; });
    return info;
  }

private:
  using AxisId = unsigned;
  using AxisIds = SmallVector<std::optional<AxisId>, 4>;

  FailureOr<AxisIds *> getOrCreateValueAxes(Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return failure();

    auto [it, inserted] = valueAxisIds.try_emplace(value);
    if (!inserted)
      return &it->second;

    it->second.reserve(type.getRank());
    for (int64_t dim = 0, rank = type.getRank(); dim < rank; ++dim) {
      int64_t extent = type.getDimSize(dim);
      if (extent == ShapedType::kDynamic)
        return failure();
      it->second.push_back(makeAxis(("v" + Twine(nextValueAxis++)).str(), extent));
    }
    return &it->second;
  }

  AxisId makeAxis(std::string name, int64_t extent) {
    AxisId id = parent.size();
    parent.push_back(id);
    axes.push_back(Axis{std::move(name), extent});
    return id;
  }

  AxisId find(AxisId id) {
    if (parent[id] == id)
      return id;
    parent[id] = find(parent[id]);
    return parent[id];
  }

  bool axesCompatible(AxisId lhs, AxisId rhs) const {
    int64_t lhsExtent = axes[lhs].extent;
    int64_t rhsExtent = axes[rhs].extent;
    return lhsExtent == ShapedType::kDynamic || rhsExtent == ShapedType::kDynamic ||
           lhsExtent == rhsExtent;
  }

  LogicalResult mergeProductFactors(Operation *op, AxisId product,
                                    ArrayRef<AxisId> incomingFactors_) {
    product = find(product);
    SmallVector<AxisId> incomingFactors =
        llvm::map_to_vector(incomingFactors_, [&](AxisId factor) { return find(factor); });

    auto it = productAxes.find(product);
    if (it == productAxes.end()) {
      productAxes[product] = std::move(incomingFactors);
      return success();
    }

    // If there is an existing list of factor axes, unify each with the incoming ones.
    // Quit if any incompatibility is seen (without returning an error).
    for (AxisId &axis : it->second)
      axis = find(axis);
    if (it->second.size() != incomingFactors.size())
      return success();
    for (auto [lhs, rhs] : llvm::zip_equal(it->second, incomingFactors)) {
      if (!axesCompatible(lhs, rhs))
        return success();
    }
    for (auto [lhs, rhs] : llvm::zip_equal(it->second, incomingFactors)) {
      if (failed(unionAxes(op, lhs, rhs)))
        return failure();
    }
    return success();
  }

  LogicalResult unionAxes(Operation *op, AxisId lhs, AxisId rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return success();
    if (!axesCompatible(lhs, rhs))
      return op->emitOpError("axis discovery found conflicting extents");
    parent[rhs] = lhs;

    auto result = success();
    if (auto it = productAxes.find(rhs); it != productAxes.end()) {
      result = mergeProductFactors(op, lhs, it->second);
      productAxes.erase(it);
    }
    return result;
  }

  LogicalResult unionMapWithLoops(Operation *op, AffineMap map, AxisIds &tensorAxes,
                                  ArrayRef<AxisId> loopAxes) {
    if (map.getNumResults() != tensorAxes.size())
      return failure();
    for (auto [tensorDim, expr] : llvm::enumerate(map.getResults())) {
      auto dim = dyn_cast<AffineDimExpr>(expr);
      if (!dim)
        continue;
      if (dim.getPosition() >= loopAxes.size())
        return failure();
      if (tensorAxes[tensorDim] &&
          failed(unionAxes(op, *tensorAxes[tensorDim], loopAxes[dim.getPosition()])))
        return failure();
    }
    return success();
  }

  LogicalResult processReassociation(Operation *op, ArrayRef<ReassociationIndices> reassociation,
                                     AxisIds &collapsedAxes, AxisIds &expandedAxes) {
    if (reassociation.size() != collapsedAxes.size())
      return failure();
    for (auto [collapsedDim, expandedDims] : llvm::enumerate(reassociation)) {
      if (!collapsedAxes[collapsedDim])
        continue;
      if (expandedDims.size() == 1) {
        int64_t expandedDim = expandedDims.front();
        if (expandedDim < 0 || expandedDim >= static_cast<int64_t>(expandedAxes.size()))
          return failure();
        if (expandedAxes[expandedDim] &&
            failed(unionAxes(op, *collapsedAxes[collapsedDim], *expandedAxes[expandedDim])))
          return failure();
        continue;
      }

      SmallVector<AxisId> factors;
      factors.reserve(expandedDims.size());
      for (int64_t expandedDim : expandedDims) {
        if (expandedDim < 0 || expandedDim >= static_cast<int64_t>(expandedAxes.size()))
          return failure();
        if (!expandedAxes[expandedDim])
          return failure();
        factors.push_back(find(*expandedAxes[expandedDim]));
      }

      if (failed(mergeProductFactors(op, *collapsedAxes[collapsedDim], factors)))
        return failure();
    }
    return success();
  }

  LogicalResult discoverCollapseShape(tensor::CollapseShapeOp collapse) {
    FailureOr<AxisIds *> expandedAxes = getOrCreateValueAxes(collapse.getSrc());
    FailureOr<AxisIds *> collapsedAxes = getOrCreateValueAxes(collapse.getResult());
    if (failed(expandedAxes) || failed(collapsedAxes))
      return collapse.emitOpError("axis discovery failed to create collapse_shape axes");
    if (failed(processReassociation(collapse, collapse.getReassociationIndices(), **collapsedAxes,
                                    **expandedAxes)))
      return collapse.emitOpError("axis discovery failed to process collapse_shape reassociation");
    return success();
  }

  LogicalResult discoverExpandShape(tensor::ExpandShapeOp expand) {
    FailureOr<AxisIds *> collapsedAxes = getOrCreateValueAxes(expand.getSrc());
    FailureOr<AxisIds *> expandedAxes = getOrCreateValueAxes(expand.getResult());
    if (failed(expandedAxes) || failed(collapsedAxes))
      return expand.emitOpError("axis discovery failed to create expand_shape axes");
    if (failed(processReassociation(expand, expand.getReassociationIndices(), **collapsedAxes,
                                    **expandedAxes)))
      return expand.emitOpError("axis discovery failed to process expand_shape reassociation");
    return success();
  }

  LogicalResult discoverGeneric(linalg::GenericOp generic, unsigned genericIndex) {
    // Create an axis for each loop dimension of linalg.generic. These axes are used as "glue"
    // to connect the tensor axes of the operands and results, so they normally won't show up in the
    // value-to-axis mapping.
    SmallVector<int64_t> loopRanges = generic.getStaticLoopRanges();
    auto iterTypes = generic.getIteratorTypesArray();
    SmallVector<AxisId> loopAxes;
    loopAxes.reserve(loopRanges.size());
    for (size_t loopDim = 0, rank = loopRanges.size(); loopDim < rank; ++loopDim) {
      int64_t extent = loopRanges[loopDim];
      if (extent == ShapedType::kDynamic)
        return generic.emitOpError("axis discovery does not support dynamic loop extents");
      StringRef prefix = iterTypes[loopDim] == utils::IteratorType::reduction ? "r" : "i";
      loopAxes.push_back(
          makeAxis(("g" + Twine(genericIndex) + "_" + prefix + Twine(loopDim)).str(), extent));
    }
    genericLoopAxisIds[generic] = loopAxes;

    auto processOperand = [&](OpOperand *operand) -> LogicalResult {
      if (!isa<RankedTensorType>(operand->get().getType()))
        return success();
      FailureOr<AxisIds *> axes = getOrCreateValueAxes(operand->get());
      if (failed(axes))
        return generic.emitOpError("axis discovery failed to create operand axes");
      if (failed(unionMapWithLoops(generic, generic.getMatchingIndexingMap(operand), **axes,
                                   loopAxes)))
        return generic.emitOpError("axis discovery failed to process operand indexing map");
      return success();
    };

    for (OpOperand *operand : generic.getDpsInputOperands()) {
      if (failed(processOperand(operand)))
        return failure();
    }

    for (OpResult result : generic->getResults()) {
      auto resultType = dyn_cast<RankedTensorType>(result.getType());
      if (!resultType)
        continue;
      AxisIds resultAxes(resultType.getRank(), std::nullopt);
      OpOperand *init = generic.getDpsInitOperand(result.getResultNumber());
      AffineMap map = generic.getMatchingIndexingMap(init);
      for (auto [tensorDim, expr] : llvm::enumerate(map.getResults())) {
        if (auto dim = dyn_cast<AffineDimExpr>(expr))
          resultAxes[tensorDim] = loopAxes[dim.getPosition()];
      }
      valueAxisIds[result] = std::move(resultAxes);
    }

    return success();
  }

  DenseMap<Value, AxisIds> valueAxisIds;
  DenseMap<Operation *, SmallVector<AxisId>> genericLoopAxisIds;
  DenseMap<AxisId, SmallVector<AxisId>> productAxes;
  SmallVector<AxisId> parent;
  SmallVector<Axis> axes;
  AxisId nextValueAxis = 0;
};

using AxisName = std::string;
using AxisNames = SmallVector<AxisName>;
using AxisNameMapVector = llvm::MapVector<AxisName, AxisName, llvm::StringMap<unsigned>>;

class ScopedTABuilder {
public:
  ScopedTABuilder(Operation *insertBefore, Location loc, RankedTensorType resultType)
      : context(insertBefore->getContext()), loc(loc), builder(insertBefore) {
    Block *body = new Block();
    scope = ScopeOp::create(builder, loc, resultType, ValueRange{}, getAxesAttr({}),
                            DenseI64ArrayAttr::get(context, {}));
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

  AxesAttr getAxesAttr(ArrayRef<AxisName> names) const {
    SmallVector<Attribute> axes;
    DenseSet<StringRef> seen;
    for (StringRef name : names) {
      if (!seen.insert(name).second)
        continue;
      axes.push_back(AxisAttr::get(context, name));
    }
    return AxesAttr::get(context, ArrayAttr::get(context, axes));
  }

  ExprType getExprType(Type elementType, ArrayRef<AxisName> axes) const {
    return ExprType::get(context, elementType, getAxesAttr(axes));
  }

  AxisNames collectOperandAxes(ValueRange operands) const {
    DenseSet<StringRef> seen;
    AxisNames result;
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
    SmallVector<AxisName> resultAxes;
    for (const std::optional<Axis> &axis : dimAxes) {
      if (axis) {
        indices.push_back(materializeAxis(*axis));
        resultAxes.push_back(axis->name);
      } else {
        indices.push_back(indexZero());
      }
    }

    return annotate(
        AtOp::create(builder, loc, getExprType(elementType, resultAxes), source, indices));
  }

  FailureOr<Value> atExpandedSource(Value source, ArrayRef<ReassociationIndices> reassociation,
                                    const TensorAxes &resultDimAxes, Type elementType) {
    if (static_cast<int64_t>(reassociation.size()) !=
        cast<RankedTensorType>(source.getType()).getRank())
      return emitError(source.getLoc(), "expand_shape reassociation does not match source rank");

    SmallVector<Value> indices;
    indices.reserve(reassociation.size());
    for (auto &group : reassociation) {
      SmallVector<Value> groupIndices;
      SmallVector<int64_t> basis;
      for (int64_t resultDim : group) {
        if (resultDim < 0 || resultDim >= static_cast<int64_t>(resultDimAxes.size()))
          return emitError(source.getLoc(), "expand_shape reassociation references invalid dim");
        const std::optional<Axis> &axis = resultDimAxes[resultDim];
        if (!axis)
          continue;
        if (axis->extent == ShapedType::kDynamic)
          return emitError(source.getLoc(), "cannot linearize dynamic expanded axis");
        groupIndices.push_back(materializeAxis(*axis));
        basis.push_back(axis->extent);
      }

      if (groupIndices.empty())
        indices.push_back(indexZero());
      else if (groupIndices.size() == 1)
        indices.push_back(groupIndices.front());
      else
        indices.push_back(affine::AffineLinearizeIndexOp::create(builder, loc, groupIndices, basis,
                                                                 /*disjoint=*/true));
    }

    AxisNames resultAxes;
    for (const std::optional<Axis> &axis : resultDimAxes)
      if (axis)
        resultAxes.push_back(axis->name);
    return annotate(
        AtOp::create(builder, loc, getExprType(elementType, resultAxes), source, indices));
  }

  Value constant(TypedAttr value) {
    return annotate(ConstantOp::create(builder, loc, getExprType(value.getType(), {}), value));
  }

  FailureOr<Value> index(const Axis &axis, Type elementType) {
    if (!elementType)
      elementType = builder.getIndexType();
    return annotate(IndexOp::create(builder, loc, getExprType(elementType, {axis.name}),
                                    materializeAxis(axis)));
  }

  template <typename CmpOp, typename PredicateTy>
  Value cmp(PredicateTy predicate, Value lhs, Value rhs) {
    auto axes = collectOperandAxes({lhs, rhs});
    return annotate(CmpOp::create(builder, loc, getExprType(IntegerType::get(context, 1), axes),
                                  predicate, lhs, rhs));
  }

  Value select(Value condition, Value trueValue, Value falseValue) {
    auto axes = collectOperandAxes({trueValue, falseValue, condition});
    auto trueType = cast<ExprType>(trueValue.getType());
    return annotate(SelectOp::create(builder, loc, getExprType(trueType.getElementType(), axes),
                                     condition, trueValue, falseValue));
  }

  template <typename OpTy> Value unary(Type elementType, Value input) {
    auto axes = collectOperandAxes({input});
    return annotate(OpTy::create(builder, loc, getExprType(elementType, axes), input));
  }

  template <typename OpTy> Value binary(Type elementType, Value lhs, Value rhs) {
    auto axes = collectOperandAxes({lhs, rhs});
    return annotate(OpTy::create(builder, loc, getExprType(elementType, axes), lhs, rhs));
  }

  Value reduce(ReduceKind kind, Value input, ArrayRef<AxisName> reductionAxes, Type elementType,
               ArrayRef<AxisName> resultAxes) {
    return annotate(ReduceOp::create(builder, loc, getExprType(elementType, resultAxes), kind,
                                     input, Value(), getAxesAttr(reductionAxes)));
  }

  Value subst(Value input, const AxisNameMapVector &replacements) {
    auto inputType = cast<ExprType>(input.getType());
    AxisNames resultAxes;
    AxisNames fromAxes, toAxes;
    for (Attribute attr : inputType.getAxes().getAxes()) {
      StringRef axis = cast<AxisAttr>(attr).getName().getValue();
      auto replacement = replacements.find(axis.str());
      if (replacement == replacements.end() || replacement->second == axis) {
        resultAxes.push_back(axis.str());
      } else {
        resultAxes.push_back(replacement->second);
        fromAxes.push_back(axis.str());
        toAxes.push_back(replacement->second);
      }
    }
    if (fromAxes.empty())
      return input;

    auto exprType = getExprType(inputType.getElementType(), resultAxes);
    return annotate(
        SubstOp::create(builder, loc, exprType, input, getAxesAttr(fromAxes), getAxesAttr(toAxes)));
  }

  void yield(Value value) { YieldOp::create(builder, loc, value); }

  // Normalize emitted axis names after the final expression is known: output axes become i*,
  // remaining internal axes become j*, and all TA axis attrs/types are rewritten consistently.
  void relabelAxesForOutput(Value output) {
    auto outputType = cast<ExprType>(output.getType());
    AxisNameMapVector axisRenames;
    unsigned nextOutputAxis = 0, nextInternalAxis = 0;
    for (Attribute attr : outputType.getAxes().getAxes()) {
      AxisName oldName = cast<AxisAttr>(attr).getName().getValue().str();
      if (!axes.contains(oldName) || axisRenames.contains(oldName))
        continue;
      axisRenames.insert({oldName, "i" + std::to_string(nextOutputAxis++)});
    }
    for (auto &entry : axes) {
      if (!axisRenames.contains(entry.first))
        axisRenames.insert({entry.first, "j" + std::to_string(nextInternalAxis++)});
    }

    AxisNames scopeAxisNames;
    SmallVector<int64_t> staticExtents;
    Block &body = scope.getBody().front();
    unsigned oldNumAxes = body.getNumArguments();
    for (auto &[oldName, newName] : axisRenames) {
      auto &axis = axes[oldName];
      BlockArgument arg = body.addArgument(builder.getIndexType(), loc);
      axis.value.replaceAllUsesWith(arg);
      axis.value = arg;
      scopeAxisNames.push_back(newName);
      staticExtents.push_back(axis.extent);
    }
    if (oldNumAxes)
      body.eraseArguments(0, oldNumAxes);
    scope.setAxesAttr(getAxesAttr(scopeAxisNames));
    scope.setStaticExtentsAttr(DenseI64ArrayAttr::get(context, staticExtents));

    auto renameAxes = [&axisRenames, this](AxesAttr axesAttr) {
      SmallVector<Attribute> renamed;
      renamed.reserve(axesAttr.getAxes().size());
      for (Attribute attr : axesAttr.getAxes()) {
        StringRef oldName = cast<AxisAttr>(attr).getName().getValue();
        auto it = axisRenames.find(oldName.str());
        renamed.push_back(AxisAttr::get(context, it == axisRenames.end() ? oldName : it->second));
      }
      return AxesAttr::get(context, ArrayAttr::get(context, renamed));
    };

    scope.getBody().walk([&](Operation *op) {
      for (NamedAttribute attr : llvm::to_vector(op->getAttrs())) {
        if (auto axesAttr = dyn_cast<AxesAttr>(attr.getValue()))
          op->setAttr(attr.getName(), renameAxes(axesAttr));
      }
      for (OpResult result : op->getResults()) {
        if (auto exprType = dyn_cast<ExprType>(result.getType()))
          result.setType(
              ExprType::get(context, exprType.getElementType(), renameAxes(exprType.getAxes())));
      }
    });
  }

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

  Value materializeAxis(const Axis &axis) {
    auto it = axes.find(axis.name);
    if (it != axes.end())
      return it->second.value;

    Value arg = scope.getBody().front().addArgument(builder.getIndexType(), loc);
    axes.try_emplace(axis.name, MaterializedAxis{arg, axis.extent});

    AxisNames axisNames;
    SmallVector<int64_t> staticExtents;
    for (auto &entry : axes) {
      axisNames.push_back(entry.first);
      staticExtents.push_back(entry.second.extent);
    }
    scope.setAxesAttr(getAxesAttr(axisNames));
    scope.setStaticExtentsAttr(DenseI64ArrayAttr::get(context, staticExtents));
    return arg;
  }

  struct MaterializedAxis {
    Value value;
    int64_t extent;
  };

  MLIRContext *context;
  const Location loc;
  OpBuilder builder;

  Value zero;
  ScopeOp scope;
  std::optional<int64_t> importGroup;
  llvm::MapVector<AxisName, MaterializedAxis, llvm::StringMap<unsigned>> axes;
};

struct TensorValueInfo {
  Value expr;
  TensorAxes axes;
};

class FunctionEmitter {
public:
  FunctionEmitter(ScopedTABuilder &ta, const DiscoveredAxisInfo &axisInfo)
      : ta(ta), axisInfo(axisInfo) {}

  LogicalResult emitExpandShape(tensor::ExpandShapeOp expand) {
    auto resultType = dyn_cast<RankedTensorType>(expand.getResult().getType());
    if (!resultType)
      return success();

    auto resultAxesIt = axisInfo.valueAxes.find(expand.getResult());
    if (resultAxesIt == axisInfo.valueAxes.end())
      return expand.emitOpError("missing discovered expand_shape result axes");

    FailureOr<Value> expr;
    if (valueMap.contains(expand.getSrc())) {
      FailureOr<TensorAxes> sourceAxes = projectExpandSourceAxes(expand, resultAxesIt->second);
      if (failed(sourceAxes))
        return expand.emitOpError("failed to project expand_shape source axes");
      auto sourceType = cast<RankedTensorType>(expand.getSrc().getType());
      expr = translateValue(expand.getSrc(), *sourceAxes, sourceType.getElementType());
    } else {
      expr = ta.atExpandedSource(expand.getSrc(), expand.getReassociationIndices(),
                                 resultAxesIt->second, resultType.getElementType());
    }
    if (failed(expr))
      return failure();

    valueMap[expand.getResult()] =
        TensorValueInfo{*expr, getPresentTensorAxes(*expr, resultAxesIt->second)};
    return success();
  }

  LogicalResult emitGenericOp(linalg::GenericOp generic, unsigned genericIndex) {
    ScopedTABuilder::ImportGroupGuard guard(ta, genericIndex);
    Block &block = generic.getRegion().front();
    DenseMap<Value, Value> env;
    auto loopIt = axisInfo.genericLoopAxes.find(generic);
    if (loopIt == axisInfo.genericLoopAxes.end())
      return generic.emitOpError("missing discovered loop axes");
    ArrayRef<Axis> loopAxes = loopIt->second;

    for (auto [index, input] : llvm::enumerate(generic.getInputs())) {
      FailureOr<Value> expr;
      if (auto type = dyn_cast<RankedTensorType>(input.getType())) {
        auto indexingMapExprs =
            generic.getMatchingIndexingMap(&generic->getOpOperand(index)).getResults();
        auto axes = llvm::map_to_vector(
            indexingMapExprs, [&](const AffineExpr &expr) -> std::optional<Axis> {
              auto dim = dyn_cast<AffineDimExpr>(expr);
              return dim ? loopAxes[dim.getPosition()] : std::optional<Axis>();
            });
        expr = translateValue(input, axes, type.getElementType());
      } else {
        expr = translateScalarOp(env, loopAxes, input);
      }
      if (failed(expr))
        return failure();
      env[block.getArgument(index)] = *expr;
    }

    SmallVector<Axis> reductionAxes;
    for (auto [index, iterator] : llvm::enumerate(generic.getIteratorTypesArray()))
      if (iterator == utils::IteratorType::reduction)
        reductionAxes.push_back(loopAxes[index]);

    auto yield = cast<linalg::YieldOp>(block.getTerminator());
    for (OpResult result : generic->getResults()) {
      auto resultType = dyn_cast<RankedTensorType>(result.getType());
      if (result.use_empty() || !resultType)
        continue;
      auto resultAxesIt = axisInfo.valueAxes.find(result);
      if (resultAxesIt == axisInfo.valueAxes.end())
        return generic.emitOpError("missing discovered result axes");

      Value yielded = yield.getOperand(result.getResultNumber()), expr;
      if (reductionAxes.empty()) {
        auto translated = translateScalarOp(env, loopAxes, yielded);
        if (failed(translated))
          return failure();
        expr = *translated;
      } else {
        FailureOr<BinaryReductionCombinerMatch> match =
            matchBinaryReductionCombiner(generic, result.getResultNumber());
        if (failed(match))
          return failure();
        ReduceKind kind;
        if (isa<arith::AddFOp>(match->combiner))
          kind = ReduceKind::Add;
        else if (isa<arith::MulFOp>(match->combiner))
          kind = ReduceKind::Mul;
        else if (isa<arith::MaximumFOp>(match->combiner))
          kind = ReduceKind::Max;
        else if (isa<arith::MinimumFOp>(match->combiner))
          kind = ReduceKind::Min;
        else
          return generic.emitOpError("unsupported reduction combiner");
        auto translated = translateScalarOp(env, loopAxes, match->nonAccumulator);
        if (failed(translated))
          return failure();
        auto redAxes =
            llvm::map_to_vector(reductionAxes, [](const Axis &axis) { return axis.name; });
        AxisNames resultAxes;
        for (const std::optional<Axis> &axis : resultAxesIt->second)
          if (axis)
            resultAxes.push_back(axis->name);
        expr = ta.reduce(kind, *translated, redAxes, resultType.getElementType(), resultAxes);
      }
      // Use `getPresentTensorAxes` to trim "dummy" dimensions: axis discovery result
      // `resultAxesIt` describes the result tensor shape, but the linalg.generic op we're looking
      // at now may not actually traverse all dims. For example, an indexing map
      // `(d0, d1, d2) -> (d0, d1, 0)` would imply the last dimension is a dummy one.
      valueMap[result] = TensorValueInfo{expr, getPresentTensorAxes(expr, resultAxesIt->second)};
    }
    return success();
  }

  FailureOr<Value> translateFunctionResult(Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return emitError(value.getLoc()) << "ta importer expected a ranked tensor return value";
    auto axesIt = axisInfo.valueAxes.find(value);
    if (axesIt == axisInfo.valueAxes.end())
      return emitError(value.getLoc()) << "missing discovered return axes";
    return translateValue(value, axesIt->second, type.getElementType());
  }

private:
  FailureOr<Value> translateScalarOp(DenseMap<Value, Value> &env, ArrayRef<Axis> loopAxes,
                                     Value value) {
#define TRANSLATE_UNARY_ARG(def, arg)                                                              \
  auto(arg) = translateScalarOp(env, loopAxes, (def)->getOperand(0));                              \
  if (failed(input))                                                                               \
    return failure();

#define TRANSLATE_BINARY_ARGS(def, lhs, rhs)                                                       \
  auto(lhs) = translateScalarOp(env, loopAxes, (def)->getOperand(0)),                              \
  (rhs) = translateScalarOp(env, loopAxes, (def)->getOperand(1));                                  \
  if (failed(lhs) || failed(rhs))                                                                  \
    return failure();

#define BINARY_OP(OpTy)                                                                            \
  if (isa<arith::OpTy>(def)) {                                                                     \
    TRANSLATE_BINARY_ARGS(def, lhs, rhs)                                                           \
    return ta.binary<OpTy>(value.getType(), *lhs, *rhs);                                           \
  }

    if (auto it = env.find(value); it != env.end())
      return it->second;
    Operation *def = value.getDefiningOp();
    if (!def)
      return emitError(value.getLoc()) << "unsupported scalar block argument";

    if (auto constant = dyn_cast<arith::ConstantOp>(def))
      return ta.constant(cast<TypedAttr>(constant.getValue()));
    if (auto indexCast = dyn_cast<arith::IndexCastOp>(def)) {
      if (auto index = indexCast.getIn().getDefiningOp<linalg::IndexOp>()) {
        if (index.getDim() >= loopAxes.size())
          return index.emitOpError("references missing TA loop axis");
        return ta.index(loopAxes[index.getDim()], value.getType());
      }
    }
    if (auto index = dyn_cast<linalg::IndexOp>(def)) {
      if (index.getDim() >= loopAxes.size())
        return index.emitOpError("references missing TA loop axis");
      return ta.index(loopAxes[index.getDim()], value.getType());
    }
    if (isa<arith::ExtFOp, arith::TruncFOp, arith::SIToFPOp, arith::IndexCastOp>(def)) {
      TRANSLATE_UNARY_ARG(def, input);
      return ta.unary<CastOp>(value.getType(), *input);
    }
    if (isa<math::ExpOp>(def)) {
      TRANSLATE_UNARY_ARG(def, input);
      return ta.unary<ExpOp>(value.getType(), *input);
    }
    BINARY_OP(AddFOp);
    BINARY_OP(SubFOp);
    BINARY_OP(MulFOp);
    BINARY_OP(DivFOp);
    BINARY_OP(MaximumFOp);
    BINARY_OP(MinimumFOp);
    BINARY_OP(SubIOp);
    BINARY_OP(AndIOp);
    if (auto cmpf = dyn_cast<arith::CmpFOp>(def)) {
      TRANSLATE_BINARY_ARGS(def, lhs, rhs);
      return ta.cmp<CmpFOp>(cmpf.getPredicate(), *lhs, *rhs);
    }
    if (auto cmpi = dyn_cast<arith::CmpIOp>(def)) {
      TRANSLATE_BINARY_ARGS(def, lhs, rhs);
      return ta.cmp<CmpIOp>(cmpi.getPredicate(), *lhs, *rhs);
    }
    if (isa<arith::SelectOp>(def)) {
      auto condition = translateScalarOp(env, loopAxes, def->getOperand(0)),
           trueValue = translateScalarOp(env, loopAxes, def->getOperand(1)),
           falseValue = translateScalarOp(env, loopAxes, def->getOperand(2));
      if (failed(condition) || failed(trueValue) || failed(falseValue))
        return failure();
      return ta.select(*condition, *trueValue, *falseValue);
    }
    return def->emitOpError("unsupported scalar op for ta import: ") << def->getName();
#undef BINARY_OP
#undef TRANSLATE_UNARY_ARG
#undef TRANSLATE_BINARY_ARGS
  }

  FailureOr<Value> translateValue(Value value, const TensorAxes &targetAxes, Type scalarTy) {
    auto it = valueMap.find(value);
    if (it != valueMap.end()) {
      // Apply a relabeling to the previously translated expression if the target axes differ from
      // the original discovery result.
      if (it->second.axes.size() != targetAxes.size())
        return emitError(value.getLoc(), "cannot relabel tensor with different rank");
      AxisNameMapVector replacements;
      for (auto [fromAxis, toAxis] : llvm::zip_equal(it->second.axes, targetAxes)) {
        if (!fromAxis)
          continue;
        if (!toAxis)
          return emitError(value.getLoc(), "cannot erase tensor axis during relabel");
        auto [replacement, inserted] = replacements.try_emplace(fromAxis->name, toAxis->name);
        if (!inserted && replacement->second != toAxis->name)
          return emitError(value.getLoc(), "cannot relabel tensor axis to multiple targets");
      }
      return ta.subst(it->second.expr, replacements);
    }

    return ta.at(value, targetAxes, scalarTy);
  }

  // Return tensor-dimension axes for a translated value, preserving tensor rank but replacing
  // axes absent from the expression support with nullopt. For example, if a tensor has axes
  // `[b, h, i, u]` but the emitted expr has type `expr<[b, h, i]>`, then `u` is a dummy
  // dimension and the recorded axes become `[b, h, i, _]`.
  TensorAxes getPresentTensorAxes(Value expr, const TensorAxes &fullAxes) {
    auto exprType = cast<ExprType>(expr.getType());
    DenseSet<StringRef> present;
    for (Attribute attr : exprType.getAxes().getAxes())
      present.insert(cast<AxisAttr>(attr).getName().getValue());

    TensorAxes result;
    result.reserve(fullAxes.size());
    for (const std::optional<Axis> &axis : fullAxes)
      result.push_back(axis && present.contains(axis->name) ? axis : std::optional<Axis>{});
    return result;
  }

  // Project an expanded result back to source axes so `emitExpandShape` can reuse a translated
  // source expression. Example: the same tensor<1024> source may be expanded as row
  // tensor<1x1024> or column tensor<1024x1>, so choose the source-axis target from this expand.
  FailureOr<TensorAxes> projectExpandSourceAxes(tensor::ExpandShapeOp expand,
                                                const TensorAxes &targetAxes) {
    auto sourceIt = axisInfo.valueAxes.find(expand.getSrc());
    if (sourceIt == axisInfo.valueAxes.end() ||
        sourceIt->second.size() != expand.getReassociationIndices().size())
      return failure();

    TensorAxes sourceAxes;
    sourceAxes.reserve(sourceIt->second.size());
    for (auto [sourceDim, group] : llvm::enumerate(expand.getReassociationIndices())) {
      const std::optional<Axis> &sourceAxis = sourceIt->second[sourceDim];
      if (!sourceAxis) {
        sourceAxes.push_back(std::nullopt);
        continue;
      }

      std::optional<Axis> selected;
      SmallVector<Axis> extentMatches;
      for (int64_t dim : group) {
        if (dim < 0 || dim >= static_cast<int64_t>(targetAxes.size()))
          return failure();
        const std::optional<Axis> &targetAxis = targetAxes[dim];
        if (!targetAxis)
          continue;
        if (targetAxis->name == sourceAxis->name) {
          selected = targetAxis;
          break;
        }
        if (targetAxis->extent == sourceAxis->extent)
          extentMatches.push_back(*targetAxis);
      }
      if (!selected && extentMatches.size() == 1)
        selected = extentMatches.front();
      sourceAxes.push_back(std::move(selected));
    }
    return sourceAxes;
  }

  ScopedTABuilder &ta;
  const DiscoveredAxisInfo &axisInfo;
  DenseMap<Value, TensorValueInfo> valueMap;
};

static LogicalResult importFunctionAsTA(func::FuncOp func, func::ReturnOp returnOp,
                                        RankedTensorType resultType) {
  ForwardAxisDiscovery discovery;
  FailureOr<DiscoveredAxisInfo> axisInfo = discovery.run(func);
  if (failed(axisInfo))
    return failure();

  SmallVector<Operation *> oldOps;
  for (Operation &op : func.front().without_terminator())
    oldOps.push_back(&op);

  ScopedTABuilder ta(returnOp, returnOp.getLoc(), resultType);
  FunctionEmitter emitter(ta, *axisInfo);
  unsigned genericIndex = 0;
  for (Operation *op : oldOps) {
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
      if (failed(emitter.emitExpandShape(expand)))
        return failure();
    } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      if (failed(emitter.emitGenericOp(generic, genericIndex++)))
        return failure();
    }
  }

  FailureOr<Value> result = emitter.translateFunctionResult(returnOp.getOperand(0));
  if (failed(result))
    return failure();
  ta.yield(*result);
  ta.relabelAxesForOutput(*result);
  returnOp.setOperand(0, ta.getScope()->getResult(0));

  for (Operation *op : llvm::reverse(oldOps))
    if (op->use_empty())
      op->erase();
  return success();
}

/* Code for expandFullyWrappedGenericOps begins: */

using LoopExpansion = SmallVector<SmallVector<int64_t>>;

struct FullyWrappedGenericMatch {
  linalg::GenericOp generic;
  SmallVector<tensor::CollapseShapeOp> inputCollapses;
  SmallVector<tensor::ExpandShapeOp> resultExpands;
  LoopExpansion loopExpansion;
};

// Infer how each original linalg loop dimension is split in the expanded op
// from one wrapped operand. For example, if an operand map projects loop d0 to
// tensor dim 0 and that tensor dim was collapsed from [0, 1], then loop d0
// expands to two loop dimensions in the rewritten generic.
static FailureOr<LoopExpansion>
computeLoopExpansion(AffineMap seedMap, ArrayRef<ReassociationIndices> seedReassociation) {
  if (seedMap.getNumResults() != seedReassociation.size())
    return failure();

  SmallVector<unsigned> numExpandedDims(seedMap.getNumDims(), 1);
  for (auto [resultIndex, expr] : llvm::enumerate(seedMap.getResults())) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim || seedReassociation[resultIndex].empty())
      return failure();
    numExpandedDims[dim.getPosition()] = seedReassociation[resultIndex].size();
  }

  LoopExpansion loopExpansion;
  loopExpansion.reserve(numExpandedDims.size());
  int64_t nextExpandedDim = 0;
  for (unsigned dimCount : numExpandedDims) {
    SmallVector<int64_t> expandedDims;
    expandedDims.reserve(dimCount);
    for (unsigned i = 0; i < dimCount; ++i)
      expandedDims.push_back(nextExpandedDim++);
    loopExpansion.push_back(std::move(expandedDims));
  }
  return loopExpansion;
}

static LogicalResult matchExpectedReassociation(AffineMap indexingMap,
                                                ArrayRef<SmallVector<int64_t>> loopExpandedDims,
                                                ArrayRef<ReassociationIndices> reassocs) {
  if (indexingMap.getNumResults() != reassocs.size())
    return failure();
  int64_t nextTensorDim = 0;
  for (auto [resultIndex, expr] : llvm::enumerate(indexingMap.getResults())) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim || dim.getPosition() >= loopExpandedDims.size())
      return failure();
    auto &expectedDims = loopExpandedDims[dim.getPosition()];
    auto &reassoc = reassocs[resultIndex];
    if (reassoc.size() != expectedDims.size())
      return failure();
    for (size_t i = 0, e = expectedDims.size(); i < e; ++i)
      if (reassoc[i] != nextTensorDim++)
        return failure();
  }
  return success();
}

static FailureOr<AffineMap> getExpandedIndexingMap(OpBuilder &builder, AffineMap indexingMap,
                                                   const LoopExpansion &loopExpansion) {
  SmallVector<AffineExpr> newExprs;
  for (AffineExpr expr : indexingMap.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim || dim.getPosition() >= loopExpansion.size())
      return failure();

    for (int64_t expandedDim : loopExpansion[dim.getPosition()])
      newExprs.push_back(builder.getAffineDimExpr(static_cast<unsigned>(expandedDim)));
  }

  size_t expandedLoopRank = 0;
  for (auto &expandedDims : loopExpansion)
    expandedLoopRank += expandedDims.size();
  return AffineMap::get(expandedLoopRank, indexingMap.getNumSymbols(), newExprs,
                        builder.getContext());
}

static SmallVector<ReassociationIndices>
getReassociationForExpansion(AffineMap indexingMap, const LoopExpansion &loopExpansion) {
  SmallVector<ReassociationIndices> reassociation;
  int64_t nextTensorDim = 0;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned dim = cast<AffineDimExpr>(expr).getPosition();
    ReassociationIndices group;
    group.reserve(loopExpansion[dim].size());
    for (size_t i = 0, e = loopExpansion[dim].size(); i < e; ++i)
      group.push_back(nextTensorDim++);
    reassociation.push_back(std::move(group));
  }
  return reassociation;
}

static bool hasOnlyProjectedPermutationIndexingMaps(linalg::GenericOp generic) {
  return llvm::all_of(generic.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isProjectedPermutation(); });
}

static bool hasLinalgIndexOps(linalg::GenericOp generic) {
  return !generic.getRegion().front().getOps<linalg::IndexOp>().empty();
}

static FailureOr<FullyWrappedGenericMatch> matchFullyWrappedGeneric(linalg::GenericOp genericOp) {
  if (!genericOp.hasPureTensorSemantics() || !hasOnlyProjectedPermutationIndexingMaps(genericOp) ||
      hasLinalgIndexOps(genericOp) || genericOp->getNumResults() == 0)
    return failure();

  OpOperand *seedOperand = nullptr;
  tensor::CollapseShapeOp seedCollapse;
  SmallVector<tensor::CollapseShapeOp> inputCollapses;
  inputCollapses.reserve(genericOp.getNumDpsInputs());
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    if (!isa<RankedTensorType>(operand->get().getType()))
      return failure();

    auto collapse = operand->get().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapse || !collapse->hasOneUse())
      return failure();

    if (!seedOperand) {
      seedOperand = operand;
      seedCollapse = collapse;
    }
    inputCollapses.push_back(collapse);
  }
  if (!seedOperand)
    return failure();

  auto loopExpansion = computeLoopExpansion(genericOp.getMatchingIndexingMap(seedOperand),
                                            seedCollapse.getReassociationIndices());
  if (failed(loopExpansion))
    return failure();

  for (auto [operand, collapse] :
       llvm::zip_equal(genericOp.getDpsInputOperands(), inputCollapses)) {
    if (failed(matchExpectedReassociation(genericOp.getMatchingIndexingMap(operand), *loopExpansion,
                                          collapse.getReassociationIndices())))
      return failure();
  }

  SmallVector<tensor::ExpandShapeOp> resultExpands;
  resultExpands.reserve(genericOp->getNumResults());
  for (OpResult result : genericOp->getResults()) {
    if (!isa<RankedTensorType>(result.getType()) || !result.hasOneUse())
      return failure();

    auto expand = dyn_cast<tensor::ExpandShapeOp>(*result.getUsers().begin());
    if (!expand || expand.getSrc() != result)
      return failure();

    OpOperand *initOperand = genericOp.getDpsInitOperand(result.getResultNumber());
    if (failed(matchExpectedReassociation(genericOp.getMatchingIndexingMap(initOperand),
                                          *loopExpansion, expand.getReassociationIndices())))
      return failure();
    resultExpands.push_back(expand);
  }

  return FullyWrappedGenericMatch{genericOp, std::move(inputCollapses), std::move(resultExpands),
                                  *loopExpansion};
}

static void debugPrintMatch(const FullyWrappedGenericMatch &match) {
  LLVM_DEBUG({
    llvm::dbgs() << "linalg-to-ta: fully wrapped linalg.generic candidate:\n";
    match.generic->print(llvm::dbgs());
    llvm::dbgs() << "\n";
    for (tensor::CollapseShapeOp collapse : match.inputCollapses) {
      llvm::dbgs() << "  input collapse: ";
      collapse->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
    for (tensor::ExpandShapeOp expand : match.resultExpands) {
      llvm::dbgs() << "  result expand: ";
      expand->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });
}

static LogicalResult rewriteFullyWrappedGeneric(FullyWrappedGenericMatch match) {
  IRRewriter rewriter(match.generic.getContext());
  rewriter.setInsertionPoint(match.resultExpands.front());
  Location loc = match.generic.getLoc();

  SmallVector<Type> resultTypes;
  SmallVector<Value> outputs;
  resultTypes.reserve(match.resultExpands.size());
  outputs.reserve(match.resultExpands.size());
  for (auto [index, expansion] : llvm::enumerate(match.resultExpands)) {
    RankedTensorType resultType = expansion.getResultType();
    resultTypes.push_back(resultType);
    OpOperand *init = match.generic.getDpsInitOperand(static_cast<int64_t>(index));
    AffineMap indexingMap = match.generic.getMatchingIndexingMap(init);
    SmallVector<ReassociationIndices> reassociation =
        getReassociationForExpansion(indexingMap, match.loopExpansion);
    outputs.push_back(tensor::ExpandShapeOp::create(
        rewriter, loc, resultType, init->get(), reassociation, expansion.getMixedOutputShape()));
  }

  OpBuilder mapBuilder(match.generic.getContext());
  SmallVector<AffineMap> expandedIndexingMaps;
  expandedIndexingMaps.reserve(match.generic.getIndexingMapsArray().size());
  for (AffineMap map : match.generic.getIndexingMapsArray()) {
    FailureOr<AffineMap> expandedMap = getExpandedIndexingMap(mapBuilder, map, match.loopExpansion);
    if (failed(expandedMap))
      return failure();
    expandedIndexingMaps.push_back(*expandedMap);
  }

  size_t expandedLoopRank = 0;
  for (auto &expandedDims : match.loopExpansion)
    expandedLoopRank += expandedDims.size();
  SmallVector<utils::IteratorType> expandedIteratorTypes(expandedLoopRank,
                                                         utils::IteratorType::parallel);
  for (auto [index, iterator] : llvm::enumerate(match.generic.getIteratorTypesArray()))
    for (int64_t expandedDim : match.loopExpansion[index])
      expandedIteratorTypes[expandedDim] = iterator;

  auto expandedInputs = llvm::to_vector(llvm::map_range(
      match.inputCollapses, [](auto collapse) { return (Value)collapse.getSrc(); }));
  auto expandedGeneric =
      linalg::GenericOp::create(rewriter, loc, resultTypes, expandedInputs, outputs,
                                expandedIndexingMaps, expandedIteratorTypes);
  rewriter.cloneRegionBefore(match.generic.getRegion(), expandedGeneric.getRegion(),
                             expandedGeneric.getRegion().begin());

  for (auto [index, expand] : llvm::enumerate(match.resultExpands))
    rewriter.replaceOp(expand, expandedGeneric->getResult(index));

  rewriter.eraseOp(match.generic);
  for (tensor::CollapseShapeOp collapse : match.inputCollapses)
    if (collapse->use_empty())
      rewriter.eraseOp(collapse);

  return success();
}

static LogicalResult expandFullyWrappedGenericOps(func::FuncOp func) {
  SmallVector<linalg::GenericOp> generics;
  func.walk([&](linalg::GenericOp generic) { generics.push_back(generic); });

  for (linalg::GenericOp generic : generics) {
    FailureOr<FullyWrappedGenericMatch> match = matchFullyWrappedGeneric(generic);
    if (failed(match))
      continue;

    debugPrintMatch(*match);
    if (failed(rewriteFullyWrappedGeneric(std::move(*match)))) {
      LLVM_DEBUG(llvm::dbgs() << "linalg-to-ta: skipped candidate; rewrite failed\n");
      continue;
    }
  }

  return success();
}

} // namespace

void registerLinalgToTAPass() { PassRegistration<ImportLinalgToTAPass>(); }

} // namespace ta

#undef DEBUG_TYPE
