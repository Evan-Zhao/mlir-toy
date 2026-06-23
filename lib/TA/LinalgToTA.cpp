#include "TA/TADialect.h"
#include "TA/TAPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

LogicalResult discoverAndPrintAxes(func::FuncOp func);

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
    llvm::dbgs() << "func = " << func << "\n";
    if (failed(discoverAndPrintAxes(func))) {
      signalPassFailure();
      return;
    }
  }
};

struct Axis {
  std::string name;
  int64_t extent = ShapedType::kDynamic;
};

using TensorAxes = SmallVector<std::optional<Axis>, 4>;

struct DiscoveredAxisInfo {
  DenseMap<Value, TensorAxes> valueAxes;
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
        return id ? axisAttrs[find(*id)] : std::optional<Axis>{};
      });
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
    axisAttrs.push_back(Axis{std::move(name), extent});
    return id;
  }

  AxisId find(AxisId id) {
    if (parent[id] == id)
      return id;
    parent[id] = find(parent[id]);
    return parent[id];
  }

  bool axesCompatible(AxisId lhs, AxisId rhs) const {
    int64_t lhsExtent = axisAttrs[lhs].extent;
    int64_t rhsExtent = axisAttrs[rhs].extent;
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
    for (int64_t index = 0, end = generic.getNumDpsInits(); index < end; ++index) {
      if (failed(processOperand(generic.getDpsInitOperand(index))))
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
  DenseMap<AxisId, SmallVector<AxisId>> productAxes;
  SmallVector<AxisId> parent;
  SmallVector<Axis> axisAttrs;
  AxisId nextValueAxis = 0;
};

static void printDiscoveredAxes(func::FuncOp func, const DiscoveredAxisInfo &info) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  AsmState asmState(func);
  llvm::StringMap<unsigned> relabeledAxes;
  auto printValueAxes = [&](Value value) {
    if (!isa<RankedTensorType>(value.getType()))
      return;
    os << "  ";
    value.printAsOperand(os, asmState);
    os << " :";
    auto it = info.valueAxes.find(value);
    if (it == info.valueAxes.end()) {
      os << " (unknown axes)\n";
      return;
    }
    auto &axes = it->second;
    os << " [";
    llvm::interleaveComma(axes, os, [&](const std::optional<Axis> &axis) {
      if (!axis) {
        os << "_";
      } else {
        auto [it, inserted] = relabeledAxes.try_emplace(axis->name, relabeledAxes.size());
        os << "v" << it->second << "(" << axis->extent << ")";
      }
    });
    os << "]\n";
  };

  os << "linalg-to-ta axis discovery for @" << func.getName() << ":\n";
  for (BlockArgument arg : func.front().getArguments())
    printValueAxes(arg);
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (Value result : op->getResults())
      printValueAxes(result);
  });
  for (auto &entry : info.valueAxes) {
    if (auto blockArg = dyn_cast<BlockArgument>(entry.first);
        blockArg && blockArg.getOwner()->getParentOp() == func)
      continue;
    if (auto result = dyn_cast<OpResult>(entry.first);
        result && func->isAncestor(result.getOwner()))
      continue;
    printValueAxes(entry.first);
  }
  os.flush();
  llvm::errs() << buffer;
}

static LogicalResult discoverAndPrintAxes(func::FuncOp func) {
  ForwardAxisDiscovery discovery;
  FailureOr<DiscoveredAxisInfo> info = discovery.run(func);
  if (failed(info))
    return failure();
  printDiscoveredAxes(func, *info);
  return success();
}

} // namespace

void registerLinalgToTAPass() { PassRegistration<ImportLinalgToTAPass>(); }

} // namespace ta
