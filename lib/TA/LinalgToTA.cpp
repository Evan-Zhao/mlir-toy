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

// Normalize linalg.generic ops of the form:
//   collapse_shape(inputs) -> linalg.generic -> expand_shape(results)
// into a higher-rank linalg.generic over the original input/result shapes.
// This keeps axis discovery forward-only by eliminating matched reshape
// sandwiches before the TA importer reasons about tensor axes.
static LogicalResult expandFullyWrappedGenericOps(func::FuncOp func);

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
    if (func.empty())
      return;
    if (failed(expandFullyWrappedGenericOps(func)))
      return signalPassFailure();
    llvm::dbgs() << "func = " << func << "\n";
    if (failed(discoverAndPrintAxes(func)))
      return signalPassFailure();
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
