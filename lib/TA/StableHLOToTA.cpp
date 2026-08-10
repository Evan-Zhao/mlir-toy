#include "TA/TAOps.h"
#include "TA/TAPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "stablehlo-to-ta"

namespace ta {
#define GEN_PASS_DEF_STABLEHLOTOTAPASS
#include "TAPasses.h.inc"

using namespace mlir;

namespace {

// The importer runs in two phases. AxisDiscovery first assigns logical axes to every tensor
// dimension, unifies dimensions related by StableHLO attributes, and determines which unit axes
// remain structurally present. FunctionEmitter then replays the dataflow as scalar TA expressions
// over those axes. Keeping discovery separate is important for dot_general: an axis may acquire its
// final identity only after several producers and consumers have been visited.
//
// StableHLO operations remain in place around the imported islands. Each supported value crossing
// into unsupported IR or a function return becomes the root of a single-result TA scope.

// An Axis is a logical iteration variable, not necessarily one physical tensor dimension. Names
// are provisional during discovery and are normalized to output i* and internal j* names later.
struct Axis {
  std::string name;
  int64_t extent = ShapedType::kDynamic;
};

// Preserve tensor rank while allowing dimensions which do not occur in an expression's support.
// A null entry means that the expression is constant along that tensor dimension. This is how a
// scalar broadcast or an expanding size-one dimension is represented without a TA broadcast op.
using TensorAxes = SmallVector<std::optional<Axis>>;

// This is the stable, name-based form of discovery output consumed by FunctionEmitter. Discovery
// itself uses compact union-find IDs because axis equivalences are still changing during the walk.
struct DiscoveredAxisInfo {
  DenseMap<Value, TensorAxes> valueAxes;
  SmallVector<Value> roots;
};

// StableHLO reshape carries source and result shapes but no tensor-dialect reassociation. Infer the
// reassociation accepted by tensor.expand_shape/tensor.collapse_shape when possible. Equal-rank
// reshapes are supported only when they are identity reshapes; rank-zero reshapes are the
// singleton-only case which has no reassociation groups.
static FailureOr<SmallVector<ReassociationIndices>>
inferReshapeReassociation(stablehlo::ReshapeOp op) {
  auto sourceType = cast<RankedTensorType>(op.getOperand().getType());
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  if (sourceType.getRank() == resultType.getRank()) {
    if (sourceType.getShape() != resultType.getShape()) {
      op.emitOpError("equal-rank non-identity reshapes are not reassociative");
      return failure();
    }
    SmallVector<ReassociationIndices> identity;
    for (int64_t dim = 0; dim < sourceType.getRank(); ++dim)
      identity.push_back({dim});
    return identity;
  }
  if (sourceType.getRank() == 0 || resultType.getRank() == 0)
    return SmallVector<ReassociationIndices>{};
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(sourceType, resultType);
  if (!reassociation) {
    op.emitOpError("reshape is not representable as a reassociative reshape");
    return failure();
  }
  return std::move(*reassociation);
}

static bool isDeferredProductCollapse(Operation *op) {
  auto reshape = dyn_cast<stablehlo::ReshapeOp>(op);
  if (!reshape)
    return false;
  auto sourceType = cast<RankedTensorType>(reshape.getOperand().getType());
  auto resultType = cast<RankedTensorType>(reshape.getResult().getType());
  if (sourceType.getRank() <= resultType.getRank())
    return false;
  auto reassociation = inferReshapeReassociation(reshape);
  return succeeded(reassociation) &&
         llvm::any_of(*reassociation,
                      [](ReassociationIndicesRef group) { return group.size() > 1; });
}

// Keep this predicate in one place so root selection and per-scope emission agree on where a TA
// island may continue. Unsupported operations remain in the surrounding function as tensor
// producers or consumers.
static bool isSupportedStableHLOOp(Operation *op) {
  if (isDeferredProductCollapse(op))
    return false;
  return isa<arith::ConstantOp, stablehlo::ConstantOp, stablehlo::IotaOp, stablehlo::DynamicIotaOp,
             stablehlo::ConvertOp, stablehlo::ExpOp, stablehlo::AddOp, stablehlo::SubtractOp,
             stablehlo::MulOp, stablehlo::DivOp, stablehlo::MaxOp, stablehlo::MinOp,
             stablehlo::AndOp, stablehlo::CompareOp, stablehlo::SelectOp, stablehlo::TransposeOp,
             stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp, stablehlo::ReduceOp,
             stablehlo::DotGeneralOp>(op);
}

/// Discover tensor-axis equality and structural presence before emitting TA.
///
/// StableHLO makes most equalities explicit in operation dimension-number attributes, unlike
/// linalg.generic where indexing maps and synthetic loop axes provide the connections. Every
/// static tensor dimension starts in its own union-find set. A forward walk merges sets when two
/// dimensions denote the same logical coordinate. Once all unions are complete, presence analysis
/// retains structural unit axes while omitting unit factors used only for expanding broadcasts.
class AxisDiscovery {
public:
  FailureOr<DiscoveredAxisInfo> run(func::FuncOp func) {
    for (BlockArgument argument : func.getArguments())
      if (isa<RankedTensorType>(argument.getType()) && failed(getOrCreateValueAxes(argument)))
        return failure();

    for (Operation &op : func.front().without_terminator()) {
      LogicalResult result = success();
      if (isa<stablehlo::ConvertOp, stablehlo::ExpOp, stablehlo::AddOp, stablehlo::SubtractOp,
              stablehlo::MulOp, stablehlo::DivOp, stablehlo::MaxOp, stablehlo::MinOp,
              stablehlo::AndOp, stablehlo::CompareOp>(&op))
        result = discoverSameShape(&op);
      else if (auto select = dyn_cast<stablehlo::SelectOp>(&op))
        result = discoverSelect(select);
      else if (auto transpose = dyn_cast<stablehlo::TransposeOp>(&op))
        result = discoverTranspose(transpose);
      else if (auto broadcast = dyn_cast<stablehlo::BroadcastInDimOp>(&op))
        result = discoverBroadcast(broadcast);
      else if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(&op))
        result = discoverReshape(reshape);
      else if (auto reduce = dyn_cast<stablehlo::ReduceOp>(&op))
        result = discoverReduce(reduce);
      else if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(&op))
        result = discoverDot(dot);
      else
        for (Value value : llvm::concat<Value>(op.getOperands(), op.getResults()))
          if (isa<RankedTensorType>(value.getType()) && failed(getOrCreateValueAxes(value)))
            return op.emitOpError("axis discovery requires static ranked tensors");
      if (failed(result))
        return failure();
    }

    auto returnOp = cast<func::ReturnOp>(func.front().getTerminator());
    for (Value value : returnOp.getOperands())
      if (isa<RankedTensorType>(value.getType()) && failed(getOrCreateValueAxes(value)))
        return returnOp.emitOpError("axis discovery requires static ranked tensors");

    SmallVector<Value> roots = collectRoots(func);
    discoverPresentAxes(func, roots);

    // Freeze union-find representatives into value-owned Axis records. Unit dimensions omitted by
    // presence analysis become null tensor axes; all non-unit dimensions remain present.
    DiscoveredAxisInfo info;
    info.roots = std::move(roots);
    for (auto &[value, ids] : valueAxisIds)
      info.valueAxes[value] = llvm::map_to_vector(ids, [&](AxisId id) -> std::optional<Axis> {
        id = find(id);
        if (axes[id].extent == 1 && !presentAxes.test(id))
          return std::nullopt;
        return axes[id];
      });
    return info;
  }

private:
  using AxisId = unsigned;
  using AxisIds = SmallVector<AxisId, 4>;

  // Return one union-find ID per tensor dimension. ArrayRef keeps call sites lightweight while the
  // vectors remain owned by valueAxisIds for the lifetime of discovery. Dynamic shapes are
  // rejected here so all later axis materialization can use compile-time extents.
  FailureOr<AxisIds> getOrCreateValueAxes(Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !type.hasStaticShape())
      return failure();
    auto [it, inserted] = valueAxisIds.try_emplace(value);
    if (!inserted)
      return it->second;
    for (int64_t extent : type.getShape())
      it->second.push_back(makeAxis(("v" + Twine(nextAxisName++)).str(), extent));
    return it->second;
  }

  AxisId makeAxis(std::string name, int64_t extent) {
    AxisId id = parent.size();
    parent.push_back(id);
    axes.push_back(Axis{std::move(name), extent});
    return id;
  }

  // Standard path-compressed union-find. The Axis stored at the representative supplies both the
  // eventual name and the static extent for every member of the equivalence class.
  AxisId find(AxisId id) {
    if (parent[id] != id)
      parent[id] = find(parent[id]);
    return parent[id];
  }

  SmallVector<Value> collectRoots(func::FuncOp func) {
    auto hasUnsupportedUser = [](Value value) {
      return llvm::any_of(value.getUses(),
                          [](OpOperand &use) { return !isSupportedStableHLOOp(use.getOwner()); });
    };

    SmallVector<Value> roots;
    for (Operation &op : func.front().without_terminator()) {
      if (!isSupportedStableHLOOp(&op))
        continue;
      for (OpResult result : op.getResults())
        if (isa<RankedTensorType>(result.getType()) && hasUnsupportedUser(result))
          roots.push_back(result);
    }
    return roots;
  }

  void markPresent(AxisId id) { presentAxes.set(find(id)); }

  void markValueDimPresent(Value value, int64_t dim) {
    auto it = valueAxisIds.find(value);
    if (it == valueAxisIds.end() || dim < 0 || dim >= static_cast<int64_t>(it->second.size()))
      return;
    markPresent(it->second[dim]);
  }

  // Equality discovery already propagates presence through elementwise operations, transposes,
  // equal-size broadcasts, and visible dot/reduction dimensions. Seed external boundaries and
  // internal iteration-only dimensions, then carry factor demand back to collapsed products.
  void discoverPresentAxes(func::FuncOp func, ArrayRef<Value> roots) {
    presentAxes.resize(parent.size());

    // Presence analysis only decides the fate of unit axes. Non-unit coordinates always affect
    // indexing and remain part of expression support.
    for (AxisId id = 0; id < axes.size(); ++id)
      if (axes[find(id)].extent != 1)
        markPresent(id);

    for (Value root : roots) {
      auto it = valueAxisIds.find(root);
      if (it != valueAxisIds.end())
        for (AxisId id : it->second)
          markPresent(id);
    }

    for (Operation &operation : func.front().without_terminator()) {
      if (auto iota = dyn_cast<stablehlo::IotaOp>(&operation))
        markValueDimPresent(iota.getResult(), iota.getIotaDimension());
      else if (auto iota = dyn_cast<stablehlo::DynamicIotaOp>(&operation))
        markValueDimPresent(iota.getResult(), iota.getIotaDimension());
      else if (auto reduce = dyn_cast<stablehlo::ReduceOp>(&operation))
        for (int64_t dim : reduce.getDimensions())
          markValueDimPresent(reduce.getInputs().front(), dim);
      else if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(&operation)) {
        auto dims = dot.getDotDimensionNumbers();
        for (int64_t dim : dims.getLhsContractingDimensions())
          markValueDimPresent(dot.getLhs(), dim);
        for (int64_t dim : dims.getRhsContractingDimensions())
          markValueDimPresent(dot.getRhs(), dim);
      }
    }

    bool changed;
    do {
      changed = false;
      for (auto &[product_, factors] : productAxes) {
        AxisId product = find(product_);
        if (presentAxes.test(product))
          continue;
        if (llvm::any_of(factors, [&](AxisId factor) { return presentAxes.test(find(factor)); })) {
          presentAxes.set(product);
          changed = true;
        }
      }
    } while (changed);
  }

  bool axesCompatible(AxisId lhs, AxisId rhs) const {
    int64_t lhsExtent = axes[lhs].extent;
    int64_t rhsExtent = axes[rhs].extent;
    return lhsExtent == ShapedType::kDynamic || rhsExtent == ShapedType::kDynamic ||
           lhsExtent == rhsExtent;
  }

  // Keep the relationship between a collapsed product axis and its expanded factor axes. If
  // another reshape exposes the same product with a compatible factorization, corresponding
  // factors become the same logical axes.
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

    SmallVector<AxisId> existingFactors = it->second;
    for (AxisId &axis : existingFactors)
      axis = find(axis);
    if (existingFactors.size() != incomingFactors.size())
      return success();
    for (auto [lhs, rhs] : llvm::zip_equal(existingFactors, incomingFactors))
      if (!axesCompatible(lhs, rhs))
        return success();
    for (auto [lhs, rhs] : llvm::zip_equal(existingFactors, incomingFactors))
      if (failed(unite(op, lhs, rhs)))
        return failure();
    return success();
  }

  // Axis equality also implies extent equality. StableHLO verification normally guarantees this,
  // but checking it here prevents malformed or partially transformed IR from producing invalid TA.
  LogicalResult unite(Operation *op, AxisId lhs, AxisId rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return success();
    if (!axesCompatible(lhs, rhs))
      return op->emitOpError("axis discovery found conflicting extents");
    parent[rhs] = lhs;

    if (auto it = productAxes.find(rhs); it != productAxes.end()) {
      SmallVector<AxisId> rhsFactors = std::move(it->second);
      productAxes.erase(it);
      return mergeProductFactors(op, lhs, rhsFactors);
    }
    return success();
  }

  LogicalResult processReassociation(Operation *op, ArrayRef<ReassociationIndices> reassociation,
                                     ArrayRef<AxisId> collapsedAxes,
                                     ArrayRef<AxisId> expandedAxes) {
    if (reassociation.size() != collapsedAxes.size())
      return failure();
    for (auto [collapsedDim, expandedDims] : llvm::enumerate(reassociation)) {
      if (expandedDims.size() == 1) {
        int64_t expandedDim = expandedDims.front();
        if (expandedDim < 0 || expandedDim >= static_cast<int64_t>(expandedAxes.size()) ||
            failed(unite(op, collapsedAxes[collapsedDim], expandedAxes[expandedDim])))
          return failure();
        continue;
      }

      SmallVector<AxisId> factors;
      factors.reserve(expandedDims.size());
      for (int64_t expandedDim : expandedDims) {
        if (expandedDim < 0 || expandedDim >= static_cast<int64_t>(expandedAxes.size()))
          return failure();
        factors.push_back(find(expandedAxes[expandedDim]));
      }
      if (failed(mergeProductFactors(op, collapsedAxes[collapsedDim], factors)))
        return failure();
    }
    return success();
  }

  LogicalResult discoverSameShape(Operation *op) {
    if (op->getNumResults() != 1)
      return op->emitOpError("expected one result");
    auto result = getOrCreateValueAxes(op->getResult(0));
    if (failed(result))
      return op->emitOpError("axis discovery requires static ranked tensors");
    for (Value operand : op->getOperands()) {
      auto operandAxes = getOrCreateValueAxes(operand);
      if (failed(operandAxes) || operandAxes->size() != result->size())
        return op->emitOpError("expected same-rank tensor operands and result");
      for (auto [operandAxis, resultAxis] : llvm::zip_equal(*operandAxes, *result))
        if (failed(unite(op, operandAxis, resultAxis)))
          return failure();
    }
    return success();
  }

  LogicalResult discoverSelect(stablehlo::SelectOp op) {
    auto result = getOrCreateValueAxes(op.getResult());
    auto trueValue = getOrCreateValueAxes(op.getOnTrue());
    auto falseValue = getOrCreateValueAxes(op.getOnFalse());
    auto pred = getOrCreateValueAxes(op.getPred());
    if (failed(result) || failed(trueValue) || failed(falseValue) || failed(pred) ||
        trueValue->size() != result->size() || falseValue->size() != result->size() ||
        (!pred->empty() && pred->size() != result->size()))
      return op.emitOpError("expected scalar predicate or same-shape select operands");

    for (auto [trueAxis, falseAxis, resultAxis] : llvm::zip_equal(*trueValue, *falseValue, *result))
      if (failed(unite(op, trueAxis, resultAxis)) || failed(unite(op, falseAxis, resultAxis)))
        return failure();
    if (!pred->empty())
      for (auto [predAxis, resultAxis] : llvm::zip_equal(*pred, *result))
        if (failed(unite(op, predAxis, resultAxis)))
          return failure();
    return success();
  }

  // StableHLO's permutation is result-dimension -> input-dimension. Unifying in that direction
  // records the permutation in result metadata; no explicit transpose operation is needed in TA.
  LogicalResult discoverTranspose(stablehlo::TransposeOp op) {
    auto input = getOrCreateValueAxes(op.getOperand());
    auto result = getOrCreateValueAxes(op.getResult());
    ArrayRef<int64_t> permutation = op.getPermutation();
    if (failed(input) || failed(result) || permutation.size() != result->size() ||
        input->size() != result->size())
      return op.emitOpError("invalid transpose axes");
    for (auto [resultDim, inputDim] : llvm::enumerate(permutation)) {
      if (inputDim < 0 || inputDim >= static_cast<int64_t>(input->size()) ||
          failed(unite(op, (*input)[inputDim], (*result)[resultDim])))
        return op.emitOpError("invalid transpose permutation");
    }
    return success();
  }

  LogicalResult discoverBroadcast(stablehlo::BroadcastInDimOp op) {
    auto input = getOrCreateValueAxes(op.getOperand());
    auto result = getOrCreateValueAxes(op.getResult());
    ArrayRef<int64_t> dimensions = op.getBroadcastDimensions();
    if (failed(input) || failed(result) || dimensions.size() != input->size())
      return op.emitOpError("invalid broadcast axes");
    auto inputType = cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    for (auto [inputDim, resultDim] : llvm::enumerate(dimensions)) {
      if (resultDim < 0 || resultDim >= resultType.getRank())
        return op.emitOpError("invalid broadcast dimension");
      int64_t inputExtent = inputType.getDimSize(inputDim);
      int64_t resultExtent = resultType.getDimSize(resultDim);
      // Equal extents mean the source coordinate survives the broadcast and can share the result
      // axis. An expanding size-one dimension is indexed at zero instead and does not carry the
      // destination axis into the source expression.
      if (inputExtent == resultExtent &&
          failed(unite(op, (*input)[inputDim], (*result)[resultDim])))
        return failure();
    }
    return success();
  }

  LogicalResult discoverReshape(stablehlo::ReshapeOp op) {
    auto input = getOrCreateValueAxes(op.getOperand());
    auto result = getOrCreateValueAxes(op.getResult());
    if (failed(input) || failed(result))
      return op.emitOpError("axis discovery requires static ranked tensors");
    auto reassociation = inferReshapeReassociation(op);
    if (failed(reassociation))
      return failure();

    auto inputType = cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    if (inputType.getRank() <= resultType.getRank())
      return processReassociation(op, *reassociation, *input, *result);
    return processReassociation(op, *reassociation, *result, *input);
  }

  // StableHLO reduction results list the unreduced input dimensions in input order. Walk the input
  // once, skip dimensions named by the attribute, and merge each survivor with the next result
  // dimension. Reduced axes intentionally remain available as internal TA axes.
  LogicalResult discoverReduce(stablehlo::ReduceOp op) {
    if (op.getInputs().size() != 1 || op.getNumResults() != 1)
      return op.emitOpError("only single-input reductions are supported");
    auto inputR = getOrCreateValueAxes(op.getInputs().front());
    auto resultR = getOrCreateValueAxes(op.getResult(0));
    if (failed(inputR) || failed(resultR))
      return op.emitOpError("axis discovery requires static ranked tensors");
    auto &result = *resultR;
    DenseSet<int64_t> reduced(op.getDimensions().begin(), op.getDimensions().end());
    unsigned resultDim = 0;
    for (auto [inputDim, inputAxis] : llvm::enumerate(*inputR)) {
      if (reduced.contains(static_cast<int64_t>(inputDim)))
        continue;
      if (resultDim >= result.size() || failed(unite(op, inputAxis, result[resultDim])))
        return op.emitOpError("reduction dimensions do not match result rank");
      resultDim++;
    }
    if (resultDim != result.size())
      return op.emitOpError("reduction dimensions do not match result rank");
    return success();
  }

  // A dot_general result is ordered as batching dimensions, lhs free dimensions, then rhs free
  // dimensions. Contracting dimensions do not appear in the result, but lhs and rhs contracting
  // pairs must be unified so the emitter can reduce their product over one shared axis.
  LogicalResult discoverDot(stablehlo::DotGeneralOp op) {
    auto lhsR = getOrCreateValueAxes(op.getLhs());
    auto rhsR = getOrCreateValueAxes(op.getRhs());
    auto resultR = getOrCreateValueAxes(op.getResult());
    if (failed(lhsR) || failed(rhsR) || failed(resultR))
      return op.emitOpError("axis discovery requires static ranked tensors");
    auto &lhs = *lhsR, &rhs = *rhsR, &result = *resultR;

    auto dims = op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatch = dims.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatch = dims.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContract = dims.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContract = dims.getRhsContractingDimensions();
    if (lhsBatch.size() != rhsBatch.size() || lhsContract.size() != rhsContract.size())
      return op.emitOpError("mismatched dot dimension numbers");

    auto validDim = [](int64_t dim, size_t rank) {
      return dim >= 0 && dim < static_cast<int64_t>(rank);
    };
    // Batch pairs are visible in all three tensors. Process them first because StableHLO places
    // them at the front of the result regardless of their positions in either operand.
    unsigned resultDim = 0;
    for (auto [lhsDim, rhsDim] : llvm::zip_equal(lhsBatch, rhsBatch)) {
      if (!validDim(lhsDim, lhs.size()) || !validDim(rhsDim, rhs.size()) ||
          resultDim >= result.size())
        return op.emitOpError("invalid dot batching dimensions");
      if (failed(unite(op, lhs[lhsDim], rhs[rhsDim])) ||
          failed(unite(op, lhs[lhsDim], result[resultDim])))
        return failure();
      ++resultDim;
    }

    // Append one operand's free dimensions to the result. Batch and contracting dimensions are
    // excluded; iterating the original operand order implements StableHLO's result layout rule.
    auto uniteDotDims = [&](auto &batch, auto &contract, auto &axes) -> LogicalResult {
      DenseSet<int64_t> skipped(batch.begin(), batch.end());
      skipped.insert(contract.begin(), contract.end());
      for (auto [dim, axis] : llvm::enumerate(axes)) {
        if (skipped.contains(static_cast<int64_t>(dim)))
          continue;
        if (resultDim >= result.size())
          return op.emitOpError("dot dimensions do not match result rank");
        if (failed(unite(op, axis, result[resultDim])))
          return failure();
        ++resultDim;
      }
      return success();
    };
    if (failed(uniteDotDims(lhsBatch, lhsContract, lhs)) ||
        failed(uniteDotDims(rhsBatch, rhsContract, rhs)))
      return failure();
    // Contracting pairs share an internal coordinate but have no corresponding result dimension.
    // They are processed last because they do not advance resultDim.
    for (auto [lhsDim, rhsDim] : llvm::zip_equal(lhsContract, rhsContract)) {
      if (!validDim(lhsDim, lhs.size()) || !validDim(rhsDim, rhs.size()) ||
          failed(unite(op, lhs[lhsDim], rhs[rhsDim])))
        return op.emitOpError("invalid dot contracting dimensions");
    }
    if (resultDim != result.size())
      return op.emitOpError("dot dimensions do not match result rank");
    return success();
  }

  DenseMap<Value, AxisIds> valueAxisIds;
  DenseMap<AxisId, SmallVector<AxisId>> productAxes;
  SmallVector<AxisId> parent;
  SmallVector<Axis> axes;
  llvm::SmallBitVector presentAxes;
  unsigned nextAxisName = 0;
};

using AxisName = std::string;
using AxisNames = SmallVector<AxisName>;
using AxisNameMapVector = llvm::MapVector<AxisName, AxisName, llvm::StringMap<unsigned>>;

// The builder owns one ta.scope and lazily adds its index block arguments as expressions begin to
// use axes. StableHLO operation locations and import-group tags are applied through the guard
// below, so all TA operations emitted for one source operation can later be materialized together.
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

  // Temporarily associate newly created TA operations with one StableHLO source operation. The
  // ta-to-linalg lowering uses this grouping to choose useful materialization boundaries.
  class ImportGroupGuard {
  public:
    ImportGroupGuard(ScopedTABuilder &ta, int64_t group, Location loc)
        : ta(ta), oldGroup(ta.importGroup), oldLoc(ta.loc) {
      ta.importGroup = group;
      ta.loc = loc;
    }
    ~ImportGroupGuard() {
      ta.importGroup = oldGroup;
      ta.loc = oldLoc;
    }

  private:
    ScopedTABuilder &ta;
    std::optional<int64_t> oldGroup;
    Location oldLoc;
  };

  ScopeOp getScope() const { return scope; }

  AxesAttr getAxesAttr(ArrayRef<AxisName> names) const {
    SmallVector<Attribute> result;
    DenseSet<StringRef> seen;
    for (StringRef name : names)
      if (seen.insert(name).second)
        result.push_back(AxisAttr::get(context, name));
    return AxesAttr::get(context, ArrayAttr::get(context, result));
  }

  ExprType getExprType(Type elementType, ArrayRef<AxisName> axes) const {
    return ExprType::get(context, elementType, getAxesAttr(axes));
  }

  // TA elementwise result axes are the ordered union of operand axes. Keep the first occurrence of
  // each name so operand order remains visible and agrees with the TA verifier's inference rule.
  AxisNames collectOperandAxes(ValueRange operands) const {
    DenseSet<StringRef> seen;
    AxisNames result;
    for (Value operand : operands)
      for (Attribute attr : cast<ExprType>(operand.getType()).getAxes().getAxes()) {
        StringRef name = cast<AxisAttr>(attr).getName().getValue();
        if (seen.insert(name).second)
          result.push_back(name.str());
      }
    return result;
  }

  // Observe a tensor using one coordinate per physical dimension. Missing logical axes become a
  // literal zero index, which implements scalar and singleton broadcasts without adding that axis
  // to the resulting expression type.
  Value at(Value source, const TensorAxes &dimAxes, Type elementType) {
    SmallVector<Value> indices;
    AxisNames resultAxes;
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

  // Observe the lower-rank source of a reassociative expansion. Each expanded group is linearized
  // into the corresponding source coordinate.
  FailureOr<Value> atExpandedSource(Value source, ArrayRef<ReassociationIndices> reassociation,
                                    const TensorAxes &resultDimAxes, Type elementType) {
    int64_t sourceRank = cast<RankedTensorType>(source.getType()).getRank();
    if (static_cast<int64_t>(reassociation.size()) != sourceRank)
      return emitError(source.getLoc(), "reshape reassociation does not match source rank");

    SmallVector<Value> indices;
    indices.reserve(reassociation.size());
    AxisNames resultAxes;
    resultAxes.reserve(resultDimAxes.size());
    for (const ReassociationIndices &group : reassociation) {
      SmallVector<Value> groupIndices;
      SmallVector<int64_t> basis;
      for (int64_t resultDim : group) {
        if (resultDim < 0 || resultDim >= static_cast<int64_t>(resultDimAxes.size()))
          return emitError(source.getLoc(), "reshape reassociation references invalid dimension");
        const std::optional<Axis> &axis = resultDimAxes[resultDim];
        // Presence analysis has already omitted factors used only for expanding broadcasts.
        // Retain every remaining factor, including structural unit axes such as MQA's KV head.
        if (!axis)
          continue;
        if (axis->extent == ShapedType::kDynamic)
          return emitError(source.getLoc(), "cannot linearize dynamic expanded axis");
        groupIndices.push_back(materializeAxis(*axis));
        basis.push_back(axis->extent);
        resultAxes.push_back(axis->name);
      }

      if (groupIndices.empty())
        indices.push_back(indexZero());
      else if (groupIndices.size() == 1)
        indices.push_back(groupIndices.front());
      else
        indices.push_back(affine::AffineLinearizeIndexOp::create(builder, loc, groupIndices, basis,
                                                                 /*disjoint=*/true));
    }

    return annotate(
        AtOp::create(builder, loc, getExprType(elementType, resultAxes), source, indices));
  }

  Value constant(TypedAttr value) {
    return annotate(ConstantOp::create(builder, loc, getExprType(value.getType(), {}), value));
  }

  Value index(const Axis &axis, Type elementType) {
    return annotate(IndexOp::create(builder, loc, getExprType(elementType, {axis.name}),
                                    materializeAxis(axis)));
  }

  template <typename OpTy> Value unary(Type elementType, Value input) {
    return annotate(
        OpTy::create(builder, loc, getExprType(elementType, collectOperandAxes({input})), input));
  }

  template <typename OpTy> Value binary(Type elementType, Value lhs, Value rhs) {
    return annotate(OpTy::create(
        builder, loc, getExprType(elementType, collectOperandAxes({lhs, rhs})), lhs, rhs));
  }

  Value cmpf(arith::CmpFPredicate predicate, Value lhs, Value rhs) {
    return annotate(CmpFOp::create(builder, loc,
                                   getExprType(builder.getI1Type(), collectOperandAxes({lhs, rhs})),
                                   predicate, lhs, rhs));
  }

  Value cmpi(arith::CmpIPredicate predicate, Value lhs, Value rhs) {
    return annotate(CmpIOp::create(builder, loc,
                                   getExprType(builder.getI1Type(), collectOperandAxes({lhs, rhs})),
                                   predicate, lhs, rhs));
  }

  Value select(Type elementType, Value condition, Value trueValue, Value falseValue) {
    return annotate(SelectOp::create(
        builder, loc,
        getExprType(elementType, collectOperandAxes({trueValue, falseValue, condition})), condition,
        trueValue, falseValue));
  }

  // Reduction removes named axes from expression support. The StableHLO importer validates the
  // identity separately, so the bodyless TA reduction does not carry an identity operand here.
  Value reduce(ReduceKind kind, Value input, ArrayRef<AxisName> reductionAxes, Type elementType) {
    DenseSet<StringRef> reduced(reductionAxes.begin(), reductionAxes.end());
    AxisNames resultAxes;
    for (Attribute attr : cast<ExprType>(input.getType()).getAxes().getAxes()) {
      StringRef axis = cast<AxisAttr>(attr).getName().getValue();
      if (!reduced.contains(axis))
        resultAxes.push_back(axis.str());
    }
    return annotate(ReduceOp::create(builder, loc, getExprType(elementType, resultAxes), kind,
                                     input, Value(), getAxesAttr(reductionAxes)));
  }

  // Relabel an already translated expression for a new consumer. Substitution changes logical
  // coordinates without materializing a transpose or copying tensor data. Identity mappings are
  // omitted to avoid creating no-op ta.subst operations.
  Value subst(Value input, const AxisNameMapVector &replacements) {
    auto inputType = cast<ExprType>(input.getType());
    AxisNames resultAxes, fromAxes, toAxes;
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
    return annotate(SubstOp::create(builder, loc,
                                    getExprType(inputType.getElementType(), resultAxes), input,
                                    getAxesAttr(fromAxes), getAxesAttr(toAxes)));
  }

  void yield(Value value) { YieldOp::create(builder, loc, value); }

  // Materialize an expression in its natural TA axis order, then adapt that tensor back to the
  // original StableHLO dimension order. A linalg.transpose restores axis order and a
  // linalg.broadcast restores dimensions omitted from expression support, so an island boundary
  // does not constrain TA's internal order and remains visible to Linalg producer fusion.
  FailureOr<Value> materializeResult(Value output, const TensorAxes &targetAxes,
                                     RankedTensorType targetType, Location transposeLoc) {
    auto exprType = cast<ExprType>(output.getType());
    SmallVector<int64_t> exprShape;
    SmallVector<int64_t> broadcastDims;
    DenseSet<int64_t> usedTargetDims;
    for (Attribute attr : exprType.getAxes().getAxes()) {
      StringRef name = cast<AxisAttr>(attr).getName().getValue();
      auto materialized = axes.find(name.str());
      if (materialized == axes.end())
        return emitError(output.getLoc(), "missing materialized extent for result axis");
      exprShape.push_back(materialized->second.extent);

      std::optional<int64_t> targetDim;
      for (auto [dim, axis] : llvm::enumerate(targetAxes))
        if (axis && axis->name == name) {
          if (targetDim)
            return emitError(output.getLoc(), "result axis maps to multiple tensor dimensions");
          targetDim = static_cast<int64_t>(dim);
        }
      if (!targetDim || !usedTargetDims.insert(*targetDim).second)
        return emitError(output.getLoc(), "expression axis is absent from result tensor axes");
      broadcastDims.push_back(*targetDim);
    }

    auto exprTensorType = RankedTensorType::get(exprShape, exprType.getElementType());

    // A splat StableHLO constant has no expression axes. Materialize it as an
    // arith tensor constant directly instead of wrapping it in an axis-free TA
    // scope, which would later become a rank-zero linalg.generic. When the
    // boundary type is static, expand the splat attribute directly to that
    // type instead of creating a rank-zero constant plus linalg.broadcast.
    if (auto constant = output.getDefiningOp<ConstantOp>();
        constant && exprType.getAxes().getAxes().empty()) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(scope);
      RankedTensorType constantType = targetType.hasStaticShape() ? targetType : exprTensorType;
      auto tensorValue = DenseElementsAttr::get(constantType, constant.getValue());
      Value adapted = arith::ConstantOp::create(builder, loc, tensorValue);
      if (constantType != targetType) {
        Value init = tensor::EmptyOp::create(builder, loc, targetType.getShape(),
                                             targetType.getElementType());
        SmallVector<int64_t> addedDimensions(targetType.getRank());
        std::iota(addedDimensions.begin(), addedDimensions.end(), 0);
        adapted = linalg::BroadcastOp::create(builder, loc, adapted, init, addedDimensions)
                      .getResult()[0];
      }
      scope.erase();
      return adapted;
    }

    scope.getResult().setType(exprTensorType);
    yield(output);
    relabelAxesForOutput(output);

    bool identityLayout = exprTensorType == targetType;
    for (auto [dim, targetDim] : llvm::enumerate(broadcastDims))
      identityLayout &= targetDim == static_cast<int64_t>(dim);
    if (identityLayout)
      return scope.getResult();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(scope);

    // linalg.transpose uses output-dimension -> input-dimension permutations.
    // Order source dimensions by the target dimensions to which they map.
    SmallVector<int64_t> permutation(exprTensorType.getRank());
    std::iota(permutation.begin(), permutation.end(), 0);
    llvm::sort(permutation,
               [&](int64_t lhs, int64_t rhs) { return broadcastDims[lhs] < broadcastDims[rhs]; });

    Value adapted = scope.getResult();
    bool needsTranspose = llvm::any_of(llvm::enumerate(permutation), [](auto entry) {
      return static_cast<int64_t>(entry.index()) != entry.value();
    });
    if (needsTranspose) {
      SmallVector<int64_t> transposedShape;
      for (int64_t inputDim : permutation)
        transposedShape.push_back(exprTensorType.getDimSize(inputDim));
      Value init = tensor::EmptyOp::create(builder, transposeLoc, transposedShape,
                                           exprTensorType.getElementType());
      adapted = linalg::TransposeOp::create(builder, transposeLoc, adapted, init, permutation)
                    .getResult()[0];
    }

    DenseSet<int64_t> mappedTargetDims(broadcastDims.begin(), broadcastDims.end());
    SmallVector<int64_t> addedDimensions;
    for (int64_t dim = 0; dim < targetType.getRank(); ++dim)
      if (!mappedTargetDims.contains(dim))
        addedDimensions.push_back(dim);

    if (!addedDimensions.empty()) {
      Value init =
          tensor::EmptyOp::create(builder, loc, targetType.getShape(), targetType.getElementType());
      adapted =
          linalg::BroadcastOp::create(builder, loc, adapted, init, addedDimensions).getResult()[0];
    }
    return adapted;
  }

private:
  // Discovery names are deliberately unstable and verbose. Once the yielded expression is known,
  // assign i* names to its axes in result order and j* names to every remaining internal axis. This
  // convention makes imported TA deterministic and keeps output dimensions visually distinct from
  // reduction-only coordinates.
  //
  // Block arguments cannot be reordered in place. Add arguments in normalized order, replace uses
  // of the old arguments, then erase the old prefix. Finally rewrite every AxesAttr and ExprType in
  // the scope so names on operations, types, and block arguments remain consistent.
  void relabelAxesForOutput(Value output) {
    auto outputType = cast<ExprType>(output.getType());
    AxisNameMapVector renames;
    unsigned nextOutput = 0, nextInternal = 0;
    for (Attribute attr : outputType.getAxes().getAxes()) {
      AxisName oldName = cast<AxisAttr>(attr).getName().getValue().str();
      if (axes.contains(oldName) && !renames.contains(oldName))
        renames.insert({oldName, "i" + std::to_string(nextOutput++)});
    }
    for (auto &entry : axes)
      if (!renames.contains(entry.first))
        renames.insert({entry.first, "j" + std::to_string(nextInternal++)});

    AxisNames scopeAxes;
    SmallVector<int64_t> extents;
    Block &body = scope.getBody().front();
    unsigned oldNumAxes = body.getNumArguments();
    for (auto &[oldName, newName] : renames) {
      MaterializedAxis &axis = axes[oldName];
      BlockArgument argument = body.addArgument(builder.getIndexType(), loc);
      axis.value.replaceAllUsesWith(argument);
      axis.value = argument;
      scopeAxes.push_back(newName);
      extents.push_back(axis.extent);
    }
    if (oldNumAxes)
      body.eraseArguments(0, oldNumAxes);
    scope.setAxesAttr(getAxesAttr(scopeAxes));
    scope.setStaticExtentsAttr(DenseI64ArrayAttr::get(context, extents));

    // Attribute and type rewriting is intentionally local to the new scope. StableHLO operations
    // outside it still use ordinary tensor types and never refer to these provisional names.
    auto renameAxes = [&](AxesAttr attr) {
      SmallVector<Attribute> result;
      for (Attribute axisAttr : attr.getAxes()) {
        StringRef oldName = cast<AxisAttr>(axisAttr).getName().getValue();
        auto it = renames.find(oldName.str());
        result.push_back(AxisAttr::get(context, it == renames.end() ? oldName : it->second));
      }
      return AxesAttr::get(context, ArrayAttr::get(context, result));
    };
    scope.getBody().walk([&](Operation *op) {
      for (NamedAttribute attr : llvm::to_vector(op->getAttrs()))
        if (auto axesAttr = dyn_cast<AxesAttr>(attr.getValue()))
          op->setAttr(attr.getName(), renameAxes(axesAttr));
      for (OpResult result : op->getResults())
        if (auto exprType = dyn_cast<ExprType>(result.getType()))
          result.setType(
              ExprType::get(context, exprType.getElementType(), renameAxes(exprType.getAxes())));
    });
  }

  template <typename OpTy> Value annotate(OpTy op) const {
    if (importGroup)
      op->setAttr("ta.import_group", IntegerAttr::get(IntegerType::get(context, 64), *importGroup));
    return op.getResult();
  }

  // All absent tensor dimensions can share one index constant. Insert it at scope entry so it
  // dominates every ta.at regardless of the source operation currently being emitted.
  Value indexZero() {
    if (zero)
      return zero;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&scope.getBody().front());
    zero = arith::ConstantIndexOp::create(builder, loc, 0);
    return zero;
  }

  // Materialize each logical axis exactly once as a scope block argument. MapVector preserves first
  // use order until relabelAxesForOutput performs the final output-first ordering.
  Value materializeAxis(const Axis &axis) {
    auto it = axes.find(axis.name);
    if (it != axes.end())
      return it->second.value;
    Value argument = scope.getBody().front().addArgument(builder.getIndexType(), loc);
    axes.try_emplace(axis.name, MaterializedAxis{argument, axis.extent});
    return argument;
  }

  struct MaterializedAxis {
    Value value;
    int64_t extent;
  };

  MLIRContext *context;
  Location loc;
  OpBuilder builder;
  Value zero;
  ScopeOp scope;
  std::optional<int64_t> importGroup;
  llvm::MapVector<AxisName, MaterializedAxis, llvm::StringMap<unsigned>> axes;
};

// Keep both the TA expression and its interpretation as tensor dimensions. The expression type
// only stores present axes, while this rank-preserving vector is needed to translate later
// broadcasts, reshapes, and consumers which request different logical names.
struct TensorValueInfo {
  Value expr;
  TensorAxes axes;
};

/// Replay supported StableHLO tensor dataflow as one scalar TA expression DAG.
///
/// Values already emitted are reused through valueMap. Function arguments and values defined
/// outside the current supported component are observed lazily with ta.at, while splat constants
/// become axis-free expressions.
class FunctionEmitter {
public:
  FunctionEmitter(ScopedTABuilder &ta, const DiscoveredAxisInfo &axisInfo,
                  const DenseSet<Operation *> &component, int64_t &nextImportGroup)
      : ta(ta), axisInfo(axisInfo), component(component), nextImportGroup(nextImportGroup) {}

  // Dispatch in source order so valueMap normally contains each producer before its consumers.
  // View-like operations only update tensor-axis metadata; arithmetic creates new TA operations.
  LogicalResult emit(Operation *op) {
    if (!component.contains(op))
      return success();
    ScopedTABuilder::ImportGroupGuard guard(ta, nextImportGroup++, op->getLoc());
    if (auto constant = dyn_cast<stablehlo::ConstantOp>(op))
      return emitConstant(constant);
    if (auto iota = dyn_cast<stablehlo::IotaOp>(op))
      return emitIota(iota);
    if (auto iota = dyn_cast<stablehlo::DynamicIotaOp>(op))
      return emitIota(iota);
    if (auto convert = dyn_cast<stablehlo::ConvertOp>(op)) {
      auto inputType = cast<RankedTensorType>(convert.getOperand().getType());
      auto resultType = cast<RankedTensorType>(convert.getResult().getType());
      if (inputType.getElementType() == resultType.getElementType())
        return emitViewLike(convert.getOperand(), convert, convert.getResult());
      return emitUnary<CastOp>(convert);
    }
    if (auto exponential = dyn_cast<stablehlo::ExpOp>(op))
      return emitUnary<ExpOp>(exponential);
    if (auto add = dyn_cast<stablehlo::AddOp>(op))
      return emitBinary<AddOp>(add);
    if (auto subtract = dyn_cast<stablehlo::SubtractOp>(op))
      return emitBinary<SubOp>(subtract);
    if (auto multiply = dyn_cast<stablehlo::MulOp>(op))
      return emitBinary<MulOp>(multiply);
    if (auto divide = dyn_cast<stablehlo::DivOp>(op))
      return emitBinary<DivOp>(divide);
    if (auto maximum = dyn_cast<stablehlo::MaxOp>(op))
      return emitBinary<MaximumOp>(maximum);
    if (auto minimum = dyn_cast<stablehlo::MinOp>(op))
      return emitBinary<MinimumOp>(minimum);
    if (auto andOp = dyn_cast<stablehlo::AndOp>(op))
      return emitBinary<AndOp>(andOp);
    if (auto compare = dyn_cast<stablehlo::CompareOp>(op))
      return emitCompare(compare);
    if (auto select = dyn_cast<stablehlo::SelectOp>(op))
      return emitSelect(select);
    if (auto transpose = dyn_cast<stablehlo::TransposeOp>(op))
      return emitViewLike(transpose.getOperand(), transpose, transpose.getResult());
    if (auto broadcast = dyn_cast<stablehlo::BroadcastInDimOp>(op))
      return emitBroadcast(broadcast);
    if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(op))
      return emitReshape(reshape);
    if (auto reduce = dyn_cast<stablehlo::ReduceOp>(op))
      return emitReduce(reduce);
    if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(op))
      return emitDot(dot);
    return success();
  }

  FailureOr<Value> translateRoot(Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return emitError(value.getLoc(), "ta importer expected a ranked tensor root value");
    auto axes = lookupAxes(value);
    if (failed(axes))
      return failure();
    return translateValue(value, **axes, type.getElementType());
  }

private:
  FailureOr<const TensorAxes *> lookupAxes(Value value) const {
    auto it = axisInfo.valueAxes.find(value);
    if (it == axisInfo.valueAxes.end()) {
      emitError(value.getLoc(), "missing discovered tensor axes");
      return failure();
    }
    return &it->second;
  }

  static FailureOr<TypedAttr> getSplatValue(Attribute value) {
    auto elements = dyn_cast<ElementsAttr>(value);
    if (!elements || !elements.isSplat())
      return failure();
    return dyn_cast<TypedAttr>(elements.getSplatValue<Attribute>());
  }

  LogicalResult emitConstant(stablehlo::ConstantOp op) {
    auto value = getSplatValue(op.getValue());
    if (failed(value))
      return op.emitOpError("only splat constants are supported");
    auto type = cast<RankedTensorType>(op.getResult().getType());
    valueMap[op.getResult()] =
        TensorValueInfo{ta.constant(*value), TensorAxes(type.getRank(), std::nullopt)};
    return success();
  }

  template <typename IotaOp> LogicalResult emitIota(IotaOp op) {
    auto resultAxes = lookupAxes(op.getResult());
    if (failed(resultAxes))
      return failure();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    uint64_t dimension = op.getIotaDimension();
    if (dimension >= static_cast<uint64_t>(resultType.getRank()) || !(**resultAxes)[dimension])
      return op.emitOpError("invalid iota dimension");

    Type elementType = resultType.getElementType();
    Value expr;
    if (elementType.isSignlessInteger()) {
      expr = ta.index(*(**resultAxes)[dimension], elementType);
    } else if (isa<FloatType>(elementType)) {
      Value index = ta.index(*(**resultAxes)[dimension], IntegerType::get(op.getContext(), 64));
      expr = ta.unary<CastOp>(elementType, index);
    } else {
      return op.emitOpError("only signless integer and floating-point iotas are supported");
    }
    valueMap[op.getResult()] = {expr, getPresentTensorAxes(expr, **resultAxes)};
    return success();
  }

  template <typename TAOp, typename StableOp> LogicalResult emitUnary(StableOp op) {
    auto inputAxes = lookupAxes(op.getOperand());
    auto resultAxes = lookupAxes(op.getResult());
    if (failed(inputAxes) || failed(resultAxes))
      return failure();
    auto inputType = cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto input = translateValue(op.getOperand(), **inputAxes, inputType.getElementType());
    if (failed(input))
      return failure();
    Value expr = ta.unary<TAOp>(resultType.getElementType(), *input);
    valueMap[op.getResult()] = {expr, getPresentTensorAxes(expr, **resultAxes)};
    return success();
  }

  template <typename TAOp, typename StableOp> LogicalResult emitBinary(StableOp op) {
    auto lhsAxes = lookupAxes(op.getLhs());
    auto rhsAxes = lookupAxes(op.getRhs());
    auto resultAxes = lookupAxes(op.getResult());
    if (failed(lhsAxes) || failed(rhsAxes) || failed(resultAxes))
      return failure();
    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Type elementType = resultType.getElementType();
    auto lhs = translateValue(op.getLhs(), **lhsAxes, lhsType.getElementType());
    auto rhs = translateValue(op.getRhs(), **rhsAxes, rhsType.getElementType());
    if (failed(lhs) || failed(rhs))
      return failure();
    Value expr = ta.binary<TAOp>(resultType.getElementType(), *lhs, *rhs);
    valueMap[op.getResult()] = {expr, getPresentTensorAxes(expr, **resultAxes)};
    return success();
  }

  LogicalResult emitCompare(stablehlo::CompareOp op) {
    auto lhsAxes = lookupAxes(op.getLhs());
    auto rhsAxes = lookupAxes(op.getRhs());
    auto resultAxes = lookupAxes(op.getResult());
    if (failed(lhsAxes) || failed(rhsAxes) || failed(resultAxes))
      return failure();

    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());
    Type elementType = lhsType.getElementType();
    if (rhsType.getElementType() != elementType)
      return op.emitOpError("comparison operands must have matching element types");
    auto lhs = translateValue(op.getLhs(), **lhsAxes, elementType);
    auto rhs = translateValue(op.getRhs(), **rhsAxes, elementType);
    if (failed(lhs) || failed(rhs))
      return failure();

    static const arith::CmpFPredicate cmpFPreds[] = {
        arith::CmpFPredicate::OEQ, arith::CmpFPredicate::UNE, arith::CmpFPredicate::OGE,
        arith::CmpFPredicate::OGT, arith::CmpFPredicate::OLE, arith::CmpFPredicate::OLT};
    static const arith::CmpIPredicate cmpISignedPreds[] = {
        arith::CmpIPredicate::eq,  arith::CmpIPredicate::ne,  arith::CmpIPredicate::sge,
        arith::CmpIPredicate::sgt, arith::CmpIPredicate::sle, arith::CmpIPredicate::slt};
    static const arith::CmpIPredicate cmpIUnsignedPreds[] = {
        arith::CmpIPredicate::eq,  arith::CmpIPredicate::ne,  arith::CmpIPredicate::uge,
        arith::CmpIPredicate::ugt, arith::CmpIPredicate::ule, arith::CmpIPredicate::ult};
    std::optional<stablehlo::ComparisonType> comparisonType = op.getCompareType();
    if (!comparisonType)
      return op.emitOpError("comparison type must be specified");
    auto cmpDirection = static_cast<unsigned>(op.getComparisonDirection());
    Value expr;
    if (isa<FloatType>(elementType)) {
      if (*comparisonType != stablehlo::ComparisonType::FLOAT)
        return op.emitOpError("only FLOAT floating-point comparisons are supported");
      arith::CmpFPredicate predicate = cmpFPreds[cmpDirection];
      expr = ta.cmpf(predicate, *lhs, *rhs);
    } else if (isa<IntegerType>(elementType)) {
      arith::CmpIPredicate predicate;
      if (*comparisonType == stablehlo::ComparisonType::SIGNED)
        predicate = cmpISignedPreds[cmpDirection];
      else if (*comparisonType == stablehlo::ComparisonType::UNSIGNED)
        predicate = cmpIUnsignedPreds[cmpDirection];
      else
        return op.emitOpError("integer comparison must be SIGNED or UNSIGNED");
      expr = ta.cmpi(predicate, *lhs, *rhs);
    } else {
      return op.emitOpError("only floating-point and integer comparisons are supported");
    }

    valueMap[op.getResult()] = {expr, getPresentTensorAxes(expr, **resultAxes)};
    return success();
  }

  LogicalResult emitSelect(stablehlo::SelectOp op) {
    auto predAxes = lookupAxes(op.getPred());
    auto trueAxes = lookupAxes(op.getOnTrue());
    auto falseAxes = lookupAxes(op.getOnFalse());
    auto resultAxes = lookupAxes(op.getResult());
    if (failed(predAxes) || failed(trueAxes) || failed(falseAxes) || failed(resultAxes))
      return failure();

    auto predType = cast<RankedTensorType>(op.getPred().getType());
    auto trueType = cast<RankedTensorType>(op.getOnTrue().getType());
    auto falseType = cast<RankedTensorType>(op.getOnFalse().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Type elementType = resultType.getElementType();
    if (!predType.getElementType().isInteger(1))
      return op.emitOpError("select predicate must have i1 element type");
    if (trueType.getElementType() != elementType || falseType.getElementType() != elementType ||
        !isa<FloatType, IntegerType>(elementType))
      return op.emitOpError("only matching floating-point or integer select values are supported");

    auto pred = translateValue(op.getPred(), **predAxes, predType.getElementType());
    auto trueValue = translateValue(op.getOnTrue(), **trueAxes, elementType);
    auto falseValue = translateValue(op.getOnFalse(), **falseAxes, elementType);
    if (failed(pred) || failed(trueValue) || failed(falseValue))
      return failure();
    Value expr = ta.select(elementType, *pred, *trueValue, *falseValue);
    valueMap[op.getResult()] = {expr, getPresentTensorAxes(expr, **resultAxes)};
    return success();
  }

  // Transpose has no scalar computation to emit. Discovery has already permuted the result's
  // tensor-dimension metadata, while the underlying expression retains the source indexing order.
  template <typename StableOp> LogicalResult emitViewLike(Value input, StableOp op, Value result) {
    auto inputAxes = lookupAxes(input);
    auto resultAxes = lookupAxes(result);
    if (failed(inputAxes) || failed(resultAxes))
      return failure();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto expr = translateValue(input, **inputAxes, inputType.getElementType());
    if (failed(expr))
      return failure();
    valueMap[result] = {*expr, getPresentTensorAxes(*expr, **resultAxes)};
    return success();
  }

  // Project result axes back through broadcast_dimensions to obtain source tensor coordinates.
  // Equal-size mapped dimensions retain the result axis. Expanded source dimensions remain null,
  // causing translateValue or ta.at to treat them as constant-zero coordinates.
  LogicalResult emitBroadcast(stablehlo::BroadcastInDimOp op) {
    auto resultAxes = lookupAxes(op.getResult());
    if (failed(resultAxes))
      return failure();
    TensorAxes projectedInputAxes(op.getBroadcastDimensions().size());
    auto inputType = cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    for (auto [inputDim, resultDim] : llvm::enumerate(op.getBroadcastDimensions()))
      if (inputType.getDimSize(inputDim) == resultType.getDimSize(resultDim))
        projectedInputAxes[inputDim] = (**resultAxes)[resultDim];
    auto expr = translateValue(op.getOperand(), projectedInputAxes, inputType.getElementType());
    if (failed(expr))
      return failure();
    valueMap[op.getResult()] = {*expr, getPresentTensorAxes(*expr, **resultAxes)};
    return success();
  }

  // Project an expanded result back to source axes so a previously translated source expression
  // can be reused.
  FailureOr<TensorAxes> projectExpandSourceAxes(Value source,
                                                ArrayRef<ReassociationIndices> reassociation,
                                                const TensorAxes &targetAxes) {
    auto sourceAxes = lookupAxes(source);
    if (failed(sourceAxes) || (*sourceAxes)->size() != reassociation.size())
      return failure();

    TensorAxes projected;
    projected.reserve((*sourceAxes)->size());
    for (auto [sourceDim, group] : llvm::enumerate(reassociation)) {
      const std::optional<Axis> &sourceAxis = (**sourceAxes)[sourceDim];
      if (!sourceAxis) {
        projected.push_back(std::nullopt);
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
      projected.push_back(std::move(selected));
    }
    return projected;
  }

  // Collapsing singleton dimensions is the inverse projection needed by existing StableHLO
  // payloads. A true product collapse would require delinearizing one TA axis into several source
  // axes and remains unsupported.
  FailureOr<TensorAxes> projectCollapsedSourceAxes(stablehlo::ReshapeOp op,
                                                   ArrayRef<ReassociationIndices> reassociation,
                                                   const TensorAxes &resultAxes) {
    auto inputType = cast<RankedTensorType>(op.getOperand().getType());
    TensorAxes projected(inputType.getRank(), std::nullopt);
    if (reassociation.empty())
      return projected;
    if (reassociation.size() != resultAxes.size())
      return failure();

    for (auto [resultDim, group] : llvm::enumerate(reassociation)) {
      const std::optional<Axis> &resultAxis = resultAxes[resultDim];
      if (!resultAxis)
        continue;
      std::optional<int64_t> selected;
      for (int64_t inputDim : group) {
        if (inputDim < 0 || inputDim >= inputType.getRank())
          return failure();
        if (inputType.getDimSize(inputDim) == resultAxis->extent) {
          if (selected)
            return op.emitOpError("ambiguous collapsed reshape axis");
          selected = inputDim;
        }
      }
      if (!selected)
        return op.emitOpError("product collapse of non-unit dimensions is not supported");
      projected[*selected] = resultAxis;
    }
    return projected;
  }

  LogicalResult emitReshape(stablehlo::ReshapeOp op) {
    auto resultAxes = lookupAxes(op.getResult());
    auto reassociation = inferReshapeReassociation(op);
    if (failed(resultAxes) || failed(reassociation))
      return failure();
    auto inputType = cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    FailureOr<Value> expr;
    if (inputType.getRank() < resultType.getRank()) {
      if (valueMap.contains(op.getOperand())) {
        auto sourceAxes = projectExpandSourceAxes(op.getOperand(), *reassociation, **resultAxes);
        if (failed(sourceAxes))
          return op.emitOpError("failed to project expanded source axes");
        expr = translateValue(op.getOperand(), *sourceAxes, inputType.getElementType());
      } else {
        expr = ta.atExpandedSource(op.getOperand(), *reassociation, **resultAxes,
                                   inputType.getElementType());
      }
    } else if (inputType.getRank() > resultType.getRank()) {
      auto sourceAxes = projectCollapsedSourceAxes(op, *reassociation, **resultAxes);
      if (failed(sourceAxes))
        return failure();
      expr = translateValue(op.getOperand(), *sourceAxes, inputType.getElementType());
    } else {
      expr = translateValue(op.getOperand(), **resultAxes, inputType.getElementType());
    }
    if (failed(expr))
      return failure();
    valueMap[op.getResult()] = {*expr, getPresentTensorAxes(*expr, **resultAxes)};
    return success();
  }

  // Match the canonical two-argument StableHLO reduction region. Requiring the returned combiner
  // to consume both block arguments directly excludes maps, nested expressions, and reducers whose
  // semantics cannot be represented by one bodyless ta.reduce.
  FailureOr<ReduceKind> matchReductionKind(stablehlo::ReduceOp op) {
    if (op.getInputs().size() != 1 || op.getInitValues().size() != 1 || op.getNumResults() != 1)
      return op.emitOpError("only single-input reductions are supported");
    Block &body = op.getBody().front();
    auto terminator = dyn_cast<stablehlo::ReturnOp>(body.getTerminator());
    if (body.getNumArguments() != 2 || !terminator || terminator.getNumOperands() != 1)
      return op.emitOpError("expected a binary reduction body");
    Operation *combiner = terminator.getOperand(0).getDefiningOp();
    if (!combiner || combiner->getNumOperands() != 2 ||
        !llvm::is_contained(combiner->getOperands(), body.getArgument(0)) ||
        !llvm::is_contained(combiner->getOperands(), body.getArgument(1)))
      return op.emitOpError("expected a direct binary reduction combiner");
    if (isa<stablehlo::AddOp>(combiner))
      return ReduceKind::Add;
    if (isa<stablehlo::MaxOp>(combiner))
      return ReduceKind::Max;
    return op.emitOpError("only add and maximum reductions are supported");
  }

  // TA reductions encode mathematical reduction kind but not arbitrary StableHLO initialization.
  // Accept only the identities whose omission is semantics-preserving: zero for add and negative
  // infinity (or the minimum signed integer) for maximum.
  LogicalResult verifyReductionIdentity(stablehlo::ReduceOp op, ReduceKind kind) {
    Value init = op.getInitValues().front();
    FailureOr<TypedAttr> value = failure();
    if (auto constant = init.getDefiningOp<stablehlo::ConstantOp>())
      value = getSplatValue(constant.getValue());
    else if (auto constant = init.getDefiningOp<arith::ConstantOp>())
      value = getSplatValue(constant.getValue());
    if (failed(value))
      return op.emitOpError("reduction init must be a splat identity constant");

    bool isIdentity = false;
    if (auto floatValue = dyn_cast<FloatAttr>(*value)) {
      if (kind == ReduceKind::Add)
        isIdentity = floatValue.getValue().isZero();
      else
        isIdentity = floatValue.getValue().isInfinity() && floatValue.getValue().isNegative();
    } else if (auto integerValue = dyn_cast<IntegerAttr>(*value)) {
      if (kind == ReduceKind::Add)
        isIdentity = integerValue.getValue().isZero();
      else
        isIdentity = integerValue.getValue().isMinSignedValue();
    }
    if (!isIdentity)
      return op.emitOpError("reduction init is not the canonical identity");
    return success();
  }

  LogicalResult emitReduce(stablehlo::ReduceOp op) {
    auto kind = matchReductionKind(op);
    if (failed(kind) || failed(verifyReductionIdentity(op, *kind)))
      return failure();
    Value inputValue = op.getInputs().front();
    Value resultValue = op.getResult(0);
    auto inputAxes = lookupAxes(inputValue);
    auto resultAxes = lookupAxes(resultValue);
    if (failed(inputAxes) || failed(resultAxes))
      return failure();
    auto inputType = cast<RankedTensorType>(inputValue.getType());
    auto resultType = cast<RankedTensorType>(resultValue.getType());
    if (!isa<FloatType>(resultType.getElementType()))
      return op.emitOpError("only floating-point reductions are supported");
    auto input = translateValue(inputValue, **inputAxes, inputType.getElementType());
    if (failed(input))
      return failure();
    AxisNames reductionAxes;
    for (int64_t dim : op.getDimensions()) {
      if (dim < 0 || dim >= static_cast<int64_t>((*inputAxes)->size()))
        return op.emitOpError("invalid reduction dimension");
      reductionAxes.push_back((**inputAxes)[dim]->name);
    }
    Value expr = ta.reduce(*kind, *input, reductionAxes, resultType.getElementType());
    valueMap[resultValue] = {expr, getPresentTensorAxes(expr, **resultAxes)};
    return success();
  }

  // Axis discovery makes each contracting pair share one name. A dot is therefore ordinary
  // elementwise multiplication over the ordered union of operand axes followed by an add reduction
  // over the lhs contracting axes. Batch and free axes survive automatically.
  LogicalResult emitDot(stablehlo::DotGeneralOp op) {
    auto lhsAxes = lookupAxes(op.getLhs());
    auto rhsAxes = lookupAxes(op.getRhs());
    auto resultAxes = lookupAxes(op.getResult());
    if (failed(lhsAxes) || failed(rhsAxes) || failed(resultAxes))
      return failure();
    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    if (!isa<FloatType>(lhsType.getElementType()) ||
        lhsType.getElementType() != rhsType.getElementType() ||
        !isa<FloatType>(resultType.getElementType()))
      return op.emitOpError("only floating-point dots with matching input types are supported");
    auto lhs = translateValue(op.getLhs(), **lhsAxes, lhsType.getElementType());
    auto rhs = translateValue(op.getRhs(), **rhsAxes, rhsType.getElementType());
    if (failed(lhs) || failed(rhs))
      return failure();
    if (lhsType.getElementType() != resultType.getElementType()) {
      lhs = ta.unary<CastOp>(resultType.getElementType(), *lhs);
      rhs = ta.unary<CastOp>(resultType.getElementType(), *rhs);
    }
    Value product = ta.binary<MulOp>(resultType.getElementType(), *lhs, *rhs);
    AxisNames reductionAxes;
    for (int64_t dim : op.getDotDimensionNumbers().getLhsContractingDimensions())
      reductionAxes.push_back((**lhsAxes)[dim]->name);
    Value expr = ta.reduce(ReduceKind::Add, product, reductionAxes, resultType.getElementType());
    valueMap[op.getResult()] = {expr, getPresentTensorAxes(expr, **resultAxes)};
    return success();
  }

  // Translate one tensor value as observed under targetAxes.
  //
  // For an emitted value, compare tensor dimensions positionally and build a simultaneous axis
  // substitution. Consumers may legitimately request different names, for example when the same
  // expression participates in different broadcasts. Dropping an axis is valid only for extent
  // one; reducing a one-element domain is an identity and removes it from TA expression support.
  //
  // Values without a valueMap entry are component leaves. Splat constants are lifted directly;
  // function arguments, unsupported results, and earlier scope results become ta.at observations.
  // A supported definition in the component without a map entry indicates an emitter bug.
  FailureOr<Value> translateValue(Value value, const TensorAxes &targetAxes, Type elementType) {
    auto it = valueMap.find(value);
    if (it != valueMap.end()) {
      if (it->second.axes.size() != targetAxes.size())
        return emitError(value.getLoc(), "cannot relabel tensor with different rank");
      AxisNameMapVector replacements;
      AxisNames erasedUnitAxes;
      for (auto [fromAxis, toAxis] : llvm::zip_equal(it->second.axes, targetAxes)) {
        if (!fromAxis)
          continue;
        if (!toAxis) {
          if (fromAxis->extent != 1)
            return emitError(value.getLoc(), "cannot erase non-unit tensor axis during relabel");
          erasedUnitAxes.push_back(fromAxis->name);
          continue;
        }
        auto [replacement, inserted] = replacements.try_emplace(fromAxis->name, toAxis->name);
        if (!inserted && replacement->second != toAxis->name)
          return emitError(value.getLoc(), "cannot relabel tensor axis to multiple targets");
      }
      Value expr = ta.subst(it->second.expr, replacements);
      // A reduction over a one-element axis is an identity and is the TA way
      // to forget a singleton dimension already materialized by a producer.
      if (!erasedUnitAxes.empty())
        expr = ta.reduce(ReduceKind::Add, expr, erasedUnitAxes,
                         cast<ExprType>(expr.getType()).getElementType());
      return expr;
    }

    if (auto stableConstant = value.getDefiningOp<stablehlo::ConstantOp>()) {
      auto splat = getSplatValue(stableConstant.getValue());
      if (succeeded(splat))
        return ta.constant(*splat);
    }
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      auto splat = getSplatValue(constant.getValue());
      if (succeeded(splat))
        return ta.constant(*splat);
    }
    if (Operation *def = value.getDefiningOp(); def && component.contains(def))
      return def->emitOpError("supported operation was not emitted into its TA scope: ")
             << def->getName();
    return ta.at(value, targetAxes, elementType);
  }

  // Convert full discovery metadata into the rank-preserving metadata for one emitted expression.
  // Dimensions whose axis is absent from the ExprType become null but retain their tensor position.
  // This distinction is essential when a later broadcast maps tensor dimensions by number.
  TensorAxes getPresentTensorAxes(Value expr, const TensorAxes &fullAxes) const {
    DenseSet<StringRef> present;
    for (Attribute attr : cast<ExprType>(expr.getType()).getAxes().getAxes())
      present.insert(cast<AxisAttr>(attr).getName().getValue());
    TensorAxes result;
    for (const std::optional<Axis> &axis : fullAxes)
      result.push_back(axis && present.contains(axis->name) ? axis : std::optional<Axis>{});
    return result;
  }

  ScopedTABuilder &ta;
  const DiscoveredAxisInfo &axisInfo;
  const DenseSet<Operation *> &component;
  int64_t &nextImportGroup;
  DenseMap<Value, TensorValueInfo> valueMap;
};

// Collect the maximal supported backward slice for one tensor root. Unsupported definitions and
// values materialized by an earlier scope are leaves which will be observed with ta.at.
static void collectSupportedComponent(Value value, Block *block, DenseSet<Operation *> &component) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != block || !isSupportedStableHLOOp(def) ||
      !component.insert(def).second)
    return;
  // A statically shaped dynamic_iota is determined by its result type. Its shape operand is
  // control metadata, not tensor dataflow, and exporter-generated shape arithmetic may use ops
  // outside the TA expression subset.
  if (isa<stablehlo::DynamicIotaOp>(def))
    return;
  for (Value operand : def->getOperands())
    if (isa<RankedTensorType>(operand.getType()))
      collectSupportedComponent(operand, block, component);
}

static LogicalResult importFunctionAsTA(func::FuncOp func) {
  AxisDiscovery discovery;
  auto axisInfoR = discovery.run(func);
  if (failed(axisInfoR))
    return failure();
  DiscoveredAxisInfo &axisInfo = *axisInfoR;

  // Process roots in producer order. If an early root also feeds supported operations, replacing
  // all uses cuts that edge and later scopes observe the already materialized tensor instead of
  // cloning the producer computation.
  int64_t nextImportGroup = 0;
  for (Value root : axisInfo.roots) {
    Operation *rootDef = root.getDefiningOp();
    auto resultType = dyn_cast<RankedTensorType>(root.getType());
    if (!rootDef || !resultType)
      return failure();

    DenseSet<Operation *> component;
    collectSupportedComponent(root, rootDef->getBlock(), component);
    if (component.empty())
      return success();

    // View-like transposes are represented as axis metadata in TA and are
    // materialized only at the island boundary. Preserve both the transpose
    // provenance and the root operation that caused materialization.
    SmallVector<Location> materializationLocs;
    for (Operation &op : rootDef->getBlock()->without_terminator())
      if (component.contains(&op) && isa<stablehlo::TransposeOp>(op))
        materializationLocs.push_back(op.getLoc());
    materializationLocs.push_back(rootDef->getLoc());
    Location materializationLoc = FusedLoc::get(func.getContext(), materializationLocs);

    ScopedTABuilder ta(rootDef, rootDef->getLoc(), resultType);
    FunctionEmitter emitter(ta, axisInfo, component, nextImportGroup);
    for (Operation &op : rootDef->getBlock()->without_terminator())
      if (failed(emitter.emit(&op)))
        return failure();

    FailureOr<Value> expr = emitter.translateRoot(root);
    auto axes = axisInfo.valueAxes.find(root);
    if (failed(expr) || axes == axisInfo.valueAxes.end())
      return failure();
    FailureOr<Value> materialized =
        ta.materializeResult(*expr, axes->second, resultType, materializationLoc);
    if (failed(materialized))
      return failure();

    // Downstream components see the adapted tensor with exactly the original root's dimension axes.
    // Recording the alias keeps the one global discovery result valid as scopes are introduced.
    axisInfo.valueAxes[*materialized] = axes->second;
    root.replaceAllUsesWith(*materialized);

    SmallVector<Operation *> componentOps;
    for (Operation &op : rootDef->getBlock()->without_terminator())
      if (component.contains(&op))
        componentOps.push_back(&op);
    for (Operation *op : llvm::reverse(componentOps))
      if (op->use_empty())
        op->erase();
  }

  // Keep product collapses outside TA: their source expression has one axis per factor, while TA
  // substitution cannot delinearize the collapsed result axis back into those factors.
  SmallVector<stablehlo::ReshapeOp> deferredCollapses;
  func.walk([&](stablehlo::ReshapeOp reshape) {
    if (isDeferredProductCollapse(reshape))
      deferredCollapses.push_back(reshape);
  });
  for (stablehlo::ReshapeOp reshape : deferredCollapses) {
    auto reassociation = inferReshapeReassociation(reshape);
    if (failed(reassociation))
      return failure();
    OpBuilder builder(reshape);
    auto resultType = cast<RankedTensorType>(reshape.getResult().getType());
    Value collapse = tensor::CollapseShapeOp::create(builder, reshape.getLoc(), resultType,
                                                     reshape.getOperand(), *reassociation);
    reshape.replaceAllUsesWith(collapse);
    reshape.erase();
  }
  return success();
}

struct ImportStableHLOToTAPass : public impl::StableHLOToTAPassBase<ImportStableHLOToTAPass> {
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
    if (!resultType.hasStaticShape()) {
      func.emitOpError("ta importer currently expects static result shapes");
      return signalPassFailure();
    }
    if (failed(importFunctionAsTA(func)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace ta

#undef DEBUG_TYPE
