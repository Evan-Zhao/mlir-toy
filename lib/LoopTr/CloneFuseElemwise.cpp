#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>

#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

namespace mlir::transform {

template <typename T>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SmallVectorImpl<T> &vec) {
  os << "[";
  bool first = true;
  for (const T &item : vec) {
    if (!first)
      os << ", ";
    os << item;
    first = false;
  }
  return os << "]";
}

namespace {

#define BAIL(message) return emitSilenceableFailure(transform, message);

using linalg::GenericOp;
using linalg::MapOp;
using scf::ForallOp;
using scf::ForOp;

using FoldResultsT = SmallVector<OpFoldResult>;

struct LoopFusionInputs {
  SmallVector<unsigned> inLoopOperandNumbers{};
  SmallVector<Value> inLoopOperands{};
  SmallVector<FoldResultsT> offsets{}, sizes{};

  auto getInLoopOperands() const { return llvm::zip_equal(inLoopOperandNumbers, inLoopOperands); }
};

/// Lift cloneValueDefChainAtInsertionPoint from Value to [OpFoldResult].
LogicalResult remapFoldResultsInplace(RewriterBase &rewriter, IRMapping &mapping,
                                      FoldResultsT &xs) {
  for (auto &value : xs) {
    if (auto attr = dyn_cast<Attribute>(value))
      continue;
    FailureOr<Value> remapped =
        cloneValueDefChainAtInsertionPoint(rewriter, cast<Value>(value), mapping);
    if (failed(remapped))
      return failure();
    value = *remapped;
  }
  return success();
}

struct OpTileDomainInfo {
  /// This struct holds the domain information of multiple tiled operations.
  /// All the vectors here have the same length.
  /// `operations` can contain duplicate operations, by design, so that `operations[i]`
  /// corresponds to the same operation as `offsets[i]` and `sizes[i]`.
  SmallVector<TilingInterface> operations{};
  SmallVector<OpFoldResult> offsets{}, sizes{};

  FailureOr<std::pair<FoldResultsT, FoldResultsT>>
  calculateAndAppendDomain(RewriterBase &rewriter, TilingInterface opInterface,
                           const LoopFusionInputs &inputs) {
    FoldResultsT offsets_, sizes_;
    llvm::errs() << "calculateAndAppendDomain: opInterface = " << opInterface
                 << " inputs.offsets=" << inputs.offsets << " inputs.sizes=" << inputs.sizes
                 << "\n";
    auto result = opInterface.getIterationDomainTileFromOperandTiles(
        rewriter, inputs.inLoopOperandNumbers, inputs.offsets, inputs.sizes, offsets_, sizes_);
    if (failed(result))
      return failure();
    operations.append(offsets_.size(), opInterface);
    offsets.append(offsets_.begin(), offsets_.end());
    sizes.append(sizes_.begin(), sizes_.end());
    return std::make_pair(offsets_, sizes_);
  }

  FailureOr<std::pair<FoldResultsT, FoldResultsT>>
  getResultDesc(RewriterBase &rewriter, size_t resultIdx, IRMapping *optMapping) {
    FoldResultsT offsets_ = offsets, sizes_ = sizes;
    if (optMapping) {
      if (failed(remapFoldResultsInplace(rewriter, *optMapping, offsets_)) ||
          failed(remapFoldResultsInplace(rewriter, *optMapping, sizes_)))
        return failure();
    }
    FoldResultsT resultOffsets, resultSizes;
    if (failed(operations[resultIdx].getResultTilePosition(rewriter, resultIdx, offsets_, sizes_,
                                                           resultOffsets, resultSizes)))
      return failure();
    return std::make_pair(resultOffsets, resultSizes);
  }
};

FoldResultsT getTileSizes(RewriterBase &rewriter, RankedTensorType type, Value tensorTile) {
  FoldResultsT sizes;
  sizes.reserve(type.getRank());
  for (auto [dim, size] : enumerate(type.getShape())) {
    if (ShapedType::isDynamic(size)) {
      auto dimOp = tensor::DimOp::create(rewriter, tensorTile.getLoc(), tensorTile,
                                         static_cast<unsigned>(dim));
      sizes.push_back(dimOp.getResult());
    } else {
      sizes.push_back(rewriter.getIndexAttr(size));
    }
  }
  return sizes;
};

static FailureOr<std::pair<FoldResultsT, FoldResultsT>>
getTileOffsetsAndSizes(RewriterBase &rewriter, Operation *mediator, Value tensorTile) {
  if (!mediator) {
    auto tileType = cast<RankedTensorType>(tensorTile.getType());
    auto offsets = SmallVector<OpFoldResult>(tileType.getRank(), rewriter.getIndexAttr(0));
    auto sizes = getTileSizes(rewriter, tileType, tensorTile);
    return {{offsets, sizes}};
  } else if (auto loopInsert = dyn_cast<tensor::InsertSliceOp>(mediator)) {
    return {{loopInsert.getMixedOffsets(), loopInsert.getMixedSizes()}};
  } else if (auto parallelInsert = dyn_cast<tensor::ParallelInsertSliceOp>(mediator)) {
    return {{parallelInsert.getMixedOffsets(), parallelInsert.getMixedSizes()}};
  } else {
    mediator->emitError("unsupported mediator operation");
    return failure();
  }
}

template <typename LoopTy> struct ForForallSharedInterface {};

template <> struct ForForallSharedInterface<ForOp> {
  using InsertionOp = tensor::InsertSliceOp;
  static ForOp createWithExtraArgs(RewriterBase &rewriter, ForOp loop,
                                   ArrayRef<Value> newInitArgs) {
    return ForOp::create(rewriter, loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
                         loop.getStep(), newInitArgs);
  }
  static void addIVToMapping(ForOp oldLoop, IRMapping &mapping, ForOp newLoop) {
    mapping.map(oldLoop.getInductionVar(), newLoop.getInductionVar());
  }
  static void pointRewriterToOp(RewriterBase &rewriter, ForOp loop) {
    rewriter.setInsertionPointToEnd(loop.getBody());
  }
  static void pointRewriterToInsert(RewriterBase &rewriter, ForOp loop) {
    rewriter.setInsertionPointToEnd(loop.getBody());
  }
  static void updateYieldOp(RewriterBase &rewriter, ForOp oldLoop, const IRMapping &mapping,
                            ForOp newLoop, ArrayRef<InsertionOp> newInsertOps) {
    SmallVector<Value> newYieldVals;
    auto oldYield = cast<scf::YieldOp>(oldLoop.getBody()->getTerminator());
    for (Value operand : oldYield.getOperands())
      newYieldVals.push_back(mapping.lookupOrDefault(operand));
    for (auto insertOp : newInsertOps)
      newYieldVals.push_back(insertOp.getResult());
    scf::YieldOp::create(rewriter, oldLoop.getLoc(), newYieldVals);
  }
};

template <> struct ForForallSharedInterface<ForallOp> {
  using InsertionOp = tensor::ParallelInsertSliceOp;
  static ForallOp createWithExtraArgs(RewriterBase &rewriter, ForallOp loop,
                                      ArrayRef<Value> newInitArgs) {
    return ForallOp::create(rewriter, loop.getLoc(), loop.getMixedLowerBound(),
                            loop.getMixedUpperBound(), loop.getMixedStep(), newInitArgs,
                            loop.getMapping());
  }
  static void addIVToMapping(ForallOp oldLoop, IRMapping &mapping, ForallOp newLoop) {
    for (auto [oldIV, newIV] :
         llvm::zip_equal(oldLoop.getInductionVars(), newLoop.getInductionVars())) {
      mapping.map(oldIV, newIV);
    }
  }
  static void pointRewriterToOp(RewriterBase &rewriter, ForallOp loop) {
    rewriter.setInsertionPoint(loop.getBody()->getTerminator());
  }
  static void pointRewriterToInsert(RewriterBase &rewriter, ForallOp loop) {
    pointRewriterToForallParallel(rewriter, loop);
  }
  static void updateYieldOp(RewriterBase &rewriter, ForallOp oldLoop, const IRMapping &mapping,
                            ForallOp newLoop, ArrayRef<InsertionOp> loopInsertOps) {}
};

/// Clones an scf.for or scf.forall loop to carry new results `newResults`, then use the new loop to
/// replace the old one.
/// Requires the producing operation of `newResults` to already be in the loop body.
template <typename LoopTy>
FailureOr<SmallVector<OpResult>>
updateLoopWithNewCarriedResults(RewriterBase &rewriter, LoopTy &loop, ArrayRef<OpResult> newResults,
                                OpTileDomainInfo iterSpace, ArrayRef<FoldResultsT> carrySizes) {
  // Create the new loop after the existing one.
  using LoopI = ForForallSharedInterface<LoopTy>;
  rewriter.setInsertionPointAfter(loop);
  // Create a new init value for each result.
  SmallVector<Value> newInitArgs = loop.getInits();
  for (auto [result, carrySize] : zip_equal(newResults, carrySizes)) {
    auto elemType = cast<RankedTensorType>(result.getType()).getElementType();
    auto emptyOp = tensor::EmptyOp::create(rewriter, result.getLoc(), carrySize, elemType);
    newInitArgs.push_back(emptyOp.getResult());
  }
  // Create the loop.
  auto newLoop = LoopI::createWithExtraArgs(rewriter, loop, newInitArgs);
  auto loopArgs = loop.getRegionIterArgs();
  auto newLoopRegionArgs = newLoop.getRegionIterArgs();
  auto newRegionOldArgs = newLoopRegionArgs.take_front(loopArgs.size()),
       newRegionNewArgs = newLoopRegionArgs.drop_front(loopArgs.size());
  // Clone the loop body while tracking a value map.
  IRMapping mapping;
  LoopI::addIVToMapping(loop, mapping, newLoop);
  for (auto [oldArg, newArg] : zip_equal(loopArgs, newRegionOldArgs)) {
    mapping.map(oldArg, newArg);
  }
  LoopI::pointRewriterToOp(rewriter, newLoop);
  for (Operation &op : loop.getBody()->without_terminator())
    rewriter.clone(op, mapping);

  // Create a bunch of insert_slice ops, each inserting the tile result into a panel.
  // scf::ForOp has a YieldOp, so it will use `loopInsertOps`.
  // scf::ForallOp does not, but it expects insertion ops to be in a special block.
  SmallVector<typename LoopI::InsertionOp> loopInsertOps;
  for (auto [index, result] : llvm::enumerate(newResults)) {
    auto result_ = mapping.lookupOrDefault(result);
    auto resultDesc = iterSpace.getResultDesc(rewriter, index, &mapping);
    if (failed(resultDesc))
      return failure();
    auto [offsets, sizes] = *resultDesc;
    auto unitStrides = getUnitStrides(rewriter, offsets.size());
    // We want the "insert" region (which is a separate block in the case of scf.forall) to strictly
    // only contain insertion ops.
    LoopI::pointRewriterToInsert(rewriter, newLoop);
    auto insertOp = LoopI::InsertionOp::create(
        rewriter, loop.getLoc(), result_, newRegionNewArgs[index], offsets, sizes, unitStrides);
    LoopI::pointRewriterToOp(rewriter, newLoop);
    loopInsertOps.push_back(insertOp);
  }
  LoopI::updateYieldOp(rewriter, loop, mapping, newLoop, loopInsertOps);

  // Replace all uses of the old loop results with the new loop results.
  auto oldLoopResults = loop.getResults(), newLoopResults = newLoop.getResults();
  auto newLoopOldResults = newLoopResults.take_front(oldLoopResults.size()),
       newLoopNewResults = newLoopResults.take_back(oldLoopResults.size());
  for (auto [oldResult, newResult] : llvm::zip_equal(oldLoopResults, newLoopOldResults)) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }
  rewriter.eraseOp(loop);
  loop = newLoop;
  return {newLoopNewResults};
}

} // namespace

void LoopRUCloneFuseElemwise::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getElemwiseChainOpsMutable(), effects);
  onlyReadsHandle(getOuterLoopMutable(), effects);
  onlyReadsHandle(getInnerLoopMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure LoopRUCloneFuseElemwise::apply(transform::TransformRewriter &rewriter,
                                                           TransformResults &transformResults,
                                                           TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseChainOps, "elementwise", elemwiseOps)
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getOuterLoop, "outer loop", outerLoop, ForallOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getInnerLoop, "inner loop", innerLoop, ForOp);

  // Use utils to track the results of the outer loop to operations within the inner loop.
  auto loopValueTrackingMapR = getChainedLoopResultMap({outerLoop, innerLoop});
  if (failed(loopValueTrackingMapR))
    BAIL("failed to get track loop results to loop body");
  const auto &loopValueTrackingMap = *loopValueTrackingMapR;

  // Insert before the inner loop's terminator (YieldOp).
  rewriter.setInsertionPoint(innerLoop.getBody()->getTerminator());

  // Track tile size info for fusion under both the inner and outer loops.
  // innerLoopFusion -- tile per inner loop iteration, outerLoopFusion -- tile per outer loop
  // iteration, globalCarrySizes -- global (full-panel) sizes.
  OpTileDomainInfo innerCarryInfo, outerCarryInfo;
  SmallVector<FoldResultsT> globalCarrySizes;
  // Track the output of fused+tiled ops.
  SmallVector<OpResult> tileResults;
  IRMapping valueMap;
  for (Operation *&elemwiseOp : elemwiseOps) {
#define BAIL_AND_POINT(message)                                                                    \
  {                                                                                                \
    elemwiseOp->emitError() << "failed on this elementwise op";                                    \
    return emitSilenceableFailure(transform, message);                                             \
  }
    // Check if the op is an elemwise op.
    if (failed(isSingleOutputElemwiseLinalgOp(elemwiseOp)))
      BAIL_AND_POINT(
          "expected every op to be an elementwise linalg.map or linalg.generic with one result");
    // Safe to assume TilingInterface since we know it's an elementwise linalg op.
    auto elemwiseOpI = cast<TilingInterface>(elemwiseOp);

    // Track operands produced by tensor.empty().
    SmallVector<OpResult> emptyOperands;
    LoopFusionInputs innerLoopFusion, outerLoopFusion;
    // Iterate over the operands of the elemwise op, so that we know the sizes of tiled inputs. This
    // prepares for fusion. For each operand, if it is a result of `outerLoop` (call this an
    // "in-loop" operand), find the corresponding in-loop producer op.
    for (auto [index, arg] : llvm::enumerate(elemwiseOp->getOperands())) {
      auto argOpResult = dyn_cast<OpResult>(arg);
      // Seems like a function argument, etc. -- something that is always available.
      if (!argOpResult)
        continue;
      auto it = loopValueTrackingMap.find(argOpResult);
      auto updateFusionInputs = [&](auto &fusionInputs, const auto &tracked,
                                    StringRef loopName) -> LogicalResult {
        auto offsetSizeF = getTileOffsetsAndSizes(rewriter, tracked.mediator, tracked.inLoopResult);
        if (failed(offsetSizeF)) {
          elemwiseOp->emitError() << "failed to get tile offset and size for operand " << index
                                  << " under the " << loopName << " loop";
          return failure();
        }
        fusionInputs.offsets.push_back(offsetSizeF->first);
        fusionInputs.sizes.push_back(offsetSizeF->second);
        fusionInputs.inLoopOperandNumbers.push_back(index);
        fusionInputs.inLoopOperands.push_back(tracked.inLoopResult);
        return success();
      };
      if (it != loopValueTrackingMap.end()) {
        // This operand is a result of the outer loop.
        if (failed(updateFusionInputs(innerLoopFusion, it->second[0], "inner")))
          BAIL_AND_POINT("failed to build inner fused operand metadata");
        if (failed(updateFusionInputs(outerLoopFusion, it->second[1], "outer")))
          BAIL_AND_POINT("failed to build outer fused operand metadata");
        continue;
      }
      // Operand is not a result of the outer loop. We allow two cases: the operand is produced
      // before the loop (which we can safely access within the loop);
      // or the operand is produced by tensor.empty(), for which we'll make a new one in the loop.
      Operation *defOp = argOpResult.getDefiningOp();
      if (defOp->isBeforeInBlock(outerLoop))
        continue;
      if (isa<tensor::EmptyOp>(defOp)) {
        emptyOperands.push_back(argOpResult);
        continue;
      }
      elemwiseOp->emitError() << "operand " << index << " is not loop-tracked, "
                              << "defined before outer loop, or produced by tensor.empty()";
      defOp->emitRemark() << "offending defining op";
      BAIL("found non-loop operand defined after outer loop");
    }

    // Run the actual tiling operation. This new op is inserted at the end of the inner loop (but
    // before the YieldOp). Information on the output result of this op is appended in
    // `innerCarryInfo`.
    auto offsetsAndSizesR =
        innerCarryInfo.calculateAndAppendDomain(rewriter, elemwiseOpI, innerLoopFusion);
    if (failed(offsetsAndSizesR))
      BAIL_AND_POINT("failed to tile elemwise op under inner loop");
    auto &[offsets, sizes] = *offsetsAndSizesR;
    auto tilingResultR = elemwiseOpI.getTiledImplementation(rewriter, offsets, sizes);
    if (failed(tilingResultR) || tilingResultR->tiledOps.size() != 1)
      BAIL_AND_POINT("failed to tile elemwise op under inner loop");
    auto innerFusedOp = cast<TilingInterface>(tilingResultR->tiledOps[0]);
    // There are still two things wrong with this new op: 1. `getTiledImplementation`
    // always generate an extract_slice op, but we don't need that for in-loop operands.
    for (auto [index, operand] : innerLoopFusion.getInLoopOperands()) {
      innerFusedOp->setOperand(index, operand);
      rewriter.eraseOp(tilingResultR->generatedSlices[index]);
    }
    // ... 2. For each of `emptyOperandNumbers`, there is a tensor.empty op that we need to move
    // *before* the outer loop, and thread through the loops (as carried values).
    for (auto emptyOperand : emptyOperands) {
      rewriter.moveOpBefore(emptyOperand.getDefiningOp(), outerLoop);
    }

    // Calculate the iteration domain under the outer loop (append to `outerCarryInfo`).
    if (failed(outerCarryInfo.calculateAndAppendDomain(rewriter, elemwiseOpI, outerLoopFusion)))
      BAIL_AND_POINT("failed to calculate iteration domain under outer loop");
    // Add the results of the fused op to `tileResults`; add global panel sizes to
    // `globalCarrySizes`; map the results of the old elemwise op to the local tiled op, so later
    // elemwise ops can reposition.
    for (auto [global, local] :
         llvm::zip_equal(elemwiseOp->getResults(), innerFusedOp->getResults())) {
      tileResults.push_back(local);
      auto resultType = cast<RankedTensorType>(global.getType());
      globalCarrySizes.push_back(getTileSizes(rewriter, resultType, global));
      valueMap.map(global, local);
    }
    elemwiseOp = innerFusedOp;
  }

  // Place the carry results on the loops. Inner loop first.
  auto innerLoopNewResultsR = updateLoopWithNewCarriedResults(rewriter, innerLoop, tileResults,
                                                              innerCarryInfo, outerCarryInfo.sizes);
  if (failed(innerLoopNewResultsR))
    BAIL("failed to create new inner loop");
  // Then, propagate the new results of the inner loop to the outer loop.
  auto outerLoopNewResultsR = updateLoopWithNewCarriedResults(
      rewriter, outerLoop, *innerLoopNewResultsR, outerCarryInfo, globalCarrySizes);
  if (failed(outerLoopNewResultsR))
    BAIL("failed to create new outer loop");

  // Just return elemwiseOps because we've updated that vector inplace.
  transformResults.set(getOperation()->getResult(0), elemwiseOps);
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
