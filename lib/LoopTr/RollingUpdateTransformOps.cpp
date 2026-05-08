#include "LoopTr/LoopTransformOps.h"

#include "LoopTr/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <deque>
#include <optional>
#include <variant>

using namespace mlir;

namespace {

bool isReductionLike(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  return llvm::any_of(linalgOp.getIteratorTypesArray(), [](auto iteratorType) {
    return iteratorType == utils::IteratorType::reduction;
  });
}

LogicalResult verifyElementwiseMap(linalg::MapOp map) {
  if (map.getNumDpsInits() != 1)
    return failure();
  if (map->getNumResults() != 1)
    return failure();
  return success();
}

LogicalResult verifyElementwiseGeneric(linalg::GenericOp generic) {
  if (generic.getNumDpsInits() != 1)
    return failure();
  if (generic->getNumResults() != 1)
    return failure();
  if (!generic.isAllParallelLoops())
    return failure();
  if (!generic.hasPureTensorSemantics())
    return failure();
  if (!llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isProjectedPermutation(); }))
    return failure();
  if (!generic.getIndexingMapsArray().back().isIdentity())
    return failure();
  return success();
}

LogicalResult verifyForceFusibleElementwiseOp(Operation *op) {
  if (auto map = dyn_cast<linalg::MapOp>(op))
    return verifyElementwiseMap(map);
  if (auto generic = dyn_cast<linalg::GenericOp>(op))
    return verifyElementwiseGeneric(generic);
  return failure();
}

SmallVector<Value> getElementwiseInputs(Operation *op) {
  if (auto map = dyn_cast<linalg::MapOp>(op))
    return llvm::to_vector(map.getInputs());
  return llvm::to_vector(cast<linalg::GenericOp>(op).getInputs());
}

FailureOr<Value> createDestinationForResultAtInsertionPoint(RewriterBase &rewriter,
                                                            OpResult opResult) {
  auto tensorType = dyn_cast<TensorType>(opResult.getType());
  if (!tensorType)
    return failure();

  SmallVector<OpFoldResult> mixedSizes;
  if (!tensorType.hasStaticShape()) {
    ReifiedRankedShapedTypeDims reifiedShapes;
    if (failed(reifyResultShapes(rewriter, opResult.getDefiningOp(), reifiedShapes)))
      return failure();
    mixedSizes = reifiedShapes[opResult.getResultNumber()];
  } else {
    for (int64_t size : tensorType.getShape())
      mixedSizes.push_back(rewriter.getIndexAttr(size));
  }

  return tensor::EmptyOp::create(rewriter, opResult.getLoc(), mixedSizes,
                                 tensorType.getElementType())
      .getResult();
}

struct SidecarValueInfo {
  Value fullValue, tileValue;
  SmallVector<OpFoldResult> tileOffsets, tileSizes;
};

using FoldResultsT = SmallVector<OpFoldResult>;

std::variant<DiagnosedSilenceableFailure, std::tuple<TilingInterface, FoldResultsT, FoldResultsT>>
createOneTiledSidecarOp(transform::TransformRewriter &rewriter,
                        const transform::TransformOpInterface &transform,
                        TilingInterface clonedConsumerOp,
                        const SmallVectorImpl<Value> &originalInputs, scf::ForOp &currentLoop,
                        const DenseMap<Value, SidecarValueInfo> &sidecarValueInfo,
                        const DenseMap<unsigned, tensor::InsertSliceOp> &loopResultInserts) {
  SmallVector<unsigned> fusedOperandNumbers;
  SmallVector<SmallVector<OpFoldResult>> fusedOffsets;
  SmallVector<SmallVector<OpFoldResult>> fusedSizes;
  SmallVector<Value> fusedTiles;
  for (auto [operandNum, originalInput] : llvm::enumerate(originalInputs)) {
    if (auto it = sidecarValueInfo.find(originalInput); it != sidecarValueInfo.end()) {
      fusedOperandNumbers.push_back(operandNum);
      fusedOffsets.push_back(it->second.tileOffsets);
      fusedSizes.push_back(it->second.tileSizes);
      fusedTiles.push_back(it->second.tileValue);
      continue;
    }

    auto inputResult = dyn_cast<OpResult>(originalInput);
    if (!inputResult || inputResult.getOwner() != currentLoop.getOperation())
      continue;
    auto insertIt = loopResultInserts.find(inputResult.getResultNumber());
    if (insertIt == loopResultInserts.end())
      continue;
    auto loopInsert = insertIt->second;
    fusedOperandNumbers.push_back(operandNum);
    fusedOffsets.push_back(loopInsert.getMixedOffsets());
    fusedSizes.push_back(loopInsert.getMixedSizes());
    fusedTiles.push_back(loopInsert.getSource());
  }
  if (fusedOperandNumbers.empty())
    return emitSilenceableFailure(transform,
                                  "expected each elementwise op in the chain to consume at "
                                  "least one loop-local or prior sidecar value");

  FailureOr<TilingResult> tiledSidecar = clonedConsumerOp.getTiledImplementationFromOperandTiles(
      rewriter, fusedOperandNumbers, fusedOffsets, fusedSizes);
  if (failed(tiledSidecar) || tiledSidecar->tiledOps.empty() ||
      tiledSidecar->tiledValues.size() != 1)
    return emitSilenceableFailure(transform,
                                  "failed to tile sidecar elementwise op from operand tiles");

  auto tiledSidecarOp = cast<TilingInterface>(tiledSidecar->tiledOps[0]);
  for (auto [fusedOperandNum, fusedTile] : llvm::zip_equal(fusedOperandNumbers, fusedTiles)) {
    rewriter.replaceAllUsesWith(tiledSidecarOp->getOperand(fusedOperandNum), fusedTile);
  }
  rewriter.eraseOp(clonedConsumerOp);

  SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;
  if (failed(tiledSidecarOp.getIterationDomainTileFromOperandTiles(
          rewriter, fusedOperandNumbers, fusedOffsets, fusedSizes, iterDomainOffsets,
          iterDomainSizes)))
    return emitSilenceableFailure(transform,
                                  "failed to get iteration domain tile from sidecar operand tiles");

  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  if (failed(tiledSidecarOp.getResultTilePosition(rewriter, 0, iterDomainOffsets, iterDomainSizes,
                                                  resultOffsets, resultSizes)))
    return emitSilenceableFailure(transform,
                                  "failed to get result tile position for sidecar elementwise op");

  return std::make_tuple(tiledSidecarOp, resultOffsets, resultSizes);
}

/// Clone the given elementwise ops, tile them according to `currentLoop`, and fuse them into the
/// loop body.
std::variant<DiagnosedSilenceableFailure, std::pair<scf::ForOp, SmallVector<Operation *>>>
createSidecarOpsInForLoop(transform::TransformRewriter &rewriter,
                          const transform::TransformOpInterface &transform,
                          ArrayRef<Operation *> elemwiseOps, scf::ForOp currentLoop) {
  if (elemwiseOps.empty())
    return emitSilenceableFailure(transform, "expected at least one elementwise payload op");
  if (!currentLoop)
    return emitSilenceableFailure(transform, "expected the streaming loop to be an scf.for");

  for (Operation *elemwiseOp : elemwiseOps)
    if (failed(verifyForceFusibleElementwiseOp(elemwiseOp)))
      return emitSilenceableFailure(transform,
                                    "expected every op in the chain to be an elementwise "
                                    "linalg.map or linalg.generic with one tensor result");

  // Step 1. Each elemwise op to fuse may introduce extra loop-carried values.
  // Rebuild the streaming loop once with extra iter_arg needed to hold these
  // values.
  // 1.1. Enlist these values and clone the loop.
  unsigned oldNumResults = currentLoop.getNumResults();
  rewriter.setInsertionPoint(currentLoop);
  SmallVector<Value> newInitArgs = llvm::to_vector(currentLoop.getInitArgs());
  for (Operation *elemwiseOp : elemwiseOps) {
    auto destination =
        createDestinationForResultAtInsertionPoint(rewriter, elemwiseOp->getResult(0));
    if (failed(destination))
      return emitSilenceableFailure(transform,
                                    "failed to create a dominating destination tensor for "
                                    "an elementwise sidecar result");
    newInitArgs.push_back(*destination);
  }
  auto newLoop =
      scf::ForOp::create(rewriter, currentLoop.getLoc(), currentLoop.getLowerBound(),
                         currentLoop.getUpperBound(), currentLoop.getStep(), newInitArgs);

  // 1.2. Set up old value to new value mapping for the loop induction variable
  // and region iter args.
  auto *newBody = newLoop.getBody();
  IRMapping mapping;
  mapping.map(currentLoop.getInductionVar(), newLoop.getInductionVar());
  for (auto [index, oldArg] : llvm::enumerate(currentLoop.getRegionIterArgs()))
    mapping.map(oldArg, newLoop.getRegionIterArgs()[index]);

  // 1.3. Clone the loop body except the yield operation.
  rewriter.setInsertionPointToEnd(newBody);
  for (Operation &op : currentLoop.getBody()->without_terminator())
    rewriter.clone(op, mapping);

  // 1.4. Build cloned yield operands and a map from loop result index to the
  // cloned tensor.insert_slice that produces it.
  auto oldYield = cast<scf::YieldOp>(currentLoop.getBody()->getTerminator());
  SmallVector<Value> clonedYieldOperands;
  clonedYieldOperands.reserve(oldYield.getNumOperands());
  DenseMap<unsigned, tensor::InsertSliceOp> loopResultInserts;
  for (auto [idx, operand] : llvm::enumerate(oldYield.getOperands())) {
    Value cloned = mapping.lookupOrDefault(operand);
    clonedYieldOperands.push_back(cloned);
    if (auto insertSlice = cloned.getDefiningOp<tensor::InsertSliceOp>())
      loopResultInserts[idx] = insertSlice;
  }
  rewriter.setInsertionPointToEnd(newBody);

  // Step 2. Clone each elementwise op under the rebuilt loop in the original
  // producer-to-consumer order, using previously cloned sidecar tiles when the
  // chain depends on earlier elementwise results.
  DenseMap<Value, Value> loopResultValues;
  for (auto [index, result] : llvm::enumerate(currentLoop.getResults()))
    loopResultValues[result] = newLoop.getRegionIterArgs()[index];
  SmallVector<Operation *> sidecarOps;
  sidecarOps.reserve(elemwiseOps.size());
  SmallVector<Value> sidecarYieldOperands;
  sidecarYieldOperands.reserve(elemwiseOps.size());
  DenseMap<Value, SidecarValueInfo> sidecarValueInfo;

  for (auto [index, elemwiseOp] : llvm::enumerate(elemwiseOps)) {
    SmallVector<Value> originalInputs = getElementwiseInputs(elemwiseOp);
    SmallVector<Value> remappedFullInputs = originalInputs;
    for (Value &input : remappedFullInputs) {
      if (auto mapped = sidecarValueInfo.find(input); mapped != sidecarValueInfo.end()) {
        input = mapped->second.fullValue;
      } else if (auto mapped = loopResultValues.find(input); mapped != loopResultValues.end()) {
        input = mapped->second;
      }
    }

    SmallVector<Value> remappedOperands = remappedFullInputs;
    Value sidecarFullTensor = newLoop.getRegionIterArgs()[oldNumResults + index];
    remappedOperands.push_back(sidecarFullTensor);
    Operation *clonedConsumer =
        mlir::clone(rewriter, elemwiseOp, elemwiseOp->getResultTypes(), remappedOperands);
    auto clonedConsumerOp = dyn_cast<TilingInterface>(clonedConsumer);
    if (!clonedConsumerOp)
      return emitSilenceableFailure(transform, "expected cloned elementwise op to implement "
                                               "TilingInterface");

    auto tileOpResult =
        createOneTiledSidecarOp(rewriter, transform, clonedConsumerOp, originalInputs, currentLoop,
                                sidecarValueInfo, loopResultInserts);
    if (std::holds_alternative<DiagnosedSilenceableFailure>(tileOpResult))
      return std::get<DiagnosedSilenceableFailure>(std::move(tileOpResult));
    auto [tiledSidecarOp, resultOffsets, resultSizes] = std::get<1>(tileOpResult);

    SmallVector<OpFoldResult> resultStrides(resultOffsets.size(), rewriter.getIndexAttr(1));
    auto sidecarInsert =
        tensor::InsertSliceOp::create(rewriter, elemwiseOp->getLoc(), tiledSidecarOp->getResult(0),
                                      sidecarFullTensor, resultOffsets, resultSizes, resultStrides);

    sidecarOps.push_back(tiledSidecarOp.getOperation());
    sidecarValueInfo[elemwiseOp->getResult(0)] = SidecarValueInfo{
        .fullValue = sidecarFullTensor,
        .tileValue = tiledSidecarOp->getResult(0),
        .tileOffsets = resultOffsets,
        .tileSizes = resultSizes,
    };
    sidecarYieldOperands.push_back(sidecarInsert.getResult());
  }

  // Step 3. Publish all sidecar tensors as extra loop results, then replace
  // the old loop while leaving the original out-of-loop chain untouched.
  SmallVector<Value> newYieldOperands = clonedYieldOperands;
  newYieldOperands.append(sidecarYieldOperands.begin(), sidecarYieldOperands.end());
  scf::YieldOp::create(rewriter, currentLoop.getLoc(), newYieldOperands);
  rewriter.replaceOp(currentLoop, newLoop.getResults().take_front(oldNumResults));
  return std::make_pair(newLoop, sidecarOps);
}

} // namespace

namespace mlir {
namespace transform {

DiagnosedSilenceableFailure
LoopRURollingUpdateNextReduction::apply(transform::TransformRewriter &rewriter,
                                        TransformResults &transformResults, TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getProducerOp, "producer", producer);

  // Forward BFS: find the nearest reduction.
  SmallPtrSet<Operation *, 16> visited({producer});
  std::deque<Operation *> queue({producer});
  Operation *reduce = nullptr;
  while (!queue.empty()) {
    Operation *current = queue.front();
    queue.pop_front();
    if (current != producer && isReductionLike(current)) {
      reduce = current;
      break;
    }
    for (Value result : current->getOpResults())
      for (Operation *user : result.getUsers())
        if (visited.insert(user).second)
          queue.push_back(user);
  }
  if (!reduce)
    return emitSilenceableFailure(transform, "no reduction reachable from producer op");

  // Backward walk from `reduce`, bounded by `visited`.
  SmallVector<Operation *> elemwiseOps;
  {
    SmallPtrSet<Operation *, 16> bvisited({reduce});
    std::deque<Operation *> bqueue({reduce});
    while (!bqueue.empty()) {
      Operation *current = bqueue.front();
      bqueue.pop_front();
      if (current != reduce) {
        if (failed(verifyForceFusibleElementwiseOp(current)))
          return emitSilenceableFailure(
              transform, "expected all ops between producer_op and reduce_op to be elementwise");
        elemwiseOps.push_back(current);
      }
      for (Value operand : current->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (defOp && defOp != producer && visited.contains(defOp))
          if (bvisited.insert(defOp).second)
            bqueue.push_back(defOp);
      }
    }
  }
  llvm::sort(elemwiseOps, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  transformResults.set(getOperation()->getResult(0), {reduce});
  transformResults.set(getOperation()->getResult(1), elemwiseOps);
  return DiagnosedSilenceableFailure::success();
}

void LoopRUCloneFuseElemwise::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getElemwiseChainOpsMutable(), effects);

  onlyReadsHandle(getOuterLoopMutable(), effects);
  onlyReadsHandle(getInnerLoopMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure LoopRUCloneFuseElemwise::apply(transform::TransformRewriter &rewriter,
                                                           TransformResults &transformResults,
                                                           TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  SmallVector<Operation *> elemwiseOps =
      llvm::to_vector(state.getPayloadOps(getElemwiseChainOps()));
  if (elemwiseOps.empty())
    return emitSilenceableFailure(transform, "expected at least one elementwise payload op");
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getOuterLoop, "outer loop", outerLoop,
                               scf::ForallOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getInnerLoop, "inner loop", innerLoop, scf::ForOp);

  // Step 1. Fuse the elemwise consumer into the scf.forall using the same
  //         logic as fuse_elemwise_into_producer (upstream tileAndFuseConsumer).
  SmallVector<LoopLikeOpInterface> loopNest{outerLoop};
  FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult =
      scf::tileAndFuseConsumer(rewriter, elemwiseOps.front(), loopNest);
  if (failed(fuseResult))
    return emitSilenceableFailure(transform,
                                  "failed to fuse elementwise consumer into forall loop");
  // loopNest[0] has been updated to the new forall by tileAndFuseConsumer.
  // The inner for pointer (innerForOp) is still valid because mergeBlocks
  // moved it (didn't clone) to the new forall body.
  scf::ForallOp newOuterLoop = cast<scf::ForallOp>(loopNest[0].getOperation());

  // Step 2. Verify the inner for is still inside the new forall.
  if (innerLoop->getParentOp() != newOuterLoop.getOperation())
    return emitDefiniteFailure(
        "inner for-loop should have been moved to new forall via mergeBlocks");

  // Step 3. Create sidecar ops for the tiled elemwise ops under the inner for.
  auto sidecarResult =
      createSidecarOpsInForLoop(rewriter, transform, fuseResult->tiledOps, innerLoop);
  if (std::holds_alternative<DiagnosedSilenceableFailure>(sidecarResult))
    return std::get<DiagnosedSilenceableFailure>(std::move(sidecarResult));
  auto [updatedInnerFor, sidecarOps] = std::get<1>(std::move(sidecarResult));

  // Step 4. Return all three results.
  transformResults.set(getOperation()->getResult(0), sidecarOps);
  transformResults.set(getOperation()->getResult(1), {newOuterLoop.getOperation()});
  transformResults.set(getOperation()->getResult(2), {updatedInnerFor.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace mlir
