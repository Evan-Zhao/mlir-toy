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
#include <mlir/Interfaces/SideEffectInterfaces.h>
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

FailureOr<Value> createDestinationLikeValueAtInsertionPoint(RewriterBase &rewriter,
                                                            Value likeValue) {
  auto tensorType = dyn_cast<RankedTensorType>(likeValue.getType());
  if (!tensorType)
    return failure();

  SmallVector<OpFoldResult> mixedSizes;
  mixedSizes.reserve(tensorType.getRank());
  for (auto [index, size] : llvm::enumerate(tensorType.getShape())) {
    if (ShapedType::isDynamic(size)) {
      mixedSizes.push_back(
          tensor::DimOp::create(rewriter, likeValue.getLoc(), likeValue, index).getResult());
      continue;
    }
    mixedSizes.push_back(rewriter.getIndexAttr(size));
  }

  return tensor::EmptyOp::create(rewriter, likeValue.getLoc(), mixedSizes,
                                 tensorType.getElementType())
      .getResult();
}

struct SidecarValueInfo {
  Value fullValue, tileValue;
  SmallVector<OpFoldResult> tileOffsets, tileSizes;
};

struct OuterRelayInfo {
  unsigned anchorResultNumber;
};

bool canRepresentSidecarResult(Value candidate, OpResult result) {
  auto candidateType = dyn_cast<RankedTensorType>(candidate.getType());
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!candidateType || !resultType)
    return false;
  return candidateType.getRank() == resultType.getRank() &&
         candidateType.getElementType() == resultType.getElementType();
}

std::variant<DiagnosedSilenceableFailure, DenseMap<Value, unsigned>>
getSeededLoopResultIndices(transform::TransformOpInterface transform,
                           ArrayRef<Operation *> elemwiseOps, scf::ForallOp outerLoop,
                           scf::ForOp currentLoop) {
  DenseMap<Value, unsigned> seededLoopResults;
  llvm::SmallDenseSet<Value> neededInputs;
  for (Operation *elemwiseOp : elemwiseOps)
    neededInputs.insert_range(getElementwiseInputs(elemwiseOp));

  for (OpResult outerResult : outerLoop.getResults()) {
    if (!neededInputs.contains(outerResult))
      continue;

    BlockArgument blockArg = outerLoop.getTiedBlockArgument(outerResult);
    SmallVector<Operation *> combiningOps = outerLoop.getCombiningOps(blockArg);
    if (combiningOps.size() != 1)
      return emitSilenceableFailure(
          transform, "expected each referenced outer loop result to have exactly one "
                     "combining op");

    auto insertSlice = dyn_cast<tensor::ParallelInsertSliceOp>(combiningOps[0]);
    if (!insertSlice)
      return emitSilenceableFailure(transform,
                                    "expected referenced outer loop results to be produced by "
                                    "tensor.parallel_insert_slice");

    auto sourceResult = dyn_cast<OpResult>(insertSlice.getSource());
    if (!sourceResult || sourceResult.getOwner() != currentLoop.getOperation())
      return emitSilenceableFailure(
          transform, "expected referenced outer loop results to be sourced from the inner "
                     "for-loop results");

    seededLoopResults[outerResult] = sourceResult.getResultNumber();
  }

  return seededLoopResults;
}

using FoldResultsT = SmallVector<OpFoldResult>;

std::variant<DiagnosedSilenceableFailure, std::tuple<TilingInterface, FoldResultsT, FoldResultsT>>
createOneTiledSidecarOp(transform::TransformRewriter &rewriter,
                        const transform::TransformOpInterface &transform,
                        TilingInterface clonedConsumerOp,
                        const SmallVectorImpl<Value> &originalInputs, scf::ForOp &currentLoop,
                        const DenseMap<Value, unsigned> &seededLoopResults,
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

    unsigned loopResultNumber;
    if (auto seeded = seededLoopResults.find(originalInput); seeded != seededLoopResults.end()) {
      loopResultNumber = seeded->second;
    } else {
      auto inputResult = dyn_cast<OpResult>(originalInput);
      if (!inputResult || inputResult.getOwner() != currentLoop.getOperation())
        continue;
      loopResultNumber = inputResult.getResultNumber();
    }

    auto insertIt = loopResultInserts.find(loopResultNumber);
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
  SmallVector<Operation *> elemwiseOps =
      llvm::to_vector(state.getPayloadOps(getElemwiseChainOps()));
  if (elemwiseOps.empty())
    return emitSilenceableFailure(transform, "expected at least one elementwise payload op");
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getOuterLoop, "outer loop", outerLoop,
                               scf::ForallOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getInnerLoop, "inner loop", innerLoop, scf::ForOp);

  for (Operation *elemwiseOp : elemwiseOps)
    if (failed(verifyForceFusibleElementwiseOp(elemwiseOp))) {
      elemwiseOp->emitError() << "this op is not a recognized elementwise operation";
      return emitSilenceableFailure(transform, "expected every op to be an elementwise linalg.map "
                                               "or linalg.generic with one tensor result");
    }

  auto seededLoopResults = getSeededLoopResultIndices(transform, elemwiseOps, outerLoop, innerLoop);
  if (std::holds_alternative<DiagnosedSilenceableFailure>(seededLoopResults))
    return std::get<DiagnosedSilenceableFailure>(std::move(seededLoopResults));
  const auto &seededLoopResultsMap = std::get<1>(seededLoopResults);

  // Step 1. Each elemwise op to fuse may introduce extra loop-carried values. Rebuild the streaming
  // loop once with extra iter_arg needed to hold these values.
  // 1.1. Enlist these values and clone the loop.
  unsigned oldNumResults = innerLoop.getNumResults();
  rewriter.setInsertionPoint(innerLoop);
  SmallVector<Value> newInitArgs = llvm::to_vector(innerLoop.getInitArgs());
  DenseMap<Value, Value> panelizedResultValues;
  DenseMap<Value, OuterRelayInfo> outerRelayInfo;
  for (Operation *elemwiseOp : elemwiseOps) {
    Value representativeValue;
    std::optional<unsigned> representativeOuterResultNumber;
    for (Value input : getElementwiseInputs(elemwiseOp)) {
      if (auto priorSidecar = panelizedResultValues.find(input);
          priorSidecar != panelizedResultValues.end() &&
          canRepresentSidecarResult(priorSidecar->second, elemwiseOp->getResult(0))) {
        representativeValue = priorSidecar->second;
        representativeOuterResultNumber = outerRelayInfo[input].anchorResultNumber;
        break;
      } else if (auto seeded = seededLoopResultsMap.find(input);
                 seeded != seededLoopResultsMap.end()) {
        Value seededValue = innerLoop.getResult(seeded->second);
        if (canRepresentSidecarResult(seededValue, elemwiseOp->getResult(0))) {
          representativeValue = seededValue;
          representativeOuterResultNumber = cast<OpResult>(input).getResultNumber();
          break;
        }
      }
    }

    if (!representativeValue)
      return emitSilenceableFailure(
          transform, "failed to infer a panelized destination type for an elementwise sidecar");

    auto destination = createDestinationLikeValueAtInsertionPoint(rewriter, representativeValue);
    if (failed(destination))
      return emitSilenceableFailure(transform,
                                    "failed to create a dominating destination tensor for "
                                    "an elementwise sidecar result");
    newInitArgs.push_back(*destination);
    panelizedResultValues[elemwiseOp->getResult(0)] = *destination;
    outerRelayInfo[elemwiseOp->getResult(0)] = OuterRelayInfo{*representativeOuterResultNumber};
  }
  auto newLoop = scf::ForOp::create(rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
                                    innerLoop.getUpperBound(), innerLoop.getStep(), newInitArgs);

  // 1.2. Set up old value to new value mapping for the loop induction variable
  // and region iter args.
  auto *newBody = newLoop.getBody();
  IRMapping mapping;
  mapping.map(innerLoop.getInductionVar(), newLoop.getInductionVar());
  for (auto [index, oldArg] : llvm::enumerate(innerLoop.getRegionIterArgs()))
    mapping.map(oldArg, newLoop.getRegionIterArgs()[index]);

  // 1.3. Clone the loop body except the yield operation.
  rewriter.setInsertionPointToEnd(newBody);
  for (Operation &op : innerLoop.getBody()->without_terminator())
    rewriter.clone(op, mapping);

  // 1.4. Build cloned yield operands and a map from loop result index to the
  // cloned tensor.insert_slice that produces it.
  auto oldYield = cast<scf::YieldOp>(innerLoop.getBody()->getTerminator());
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
  for (auto [index, result] : llvm::enumerate(innerLoop.getResults()))
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
      } else if (auto seeded = seededLoopResultsMap.find(input);
                 seeded != seededLoopResultsMap.end()) {
        input = loopResultValues[innerLoop.getResult(seeded->second)];
      } else if (auto mapped = loopResultValues.find(input); mapped != loopResultValues.end()) {
        input = mapped->second;
      }
    }

    SmallVector<Value> remappedOperands = remappedFullInputs;
    Value sidecarFullTensor = newLoop.getRegionIterArgs()[oldNumResults + index];
    remappedOperands.push_back(sidecarFullTensor);
    Operation *clonedConsumer =
        mlir::clone(rewriter, elemwiseOp, TypeRange(sidecarFullTensor.getType()), remappedOperands);
    auto clonedConsumerOp = dyn_cast<TilingInterface>(clonedConsumer);
    if (!clonedConsumerOp)
      return emitSilenceableFailure(transform, "expected cloned elementwise op to implement "
                                               "TilingInterface");

    auto tileOpResult =
        createOneTiledSidecarOp(rewriter, transform, clonedConsumerOp, originalInputs, innerLoop,
                                seededLoopResultsMap, sidecarValueInfo, loopResultInserts);
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
  scf::YieldOp::create(rewriter, innerLoop.getLoc(), newYieldOperands);
  rewriter.replaceOp(innerLoop, newLoop.getResults().take_front(oldNumResults));

  // Step 4. Relay the additional inner-loop results through the outer forall as
  // extra shared_outs/results, using the same parallel-insert tiling pattern as
  // the representative seeded outer result.
  unsigned oldNumOuterResults = outerLoop.getNumResults();
  rewriter.setInsertionPoint(outerLoop);
  SmallVector<Value> newOuterOutputs = llvm::to_vector(outerLoop.getOutputs());
  for (Operation *elemwiseOp : elemwiseOps) {
    unsigned anchorResultNumber = outerRelayInfo[elemwiseOp->getResult(0)].anchorResultNumber;
    auto outerInit = createDestinationLikeValueAtInsertionPoint(
        rewriter, outerLoop.getOutputs()[anchorResultNumber]);
    if (failed(outerInit))
      return emitSilenceableFailure(transform,
                                    "failed to create a dominating destination tensor for "
                                    "an outer sidecar result");
    newOuterOutputs.push_back(*outerInit);
  }

  auto newOuterLoop = scf::ForallOp::create(
      rewriter, outerLoop.getLoc(), outerLoop.getMixedLowerBound(), outerLoop.getMixedUpperBound(),
      outerLoop.getMixedStep(), newOuterOutputs, outerLoop.getMapping(),
      [](OpBuilder &, Location, ValueRange) {});
  Block *oldOuterBody = outerLoop.getBody();
  Block *newOuterBody = newOuterLoop.getBody();
  rewriter.mergeBlocks(oldOuterBody, newOuterBody,
                       newOuterBody->getArguments().take_front(oldOuterBody->getNumArguments()));

  auto newOuterTerminator = newOuterLoop.getTerminator();
  rewriter.setInsertionPointToEnd(newOuterTerminator.getBody());
  ValueRange extraOuterRegionArgs = newOuterLoop.getRegionIterArgs().take_back(elemwiseOps.size());
  ValueRange extraInnerResults = newLoop.getResults().drop_front(oldNumResults);
  for (auto [index, elemwiseOp] : llvm::enumerate(elemwiseOps)) {
    unsigned anchorResultNumber = outerRelayInfo[elemwiseOp->getResult(0)].anchorResultNumber;
    BlockArgument anchorBlockArg = newOuterLoop.getTiedBlockArgument(
        cast<OpResult>(newOuterLoop.getResult(anchorResultNumber)));
    SmallVector<Operation *> combiningOps = newOuterLoop.getCombiningOps(anchorBlockArg);
    if (combiningOps.size() != 1)
      return emitSilenceableFailure(transform,
                                    "expected each relayed outer loop result to have exactly one "
                                    "combining op");
    auto anchorInsert = dyn_cast<tensor::ParallelInsertSliceOp>(combiningOps[0]);
    if (!anchorInsert)
      return emitSilenceableFailure(transform,
                                    "expected relayed outer loop results to be produced by "
                                    "tensor.parallel_insert_slice");

    tensor::ParallelInsertSliceOp::create(
        rewriter, elemwiseOp->getLoc(), extraInnerResults[index], extraOuterRegionArgs[index],
        anchorInsert.getMixedOffsets(), anchorInsert.getMixedSizes(),
        anchorInsert.getMixedStrides());
  }
  rewriter.replaceOp(outerLoop, newOuterLoop.getResults().take_front(oldNumOuterResults));

  // Step 5. Return the cloned sidecar chain. The loop handles are preserved and
  // updated by replacement tracking.
  transformResults.set(getOperation()->getResult(0), sidecarOps);
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace mlir
