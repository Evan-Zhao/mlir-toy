#include "LoopTr/LoopTransformOps.h"

#include "LoopTr/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"
#include <deque>
#include <optional>
#include <variant>

#define BAIL(message) return emitSilenceableFailure(transform, message);

namespace {

using namespace mlir;
using linalg::GenericOp;
using scf::ForallOp;
using scf::ForOp;
using transform::TransformOpInterface;

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

LogicalResult verifyElementwiseGeneric(GenericOp generic) {
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
  if (auto generic = dyn_cast<GenericOp>(op))
    return verifyElementwiseGeneric(generic);
  return failure();
}

SmallVector<Value> getElementwiseInputs(Operation *op) {
  if (auto map = dyn_cast<linalg::MapOp>(op))
    return llvm::to_vector(map.getInputs());
  return llvm::to_vector(cast<GenericOp>(op).getInputs());
}

/// Creates a fresh tensor.empty whose shape and element type match `likeValue`.
/// This is used to seed new loop-carried tensors when cloning sidecar chains.
FailureOr<Value> createDestinationLikeValueAtInsertionPoint(RewriterBase &rewriter,
                                                            Value likeValue) {
  auto tensorType = dyn_cast<RankedTensorType>(likeValue.getType());
  if (!tensorType)
    return failure();

  SmallVector<OpFoldResult> mixedSizes;
  mixedSizes.reserve(tensorType.getRank());
  for (auto [index, size] : llvm::enumerate(tensorType.getShape())) {
    if (ShapedType::isDynamic(size)) {
      auto dimOp = tensor::DimOp::create(rewriter, likeValue.getLoc(), likeValue,
                                         static_cast<int64_t>(index));
      mixedSizes.push_back(dimOp.getResult());
    } else {
      mixedSizes.push_back(rewriter.getIndexAttr(size));
    }
  }

  return tensor::EmptyOp::create(rewriter, likeValue.getLoc(), mixedSizes,
                                 tensorType.getElementType())
      .getResult();
}

/// Clones the defining chain of `value` only as far as needed to make it dominate
/// the current insertion point. Existing dominating definitions are reused.
FailureOr<Value> cloneValueDefChainAtInsertionPoint(RewriterBase &rewriter, Value value,
                                                    IRMapping &mapping) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  Operation *def = value.getDefiningOp();
  if (!def)
    return value;

  Block *insertBlock = rewriter.getInsertionBlock();
  auto insertPoint = rewriter.getInsertionPoint();
  Operation *insertPointOp = insertPoint == insertBlock->end() ? nullptr : &*insertPoint;
  if (!insertPointOp || def->getBlock() != insertBlock || def->isBeforeInBlock(insertPointOp))
    return value;

  IRMapping localMapping = mapping;
  for (Value operand : def->getOperands()) {
    FailureOr<Value> remappedOperand =
        cloneValueDefChainAtInsertionPoint(rewriter, operand, mapping);
    if (failed(remappedOperand))
      return failure();
    localMapping.map(operand, *remappedOperand);
  }

  Operation *cloned = rewriter.clone(*def, localMapping);
  for (auto [oldResult, newResult] : llvm::zip_equal(def->getResults(), cloned->getResults()))
    mapping.map(oldResult, newResult);
  return mapping.lookup(value);
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

/// For each outer-loop result consumed by the elementwise chain, record which
/// inner-loop result number feeds it through a tensor.parallel_insert_slice.
std::variant<DiagnosedSilenceableFailure, DenseMap<Value, unsigned>>
getSeededLoopResultIndices(transform::TransformOpInterface transform,
                           ArrayRef<Operation *> elemwiseOps, ForallOp outerLoop,
                           ForOp currentLoop) {
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

/// Re-tile one cloned sidecar op from loop-local producer tiles, then report
/// the tiled op together with the slice position of its result in the full tensor.
std::variant<DiagnosedSilenceableFailure, std::tuple<TilingInterface, FoldResultsT, FoldResultsT>>
createOneTiledSidecarOp(transform::TransformRewriter &rewriter,
                        const transform::TransformOpInterface &transform,
                        TilingInterface clonedConsumerOp,
                        const SmallVectorImpl<Value> &originalInputs, ForOp &currentLoop,
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

/// Builds a map from in-loop producer operations in `innerLoop` to the
/// corresponding relayed result of `outerLoop`.
FailureOr<DenseMap<Operation *, OpResult>> getOpToLoopResultMap(ForallOp outerLoop,
                                                                ForOp innerLoop) {
  auto loopResultRelaysF = getNestedLoopResultRelays({outerLoop, innerLoop});
  if (failed(loopResultRelaysF))
    return failure();
  // loopResultRelays[i][j] refers to the j-th result of the i-th loop
  // and describes where that result comes from, so we'll need to chain it over every i.
  const auto &loopResultRelays = *loopResultRelaysF;
  assert(loopResultRelays.size() == 2);
  // This is a chained map that maps from loop-i returned result to innermost body result.
  DenseMap<OpResult, OpResult> chainedMap;
  // First populate it with inner loop return -> body entries
  for (auto relay : loopResultRelays[1]) {
    chainedMap.try_emplace(relay.loopReturnResult, relay.inLoopResult);
  }
  // Then chain it with outer loop return -> body, where outer loop's body result is the inner
  // loop's returned result.
  for (auto relay : loopResultRelays[0]) {
    auto it = chainedMap.find(relay.inLoopResult);
    if (it != chainedMap.end()) {
      auto mappedValue = it->second;
      chainedMap.erase(it);
      chainedMap.try_emplace(relay.loopReturnResult, mappedValue);
    }
  }
  // Flip the map to get what we want to return.
  DenseMap<Operation *, OpResult> innerBodyToOuterRet;
  for (auto &[outerRet, innerBody] : chainedMap) {
    if (outerRet.getDefiningOp() == outerLoop)
      innerBodyToOuterRet.try_emplace(innerBody.getDefiningOp(), outerRet);
  }
  return innerBodyToOuterRet;
}

DiagnosedSilenceableFailure fuseReduceInLoopNest(TransformOpInterface transform,
                                                 RewriterBase &rewriter, ForallOp &outerLoop,
                                                 ForOp &innerLoop, GenericOp &reduce, size_t redDim,
                                                 SmallVector<Operation *> &elemwiseOrig,
                                                 SmallVector<Operation *> &elemwiseSidecars) {
  // Step 1. Find the elementwise ops that the reduction reads. For each elementwise `e_i`, there is
  // a cloned "sidecar" version `s_i` under the nested loop. `s_i` computes a tile of result at a
  // time, which accumulates over the loop iterations, so the outer loop has a corresponding result
  // `r_j`. The following chunk of code finds this `j` (which is `resultNumber` below).
  auto opToLoopResultMap = getOpToLoopResultMap(outerLoop, innerLoop);
  if (failed(opToLoopResultMap))
    BAIL("failed to map loop return values to in-loop operations that produce them");
  SmallVector<std::pair<Operation *, unsigned>> sidecarsUsedByReduce;
  for (auto [elemwiseOp, sidecarOp] : llvm::zip_equal(elemwiseOrig, elemwiseSidecars)) {
    Value originalResult = elemwiseOp->getResult(0);
    bool usedByReduce = llvm::any_of(originalResult.getUses(),
                                     [&](OpOperand &use) { return use.getOwner() == reduce; });
    if (!usedByReduce)
      continue;
    auto it = opToLoopResultMap->find(sidecarOp);
    if (it == opToLoopResultMap->end()) {
      sidecarOp->emitRemark("this sidecar op");
      BAIL("cannot trace the output of a sidecar operation to an output of the outer loop");
    }
    sidecarsUsedByReduce.emplace_back(sidecarOp, it->second.getResultNumber());
  }
  if (size_t size = sidecarsUsedByReduce.size(); size != 1) {
    reduce->emitRemark("this reduction:");
    BAIL("expected the reduction to consume exactly one elementwise result; got " +
         std::to_string(size) + " results");
  }
  auto [sidecarOp, resultNumber] = sidecarsUsedByReduce.front();

  // Step 2. Add a fresh reduction slot to the outer forall, recover the sidecar
  // slice geometry from the chosen relayed result, and seed the inner reduction tile.
  unsigned oldNumOuterResults = outerLoop.getNumResults();
  rewriter.setInsertionPoint(outerLoop);
  IRMapping reductionInitMapping;
  FailureOr<Value> reductionInit = cloneValueDefChainAtInsertionPoint(
      rewriter, reduce.getDpsInits().front(), reductionInitMapping);
  if (failed(reductionInit))
    BAIL("failed to clone a dominating init tensor for the fused reduction");
  SmallVector<Value> newOuterOutputs = llvm::to_vector(outerLoop.getOutputs());
  newOuterOutputs.push_back(*reductionInit);
  auto newOuterLoop =
      ForallOp::create(rewriter, outerLoop.getLoc(), outerLoop.getMixedLowerBound(),
                       outerLoop.getMixedUpperBound(), outerLoop.getMixedStep(), newOuterOutputs,
                       outerLoop.getMapping(), [](OpBuilder &, Location, ValueRange) {});
  Block *oldOuterBody = outerLoop.getBody();
  Block *newOuterBody = newOuterLoop.getBody();
  rewriter.mergeBlocks(oldOuterBody, newOuterBody,
                       newOuterBody->getArguments().take_front(oldOuterBody->getNumArguments()));

  auto producerOuterResult = cast<OpResult>(newOuterLoop->getResult(resultNumber));
  auto producerOuterInsertF =
      getParallelInsertSliceForLoopResult(newOuterLoop, producerOuterResult);
  if (failed(producerOuterInsertF))
    BAIL("failed to find the outer-loop relay for the reduction producer sidecar");
  auto producerOuterInsert = *producerOuterInsertF;

  auto dropAt = [](SmallVector<OpFoldResult> values, uint64_t index) {
    values.erase(values.begin() + index);
    return values;
  };
  SmallVector<OpFoldResult> reductionOffsets =
      dropAt(producerOuterInsert.getMixedOffsets(), redDim);
  SmallVector<OpFoldResult> reductionSizes = dropAt(producerOuterInsert.getMixedSizes(), redDim);
  SmallVector<OpFoldResult> reductionStrides = getUnitStrides(rewriter, reductionOffsets.size());

  // Step 3. Rebuild the inner loop with one extra iter_arg/result for the fused
  // reduction, then relay that new result through the rebuilt outer forall.
  unsigned oldNumInnerResults = innerLoop.getNumResults();
  rewriter.setInsertionPoint(innerLoop);
  Value reductionTileInit = createExtractSliceFromState(
      rewriter, reduce.getLoc(), newOuterLoop.getRegionIterArgs().back(), reductionOffsets,
      reductionSizes, reductionStrides);
  SmallVector<Value> newInnerInitArgs = llvm::to_vector(innerLoop.getInitArgs());
  newInnerInitArgs.push_back(reductionTileInit);
  auto newInnerLoop =
      ForOp::create(rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
                    innerLoop.getUpperBound(), innerLoop.getStep(), newInnerInitArgs);

  auto *newInnerBody = newInnerLoop.getBody();
  IRMapping mapping;
  mapping.map(innerLoop.getInductionVar(), newInnerLoop.getInductionVar());
  for (auto [index, oldArg] : llvm::enumerate(innerLoop.getRegionIterArgs()))
    mapping.map(oldArg, newInnerLoop.getRegionIterArgs()[index]);

  rewriter.setInsertionPointToEnd(newInnerBody);
  for (Operation &op : innerLoop.getBody()->without_terminator())
    rewriter.clone(op, mapping);

  auto oldYield = cast<scf::YieldOp>(innerLoop.getBody()->getTerminator());
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(oldYield.getNumOperands() + 1);
  for (Value operand : oldYield.getOperands())
    newYieldOperands.push_back(mapping.lookupOrDefault(operand));

  auto clonedSidecarResult = dyn_cast<OpResult>(mapping.lookupOrDefault(sidecarOp->getResult(0)));
  if (!clonedSidecarResult)
    BAIL("failed to remap the sidecar op result into the rebuilt inner loop");
  rewriter.setInsertionPointToEnd(newInnerBody);
  auto fusedReduction =
      cloneGenericOnTile(rewriter, reduce, clonedSidecarResult,
                         newInnerLoop.getRegionIterArgs().back(), reduce.getLoc());
  newYieldOperands.push_back(fusedReduction.getResult(0));
  scf::YieldOp::create(rewriter, innerLoop.getLoc(), newYieldOperands);
  rewriter.replaceOp(innerLoop, newInnerLoop.getResults().take_front(oldNumInnerResults));

  pointRewriterToForallParallel(rewriter, newOuterLoop);
  tensor::ParallelInsertSliceOp::create(rewriter, reduce.getLoc(), newInnerLoop.getResults().back(),
                                        newOuterLoop.getRegionIterArgs().back(), reductionOffsets,
                                        reductionSizes, reductionStrides);
  rewriter.replaceOp(outerLoop, newOuterLoop.getResults().take_front(oldNumOuterResults));
  rewriter.replaceOp(reduce, newOuterLoop.getResults().back());

  // Step 4. Remap operations to their cloned counterparts, since the original ops were erased
  // during the rebuild.
  reduce = fusedReduction;
  outerLoop = newOuterLoop;
  innerLoop = newInnerLoop;
  // For elemwise ops (no need to update original elemwise, because they weren't changed)
  for (size_t i = 0; i < elemwiseSidecars.size(); ++i) {
    elemwiseSidecars[i] = dyn_cast_if_present<GenericOp>(
        mapping.lookupOrDefault(elemwiseSidecars[i]->getResult(0)).getDefiningOp());
    if (!elemwiseSidecars[i])
      BAIL("failed to remap elemwise (sidecars) ops after fusing reduction into the loop nest");
  }
  return DiagnosedSilenceableFailure::success();
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
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseChainOps, "elementwise", elemwiseOps)
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getOuterLoop, "outer loop", outerLoop, ForallOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getInnerLoop, "inner loop", innerLoop, ForOp);

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

  // Step 1. Rebuild the inner streaming loop with one extra iter_arg/result
  // per sidecar tensor that will be materialized under the loop.
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
  auto newLoop = ForOp::create(rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
                               innerLoop.getUpperBound(), innerLoop.getStep(), newInitArgs);

  // Clone the old loop body under the replacement loop.
  auto *newBody = newLoop.getBody();
  IRMapping mapping;
  mapping.map(innerLoop.getInductionVar(), newLoop.getInductionVar());
  for (auto [index, oldArg] : llvm::enumerate(innerLoop.getRegionIterArgs()))
    mapping.map(oldArg, newLoop.getRegionIterArgs()[index]);

  rewriter.setInsertionPointToEnd(newBody);
  for (Operation &op : innerLoop.getBody()->without_terminator())
    rewriter.clone(op, mapping);

  // Recover the cloned loop-carried tensors and the insert_slice ops that update them.
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

  // Step 2. Clone the elementwise chain under the rebuilt loop, tile each op
  // from loop-local producer tiles, and append the new sidecar tensors to the yield.
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

  // Notify the rewriter about the operation replacements. This allows all handles in the same
  // transform-dialect program to stay valid.
  // The `rewriter.replaceOp` step next will not update sub-operations recursively, so we'll need to
  // do this ourselves.
  for (auto &[k, v] : mapping.getOperationMap()) {
    // This fails when `k` is not tracked by any handle, and that is not a problem.
    if (failed(rewriter.notifyPayloadOperationReplaced(k, v)))
      continue;
  }
  // Update the loop.
  rewriter.replaceOp(innerLoop, newLoop.getResults().take_front(oldNumResults));

  // Step 3. Relay the new inner-loop results through the outer forall using the
  // same slice pattern as the already-relayed producer tensors.
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

  auto newOuterLoop =
      ForallOp::create(rewriter, outerLoop.getLoc(), outerLoop.getMixedLowerBound(),
                       outerLoop.getMixedUpperBound(), outerLoop.getMixedStep(), newOuterOutputs,
                       outerLoop.getMapping(), [](OpBuilder &, Location, ValueRange) {});
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

  transformResults.set(getOperation()->getResult(0), sidecarOps);
  return DiagnosedSilenceableFailure::success();
}

void LoopRURepairReductionFrontier::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getProducerReducesMutable(), effects);
  consumesHandle(getThisReduceMutable(), effects);
  onlyReadsHandle(getElemwiseOrigMutable(), effects);
  onlyReadsHandle(getElemwiseSidecarsMutable(), effects);
  onlyReadsHandle(getOuterLoopMutable(), effects);
  onlyReadsHandle(getInnerLoopMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
LoopRURepairReductionFrontier::apply(transform::TransformRewriter &rewriter,
                                     TransformResults &transformResults, TransformState &state) {
  // Do some basic validation.
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_NON_EMPTY_OPS(state, transform, getProducerReduces, "producer reductions", producerReds);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getThisReduce, "this reduction", thisRed,
                               GenericOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getOuterLoop, "outer loop", outerLoop, ForallOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getInnerLoop, "inner loop", innerLoop, ForOp);
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseOrig, "original elementwise", elemwiseOrig);
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseSidecars, "sidecar elementwise",
                      elemwiseSidecars);
  if (elemwiseOrig.size() != elemwiseSidecars.size())
    BAIL("expected the original and sidecar elementwise chains to have the same size");

  // Get the reduce axis of the reduction.
  auto redDimOrF = matchUnarySingleReductionGeneric(thisRed);
  if (failed(redDimOrF))
    BAIL("expected reduce to be a unary single-reduction linalg.generic");
  auto redDim = *redDimOrF;

  // Fuse the reduce operation into the loop nest, changing its input from `elemwiseOrig` to
  // `elemwiseSidecars`. This function takes `elemwiseOrig`, `outerLoop`, etc. by reference,
  // and updates them to point to new operations.
  auto fuseResult = fuseReduceInLoopNest(transform, rewriter, outerLoop, innerLoop, thisRed, redDim,
                                         elemwiseOrig, elemwiseSidecars);
  if (!fuseResult.succeeded())
    return fuseResult;

  // Fuse sidecar ops into `reduce` and other sidecar ops, in a TVM "compute_inline" manner.
  SmallPtrSet<Operation *, 4> sidecarSet(elemwiseSidecars.begin(), elemwiseSidecars.end());
  // MLIR "compute-inlining" is provided by `linalg::fuseElementwiseOps`, which takes only an
  // operand on the consumer side, and "pulls in" the producers.
  // We figure out which operand of the consumer is provided by one of the sidecar ops.
  // Returning the operand number because OpOperand is not copyable.
  auto findFusableOperand =
      [&sidecarSet](Operation *consumer) -> std::optional<std::pair<Operation *, unsigned>> {
    for (auto &operand : consumer->getOpOperands()) {
      auto producer = operand.get().getDefiningOp();
      if (producer && sidecarSet.count(producer)) {
        return std::make_pair(producer, operand.getOperandNumber());
      }
    }
    return std::nullopt;
  };

  rewriter.setInsertionPointAfter(thisRed);
  Operation *consumer = thisRed;
  while (auto nextFusionTarget = findFusableOperand(consumer)) {
    // `producer` is guaranteed to be a sidecar op.
    auto [producer, consumerOpndNum] = *nextFusionTarget;
    FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
        linalg::fuseElementwiseOps(rewriter, &consumer->getOpOperand(consumerOpndNum));
    if (failed(fusionResult)) {
      producer->emitError("when fusing this op...");
      consumer->emitError("into this op...");
      BAIL("failed to fuse sidecar and reduce ops");
    }
    consumer = fusionResult->fusedOp;
  }
  llvm::errs() << "Fusion succeeded and produced " << *consumer << "\n";

  transformResults.set(getOperation()->getResult(0), {thisRed.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace mlir
