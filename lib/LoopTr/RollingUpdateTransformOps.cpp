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
#include "llvm/ADT/STLExtras.h"

#include <deque>
#include <optional>

namespace {

using namespace mlir;
using linalg::GenericOp;
using scf::ForallOp;
using scf::ForOp;
using transform::TransformOpInterface;

#define BAIL(message) return emitSilenceableFailure(transform, message);

bool isReductionLike(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  return llvm::any_of(linalgOp.getIteratorTypesArray(), [](auto iteratorType) {
    return iteratorType == utils::IteratorType::reduction;
  });
}

/// Builds a map from in-loop producer operations in `innerLoop` to the
/// corresponding relayed result of `outerLoop`.
FailureOr<DenseMap<Operation *, OpResult>> getOpToLoopResultMap(ForallOp outerLoop,
                                                                ForOp innerLoop) {
  auto chainedMapF = getChainedLoopResultMap({outerLoop, innerLoop});
  if (failed(chainedMapF))
    return failure();

  DenseMap<Operation *, OpResult> innerBodyToOuterRet;
  for (const auto &[outerRet, relays] : *chainedMapF) {
    assert(!relays.empty());
    if (outerRet.getDefiningOp() != outerLoop)
      continue;
    if (Operation *innerProducer = relays.front().inLoopResult.getDefiningOp())
      innerBodyToOuterRet.try_emplace(innerProducer, outerRet);
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
    BAIL("no reduction reachable from producer op");

  // Backward walk from `reduce`, bounded by `visited`.
  SmallVector<Operation *> elemwiseOps;
  {
    SmallPtrSet<Operation *, 16> bvisited({reduce});
    std::deque<Operation *> bqueue({reduce});
    while (!bqueue.empty()) {
      Operation *current = bqueue.front();
      bqueue.pop_front();
      if (current != reduce) {
        if (failed(isSingleOutputElemwiseLinalgOp(current)))
          BAIL("expected all ops between producer_op and reduce_op to be elementwise");
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

void LoopRURepairReductionFrontier::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getReduceMutable(), effects);
  consumesHandle(getElemwiseOrigMutable(), effects);
  consumesHandle(getElemwiseSidecarsMutable(), effects);

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
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getReduce, "reduce", reduce, GenericOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getOuterLoop, "outer loop", outerLoop, ForallOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getInnerLoop, "inner loop", innerLoop, ForOp);
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseOrig, "original elementwise", elemwiseOrig);
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseSidecars, "sidecar elementwise",
                      elemwiseSidecars);
  if (elemwiseOrig.size() != elemwiseSidecars.size())
    BAIL("expected the original and sidecar elementwise chains to have the same size");

  // Get the reduce axis of the reduction.
  auto redDimOrF = matchUnarySingleReductionGeneric(reduce);
  if (failed(redDimOrF))
    BAIL("expected reduce to be a unary single-reduction linalg.generic");
  auto redDim = *redDimOrF;

  // Fuse the reduce operation into the loop nest, changing its input from `elemwiseOrig` to
  // `elemwiseSidecars`. This function takes `elemwiseOrig`, `outerLoop`, etc. by reference,
  // and updates them to point to new operations.
  auto fuseResult = fuseReduceInLoopNest(transform, rewriter, outerLoop, innerLoop, reduce, redDim,
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

  rewriter.setInsertionPointAfter(reduce);
  Operation *consumer = reduce;
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

  transformResults.set(getOperation()->getResult(0), {reduce.getOperation()});
  transformResults.set(getOperation()->getResult(1), elemwiseSidecars);
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace mlir
