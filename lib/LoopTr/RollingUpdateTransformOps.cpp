#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
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

DiagnosedSilenceableFailure fuseReduceInLoopNest(TransformOpInterface transform,
                                                 RewriterBase &rewriter, ForallOp &outerLoop,
                                                 ForOp &innerLoop, GenericOp &reduce,
                                                 SmallVector<Operation *> &elemwiseOrig,
                                                 SmallVector<Operation *> &elemwiseSidecars) {
  // Step 1. Build a value map that rewires the original elementwise chain to the corresponding
  // sidecar values relayed by the outer loop.
  auto chainedMapR = getChainedLoopResultMap({outerLoop, innerLoop});
  if (failed(chainedMapR))
    BAIL("failed to map loop return values to in-loop operations that produce them");
  DenseMap<Operation *, SmallVector<OpResult>> opToLoopResultMap;
  for (const auto &[outerRet, relays] : *chainedMapR) {
    assert(!relays.empty());
    if (Operation *innerProducer = relays.front().inLoopResult.getDefiningOp())
      opToLoopResultMap[innerProducer].push_back(outerRet);
  }
  IRMapping stagedReductionMapping;
  for (auto [elemwiseOp, sidecarOp] : llvm::zip_equal(elemwiseOrig, elemwiseSidecars)) {
    auto it = opToLoopResultMap.find(sidecarOp);
    if (it == opToLoopResultMap.end()) {
      sidecarOp->emitRemark("this sidecar op");
      BAIL("cannot trace the output of a sidecar operation to an output of the outer loop");
    }
    for (auto [opResult, loopResult] : llvm::zip_equal(elemwiseOp->getResults(), it->second))
      stagedReductionMapping.map(opResult, loopResult);
  }

  // Step 2. Stage the reduction as a normal consumer of the sidecar loop result, then delegate the
  // outer+inner loop fusion mechanics to the shared helper.
  rewriter.setInsertionPoint(reduce);
  auto stagedReduce = rewriter.clone(*reduce, stagedReductionMapping);
  // This cloning may have inserted some operations after the outer loop, which prevents the
  // fusion from working. We'll try and move them before the inner loop.
  if (failed(recursiveMoveOperandsBeforeOp(*stagedReduce, rewriter, *outerLoop)))
    BAIL("failed to move operands before the outer loop");
  auto fused = tileAndFuseConsumerIntoDoubleLoops(rewriter, outerLoop, innerLoop, *stagedReduce);
  if (failed(fused))
    BAIL("failed to fuse staged reduction into the loop nest");
  auto [fusedOuterLoop, fusedInnerLoop] = *fused;
  rewriter.eraseOp(stagedReduce);
  rewriter.eraseOp(fusedOuterLoop);
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

  // Fuse the reduce operation into the loop nest, changing its input from `elemwiseOrig` to
  // `elemwiseSidecars`. This function takes `elemwiseOrig`, `outerLoop`, etc. by reference,
  // and updates them to point to new operations.
  auto fuseResult = fuseReduceInLoopNest(transform, rewriter, outerLoop, innerLoop, thisRed,
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
