#include "LoopTr/FusionExprSolver.h"
#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/STLExtras.h"

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

template <typename T>
DiagnosedSilenceableFailure
fuseReduceInLoopNest(TransformOpInterface transform, RewriterBase &rewriter, ForallOp &outerLoop,
                     ForOp &innerLoop, GenericOp &reduce, T elemwiseOpPairs) {
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
  for (auto [elemwiseOp, sidecarOp] : elemwiseOpPairs) {
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
  auto [fusedOuter, fusedInner] = *fused;
  rewriter.eraseOp(stagedReduce);
  rewriter.eraseOp(fusedOuter);
  // Replace the old out-of-loop reduction with the results of the outer loop (which carries the
  // result of the fused reduction).
  rewriter.replaceOp(reduce, outerLoop->getResults().take_back(reduce->getNumResults()));
  reduce = cast<GenericOp>(fusedInner);
  return DiagnosedSilenceableFailure::success();
}

} // namespace

namespace mlir {
namespace transform {

void FusionCloneFuseElemwiseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getElemwiseChainOpsMutable(), effects);
  onlyReadsHandle(getOuterLoopMutable(), effects);
  onlyReadsHandle(getInnerLoopMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure FusionCloneFuseElemwiseOp::apply(transform::TransformRewriter &rewriter,
                                                             TransformResults &transformResults,
                                                             TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseChainOps, "elementwise", elemwiseOps)
  ForallOp outerLoop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getOuterLoop, "outer loop", outerLoop, ForallOp);
  ForOp innerLoop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getInnerLoop, "inner loop", innerLoop, ForOp);

  IRMapping mapping;
  for (Operation *&elemwiseOp : elemwiseOps) {
#define BAIL_AND_POINT(message)                                                                    \
  {                                                                                                \
    elemwiseOp->emitError() << "failed on this elementwise op";                                    \
    return emitSilenceableFailure(transform, message);                                             \
  }
    if (failed(isSingleOutputElemwiseLinalgOp(elemwiseOp)))
      BAIL_AND_POINT(
          "expected every op to be an elementwise linalg.map or linalg.generic with one result");

    rewriter.setInsertionPoint(elemwiseOp);
    auto newElemwiseOp = rewriter.clone(*elemwiseOp, mapping);
    if (failed(recursiveMoveOperandsBeforeOp(*newElemwiseOp, rewriter, *outerLoop)))
      BAIL_AND_POINT("failed to move operands before the outer loop");

    auto fuseResult =
        tileAndFuseConsumerIntoDoubleLoops(rewriter, outerLoop, innerLoop, *newElemwiseOp);
    if (failed(fuseResult))
      BAIL_AND_POINT("failed to fuse consumer into double loops");
    auto [outerFusedOp, innerFusedOp] = *fuseResult;

    auto newLoopResults = outerLoop->getResults().take_back(elemwiseOp->getNumResults());
    for (auto [oldResult, newLoopResult] :
         llvm::zip_equal(elemwiseOp->getResults(), newLoopResults)) {
      mapping.map(oldResult, newLoopResult);
    }

    rewriter.eraseOp(newElemwiseOp);
    rewriter.eraseOp(outerFusedOp);
    elemwiseOp = innerFusedOp;

    eliminateLocalCommonSubexpressions(rewriter, outerLoop.getOperation());
  }

  transformResults.set(getOperation()->getResult(0), elemwiseOps);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure FusionFindNextReductionOp::apply(transform::TransformRewriter &rewriter,
                                                             TransformResults &transformResults,
                                                             TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getProducerOp, "producer", producer);
  std::optional<uint64_t> resultNumber = getResultNumber();
  if (resultNumber && *resultNumber >= producer->getNumResults())
    BAIL("result number is out of range for producer op");

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
    SmallVector<Value> results;
    if (current == producer && resultNumber)
      results.push_back(producer->getResult(*resultNumber));
    else
      llvm::append_range(results, current->getOpResults());
    for (Value result : results)
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
        if (failed(isSingleOutputElemwiseLinalgOp(current))) {
          current->emitRemark("this op is not a single-output elementwise linalg op");
          BAIL("expected all ops between producer_op and reduce_op to be elementwise");
        }
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

void FusionRepairReductionFrontierOp::getEffects(
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
FusionRepairReductionFrontierOp::apply(transform::TransformRewriter &rewriter,
                                       TransformResults &transformResults, TransformState &state) {
  // Do some basic validation.
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_NON_EMPTY_OPS(state, transform, getProducerReduces, "producer reductions", producerReds);
  GenericOp thisRed;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getThisReduce, "this reduction", thisRed,
                               GenericOp);
  ForallOp outerLoop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getOuterLoop, "outer loop", outerLoop, ForallOp);
  ForOp innerLoop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getInnerLoop, "inner loop", innerLoop, ForOp);
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseOrig, "original elementwise", elemwiseOrig);
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseSidecars, "sidecar elementwise",
                      elemwiseSidecars);
  if (elemwiseOrig.size() != elemwiseSidecars.size())
    BAIL("expected the original and sidecar elementwise chains to have the same size");
  auto redDimR = matchOneDimReductionGeneric(thisRed);
  if (failed(redDimR))
    BAIL("expected reduce to be a single-dim reduction linalg.generic");

  // Fuse the reduce operation into the loop nest, changing its input from `elemwiseOrig` to
  // `elemwiseSidecars`. This function takes `elemwiseOrig`, `outerLoop`, etc. by reference,
  // and updates them to point to new operations.
  auto fuseResult = fuseReduceInLoopNest(transform, rewriter, outerLoop, innerLoop, thisRed,
                                         llvm::zip_equal(elemwiseOrig, elemwiseSidecars));
  if (!fuseResult.succeeded())
    return fuseResult;

  // Extract scalar expressions that describe the reduction and its producers, then send them to the
  // solver to get a repair term (h-expression).
  auto repairExpr = solveFusionRepairExpr(rewriter, producerReds, thisRed, elemwiseSidecars);
  if (failed(repairExpr))
    BAIL("failed to solve rolling updater expressions");

  SmallVector<FusionRepairReductionBinding> repairBindings;
  for (Operation *producerRed : producerReds) {
    auto producerOp = dyn_cast<DestinationStyleOpInterface>(producerRed);
    if (!producerOp)
      BAIL("expected producer reduction to implement DestinationStyleOpInterface");
    for (OpResult result : producerRed->getResults()) {
      auto init = producerOp.getDpsInitOperand(result.getResultNumber());
      if (!init)
        BAIL("failed to find producer reduction init for repair binding");
      repairBindings.push_back({result, init->get(), result});
    }
  }

  // Build a new linalg.generic that applies the repair term.
  auto repairUpdateOp = repairExpr->build(rewriter, thisRed.getLoc(), repairBindings,
                                          thisRed.getDpsInitOperand(0)->get(),
                                          FusionRepairTermMode::RollingUpdate, *redDimR);
  if (failed(repairUpdateOp))
    BAIL("failed to build linalg.generic around the h-expression returned by the solver");

  // Clone `thisRed`, but replace the accumulator with the output of `repairUpdateOp`.
  IRMapping repairedReduceMapping;
  repairedReduceMapping.map(thisRed.getDpsInitOperand(0)->get(), repairUpdateOp->getResult(0));
  auto repairedReduceOp = cast<GenericOp>(rewriter.clone(*thisRed, repairedReduceMapping));
  rewriter.replaceOp(thisRed, repairedReduceOp);

  transformResults.set(getOperation()->getResult(0), {repairedReduceOp.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace mlir
