#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/PythonSolver.h"
#include "LoopTr/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#define BAIL(message) return emitSilenceableFailure(transform, message);

namespace {

using namespace mlir;
namespace json = llvm::json;
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

struct SelfReductionMatch {
  BlockArgument accumulatorArg;
  Value yieldValue;
  Value reduceOperand;
  Operation *reduceCombiner;
};

FailureOr<SelfReductionMatch> matchSelfReductionConsumer(GenericOp reduceOp, OpResult redResult) {
  // Get the in-body value that corresponds to the reduction result, then check if it's a reasonable
  // reduction (scalar) operation, like addF.
  unsigned resultNumber = redResult.getResultNumber();
  auto yield = cast<linalg::YieldOp>(reduceOp.getBody()->getTerminator());
  Value yieldValue = yield.getOperand(resultNumber);
  Operation *combiner = yieldValue.getDefiningOp();
  if (!combiner || combiner->getNumOperands() != 2 || combiner->getNumResults() != 1) {
    reduceOp.emitError() << "expected the reduction combiner to have 2 operands and 1 result";
    if (combiner)
      combiner->emitRemark() << "this is the reduction combiner";
    return failure();
  }
  Value lhs = combiner->getOperand(0), rhs = combiner->getOperand(1);

  // Find the accumulator as an argument of the block.
  unsigned numInputs = reduceOp.getNumDpsInputs();
  auto accumulatorArg = reduceOp.getBlock()->getArgument(numInputs + resultNumber);
  // We're then expecting the other argument of the combiner is the result to fold over.
  Value otherArg = lhs == accumulatorArg ? rhs : lhs;

  return SelfReductionMatch{
      .accumulatorArg = accumulatorArg,
      .yieldValue = yieldValue,
      .reduceOperand = otherArg,
      .reduceCombiner = combiner,
  };
}

struct RollingUpdateSolverInput {
  DenseMap<OpResult, std::string> reductionVars;
  std::string accumulatorVar;
  json::Value fExpr, gExpr;
};

/// Extract scalar expressions that describe a reduction and its elemwise producers, all the way up
/// to previous reductions. Extracted expressions can be sent to the "repair term solver".
//
/// @param rewriter The rewriter to use for creating new ops.
/// @param producingReds A sequence of producing reduction ops.
/// @param thisRed The reduction operation to extract expressions for.
/// @param elemwiseSidecars A sequence of elementwise ops in between the reduction and its
/// producers.
FailureOr<RollingUpdateSolverInput>
extractRepairInputExprs(RewriterBase &rewriter, ArrayRef<Operation *> producingReds,
                        GenericOp thisRed, ArrayRef<Operation *> elemwiseSidecars) {
  // Step 1. fuse and fold sidecar ops all into `thisRed` in a TVM "compute_inline" manner, until we
  // end up with a single `currentOp`.
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
  // Also track which result is the reduction result.
  OpResult reductionResult = cast<OpResult>(thisRed->getResult(0));
  GenericOp currentOp = thisRed;
  while (auto nextFusionTarget = findFusableOperand(currentOp)) {
    // `producer` is guaranteed to be a sidecar op.
    auto [producer, consumerOpndNum] = *nextFusionTarget;
    FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
        linalg::fuseElementwiseOps(rewriter, &currentOp->getOpOperand(consumerOpndNum));
    if (failed(fusionResult)) {
      producer->emitError("failed to fuse this op...");
      currentOp->emitError("into this op...");
      return failure();
    }
    auto it = fusionResult->replacements.find(reductionResult);
    if (it == fusionResult->replacements.end())
      return failure();
    reductionResult = cast<OpResult>(it->second);
    currentOp = cast<GenericOp>(fusionResult->fusedOp);
  }

  // Step 2. Check this currentOp is a reduction, and get some information about it.
  auto match = matchSelfReductionConsumer(currentOp, reductionResult);
  if (failed(match))
    return failure();

  // Step 3. Look at the body of currentOp; mark and name the block arguments by whether they
  // come from `producingReds` or not.
  SmallPtrSet<Value, 4> prodRedResults;
  for (Operation *prodRed : producingReds) {
    prodRedResults.insert(prodRed->result_begin(), prodRed->result_end());
  }
  auto genericBodyBlk = currentOp.getBlock();
  size_t nInputs = currentOp.getNumDpsInputs();
  size_t rCounter = 0, cCounter = 0;
  DenseMap<Value, std::string> gExprVarNames;
  DenseMap<OpResult, std::string> reductionVarNames;
  for (size_t i = 0; i < nInputs; ++i) {
    Value operand = currentOp.getOperand(i);
    BlockArgument blkArg = genericBodyBlk->getArgument(i);
    if (prodRedResults.contains(operand)) {
      // This operand is produced by one of the producing reductions. Name it "r{i}".
      auto rName = "r" + std::to_string(rCounter++);
      gExprVarNames[blkArg] = rName;
      // Map from the producer reduction result to the variable name. This will be useful when we
      // build a program from the h-expression later.
      reductionVarNames[cast<OpResult>(operand)] = rName;
    } else {
      // This operand is not produced by the reductions. Name it "c{i}".
      gExprVarNames[blkArg] = "c" + std::to_string(cCounter++);
    }
  }

  // Step 4. Extract the g expression from reduceOperand upwards.
  auto gExpr = serializeMLIRExprToJSON(match->reduceOperand, gExprVarNames, currentOp);
  if (failed(gExpr))
    return failure();
  // Step 5. Similarly extract the f expression from yieldValue upwards (which should stop soon
  // because there is only one operation to extract)
  DenseMap<Value, std::string> fExprVarNames{
      {match->accumulatorArg, "acc"},
      {match->reduceOperand, "x"},
  };
  auto fExpr = serializeMLIRExprToJSON(match->yieldValue, fExprVarNames, currentOp);
  if (failed(fExpr))
    return failure();

  return RollingUpdateSolverInput{
      .reductionVars = std::move(reductionVarNames),
      .accumulatorVar = "acc",
      .fExpr = std::move(*fExpr),
      .gExpr = std::move(*gExpr),
  };
}

/// Make an elemwise linalg.generic op that applies the repair term found by the solver.
FailureOr<GenericOp> buildLinalgFromRepairTerm(RewriterBase &rewriter, const json::Value &hExpr,
                                               GenericOp sourceReduce,
                                               const RollingUpdateSolverInput &solverInput) {
  auto loc = sourceReduce.getLoc();
  if (sourceReduce.getNumResults() != 1)
    return failure();

  // Deserialize the h-expression into a sequence of MLIR scalar operations, dumped into a scratch
  // block, then go back to the old insertion point.
  Block scratchBlock;
  DeserializedValueExpr deserialized;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&scratchBlock);
    auto deserializedR = deserializeMLIRExprFromJSON(hExpr, rewriter, loc);
    if (failed(deserializedR))
      return failure();
    deserialized = *std::move(deserializedR);
  }

  // Start building a linalg.generic around the scalar code in the scratch block.
  // We need to find the input operands to use for this generic.
  auto *sourceInitOperand = sourceReduce.getDpsInitOperand(0);
  if (!sourceInitOperand)
    return failure();
  Value outputTensor = sourceInitOperand->get();
  llvm::StringMap<Value> inputBindingsByName;
  inputBindingsByName[solverInput.accumulatorVar] = outputTensor;
  for (const auto &[opResult, varName] : solverInput.reductionVars) {
    auto producerOp = dyn_cast<DestinationStyleOpInterface>(opResult.getDefiningOp());
    if (!producerOp)
      return failure();
    // Map the variable for the new result (with a prime, like r0') to `opResult`,
    // This prime thing is a convension assumed by the solver.
    inputBindingsByName[varName + "'"] = opResult;
    // and map the variable for the old result (e.g. r0) to the init value of the producer
    // reduction, which is the "previous iteration" value of this reduction.
    auto initValue = producerOp.getDpsInitOperand(opResult.getResultNumber());
    inputBindingsByName[varName] = initValue->get();
  }

  // Start building the linalg.generic. Get a list of inputs sorted by variable names.
  auto sortedVars =
      llvm::to_vector(llvm::map_range(deserialized.variablesByName, [](const auto &it) {
        return std::make_pair(it.getKey().str(), it.getValue());
      }));
  llvm::sort(sortedVars, [](const auto &a, const auto &b) { return a.first < b.first; });
  // Look up each input in `inputBindingsByName`.
  SmallVector<Value> inputTensors;
  inputTensors.reserve(sortedVars.size());
  for (const auto &[name, _] : sortedVars) {
    auto it = inputBindingsByName.find(name);
    if (it == inputBindingsByName.end()) {
      llvm::errs() << "no tensor binding provided for symbolic variable `" << name << "`\n";
      return failure();
    }
    inputTensors.push_back(it->second);
  }
  // We also need indexing maps. We're assuming this op will be elemwise, so all the maps are just
  // identities. N for inputs and one for output.
  auto sourceOutType = dyn_cast<RankedTensorType>(sourceReduce.getResult(0).getType());
  if (!sourceOutType)
    return failure();
  unsigned outRank = sourceOutType.getRank();
  AffineMap identityMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());
  SmallVector<AffineMap> indexingMaps(sortedVars.size() + 1, identityMap);
  SmallVector<utils::IteratorType> iteratorTypes(sortedVars.size(), utils::IteratorType::parallel);

  return linalg::GenericOp::create(
      rewriter, loc, TypeRange{outputTensor.getType()}, inputTensors, ValueRange{outputTensor},
      indexingMaps, iteratorTypes, [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
        IRMapping mapping;
        for (auto [pair, arg] :
             llvm::zip_equal(sortedVars, newArgs.take_front(inputTensors.size())))
          mapping.map(pair.second, arg);
        for (Operation &op : scratchBlock)
          builder.clone(op, mapping);
        Value mappedResult = mapping.lookupOrDefault(deserialized.result);
        linalg::YieldOp::create(builder, nestedLoc, mappedResult);
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
  auto redDimOrF = matchUnarySingleReductionGeneric(thisRed);
  if (failed(redDimOrF))
    BAIL("expected reduce to be a unary single-reduction linalg.generic");

  // Fuse the reduce operation into the loop nest, changing its input from `elemwiseOrig` to
  // `elemwiseSidecars`. This function takes `elemwiseOrig`, `outerLoop`, etc. by reference,
  // and updates them to point to new operations.
  auto fuseResult = fuseReduceInLoopNest(transform, rewriter, outerLoop, innerLoop, thisRed,
                                         llvm::zip_equal(elemwiseOrig, elemwiseSidecars));
  if (!fuseResult.succeeded())
    return fuseResult;
  if (failed(foldRewriteTensorExtractInserts(rewriter, *outerLoop)))
    BAIL("failed to apply merge consecutive insert/extract_slice patterns");

  // Extract scalar expressions that describe the reduction and its producers, then send them to the
  // solver to get a repair term (h-expression).
  rewriter.setInsertionPointAfter(thisRed);
  auto solverInputR = extractRepairInputExprs(rewriter, producerReds, thisRed, elemwiseSidecars);
  if (failed(solverInputR))
    BAIL("failed to extract reducer/elemwise expressions from the program");
  auto redVars = llvm::to_vector(
      llvm::map_range(solverInputR->reductionVars, [](const auto &it) { return it.second; }));
  auto hExpr =
      solveRollingUpdaterWithPython(solverInputR->fExpr, solverInputR->gExpr, redVars, "acc");
  if (!hExpr)
    BAIL("failed to solve rolling updater with Python: " + llvm::toString(hExpr.takeError()));

  // Build a new linalg.generic that applies the repair term.
  auto repairUpdateOp = buildLinalgFromRepairTerm(rewriter, *hExpr, thisRed, *solverInputR);
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
