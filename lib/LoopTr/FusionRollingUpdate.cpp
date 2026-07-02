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
#include "llvm/ADT/ScopeExit.h"
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

static FailureOr<AffineMap> dropDomainDim(AffineMap map, unsigned droppedDim) {
  MLIRContext *ctx = map.getContext();
  unsigned oldNumDims = map.getNumDims();
  if (droppedDim >= oldNumDims)
    return failure();

  // Verify dropped dim was not referenced.
  for (AffineExpr expr : map.getResults()) {
    if (expr.isFunctionOfDim(droppedDim))
      return failure();
  }
  // First create a (d0, d1, ..., d{n-1}) vector, then insert a placeholder `0` at `droppedDim`.
  SmallVector<AffineExpr> dimRepls = llvm::to_vector(llvm::map_range(
      llvm::index_range(0, oldNumDims), [&](size_t i) { return getAffineDimExpr(i, ctx); }));
  dimRepls.insert(dimRepls.begin() + droppedDim, getAffineConstantExpr(0, ctx));

  return map.replaceDimsAndSymbols(dimRepls, map.getResults(), oldNumDims - 1, map.getNumSymbols());
}

struct LinalgProvenance {
  OpResult tileValue;
  AffineMap indexMap;
  std::string varName;
};

// TODO: better name for this struct. Not all fields are used by the solver only. `reductionVars`
// provides tensor-value provenance for scalar variables in the g expression, which is used when
// reconstructing a linalg.generic from the h-expression after the solver is done.
struct RollingUpdateSolverInput {
  SmallVector<LinalgProvenance> varProvenances;
  std::string accVarName;
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
    // Erase intermediate fusion result -- we don't need them and they take up space in the IR.
    if (currentOp != thisRed)
      rewriter.eraseOp(currentOp);
    reductionResult = cast<OpResult>(it->second);
    currentOp = cast<GenericOp>(fusionResult->fusedOp);
  }
  auto scopeGuard = llvm::scope_exit([&]() {
    if (currentOp != thisRed)
      rewriter.eraseOp(currentOp);
  });

  // Step 2. Check this currentOp is a reduction, and get some information about it.
  auto match = matchBinaryReductionCombiner(currentOp, reductionResult.getResultNumber(),
                                            /*emitDiagnostics=*/true);
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
  SmallVector<LinalgProvenance> reductionVars;
  for (size_t i = 0; i < nInputs; ++i) {
    Value operand = currentOp.getOperand(i);
    AffineMap indexMap = currentOp.getIndexingMapsArray()[i];
    BlockArgument blkArg = genericBodyBlk->getArgument(i);
    if (prodRedResults.contains(operand)) {
      // This operand is produced by one of the producing reductions. Name it "r{i}".
      auto rName = "r" + std::to_string(rCounter++);
      gExprVarNames[blkArg] = rName;
      // Map from the producer reduction result to the variable name. This will be useful when we
      // build a program from the h-expression later.
      reductionVars.emplace_back(cast<OpResult>(operand), indexMap, rName);
    } else {
      // This operand is not produced by the reductions. Name it "c{i}".
      gExprVarNames[blkArg] = "c" + std::to_string(cCounter++);
    }
  }

  // Step 4. Extract the g expression from reduceOperand upwards.
  auto gExpr = serializeMLIRExprToJSON(match->nonAccumulator, gExprVarNames, currentOp);
  if (failed(gExpr)) {
    currentOp->emitRemark("this is the compute operation we're extracting from");
    return failure();
  }
  // Step 5. Similarly extract the f expression from yieldValue upwards (which should stop soon
  // because there is only one operation to extract)
  static const std::string accVarName = "acc";
  DenseMap<Value, std::string> fExprVarNames{
      {match->accumulatorArg, accVarName},
      {match->nonAccumulator, "x"},
  };
  auto fExpr = serializeMLIRExprToJSON(match->yieldedValue, fExprVarNames, currentOp);
  if (failed(fExpr))
    return failure();

  return RollingUpdateSolverInput{
      .varProvenances = std::move(reductionVars),
      .accVarName = accVarName,
      .fExpr = std::move(*fExpr),
      .gExpr = std::move(*gExpr),
  };
}

/// Make an elemwise linalg.generic op that applies the repair term found by the solver.
FailureOr<GenericOp> buildLinalgFromRepairTerm(RewriterBase &rewriter, const json::Value &hExpr,
                                               GenericOp sourceReduce, size_t reduceDim,
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

  auto dpsInit = sourceReduce.getDpsInitOperand(0)->get();
  // Since we checked there is only one result, this is the indexing map for the output tensor.
  auto resultMap = sourceReduce.getIndexingMapsArray()[sourceReduce.getNumDpsInputs() + 0];

  // Start building a linalg.generic around the scalar code in the scratch block.
  // We need to find the input operands to use for this generic.
  llvm::StringMap<std::pair<Value, AffineMap>> varNameToValue;
  varNameToValue[solverInput.accVarName] = {dpsInit, resultMap};
  for (const auto &[opResult, sourceMap, varName] : solverInput.varProvenances) {
    auto producerOp = dyn_cast<DestinationStyleOpInterface>(opResult.getDefiningOp());
    if (!producerOp)
      return failure();
    // Map the variable for the new result (with a prime, like r0') to `opResult`,
    // This prime thing is a convension assumed by the solver.
    varNameToValue[varName + "'"] = {opResult, sourceMap};
    // and map the variable for the old result (e.g. r0) to the init value of the producer
    // reduction, which is the "previous iteration" value of this reduction.
    auto initValue = producerOp.getDpsInitOperand(opResult.getResultNumber());
    varNameToValue[varName] = {initValue->get(), sourceMap};
  }

  // Get a list of scalar variables for inputs, sorted by variable name.
  auto sortedVars =
      llvm::to_vector(llvm::map_range(deserialized.variablesByName, [](const auto &it) {
        return std::make_pair(it.getKey().str(), it.getValue());
      }));
  llvm::sort(sortedVars, [](const auto &a, const auto &b) { return a.first < b.first; });
  // Look up each variable name in `varNameToProv`, and build a list of operands for the linalg op,
  // and their corresponding indexing maps.
  SmallVector<Value> inputTensors;
  inputTensors.reserve(sortedVars.size());
  SmallVector<AffineMap> indexingMaps;
  indexingMaps.reserve(sortedVars.size() + 1);
  for (const auto &[name, _] : sortedVars) {
    auto it = varNameToValue.find(name);
    if (it == varNameToValue.end()) {
      llvm::errs() << "no tensor binding provided for symbolic variable `" << name << "`\n";
      return failure();
    }
    inputTensors.push_back(it->second.first);
    indexingMaps.push_back(it->second.second);
  }
  // One more for the DPS init operand.
  indexingMaps.push_back(resultMap);
  // Drop the reduction dimension from the domain of each indexing map, since we're creating an
  // elemwise op here.
  for (auto &map : indexingMaps) {
    auto trimmedIndexMap = dropDomainDim(map, reduceDim);
    if (failed(trimmedIndexMap))
      return failure();
    map = *trimmedIndexMap;
  }

  // This op is elementwise, so all iterators are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(indexingMaps.back().getNumDims(),
                                                 utils::IteratorType::parallel);
  return linalg::GenericOp::create(
      rewriter, loc, TypeRange{dpsInit.getType()}, inputTensors, ValueRange{dpsInit}, indexingMaps,
      iteratorTypes, [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
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
  rewriter.setInsertionPointAfter(thisRed);
  auto solverInputR = extractRepairInputExprs(rewriter, producerReds, thisRed, elemwiseSidecars);
  if (failed(solverInputR))
    BAIL("failed to extract reducer/elemwise expressions from the program");
  auto redVars = llvm::to_vector(llvm::map_range(
      solverInputR->varProvenances, [](const LinalgProvenance &prov) { return prov.varName; }));
  auto hExpr =
      solveRollingUpdaterWithPython(solverInputR->fExpr, solverInputR->gExpr, redVars, "acc");
  if (!hExpr)
    BAIL("failed to solve rolling updater with Python: " + llvm::toString(hExpr.takeError()));

  // Build a new linalg.generic that applies the repair term.
  auto repairUpdateOp =
      buildLinalgFromRepairTerm(rewriter, *hExpr, thisRed, *redDimR, *solverInputR);
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
