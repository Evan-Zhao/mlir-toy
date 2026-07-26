#include "LoopTr/FusionExprSolver.h"
#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/PartialReduction.h"
#include "LoopTr/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include <llvm/ADT/ArrayRef.h>

namespace {

#define BAIL(message) return emitSilenceableFailure(transform, message);

using namespace mlir;

using scf::ForallOp;
using tensor::ParallelInsertSliceOp;
using transform::TransformOpInterface;

struct ForallResultRelay {
  OpResult inLoopResult;
  OpResult loopReturnResult;
  ParallelInsertSliceOp insertSlice;
  // Overrides the offsets and sizes of the insertSlice.
  SmallVector<OpFoldResult> offsets, sizes;
};

using RelayMap = DenseMap<OpResult, ForallResultRelay>;

SmallVector<OpFoldResult> dropAt(SmallVector<OpFoldResult> values, unsigned index) {
  values.erase(values.begin() + index);
  return values;
}

FailureOr<uint64_t> getSingleReductionDim(Operation *op) {
  if (auto generic = dyn_cast<linalg::GenericOp>(op))
    return getReductionIteratorIndex(generic);
  if (auto reduce = dyn_cast<linalg::ReduceOp>(op)) {
    ArrayRef<int64_t> dimensions = reduce.getDimensions();
    if (dimensions.size() != 1 || dimensions.front() < 0)
      return failure();
    return static_cast<uint64_t>(dimensions.front());
  }
  return failure();
}

// A specialized version of getChainedLoopResultMap that only works for a single forall loop.
// Compared to getChainedLoopResultMap, it always check that the mediator is a
// tensor.parallel_insert_slice op.
// Returns a pair of maps, first one keyed by loop results, second one by in-loop (tile) results.
FailureOr<std::pair<RelayMap, RelayMap>>
getForallLoopResultMaps(scf::ForallOp loop,
                        std::optional<size_t> takeFirstNResults = std::nullopt) {
  auto loopResultMap = getChainedLoopResultMap({loop});
  if (failed(loopResultMap))
    return failure();
  // Traverse loopResultMap to (1) ensure each `mediator` is a tensor.parallel_insert_slice, and
  // (2) make a reverse map keyed by in-loop tile results.
  DenseMap<OpResult, ForallResultRelay> loopResultToRelay, tileToRelay;
  for (const auto &[loopResult, relays] : *loopResultMap) {
    // Skip results that are beyond the specified range.
    if (takeFirstNResults && loopResult.getResultNumber() >= takeFirstNResults)
      continue;
    assert(relays.size() == 1 && "expected exactly one relay for each forall loop result");
    auto &relay = relays.front();
    auto parallelInsert = dyn_cast_if_present<tensor::ParallelInsertSliceOp>(relay.mediator);
    if (!parallelInsert)
      return loop.emitError() << "forall result #" << loopResult.getResultNumber()
                              << " is not published via a tensor.parallel_insert_slice";
    auto forallRelay =
        ForallResultRelay{relay.inLoopResult, loopResult, parallelInsert,
                          parallelInsert.getMixedOffsets(), parallelInsert.getMixedSizes()};
    loopResultToRelay[loopResult] = tileToRelay[relay.inLoopResult] = forallRelay;
  }
  return std::make_pair(std::move(loopResultToRelay), std::move(tileToRelay));
}

FailureOr<DenseMap<OpResult, ForallResultRelay>>
mapWriteBackResultsToTilesInLoop(scf::ForallOp loop, size_t nOldResults, auto &&pairs) {
  // Connect loop results to their "in-loop tile" results.
  auto loopResultMaps = getForallLoopResultMaps(loop, nOldResults);
  if (failed(loopResultMaps))
    return failure();
  // Pair write-back and rfactor reductions.
  // Start from the forward map `opResultToRelay` and replace rfactor-produced loop results with
  // writeback results. Then later we can use this opResultToRelay as a subst map.
  auto [opResultToRelay, tileToRelay] = std::move(*loopResultMaps);
  for (auto [wbOp, rfOp] : pairs) {
    size_t nWbResults = wbOp->getNumResults(), nRfResults = rfOp->getNumResults();
    if (nWbResults != nRfResults) {
      wbOp->emitError() << "this write-back reduction has different number of results ("
                        << nWbResults << " results) from its rfactor counterpart";
      return rfOp->emitError() << "this rfactor reduction has " << nRfResults << " results";
    }
    FailureOr<uint64_t> reductionDim = getSingleReductionDim(wbOp);
    if (failed(reductionDim))
      return wbOp->emitError()
             << "expected write-back reduction to have exactly one reduction dimension";
    for (size_t i = 0; i < nWbResults; ++i) {
      OpResult wbResult = wbOp->getResult(i), rfResult = rfOp->getResult(i);
      auto relayIt = tileToRelay.find(rfResult);
      if (relayIt == tileToRelay.end())
        return rfOp->emitError() << "result #" << i
                                 << " of this rfactor op is not published to a forall loop result";
      auto relay = relayIt->second;
      relay.offsets = dropAt(relay.offsets, static_cast<unsigned>(*reductionDim));
      relay.sizes = dropAt(relay.sizes, static_cast<unsigned>(*reductionDim));
      opResultToRelay[wbResult] = relay;
      // Remove the loop result produced by the rfactor from the map.
      opResultToRelay.erase(relay.loopReturnResult);
    }
  }
  return opResultToRelay;
}

struct ElemwiseOpOperandInfo {
  SmallVector<unsigned> operandNumbers;
  SmallVector<SmallVector<OpFoldResult>> offsets;
  SmallVector<SmallVector<OpFoldResult>> sizes;
  SmallVector<Value> replacements;
};

FailureOr<ElemwiseOpOperandInfo>
collectOperandTiles(linalg::LinalgOp elemwiseOp,
                    const DenseMap<OpResult, ForallResultRelay> &opResultToRelay) {
  ElemwiseOpOperandInfo input;
  for (OpOperand *operand : elemwiseOp.getDpsInputOperands()) {
    auto relayIt = opResultToRelay.find(dyn_cast<OpResult>(operand->get()));
    if (relayIt == opResultToRelay.end()) {
      elemwiseOp.emitError() << "failed to get in-loop tile for operand #"
                             << operand->getOperandNumber();
      return failure();
    }
    const ForallResultRelay &relay = relayIt->second;
    input.operandNumbers.push_back(operand->getOperandNumber());
    input.offsets.push_back(relay.offsets);
    input.sizes.push_back(relay.sizes);
    input.replacements.push_back(relay.inLoopResult);
  }
  return input;
}

void patchTiledOpInputs(RewriterBase &rewriter, linalg::LinalgOp tiledOp,
                        const ElemwiseOpOperandInfo &operandInfo) {
  for (auto [opndNum, replacement] :
       llvm::zip_equal(operandInfo.operandNumbers, operandInfo.replacements)) {
    rewriter.modifyOpInPlace(tiledOp, [&]() { tiledOp->setOperand(opndNum, replacement); });
  }
}

FailureOr<SmallVector<tensor::ExtractSliceOp>>
patchTiledOpDpsInits(RewriterBase &rewriter, linalg::LinalgOp tiledOp,
                     ArrayRef<BlockArgument> loopOutArgs) {
  rewriter.setInsertionPoint(tiledOp);
  SmallVector<tensor::ExtractSliceOp> slices;
  for (auto [idx, outArg] : llvm::enumerate(tiledOp.getDpsInitsMutable())) {
    auto oldOutSlice = outArg.get().getDefiningOp<tensor::ExtractSliceOp>();
    if (!oldOutSlice) {
      tiledOp.emitError() << "expected DPS init argument #" << idx
                          << " of this op to be defined by a tensor.extract_slice";
      return failure();
    }
    auto newOutSlice = tensor::ExtractSliceOp::create(
        rewriter, oldOutSlice.getLoc(), oldOutSlice.getType(), loopOutArgs[idx],
        oldOutSlice.getMixedOffsets(), oldOutSlice.getMixedSizes(), oldOutSlice.getMixedStrides());
    rewriter.modifyOpInPlace(tiledOp, [&]() { outArg.set(newOutSlice); });
    slices.push_back(newOutSlice);
  }
  return slices;
}

LogicalResult isElemwiseLinalgOp(Operation *op) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic)
    return failure();
  return linalg::isElementwise(generic) ? success() : failure();
}

FailureOr<IRMapping> matchElemwisePairsGetSubstMap(ArrayRef<Operation *> outLoopOps,
                                                   scf::ForallOp loop,
                                                   ArrayRef<Operation *> inLoopOps,
                                                   const RelayMap &tileToLoopRelay) {
  // This isn't a good place to emit a diagnostic for this error (we don't have the handle).
  // It's recommended that the caller checks for equal length.
  if (inLoopOps.size() != outLoopOps.size())
    return failure();

  // Check that all the in-loop and out-loop ops are valid ops (such as elementwise linalg op), and
  // that the in-loop ops are directly inside the loop.
  // Check that each in-loop op has the same number of results as its corresponding out-loop op.
  // Finally pair them and build a subst map.
  IRMapping mapping;
  for (auto [inOp, outOp] : llvm::zip_equal(inLoopOps, outLoopOps)) {
    if (failed(isElemwiseLinalgOp(inOp)))
      return inOp->emitError() << "expected this op to be an elementwise linalg op";
    if (failed(isElemwiseLinalgOp(outOp)))
      return outOp->emitError() << "expected this op to be an elementwise linalg op";
    if (inOp->getParentOp() != loop)
      return inOp->emitError() << "expected this op to be directly inside the loop";

    size_t nInResults = inOp->getNumResults(), nOutResults = outOp->getNumResults();
    if (nInResults != nOutResults) {
      inOp->emitError() << "this in-loop sidecar op has " << nInResults << " result(s)";
      return outOp->emitError() << "this out-loop op has " << nOutResults
                                << " result(s), expected the same number of results";
    }
    for (size_t idx = 0; idx < inOp->getNumResults(); ++idx) {
      auto it = tileToLoopRelay.find(inOp->getResult(idx));
      if (it == tileToLoopRelay.end())
        return inOp->emitError() << "result #" << idx
                                 << " of this in-loop op is not published to a loop result";
      mapping.map(outOp->getResult(idx), it->second.loopReturnResult);
    }
  }
  return mapping;
}

} // namespace

namespace mlir::transform {

void FusionCloneFuseRfactorElemwiseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getElemwiseChainOpsMutable(), effects);
  onlyReadsHandle(getForallLoopMutable(), effects);
  onlyReadsHandle(getWritebackReduceOpsMutable(), effects);
  onlyReadsHandle(getRfactorReduceOpsMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
FusionCloneFuseRfactorElemwiseOp::apply(transform::TransformRewriter &rewriter,
                                        TransformResults &transformResults, TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());

  // Step 1. Get and validate input ops.
  ForallOp forallLoop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForallLoop, "forall loop", forallLoop,
                               ForallOp);
  SmallVector<linalg::LinalgOp> elemwiseOps;
  for (auto op : state.getPayloadOps(getElemwiseChainOps())) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp || failed(isSingleOutputElemwiseLinalgOp(op))) {
      op->emitError() << "expected an elementwise linalg op";
      BAIL("expected every elementwise op to be a single-result elementwise linalg op");
    }
    elemwiseOps.push_back(linalgOp);
  }
  CHECK_NON_EMPTY_OPS(state, transform, getWritebackReduceOps, "write-back reduction",
                      writebackOps);
  CHECK_NON_EMPTY_OPS(state, transform, getRfactorReduceOps, "r-factor reduction", rfactorOps);
  if (writebackOps.size() != rfactorOps.size())
    BAIL("expected the same number of write-back and rfactor reduction ops");

  // Step 2. Make a new forall loop to hold all the results of all elemwise ops to be fused.
  // We may not end up needing that many, but it's easier to only clone the loop once,
  // and scf.forall knows how to remove unused results later.
  size_t nOldResults = forallLoop.getNumResults();
  SmallVector<Value> newOutArgs = forallLoop.getOutputs();
  rewriter.setInsertionPoint(forallLoop);
  {
    // Move DPS init operands of the elementwise ops before the forall loop, so the new forall loop
    // can take them as init values.
    IRMapping localMapping;
    for (auto elemwiseOp : elemwiseOps) {
      for (auto initVal : elemwiseOp.getDpsInits()) {
        auto movedValues = makeValuesAvailableAtInsertionPoint(rewriter, {initVal}, localMapping,
                                                               DefChainAction::Move);
        if (failed(movedValues)) {
          ::emitRemark(initVal.getLoc()) << "when cloning this value and its use-def chain up";
          BAIL("failed to move the DPS init operand of an elementwise op before the forall loop");
        }
        newOutArgs.push_back(movedValues->front());
      }
    }
  }
  rewriter.setInsertionPoint(forallLoop);
  ForallOutputExtension extension = cloneForallWithAppendedOutputs(
      rewriter, forallLoop, ValueRange(newOutArgs).drop_front(nOldResults));
  scf::ForallOp newForall = extension.forall;
  // Collect the region arguments of the new forall that correspond to the outputs of the
  // elementwise ops. We'll need that when we make the DPS init operands of the fused elemwise ops.
  SmallVector<SmallVector<BlockArgument>> loopOutArgsByElemwise;
  {
    auto regionArgs = newForall.getRegionOutArgs().drop_front(nOldResults);
    for (auto elemwiseOp : elemwiseOps) {
      size_t nResults = elemwiseOp.getNumDpsInits();
      auto opRegionArgs = regionArgs.take_front(nResults);
      loopOutArgsByElemwise.push_back(SmallVector<BlockArgument>(opRegionArgs));
      regionArgs = regionArgs.drop_front(nResults);
    }
  }
  IRMapping mapping = std::move(extension.mapping);
  // Map rfactor ops from the old loop to the new loop.
  for (auto &op : rfactorOps) {
    if (op->getParentOp() != forallLoop) {
      op->emitRemark() << "this rfactor reduction op";
      BAIL("expected rfactor reduction to be inside the forall loop");
    }
    Operation *newOp = mapping.lookup(op);
    if (failed(rewriter.notifyPayloadOperationReplaced(op, newOp)))
      BAIL("failed to preserve the rfactor reduction handle");
    op = newOp;
  }
  // Replace uses of the old forall results with new forall results, then erase the old loop.
  rewriter.replaceOp(forallLoop, newForall.getResults().take_front(nOldResults));

  // Step 3. Make a map from the results of the write-back reductions to operations in the loop body
  // that produce the corresponding tiles.
  auto opResultToRelayR = mapWriteBackResultsToTilesInLoop(
      newForall, nOldResults, llvm::zip_equal(writebackOps, rfactorOps));
  if (failed(opResultToRelayR))
    BAIL("failed to map write-back reduction results to in-loop tile-producing ops");
  auto opResultToRelay = std::move(*opResultToRelayR);

  // Step 3. Start fusing the elementwise ops.
  SmallVector<Operation *> sidecarOps;
  for (auto [idx, elemwiseOp] : enumerate(elemwiseOps)) {
    // Collect tile information for the operands of the elementwise op, such as offsets and sizes.
    auto operandInfo = collectOperandTiles(elemwiseOp, opResultToRelay);
    if (failed(operandInfo))
      BAIL("failed to collect operand tiles for elementwise op");
    auto tilingOp = cast<TilingInterface>(elemwiseOp.getOperation());

    // Set the insertion point inside the loop body, and use getTiledImplementationFromOperandTiles
    // to create a tiled version of the elementwise op.
    rewriter.setInsertionPoint(newForall.getTerminator());
    auto tilingResult = tilingOp.getTiledImplementationFromOperandTiles(
        rewriter, operandInfo->operandNumbers, operandInfo->offsets, operandInfo->sizes);
    if (failed(tilingResult))
      BAIL("failed to create tiled implementation of elementwise op");
    assert(tilingResult->tiledOps.size() == 1 && "expected exactly one tiled op");
    auto tiledLinalgOp = cast<linalg::LinalgOp>(tilingResult->tiledOps.front());

    // That tiling method is rather mechanical. Since tilingOp uses newForall results, the method
    // created slices over loop results, but since we're in the loop body, that is surely illegal.
    // We already have the input values for the tiled op, so we can just apply these values.
    patchTiledOpInputs(rewriter, tiledLinalgOp, *operandInfo);
    // Then for the DPS init operands, we'll need to slice the loop's block arguments and feed those
    // slices to the tiled op.
    auto &blockArgs = loopOutArgsByElemwise[idx];
    auto slices = patchTiledOpDpsInits(rewriter, tiledLinalgOp, blockArgs);
    if (failed(slices))
      BAIL("failed to patch tiled elementwise output operand");

    // Add parallel_insert_slice ops to the loop for these new results that tiledLinalgOp produces.
    // Also add these results to opResultToRelay.
    pointBuilderToForallParallel(rewriter, newForall);
    for (size_t i = 0; i < tiledLinalgOp->getNumResults(); ++i) {
      auto sliceOp = (*slices)[i];
      auto tileResult = tiledLinalgOp->getResult(i);
      auto insertOp = tensor::ParallelInsertSliceOp::create(
          rewriter, newForall.getLoc(), tileResult, blockArgs[i], sliceOp.getMixedOffsets(),
          sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
      auto loopResult = newForall.getTiedOpResult(newForall.getTiedOpOperand(blockArgs[i]));
      auto relay = ForallResultRelay{tileResult, loopResult, insertOp, sliceOp.getMixedOffsets(),
                                     sliceOp.getMixedSizes()};
      opResultToRelay[elemwiseOp->getResult(i)] = relay;
    }

    // Remove slice ops created by the tiling method (remember they are illegal).
    for (Operation *slice : tilingResult->generatedSlices) {
      if (!slice->use_empty()) {
        slice->emitError() << "expected generated slice to be unused after patching tiled op";
        BAIL("failed to remove temporary tiled operand slice");
      }
      rewriter.eraseOp(slice);
    }
    sidecarOps.push_back(tiledLinalgOp);
  }

  transformResults.set(getOperation()->getResult(0), sidecarOps);
  return DiagnosedSilenceableFailure::success();
}

void FusionRepairRfactorReductionFrontierOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getReducesWbMutable(), effects);
  onlyReadsHandle(getReducesRfMutable(), effects);
  consumesHandle(getThisReduceMutable(), effects);
  onlyReadsHandle(getElemwiseOrigMutable(), effects);
  onlyReadsHandle(getElemwiseSidecarsMutable(), effects);
  onlyReadsHandle(getForallLoopMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
FusionRepairRfactorReductionFrontierOp::apply(transform::TransformRewriter &rewriter,
                                              TransformResults &transformResults,
                                              TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());

  // Step 1. Validate the forall loop; get a relay map from in-loop (tiled) ops to loop's return
  // results.
  ForallOp forallLoop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForallLoop, "forall loop", forallLoop,
                               ForallOp);
  auto loopRelayResult = getForallLoopResultMaps(forallLoop);
  if (failed(loopRelayResult))
    BAIL("failed to get loop result map for the forall loop");
  auto &tileToLoopRelay = loopRelayResult->second;

  // Step 2. Validate the elementwise ops; match the original and sidecar elementwise ops, and build
  // a subst map from the results of the original ops to the loop result produced by sidecar ops.
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseOrig, "original elementwise", elemwiseOrig);
  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseSidecars, "sidecar elementwise",
                      elemwiseSidecars);
  auto substMapping =
      matchElemwisePairsGetSubstMap(elemwiseOrig, forallLoop, elemwiseSidecars, tileToLoopRelay);
  if (failed(substMapping))
    BAIL("failed to match original and sidecar elementwise ops and their results");

  // Step 3. Validate the rfactor / writeback reductions and the "this reduction" op.
  CHECK_NON_EMPTY_OPS(state, transform, getReducesWb, "producer write-back reductions", reducesWb);
  CHECK_NON_EMPTY_OPS(state, transform, getReducesRf, "producer rfactor reductions", reducesRf);
  if (reducesWb.size() != reducesRf.size())
    BAIL("expected the same number of producer write-back and rfactor reductions");
  linalg::GenericOp thisReduce;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getThisReduce, "this reduction", thisReduce,
                               linalg::GenericOp);
  auto thisReducePR = dyn_cast<PartialReductionOpInterface>(thisReduce.getOperation());
  if (!thisReducePR)
    BAIL("expected this reduction to implement PartialReductionOpInterface");

  // Step 4. Remap the reduction under `substMapping` to use loop results produced by sidecars.
  {
    rewriter.setInsertionPoint(forallLoop);
    auto clonedReduce = cast<linalg::GenericOp>(rewriter.clone(*thisReduce, *substMapping));
    rewriter.replaceOp(thisReduce, clonedReduce);
    thisReduce = clonedReduce;
  }
  // Move DPS init operands of clonedReduce before the forall loop.
  if (failed(recursiveMoveOperandsBeforeOp(*thisReduce, rewriter, *forallLoop)))
    BAIL("failed to move staged reduction operands before the forall loop");
  // Now we can detect a split plan for the reduction once it uses loop outputs.
  auto splitPlan = detectReductionForallSplit(transform, forallLoop, thisReduce);
  if (failed(splitPlan))
    BAIL("failed to detect a split plan for the reduction");

  // Step 5. Fuse the reduction op into the forall loop while r-factoring it, using the same logic
  // as in ScfFusePartialReductionIntoForallOp.
  auto rfactorResult =
      rFactorReductionUnderForall(transform, rewriter, forallLoop, thisReduce, *splitPlan);
  if (failed(rfactorResult))
    BAIL("failed to split the forall loop for the reduction");
  // Update reducesRf and elemwiseSidecars to point to the cloned ops inside the new forall loop.
  auto &clonedOps = rfactorResult->clonedOps;
  DenseMap<Operation *, Operation *> clonedOpMap(clonedOps.begin(), clonedOps.end());
  auto updateOps = [&clonedOpMap, &rewriter](MutableArrayRef<Operation *> ops) -> LogicalResult {
    for (auto &op : ops) {
      auto it = clonedOpMap.find(op);
      if (it == clonedOpMap.end())
        return op->emitError() << "failed to find this op in the cloned forall loop";
      if (failed(rewriter.notifyPayloadOperationReplaced(op, it->second)))
        return op->emitError() << "failed to preserve the transform handle for this cloned op";
      op = cast<linalg::GenericOp>(it->second);
    }
    return success();
  };
  if (failed(updateOps(reducesRf)) || failed(updateOps(elemwiseSidecars)))
    BAIL("failed to remap producer reductions into the rebuilt forall");

  // Step 6. Call the repair expression solver to get the H expression.
  auto repairExpr =
      solveFusionRepairExpr(rewriter, reducesRf, rfactorResult->rFactorOp, elemwiseSidecars);
  if (failed(repairExpr))
    BAIL("failed to solve rolling updater expressions");

  // Step 7. Start collecting values we want to feed `repairExpr` to build a new linalg.generic op.
  // Find an insertion point, which needs to be after all the producer writeback reductions.
  Operation *insertAfter = reducesWb.front();
  for (Operation *wbOp : reducesWb) {
    if (wbOp->getBlock() != insertAfter->getBlock())
      BAIL("expected producer writeback reductions to be in the same block");
    if (insertAfter->isBeforeInBlock(wbOp))
      insertAfter = wbOp;
  }
  rewriter.setInsertionPointAfter(insertAfter);

  // Collect the "panel" (full tensor) for accumulator and reducesRf operations.
  // Call `getForallLoopResultMaps` again (simplest way to update the relay map).
  loopRelayResult = getForallLoopResultMaps(rfactorResult->newForall);
  if (failed(loopRelayResult))
    BAIL("failed to get loop result map for the rebuilt forall loop");
  DenseMap<OpResult, OpResult> rfactorTensorByTile;
  for (const auto &[tileResult, relay] : loopRelayResult->second)
    rfactorTensorByTile[tileResult] = relay.loopReturnResult;
  auto it = rfactorTensorByTile.find(rfactorResult->rFactorOp->getResult(0));
  if (it == rfactorTensorByTile.end())
    BAIL("failed to find the loop result for the repair accumulator tensor");
  auto accTensor = it->second;
  SmallVector<FusionRepairReductionBinding> repairBindings;
  for (auto [rfOp, wbOp] : llvm::zip_equal(reducesRf, reducesWb)) {
    for (auto [rfResult, wbResult] : llvm::zip_equal(rfOp->getResults(), wbOp->getResults())) {
      it = rfactorTensorByTile.find(rfResult);
      if (it == rfactorTensorByTile.end())
        BAIL("failed to find the loop result for a repair reduction tensor");
      repairBindings.emplace_back(rfResult, it->second, wbResult);
    }
  }

  // Step 8. Build a new linalg.generic that applies the repair term in split-k update mode.
  auto repairUpdateOp =
      repairExpr->build(rewriter, rfactorResult->rFactorOp.getLoc(), repairBindings, accTensor,
                        FusionRepairTermMode::SplitKUpdate, splitPlan->opRedDim);
  if (failed(repairUpdateOp))
    BAIL("failed to build linalg.generic around the h-expression returned by the solver");

  // Step 9. Update the writeback op of "this reduction" to use the calculated repair term.
  IRMapping repairedWritebackMapping;
  repairedWritebackMapping.map(rfactorResult->writebackOp.getDpsInputOperand(0)->get(),
                               repairUpdateOp->getResult(0));
  rewriter.setInsertionPointAfter(*repairUpdateOp);
  auto repairedWritebackOp =
      cast<linalg::ReduceOp>(rewriter.clone(*rfactorResult->writebackOp, repairedWritebackMapping));
  rewriter.replaceOp(rfactorResult->writebackOp, repairedWritebackOp);

  rewriter.replaceOp(thisReduce, repairedWritebackOp);
  rewriter.replaceOp(forallLoop,
                     rfactorResult->newForall.getResults().take_front(forallLoop.getNumResults()));
  transformResults.set(getOperation()->getResult(0), {rfactorResult->rFactorOp.getOperation()});
  transformResults.set(getOperation()->getResult(1), {repairedWritebackOp});
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
