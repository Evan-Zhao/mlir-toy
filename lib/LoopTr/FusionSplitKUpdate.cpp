#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

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

FailureOr<DenseMap<OpResult, ForallResultRelay>>
mapWriteBackResultsToTilesInLoop(scf::ForallOp loop, size_t nOldResults, auto &&pairs) {
  // Connect loop results to their "in-loop tile" results, which are results produced by ops
  // in the loop body. These are published via tensor.parallel_insert_slice ops for scf.forall
  // loops, so we find these too.
  auto loopResultMap = getChainedLoopResultMap({loop});
  if (failed(loopResultMap))
    return failure();
  // Traverse loopResultMap to (1) ensure each `mediator` is a tensor.parallel_insert_slice, and
  // (2) make a reverse map keyed by in-loop tile results.
  DenseMap<OpResult, ForallResultRelay> loopResultToRelay, tileToRelay;
  for (const auto &[loopResult, relays] : *loopResultMap) {
    // Skip results that are not from the original forall loop.
    if (loopResult.getResultNumber() >= nOldResults)
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

  // Pair write-back and rfactor reductions.
  // Take loopResultToRelay and replace rfactor-produced loop results with writeback results.
  // Then later we can use this opResultToRelay as a subst map.
  DenseMap<OpResult, ForallResultRelay> opResultToRelay = std::move(loopResultToRelay);
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
        auto newInitVal = cloneValueDefChainAtInsertionPoint(rewriter, initVal, localMapping);
        if (failed(newInitVal)) {
          ::emitRemark(initVal.getLoc()) << "when cloning this value and its use-def chain up";
          BAIL("failed to move the DPS init operand of an elementwise op before the forall loop");
        }
        if (*newInitVal != initVal) {
          rewriter.replaceAllUsesWith(initVal, *newInitVal);
          rewriter.eraseOp(initVal.getDefiningOp());
        }
        newOutArgs.push_back(*newInitVal);
      }
    }
  }
  auto newForall =
      scf::ForallOp::create(rewriter, forallLoop.getLoc(), forallLoop.getMixedLowerBound(),
                            forallLoop.getMixedUpperBound(), forallLoop.getMixedStep(), newOutArgs,
                            forallLoop.getMapping());
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
  // Map old forall induction vars and region args to the new forall.
  IRMapping mapping;
  mapping.map(forallLoop.getInductionVars(), newForall.getInductionVars());
  mapping.map(forallLoop.getRegionOutArgs(), newForall.getRegionOutArgs().take_front(nOldResults));
  // Clone loop body ops.
  rewriter.setInsertionPointToStart(newForall.getBody());
  auto clonedBodyOps = cloneBlockWithoutTerminator(rewriter, *forallLoop.getBody(), mapping);
  // Clone the terminator (tensor.parallel_insert_slice ops).
  pointRewriterToForallParallel(rewriter, newForall);
  for (Operation &oldCombiningOp : forallLoop.getTerminator())
    rewriter.clone(oldCombiningOp, mapping);
  // Map rfactor ops from the old loop to the new loop.
  for (auto &op : rfactorOps) {
    if (op->getParentOp() != forallLoop) {
      op->emitRemark() << "this rfactor reduction op";
      BAIL("expected rfactor reduction to be inside the forall loop");
    }
    op = mapping.lookup(op);
  }
  // Notify the rewriter of op replacements.
  for (auto [oldOp, newOp] : clonedBodyOps)
    auto _ = rewriter.notifyPayloadOperationReplaced(oldOp, newOp);
  if (failed(rewriter.notifyPayloadOperationReplaced(forallLoop, newForall.getOperation())))
    BAIL("failed to preserve the scf.forall handle");
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
    pointRewriterToForallParallel(rewriter, newForall);
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

DiagnosedSilenceableFailure FusionRepairRfactorReductionFrontierOp::apply(
    transform::TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  (void)rewriter;
  (void)transformResults;
  (void)state;
  BAIL("transform.fusion.repair_rfactor_reduction_frontier is declared but not implemented yet");
}

} // namespace mlir::transform
