#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#define BAIL(message) return emitSilenceableFailure(transform, message)

using namespace mlir;

namespace mlir::transform {

namespace {

bool dependsOn(OpFoldResult value, Value ancestor) {
  auto dynamicValue = dyn_cast<Value>(value);
  if (!dynamicValue)
    return false;
  SmallVector<Value> worklist{dynamicValue};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (current == ancestor)
      return true;
    if (!visited.insert(current).second)
      continue;
    if (Operation *definingOp = current.getDefiningOp())
      llvm::append_range(worklist, definingOp->getOperands());
  }
  return false;
}

/// The destination subset published by one forall worker in one for iteration.
/// `varyingDimension` identifies the offset driven by the outer for IV, if any:
/// `[worker, iv] [1, 1]` varies in dimension 1, while `[worker, 0] [1, 16]`
/// is invariant. Sizes and strides are required to be invariant.
struct WorkerRegion {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  std::optional<size_t> varyingDimension;
};

/// Copy a parallel publication's subset and classify its dependence on the outer scf.for IV.
FailureOr<WorkerRegion> deriveWorkerRegion(tensor::ParallelInsertSliceOp publication,
                                           Value forInduction) {
  auto emitSubsetError = [&](OpFoldResult expression,
                             StringRef message) -> FailureOr<WorkerRegion> {
    Operation *anchor = publication;
    if (auto value = dyn_cast<Value>(expression); value && value.getDefiningOp())
      anchor = value.getDefiningOp();
    return anchor->emitError(message);
  };

  SmallVector<OpFoldResult> offsets = llvm::to_vector(publication.getMixedOffsets());
  SmallVector<OpFoldResult> sizes = llvm::to_vector(publication.getMixedSizes());
  SmallVector<OpFoldResult> strides = llvm::to_vector(publication.getMixedStrides());
  std::optional<size_t> varyingDimension;
  for (auto [dimension, offset, size, stride] : llvm::enumerate(offsets, sizes, strides)) {
    if (dependsOn(size, forInduction))
      return emitSubsetError(size, "expected publication size to be invariant under the scf.for");
    if (dependsOn(stride, forInduction))
      return emitSubsetError(stride,
                             "expected publication stride to be invariant under the scf.for");
    if (!dependsOn(offset, forInduction))
      continue;
    if (varyingDimension)
      return emitSubsetError(offset,
                             "expected at most one publication dimension to vary with the scf.for");
    if (dyn_cast<Value>(offset) != forInduction)
      return emitSubsetError(offset, "expected a varying publication offset to be exactly the "
                                     "scf.for induction variable");
    if (!isConstantIntValue(size, 1))
      return emitSubsetError(size, "expected a publication varying with the scf.for to have size "
                                   "one in that dimension");
    varyingDimension = dimension;
  }
  return WorkerRegion{.offsets = std::move(offsets),
                      .sizes = std::move(sizes),
                      .strides = std::move(strides),
                      .varyingDimension = varyingDimension};
}

/// Connects one outer for-carried value to its inner forall publication:
///
///   for init/iter_arg/result -> forall result -> parallel_insert(publishedValue)
///
/// The worker region determines the tile or slab that will replace the full
/// tensor iter_arg after interchange.
struct LoopCarriedResultPlan {
  BlockArgument forIterArg;
  Value forInit;
  tensor::ParallelInsertSliceOp publication;
  /// The point published by one worker in one sequential iteration.
  WorkerRegion publicationRegion;
  SmallVector<tensor::ExtractSliceOp> recurrenceReads;
  SmallVector<tensor::ExtractSliceOp> destinationReads;
  Value destinationInit;
};

/// Build one complete carry plan, including all subset reads that will be
/// replaced during cloning. This keeps validation and use discovery in one
/// traversal before payload mutation begins.
FailureOr<LoopCarriedResultPlan> buildLoopCarriedResultPlan(scf::ForOp forLoop,
                                                            scf::ForallOp forallLoop,
                                                            OpResult forResult,
                                                            const LoopResultRelay &forallRelay) {
  auto publication = dyn_cast_or_null<tensor::ParallelInsertSliceOp>(forallRelay.mediator);
  if (!publication) {
    forallLoop.emitError("expected every relayed scf.forall result to have one "
                         "tensor.parallel_insert_slice combining op");
    return failure();
  }
  auto publicationRegion = deriveWorkerRegion(publication, forLoop.getInductionVar());
  if (failed(publicationRegion))
    return failure();

  auto hasSameRegion = [](tensor::ExtractSliceOp slice, tensor::ParallelInsertSliceOp publication) {
    auto sliceI = cast<OffsetSizeAndStrideOpInterface>(slice.getOperation());
    auto publicationI = cast<OffsetSizeAndStrideOpInterface>(publication.getOperation());
    return publicationI.isSameAs(sliceI, isEqualConstantIntOrValue);
  };
  auto collectReads = [&](Value tensor, Operation *allowedUser,
                          StringRef userName) -> FailureOr<SmallVector<tensor::ExtractSliceOp>> {
    SmallVector<tensor::ExtractSliceOp> reads;
    for (OpOperand &use : tensor.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == allowedUser)
        continue;
      auto slice = dyn_cast<tensor::ExtractSliceOp>(owner);
      if (!slice || slice.getSource() != tensor || !hasSameRegion(slice, publication))
        return owner->emitError() << "expected the " << userName
                                  << " to be used through a direct extract_slice matching its "
                                     "publication region: ";
      reads.push_back(slice);
    }
    return reads;
  };

  unsigned resultNumber = forResult.getResultNumber();
  BlockArgument forIterArg = forLoop.getRegionIterArgs()[resultNumber];
  auto recurrenceReads = collectReads(forIterArg, forallLoop, "scf.for iter_arg");
  if (failed(recurrenceReads))
    return failure();

  BlockArgument destination = cast<BlockArgument>(publication.getDest());
  auto destinationReads = collectReads(destination, publication, "scf.forall destination");
  if (failed(destinationReads))
    return failure();

  Value destinationInit = forallLoop.getTiedOpOperand(destination)->get();
  if (destinationInit != forIterArg) {
    Operation *parentOp = destinationInit.getParentBlock()->getParentOp();
    if (parentOp == forLoop.getOperation() || forLoop->isAncestor(parentOp)) {
      publication.emitError("expected a distinct forall scratch destination to be defined "
                            "outside the scf.for");
      return failure();
    }
  }
  return LoopCarriedResultPlan{
      .forIterArg = forIterArg,
      .forInit = forLoop.getInitArgs()[resultNumber],
      .publication = publication,
      .publicationRegion = std::move(*publicationRegion),
      .recurrenceReads = std::move(*recurrenceReads),
      .destinationReads = std::move(*destinationReads),
      .destinationInit = destinationInit,
  };
}

struct MaterializedWorkerCarry {
  Value workerInit;
  /// A worker-local view of a distinct forall scratch destination, if any.
  Value workerDestinationInit;
  /// The full-tensor subset occupied by this worker's tile or sequential slab.
  WorkerRegion workerRegion;
};

struct OuterForallScaffold {
  scf::ForallOp forallLoop;
  Range normalizedForRange;
  IRMapping mapping;
  /// Ordered identically to the analyzed loop-carried result plans.
  SmallVector<MaterializedWorkerCarry> workerCarries;
};

/// Create the outer worker loop and extract each worker's initial tile or slab.
OuterForallScaffold createOuterForall(RewriterBase &rewriter, scf::ForOp forLoop,
                                      scf::ForallOp forallLoop,
                                      ArrayRef<LoopCarriedResultPlan> resultPlans) {
  Location loc = forallLoop.getLoc();
  rewriter.setInsertionPoint(forLoop);
  Range normalizedForRange = emitNormalizedLoopBounds(rewriter, loc, forLoop.getLowerBound(),
                                                      forLoop.getUpperBound(), forLoop.getStep());

  SmallVector<Value> outputs = llvm::map_to_vector(
      resultPlans, [](const LoopCarriedResultPlan &plan) { return plan.forInit; });
  auto newForall = scf::ForallOp::create(rewriter, loc, forallLoop.getMixedLowerBound(),
                                         forallLoop.getMixedUpperBound(), forallLoop.getMixedStep(),
                                         outputs, forallLoop.getMapping());

  IRMapping mapping;
  mapping.map(forallLoop.getOperation(), newForall.getOperation());
  mapping.map(forallLoop.getInductionVars(), newForall.getInductionVars());

  auto makeWorkerIndexAvailable = [&](OpFoldResult value) -> OpFoldResult {
    auto dynamicValue = dyn_cast<Value>(value);
    if (!dynamicValue)
      return value;
    auto available = makeValuesAvailableAtInsertionPoint(rewriter, {dynamicValue}, mapping,
                                                         DefChainAction::Clone);
    assert(succeeded(available) && "cloning a definition chain cannot fail");
    return available->front();
  };

  SmallVector<MaterializedWorkerCarry> workerCarries;
  workerCarries.reserve(resultPlans.size());
  rewriter.setInsertionPoint(newForall.getTerminator());
  for (const LoopCarriedResultPlan &plan : resultPlans) {
    WorkerRegion workerRegion = plan.publicationRegion;
    for (auto [dimension, offset, size, stride] :
         llvm::enumerate(workerRegion.offsets, workerRegion.sizes, workerRegion.strides)) {
      if (plan.publicationRegion.varyingDimension == dimension) {
        offset = forLoop.getLowerBound();
        size = normalizedForRange.size;
        stride = forLoop.getStep();
      } else {
        offset = makeWorkerIndexAvailable(offset);
        size = makeWorkerIndexAvailable(size);
        stride = makeWorkerIndexAvailable(stride);
      }
    }
    Value workerInit =
        createExtractSliceFromState(rewriter, loc, plan.forInit, workerRegion.offsets,
                                    workerRegion.sizes, workerRegion.strides);
    Value workerDestinationInit;
    if (plan.destinationInit != plan.forIterArg) {
      workerDestinationInit =
          createExtractSliceFromState(rewriter, loc, plan.destinationInit, workerRegion.offsets,
                                      workerRegion.sizes, workerRegion.strides);
    }
    workerCarries.push_back(MaterializedWorkerCarry{.workerInit = workerInit,
                                                    .workerDestinationInit = workerDestinationInit,
                                                    .workerRegion = std::move(workerRegion)});
  }
  return OuterForallScaffold{.forallLoop = newForall,
                             .normalizedForRange = normalizedForRange,
                             .mapping = std::move(mapping),
                             .workerCarries = std::move(workerCarries)};
}

struct InnerForScaffold {
  scf::ForOp forLoop;
  IRMapping mapping;
  /// Ordered identically to the analyzed loop-carried result plans.
  SmallVector<BlockArgument> workerIterArgs;
};

/// Create the normalized sequential loop. The old full-tensor iter_args are
/// deliberately left unmapped until their direct subset reads are localized.
InnerForScaffold createInnerFor(RewriterBase &rewriter, scf::ForOp oldFor,
                                const OuterForallScaffold &outerForall) {
  Location loc = oldFor.getLoc();
  scf::ForallOp newForall = outerForall.forallLoop;
  rewriter.setInsertionPoint(newForall.getTerminator());
  Value lowerBound =
      getValueOrCreateConstantIndexOp(rewriter, loc, outerForall.normalizedForRange.offset);
  Value upperBound =
      getValueOrCreateConstantIndexOp(rewriter, loc, outerForall.normalizedForRange.size);
  Value step =
      getValueOrCreateConstantIndexOp(rewriter, loc, outerForall.normalizedForRange.stride);
  SmallVector<Value> initArgs =
      llvm::map_to_vector(outerForall.workerCarries,
                          [](const MaterializedWorkerCarry &carry) { return carry.workerInit; });
  auto newFor = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step, initArgs);
  rewriter.setInsertionPointToEnd(newFor.getBody());
  scf::YieldOp::create(rewriter, loc, newFor.getRegionIterArgs());

  rewriter.setInsertionPointToStart(newFor.getBody());
  AffineExpr normalizedIv, lowerBoundExpr, stepExpr;
  bindDims(rewriter.getContext(), normalizedIv);
  bindSymbols(rewriter.getContext(), lowerBoundExpr, stepExpr);
  OpFoldResult logicalInduction = affine::makeComposedFoldedAffineApply(
      rewriter, loc, normalizedIv * stepExpr + lowerBoundExpr,
      {newFor.getInductionVar(), oldFor.getLowerBound(), oldFor.getStep()});
  Value logicalInductionValue = getValueOrCreateConstantIndexOp(rewriter, loc, logicalInduction);

  IRMapping mapping = outerForall.mapping;
  mapping.map(oldFor.getOperation(), newFor.getOperation());
  mapping.map(oldFor.getInductionVar(), logicalInductionValue);
  return InnerForScaffold{.forLoop = newFor,
                          .mapping = std::move(mapping),
                          .workerIterArgs = llvm::to_vector(newFor.getRegionIterArgs())};
}

/// Build the worker-local region read or written by one original scf.for iteration. A "point" is
/// one position along the sequentially varying dimension; it may still be a multidimensional tile
/// in the other dimensions.
///
/// After interchange, a worker slab holds every sequential point for one scf.forall worker. `%k`
/// is the normalized IV of the rebuilt inner scf.for (`0 .. trip_count`), with the original IV
/// reconstructed as `%t = lower_bound + %k * step`. The returned region describes both:
///
///   %point = tensor.extract_slice %slab[0, %k] [1, 1] [1, 1]
///   %next = tensor.insert_slice %updated into %slab[0, %k] [1, 1] [1, 1]
///
/// Invariant publications do not need a point region because their worker-local tile is reused by
/// every sequential iteration.
WorkerRegion getLocalPointRegion(RewriterBase &rewriter, const WorkerRegion &publicationRegion,
                                 Value normalizedInduction, const IRMapping &mapping) {
  auto remapFoldResult = [&mapping](OpFoldResult value) {
    auto dynamicValue = dyn_cast<Value>(value);
    return dynamicValue ? OpFoldResult(mapping.lookupOrDefault(dynamicValue)) : value;
  };
  SmallVector<OpFoldResult> offsets(publicationRegion.offsets.size(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = llvm::map_to_vector(publicationRegion.sizes, remapFoldResult);
  SmallVector<OpFoldResult> strides =
      llvm::map_to_vector(publicationRegion.strides, remapFoldResult);
  size_t varyingDim = *publicationRegion.varyingDimension;
  offsets[varyingDim] = normalizedInduction;
  strides[varyingDim] = rewriter.getIndexAttr(1);
  return WorkerRegion{.offsets = std::move(offsets),
                      .sizes = std::move(sizes),
                      .strides = std::move(strides),
                      .varyingDimension = varyingDim};
}

void cloneAndRemapReads(RewriterBase &rewriter, scf::ForallOp oldForall,
                        ArrayRef<LoopCarriedResultPlan> resultPlans,
                        const OuterForallScaffold &outerForall, InnerForScaffold &innerFor) {
  rewriter.setInsertionPoint(innerFor.forLoop.getBody()->getTerminator());
  Value innerIV = innerFor.forLoop.getInductionVar();
  auto createLocalPoint = [&](Value localTensor, const WorkerRegion &publicationRegion) {
    if (!publicationRegion.varyingDimension)
      return localTensor;
    WorkerRegion pointRegion =
        getLocalPointRegion(rewriter, publicationRegion, innerIV, innerFor.mapping);
    return createExtractSliceFromState(rewriter, oldForall.getLoc(), localTensor,
                                       pointRegion.offsets, pointRegion.sizes, pointRegion.strides);
  };
  auto mapPlannedReads = [&](ArrayRef<tensor::ExtractSliceOp> reads, Value localPoint,
                             Value localTensor) {
    Operation *localPointOp = localPoint == localTensor ? nullptr : localPoint.getDefiningOp();
    for (tensor::ExtractSliceOp read : reads) {
      if (localPointOp)
        innerFor.mapping.map(read.getOperation(), localPointOp);
      innerFor.mapping.map(read.getResult(), localPoint);
    }
  };

  for (auto [plan, carry, iterArg] :
       llvm::zip_equal(resultPlans, outerForall.workerCarries, innerFor.workerIterArgs)) {
    Value recurrencePoint;
    if (!plan.recurrenceReads.empty()) {
      recurrencePoint = createLocalPoint(iterArg, plan.publicationRegion);
      mapPlannedReads(plan.recurrenceReads, recurrencePoint, iterArg);
    }

    if (plan.destinationReads.empty())
      continue;
    Value localDestination = carry.workerDestinationInit ? carry.workerDestinationInit : iterArg;
    Value destinationPoint = localDestination == iterArg && recurrencePoint
                                 ? recurrencePoint
                                 : createLocalPoint(localDestination, plan.publicationRegion);
    mapPlannedReads(plan.destinationReads, destinationPoint, localDestination);
  }
}

void cloneNormalComputationOps(RewriterBase &rewriter, scf::ForallOp oldForall,
                               InnerForScaffold &innerFor) {
  rewriter.setInsertionPoint(innerFor.forLoop.getBody()->getTerminator());
  for (Operation &op : oldForall.getBody()->without_terminator()) {
    if (innerFor.mapping.lookupOrNull(&op))
      continue;
    if (op.getNumResults() != 0 && llvm::all_of(op.getResults(), [&](Value result) {
          return innerFor.mapping.contains(result);
        }))
      continue;
    rewriter.clone(op, innerFor.mapping);
  }
}

void cloneAndRemapWritesAndForYield(RewriterBase &rewriter, scf::ForOp oldFor,
                                    ArrayRef<LoopCarriedResultPlan> resultPlans,
                                    InnerForScaffold &innerFor) {
  rewriter.setInsertionPoint(innerFor.forLoop.getBody()->getTerminator());
  SmallVector<Value> nextCarries;
  nextCarries.reserve(resultPlans.size());
  for (auto [plan, iterArg] : llvm::zip_equal(resultPlans, innerFor.workerIterArgs)) {
    tensor::ParallelInsertSliceOp oldPublication = plan.publication;
    Value publishedPoint = innerFor.mapping.lookupOrDefault(oldPublication.getSource());
    if (!plan.publicationRegion.varyingDimension) {
      nextCarries.push_back(publishedPoint);
      continue;
    }
    WorkerRegion pointRegion = getLocalPointRegion(
        rewriter, plan.publicationRegion, innerFor.forLoop.getInductionVar(), innerFor.mapping);
    nextCarries.push_back(
        tensor::InsertSliceOp::create(rewriter, oldPublication.getLoc(), publishedPoint, iterArg,
                                      pointRegion.offsets, pointRegion.sizes, pointRegion.strides));
  }

  auto oldYield = cast<scf::YieldOp>(oldFor.getBody()->getTerminator());
  auto newYield = cast<scf::YieldOp>(innerFor.forLoop.getBody()->getTerminator());
  rewriter.setInsertionPoint(newYield);
  auto replacementYield = scf::YieldOp::create(rewriter, oldYield.getLoc(), nextCarries);
  innerFor.mapping.map(oldYield.getOperation(), replacementYield.getOperation());
  rewriter.eraseOp(newYield);
}

void cloneAndRemapParallelInsertSlices(RewriterBase &rewriter, scf::ForallOp oldForall,
                                       ArrayRef<LoopCarriedResultPlan> resultPlans,
                                       const OuterForallScaffold &outerForall,
                                       InnerForScaffold &innerFor) {
  scf::ForallOp newForall = outerForall.forallLoop;
  pointBuilderToForallParallel(rewriter, newForall);
  for (auto [index, plan, carry] : llvm::enumerate(resultPlans, outerForall.workerCarries)) {
    tensor::ParallelInsertSliceOp oldPublication = plan.publication;
    auto publication = tensor::ParallelInsertSliceOp::create(
        rewriter, oldPublication.getLoc(), innerFor.forLoop.getResult(index),
        newForall.getRegionOutArgs()[index], carry.workerRegion.offsets, carry.workerRegion.sizes,
        carry.workerRegion.strides);
    innerFor.mapping.map(oldPublication.getOperation(), publication.getOperation());
  }
  innerFor.mapping.map(oldForall.getTerminator().getOperation(),
                       newForall.getTerminator().getOperation());
}

} // namespace

void ScfInterchangeForAndForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getForLoopMutable(), effects);
  onlyReadsHandle(getForallLoopMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure ScfInterchangeForAndForallOp::apply(TransformRewriter &rewriter,
                                                                TransformResults &transformResults,
                                                                TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  scf::ForOp forLoop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForLoop, "for loop", forLoop, scf::ForOp);
  scf::ForallOp forallLoop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForallLoop, "forall loop", forallLoop,
                               scf::ForallOp);
  if (forallLoop->getParentOp() != forLoop.getOperation())
    BAIL("expected the scf.forall to be directly nested in the scf.for");
  Block *forBody = forLoop.getBody();
  if (&forBody->front() != forallLoop.getOperation() ||
      forallLoop->getNextNode() != forBody->getTerminator())
    BAIL("expected a perfect nest with the scf.forall as the only non-terminator operation in "
         "the scf.for");
  if (!forLoop.getInductionVar().getType().isIndex())
    BAIL("expected the scf.for induction variable to have index type");
  std::optional<int64_t> step = getConstantIntValue(OpFoldResult(forLoop.getStep()));
  if (!step || *step <= 0)
    BAIL("expected the scf.for step to be a positive constant");

  FailureOr<DenseMap<OpResult, LoopResultRelaysT>> chainedResultMap =
      getChainedLoopResultMap({forLoop, forallLoop});
  if (failed(chainedResultMap) || chainedResultMap->size() != forLoop.getNumResults())
    BAIL("failed to build loop result relay chains");
  DenseSet<OpResult> relayedForallResults;
  for (const LoopResultRelaysT &relays : llvm::make_second_range(*chainedResultMap)) {
    assert(relays.size() == 2);
    if (!relayedForallResults.insert(relays.front().loopReturnResult).second)
      BAIL("expected every scf.forall result to relay to exactly one scf.for result");
  }
  if (relayedForallResults.size() != forallLoop.getNumResults())
    BAIL("expected every scf.forall result to relay to exactly one scf.for result");

  SmallVector<LoopCarriedResultPlan, 2> recurrencePlans;
  recurrencePlans.reserve(chainedResultMap->size());
  for (const auto &[forResult, relays] : *chainedResultMap) {
    auto resultPlan = buildLoopCarriedResultPlan(forLoop, forallLoop, forResult, relays.front());
    if (failed(resultPlan))
      BAIL("failed to build a localizable loop-carried result plan");
    recurrencePlans.push_back(std::move(*resultPlan));
  }
  llvm::sort(recurrencePlans,
             [](const LoopCarriedResultPlan &lhs, const LoopCarriedResultPlan &rhs) {
               return lhs.forIterArg.getArgNumber() < rhs.forIterArg.getArgNumber();
             });

  // Step 1: Clone the scf.forall as the new outer loop and materialize worker-local initial values.
  OuterForallScaffold outerForall =
      createOuterForall(rewriter, forLoop, forallLoop, recurrencePlans);
  // Step 2: Clone the scf.for as the normalized inner recurrence loop.
  InnerForScaffold innerFor = createInnerFor(rewriter, forLoop, outerForall);
  // Step 3: Clone and remap full-tensor reads to worker-local tiles or points. The rebuilt scf.for
  // carries only these local values, so the old full-tensor iter_args remain intentionally
  // unmapped.
  cloneAndRemapReads(rewriter, forallLoop, recurrencePlans, outerForall, innerFor);
  // Step 4: Clone the normal computation between the localized reads and publications.
  cloneNormalComputationOps(rewriter, forallLoop, innerFor);
  // Step 5: Clone and remap worker writes into the scf.for yield.
  cloneAndRemapWritesAndForYield(rewriter, forLoop, recurrencePlans, innerFor);
  // Step 6: Clone and remap final worker publications into the outer scf.forall.
  cloneAndRemapParallelInsertSlices(rewriter, forallLoop, recurrencePlans, outerForall, innerFor);

  SmallVector<std::pair<Operation *, Operation *>> clonedBodyOps;
  for (Operation &oldOp : forallLoop.getBody()->without_terminator())
    if (Operation *newOp = innerFor.mapping.lookupOrNull(&oldOp))
      clonedBodyOps.emplace_back(&oldOp, newOp);
  notifyClonedOpsRecursively(rewriter, clonedBodyOps);
  if (failed(rewriter.notifyPayloadOperationReplaced(forallLoop, outerForall.forallLoop)))
    BAIL("failed to track the rebuilt scf.forall");
  if (failed(rewriter.notifyPayloadOperationReplaced(forLoop, innerFor.forLoop)))
    BAIL("failed to track the rebuilt scf.for");
  rewriter.replaceOp(forLoop, outerForall.forallLoop.getResults());

  transformResults.set(getOperation()->getResult(0), {outerForall.forallLoop.getOperation()});
  transformResults.set(getOperation()->getResult(1), {innerFor.forLoop.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
