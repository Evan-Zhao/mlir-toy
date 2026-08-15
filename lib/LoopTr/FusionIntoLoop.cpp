#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>

#define DEBUG_TYPE "fusion-into-loop"

namespace mlir::transform {

#define BAIL(message) return emitSilenceableFailure(transform, message);

void FusionGreedyConsumersIntoProducerOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getProducerLoopMutable(), effects);
  onlyReadsHandle(getStopOpMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

enum class TensorRegionTransferDirection : uint8_t { Pullback, Pushforward };

enum class TensorValueTransform : uint8_t {
  Identity,
  ExtractSlice,
  InsertSlice,
  CollapseShape,
  ExpandShape
};

/// A rectangular strided region expressed in the coordinates of `tensor`.
/// Region arrays always have the rank of the full tensor, even when the local
/// tile value is rank-reduced.
struct StridedTensorRegion {
  Value tensor;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};

/// Describes the value transformations that accompany a forall publication:
///
///   %new_init = <initTransform> %old_init
///   scf.forall shared_outs(%new_arg = %new_init) {
///     %old_arg = <blockArgumentTransform> %new_arg
///   }
///
/// The structural executor decides where to materialize these values.
struct ForallDestMaterialization {
  TensorValueTransform initTransform;
  TensorValueTransform blockArgumentTransform;
};

/// Describes the coordinate transfer and the value transformations needed at
/// its boundaries. `localTransform` maps a source-coordinate tile to a
/// result-coordinate tile; slice composition can make this the identity.
struct TensorRegionTransfer {
  StridedTensorRegion mappedRegion;
  TensorValueTransform localTransform;
  std::optional<ForallDestMaterialization> forallDestination;
};

static void printTensorRegion(raw_ostream &os, const StridedTensorRegion &region) {
  auto printFoldResults = [&](StringRef label, ArrayRef<OpFoldResult> values) {
    os << label << " = [";
    llvm::interleaveComma(values, os, [&](OpFoldResult value) {
      if (auto dynamic = dyn_cast<Value>(value))
        dynamic.printAsOperand(os, OpPrintingFlags());
      else
        os << cast<Attribute>(value);
    });
    os << "]";
  };

  os << "tensor = ";
  region.tensor.printAsOperand(os, OpPrintingFlags());
  os << ", ";
  printFoldResults("offsets", region.offsets);
  os << ", ";
  printFoldResults("sizes", region.sizes);
  os << ", ";
  printFoldResults("strides", region.strides);
}

static StringRef getTensorValueTransformName(TensorValueTransform transform) {
  switch (transform) {
  case TensorValueTransform::Identity:
    return "identity";
  case TensorValueTransform::ExtractSlice:
    return "tensor.extract_slice";
  case TensorValueTransform::InsertSlice:
    return "tensor.insert_slice";
  case TensorValueTransform::CollapseShape:
    return "tensor.collapse_shape";
  case TensorValueTransform::ExpandShape:
    return "tensor.expand_shape";
  }
  llvm_unreachable("unknown tensor value transform");
}

static void printTensorRegionTransfer(raw_ostream &os, Operation *shapeOp,
                                      TensorRegionTransferDirection direction,
                                      const StridedTensorRegion &knownRegion,
                                      const TensorRegionTransfer &transfer) {
  os << "tensor region transfer through " << shapeOp->getName() << " ("
     << (direction == TensorRegionTransferDirection::Pullback ? "pullback" : "pushforward")
     << ")\n  known:  ";
  printTensorRegion(os, knownRegion);
  os << "\n  mapped: ";
  printTensorRegion(os, transfer.mappedRegion);
  os << "\n  local transform: " << getTensorValueTransformName(transfer.localTransform);
  if (transfer.forallDestination) {
    os << "\n  forall init transform: "
       << getTensorValueTransformName(transfer.forallDestination->initTransform)
       << "\n  forall block argument transform: "
       << getTensorValueTransformName(transfer.forallDestination->blockArgumentTransform);
  }
  os << "\n";
}

static bool hasValidRegionRank(const StridedTensorRegion &region) {
  auto type = dyn_cast<RankedTensorType>(region.tensor.getType());
  return type && static_cast<int64_t>(region.offsets.size()) == type.getRank() &&
         region.sizes.size() == region.offsets.size() &&
         region.strides.size() == region.offsets.size();
}

static FailureOr<StridedTensorRegion> composeSliceRegions(
    RewriterBase &rewriter, Location loc, Value mappedTensor, ArrayRef<OpFoldResult> outerOffsets,
    ArrayRef<OpFoldResult> outerSizes, ArrayRef<OpFoldResult> outerStrides,
    const llvm::SmallBitVector &outerDroppedDims, const StridedTensorRegion &innerRegion) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  if (failed(affine::mergeOffsetsSizesAndStrides(
          rewriter, loc, outerOffsets, outerSizes, outerStrides, outerDroppedDims,
          innerRegion.offsets, innerRegion.sizes, innerRegion.strides, offsets, sizes, strides)))
    return failure();
  return StridedTensorRegion{mappedTensor, std::move(offsets), std::move(sizes),
                             std::move(strides)};
}

static FailureOr<StridedTensorRegion> mapReshapeRegion(RewriterBase &rewriter, Location loc,
                                                       const StridedTensorRegion &region,
                                                       Value mappedTensor,
                                                       ArrayRef<ReassociationIndices> reassociation,
                                                       bool collapseRegion) {
  if (!hasValidRegionRank(region))
    return failure();

  OpBuilder::InsertionGuard guard(rewriter);
  auto equivalentSlice = tensor::ExtractSliceOp::create(
      rewriter, loc, region.tensor, region.offsets, region.sizes, region.strides);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  LogicalResult mapped =
      collapseRegion ? tensor::getCollapsedExtractSliceInfo(rewriter, equivalentSlice,
                                                            reassociation, offsets, sizes, strides)
                     : tensor::getExpandedExtractSliceInfo(rewriter, equivalentSlice, reassociation,
                                                           mappedTensor, offsets, sizes, strides);
  rewriter.eraseOp(equivalentSlice);
  if (failed(mapped))
    return failure();
  return StridedTensorRegion{mappedTensor, std::move(offsets), std::move(sizes),
                             std::move(strides)};
}

/// Transfer one rectangular region across a tensor shape operation without
/// rewriting that operation or its surrounding loop. The two piecewise cases
/// are intentionally excluded: pushing a region through extract_slice and
/// pulling a region through insert_slice.
static FailureOr<TensorRegionTransfer> transferTensorRegion(
    RewriterBase &rewriter, Operation *shapeOp, TensorRegionTransferDirection direction,
    const StridedTensorRegion &knownRegion, std::optional<unsigned> operandNumber = std::nullopt) {
  if (!hasValidRegionRank(knownRegion))
    return failure();

  auto succeed = [&](StridedTensorRegion mappedRegion, TensorValueTransform localTransform,
                     std::optional<ForallDestMaterialization> forallDestination =
                         std::nullopt) -> FailureOr<TensorRegionTransfer> {
    TensorRegionTransfer transfer{std::move(mappedRegion), localTransform, forallDestination};
    LLVM_DEBUG(printTensorRegionTransfer(llvm::dbgs(), shapeOp, direction, knownRegion, transfer));
    return transfer;
  };

  if (auto extract = dyn_cast<tensor::ExtractSliceOp>(shapeOp)) {
    if (direction != TensorRegionTransferDirection::Pullback || operandNumber ||
        knownRegion.tensor.getType() != extract.getResultType())
      return failure();
    FailureOr<StridedTensorRegion> mapped = composeSliceRegions(
        rewriter, extract.getLoc(), extract.getSource(), extract.getMixedOffsets(),
        extract.getMixedSizes(), extract.getMixedStrides(), extract.getDroppedDims(), knownRegion);
    if (failed(mapped))
      return failure();
    return succeed(std::move(*mapped), TensorValueTransform::Identity);
  }

  if (auto insert = dyn_cast<tensor::InsertSliceOp>(shapeOp)) {
    if (direction != TensorRegionTransferDirection::Pushforward || operandNumber != 0 ||
        knownRegion.tensor.getType() != insert.getSourceType())
      return failure();
    FailureOr<StridedTensorRegion> mapped = composeSliceRegions(
        rewriter, insert.getLoc(), insert.getResult(), insert.getMixedOffsets(),
        insert.getMixedSizes(), insert.getMixedStrides(), insert.getDroppedDims(), knownRegion);
    if (failed(mapped))
      return failure();
    return succeed(std::move(*mapped), TensorValueTransform::Identity,
                   ForallDestMaterialization{TensorValueTransform::InsertSlice,
                                             TensorValueTransform::ExtractSlice});
  }

  if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(shapeOp)) {
    bool pullback = direction == TensorRegionTransferDirection::Pullback;
    if (pullback ? operandNumber.has_value() : operandNumber != 0)
      return failure();
    Type expectedType = pullback ? collapse.getResultType() : collapse.getSrcType();
    if (knownRegion.tensor.getType() != expectedType)
      return failure();
    FailureOr<StridedTensorRegion> mapped =
        mapReshapeRegion(rewriter, collapse.getLoc(), knownRegion,
                         pullback ? collapse.getSrc() : collapse.getResult(),
                         collapse.getReassociationIndices(), /*collapseRegion=*/!pullback);
    if (failed(mapped))
      return failure();
    std::optional<ForallDestMaterialization> forallDestination;
    if (!pullback)
      forallDestination = ForallDestMaterialization{TensorValueTransform::CollapseShape,
                                                    TensorValueTransform::ExpandShape};
    return succeed(std::move(*mapped), TensorValueTransform::CollapseShape, forallDestination);
  }

  if (auto expand = dyn_cast<tensor::ExpandShapeOp>(shapeOp)) {
    bool pullback = direction == TensorRegionTransferDirection::Pullback;
    if (pullback ? operandNumber.has_value() : operandNumber != 0)
      return failure();
    Type expectedType = pullback ? expand.getResultType() : expand.getSrcType();
    if (knownRegion.tensor.getType() != expectedType)
      return failure();
    FailureOr<StridedTensorRegion> mapped = mapReshapeRegion(
        rewriter, expand.getLoc(), knownRegion, pullback ? expand.getSrc() : expand.getResult(),
        expand.getReassociationIndices(), /*collapseRegion=*/pullback);
    if (failed(mapped))
      return failure();
    std::optional<ForallDestMaterialization> forallDestination;
    if (!pullback)
      forallDestination = ForallDestMaterialization{TensorValueTransform::ExpandShape,
                                                    TensorValueTransform::CollapseShape};
    return succeed(std::move(*mapped), TensorValueTransform::ExpandShape, forallDestination);
  }

  return failure();
}

static SmallVector<ReassociationIndices> getReassociation(Operation *shapeOp) {
  if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(shapeOp))
    return collapse.getReassociationIndices();
  return cast<tensor::ExpandShapeOp>(shapeOp).getReassociationIndices();
}

static Value materializeTensorValueTransform(RewriterBase &rewriter, Location loc,
                                             TensorValueTransform transform, Operation *shapeOp,
                                             Value input, RankedTensorType resultType,
                                             ArrayRef<OpFoldResult> resultShape = {}) {
  switch (transform) {
  case TensorValueTransform::Identity:
    return input;
  case TensorValueTransform::ExtractSlice: {
    auto insert = cast<tensor::InsertSliceOp>(shapeOp);
    return tensor::ExtractSliceOp::create(rewriter, loc, resultType, input,
                                          insert.getMixedOffsets(), insert.getMixedSizes(),
                                          insert.getMixedStrides())
        .getResult();
  }
  case TensorValueTransform::InsertSlice: {
    auto insert = cast<tensor::InsertSliceOp>(shapeOp);
    return tensor::InsertSliceOp::create(rewriter, loc, input, insert.getDest(),
                                         insert.getMixedOffsets(), insert.getMixedSizes(),
                                         insert.getMixedStrides())
        .getResult();
  }
  case TensorValueTransform::CollapseShape:
    return tensor::CollapseShapeOp::create(rewriter, loc, resultType, input,
                                           getReassociation(shapeOp))
        .getResult();
  case TensorValueTransform::ExpandShape:
    return tensor::ExpandShapeOp::create(rewriter, loc, resultType, input,
                                         getReassociation(shapeOp), resultShape)
        .getResult();
  }
  llvm_unreachable("unknown tensor value transform");
}

static tensor::ExtractSliceOp pullTensorProducerThroughSlice(RewriterBase &rewriter,
                                                             tensor::ExtractSliceOp slice,
                                                             Operation *shapeProducer,
                                                             const TensorRegionTransfer &transfer) {
  rewriter.setInsertionPoint(slice);
  RankedTensorType sourceTileType = slice.getResultType();
  if (transfer.localTransform != TensorValueTransform::Identity)
    sourceTileType = tensor::ExtractSliceOp::inferResultType(
        cast<RankedTensorType>(transfer.mappedRegion.tensor.getType()),
        transfer.mappedRegion.sizes);
  auto sourceTile = tensor::ExtractSliceOp::create(
      rewriter, slice.getLoc(), sourceTileType, transfer.mappedRegion.tensor,
      transfer.mappedRegion.offsets, transfer.mappedRegion.sizes, transfer.mappedRegion.strides);
  Value replacement = materializeTensorValueTransform(
      rewriter, slice.getLoc(), transfer.localTransform, shapeProducer, sourceTile,
      slice.getResultType(), slice.getMixedSizes());
  rewriter.replaceOp(slice, replacement);
  return sourceTile;
}

struct TensorConsumerFusionResult {
  scf::ForallOp loop;
  SmallVector<Operation *> fusedOps;
};

static FailureOr<TensorConsumerFusionResult> pushTensorConsumerThroughForall(RewriterBase &rewriter,
                                                                             scf::ForallOp loop,
                                                                             unsigned resultNumber,
                                                                             Operation *consumer) {
  SmallVector<OpOperand *> transferredOperands;
  for (OpOperand &operand : consumer->getOpOperands())
    if (operand.get() == loop.getResult(resultNumber))
      transferredOperands.push_back(&operand);
  if (!llvm::hasSingleElement(transferredOperands) || consumer->getNumResults() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "tensor consumer transfer requires one transferred operand and one result\n");
    return failure();
  }
  unsigned operandNumber = transferredOperands.front()->getOperandNumber();

  SmallVector<Operation *> yieldingOps;
  for (Operation &op : loop.getTerminator().getYieldingOps())
    yieldingOps.push_back(&op);
  tensor::ParallelInsertSliceOp oldInsert;
  Value selectedOutArg = loop.getRegionOutArgs()[resultNumber];
  for (Operation *yieldingOp : yieldingOps) {
    auto insert = dyn_cast<tensor::ParallelInsertSliceOp>(yieldingOp);
    if (insert && insert.getDest() == selectedOutArg) {
      if (oldInsert)
        return failure();
      oldInsert = insert;
    }
  }
  if (!oldInsert) {
    LLVM_DEBUG(llvm::dbgs() << "tensor consumer transfer found no unique publication\n");
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop.getTerminator());
  StridedTensorRegion publishedRegion{
      oldInsert.getDest(), llvm::to_vector(oldInsert.getMixedOffsets()),
      llvm::to_vector(oldInsert.getMixedSizes()), llvm::to_vector(oldInsert.getMixedStrides())};
  LLVM_DEBUG({
    llvm::dbgs() << "push tensor consumer through forall result " << resultNumber << ":\n  ";
    consumer->print(llvm::dbgs());
    llvm::dbgs() << "\n  publication: ";
    printTensorRegion(llvm::dbgs(), publishedRegion);
    llvm::dbgs() << "\n";
  });
  FailureOr<TensorRegionTransfer> transfer =
      transferTensorRegion(rewriter, consumer, TensorRegionTransferDirection::Pushforward,
                           publishedRegion, operandNumber);
  if (failed(transfer) || !transfer->forallDestination)
    return failure();

  Location loc = consumer->getLoc();
  Value oldInit = loop.getOutputs()[resultNumber];
  SmallVector<OpFoldResult> sourceShape = getMixedTensorSizes(rewriter, loc, oldInit);
  SmallVector<OpFoldResult> resultShape;
  if (auto expand = dyn_cast<tensor::ExpandShapeOp>(consumer))
    resultShape = expand.getMixedOutputShape();

  rewriter.setInsertionPoint(loop);
  auto resultType = cast<RankedTensorType>(consumer->getResult(0).getType());
  Value newInit =
      materializeTensorValueTransform(rewriter, loc, transfer->forallDestination->initTransform,
                                      consumer, oldInit, resultType, resultShape);
  SmallVector<Value> newInits(loop.getOutputs());
  newInits[resultNumber] = newInit;
  auto newLoop = scf::ForallOp::create(rewriter, loop.getLoc(), loop.getMixedLowerBound(),
                                       loop.getMixedUpperBound(), loop.getMixedStep(), newInits,
                                       loop.getMapping());
  newLoop->setAttrs(loop->getAttrs());

  IRMapping mapping;
  mapping.map(loop.getInductionVars(), newLoop.getInductionVars());
  for (auto [index, oldArg] : llvm::enumerate(loop.getRegionOutArgs())) {
    Value newArg = newLoop.getRegionOutArgs()[index];
    if (index == resultNumber) {
      rewriter.setInsertionPointToStart(newLoop.getBody());
      newArg = materializeTensorValueTransform(
          rewriter, loc, transfer->forallDestination->blockArgumentTransform, consumer, newArg,
          cast<RankedTensorType>(oldArg.getType()), sourceShape);
    }
    mapping.map(oldArg, newArg);
  }

  rewriter.setInsertionPoint(newLoop.getTerminator());
  auto clonedOps = cloneBlockWithoutTerminator(rewriter, *loop.getBody(), mapping);
  auto mapFoldResult = [&](OpFoldResult value) -> OpFoldResult {
    if (auto attr = dyn_cast<Attribute>(value))
      return attr;
    return mapping.lookupOrDefault(cast<Value>(value));
  };
  SmallVector<OpFoldResult> offsets =
      llvm::map_to_vector(transfer->mappedRegion.offsets, mapFoldResult);
  SmallVector<OpFoldResult> sizes =
      llvm::map_to_vector(transfer->mappedRegion.sizes, mapFoldResult);
  SmallVector<OpFoldResult> strides =
      llvm::map_to_vector(transfer->mappedRegion.strides, mapFoldResult);

  Value oldTile = mapping.lookupOrDefault(oldInsert.getSource());
  RankedTensorType newTileType = cast<RankedTensorType>(oldTile.getType());
  if (transfer->localTransform == TensorValueTransform::CollapseShape)
    newTileType =
        tensor::CollapseShapeOp::inferCollapsedType(newTileType, getReassociation(consumer));
  else if (transfer->localTransform == TensorValueTransform::ExpandShape)
    newTileType = tensor::ExtractSliceOp::inferResultType(resultType, sizes);
  Value newTile = materializeTensorValueTransform(rewriter, loc, transfer->localTransform, consumer,
                                                  oldTile, newTileType, sizes);

  SmallVector<std::pair<Operation *, Operation *>> clonedYields;
  Operation *newInsert = nullptr;
  pointBuilderToForallParallel(rewriter, newLoop);
  for (Operation *yieldingOp : yieldingOps) {
    if (yieldingOp == oldInsert) {
      newInsert = tensor::ParallelInsertSliceOp::create(rewriter, oldInsert.getLoc(), newTile,
                                                        newLoop.getRegionOutArgs()[resultNumber],
                                                        offsets, sizes, strides);
      continue;
    }
    Operation *cloned = rewriter.clone(*yieldingOp, mapping);
    clonedYields.emplace_back(yieldingOp, cloned);
  }

  notifyClonedOpsRecursively(rewriter, clonedOps);
  notifyClonedOpsRecursively(rewriter, clonedYields);
  SmallVector<Operation *> fusedOps;
  if (Operation *tileTransform = newTile.getDefiningOp();
      tileTransform && tileTransform != oldTile.getDefiningOp())
    fusedOps.push_back(tileTransform);
  fusedOps.push_back(newInsert);

  rewriter.setInsertionPointAfter(newLoop);
  Value oldResultView = materializeTensorValueTransform(
      rewriter, loc, transfer->forallDestination->blockArgumentTransform, consumer,
      newLoop.getResult(resultNumber),
      cast<RankedTensorType>(loop.getResult(resultNumber).getType()), sourceShape);
  SmallVector<Value> loopReplacements(newLoop.getResults());
  loopReplacements[resultNumber] = oldResultView;

  Operation *oldResultViewOp = oldResultView.getDefiningOp();
  rewriter.replaceOp(consumer, newLoop.getResult(resultNumber));
  if (auto *listener = dyn_cast_if_present<RewriterBase::Listener>(rewriter.getListener()))
    listener->notifyOperationReplaced(loop, newLoop);
  rewriter.replaceOp(loop, loopReplacements);
  if (isOpTriviallyDead(oldResultViewOp))
    rewriter.eraseOp(oldResultViewOp);
  return TensorConsumerFusionResult{newLoop, std::move(fusedOps)};
}

static FailureOr<scf::ForallOp> canonicalizeForLoop(RewriterBase &rewriter, scf::ForallOp loop) {
  RewritePatternSet patterns(rewriter.getContext());
  scf::ForallOp::getCanonicalizationPatterns(patterns, rewriter.getContext());
  SmallVector<Operation *> trackedLoops{loop};
  TrackedOperationsListener listener(trackedLoops, rewriter.getListener());
  GreedyRewriteConfig config;
  config.setListener(&listener);
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  if (failed(applyOpPatternsGreedily({loop}, FrozenRewritePatternSet(std::move(patterns)), config)))
    return failure();
  if (!llvm::hasSingleElement(trackedLoops))
    return failure();
  return dyn_cast<scf::ForallOp>(trackedLoops.front());
}

DiagnosedSilenceableFailure
FusionGreedyConsumersIntoProducerOp::apply(transform::TransformRewriter &rewriter,
                                           TransformResults &transformResults,
                                           TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  scf::ForallOp loop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getProducerLoop, "producer loop", loop,
                               scf::ForallOp);
  DenseSet<Operation *> stopOps;
  if (Value stopOpHandle = getStopOp()) {
    auto stopOpRange = state.getPayloadOps(stopOpHandle);
    stopOps.insert(stopOpRange.begin(), stopOpRange.end());
  }
  size_t resultNumber = getResultNumber();
  if (resultNumber >= loop->getNumResults())
    BAIL("result number is out of range for producer loop");
  const bool inlineElemwise = getInlineElementwise();

  SmallVector<Operation *> fusedOps;
  DenseSet<Operation *> failedConsumers;
  OpBuilder::Listener *previousListener = rewriter.getListener();
  TrackedOperationsListener fusedOpsListener(fusedOps, previousListener);
  rewriter.setListener(&fusedOpsListener);
  auto restoreListener = llvm::scope_exit([&]() { rewriter.setListener(previousListener); });

  bool followFusedChain = false;
  while (true) {
    // Run CSE on the loop body because fusion may fail without it (fusion compares indices by
    // operation equality of affine ops, so we want to make sure that we don't have duplicate affine
    // ops in the loop body).
    eliminateLocalCommonSubexpressions(rewriter, loop);

    SmallVector<Operation *> consumers;
    for (Operation *consumer : loop->getResult(resultNumber).getUsers()) {
      if (failedConsumers.contains(consumer))
        continue;
      if (stopOps.contains(consumer)) {
        // Stop all fusing when we reach the stop op.
        transformResults.set(getOperation()->getResult(0), fusedOps);
        if (!failedConsumers.empty())
          transform.emitRemark("did not fuse all discovered consumers into the producer loop");
        return DiagnosedSilenceableFailure::success();
      }
      consumers.push_back(consumer);
    }
    if (consumers.empty()) {
      // No consumer found -- we are done.
      transformResults.set(getOperation()->getResult(0), fusedOps);
      if (!failedConsumers.empty())
        transform.emitRemark("did not fuse all discovered consumers into the producer loop");
      return DiagnosedSilenceableFailure::success();
    }
    // Sort by their position in the block so that we fuse consumers in program order.
    llvm::sort(consumers, [](Operation *lhs, Operation *rhs) {
      return lhs->getBlock() == rhs->getBlock() && lhs->isBeforeInBlock(rhs);
    });

    SmallVector<Operation *> nextFrontierUsers;
    for (Operation *consumer : consumers) {
      LLVM_DEBUG(llvm::dbgs() << "greedy consumer-fusion candidate: " << consumer->getName()
                              << "\n");
      auto genericConsumer = dyn_cast<linalg::GenericOp>(consumer);
      if (inlineElemwise && genericConsumer) {
        FailureOr<ElementwiseInlineResult> inlineResult =
            greedyInlineElementwiseProducers(rewriter, genericConsumer);
        if (failed(inlineResult))
          BAIL("failed to inline elementwise producer into consumer");
        consumer = inlineResult->fusedOp;
      }

      SmallVector<Value> externalOperands;
      for (Value operand : consumer->getOperands())
        if (operand.getDefiningOp() != loop)
          externalOperands.push_back(operand);
      rewriter.setInsertionPoint(loop);
      IRMapping availableValues;
      if (failed(makeValuesAvailableAtInsertionPoint(rewriter, externalOperands, availableValues,
                                                     DefChainAction::Move)))
        BAIL("failed to make consumer operands available before the producer loop");

      unsigned loopOperandCount = llvm::count_if(
          consumer->getOperands(), [&](Value operand) { return operand.getDefiningOp() == loop; });
      bool advanceFrontier = followFusedChain || loopOperandCount > 1;
      if (advanceFrontier && consumers.size() == 1 && consumer->getNumResults() == 1)
        nextFrontierUsers = llvm::map_to_vector(consumer->getResult(0).getUsers(),
                                                [](Operation *user) { return user; });

      SmallVector<LoopLikeOpInterface> loops{loop};
      FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult =
          tileAndFuseConsumerWithDebug(rewriter, *consumer, loops);
      if (failed(fuseResult) || fuseResult->tiledOps.empty()) {
        FailureOr<TensorConsumerFusionResult> tensorFusion =
            pushTensorConsumerThroughForall(rewriter, loop, resultNumber, consumer);
        if (failed(tensorFusion)) {
          consumer->emitRemark("failed to fuse this consumer into the producer loop");
          failedConsumers.insert(consumer);
          continue;
        }
        loop = tensorFusion->loop;
        followFusedChain = advanceFrontier;
        fusedOps.append(tensorFusion->fusedOps);
        continue;
      }
      loop = cast<scf::ForallOp>(loops.front());
      followFusedChain = advanceFrontier;
      fusedOps.append(fuseResult->tiledOps);
      if (isOpTriviallyDead(consumer))
        rewriter.eraseOp(consumer);
    }

    FailureOr<scf::ForallOp> canonicalizedLoop = canonicalizeForLoop(rewriter, loop);
    if (failed(canonicalizedLoop))
      BAIL("failed to canonicalize the producer loop after consumer fusion");
    loop = *canonicalizedLoop;
    SmallVector<unsigned> frontierResults;
    for (auto [index, result] : llvm::enumerate(loop.getResults()))
      if (llvm::any_of(result.getUsers(), [&](Operation *user) {
            return llvm::is_contained(nextFrontierUsers, user);
          }))
        frontierResults.push_back(index);
    if (llvm::hasSingleElement(frontierResults))
      resultNumber = frontierResults.front();
    else if (resultNumber >= loop->getNumResults())
      BAIL("result number is out of range for producer loop after canonicalization");
  }
}

void FusionGreedyInputProducersIntoConsumerOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getConsumerOpMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

/// Return true when `use` carries an initialization value rather than a data
/// input whose producer should be pulled into the consumer loop.
static bool isInitUse(OpOperand &use) {
  if (auto dps = dyn_cast<DestinationStyleOpInterface>(use.getOwner()))
    if (dps.isDpsInit(&use))
      return true;

  if (auto loop = dyn_cast<LoopLikeOpInterface>(use.getOwner()))
    if (loop.getTiedLoopRegionIterArg(&use))
      return true;

  return false;
}

static bool hasDataUse(tensor::ExtractSliceOp slice) {
  return llvm::any_of(slice->getUses(), [](OpOperand &use) { return !isInitUse(use); });
}

/// Return true when the forall result can replace this escaping use without
/// changing dominance. The forall's own initialization edge is excluded.
static bool isReplaceableEscapingUse(OpOperand &use, scf::ForallOp forallOp) {
  Operation *user = use.getOwner();
  if (user == forallOp || forallOp->isProperAncestor(user))
    return false;

  Operation *ancestor = user;
  while (ancestor && ancestor->getBlock() != forallOp->getBlock())
    ancestor = ancestor->getParentOp();
  return ancestor && forallOp->isBeforeInBlock(ancestor);
}

DiagnosedSilenceableFailure
FusionGreedyInputProducersIntoConsumerOp::apply(transform::TransformRewriter &rewriter,
                                                TransformResults &transformResults,
                                                TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_NON_EMPTY_OPS(state, transform, getConsumerOp, "consumer loop", consumerLoops);

  // The consumer handle denotes a loop nest. Infer its outer-to-inner order
  // from payload ancestry rather than relying on the order in the handle.
  llvm::sort(consumerLoops, [](Operation *lhs, Operation *rhs) {
    auto getDepth = [](Operation *op) {
      unsigned depth = 0;
      for (Operation *parent = op->getParentOp(); parent; parent = parent->getParentOp())
        ++depth;
      return depth;
    };
    return getDepth(lhs) < getDepth(rhs);
  });

  SmallVector<LoopLikeOpInterface> loops;
  loops.reserve(consumerLoops.size());
  for (auto [index, loop] : llvm::enumerate(consumerLoops)) {
    auto loopLike = dyn_cast<LoopLikeOpInterface>(loop);
    if (!loopLike)
      BAIL("expected every consumer loop to implement LoopLikeOpInterface");
    if (index > 0 && !consumerLoops[index - 1]->isProperAncestor(loop))
      BAIL("expected consumer loops to form a strictly nested loop nest");
    loops.push_back(loopLike);
  }
  if (!isa<scf::ForallOp>(loops.front().getOperation()))
    BAIL("expected the outermost consumer loop to be an scf.forall");

  struct WorkItem {
    tensor::ExtractSliceOp slice;
    unsigned loopDepth;
  };

  SmallVector<WorkItem> initialWorklist;
  auto enqueue = [&](tensor::ExtractSliceOp slice, unsigned loopDepth,
                     SmallVectorImpl<WorkItem> &items) {
    Operation *fusionBoundary = loops[loopDepth].getOperation();
    auto source = dyn_cast<OpResult>(slice.getSource());
    if (!source || fusionBoundary->isAncestor(source.getOwner()) || !hasDataUse(slice))
      return;
    items.push_back(WorkItem{slice, loopDepth});
  };

  // Each boundary slice is placed at the deepest selected loop that contains
  // it. In particular, a slice used after an inner loop is fused only into the
  // surrounding outer loop.
  loops.front()->walk([&](tensor::ExtractSliceOp slice) {
    std::optional<unsigned> deepestLoop;
    for (auto [index, loop] : llvm::enumerate(loops))
      if (loop->isAncestor(slice))
        deepestLoop = index;
    if (deepestLoop)
      enqueue(slice, *deepestLoop, initialWorklist);
  });
  llvm::stable_sort(initialWorklist, [](const WorkItem &lhs, const WorkItem &rhs) {
    return lhs.loopDepth > rhs.loopDepth;
  });
  std::deque<WorkItem> worklist(initialWorklist.begin(), initialWorklist.end());

  // tileAndFuseProducerOfSlice speculatively inserts a cloned producer, a
  // cloned candidate slice, and possibly tiling helpers immediately before the
  // candidate slice. If tiling fails, the utility currently leaves those
  // operations behind. Remove the trivially dead part of that insertion
  // interval without disturbing unrelated operations in the loop.
  auto eraseDeadFailedFusionInsertions = [&](Operation *previous, tensor::ExtractSliceOp anchor) {
    SmallVector<Operation *> inserted;
    Operation *operation = previous ? previous->getNextNode() : &anchor->getBlock()->front();
    for (; operation != anchor; operation = operation->getNextNode())
      inserted.push_back(operation);
    for (Operation *insertedOp : llvm::reverse(inserted))
      if (isOpTriviallyDead(insertedOp))
        rewriter.eraseOp(insertedOp);
  };

  SmallVector<Operation *> fusedOps;
  bool hasFailure = false;
  while (!worklist.empty()) {
    WorkItem item = worklist.front();
    worklist.pop_front();
    tensor::ExtractSliceOp slice = item.slice;
    LLVM_DEBUG({
      llvm::dbgs() << "greedy producer-fusion candidate: ";
      if (Operation *producer = slice.getSource().getDefiningOp())
        llvm::dbgs() << producer->getName();
      else
        llvm::dbgs() << "block argument";
      llvm::dbgs() << "\n";
    });

    MutableArrayRef<LoopLikeOpInterface> loopPrefix(loops);
    loopPrefix = loopPrefix.take_front(item.loopDepth + 1);
    Operation *tensorProducer = slice.getSource().getDefiningOp();
    if (isa_and_nonnull<tensor::ExtractSliceOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
            tensorProducer)) {
      rewriter.setInsertionPoint(slice);
      StridedTensorRegion requestedRegion{
          slice.getSource(), llvm::to_vector(slice.getMixedOffsets()),
          llvm::to_vector(slice.getMixedSizes()), llvm::to_vector(slice.getMixedStrides())};
      FailureOr<TensorRegionTransfer> transfer = transferTensorRegion(
          rewriter, tensorProducer, TensorRegionTransferDirection::Pullback, requestedRegion);
      if (failed(transfer)) {
        slice.emitRemark("failed to pull this tensor producer through extract_slice");
        hasFailure = true;
        continue;
      }

      tensor::ExtractSliceOp sourceTile =
          pullTensorProducerThroughSlice(rewriter, slice, tensorProducer, *transfer);
      SmallVector<WorkItem> generatedItems;
      enqueue(sourceTile, item.loopDepth, generatedItems);
      worklist.insert(worklist.begin(), generatedItems.begin(), generatedItems.end());
      if (isOpTriviallyDead(tensorProducer)) {
        llvm::erase_if(worklist,
                       [&](const WorkItem &pending) { return pending.slice == tensorProducer; });
        rewriter.eraseOp(tensorProducer);
      }
      continue;
    }

    Operation *previous = slice->getPrevNode();
    std::optional<scf::SCFFuseProducerOfSliceResult> fused =
        scf::tileAndFuseProducerOfSlice(rewriter, slice, loopPrefix);
    if (!fused) {
      eraseDeadFailedFusionInsertions(previous, slice);
      if (Operation *source = slice.getSource().getDefiningOp())
        source->emitRemark("failed to fuse this op into the consumer loop nest");
      else
        slice.emitRemark("failed to fuse the producer of this slice into the consumer loop nest");
      hasFailure = true;
      continue;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "greedy input-producer fusion into loop depth " << item.loopDepth << ":\n"
                   << *fused->origProducer.getOwner() << "\n";
    });
    fusedOps.append(fused->tiledOps);

    SmallVector<Operation *> generatedSlices = fused->generatedSlices;
    Operation *originalProducer = fused->origProducer.getOwner();
    SmallVector<unsigned> escapingResults;
    auto outerForall = cast<scf::ForallOp>(loops.front().getOperation());
    for (auto [resultNumber, result] : llvm::enumerate(originalProducer->getResults()))
      if (llvm::any_of(result.getUses(),
                       [&](OpOperand &use) { return isReplaceableEscapingUse(use, outerForall); }))
        escapingResults.push_back(resultNumber);

    // Reconstruct fused values that still escape the loop. This turns
    //
    //   %p = producer
    //   %r = scf.forall { use tile(%p) }
    //   use %p
    //
    // into a forall with an additional shared_out/result, allowing the
    // original full-tensor producer to become dead. The upstream helper only
    // supports a forall as the innermost selected loop, so nested placement is
    // left unchanged until it has an escaping producer at the forall level.
    if (!escapingResults.empty() && item.loopDepth == 0) {
      FailureOr<SmallVector<Operation *>> reconstructionSlices =
          scf::yieldReplacementForFusedProducer(rewriter, slice, *fused, loops, escapingResults);
      if (failed(reconstructionSlices))
        BAIL("failed to reconstruct an escaping fused producer from the consumer loop");
      generatedSlices.append(*reconstructionSlices);

      outerForall = cast<scf::ForallOp>(loops.front().getOperation());
      ValueRange reconstructed = outerForall.getResults().take_back(escapingResults.size());
      for (auto [resultNumber, replacement] : llvm::zip_equal(escapingResults, reconstructed)) {
        Value original = originalProducer->getResult(resultNumber);
        original.replaceUsesWithIf(replacement, [&](OpOperand &use) {
          return isReplaceableEscapingUse(use, outerForall);
        });
      }
    }

    SmallVector<WorkItem> generatedItems;
    for (Operation *generated : generatedSlices)
      if (auto generatedSlice = dyn_cast<tensor::ExtractSliceOp>(generated))
        enqueue(generatedSlice, item.loopDepth, generatedItems);
    worklist.insert(worklist.end(), generatedItems.begin(), generatedItems.end());

    // tileAndFuseProducerOfSlice replaces the uses of the slice but
    // intentionally leaves the now-dead operation behind.
    if (slice->use_empty())
      rewriter.eraseOp(slice);
    if (isOpTriviallyDead(originalProducer))
      rewriter.eraseOp(originalProducer);
  }

  SmallVector<Operation *> updatedLoops =
      llvm::map_to_vector(loops, [](LoopLikeOpInterface loop) { return loop.getOperation(); });
  transformResults.set(getOperation()->getResult(0), fusedOps);
  transformResults.set(getOperation()->getResult(1), updatedLoops);
  if (hasFailure)
    transform.emitRemark("failed to fuse some producers into the consumer loop nest");
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
