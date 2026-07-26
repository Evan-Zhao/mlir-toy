#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/PartialReduction.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::transform {

namespace {

using OpPairs = SmallVector<std::pair<Operation *, Operation *>>;

#define BAIL(message) return emitSilenceableFailure(transform, message);

std::optional<unsigned> findLoopIvIndex(Value value, ArrayRef<Value> ivs) {
  if (!value)
    return std::nullopt;
  for (auto [index, iv] : llvm::enumerate(ivs))
    if (value == iv)
      return index;

  auto affineApply = value.getDefiningOp<affine::AffineApplyOp>();
  if (!affineApply || affineApply.getNumOperands() != 1)
    return std::nullopt;
  for (auto [index, iv] : llvm::enumerate(ivs))
    if (affineApply.getOperand(0) == iv)
      return index;
  return std::nullopt;
}

FailureOr<OpFoldResult> remapAffineIndex(RewriterBase &rewriter, Location loc, OpFoldResult ofr,
                                         const DenseMap<Value, Value> &mapping) {
  if (auto attr = dyn_cast<Attribute>(ofr))
    return success(attr);
  Value value = cast<Value>(ofr);
  if (auto it = mapping.find(value); it != mapping.end())
    return success(it->second);
  auto affineApply = value.getDefiningOp<affine::AffineApplyOp>();
  if (!affineApply)
    return failure();
  SmallVector<Value> remappedOperands;
  for (Value operand : affineApply.getOperands()) {
    if (auto it = mapping.find(operand); it != mapping.end())
      remappedOperands.push_back(it->second);
    else
      remappedOperands.push_back(operand);
  }
  auto affineOp =
      affine::AffineApplyOp::create(rewriter, loc, affineApply.getAffineMap(), remappedOperands);
  return success(affineOp.getResult());
}

using OpFoldResults = SmallVector<OpFoldResult>;

// Remaps the offsets, sizes, and strides of a view operation (e.g., tensor.extract_slice)
// according to a provided mapping of values.
FailureOr<std::tuple<OpFoldResults, OpFoldResults, OpFoldResults>>
remapAffineOffsetSizeStride(RewriterBase &rewriter, Location loc,
                            OffsetSizeAndStrideOpInterface view,
                            const DenseMap<Value, Value> &mapping) {
  auto remapAffineIndices = [&](SmallVectorImpl<OpFoldResult> &values) {
    for (OpFoldResult &value : values) {
      auto result = remapAffineIndex(rewriter, loc, value, mapping);
      if (failed(result))
        return failure();
      value = *result;
    }
    return success();
  };
  auto offsets = view.getMixedOffsets(), sizes = view.getMixedSizes(),
       strides = view.getMixedStrides();
  if (failed(remapAffineIndices(offsets)) || failed(remapAffineIndices(sizes)) ||
      failed(remapAffineIndices(strides)))
    return failure();
  return std::make_tuple(offsets, sizes, strides);
}

SmallVector<OpFoldResult> dropAt(SmallVector<OpFoldResult> values, uint64_t index) {
  values.erase(values.begin() + index);
  return values;
}

struct FoldedTensorInfo {
  RankedTensorType oldType;
  RankedTensorType newType;
  SmallVector<ReassociationIndices> reassociation;

  SmallVector<OpFoldResult> collapseSliceParams(ArrayRef<OpFoldResult> params) const {
    SmallVector<OpFoldResult> collapsed;
    collapsed.reserve(reassociation.size());
    for (const ReassociationIndices &group : reassociation) {
      int64_t selectedDim = group.back();
      for (int64_t dim : group) {
        if (oldType.getDimSize(dim) != 1) {
          selectedDim = dim;
          break;
        }
      }
      collapsed.push_back(params[selectedDim]);
    }
    return collapsed;
  }
};

SmallVector<OpFoldResult> getStaticMixedSizes(OpBuilder &builder, RankedTensorType type) {
  SmallVector<OpFoldResult> sizes;
  sizes.reserve(type.getRank());
  for (int64_t size : type.getShape())
    sizes.push_back(builder.getIndexAttr(size));
  return sizes;
}

FailureOr<FoldedTensorInfo> getFoldedTensorInfo(OpBuilder &builder, Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || tensorType.getEncoding())
    return failure();
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      linalg::getReassociationMapForFoldingUnitDims(getStaticMixedSizes(builder, tensorType));
  if (!reassociation)
    return failure();
  auto newType = tensor::CollapseShapeOp::inferCollapsedType(tensorType, *reassociation);
  if (newType == tensorType)
    return failure(); // Here means "nothing to do".
  return FoldedTensorInfo{
      .oldType = tensorType, .newType = newType, .reassociation = std::move(*reassociation)};
}

FailureOr<SmallVector<std::optional<FoldedTensorInfo>>> getFoldedTensorInfos(OpBuilder &builder,
                                                                             ValueRange values) {
  using RetT = SmallVector<std::optional<FoldedTensorInfo>>;
  RetT infos;
  infos.reserve(values.size());
  bool changed = false;
  for (Value v : values) {
    FailureOr<FoldedTensorInfo> info = getFoldedTensorInfo(builder, v.getType());
    changed |= succeeded(info);
    infos.push_back(succeeded(info) ? info : std::optional<FoldedTensorInfo>{});
  }
  return changed ? success(infos) : FailureOr<RetT>{};
}

Value collapseTensor(OpBuilder &builder, Location loc, Value value, const FoldedTensorInfo &info) {
  return tensor::CollapseShapeOp::create(builder, loc, info.newType, value, info.reassociation);
}

Value expandTensor(OpBuilder &builder, Location loc, Value value, const FoldedTensorInfo &info) {
  return tensor::ExpandShapeOp::create(builder, loc, info.oldType, value, info.reassociation);
}

void notifyReplacedRecursively(Operation *oldOp, RewriterBase &rewriter, Operation *newOp) {
  auto *listener = dyn_cast_if_present<RewriterBase::Listener>(rewriter.getListener());
  if (!listener)
    return;

  SmallVector<Operation *> oldOps, newOps;
  oldOp->walk<WalkOrder::PreOrder>([&](Operation *op) { oldOps.push_back(op); });
  newOp->walk<WalkOrder::PreOrder>([&](Operation *op) { newOps.push_back(op); });
  if (oldOps.size() != newOps.size())
    return;

  for (auto [oldNested, newNested] : llvm::zip(oldOps, newOps))
    listener->notifyOperationReplaced(oldNested, newNested);
}

template <typename OpRange>
void notifyClonedOpsRecursively(RewriterBase &rewriter, OpRange &&clonedOps) {
  for (auto [oldOp, newOp] : clonedOps)
    notifyReplacedRecursively(oldOp, rewriter, newOp);
}

void notifyLoopReplaced(RewriterBase &rewriter, Operation *oldLoop, Operation *newLoop) {
  if (auto *listener = dyn_cast_if_present<RewriterBase::Listener>(rewriter.getListener()))
    listener->notifyOperationReplaced(oldLoop, newLoop);
}

std::optional<unsigned> findForallOutArgIndex(scf::ForallOp loop, Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != loop.getBody())
    return std::nullopt;
  unsigned rank = loop.getRank();
  if (blockArg.getArgNumber() < rank)
    return std::nullopt;
  unsigned index = blockArg.getArgNumber() - rank;
  if (index >= loop.getNumResults())
    return std::nullopt;
  return index;
}

FailureOr<Operation *> cloneOrRewriteForallCombiningOp(
    scf::ForallOp oldLoop, RewriterBase &rewriter, Operation &oldOp, const IRMapping &mapping,
    ArrayRef<std::optional<FoldedTensorInfo>> infos, scf::ForallOp newLoop) {
  auto insert = cast<tensor::ParallelInsertSliceOp>(&oldOp);
  auto loc = insert.getLoc();

  Value oldDest = insert.getDest();
  std::optional<unsigned> index = findForallOutArgIndex(oldLoop, oldDest);
  if (!index)
    return failure();

  Value source = mapping.lookupOrDefault(insert.getSource());
  Value dest = newLoop.getRegionOutArgs()[*index];
  auto viewTriple = remapAffineOffsetSizeStride(rewriter, loc, insert, mapping.getValueMap());
  if (failed(viewTriple))
    return failure();
  auto [offsets, sizes, strides] = *viewTriple;

  const auto &info = infos[*index];
  if (info) {
    auto sourceType = dyn_cast<RankedTensorType>(source.getType());
    if (!sourceType)
      return failure();
    offsets = info->collapseSliceParams(offsets);
    sizes = info->collapseSliceParams(sizes);
    strides = info->collapseSliceParams(strides);

    RankedTensorType expectedSourceType =
        tensor::ExtractSliceOp::inferResultType(info->newType, sizes);
    auto isRankReduced = [](RankedTensorType lhs, RankedTensorType rhs) {
      return isRankReducedType(lhs, rhs) == SliceVerificationResult::Success;
    };
    if (isRankReduced(expectedSourceType, sourceType)) {
      // Source is already a legal rank-reduced tile for the rewritten slice.
    } else if (auto sourceInfo = getFoldedTensorInfo(rewriter, sourceType);
               succeeded(sourceInfo) && isRankReduced(expectedSourceType, sourceInfo->newType)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(newLoop.getTerminator());
      source = tensor::CollapseShapeOp::create(rewriter, insert.getLoc(), sourceInfo->newType,
                                               source, sourceInfo->reassociation);
    } else {
      return failure();
    }
  } else {
    dest = mapping.lookupOrDefault(oldDest);
  }

  return {tensor::ParallelInsertSliceOp::create(rewriter, insert.getLoc(), source, dest, offsets,
                                                sizes, strides)};
}

template <typename LoopOp> struct LoopSharedTrait {};

template <> struct LoopSharedTrait<scf::ForallOp> {
  static ValueRange getInitOrOut(scf::ForallOp loop) { return loop.getOutputs(); }
  static scf::ForallOp createLike(OpBuilder &builder, scf::ForallOp oldLoop, ValueRange outputs) {
    return scf::ForallOp::create(builder, oldLoop.getLoc(), oldLoop.getMixedLowerBound(),
                                 oldLoop.getMixedUpperBound(), oldLoop.getMixedStep(), outputs,
                                 oldLoop.getMapping());
  }

  static void mapInductionVars(IRMapping &mapping, scf::ForallOp oldLoop, scf::ForallOp newLoop) {
    for (auto [oldIv, newIv] : llvm::zip(oldLoop.getInductionVars(), newLoop.getInductionVars()))
      mapping.map(oldIv, newIv);
  }

  static FailureOr<OpPairs>
  cloneCombiningOps(scf::ForallOp oldLoop, RewriterBase &rewriter, const IRMapping &mapping,
                    const SmallVector<std::optional<FoldedTensorInfo>> &infos,
                    scf::ForallOp newLoop) {
    OpPairs clonedCombiningOps;
    pointBuilderToForallParallel(rewriter, newLoop);
    for (Operation &oldCombiningOp : oldLoop.getTerminator().getYieldingOps()) {
      FailureOr<Operation *> newCombiningOp = cloneOrRewriteForallCombiningOp(
          oldLoop, rewriter, oldCombiningOp, mapping, infos, newLoop);
      if (failed(newCombiningOp)) {
        oldCombiningOp.emitError() << "failed to clone or rewrite this combining operation";
        return failure();
      }
      clonedCombiningOps.emplace_back(&oldCombiningOp, *newCombiningOp);
    }
    return clonedCombiningOps;
  }
};

template <> struct LoopSharedTrait<scf::ForOp> {
  static ValueRange getInitOrOut(scf::ForOp loop) { return loop.getInitArgs(); }
  static scf::ForOp createLike(OpBuilder &builder, scf::ForOp oldLoop, ValueRange inits) {
    return scf::ForOp::create(builder, oldLoop.getLoc(), oldLoop.getLowerBound(),
                              oldLoop.getUpperBound(), oldLoop.getStep(), inits,
                              /*bodyBuilder=*/nullptr, oldLoop.getUnsignedCmp());
  }
  static void mapInductionVars(IRMapping &mapping, scf::ForOp oldLoop, scf::ForOp newLoop) {
    mapping.map(oldLoop.getInductionVar(), newLoop.getInductionVar());
  }

  static FailureOr<OpPairs>
  cloneCombiningOps(scf::ForOp oldLoop, RewriterBase &rewriter, const IRMapping &mapping,
                    const SmallVector<std::optional<FoldedTensorInfo>> &infos, scf::ForOp newLoop) {
    auto oldYield = cast<scf::YieldOp>(oldLoop.getBody()->getTerminator());
    rewriter.setInsertionPointToEnd(newLoop.getBody());
    SmallVector<Value> newYieldOperands;
    newYieldOperands.reserve(oldYield.getNumOperands());
    auto loc = oldYield.getLoc();
    for (auto [yielded, info] : llvm::zip(oldYield.getOperands(), infos)) {
      Value mapped = mapping.lookupOrDefault(yielded);
      newYieldOperands.push_back(info ? collapseTensor(rewriter, loc, mapped, *info) : mapped);
    }
    auto newYield = scf::YieldOp::create(rewriter, loc, newYieldOperands);
    return OpPairs{{oldYield, newYield}};
  }
};

template <typename LoopOp>
FailureOr<LoopOp> foldUnitExtentDimsInLoop(PatternRewriter &rewriter, LoopOp loop) {
  auto infosR = getFoldedTensorInfos(rewriter, loop.getResults());
  if (failed(infosR))
    return failure();
  SmallVector<std::optional<FoldedTensorInfo>> &infos = *infosR;
  using LoopTrait = LoopSharedTrait<LoopOp>;

  Location loc = loop.getLoc();
  rewriter.setInsertionPoint(loop);
  SmallVector<Value> newOuts = LoopTrait::getInitOrOut(loop);
  for (size_t i = 0; i < newOuts.size(); ++i)
    if (infos[i])
      newOuts[i] = collapseTensor(rewriter, loc, newOuts[i], *infos[i]);

  // Create the new loop with the same bounds and steps, but with the new outputs.
  auto newLoop = LoopTrait::createLike(rewriter, loop, newOuts);
  newLoop->setAttrs(loop->getAttrs());
  // Map the induction variables and the region arguments of the old loop to the new loop.
  IRMapping mapping;
  LoopTrait::mapInductionVars(mapping, loop, newLoop);
  rewriter.setInsertionPointToStart(newLoop.getBody());
  for (auto [oldArg, newArg, info] :
       llvm::zip(loop.getRegionIterArgs(), newLoop.getRegionIterArgs(), infos))
    mapping.map(oldArg, info ? expandTensor(rewriter, loc, newArg, *info) : newArg);
  // Clone loop body operations, adding to `mapping` as we go.
  OpPairs clonedOps = cloneBlockWithoutTerminator(rewriter, *loop.getBody(), mapping);
  // Clone the "combining operations" (e.g., the yield operations in scf.for and parallel insert in
  // scf.forall).
  auto clonedCombiningOpsR = LoopTrait::cloneCombiningOps(loop, rewriter, mapping, infos, newLoop);
  if (failed(clonedCombiningOpsR))
    return failure();

  notifyClonedOpsRecursively(rewriter, clonedOps);
  notifyClonedOpsRecursively(rewriter, *clonedCombiningOpsR);
  notifyLoopReplaced(rewriter, loop, newLoop);

  rewriter.setInsertionPointAfter(newLoop);
  SmallVector<Value> replacements = newLoop.getResults();
  for (size_t i = 0; i < replacements.size(); ++i)
    if (infos[i])
      replacements[i] = expandTensor(rewriter, loc, replacements[i], *infos[i]);
  rewriter.replaceOp(loop, replacements);
  return newLoop;
}

template <typename LoopOp> struct FoldUnitExtentDimsInLoopPattern : OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loop, PatternRewriter &rewriter) const override {
    return success(succeeded(foldUnitExtentDimsInLoop(rewriter, loop)));
  }
};

struct SplitForallIntoForResult {
  struct TileSlice {
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
  };

  scf::ForallOp newForall;
  scf::ForOp innerFor;
  Value outerTile, innerTile;
  TileSlice reductionSlice;
  OpPairs clonedOps;
};

FailureOr<SplitForallIntoForResult>
splitForallDimensionForReduction(TransformOpInterface transform, RewriterBase &rewriter,
                                 scf::ForallOp loop, ReductionForallSplitPlan &plan) {
  auto loc = loop.getLoc();
  auto splitArray = [](SmallVector<OpFoldResult> xs,
                       unsigned index) -> std::pair<SmallVector<OpFoldResult>, OpFoldResult> {
    OpFoldResult removed = xs[index];
    xs.erase(xs.begin() + index);
    return {std::move(xs), removed};
  };
  auto [outerLbs, innerLb] = splitArray(loop.getMixedLowerBound(), plan.removedIvIndex);
  auto [outerUbs, innerUb] = splitArray(loop.getMixedUpperBound(), plan.removedIvIndex);
  auto [outerSteps, innerStep] = splitArray(loop.getMixedStep(), plan.removedIvIndex);
  SmallVector<Value> newOutputs = llvm::to_vector(loop.getOutputs());
  newOutputs.push_back(plan.reductionInit);
  auto newForall =
      scf::ForallOp::create(rewriter, loc, outerLbs, outerUbs, outerSteps, newOutputs, {});

  rewriter.setInsertionPointToStart(newForall.getBody());
  Value outerProducerArg = newForall.getRegionOutArgs().front();
  Value outerReductionArg = newForall.getRegionOutArgs().back();
  Value innerLbV = getValueOrCreateConstantIndexOp(rewriter, loc, innerLb);
  Value innerUbV = getValueOrCreateConstantIndexOp(rewriter, loc, innerUb);
  Value innerStepV = getValueOrCreateConstantIndexOp(rewriter, loc, innerStep);

  IRMapping outerIvMapping;
  unsigned newOuterIvPos = 0;
  for (auto [index, oldIv] : llvm::enumerate(loop.getInductionVars())) {
    if (index == plan.removedIvIndex) {
      outerIvMapping.map(oldIv, innerLbV);
    } else {
      outerIvMapping.map(oldIv, newForall.getInductionVars()[newOuterIvPos++]);
    }
  }
  outerIvMapping.map(loop.getRegionOutArgs().front(), outerProducerArg);

  auto viewTriple =
      remapAffineOffsetSizeStride(rewriter, loc, plan.producerInsert, outerIvMapping.getValueMap());
  if (failed(viewTriple)) {
    plan.producerInsert.emitError() << "failed to remap offsets of this operation";
    return failure();
  }
  auto [outerTileOffsets, outerTileSizes, outerTileStrides] = *viewTriple;
  outerTileSizes[plan.producerRedDim] =
      getMixedTensorSizes(rewriter, loc, plan.loopProducedInput->get())[plan.producerRedDim];
  Value tileInit = createExtractSliceFromState(rewriter, loc, outerProducerArg, outerTileOffsets,
                                               outerTileSizes, outerTileStrides);

  auto redTileOffsets = dropAt(outerTileOffsets, plan.producerRedDim);
  auto redTileSizes = dropAt(outerTileSizes, plan.producerRedDim);
  auto redTileStrides = dropAt(outerTileStrides, plan.producerRedDim);
  Value redTileInit = createExtractSliceFromState(rewriter, loc, outerReductionArg, redTileOffsets,
                                                  redTileSizes, redTileStrides);

  // Create the inner loop that iterates over the removed induction variable.
  auto forLoop = scf::ForOp::create(rewriter, loc, innerLbV, innerUbV, innerStepV,
                                    ValueRange{tileInit, redTileInit});
  outerIvMapping.map(loop.getInductionVars()[plan.removedIvIndex], forLoop.getInductionVar());
  rewriter.setInsertionPointToStart(forLoop.getBody());
  OpPairs clonedOps = cloneBlockWithoutTerminator(rewriter, *loop.getBody(), outerIvMapping);

  Value outerTile = outerIvMapping.lookup(plan.producerInsert.getSource());
  rewriter.setInsertionPointToEnd(forLoop.getBody());
  auto offsets = plan.producerInsert.getMixedOffsets();
  SmallVector<OpFoldResult> localOffsets(offsets.size(), rewriter.getIndexAttr(0));
  localOffsets[plan.producerRedDim] =
      *remapAffineIndex(rewriter, loc, offsets[plan.producerRedDim], outerIvMapping.getValueMap());
  Value innerTile = tensor::InsertSliceOp::create(
      rewriter, loc, outerTile, forLoop.getRegionIterArgs()[0], localOffsets,
      plan.producerInsert.getMixedSizes(), outerTileStrides);

  pointBuilderToForallParallel(rewriter, newForall);
  tensor::ParallelInsertSliceOp::create(rewriter, loc, forLoop.getResult(0), outerProducerArg,
                                        outerTileOffsets, outerTileSizes, outerTileStrides);
  return SplitForallIntoForResult{.newForall = newForall,
                                  .innerFor = forLoop,
                                  .outerTile = outerTile,
                                  .innerTile = innerTile,
                                  .reductionSlice = {.offsets = std::move(redTileOffsets),
                                                     .sizes = std::move(redTileSizes),
                                                     .strides = std::move(redTileStrides)},
                                  .clonedOps = std::move(clonedOps)};
}

} // namespace

FailureOr<ReductionForallSplitPlan>
detectReductionForallSplit(const TransformOpInterface &transform, scf::ForallOp loop,
                           linalg::GenericOp consumer) {
  FailureOr<uint64_t> opReductionDim = matchOneDimReductionGeneric(consumer);
  if (failed(opReductionDim))
    return consumer.emitError() << "expected a single-dimension reduction linalg.generic consumer";

  SmallVector<OpOperand *> loopProducedOperands;
  for (auto [inputIndex, operand] : llvm::enumerate(consumer.getDpsInputOperands())) {
    auto operandResult = dyn_cast<OpResult>(operand->get());
    if (operandResult && operandResult.getOwner() == loop.getOperation())
      loopProducedOperands.push_back(operand);
  }
  if (loopProducedOperands.size() != 1)
    return consumer.emitError()
           << "expected exactly one input of this op to be produced by the target scf.forall";
  OpOperand *loopProducedOpnd = loopProducedOperands.front();
  auto loopResult = cast<OpResult>(loopProducedOpnd->get());

  std::optional<uint64_t> producerReductionDim;
  AffineMap producerIndexingMap =
      consumer.getIndexingMapsArray()[loopProducedOpnd->getOperandNumber()];
  for (auto [resultIndex, expr] : llvm::enumerate(producerIndexingMap.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (dimExpr && dimExpr.getPosition() == *opReductionDim) {
      producerReductionDim = resultIndex;
      break;
    }
  }
  if (!producerReductionDim)
    return consumer.emitError() << "expected the loop-produced input to be a reduction argument "
                                   "and use the reduction iterator";

  auto insertSliceR = getParallelInsertSliceForLoopResult(loop, loopResult);
  if (failed(insertSliceR)) {
    loop.emitError() << "failed to find the tensor.parallel_insert_slice operation in this loop "
                     << "that published result #" << loopResult.getResultNumber();
    return failure();
  }
  tensor::ParallelInsertSliceOp producerInsert = *insertSliceR;

  OpFoldResult reductionOffset = producerInsert.getMixedOffsets()[*producerReductionDim];
  Value reductionOffsetValue = dyn_cast<Value>(reductionOffset);
  // findLoopIvIndex handles the case where `reductionOffsetValue` is null, so we can have a single
  // point of failure reporting.
  std::optional<unsigned> removedIvIndex =
      findLoopIvIndex(reductionOffsetValue, loop.getInductionVars());
  if (!removedIvIndex) {
    producerInsert.emitError()
        << "expected the offset on dimension " << *producerReductionDim
        << " to be a dynamic value controlled by a single scf.forall induction variable; got "
        << reductionOffsetValue;
    return failure();
  }

  auto reductionInit = consumer.getDpsInits().front();
  return ReductionForallSplitPlan{loopProducedOpnd, reductionInit,   producerInsert,
                                  *removedIvIndex,  *opReductionDim, *producerReductionDim};
}

FailureOr<PartialReductionForallResult>
rFactorReductionUnderForall(TransformOpInterface transform, TransformRewriter &rewriter,
                            scf::ForallOp forall, linalg::GenericOp consumer,
                            ReductionForallSplitPlan &plan) {
  Location loc = forall.getLoc();
  auto consumerPR = dyn_cast<PartialReductionOpInterface>(consumer.getOperation());
  if (!consumerPR)
    return consumer.emitError() << "expected the consumer to implement PartialReductionInterface";

  auto makeSub = [&](OpFoldResult lhs, OpFoldResult rhs) {
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    return affine::makeComposedFoldedAffineApply(rewriter, loc, s0 - s1, {lhs, rhs});
  };
  auto makeCeilDiv = [&](OpFoldResult lhs, OpFoldResult rhs) {
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    return affine::makeComposedFoldedAffineApply(rewriter, loc, s0.ceilDiv(s1), {lhs, rhs});
  };
  auto makeFloorDiv = [&](OpFoldResult lhs, OpFoldResult rhs) {
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    return affine::makeComposedFoldedAffineApply(rewriter, loc, s0.floorDiv(s1), {lhs, rhs});
  };
  auto getForallTripCount = [&](scf::ForallOp loop, unsigned ivIndex) {
    return makeCeilDiv(
        makeSub(loop.getMixedUpperBound()[ivIndex], loop.getMixedLowerBound()[ivIndex]),
        loop.getMixedStep()[ivIndex]);
  };
  auto getForallSplitIndex = [&](scf::ForallOp loop, unsigned ivIndex) {
    return makeFloorDiv(
        makeSub(loop.getInductionVars()[ivIndex], loop.getMixedLowerBound()[ivIndex]),
        loop.getMixedStep()[ivIndex]);
  };

  // Create the init value for the RF (r-factor) tensor with shape `outputShape`. The shape is
  // based on the iteration domain with the reduction dim set to the trip count of the loop.
  auto iterDomain = consumerPR.getIterationDomain(rewriter);
  auto outputShape = llvm::map_to_vector(iterDomain, [](const Range &r) { return r.size; });
  outputShape[plan.opRedDim] = getForallTripCount(forall, plan.removedIvIndex);
  SetVector<unsigned> reductionDims;
  reductionDims.insert(static_cast<unsigned>(plan.opRedDim));
  FailureOr<SmallVector<Value>> rfInits = consumerPR.generateInitialTensorForPartialReduction(
      rewriter, consumer.getLoc(), outputShape, reductionDims);
  if (failed(rfInits) || rfInits->size() != 1)
    return consumer.emitError() << "failed to create partial reduction init tensor";

  // Create a new forall loop that has the RF init tensor as an additional output.
  ForallOutputExtension extension =
      cloneForallWithAppendedOutputs(rewriter, forall, ValueRange{rfInits->front()});
  scf::ForallOp newForall = extension.forall;
  IRMapping mapping = std::move(extension.mapping);
  auto clonedOps = std::move(extension.clonedOps);

  // Get the under-loop tile of the reduction input, and use tiling interface method to map this
  // input tile to a tile of the reduction's iter domain.
  auto producerInsert =
      dyn_cast_if_present<tensor::ParallelInsertSliceOp>(mapping.lookup(plan.producerInsert));
  if (!producerInsert)
    return plan.producerInsert.emitError()
           << "failed to find this op in the new forall loop that corresponds";
  unsigned producerInputIndex = plan.loopProducedInput->getOperandNumber();
  SmallVector<OpFoldResult> iterDomainOffsets, iterDomainShape;
  if (failed(consumerPR.getIterationDomainTileFromOperandTiles(
          rewriter, {producerInputIndex}, {producerInsert.getMixedOffsets()},
          {producerInsert.getMixedSizes()}, iterDomainOffsets, iterDomainShape)))
    return consumer.emitError()
           << "failed to infer iteration-domain tile from forall-produced input tile";

  // Create the rfactor op under the loop.
  OpFoldResult loopSplitIndex = getForallSplitIndex(newForall, plan.removedIvIndex);
  Value rfOutArg = newForall.getRegionOutArgs().back();
  rewriter.setInsertionPoint(newForall.getTerminator());
  FailureOr<TilingResult> tilingResult = consumerPR.tileToPartialReduction(
      rewriter, consumer.getLoc(), ReductionTilingStrategy::PartialReductionOuterParallel,
      {rfOutArg}, iterDomainOffsets, iterDomainShape, reductionDims, {loopSplitIndex});
  if (failed(tilingResult) || tilingResult->tiledOps.size() != 1)
    return consumer.emitError() << "failed to tile this op to a partial reduction";
  auto tiledConsumer = cast<linalg::GenericOp>(tilingResult->tiledOps.front());
  auto partialTileInit =
      tiledConsumer.getDpsInits().front().getDefiningOp<tensor::ExtractSliceOp>();
  if (!partialTileInit)
    return tiledConsumer.emitError()
           << "expected partial reduction init to be a tensor.extract_slice";

  // The tiling interface slices every input from the original op operands. For the forall-produced
  // input, use the already-cloned in-loop producer tile instead of slicing the old forall result.
  Value generatedInput = tiledConsumer->getOperand(producerInputIndex);
  rewriter.modifyOpInPlace(tiledConsumer, [&]() {
    tiledConsumer->setOperand(producerInputIndex, producerInsert.getSource());
  });
  auto inputSliceOp = generatedInput.getDefiningOp<tensor::ExtractSliceOp>();
  if (inputSliceOp && inputSliceOp->use_empty())
    rewriter.eraseOp(inputSliceOp);

  // Add a parallel insert slice to write the result of the partial reduction into the RF tensor.
  pointBuilderToForallParallel(rewriter, newForall);
  tensor::ParallelInsertSliceOp::create(
      rewriter, loc, tiledConsumer.getResult(0), rfOutArg, partialTileInit.getMixedOffsets(),
      partialTileInit.getMixedSizes(), partialTileInit.getMixedStrides());

  rewriter.setInsertionPointAfter(newForall);
  FailureOr<MergeResult> mergeResult = consumerPR.mergeReductions(
      rewriter, consumer.getLoc(), ValueRange{newForall.getResults().back()}, reductionDims);
  if (failed(mergeResult) || mergeResult->mergeOps.size() != 1)
    return consumer.emitError() << "failed to create write-back reduction";

  return PartialReductionForallResult{.newForall = newForall,
                                      .rFactorOp = tiledConsumer,
                                      .writebackOp =
                                          cast<linalg::ReduceOp>(mergeResult->mergeOps.front()),
                                      .clonedOps = std::move(clonedOps)};
}

void ScfFoldUnitExtentDimsViaReshapesPatternsOp::populatePatterns(RewritePatternSet &patterns) {
  patterns.add<FoldUnitExtentDimsInLoopPattern<scf::ForOp>,
               FoldUnitExtentDimsInLoopPattern<scf::ForallOp>>(patterns.getContext());
  linalg::populateSwapExtractSliceWithFillPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  populateRegionBranchOpInterfaceCanonicalizationPatterns(patterns, scf::ForOp::getOperationName());
#if LLVM_VERSION_MAJOR >= 23
  populateRegionBranchOpInterfaceCanonicalizationPatterns(patterns,
                                                          scf::ForallOp::getOperationName());
#endif
}

void ScfFuseReductionIntoForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getConsumerOpMutable(), effects);
  onlyReadsHandle(getForallLoopMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure ScfFuseReductionIntoForallOp::apply(TransformRewriter &rewriter,
                                                                TransformResults &transformResults,
                                                                TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  scf::ForallOp loop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForallLoop, "loop", loop, scf::ForallOp);
  linalg::GenericOp consumer;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getConsumerOp, "consumer", consumer,
                               linalg::GenericOp);

  auto splitPlan = detectReductionForallSplit(transform, loop, consumer);
  if (failed(splitPlan))
    BAIL("failed to detect a split plan for the reduction");

  rewriter.setInsertionPoint(consumer);
  auto splitR = splitForallDimensionForReduction(transform, rewriter, loop, *splitPlan);
  if (failed(splitR))
    BAIL("failed to split the forall loop for the reduction");
  SplitForallIntoForResult &split = *splitR;
  for (auto [oldOp, newOp] : split.clonedOps) {
    if (succeeded(rewriter.notifyPayloadOperationReplaced(oldOp, newOp)))
      continue;
    rewriter.silenceTrackingFailure();
  }

  rewriter.setInsertionPointToEnd(split.innerFor.getBody());
  auto fusedReduction =
      cloneGenericOnTile(rewriter, consumer, split.outerTile, split.innerFor.getRegionIterArgs()[1],
                         consumer.getLoc());
  Value reductionTile = fusedReduction.getResult(0);
  auto reductionTileType = cast<RankedTensorType>(reductionTile.getType());
  SmallVector<OpFoldResult> reductionOffsets(reductionTileType.getRank(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> reductionSizes =
      getMixedTensorSizes(rewriter, loop.getLoc(), reductionTile);
  auto insertedReduction = tensor::InsertSliceOp::create(
      rewriter, loop.getLoc(), reductionTile, split.innerFor.getRegionIterArgs()[1],
      reductionOffsets, reductionSizes, getUnitStrides(rewriter, reductionTileType.getRank()));
  scf::YieldOp::create(rewriter, loop.getLoc(),
                       ValueRange{split.innerTile, insertedReduction.getResult()});

  pointBuilderToForallParallel(rewriter, split.newForall);
  Value outerReductionArg = split.newForall.getRegionOutArgs().back();
  tensor::ParallelInsertSliceOp::create(rewriter, loop.getLoc(), split.innerFor.getResult(1),
                                        outerReductionArg, split.reductionSlice.offsets,
                                        split.reductionSlice.sizes, split.reductionSlice.strides);

  if (failed(rewriter.notifyPayloadOperationReplaced(loop, split.newForall.getOperation())))
    BAIL("failed to preserve the scf.forall handle");
  rewriter.replaceOp(consumer, split.newForall.getResults().back());
  rewriter.replaceOp(loop, split.newForall.getResults().take_front(loop.getNumResults()));

  transformResults.set(getOperation()->getResult(0), {fusedReduction.getOperation()});
  transformResults.set(getOperation()->getResult(1), {split.innerFor.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

void ScfFusePartialReductionIntoForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getConsumerOpMutable(), effects);
  onlyReadsHandle(getForallLoopMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure ScfFusePartialReductionIntoForallOp::apply(
    TransformRewriter &rewriter, TransformResults &transformResults, TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  scf::ForallOp loop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForallLoop, "loop", loop, scf::ForallOp);
  linalg::GenericOp consumer;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getConsumerOp, "consumer", consumer,
                               linalg::GenericOp);

  auto splitPlan = detectReductionForallSplit(transform, loop, consumer);
  if (failed(splitPlan))
    BAIL("failed to detect a split plan for the reduction");

  rewriter.setInsertionPoint(consumer);
  auto rFactorResult = rFactorReductionUnderForall(transform, rewriter, loop, consumer, *splitPlan);
  if (failed(rFactorResult))
    BAIL("failed to split the forall loop for the reduction");

  rewriter.replaceOp(consumer, rFactorResult->writebackOp);
  rewriter.replaceOp(loop, rFactorResult->newForall.getResults().take_front(loop.getNumResults()));

  transformResults.set(getOperation()->getResult(0), {rFactorResult->rFactorOp.getOperation()});
  transformResults.set(getOperation()->getResult(1), {rFactorResult->writebackOp});
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
