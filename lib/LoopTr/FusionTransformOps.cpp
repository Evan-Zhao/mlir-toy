#include "LoopTr/LoopTransformOps.h"

#include "LoopTr/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

FailureOr<SmallVector<LoopLikeOpInterface>> collectLoopNestRootedAt(Operation *root) {
  if (auto forallOp = dyn_cast<scf::ForallOp>(root))
    return SmallVector<LoopLikeOpInterface>{forallOp};

  auto forOp = dyn_cast<scf::ForOp>(root);
  if (!forOp)
    return failure();

  SmallVector<LoopLikeOpInterface> loops;
  loops.push_back(forOp);

  while (true) {
    auto currentFor = cast<scf::ForOp>(loops.back().getOperation());
    auto yieldOp = cast<scf::YieldOp>(currentFor.getBody()->getTerminator());

    scf::ForOp nestedFor;
    for (Value yielded : yieldOp.getOperands()) {
      auto yieldedResult = dyn_cast<OpResult>(yielded);
      if (!yieldedResult)
        continue;
      auto candidate = dyn_cast<scf::ForOp>(yieldedResult.getOwner());
      if (!candidate || candidate->getBlock() != currentFor.getBody())
        continue;
      if (nestedFor && nestedFor != candidate)
        return failure();
      nestedFor = candidate;
    }

    if (!nestedFor)
      break;
    loops.push_back(nestedFor);
  }

  return loops;
}

LogicalResult verifyUnarySingleReductionGeneric(linalg::GenericOp generic) {
  if (generic.getInputs().size() != 1 || generic.getNumDpsInits() != 1)
    return failure();
  if (generic->getNumResults() != 1)
    return failure();

  auto inputType = dyn_cast<RankedTensorType>(generic.getInputs().front().getType());
  auto resultType = dyn_cast<RankedTensorType>(generic.getResults().front().getType());
  if (!inputType || !resultType || inputType.getRank() != resultType.getRank() + 1)
    return failure();

  AffineMap inputMap = generic.getIndexingMapsArray().front();
  if (!inputMap.isIdentity())
    return failure();

  SmallVector<utils::IteratorType> iteratorTypes = llvm::to_vector(generic.getIteratorTypesArray());
  std::optional<uint64_t> reductionDim = std::nullopt;
  for (auto [index, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != utils::IteratorType::reduction)
      continue;
    if (reductionDim.has_value())
      return failure();
    reductionDim = index;
  }
  if (!reductionDim.has_value())
    return failure();

  AffineMap outputMap = generic.getIndexingMapsArray().back();
  if (outputMap.getNumResults() != resultType.getRank())
    return failure();
  for (int64_t dim = 0, outIdx = 0, e = inputType.getRank(); dim < e; ++dim) {
    if (dim == reductionDim)
      continue;
    auto expr = outputMap.getResult(outIdx++);
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() != dim)
      return failure();
  }
  return success();
}

std::optional<uint64_t> getSingleReductionDim(linalg::GenericOp generic) {
  for (auto [index, iteratorType] : llvm::enumerate(generic.getIteratorTypesArray()))
    if (iteratorType == utils::IteratorType::reduction)
      return index;
  return std::nullopt;
}

FailureOr<tensor::ParallelInsertSliceOp> getParallelInsertSliceForLoopResult(scf::ForallOp loop,
                                                                             OpResult result) {
  if (result.getOwner() != loop.getOperation())
    return failure();
  BlockArgument bbArg = loop.getTiedBlockArgument(result);
  SmallVector<Operation *> combiningOps = loop.getCombiningOps(bbArg);
  if (combiningOps.size() != 1)
    return failure();
  return dyn_cast<tensor::ParallelInsertSliceOp>(combiningOps.front());
}

template <typename T> SmallVector<T> dropAt(const SmallVector<T> &values, int64_t index) {
  SmallVector<T> result;
  result.reserve(values.size() - 1);
  for (auto [i, value] : llvm::enumerate(values))
    if (static_cast<int64_t>(i) != index)
      result.push_back(value);
  return result;
}

std::optional<unsigned> findLoopIvIndex(Value value, ArrayRef<Value> ivs) {
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

OpFoldResult remapOpFoldResult(RewriterBase &rewriter, Location loc, OpFoldResult ofr, Value from,
                               Value to) {
  if (auto attr = dyn_cast<Attribute>(ofr))
    return attr;
  Value value = cast<Value>(ofr);
  if (value == from)
    return to;
  auto affineApply = value.getDefiningOp<affine::AffineApplyOp>();
  if (!affineApply || affineApply.getNumOperands() != 1 || affineApply.getOperand(0) != from)
    return value;
  return affine::AffineApplyOp::create(rewriter, loc, affineApply.getAffineMap(), ValueRange{to})
      .getResult();
}

SmallVector<OpFoldResult> remapOpFoldResults(RewriterBase &rewriter, Location loc,
                                             ArrayRef<OpFoldResult> values,
                                             ArrayRef<std::pair<Value, Value>> mapping) {
  SmallVector<OpFoldResult> remapped = llvm::to_vector(values);
  for (auto [from, to] : mapping)
    for (OpFoldResult &value : remapped)
      value = remapOpFoldResult(rewriter, loc, value, from, to);
  return remapped;
}

RankedTensorType inferTensorTypeFromMixedSizes(ArrayRef<OpFoldResult> sizes, Type elementType) {
  SmallVector<int64_t> shape;
  shape.reserve(sizes.size());
  for (OpFoldResult size : sizes) {
    auto maybeConst = getConstantIntValue(size);
    shape.push_back(maybeConst ? *maybeConst : ShapedType::kDynamic);
  }
  return RankedTensorType::get(shape, elementType);
}

void cloneSingleRegionBody(OpBuilder &builder, Location nestedLoc, Block &oldBlock,
                           ValueRange newArgs) {
  IRMapping mapping;
  for (auto [oldArg, newArg] : llvm::zip_equal(oldBlock.getArguments(), newArgs))
    mapping.map(oldArg, newArg);

  for (Operation &op : oldBlock.without_terminator())
    builder.clone(op, mapping);

  auto oldYield = cast<linalg::YieldOp>(oldBlock.getTerminator());
  SmallVector<Value> yielded;
  yielded.reserve(oldYield.getValues().size());
  for (Value value : oldYield.getValues())
    yielded.push_back(mapping.lookup(value));
  linalg::YieldOp::create(builder, nestedLoc, yielded);
}

linalg::GenericOp cloneGenericOnTile(RewriterBase &rewriter, linalg::GenericOp sourceGeneric,
                                     Value inputTile, Value initTile, Location loc) {
  return linalg::GenericOp::create(
      rewriter, loc, TypeRange{initTile.getType()}, ValueRange{inputTile}, ValueRange{initTile},
      sourceGeneric.getIndexingMapsArray(), sourceGeneric.getIteratorTypesArray(),
      [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
        cloneSingleRegionBody(builder, nestedLoc, sourceGeneric->getRegion(0).front(), newArgs);
      });
}

} // namespace

namespace mlir {
namespace transform {

DiagnosedSilenceableFailure LoopFuseIntoProducerOp::apply(transform::TransformRewriter &rewriter,
                                                          TransformResults &transformResults,
                                                          TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getConsumerOp, "consumer", consumer);
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getProducerLoop, "producer loop", loop);

  auto loopNest = collectLoopNestRootedAt(loop);
  if (failed(loopNest))
    return emitSilenceableFailure(transform,
                                  "expected containing loop to be an scf.for/scf.forall root of a "
                                  "supported tiled loop nest");

  FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult =
      scf::tileAndFuseConsumer(rewriter, consumer, *loopNest);
  if (failed(fuseResult))
    return emitSilenceableFailure(transform,
                                  "failed to tile and fuse elementwise consumer into loop");
  if (fuseResult->tiledOps.empty())
    return emitSilenceableFailure(transform,
                                  "consumer had no operands defined by the containing loop");

  if (isOpTriviallyDead(consumer))
    rewriter.eraseOp(consumer);

  transformResults.set(getOperation()->getResult(0), fuseResult->tiledOps);
  transformResults.set(getOperation()->getResult(1),
                       ArrayRef<Operation *>{loopNest->front().getOperation()});
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
LoopFuseReduceConsumerIntoForall::apply(transform::TransformRewriter &rewriter,
                                        TransformResults &transformResults, TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForallLoop, "loop", loop, scf::ForallOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getConsumerOp, "consumer", consumer,
                               linalg::GenericOp);
  if (failed(verifyUnarySingleReductionGeneric(consumer)))
    return emitSilenceableFailure(transform,
                                  "expected a unary single-reduction linalg.generic consumer");

  if (loop.getNumResults() != 1)
    return emitSilenceableFailure(transform, "expected the scf.forall to have exactly one result");

  auto producerResult = dyn_cast<OpResult>(consumer.getInputs().front());
  if (!producerResult || producerResult.getOwner() != loop.getOperation())
    return emitSilenceableFailure(
        transform, "expected the reduction input to be produced by the target scf.forall");

  FailureOr<tensor::ParallelInsertSliceOp> maybeProducerInsert =
      getParallelInsertSliceForLoopResult(loop, producerResult);
  if (failed(maybeProducerInsert) || !*maybeProducerInsert)
    return emitSilenceableFailure(
        transform, "expected the loop result consumed by the reduction to be combined via "
                   "tensor.parallel_insert_slice");
  auto producerInsert = *maybeProducerInsert;

  auto reductionDimOpt = getSingleReductionDim(consumer);
  if (!reductionDimOpt.has_value())
    return emitSilenceableFailure(transform, "expected a single reduction dimension");
  uint64_t reductionDim = *reductionDimOpt;
  int64_t reductionDimI64 = static_cast<int64_t>(reductionDim);
  OpFoldResult reductionOffset = producerInsert.getMixedOffsets()[reductionDim];
  Value reductionOffsetValue = dyn_cast<Value>(reductionOffset);
  if (!reductionOffsetValue)
    return emitSilenceableFailure(transform, "expected the reduced producer dimension to "
                                             "have a dynamic tile offset");
  std::optional<unsigned> removedIvIndex =
      findLoopIvIndex(reductionOffsetValue, loop.getInductionVars());
  if (!removedIvIndex)
    return emitSilenceableFailure(transform,
                                  "expected the reduced producer dimension to be controlled by one "
                                  "scf.forall induction variable");

  // Step 1. Rebuild the outer forall without the reduction-tiled dimension and
  // add the reduction result tensor as a new shared output.
  auto outerLbs = dropAt(loop.getMixedLowerBound(), *removedIvIndex);
  auto outerUbs = dropAt(loop.getMixedUpperBound(), *removedIvIndex);
  auto outerSteps = dropAt(loop.getMixedStep(), *removedIvIndex);
  SmallVector<Value> newOutputs = llvm::to_vector(loop.getOutputs());
  Value reductionInit = consumer.getDpsInits().front();
  newOutputs.push_back(reductionInit);
  rewriter.setInsertionPoint(consumer);
  auto newForall = scf::ForallOp::create(rewriter, loop.getLoc(), outerLbs, outerUbs, outerSteps,
                                         newOutputs, std::nullopt, nullptr);

  // Step 2. Materialize the outer-thread slices that will be updated by the
  // new inner sequential loop.
  rewriter.setInsertionPointToStart(newForall.getBody());
  Value outerProducerArg = newForall.getRegionOutArgs().front();
  Value outerReductionArg = newForall.getRegionOutArgs().back();
  Value removedIv = loop.getInductionVars()[*removedIvIndex];
  Value removedLb = getValueOrCreateConstantIndexOp(rewriter, loop.getLoc(),
                                                    loop.getMixedLowerBound()[*removedIvIndex]);
  Value removedUb = getValueOrCreateConstantIndexOp(rewriter, loop.getLoc(),
                                                    loop.getMixedUpperBound()[*removedIvIndex]);
  Value removedStep = getValueOrCreateConstantIndexOp(rewriter, loop.getLoc(),
                                                      loop.getMixedStep()[*removedIvIndex]);

  SmallVector<std::pair<Value, Value>> outerIvMapping;
  outerIvMapping.reserve(loop.getInductionVars().size());
  unsigned newOuterIvPos = 0;
  for (auto [index, oldIv] : llvm::enumerate(loop.getInductionVars())) {
    if (index == *removedIvIndex) {
      outerIvMapping.emplace_back(oldIv, removedLb);
      continue;
    }
    outerIvMapping.emplace_back(oldIv, newForall.getInductionVars()[newOuterIvPos++]);
  }

  SmallVector<OpFoldResult> panelOffsets =
      remapOpFoldResults(rewriter, loop.getLoc(), producerInsert.getMixedOffsets(), outerIvMapping);
  SmallVector<OpFoldResult> panelSizes = llvm::to_vector(producerInsert.getMixedSizes());
  auto producerType = cast<RankedTensorType>(producerResult.getType());
  if (producerType.isDynamicDim(reductionDim)) {
    panelSizes[reductionDim] =
        tensor::DimOp::create(rewriter, loop.getLoc(), producerResult, reductionDimI64).getResult();
  } else {
    panelSizes[reductionDim] = rewriter.getIndexAttr(producerType.getDimSize(reductionDim));
  }
  SmallVector<OpFoldResult> unitStrides(panelOffsets.size(), rewriter.getIndexAttr(1));
  auto panelType = inferTensorTypeFromMixedSizes(panelSizes, producerType.getElementType());
  Value panelInit = tensor::ExtractSliceOp::create(
      rewriter, loop.getLoc(), panelType, outerProducerArg, panelOffsets, panelSizes, unitStrides);

  auto reductionType = cast<RankedTensorType>(reductionInit.getType());
  SmallVector<OpFoldResult> reductionOffsets =
      remapOpFoldResults(rewriter, loop.getLoc(),
                         dropAt(producerInsert.getMixedOffsets(), reductionDimI64), outerIvMapping);
  SmallVector<OpFoldResult> reductionSizes =
      dropAt(producerInsert.getMixedSizes(), reductionDimI64);
  SmallVector<OpFoldResult> reductionStrides(reductionOffsets.size(), rewriter.getIndexAttr(1));
  auto reductionTileType =
      inferTensorTypeFromMixedSizes(reductionSizes, reductionType.getElementType());
  Value reductionTileInit =
      tensor::ExtractSliceOp::create(rewriter, loop.getLoc(), reductionTileType, outerReductionArg,
                                     reductionOffsets, reductionSizes, reductionStrides);

  // Step 3. Create the inner sequential loop, clone the producer tile
  // computation into it, insert the producer tile into the local panel, and
  // thread the reduction state as an scf.for iter_arg.
  auto innerFor = scf::ForOp::create(rewriter, loop.getLoc(), removedLb, removedUb, removedStep,
                                     ValueRange{panelInit, reductionTileInit});
  rewriter.setInsertionPointToStart(innerFor.getBody());

  IRMapping mapping;
  newOuterIvPos = 0;
  for (auto [index, oldIv] : llvm::enumerate(loop.getInductionVars())) {
    if (index == *removedIvIndex) {
      mapping.map(oldIv, innerFor.getInductionVar());
      continue;
    }
    mapping.map(oldIv, newForall.getInductionVars()[newOuterIvPos++]);
  }
  mapping.map(loop.getRegionOutArgs().front(), outerProducerArg);
  for (Operation &op : loop.getBody()->without_terminator())
    rewriter.clone(op, mapping);

  Value mappedTile = mapping.lookup(producerInsert.getSource());
  SmallVector<OpFoldResult> localPanelOffsets;
  localPanelOffsets.reserve(panelOffsets.size());
  for (auto [dim, offset] : llvm::enumerate(producerInsert.getMixedOffsets())) {
    if (dim == reductionDim) {
      localPanelOffsets.push_back(remapOpFoldResult(rewriter, loop.getLoc(), offset, removedIv,
                                                    innerFor.getInductionVar()));
      continue;
    }
    localPanelOffsets.push_back(rewriter.getIndexAttr(0));
  }
  SmallVector<OpFoldResult> tileStrides(localPanelOffsets.size(), rewriter.getIndexAttr(1));
  auto updatedPanel = tensor::InsertSliceOp::create(
      rewriter, loop.getLoc(), mappedTile, innerFor.getRegionIterArgs()[0], localPanelOffsets,
      producerInsert.getMixedSizes(), tileStrides);
  auto fusedReduction = cloneGenericOnTile(rewriter, consumer, mappedTile,
                                           innerFor.getRegionIterArgs()[1], consumer.getLoc());
  scf::YieldOp::create(rewriter, loop.getLoc(),
                       ValueRange{updatedPanel.getResult(), fusedReduction.getResult(0)});

  // Step 4. Publish the completed producer panel and reduced tile once per
  // outer forall instance, then replace the old loop/result pair.
  rewriter.setInsertionPointToEnd(&newForall.getTerminator().getRegion().front());
  tensor::ParallelInsertSliceOp::create(rewriter, loop.getLoc(), innerFor.getResult(0),
                                        outerProducerArg, panelOffsets, panelSizes, unitStrides);
  tensor::ParallelInsertSliceOp::create(rewriter, loop.getLoc(), innerFor.getResult(1),
                                        outerReductionArg, reductionOffsets, reductionSizes,
                                        reductionStrides);

  rewriter.replaceOp(consumer, newForall.getResults().back());
  rewriter.replaceOp(loop, newForall.getResults().take_front(loop.getNumResults()));

  transformResults.set(getOperation()->getResult(0),
                       ArrayRef<Operation *>{fusedReduction.getOperation()});
  transformResults.set(getOperation()->getResult(1),
                       ArrayRef<Operation *>{newForall.getOperation()});
  transformResults.set(getOperation()->getResult(2),
                       ArrayRef<Operation *>{innerFor.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace mlir
