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
#include <variant>

using namespace mlir;

namespace {

using transform::TransformOpInterface;

FailureOr<uint64_t> matchUnarySingleReductionGeneric(linalg::GenericOp generic) {
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
  std::optional<uint64_t> reductionDim;
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
  int64_t reductionDimI64 = static_cast<int64_t>(*reductionDim);
  for (int64_t dim = 0, outIdx = 0, e = inputType.getRank(); dim < e; ++dim) {
    if (dim == reductionDimI64)
      continue;
    auto expr = outputMap.getResult(outIdx++);
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() != dim)
      return failure();
  }
  return *reductionDim;
}

/// Give one of the output values (`result`) that `loop` produces, returns the
/// `tensor::ParallelInsertSliceOp` within `loop` that produces `result`.
FailureOr<tensor::ParallelInsertSliceOp> getParallelInsertSliceForLoopResult(scf::ForallOp loop,
                                                                             OpResult result) {
  if (result.getOwner() != loop.getOperation())
    return failure();
  BlockArgument bbArg = loop.getTiedBlockArgument(result);
  SmallVector<Operation *> combiningOps = loop.getCombiningOps(bbArg);
  if (combiningOps.size() != 1)
    return failure();
  auto insertSlice = dyn_cast<tensor::ParallelInsertSliceOp>(combiningOps.front());
  if (!insertSlice)
    return failure();
  return insertSlice;
}

std::variant<tensor::ParallelInsertSliceOp, DiagnosedSilenceableFailure>
getParallelInsertSliceForLoopResult(TransformOpInterface transform, scf::ForallOp loop,
                                    OpResult result) {
  auto insertSliceOrFailure = getParallelInsertSliceForLoopResult(loop, result);
  if (failed(insertSliceOrFailure)) {
    loop->emitRemark("when analyzing this loop (scf.forall); result = ") << result;
    return emitSilenceableFailure(transform,
                                  "failed to find the tensor.parallel_insert_slice operation in "
                                  "the loop that published a loop result");
  }
  return *insertSliceOrFailure;
}

/// Returns the index of the loop IV that directly defines `value`, or that
/// feeds a trivial one-operand `affine.apply` producing `value`.
///
/// Example that matches: `%i` itself, or `%t = affine.apply #map(%i)` where
/// `%i` is one of `ivs`.
/// Example that does not match: `%t = affine.apply #map(%i, %j)`, or any value
/// produced by a deeper expression tree that is not a single-step wrapper of
/// one IV.
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

/// A plan for splitting an `scf.forall` loop `l`, whose result is consumed by a reduction `r`,
/// into two loops: an outer `scf.forall` loop and an inner `scf.for` loop.
struct ReductionForallSplitPlan {
  /// The result of `l` that is consumed by `r`.
  Value producerResult;
  /// The "yield" operation inside `l` that produces `producerResult`.
  tensor::ParallelInsertSliceOp producerInsert;
  /// The index of the iter-var (IV) in the forall loop `l` (because a forall loop may be
  /// multi-dimensional) to remove from the outer loop and re-materialize as `scf.for`.
  unsigned removedIvIndex;
  /// Reduction dimension in the consumer and producer tile.
  uint64_t reductionDim;
  /// The initial value of the reduction.
  Value reductionInit;
};

std::variant<ReductionForallSplitPlan, DiagnosedSilenceableFailure>
detectReductionForallSplit(const TransformOpInterface &transform, scf::ForallOp loop,
                           linalg::GenericOp consumer, uint64_t reductionDim) {
  // Step 1. Verify producer-consumer relationship by finding the value that connects them; extract
  // the parallel-insert-slice op in the loop that produces the value.
  auto producerResult = dyn_cast<OpResult>(consumer.getInputs().front());
  if (!producerResult || producerResult.getOwner() != loop.getOperation())
    return emitSilenceableFailure(
        transform, "expected the reduction input to be produced by the target scf.forall");
  RETURN_DIAGNOSTICS_OR_BIND_VAL(
      tensor::ParallelInsertSliceOp, producerInsert,
      getParallelInsertSliceForLoopResult(transform, loop, producerResult));

  // Step 2. Recover which producer tile dimension is reduced and confirm it
  // is controlled by a single forall IV.
  OpFoldResult reductionOffset = producerInsert.getMixedOffsets()[reductionDim];
  Value reductionOffsetValue = dyn_cast<Value>(reductionOffset);
  if (!reductionOffsetValue)
    return emitSilenceableFailure(transform, "expected the reduced producer dimension to "
                                             "have a dynamic tile offset");
  std::optional<unsigned> removedIvIndex =
      findLoopIvIndex(reductionOffsetValue, loop.getInductionVars());
  if (!removedIvIndex) {
    producerInsert->emitRemark() << "dim " << reductionDim << " of this op has offset "
                                 << reductionOffsetValue << ", which violates our assumptions";
    return emitSilenceableFailure(
        transform, "expected the reduced producer dimension to be controlled by a single "
                   "scf.forall induction variable");
  }

  // Step 3. Package the split recipe for the actual loop rewrite.
  return ReductionForallSplitPlan{
      .producerResult = producerResult,
      .producerInsert = producerInsert,
      .removedIvIndex = *removedIvIndex,
      .reductionDim = reductionDim,
      .reductionInit = consumer.getDpsInits().front(),
  };
}

/// Remaps an affine index `ofr` using the given `mapping`, returning a failure if the remap fails.
/// We'll attempt to remap `ofr` directly, then check if it is an affine apply and remap its
/// operand. `ofr` could also be a constant attribute.
FailureOr<OpFoldResult> remapAffineIndex(RewriterBase &rewriter, Location loc, OpFoldResult ofr,
                                         const DenseMap<Value, Value> &mapping) {
  // If `ofr` is an attribute (constant), return it as-is.
  if (auto attr = dyn_cast<Attribute>(ofr))
    return success(attr);
  // Handle the special case where `ofr` is `from`, returning `to` instead.
  Value value = cast<Value>(ofr);
  if (auto it = mapping.find(value); it != mapping.end())
    return success(it->second);
  // If `ofr` is an affine apply, remap its operand if it matches `from`.
  auto affineApply = value.getDefiningOp<affine::AffineApplyOp>();
  if (!affineApply)
    return failure();
  // Remap each operand of the affine apply.
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

/// Remaps multiple affine indices in `values` using the given `mapping`, returning a failure if any
/// remap fails.
FailureOr<SmallVector<OpFoldResult>> remapAffineIndices(RewriterBase &rewriter, Location loc,
                                                        SmallVector<OpFoldResult> values,
                                                        const DenseMap<Value, Value> &mapping) {
  for (OpFoldResult &value : values) {
    auto result = remapAffineIndex(rewriter, loc, value, mapping);
    if (failed(result)) {
      return failure();
    }
    value = *result;
  }
  return std::move(values);
}

SmallVector<OpFoldResult> getUnitStrides(RewriterBase &rewriter, size_t rank) {
  return SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(1));
}

Value createExtractSliceFromState(RewriterBase &rewriter, Location loc, Value fullTensor,
                                  ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides) {
  auto tensorType = cast<RankedTensorType>(fullTensor.getType());
  SmallVector<int64_t> shape;
  shape.reserve(sizes.size());
  for (OpFoldResult size : sizes) {
    auto maybeConst = getConstantIntValue(size);
    shape.push_back(maybeConst ? *maybeConst : ShapedType::kDynamic);
  }
  auto tileType = RankedTensorType::get(shape, tensorType.getElementType());
  return tensor::ExtractSliceOp::create(rewriter, loc, tileType, fullTensor, offsets, sizes,
                                        strides);
}

void pointRewriterToForallParallel(RewriterBase &rewriter, scf::ForallOp forall) {
  rewriter.setInsertionPointToEnd(&forall.getTerminator().getRegion().front());
}

struct SplitForallIntoForResult {
  /// Slice metadata used to extract or publish a tile relative to a loop-carried tensor.
  struct TileSlice {
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
  };

  /// Rebuilt outer forall with the reduction dimension removed.
  scf::ForallOp newForall;
  /// New inner sequential loop that streams along the removed dimension.
  scf::ForOp innerFor;
  /// Recalculated tiles for the return value of the forall loop.
  Value outerTile, innerTile;
  /// Slice of the reduction tile assembled by one outer forall instance.
  TileSlice reductionSlice;
};

/// Rebuilds `loop` as an outer `scf.forall` plus an inner streaming `scf.for`
/// according to `plan`, and clones the old forall body into the new inner loop.
/// In addition, add a new output/carry argument on the new forall loop and the new for loop for the
/// reduction.
/// This function does these two things at once because it is hard to modify loops for additional
/// carry arguments after the loop has been created.
std::variant<SplitForallIntoForResult, DiagnosedSilenceableFailure>
splitForallDimensionForReduction(TransformOpInterface transform, RewriterBase &rewriter,
                                 scf::ForallOp loop, ReductionForallSplitPlan &plan) {
  auto loc = loop.getLoc();
  // Step 1. Create the replacement outer forall with the reduced dimension
  // removed and the reduction result added as a new shared output.
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

  // Step 2. Materialize the tile owned by this outer forall instance.
  // Build `outerIvMapping`, which maps the IV of the removed loop dimension to the IV's lower bound
  // (like 0).
  IRMapping outerIvMapping;
  unsigned newOuterIvPos = 0;
  auto newLoopIvs = newForall.getInductionVars();
  for (auto [index, oldIv] : llvm::enumerate(loop.getInductionVars())) {
    if (index == plan.removedIvIndex) {
      outerIvMapping.map(oldIv, innerLbV);
    } else {
      outerIvMapping.map(oldIv, newForall.getInductionVars()[newOuterIvPos++]);
    }
  }
  // Also map old loop output -> new forall loop output
  outerIvMapping.map(loop.getRegionOutArgs().front(), outerProducerArg);

  // Use the map to remap the outer tile offsets and sizes, and compute the unit strides.
  // Eventually get `tileInit`, which will be used in the inner for-loop.
  auto outerTileOffsetsF = remapAffineIndices(rewriter, loc, plan.producerInsert.getMixedOffsets(),
                                              outerIvMapping.getValueMap());
  if (failed(outerTileOffsetsF)) {
    plan.producerInsert->emitRemark() << "failed to remap offsets of this operation";
    return emitSilenceableFailure(transform, "failed to remap offsets of insert operation");
  }
  const SmallVector<OpFoldResult> &outerTileOffsets = *outerTileOffsetsF;
  // Tile size: expand the size of the tensor on the reduction dimension back to the full size,
  // because we're no longer tiling along it.
  SmallVector<OpFoldResult> outerTileSizes = plan.producerInsert.getMixedSizes();
  auto producerType = cast<RankedTensorType>(plan.producerResult.getType());
  if (producerType.isDynamicDim(plan.reductionDim)) {
    outerTileSizes[plan.reductionDim] =
        tensor::DimOp::create(rewriter, loc, plan.producerResult,
                              static_cast<int64_t>(plan.reductionDim))
            .getResult();
  } else {
    outerTileSizes[plan.reductionDim] =
        rewriter.getIndexAttr(producerType.getDimSize(plan.reductionDim));
  }
  SmallVector<OpFoldResult> nDUnitStrides = getUnitStrides(rewriter, outerTileOffsets.size());
  Value tileInit = createExtractSliceFromState(rewriter, loc, outerProducerArg, outerTileOffsets,
                                               outerTileSizes, nDUnitStrides);

  // Step 3. Extract a slice of the reduction tile to use as the initial value for the inner
  // for-loop.
  auto dropAt = [](SmallVector<OpFoldResult> values, uint64_t index) {
    values.erase(values.begin() + index);
    return values;
  };
  auto redTileOffsets = dropAt(outerTileOffsets, plan.reductionDim);
  auto redTileSizes = dropAt(outerTileSizes, plan.reductionDim);
  auto nMinus1DStrides = getUnitStrides(rewriter, redTileOffsets.size());
  Value redTileInit = createExtractSliceFromState(rewriter, loc, outerReductionArg, redTileOffsets,
                                                  redTileSizes, nMinus1DStrides);

  // Step 4. Clone the old forall body under the new inner for-loop.
  // Remap the removed forall IV to the inner induction variable.
  auto forLoop = scf::ForOp::create(rewriter, loc, innerLbV, innerUbV, innerStepV,
                                    ValueRange{tileInit, redTileInit});
  outerIvMapping.map(loop.getInductionVars()[plan.removedIvIndex], forLoop.getInductionVar());
  rewriter.setInsertionPointToStart(forLoop.getBody());
  for (Operation &op : loop.getBody()->without_terminator())
    rewriter.clone(op, outerIvMapping);
  // This is the tile that this loop should write to.
  Value outerTile = outerIvMapping.lookup(plan.producerInsert.getSource());

  // Step 5. Go to the end of this loop, and create an insert-slice op for the for loop output.
  rewriter.setInsertionPointToEnd(forLoop.getBody());
  auto offsets = plan.producerInsert.getMixedOffsets();
  // Offset values are all zero except for the reduction dimension, which is remapped to use the new
  // loop induction variable.
  SmallVector<OpFoldResult> localOffsets(offsets.size(), rewriter.getIndexAttr(0));
  localOffsets[plan.reductionDim] =
      *remapAffineIndex(rewriter, loc, offsets[plan.reductionDim], outerIvMapping.getValueMap());
  Value innerTile = tensor::InsertSliceOp::create(
      rewriter, loc, outerTile, forLoop.getRegionIterArgs()[0], localOffsets,
      plan.producerInsert.getMixedSizes(), nDUnitStrides);

  // Step 5. Add a parallel insert slice op for the forall output (the old one, not the reduction
  // one), to the new forall body.
  pointRewriterToForallParallel(rewriter, newForall);
  tensor::ParallelInsertSliceOp::create(rewriter, loc, forLoop.getResult(0), outerProducerArg,
                                        outerTileOffsets, outerTileSizes, nDUnitStrides);
  return SplitForallIntoForResult{.newForall = newForall,
                                  .innerFor = forLoop,
                                  .outerTile = outerTile,
                                  .innerTile = innerTile,
                                  .reductionSlice = {.offsets = std::move(redTileOffsets),
                                                     .sizes = std::move(redTileSizes),
                                                     .strides = std::move(nMinus1DStrides)}};
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

  // Step 1. Resolve the payload ops and check that the producer is loop-like.
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getConsumerOp, "consumer", consumer);
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getProducerLoop, "producer loop", loop);

  auto loopI = dyn_cast<LoopLikeOpInterface>(loop);
  if (!loopI)
    return emitSilenceableFailure(
        transform, "expected the producer loop to implement the LoopLikeOpInterface");

  // Step 2. Delegate the actual tile-and-fuse rewrite to the upstream SCF utility.
  FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseResult =
      scf::tileAndFuseConsumer(rewriter, consumer, {loopI});
  if (failed(fuseResult))
    return emitSilenceableFailure(transform,
                                  "failed to tile and fuse elementwise consumer into loop");
  if (fuseResult->tiledOps.empty())
    return emitSilenceableFailure(transform,
                                  "consumer had no operands defined by the containing loop");

  // Step 3. Clean up the old consumer if it became dead and publish the new handles.
  if (isOpTriviallyDead(consumer))
    rewriter.eraseOp(consumer);

  transformResults.set(getOperation()->getResult(0), fuseResult->tiledOps);
  transformResults.set(getOperation()->getResult(1), {loopI.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
LoopFuseReduceConsumerIntoForall::apply(transform::TransformRewriter &rewriter,
                                        TransformResults &transformResults, TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForallLoop, "loop", loop, scf::ForallOp);
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getConsumerOp, "consumer", consumer,
                               linalg::GenericOp);
  FailureOr<uint64_t> reductionDim = matchUnarySingleReductionGeneric(consumer);
  if (failed(reductionDim))
    return emitSilenceableFailure(transform,
                                  "expected a unary single-reduction linalg.generic consumer");

  // Step 1. Detect which forall IV should become the streaming for-loop.
  RETURN_DIAGNOSTICS_OR_BIND_VAL(
      ReductionForallSplitPlan, splitPlan,
      detectReductionForallSplit(transform, loop, consumer, *reductionDim));

  // Step 2. Split the forall skeleton first, without fusing the reduction yet.
  rewriter.setInsertionPoint(consumer);
  RETURN_DIAGNOSTICS_OR_BIND_VAL(
      SplitForallIntoForResult, split,
      splitForallDimensionForReduction(transform, rewriter, loop, splitPlan));

  // Step 3. Fuse the reduction operation under the new inner for, and create a yield op for it.
  rewriter.setInsertionPointToEnd(split.innerFor.getBody());
  auto fusedReduction =
      cloneGenericOnTile(rewriter, consumer, split.outerTile, split.innerFor.getRegionIterArgs()[1],
                         consumer.getLoc());
  scf::YieldOp::create(rewriter, loop.getLoc(),
                       ValueRange{split.innerTile, fusedReduction.getResult(0)});

  // Step 4. Publish the completed reduced tile once per outer forall instance, then replace the old
  // loop/result pair.
  pointRewriterToForallParallel(rewriter, split.newForall);
  Value outerReductionArg = split.newForall.getRegionOutArgs().back();
  tensor::ParallelInsertSliceOp::create(rewriter, loop.getLoc(), split.innerFor.getResult(1),
                                        outerReductionArg, split.reductionSlice.offsets,
                                        split.reductionSlice.sizes, split.reductionSlice.strides);

  rewriter.replaceOp(consumer, split.newForall.getResults().back());
  rewriter.replaceOp(loop, split.newForall.getResults().take_front(loop.getNumResults()));

  transformResults.set(getOperation()->getResult(0), {fusedReduction.getOperation()});
  transformResults.set(getOperation()->getResult(1), {split.newForall.getOperation()});
  transformResults.set(getOperation()->getResult(2), {split.innerFor.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace mlir
