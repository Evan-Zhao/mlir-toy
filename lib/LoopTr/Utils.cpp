#include "LoopTr/Utils.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {

namespace {

/// Dispatch to the appropriate loop result mediator getter based on the loop type.
template <typename LoopOp> struct GetLoopResults;

template <> struct GetLoopResults<scf::ForallOp> {
  static FailureOr<LoopResultRelay> get(scf::ForallOp loop, OpResult result) {
    auto mediator = getParallelInsertSliceForLoopResult(loop, result);
    if (failed(mediator))
      return failure();
    auto source = dyn_cast<OpResult>(mediator->getSource());
    if (!source)
      return failure();
    return LoopResultRelay{
        .inLoopResult = source, .loopReturnResult = result, .mediator = *mediator};
  }
};

template <> struct GetLoopResults<scf::ForOp> {
  static FailureOr<LoopResultRelay> get(scf::ForOp loop, OpResult result) {
    if (result.getOwner() != loop.getOperation())
      return failure();
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    auto source = dyn_cast<OpResult>(yield.getOperand(result.getResultNumber()));
    if (!source)
      return failure();
    // If the source of the `yield` operand is an insert-slice operation, then the real source
    // is the source of this `insertSlice`.
    auto insertSlice = dyn_cast<tensor::InsertSliceOp>(source.getDefiningOp());
    if (insertSlice) {
      source = dyn_cast<OpResult>(insertSlice.getSource());
      if (!source)
        return failure();
    }
    return LoopResultRelay{
        .inLoopResult = source, .loopReturnResult = result, .mediator = insertSlice};
  }
};

template <typename LoopOp>
FailureOr<SmallVector<LoopResultRelay>> getLoopResultRelays(LoopOp loop) {
  SmallVector<LoopResultRelay> relays;
  relays.reserve(loop.getNumResults());
  for (auto [index, result] : llvm::enumerate(loop.getResults())) {
    auto resultRelay = GetLoopResults<LoopOp>::get(loop, result);
    if (failed(resultRelay)) {
      loop->emitError() << "failed to get loop result relay for result " << index
                        << " of this loop";
      return failure();
    }
    relays.push_back(*resultRelay);
  }
  return relays;
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

} // namespace

FailureOr<tensor::ParallelInsertSliceOp> getParallelInsertSliceForLoopResult(scf::ForallOp loop,
                                                                             OpResult result) {
  if (result.getOwner() != loop.getOperation())
    return failure();
  BlockArgument bbArg = loop.getTiedBlockArgument(result);
  SmallVector<Operation *> combiningOps = loop.getCombiningOps(bbArg);
  if (!llvm::hasSingleElement(combiningOps))
    return failure();
  auto insertSlice = dyn_cast<tensor::ParallelInsertSliceOp>(combiningOps.front());
  if (!insertSlice)
    return failure();
  return insertSlice;
}

FailureOr<SmallVector<SmallVector<LoopResultRelay>>>
getNestedLoopResultRelays(ArrayRef<Operation *> loops) {
  SmallVector<SmallVector<LoopResultRelay>> relaysByLoop;
  relaysByLoop.reserve(loops.size());
  for (auto [index, loop] : llvm::enumerate(loops)) {
    if (index + 1 < loops.size() && loops[index + 1]->getParentOp() != loop)
      return failure();
    FailureOr<SmallVector<LoopResultRelay>> relays = failure();
    if (auto forall = dyn_cast<scf::ForallOp>(loop)) {
      relays = getLoopResultRelays(forall);
    } else if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
      relays = getLoopResultRelays(forOp);
    }
    if (failed(relays))
      return failure();
    relaysByLoop.push_back(*relays);
  }
  return relaysByLoop;
}

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

linalg::GenericOp cloneGenericOnTile(RewriterBase &rewriter, linalg::GenericOp sourceGeneric,
                                     Value inputTile, Value initTile, Location loc) {
  return linalg::GenericOp::create(
      rewriter, loc, TypeRange{initTile.getType()}, ValueRange{inputTile}, ValueRange{initTile},
      sourceGeneric.getIndexingMapsArray(), sourceGeneric.getIteratorTypesArray(),
      [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
        cloneSingleRegionBody(builder, nestedLoc, sourceGeneric->getRegion(0).front(), newArgs);
      });
}

} // namespace mlir
