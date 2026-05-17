#include "LoopTr/Utils.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {

namespace {

LogicalResult isElementwiseMap(linalg::MapOp map) {
  if (map.getNumDpsInits() != 1)
    return failure();
  if (map->getNumResults() != 1)
    return failure();
  return success();
}

LogicalResult isElementwiseGeneric(linalg::GenericOp generic) {
  if (generic.getNumDpsInits() != 1)
    return failure();
  if (generic->getNumResults() != 1)
    return failure();
  if (!generic.isAllParallelLoops())
    return failure();
  if (!generic.hasPureTensorSemantics())
    return failure();
  if (!llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isProjectedPermutation(); }))
    return failure();
  if (!generic.getIndexingMapsArray().back().isIdentity())
    return failure();
  return success();
}

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

LogicalResult isSingleOutputElemwiseLinalgOp(Operation *op) {
  if (auto map = dyn_cast<linalg::MapOp>(op))
    return isElementwiseMap(map);
  if (auto generic = dyn_cast<linalg::GenericOp>(op))
    return isElementwiseGeneric(generic);
  return failure();
}

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

FailureOr<DenseMap<OpResult, LoopResultRelaysT>>
getChainedLoopResultMap(ArrayRef<Operation *> loops) {
  auto loopResultRelaysF = getNestedLoopResultRelays(loops);
  if (failed(loopResultRelaysF))
    return failure();
  const auto &loopResultRelays = *loopResultRelaysF;
  if (loopResultRelays.empty())
    return {};

  // Start from the innermost loop: each of its returned results is directly
  // associated with the in-loop OpResult that computes it.
  DenseMap<OpResult, LoopResultRelaysT> chainedMap;
  for (const LoopResultRelay &relay : loopResultRelays.back()) {
    chainedMap.try_emplace(relay.loopReturnResult, SmallVector<LoopResultRelay>{relay});
  }

  // Walk outward and keep only those relays that continue the chain all the
  // way to the innermost loop. After processing loop i, `chainedMap` is keyed
  // by results of loops[i].
  for (auto relayIt = loopResultRelays.rbegin() + 1; relayIt != loopResultRelays.rend();
       ++relayIt) {
    DenseMap<OpResult, LoopResultRelaysT> nextMap;
    for (const LoopResultRelay &relay : *relayIt) {
      auto cMapIt = chainedMap.find(relay.inLoopResult);
      if (cMapIt == chainedMap.end())
        continue;
      // Safe to move because each key is visited at most once.
      auto nextRelays = cMapIt->second;
      nextRelays.push_back(relay);
      nextMap.try_emplace(relay.loopReturnResult, std::move(nextRelays));
    }
    chainedMap = std::move(nextMap);
  }
  return chainedMap;
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

SmallVector<OpFoldResult> getMixedTensorSizes(RewriterBase &rewriter, Location loc, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  SmallVector<OpFoldResult> sizes;
  sizes.reserve(tensorType.getRank());
  for (auto [dim, size] : llvm::enumerate(tensorType.getShape())) {
    if (ShapedType::isDynamic(size)) {
      sizes.push_back(
          tensor::DimOp::create(rewriter, loc, tensor, static_cast<unsigned>(dim)).getResult());
    } else {
      sizes.push_back(rewriter.getIndexAttr(size));
    }
  }
  return sizes;
}

FailureOr<Value> cloneValueDefChainAtInsertionPoint(RewriterBase &rewriter, Value value,
                                                    IRMapping &mapping) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  Operation *def = value.getDefiningOp();
  if (!def)
    return value;

  Block *insertBlock = rewriter.getInsertionBlock();
  auto insertPoint = rewriter.getInsertionPoint();
  Operation *insertPointOp = insertPoint == insertBlock->end() ? nullptr : &*insertPoint;
  if (!insertPointOp || def->getBlock() != insertBlock || !insertPointOp->isBeforeInBlock(def))
    return value;

  IRMapping localMapping = mapping;
  for (Value operand : def->getOperands()) {
    FailureOr<Value> remappedOperand =
        cloneValueDefChainAtInsertionPoint(rewriter, operand, mapping);
    if (failed(remappedOperand))
      return failure();
    localMapping.map(operand, *remappedOperand);
  }

  Operation *cloned = rewriter.clone(*def, localMapping);
  for (auto [oldResult, newResult] : llvm::zip_equal(def->getResults(), cloned->getResults()))
    mapping.map(oldResult, newResult);
  return mapping.lookup(value);
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
