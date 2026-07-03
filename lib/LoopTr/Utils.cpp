#include "LoopTr/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"

namespace mlir {

namespace {

void replaceUsesAfterElementwiseFusion(RewriterBase &rewriter,
                                       const linalg::ElementwiseOpFusionResult &fusionResult,
                                       Operation *producer) {
  for (auto &[origVal, replacement] : fusionResult.replacements) {
    if (origVal.getDefiningOp() != producer)
      rewriter.replaceUsesWithIf(origVal, replacement,
                                 [&](OpOperand &use) { return use.getOwner() != producer; });
  }
}

FailureOr<Operation *> tryDirectElementwiseFusion(transform::TransformRewriter &rewriter,
                                                  linalg::LinalgOp linalgTarget,
                                                  int64_t operandNumber) {
  OpOperand &fusedOperand = linalgTarget->getOpOperand(static_cast<unsigned>(operandNumber));
  if (!linalg::areElementwiseOpsFusable(&fusedOperand))
    return static_cast<Operation *>(nullptr);

  Operation *producer = fusedOperand.get().getDefiningOp();
  rewriter.setInsertionPoint(linalgTarget);
  FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
      linalg::fuseElementwiseOps(rewriter, &fusedOperand);
  if (failed(fusionResult))
    return static_cast<Operation *>(nullptr);

  replaceUsesAfterElementwiseFusion(rewriter, *fusionResult, producer);
  if (failed(rewriter.notifyPayloadOperationReplaced(linalgTarget, fusionResult->fusedOp)))
    return failure();
  rewriter.eraseOp(linalgTarget);
  return fusionResult->fusedOp;
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

void cloneSingleRegionBody(OpBuilder &builder, Location nestedLoc, Block &oldBlock,
                           ValueRange newArgs) {
  IRMapping mapping;
  for (auto [oldArg, newArg] : llvm::zip_equal(oldBlock.getArguments(), newArgs))
    mapping.map(oldArg, newArg);

  cloneBlockWithoutTerminator(builder, oldBlock, mapping);

  auto oldYield = cast<linalg::YieldOp>(oldBlock.getTerminator());
  SmallVector<Value> yielded;
  yielded.reserve(oldYield.getValues().size());
  for (Value value : oldYield.getValues())
    yielded.push_back(mapping.lookup(value));
  linalg::YieldOp::create(builder, nestedLoc, yielded);
}

struct MatchFailureCaptureListener : public RewriterBase::ForwardingListener {
  using Base = RewriterBase::ForwardingListener;

  explicit MatchFailureCaptureListener(OpBuilder::Listener *previous) : Base(previous) {}

  void notifyMatchFailure(Location loc,
                          llvm::function_ref<void(Diagnostic &)> reasonCallback) override {
    // Preserve any existing listener behavior.
    Base::notifyMatchFailure(loc, reasonCallback);

    Diagnostic diag(loc, DiagnosticSeverity::Remark);
    reasonCallback(diag);

    std::string msg;
    llvm::raw_string_ostream os(msg);
    diag.print(os);
    os.flush();

    messages.push_back(std::move(msg));
  }

  llvm::SmallVector<std::string> messages;
};

/// Detects `tensor.insert_slice` operations that feed into the yield of a loop,
/// and moves them right before the yield. This reduces the chance of scf::tileAndFuseConsumer
/// getting confused.
LogicalResult sinkYieldInsertSlices(RewriterBase &rewriter, scf::ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (Value operand : yield.getOperands()) {
    auto insertSlice = operand.getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSlice)
      continue;
    if (insertSlice->getBlock() != yield->getBlock() ||
        !llvm::hasSingleElement(insertSlice->getUses()))
      return failure();
    rewriter.moveOpBefore(insertSlice, yield);
  }
  return success();
}

Operation *getCommonDefiningOp(ValueRange values) {
  DenseSet<Operation *> defOps;
  for (Value value : values) {
    if (value)
      defOps.insert(value.getDefiningOp());
  }
  if (defOps.contains(nullptr) || defOps.size() != 1)
    return nullptr;
  return *defOps.begin();
}

} // namespace

TrackedOperationListener::TrackedOperationListener(Operation *trackedOp,
                                                   OpBuilder::Listener *previous)
    : RewriterBase::ForwardingListener(previous), trackedOp(trackedOp) {}

void TrackedOperationListener::notifyOperationReplaced(Operation *op, Operation *newOp) {
  RewriterBase::ForwardingListener::notifyOperationReplaced(op, newOp);
  if (op == trackedOp)
    trackedOp = newOp;
}

void TrackedOperationListener::notifyOperationReplaced(Operation *op, ValueRange replacement) {
  RewriterBase::ForwardingListener::notifyOperationReplaced(op, replacement);
  if (op == trackedOp) {
    auto newOp = getCommonDefiningOp(replacement);
    if (!newOp || newOp->getName() != op->getName())
      trackedOp = nullptr;
    else
      trackedOp = newOp;
  }
}

void TrackedOperationListener::notifyOperationErased(Operation *op) {
  RewriterBase::ForwardingListener::notifyOperationErased(op);
  if (op == trackedOp)
    trackedOp = nullptr;
}

LogicalResult isSingleOutputElemwiseLinalgOp(Operation *op) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic)
    return failure();
  if (generic->getNumResults() != 1 || !generic.isAllParallelLoops() ||
      !generic.hasPureTensorSemantics())
    return failure();
  if (!llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isProjectedPermutation(); }))
    return failure();
  if (!generic.getIndexingMapsArray().back().isIdentity())
    return failure();
  return success();
}

FailureOr<uint64_t> getReductionIteratorIndex(linalg::GenericOp generic) {
  SmallVector<uint64_t> reductionDims;
  for (auto [index, iteratorType] : llvm::enumerate(generic.getIteratorTypesArray())) {
    if (iteratorType == utils::IteratorType::reduction)
      reductionDims.push_back(index);
  }
  if (reductionDims.size() != 1) {
    generic.emitError() << "expected exactly one reduction dimension, but got "
                        << reductionDims.size() << " reduction dimensions";
    return failure();
  }
  return reductionDims.front();
}

FailureOr<uint64_t> matchOneDimReductionGeneric(linalg::GenericOp generic) {
  if (generic.getNumDpsInits() != 1 || generic.getNumResults() != 1)
    return generic.emitError("expected exactly one init operand and one output");
  auto resultType = dyn_cast<RankedTensorType>(generic.getResults().front().getType());
  if (!resultType)
    return generic.emitError("expected output to be a ranked tensor");

  auto reductionDim = getReductionIteratorIndex(generic);
  if (failed(reductionDim))
    return failure();
  int64_t nLoops = static_cast<int64_t>(generic.getNumLoops());
  if (resultType.getRank() != nLoops - 1)
    return generic.emitError("expected output rank to match the number of non-reduction iterators");

  AffineMap outputMap = generic.getIndexingMapsArray().back();
  if (outputMap.getNumResults() != resultType.getRank())
    return failure();
  int64_t reductionDimI64 = static_cast<int64_t>(*reductionDim);
  for (int64_t dim = 0, outIdx = 0; dim < nLoops; ++dim) {
    if (dim == reductionDimI64)
      continue;
    auto expr = outputMap.getResult(outIdx++);
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() != dim)
      return generic.emitError(
          "expected output indexing map to be the iteration space with the reduction "
          "dimension dropped");
  }
  return *reductionDim;
}

SmallVector<std::pair<Operation *, Operation *>>
cloneBlockWithoutTerminator(OpBuilder &builder, Block &block, IRMapping &mapping) {
  SmallVector<std::pair<Operation *, Operation *>> clonedOps;
  for (Operation &op : block.without_terminator()) {
    Operation *cloned = builder.clone(op, mapping);
    clonedOps.emplace_back(&op, cloned);
  }
  return clonedOps;
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

LogicalResult recursiveMoveOperandsBeforeOp(Operation &toMoveOperands, RewriterBase &rewriter,
                                            Operation &moveBefore) {
  IRMapping mapping;
  rewriter.setInsertionPoint(&moveBefore);
  for (auto value : toMoveOperands.getOperands()) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      continue;
    // This function does not clone if the value is already defined before the insertion point of
    // the rewriter.
    auto newValue = cloneValueDefChainAtInsertionPoint(rewriter, result, mapping);
    if (failed(newValue)) {
      result.getDefiningOp()->emitRemark("when cloning this operation");
      return failure();
    }
    if (*newValue == value)
      continue;
    mapping.map(result, *newValue);
    rewriter.replaceAllUsesWith(result, *newValue);
    rewriter.eraseOp(result.getDefiningOp());
  }
  return success();
}

FailureOr<scf::SCFFuseConsumerOfSliceResult>
tileAndFuseConsumerWithDebug(RewriterBase &rewriter, Operation &consumer,
                             MutableArrayRef<LoopLikeOpInterface> loops) {
  OpBuilder::Listener *previousListener = rewriter.getListener();
  MatchFailureCaptureListener capture(previousListener);
  rewriter.setListener(&capture);
  auto restoreListener = llvm::scope_exit([&]() { rewriter.setListener(previousListener); });
  FailureOr<scf::SCFFuseConsumerOfSliceResult> result =
      scf::tileAndFuseConsumer(rewriter, &consumer, loops);
  if (failed(result)) {
    llvm::errs() << "\nCaptured match failures:\n";
    for (StringRef msg : capture.messages)
      llvm::errs() << "  - " << msg << "\n";
  }
  return result;
}

FailureOr<ElementwiseInlineResult>
greedyInlineElementwiseProducers(transform::TransformRewriter &rewriter, linalg::GenericOp target,
                                 std::optional<int64_t> operandNumber) {
  linalg::GenericOp currentOp = target;
  bool applied = false;
  while (true) {
    int64_t beginOperandNumber = operandNumber ? *operandNumber : 0;
    int64_t endOperandNumber = operandNumber ? beginOperandNumber + 1 : currentOp.getNumDpsInputs();
    bool changed = false;
    for (int64_t i = beginOperandNumber; i < endOperandNumber; ++i) {
      FailureOr<Operation *> folded = tryDirectElementwiseFusion(rewriter, currentOp, i);
      if (failed(folded))
        return failure();
      if (*folded) {
        currentOp = cast<linalg::GenericOp>(*folded);
        changed = applied = true;
        break;
      }
    }
    if (!changed)
      return ElementwiseInlineResult{.fusedOp = currentOp.getOperation(), .applied = applied};
  }
}

FailureOr<std::pair<Operation *, Operation *>>
tileAndFuseConsumerIntoDoubleLoops(RewriterBase &rewriter, scf::ForallOp &outerLoop,
                                   scf::ForOp &innerLoop, Operation &operation) {
#define BAIL_AND_POINT(msg)                                                                        \
  {                                                                                                \
    operation.emitError() << (msg);                                                                \
    operation.getParentOp()->emitRemark() << "outer scope:";                                       \
    return failure();                                                                              \
  }

  // We are going to use scf::tileAndFuseConsumer twice. While it takes a vector of loops, it can
  // only work with one scf.forall loop at a time.
  SmallVector<LoopLikeOpInterface> outerLoops{outerLoop};
  FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedIntoForall =
      tileAndFuseConsumerWithDebug(rewriter, operation, outerLoops);
  if (failed(fusedIntoForall))
    BAIL_AND_POINT("failed to fuse this operation into the outer loop");
  outerLoop = cast<scf::ForallOp>(outerLoops[0]);
  auto *outerFusedOp = fusedIntoForall->tiledOps[0];

  // Similarly, this first fusion may have inserted some operations after the inner loop, and we
  // move them before the inner loop.
  if (failed(recursiveMoveOperandsBeforeOp(*outerFusedOp, rewriter, *innerLoop)))
    BAIL_AND_POINT("failed to move operands before the inner loop");
  if (failed(sinkYieldInsertSlices(rewriter, innerLoop))) {
    innerLoop.emitError() << "failed to sink inner-loop insert_slice yield operands";
    return failure();
  }

  // Apply the same fusion on the inner loop.
  SmallVector<LoopLikeOpInterface> innerLoops{innerLoop};
  FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedIntoFor =
      tileAndFuseConsumerWithDebug(rewriter, *outerFusedOp, innerLoops);
  if (failed(fusedIntoFor))
    BAIL_AND_POINT("failed to fuse this operation into the inner loop");
  innerLoop = cast<scf::ForOp>(innerLoops[0]);
  auto *innerFusedOp = fusedIntoFor->tiledOps[0];
  return std::make_pair(outerFusedOp, innerFusedOp);
}

void eliminateLocalCommonSubexpressions(RewriterBase &rewriter, Operation *op) {
  DominanceInfo domInfo(op);
  eliminateCommonSubExpressions(rewriter, domInfo, op);
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
