#include "LoopTr/LoopTransformOps.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::transform {

namespace {

using OpPairs = SmallVector<std::pair<Operation *, Operation *>>;

#define BAIL(message) return emitSilenceableFailure(transform, message);

Operation *findEnclosingIsolatedFromAbove(Operation *op) {
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (current->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return current;
  }
  return nullptr;
}

bool isOverwriteOnlyDestUse(OpOperand &use) {
  if (auto fillOp = dyn_cast<linalg::FillOp>(use.getOwner()))
    return fillOp.getDpsInitOperand(0) == &use;

  auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
  if (!linalgOp || !linalgOp.isDpsInit(&use))
    return false;
  return !linalgOp.payloadUsesValueFromOperand(&use);
}

bool isEligibleScratchSlice(tensor::ExtractSliceOp extract) {
  if (!extract->hasOneUse())
    return false;
  OpOperand &use = *extract->use_begin();
  return isOverwriteOnlyDestUse(use);
}

Value makeEmptyLikeExtractSlice(RewriterBase &rewriter, tensor::ExtractSliceOp extract) {
  auto resultType = cast<RankedTensorType>(extract.getType());
  return tensor::EmptyOp::create(rewriter, extract.getLoc(), extract.getMixedSizes(),
                                 resultType.getElementType());
}

struct SliceFillOfEmpty final : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extract,
                                PatternRewriter &rewriter) const override {
    auto fill = extract.getSource().getDefiningOp<linalg::FillOp>();
    if (!fill || !fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
      return failure();

    Value empty = makeEmptyLikeExtractSlice(rewriter, extract);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(extract, fill.getInputs(), ValueRange{empty});
    return success();
  }
};

bool localizeScratchSlicesInFor(TransformRewriter &rewriter, scf::ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<tensor::ExtractSliceOp> toReplace;

  for (auto [index, iterArg] : llvm::enumerate(loop.getRegionIterArgs())) {
    if (!loop.getResult(index).use_empty())
      continue;

    auto insert = yield.getOperand(index).getDefiningOp<tensor::InsertSliceOp>();
    if (!insert || insert.getDest() != iterArg)
      continue;

    SmallVector<tensor::ExtractSliceOp> extracts;
    bool ok = true;
    for (OpOperand &use : iterArg.getUses()) {
      if (use.getOwner() == insert.getOperation()) {
        if (use.get() != iterArg) {
          ok = false;
          break;
        }
        continue;
      }

      auto extract = dyn_cast<tensor::ExtractSliceOp>(use.getOwner());
      if (!extract || extract.getSource() != iterArg || !isEligibleScratchSlice(extract)) {
        ok = false;
        break;
      }
      extracts.push_back(extract);
    }

    if (!ok)
      continue;
    llvm::append_range(toReplace, extracts);
  }

  for (tensor::ExtractSliceOp extract : toReplace) {
    rewriter.setInsertionPoint(extract);
    rewriter.replaceOp(extract, makeEmptyLikeExtractSlice(rewriter, extract));
  }
  return !toReplace.empty();
}

bool localizeScratchSlicesInForall(TransformRewriter &rewriter, scf::ForallOp loop) {
  SmallVector<tensor::ExtractSliceOp> toReplace;

  for (auto [index, result] : llvm::enumerate(loop.getResults())) {
    if (!result.use_empty())
      continue;

    auto insert = getParallelInsertSliceForLoopResult(loop, cast<OpResult>(result));
    if (failed(insert))
      continue;

    BlockArgument outArg = loop.getRegionOutArgs()[index];
    SmallVector<tensor::ExtractSliceOp> extracts;
    bool ok = true;
    for (OpOperand &use : outArg.getUses()) {
      if (use.getOwner() == insert->getOperation()) {
        if (use.get() != outArg) {
          ok = false;
          break;
        }
        continue;
      }

      auto extract = dyn_cast<tensor::ExtractSliceOp>(use.getOwner());
      if (!extract || extract.getSource() != outArg || !isEligibleScratchSlice(extract)) {
        ok = false;
        break;
      }
      extracts.push_back(extract);
    }

    if (!ok)
      continue;
    llvm::append_range(toReplace, extracts);
  }

  for (tensor::ExtractSliceOp extract : toReplace) {
    rewriter.setInsertionPoint(extract);
    rewriter.replaceOp(extract, makeEmptyLikeExtractSlice(rewriter, extract));
  }
  return !toReplace.empty();
}

bool localizeScratchSlices(TransformRewriter &rewriter, Operation *target) {
  bool changed = false;
  target->walk<WalkOrder::PostOrder>(
      [&](scf::ForOp loop) { changed |= localizeScratchSlicesInFor(rewriter, loop); });
  target->walk<WalkOrder::PostOrder>(
      [&](scf::ForallOp loop) { changed |= localizeScratchSlicesInForall(rewriter, loop); });
  return changed;
}

bool dropUnusedScratchForResults(TransformRewriter &rewriter, scf::ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  BitVector resultsToDrop(loop.getNumResults(), false);
  BitVector bodyArgsToDrop(loop.getBody()->getNumArguments(), false);
  BitVector operandsToDrop(loop->getNumOperands(), false);
  SmallVector<Operation *> insertsToErase;

  for (auto [index, result] : llvm::enumerate(loop.getResults())) {
    if (!result.use_empty())
      continue;

    BlockArgument iterArg = loop.getRegionIterArg(index);
    auto insert = yield.getOperand(index).getDefiningOp<tensor::InsertSliceOp>();
    if (!insert || insert.getDest() != iterArg || !insert->hasOneUse())
      continue;

    bool onlyUsedByInsert = llvm::all_of(
        iterArg.getUses(), [&](OpOperand &use) { return use.getOwner() == insert.getOperation(); });
    if (!onlyUsedByInsert)
      continue;

    resultsToDrop.set(index);
    bodyArgsToDrop.set(iterArg.getArgNumber());
    insertsToErase.push_back(insert);
  }

  if (resultsToDrop.none())
    return false;

  for (auto [index, init] : llvm::enumerate(loop.getInitArgsMutable())) {
    if (resultsToDrop[index])
      operandsToDrop.set(init.getOperandNumber());
  }

  rewriter.modifyOpInPlace(yield, [&]() { yield->eraseOperands(resultsToDrop); });
  for (Operation *insert : insertsToErase)
    rewriter.eraseOp(insert);
  rewriter.modifyOpInPlace(loop, [&]() { loop.getBody()->eraseArguments(bodyArgsToDrop); });
  rewriter.modifyOpInPlace(loop, [&]() { loop->eraseOperands(operandsToDrop); });
  rewriter.eraseOpResults(loop, resultsToDrop);
  return true;
}

bool dropUnusedScratchForResults(TransformRewriter &rewriter, Operation *target) {
  bool changed = false;
  target->walk<WalkOrder::PostOrder>(
      [&](scf::ForOp loop) { changed |= dropUnusedScratchForResults(rewriter, loop); });
  return changed;
}

bool eraseTriviallyDeadOps(TransformRewriter &rewriter, Operation *target) {
  bool changed = false;
  bool changedThisRound = true;
  while (changedThisRound) {
    changedThisRound = false;
    SmallVector<Operation *> deadOps;
    target->walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op != target && isOpTriviallyDead(op))
        deadOps.push_back(op);
    });
    for (Operation *op : deadOps) {
      if (!op->getBlock())
        continue;
      rewriter.eraseOp(op);
      changed = true;
      changedThisRound = true;
    }
  }
  return changed;
}

LogicalResult runGreedyCleanup(TransformRewriter &rewriter, Operation *target) {
  RewritePatternSet patterns(target->getContext());
  patterns.add<SliceFillOfEmpty>(target->getContext());
  linalg::populateSwapExtractSliceWithFillPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  populateRegionBranchOpInterfaceCanonicalizationPatterns(patterns, scf::ForOp::getOperationName());
#if LLVM_VERSION_MAJOR >= 23
  populateRegionBranchOpInterfaceCanonicalizationPatterns(patterns,
                                                          scf::ForallOp::getOperationName());
#endif

  GreedyRewriteConfig config;
  config.setListener(static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  return applyPatternsGreedily(target, std::move(patterns), config);
}

FailureOr<uint64_t> matchUnarySingleReductionGeneric(linalg::GenericOp generic) {
  auto reductionDim = matchOneDimReductionGeneric(generic);
  // In addition, check that there is exactly one input, and that its indexing map is the identity.
  if (generic.getInputs().size() != 1)
    return failure();
  AffineMap inputMap = generic.getIndexingMapsArray()[0];
  if (!inputMap.isIdentity())
    return failure();
  return reductionDim;
}

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

struct ReductionForallSplitPlan {
  Value producerResult;
  tensor::ParallelInsertSliceOp producerInsert;
  unsigned removedIvIndex;
  uint64_t reductionDim;
  Value reductionInit;
};

FailureOr<ReductionForallSplitPlan>
detectReductionForallSplit(const TransformOpInterface &transform, scf::ForallOp loop,
                           linalg::GenericOp consumer, uint64_t reductionDim) {
  auto producerResult = dyn_cast<OpResult>(consumer.getInputs().front());
  if (!producerResult || producerResult.getOwner() != loop.getOperation()) {
    emitError(consumer.getInputs().front().getLoc())
        << "expected the reduction input to be produced by the target scf.forall";
    return failure();
  }
  auto insertSliceF = getParallelInsertSliceForLoopResult(loop, producerResult);
  if (failed(insertSliceF)) {
    loop.emitError() << "failed to find the tensor.parallel_insert_slice operation in this loop "
                     << "that published result #" << producerResult.getResultNumber();
    return failure();
  }
  tensor::ParallelInsertSliceOp producerInsert = *insertSliceF;

  OpFoldResult reductionOffset = producerInsert.getMixedOffsets()[reductionDim];
  Value reductionOffsetValue = dyn_cast<Value>(reductionOffset);
  // findLoopIvIndex handles the case where `reductionOffsetValue` is null, so we can have a single
  // point of failure reporting.
  std::optional<unsigned> removedIvIndex =
      findLoopIvIndex(reductionOffsetValue, loop.getInductionVars());
  if (!removedIvIndex) {
    producerInsert.emitError()
        << "expected the offset on dimension " << reductionDim
        << " to be a dynamic value controlled by a single scf.forall induction variable; got "
        << reductionOffsetValue;
    return failure();
  }

  return ReductionForallSplitPlan{
      .producerResult = producerResult,
      .producerInsert = producerInsert,
      .removedIvIndex = *removedIvIndex,
      .reductionDim = reductionDim,
      .reductionInit = consumer.getDpsInits().front(),
  };
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

RankedTensorType getTensorTypeFromMixedSizes(Type sourceType, ArrayRef<OpFoldResult> sizes) {
  Type elementType = getElementTypeOrSelf(sourceType);
  SmallVector<int64_t> shape;
  shape.reserve(sizes.size());
  for (OpFoldResult size : sizes) {
    auto maybeConst = getConstantIntValue(size);
    shape.push_back(maybeConst ? *maybeConst : ShapedType::kDynamic);
  }
  return RankedTensorType::get(shape, elementType);
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
    rewriter.setInsertionPointToEnd(&newLoop.getTerminator().getRegion().front());
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

// Clones the body of a loop operation using `rewriter` and `mapping`, at the insertion point of
// `rewriter`. Returns a vector of pairs of the original and cloned operations.
template <typename LoopOp>
OpPairs cloneLoopBody(RewriterBase &rewriter, LoopOp cloneFrom, IRMapping &mapping) {
  OpPairs clonedOps;
  for (Operation &op : cloneFrom.getBody()->without_terminator()) {
    Operation *newOp = rewriter.clone(op, mapping);
    clonedOps.emplace_back(&op, newOp);
  }
  return clonedOps;
}

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
  OpPairs clonedOps = cloneLoopBody(rewriter, loop, mapping);
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
  outerTileSizes[plan.reductionDim] =
      getMixedTensorSizes(rewriter, loc, plan.producerResult)[plan.reductionDim];
  Value tileInit = createExtractSliceFromState(rewriter, loc, outerProducerArg, outerTileOffsets,
                                               outerTileSizes, outerTileStrides);

  auto redTileOffsets = dropAt(outerTileOffsets, plan.reductionDim);
  auto redTileSizes = dropAt(outerTileSizes, plan.reductionDim);
  auto redTileStrides = dropAt(outerTileStrides, plan.reductionDim);
  Value redTileInit = createExtractSliceFromState(rewriter, loc, outerReductionArg, redTileOffsets,
                                                  redTileSizes, redTileStrides);

  // Create the inner loop that iterates over the removed induction variable.
  auto forLoop = scf::ForOp::create(rewriter, loc, innerLbV, innerUbV, innerStepV,
                                    ValueRange{tileInit, redTileInit});
  outerIvMapping.map(loop.getInductionVars()[plan.removedIvIndex], forLoop.getInductionVar());
  rewriter.setInsertionPointToStart(forLoop.getBody());
  OpPairs clonedOps = cloneLoopBody(rewriter, loop, outerIvMapping);

  Value outerTile = outerIvMapping.lookup(plan.producerInsert.getSource());
  rewriter.setInsertionPointToEnd(forLoop.getBody());
  auto offsets = plan.producerInsert.getMixedOffsets();
  SmallVector<OpFoldResult> localOffsets(offsets.size(), rewriter.getIndexAttr(0));
  localOffsets[plan.reductionDim] =
      *remapAffineIndex(rewriter, loc, offsets[plan.reductionDim], outerIvMapping.getValueMap());
  Value innerTile = tensor::InsertSliceOp::create(
      rewriter, loc, outerTile, forLoop.getRegionIterArgs()[0], localOffsets,
      plan.producerInsert.getMixedSizes(), outerTileStrides);

  pointRewriterToForallParallel(rewriter, newForall);
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

struct PartialReductionForallResult {
  scf::ForallOp newForall;
  linalg::GenericOp partialReduce;
  OpPairs clonedOps;
  OpPairs clonedCombiningOps;
};

FailureOr<PartialReductionForallResult>
fusePartialReductionIntoForall(TransformOpInterface transform, RewriterBase &rewriter,
                               scf::ForallOp loop, linalg::GenericOp consumer,
                               PartialReductionOpInterface consumerPR,
                               ReductionForallSplitPlan &plan) {
  Location loc = loop.getLoc();

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

  // Create the init value for the "rf" (r-factor) tensor with the size of the reduction dimension
  // set to the trip count of the loop.
  SmallVector<OpFoldResult> partialItSizes =
      getMixedTensorSizes(rewriter, loc, plan.producerResult);
  partialItSizes[plan.reductionDim] = getForallTripCount(loop, plan.removedIvIndex);
  SetVector<unsigned> reductionDims;
  reductionDims.insert(static_cast<unsigned>(plan.reductionDim));
  FailureOr<SmallVector<Value>> rfInits = consumerPR.generateInitialTensorForPartialReduction(
      rewriter, consumer.getLoc(), partialItSizes, reductionDims);
  if (failed(rfInits) || rfInits->size() != 1) {
    consumer.emitError() << "failed to create partial reduction init tensor";
    return failure();
  }

  // Create a new forall loop that has the RF init tensor as an additional output.
  SmallVector<Value> newOutputs = llvm::to_vector(loop.getOutputs());
  newOutputs.push_back(rfInits->front());
  auto newForall =
      scf::ForallOp::create(rewriter, loc, loop.getMixedLowerBound(), loop.getMixedUpperBound(),
                            loop.getMixedStep(), newOutputs, loop.getMapping());
  IRMapping mapping;
  for (auto [oldIv, newIv] : llvm::zip(loop.getInductionVars(), newForall.getInductionVars()))
    mapping.map(oldIv, newIv);
  for (auto [oldArg, newArg] : llvm::zip(loop.getRegionOutArgs(), newForall.getRegionOutArgs()))
    mapping.map(oldArg, newArg);
  rewriter.setInsertionPointToStart(newForall.getBody());
  OpPairs clonedOps = cloneLoopBody(rewriter, loop, mapping);

  // Calculate (offset, size, stride) for a tile of the RF tensor under the loop.
  auto viewTriple =
      remapAffineOffsetSizeStride(rewriter, loc, plan.producerInsert, mapping.getValueMap());
  if (failed(viewTriple)) {
    plan.producerInsert.emitError() << "failed to remap offsets of this operation";
    return failure();
  }
  auto [producerOffsets, producerSizes, _] = *viewTriple;
  auto partialOffsets = dropAt(producerOffsets, plan.reductionDim);
  // The split index must be derived from the new induction variable. Using the old loop IV here
  // would leave a dangling operand after the old forall is replaced.
  OpFoldResult loopSplitIndex = getForallSplitIndex(newForall, plan.removedIvIndex);
  partialOffsets.push_back(loopSplitIndex);
  auto rankReducedSizes = dropAt(producerSizes, plan.reductionDim);
  auto partialSizes = rankReducedSizes;
  partialSizes.push_back(rewriter.getIndexAttr(1));
  auto partialStrides = getUnitStrides(rewriter, partialOffsets.size());

  // Extract a tile of the RF tensor.
  Value inputTile = mapping.lookup(plan.producerInsert.getSource());
  Value partialOutArg = newForall.getRegionOutArgs().back();
  auto partialTileType = getTensorTypeFromMixedSizes(
      cast<RankedTensorType>(partialOutArg.getType()).getElementType(), rankReducedSizes);
  rewriter.setInsertionPoint(newForall.getTerminator());
  auto partialTileInit = tensor::ExtractSliceOp::create(
      rewriter, loc, partialTileType, partialOutArg, partialOffsets, partialSizes, partialStrides);

  // Clone the consumer into the loop while making it read from the tile of the RF tensor.
  auto partialReduce =
      cloneGenericOnTile(rewriter, consumer, inputTile, partialTileInit, consumer.getLoc());

  // Clone the combining operations (e.g., the parallel insert slice) into the new forall loop.
  SmallVector<std::optional<FoldedTensorInfo>> noFoldInfos(loop.getNumResults());
  auto clonedCombiningOps = LoopSharedTrait<scf::ForallOp>::cloneCombiningOps(
      loop, rewriter, mapping, noFoldInfos, newForall);
  if (failed(clonedCombiningOps))
    return failure();
  // Add a parallel insert slice to write the result of the partial reduction into the RF tensor.
  pointRewriterToForallParallel(rewriter, newForall);
  tensor::ParallelInsertSliceOp::create(rewriter, loc, partialReduce.getResult(0), partialOutArg,
                                        partialOffsets, partialSizes, partialStrides);

  return PartialReductionForallResult{.newForall = newForall,
                                      .partialReduce = partialReduce,
                                      .clonedOps = std::move(clonedOps),
                                      .clonedCombiningOps = std::move(*clonedCombiningOps)};
}

} // namespace

void ScfLocalizeScratchTensorsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
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

DiagnosedSilenceableFailure ScfLocalizeScratchTensorsOp::applyToOne(TransformRewriter &rewriter,
                                                                    Operation *target,
                                                                    ApplyToEachResultList &results,
                                                                    TransformState &state) {
  (void)results;
  (void)state;

  Operation *isolatedTarget = findEnclosingIsolatedFromAbove(target);
  if (!isolatedTarget)
    return emitSilenceableFailure(target, "expected target to be nested in an isolated op");

  if (failed(runGreedyCleanup(rewriter, isolatedTarget)))
    return emitSilenceableFailure(target, "initial greedy cleanup did not converge");

  localizeScratchSlices(rewriter, isolatedTarget);
  dropUnusedScratchForResults(rewriter, isolatedTarget);
  eraseTriviallyDeadOps(rewriter, isolatedTarget);

  if (failed(runGreedyCleanup(rewriter, isolatedTarget)))
    return emitSilenceableFailure(target, "final greedy cleanup did not converge");

  return DiagnosedSilenceableFailure::success();
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

  FailureOr<uint64_t> reductionDim = matchUnarySingleReductionGeneric(consumer);
  if (failed(reductionDim))
    BAIL("expected a unary single-reduction linalg.generic consumer");
  auto splitPlan = detectReductionForallSplit(transform, loop, consumer, *reductionDim);
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

  pointRewriterToForallParallel(rewriter, split.newForall);
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

  FailureOr<uint64_t> reductionDim = matchUnarySingleReductionGeneric(consumer);
  if (failed(reductionDim))
    BAIL("expected a unary single-reduction linalg.generic consumer");
  auto partialReductionOp = dyn_cast<PartialReductionOpInterface>(consumer.getOperation());
  if (!partialReductionOp)
    BAIL("expected reduction to implement PartialReductionOpInterface");
  auto splitPlan = detectReductionForallSplit(transform, loop, consumer, *reductionDim);
  if (failed(splitPlan))
    BAIL("failed to detect a split plan for the reduction");

  rewriter.setInsertionPoint(consumer);
  auto partial = fusePartialReductionIntoForall(transform, rewriter, loop, consumer,
                                                partialReductionOp, *splitPlan);
  if (failed(partial))
    BAIL("failed to split the forall loop for the reduction");
  notifyClonedOpsRecursively(rewriter, partial->clonedOps);
  notifyClonedOpsRecursively(rewriter, partial->clonedCombiningOps);

  SetVector<unsigned> reductionDims;
  reductionDims.insert(static_cast<unsigned>(*reductionDim));
  rewriter.setInsertionPointAfter(partial->newForall);
  FailureOr<MergeResult> mergeResult = partialReductionOp.mergeReductions(
      rewriter, consumer.getLoc(), ValueRange{partial->newForall.getResults().back()},
      reductionDims);
  if (failed(mergeResult))
    BAIL("failed to create write-back reduction");
  if (mergeResult->mergeOps.size() != 1 || mergeResult->replacements.size() != 1)
    BAIL("expected exactly one write-back reduction and one replacement");

  if (failed(rewriter.notifyPayloadOperationReplaced(loop, partial->newForall.getOperation())))
    BAIL("failed to preserve the scf.forall handle");
  rewriter.replaceOp(consumer, mergeResult->replacements);
  rewriter.replaceOp(loop, partial->newForall.getResults().take_front(loop.getNumResults()));

  transformResults.set(getOperation()->getResult(0), {partial->partialReduce.getOperation()});
  transformResults.set(getOperation()->getResult(1), {mergeResult->mergeOps.front()});
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
