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

#include <variant>

using namespace mlir;

namespace mlir::transform {

namespace {

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

  auto reductionDim = getReductionIteratorIndex(generic);
  if (failed(reductionDim))
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

struct ReductionForallSplitPlan {
  Value producerResult;
  tensor::ParallelInsertSliceOp producerInsert;
  unsigned removedIvIndex;
  uint64_t reductionDim;
  Value reductionInit;
};

#define BAIL(message) return emitSilenceableFailure(transform, message);

std::variant<ReductionForallSplitPlan, DiagnosedSilenceableFailure>
detectReductionForallSplit(const TransformOpInterface &transform, scf::ForallOp loop,
                           linalg::GenericOp consumer, uint64_t reductionDim) {
  auto producerResult = dyn_cast<OpResult>(consumer.getInputs().front());
  if (!producerResult || producerResult.getOwner() != loop.getOperation())
    BAIL("expected the reduction input to be produced by the target scf.forall");
  auto insertSliceF = getParallelInsertSliceForLoopResult(loop, producerResult);
  if (failed(insertSliceF)) {
    loop.emitRemark() << "when analyzing this loop (scf.forall); #result = "
                      << producerResult.getResultNumber();
    BAIL("failed to find the tensor.parallel_insert_slice operation in the loop that published a "
         "loop result");
  }
  tensor::ParallelInsertSliceOp producerInsert = *insertSliceF;

  OpFoldResult reductionOffset = producerInsert.getMixedOffsets()[reductionDim];
  Value reductionOffsetValue = dyn_cast<Value>(reductionOffset);
  if (!reductionOffsetValue) {
    producerInsert.emitError() << "for this insert operation, on dim " << reductionDim;
    BAIL("expected the reduced producer dimension to have a dynamic tile offset");
  }

  std::optional<unsigned> removedIvIndex =
      findLoopIvIndex(reductionOffsetValue, loop.getInductionVars());
  if (!removedIvIndex) {
    producerInsert->emitRemark() << "dim " << reductionDim << " of this op has offset "
                                 << reductionOffsetValue << ", which violates our assumptions";
    BAIL("expected the reduced producer dimension to be controlled by a single scf.forall "
         "induction variable");
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

LogicalResult remapAffineIndices(RewriterBase &rewriter, Location loc,
                                 SmallVectorImpl<OpFoldResult> &values,
                                 const DenseMap<Value, Value> &mapping) {
  for (OpFoldResult &value : values) {
    auto result = remapAffineIndex(rewriter, loc, value, mapping);
    if (failed(result))
      return failure();
    value = *result;
  }
  return success();
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
  SmallVector<OpFoldResult> offsets = insert.getMixedOffsets(), sizes = insert.getMixedSizes(),
                            strides = insert.getMixedStrides();
  auto remapAffineIndicesLocal = [&](SmallVectorImpl<OpFoldResult> &values) {
    return remapAffineIndices(rewriter, loc, values, mapping.getValueMap());
  };
  if (failed(remapAffineIndicesLocal(offsets)) || failed(remapAffineIndicesLocal(sizes)) ||
      failed(remapAffineIndicesLocal(strides)))
    return failure();

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

using OpPairs = SmallVector<std::pair<Operation *, Operation *>>;

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
      if (failed(newCombiningOp))
        return failure();
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

  auto newLoop = LoopTrait::createLike(rewriter, loop, newOuts);
  newLoop->setAttrs(loop->getAttrs());

  IRMapping mapping;
  LoopTrait::mapInductionVars(mapping, loop, newLoop);
  rewriter.setInsertionPointToStart(newLoop.getBody());
  for (auto [oldArg, newArg, info] :
       llvm::zip(loop.getRegionIterArgs(), newLoop.getRegionIterArgs(), infos))
    mapping.map(oldArg, info ? expandTensor(rewriter, loc, newArg, *info) : newArg);

  OpPairs clonedOps;
  for (Operation &op : loop.getBody()->without_terminator()) {
    Operation *newOp = rewriter.clone(op, mapping);
    clonedOps.emplace_back(&op, newOp);
  }

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

std::variant<SplitForallIntoForResult, DiagnosedSilenceableFailure>
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

  auto outerTileOffsets = plan.producerInsert.getMixedOffsets();
  if (failed(remapAffineIndices(rewriter, loc, outerTileOffsets, outerIvMapping.getValueMap()))) {
    plan.producerInsert->emitRemark() << "failed to remap offsets of this operation";
    return emitSilenceableFailure(transform, "failed to remap offsets of insert operation");
  }
  SmallVector<OpFoldResult> outerTileSizes = plan.producerInsert.getMixedSizes();
  outerTileSizes[plan.reductionDim] =
      getMixedTensorSizes(rewriter, loc, plan.producerResult)[plan.reductionDim];
  SmallVector<OpFoldResult> nDUnitStrides = getUnitStrides(rewriter, outerTileOffsets.size());
  Value tileInit = createExtractSliceFromState(rewriter, loc, outerProducerArg, outerTileOffsets,
                                               outerTileSizes, nDUnitStrides);

  auto dropAt = [](SmallVector<OpFoldResult> values, uint64_t index) {
    values.erase(values.begin() + index);
    return values;
  };
  auto redTileOffsets = dropAt(outerTileOffsets, plan.reductionDim);
  auto redTileSizes = dropAt(outerTileSizes, plan.reductionDim);
  auto nMinus1DStrides = getUnitStrides(rewriter, redTileOffsets.size());
  Value redTileInit = createExtractSliceFromState(rewriter, loc, outerReductionArg, redTileOffsets,
                                                  redTileSizes, nMinus1DStrides);

  auto forLoop = scf::ForOp::create(rewriter, loc, innerLbV, innerUbV, innerStepV,
                                    ValueRange{tileInit, redTileInit});
  outerIvMapping.map(loop.getInductionVars()[plan.removedIvIndex], forLoop.getInductionVar());
  rewriter.setInsertionPointToStart(forLoop.getBody());
  OpPairs clonedOps;
  for (Operation &op : loop.getBody()->without_terminator()) {
    Operation *newOp = rewriter.clone(op, outerIvMapping);
    clonedOps.emplace_back(&op, newOp);
  }
  Value outerTile = outerIvMapping.lookup(plan.producerInsert.getSource());

  rewriter.setInsertionPointToEnd(forLoop.getBody());
  auto offsets = plan.producerInsert.getMixedOffsets();
  SmallVector<OpFoldResult> localOffsets(offsets.size(), rewriter.getIndexAttr(0));
  localOffsets[plan.reductionDim] =
      *remapAffineIndex(rewriter, loc, offsets[plan.reductionDim], outerIvMapping.getValueMap());
  Value innerTile = tensor::InsertSliceOp::create(
      rewriter, loc, outerTile, forLoop.getRegionIterArgs()[0], localOffsets,
      plan.producerInsert.getMixedSizes(), nDUnitStrides);

  pointRewriterToForallParallel(rewriter, newForall);
  tensor::ParallelInsertSliceOp::create(rewriter, loc, forLoop.getResult(0), outerProducerArg,
                                        outerTileOffsets, outerTileSizes, nDUnitStrides);
  return SplitForallIntoForResult{.newForall = newForall,
                                  .innerFor = forLoop,
                                  .outerTile = outerTile,
                                  .innerTile = innerTile,
                                  .reductionSlice = {.offsets = std::move(redTileOffsets),
                                                     .sizes = std::move(redTileSizes),
                                                     .strides = std::move(nMinus1DStrides)},
                                  .clonedOps = std::move(clonedOps)};
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
    return ::mlir::emitDefiniteFailure(target, "initial greedy cleanup did not converge");

  localizeScratchSlices(rewriter, isolatedTarget);
  dropUnusedScratchForResults(rewriter, isolatedTarget);
  eraseTriviallyDeadOps(rewriter, isolatedTarget);

  if (failed(runGreedyCleanup(rewriter, isolatedTarget)))
    return ::mlir::emitDefiniteFailure(target, "final greedy cleanup did not converge");

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
    return emitSilenceableFailure(transform,
                                  "expected a unary single-reduction linalg.generic consumer");

  RETURN_DIAGNOSTICS_OR_BIND_VAL(
      ReductionForallSplitPlan, splitPlan,
      detectReductionForallSplit(transform, loop, consumer, *reductionDim));

  rewriter.setInsertionPoint(consumer);
  RETURN_DIAGNOSTICS_OR_BIND_VAL(
      SplitForallIntoForResult, split,
      splitForallDimensionForReduction(transform, rewriter, loop, splitPlan));
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

} // namespace mlir::transform
