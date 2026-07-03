#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

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

} // namespace

void ScfLocalizeScratchTensorsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
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

} // namespace mlir::transform
