#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

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

bool localizeScratchSlicesInFor(transform::TransformRewriter &rewriter, scf::ForOp loop) {
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

bool localizeScratchSlicesInForall(transform::TransformRewriter &rewriter, scf::ForallOp loop) {
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

bool localizeScratchSlices(transform::TransformRewriter &rewriter, Operation *target) {
  bool changed = false;
  target->walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
    changed |= localizeScratchSlicesInFor(rewriter, loop);
  });
  target->walk<WalkOrder::PostOrder>([&](scf::ForallOp loop) {
    changed |= localizeScratchSlicesInForall(rewriter, loop);
  });
  return changed;
}

LogicalResult runGreedyCleanup(transform::TransformRewriter &rewriter, Operation *target) {
  RewritePatternSet patterns(target->getContext());
  linalg::populateSwapExtractSliceWithFillPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);

  GreedyRewriteConfig config;
  config.setListener(static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  return applyPatternsGreedily(target, std::move(patterns), config);
}

LogicalResult runPassCleanup(Operation *isolatedTarget) {
  PassManager pm(isolatedTarget->getContext(), isolatedTarget->getName().getStringRef());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(createCanonicalizerPass());
  return pm.run(isolatedTarget);
}

} // namespace

void LoopLocalizeScratchTensorsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure LoopLocalizeScratchTensorsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results, transform::TransformState &state) {
  (void)results;
  (void)state;

  Operation *isolatedTarget = findEnclosingIsolatedFromAbove(target);
  if (!isolatedTarget)
    return emitSilenceableFailure(target, "expected target to be nested in an isolated op");

  if (failed(runGreedyCleanup(rewriter, isolatedTarget)))
    return ::mlir::emitDefiniteFailure(target, "initial greedy cleanup did not converge");

  localizeScratchSlices(rewriter, isolatedTarget);

  if (failed(runPassCleanup(isolatedTarget)))
    return ::mlir::emitDefiniteFailure(target, "remove-dead-values cleanup failed");

  if (failed(runGreedyCleanup(rewriter, isolatedTarget)))
    return ::mlir::emitDefiniteFailure(target, "final greedy cleanup did not converge");

  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
