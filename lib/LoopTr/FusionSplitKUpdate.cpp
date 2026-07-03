#include "LoopTr/LoopTransformOps.h"
#include "LoopTr/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#define BAIL(message) return emitSilenceableFailure(transform, message);

using namespace mlir;
using scf::ForallOp;
using transform::TransformOpInterface;

static bool isReductionLike(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  return llvm::any_of(linalgOp.getIteratorTypesArray(), [](auto iteratorType) {
    return iteratorType == utils::IteratorType::reduction;
  });
}

namespace mlir::transform {

void FusionCloneFuseRfactorElemwiseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getElemwiseChainOpsMutable(), effects);
  onlyReadsHandle(getForallLoopMutable(), effects);
  onlyReadsHandle(getWritebackReduceOpsMutable(), effects);
  onlyReadsHandle(getRfactorReduceOpsMutable(), effects);

  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
FusionCloneFuseRfactorElemwiseOp::apply(transform::TransformRewriter &rewriter,
                                        TransformResults &transformResults, TransformState &state) {
  (void)rewriter;
  (void)transformResults;
  auto transform = cast<TransformOpInterface>(getOperation());

  ForallOp forallLoop;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForallLoop, "forall loop", forallLoop,
                               ForallOp);

  CHECK_NON_EMPTY_OPS(state, transform, getElemwiseChainOps, "elementwise", elemwiseOps)
  for (Operation *elemwiseOp : elemwiseOps) {
    if (failed(isSingleOutputElemwiseLinalgOp(elemwiseOp)))
      BAIL("expected every elementwise op to be a single-result elementwise linalg.generic");
  }

  CHECK_NON_EMPTY_OPS(state, transform, getWritebackReduceOps, "write-back reduction",
                      writebackOps);
  CHECK_NON_EMPTY_OPS(state, transform, getRfactorReduceOps, "r-factor reduction", rfactorOps);
  if (writebackOps.size() != rfactorOps.size())
    BAIL("expected the same number of write-back and rfactor reduction ops");
  for (auto *op : llvm::concat<Operation *>(writebackOps, rfactorOps)) {
    if (!isReductionLike(op))
      BAIL("expected write-back and rfactor ops to be reductions");
  }

  BAIL("transform.fusion.clone_fuse_rfactor_elemwise is not implemented yet");
}

} // namespace mlir::transform
