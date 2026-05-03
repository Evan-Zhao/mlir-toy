#include "LinalgExt/LinalgExtTransformOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <deque>

using namespace mlir;

namespace {

bool isReductionLike(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  return llvm::any_of(linalgOp.getIteratorTypesArray(), [](auto iteratorType) {
    return iteratorType == utils::IteratorType::reduction;
  });
}

using OpsT = SmallVector<Operation *>;

std::optional<std::pair<OpsT, OpsT>> collectRollingFrontierPath(Operation *op) {
  OpsT elementwiseOps, frontierOps;
  SmallPtrSet<Operation *, 8> visitedOps;
  std::deque<std::pair<Operation *, bool>> worklist;
  worklist.push_back({op, false});

  while (!worklist.empty()) {
    auto [currentOp, foundReduction] = worklist.front();
    worklist.pop_front();
    if (visitedOps.contains(currentOp))
      continue;
    visitedOps.insert(currentOp);
    // If the current op is a reduction, add it to the frontier and skip its
    // users.
    if (foundReduction) {
      frontierOps.push_back(currentOp);
      continue;
    }
    // Otherwise, add it to the elementwise ops list and add its users to the
    // worklist (don't include `op` itself).
    if (currentOp != op) {
      elementwiseOps.push_back(currentOp);
    }
    bool hasUsers = false;
    for (Value result : currentOp->getOpResults()) {
      for (auto user : result.getUsers()) {
        worklist.push_back({user, isReductionLike(user)});
        hasUsers = true;
      }
    }
    if (!hasUsers) {
      currentOp->emitRemark("Path ends here without seeing a reduction");
      return std::nullopt;
    }
  }
  return {{elementwiseOps, frontierOps}};
}

} // namespace

namespace mlir {
namespace transform {

DiagnosedSilenceableFailure LinalgExtRollingUpdateFwdFrontierOp::apply(
    transform::TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  SmallVector<Operation *> producers =
      llvm::to_vector(state.getPayloadOps(getProducerOp()));
  if (producers.size() != 1)
    return emitSilenceableFailure(
        transform,
        "expected exactly one producer payload op in the transform handle");

  auto frontierResult = collectRollingFrontierPath(producers.front());
  if (!frontierResult)
    return emitSilenceableFailure(
        transform, "target has no closed rolling update frontier");
  auto [elemwiseOps, frontierOps] = *frontierResult;
  transformResults.set(getOperation()->getResult(0), elemwiseOps);
  transformResults.set(getOperation()->getResult(1), frontierOps);
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace mlir
