#ifndef LOOPTR_FUSION_EXPR_SOLVER_H
#define LOOPTR_FUSION_EXPR_SOLVER_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

/// A binding from a reduction result (which implicitly corresponds to an r_i)
/// to two values: r_i (current) and r_i' (next).
struct FusionRepairReductionBinding {
  OpResult reductionResult;
  Value current;
  Value next;
};

/// A switch for FusionRepairTerm::build that controls how the linalg generic is built.
/// It influences the indexing maps and expectation of how many dims each input tensor should have.
enum class FusionRepairTermMode : uint8_t {
  RollingUpdate,
  SplitKUpdate,
};

/// A solved scalar repair term h(r0, r0', r1, r1', ..., acc), plus the producer reduction order and
/// base indexing maps needed to materialize it as a linalg.generic.
class FusionRepairTerm {
public:
  FusionRepairTerm(SmallVector<OpResult> reductionOrder, SmallVector<AffineMap> argsIndexingMaps,
                   AffineMap accIndexingMap, std::unique_ptr<Block> scalarBlock, Value scalarResult)
      : reductionOrder(std::move(reductionOrder)), argsIndexingMaps(std::move(argsIndexingMaps)),
        accIndexingMap(accIndexingMap), scalarBlock(std::move(scalarBlock)),
        scalarResult(scalarResult) {}

  FailureOr<linalg::GenericOp> build(RewriterBase &rewriter, Location loc,
                                     ArrayRef<FusionRepairReductionBinding> reduceArgs,
                                     Value accArg, FusionRepairTermMode mode,
                                     unsigned reduceDim) const;

private:
  /// Results from the producer reductions. reductionOrder[i] corresponds to r_i.
  SmallVector<OpResult> reductionOrder;
  /// Base maps from the fused reduction expression. argsIndexingMaps[i] is the
  /// common base map for r_i and r_i' before mode-specific patching.
  SmallVector<AffineMap> argsIndexingMaps;
  /// Base affine map for the accumulator/output argument before mode-specific patching.
  AffineMap accIndexingMap;
  /// Detached scalar expression block. Its 2N + 1 arguments are ordered by the
  /// solver ABI: r0, r0', r1, r1', ..., acc.
  std::unique_ptr<Block> scalarBlock;
  /// Value inside scalarBlock that represents the repair term result.
  Value scalarResult;
};

/// Extracts the frontier reduction and sidecar expression, then calls the
/// Python solver to derive the repair term.
///
/// The insertion point is moved after `thisRed`, where temporary fused
/// expression ops can be safely created during extraction.
FailureOr<FusionRepairTerm> solveFusionRepairExpr(RewriterBase &rewriter,
                                                  ArrayRef<Operation *> producingReds,
                                                  linalg::GenericOp thisRed,
                                                  ArrayRef<Operation *> elemwiseSidecars);

} // namespace mlir

#endif // LOOPTR_FUSION_EXPR_SOLVER_H
