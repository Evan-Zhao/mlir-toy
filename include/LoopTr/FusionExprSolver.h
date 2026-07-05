#ifndef LOOPTR_FUSION_EXPR_SOLVER_H
#define LOOPTR_FUSION_EXPR_SOLVER_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace mlir {

struct LinalgProvenance {
  OpResult tileValue;
  AffineMap indexMap;
  std::string varName;
};

/// Solver input extracted from a reduction frontier and its fused sidecar
/// elementwise producers.
struct FusionRepairSolverInput {
  SmallVector<LinalgProvenance> varProvenances;
  std::string accVarName;
  llvm::json::Value fExpr, gExpr;
};

/// Result of solving the repair expression. The original solver input is kept
/// because materialization still needs variable provenance.
struct FusionRepairSolverResult {
  FusionRepairSolverInput input;
  llvm::json::Value hExpr;
};

/// Extracts the frontier reduction and sidecar expression, then calls the
/// Python solver to derive the repair term.
///
/// The insertion point is moved after `thisRed`, where temporary fused
/// expression ops can be safely created during extraction.
llvm::Expected<FusionRepairSolverResult>
solveFusionRepairExpr(RewriterBase &rewriter, ArrayRef<Operation *> producingReds,
                      linalg::GenericOp thisRed, ArrayRef<Operation *> elemwiseSidecars);

/// Make an elementwise linalg.generic op that applies the repair term found by
/// the solver. This is the rolling-update binding policy: `acc` is bound to
/// the frontier DPS init, `r*` to producer reduction inits, and `r*'` to
/// producer reduction results.
FailureOr<linalg::GenericOp> buildLinalgFromRepairTerm(RewriterBase &rewriter,
                                                       const llvm::json::Value &hExpr,
                                                       linalg::GenericOp sourceReduce,
                                                       size_t reduceDim,
                                                       const FusionRepairSolverInput &solverInput);

} // namespace mlir

#endif // LOOPTR_FUSION_EXPR_SOLVER_H
