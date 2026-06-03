#ifndef TA_TAUTILS_H
#define TA_TAUTILS_H

#include "TA/TAAttrs.h"
#include "TA/TAOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace ta {

struct ScopeIndexOperand {
  llvm::SmallVector<AxisAttr, 2> axes;
  llvm::SmallVector<int64_t, 2> staticBasis;
  std::optional<int64_t> constant;

  bool isAxis() const { return axes.size() == 1 && staticBasis.empty(); }
  bool isLinearized() const { return !axes.empty() && !staticBasis.empty(); }
  bool isConstant() const { return constant.has_value(); }
};

mlir::FailureOr<llvm::SmallVector<ScopeIndexOperand>> decodeScopeIndexOperands(
    mlir::Operation *op, ScopeOp scope, mlir::ValueRange indices,
    mlir::StringRef diagnostic = "index operands must be ta.scope axes or constant indices");

mlir::FailureOr<AxesAttr> inferAxesFromScopeIndexOperands(
    mlir::Operation *op, ScopeOp scope, mlir::ValueRange indices,
    mlir::StringRef diagnostic = "index operands must be ta.scope axes or constant indices");

} // namespace ta

#endif // TA_TAUTILS_H
