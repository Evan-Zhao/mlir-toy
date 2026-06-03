#include "TA/TAUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSet.h"

namespace ta {

using namespace mlir;

FailureOr<SmallVector<ScopeIndexOperand>>
decodeScopeIndexOperands(Operation *op, ScopeOp scope, ValueRange indices, StringRef diagnostic) {
  SmallVector<ScopeIndexOperand> decoded;
  ArrayAttr scopeAxes = scope.getAxes().getAxes();
  Block &scopeBody = scope.getBody().front();

  for (Value index : indices) {
    if (auto arg = dyn_cast<BlockArgument>(index)) {
      unsigned argNumber = arg.getArgNumber();
      if (arg.getOwner() != &scopeBody || argNumber >= scopeAxes.size())
        return op->emitOpError(diagnostic);
      decoded.push_back({cast<AxisAttr>(scopeAxes[argNumber]), std::nullopt});
      continue;
    }

    if (auto constant = index.getDefiningOp<arith::ConstantIndexOp>()) {
      decoded.push_back({AxisAttr(), constant.value()});
      continue;
    }

    return op->emitOpError(diagnostic);
  }

  return decoded;
}

FailureOr<AxesAttr>
inferAxesFromScopeIndexOperands(Operation *op, ScopeOp scope, ValueRange indices,
                                StringRef diagnostic) {
  FailureOr<SmallVector<ScopeIndexOperand>> decoded =
      decodeScopeIndexOperands(op, scope, indices, diagnostic);
  if (failed(decoded))
    return failure();

  llvm::StringSet<> seen;
  SmallVector<Attribute> inferred;
  for (const ScopeIndexOperand &index : *decoded) {
    if (!index.isAxis())
      continue;
    if (seen.insert(index.axis.getName().getValue()).second)
      inferred.push_back(index.axis);
  }

  return AxesAttr::get(op->getContext(), ArrayAttr::get(op->getContext(), inferred));
}

} // namespace ta
