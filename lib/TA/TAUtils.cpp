#include "TA/TAUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSet.h"

namespace ta {

using namespace mlir;

static FailureOr<AxisAttr> decodeScopeAxis(Operation *op, ScopeOp scope, Value value,
                                           StringRef diagnostic) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return op->emitOpError(diagnostic);

  ArrayAttr scopeAxes = scope.getAxes().getAxes();
  Block &scopeBody = scope.getBody().front();
  unsigned argNumber = arg.getArgNumber();
  if (arg.getOwner() != &scopeBody || argNumber >= scopeAxes.size())
    return op->emitOpError(diagnostic);

  return cast<AxisAttr>(scopeAxes[argNumber]);
}

FailureOr<SmallVector<ScopeIndexOperand>>
decodeScopeIndexOperands(Operation *op, ScopeOp scope, ValueRange indices, StringRef diagnostic) {
  SmallVector<ScopeIndexOperand> decoded;

  for (Value index : indices) {
    if (auto arg = dyn_cast<BlockArgument>(index)) {
      FailureOr<AxisAttr> axis = decodeScopeAxis(op, scope, arg, diagnostic);
      if (failed(axis))
        return failure();
      decoded.push_back({SmallVector<AxisAttr, 2>{*axis}, {}, std::nullopt});
      continue;
    }

    if (auto constant = index.getDefiningOp<arith::ConstantIndexOp>()) {
      decoded.push_back({{}, {}, constant.value()});
      continue;
    }

    if (auto linearize = index.getDefiningOp<affine::AffineLinearizeIndexOp>()) {
      if (!linearize.getDynamicBasis().empty())
        return op->emitOpError(diagnostic);

      SmallVector<int64_t> basis(linearize.getStaticBasis());
      if (basis.size() == linearize.getMultiIndex().size())
        basis.erase(basis.begin());
      if (basis.size() + 1 != linearize.getMultiIndex().size())
        return op->emitOpError(diagnostic);

      SmallVector<AxisAttr, 2> axes;
      for (Value value : linearize.getMultiIndex()) {
        FailureOr<AxisAttr> axis = decodeScopeAxis(op, scope, value, diagnostic);
        if (failed(axis))
          return failure();
        axes.push_back(*axis);
      }
      decoded.push_back({std::move(axes), std::move(basis), std::nullopt});
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
    if (index.isConstant())
      continue;
    for (AxisAttr axis : index.axes) {
      if (seen.insert(axis.getName().getValue()).second)
        inferred.push_back(axis);
    }
  }

  return AxesAttr::get(op->getContext(), ArrayAttr::get(op->getContext(), inferred));
}

} // namespace ta
