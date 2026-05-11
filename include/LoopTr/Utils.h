#ifndef LOOPTR_UTILS_H
#define LOOPTR_UTILS_H

#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"

namespace mlir {

#define CHECK_NON_EMPTY_OPS(state, transform, getter, nameStr, varName)                            \
  SmallVector<Operation *> varName = llvm::to_vector((state).getPayloadOps(getter()));             \
  if ((varName).empty())                                                                           \
    return emitSilenceableFailure(transform, "expected at least one " nameStr " payload op");

#define CHECK_EXTRACT_UNIQUE_OP(state, transform, getter, nameStr, varName)                        \
  SmallVector<Operation *> varName##Ops = llvm::to_vector((state).getPayloadOps(getter()));        \
  if (!llvm::hasSingleElement(varName##Ops))                                                       \
    return emitSilenceableFailure(transform, "expected exactly one " nameStr " payload op, got " + \
                                                 std::to_string(varName##Ops.size()));             \
  auto(varName) = varName##Ops.front();

#define CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getter, nameStr, varName, Type)             \
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getter, nameStr, varName##1);                          \
  auto(varName) = dyn_cast<Type>(varName##1);                                                      \
  if (!(varName))                                                                                  \
    return emitSilenceableFailure(transform, "expected " nameStr " to be a " #Type);

#define RETURN_DIAGNOSTICS_OR_BIND_VAL(OtherType, var, expr)                                       \
  OtherType(var);                                                                                  \
  {                                                                                                \
    auto var##1 = expr;                                                                            \
    if (std::holds_alternative<DiagnosedSilenceableFailure>(var##1))                               \
      return std::get<DiagnosedSilenceableFailure>(std::move(var##1));                             \
    (var) = std::get<OtherType>(var##1);                                                           \
  }
} // namespace mlir

#endif // LOOPTR_UTILS_H
