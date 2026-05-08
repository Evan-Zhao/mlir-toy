#ifndef LOOPTR_UTILS_H
#define LOOPTR_UTILS_H

#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#define CHECK_EXTRACT_UNIQUE_OP(state, transform, getter, name_str, var_name)                      \
  SmallVector<Operation *> var_name##Ops = llvm::to_vector(state.getPayloadOps(getter()));         \
  if (!llvm::hasSingleElement(var_name##Ops))                                                      \
    return emitSilenceableFailure(transform, "expected exactly one " name_str                      \
                                             " payload op, got " +                                 \
                                                 std::to_string(var_name##Ops.size()));            \
  auto(var_name) = var_name##Ops.front();

#define CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getter, name_str, var_name, Type)           \
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getter, name_str, var_name##1);                        \
  auto(var_name) = dyn_cast<Type>(var_name##1);                                                    \
  if (!(var_name))                                                                                 \
    return emitSilenceableFailure(transform, "expected " name_str " to be a " #Type);

#endif // LOOPTR_UTILS_H
