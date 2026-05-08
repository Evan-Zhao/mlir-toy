#ifndef LOOPTR_LOOPTRANSFORMOPS_H
#define LOOPTR_LOOPTRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "LoopTransformOps.h.inc"

#endif // LOOPTR_LOOPTRANSFORMOPS_H
