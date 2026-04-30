#ifndef LINALGEXT_LINALGEXTTRANSFORMOPS_H
#define LINALGEXT_LINALGEXTTRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "LinalgExtTransformOps.h.inc"

#endif // LINALGEXT_LINALGEXTTRANSFORMOPS_H
