#ifndef HTILE_HTILETRANSFORMOPS_H
#define HTILE_HTILETRANSFORMOPS_H

#include "HTile/HTileOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "HTileTransformOps.h.inc"

#endif // HTILE_HTILETRANSFORMOPS_H
