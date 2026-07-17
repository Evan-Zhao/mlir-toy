#ifndef STABLEHLO_STABLEHLOTRANSFORMOPS_H
#define STABLEHLO_STABLEHLOTRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "StableHLOTransformOps.h.inc"

#endif // STABLEHLO_STABLEHLOTRANSFORMOPS_H
