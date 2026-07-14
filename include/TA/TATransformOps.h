#ifndef TA_TATRANSFORMOPS_H
#define TA_TATRANSFORMOPS_H

#include "TA/TAOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TATransformOps.h.inc"

#endif // TA_TATRANSFORMOPS_H
