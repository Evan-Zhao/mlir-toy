#ifndef TUNE_TUNETRANSFORMOPS_H
#define TUNE_TUNETRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "TuneTransformOps.h.inc"

#endif // TUNE_TUNETRANSFORMOPS_H
