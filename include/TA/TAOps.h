#ifndef TA_TAOPS_H
#define TA_TAOPS_H

#include "TA/TADialect.h"
#include "TA/TATypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "TAOps.h.inc"

#endif // TA_TAOPS_H
