#ifndef HTILE_HTILEOPS_H
#define HTILE_HTILEOPS_H

#include "HTile/HTileDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "HTileOps.h.inc"

#endif // HTILE_HTILEOPS_H
