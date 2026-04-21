#include "HTile/HTileDialect.h"
#include "HTile/HTileOps.h"
#include "mlir/IR/OpImplementation.h"

#define GET_DIALECT_DEFS
#include "HTileOpsDialect.cpp.inc"

namespace htile {
void HTileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "HTileOps.cpp.inc"
      >();
}
} // namespace htile

#define GET_OP_CLASSES
#include "HTileOps.cpp.inc"
