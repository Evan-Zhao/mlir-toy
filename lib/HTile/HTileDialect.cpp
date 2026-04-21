#include "HTile/HTileAttrs.h"
#include "HTile/HTileDialect.h"
#include "HTile/HTileOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_DIALECT_DEFS
#include "HTileOpsDialect.cpp.inc"

namespace htile {
void HTileDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "HTileAttrs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "HTileOps.cpp.inc"
      >();
}
} // namespace htile

#include "HTileEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "HTileAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "HTileOps.cpp.inc"
