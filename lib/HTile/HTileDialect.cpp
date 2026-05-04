#include "HTile/HTileDialect.h"
#include "HTile/HTileAttrs.h"
#include "HTile/HTileOps.h"
#include "HTile/HTilePasses.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
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

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "HTileDialectPlugin", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<htile::HTileDialect>();
            htile::registerHTilePasses();
          }};
}

#include "HTileEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "HTileAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "HTileOps.cpp.inc"
