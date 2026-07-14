#include "HTile/HTileDialect.h"
#include "HTile/HTilePasses.h"
#include "HTile/HTileTransformExtension.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/Compiler.h"

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "HTileDialectPlugin", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<htile::HTileDialect>();
            htile::registerHTilePasses();
            htile::registerHTileTransformExtension(*registry);
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "HTilePassPlugin", LLVM_VERSION_STRING,
          []() { htile::registerHTilePasses(); }};
}
