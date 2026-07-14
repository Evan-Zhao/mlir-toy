#include "TA/TADialect.h"
#include "TA/TAPasses.h"
#include "TA/TATransformExtension.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Compiler.h"

namespace {

void registerTAPasses() {
  ta::registerTAPasses();
  mlir::stablehlo::registerStablehloLinalgTransformsPasses();
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TADialectPlugin", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<ta::TADialect, mlir::stablehlo::StablehloDialect>();
            ta::registerTATransformExtension(*registry);
            registerTAPasses();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TAPassPlugin", LLVM_VERSION_STRING,
          []() { registerTAPasses(); }};
}
