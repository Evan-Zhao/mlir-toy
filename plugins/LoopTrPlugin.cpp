#include "LoopTr/LoopTransformExtension.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "llvm/Support/Compiler.h"

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "LoopTransformPlugin", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) { loop::registerLoopTransformExtension(*registry); }};
}
