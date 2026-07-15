#include "neptune/InitAll.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  neptune::registerAllDialects(registry);
  neptune::registerAllExtensions(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
