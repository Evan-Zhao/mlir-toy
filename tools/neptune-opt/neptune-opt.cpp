#include "neptune/InitAll.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  neptune::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  neptune::registerAllDialects(registry);
  neptune::registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Neptune optimizer driver\n", registry));
}
