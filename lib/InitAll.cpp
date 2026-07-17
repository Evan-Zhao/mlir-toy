#include "neptune/InitAll.h"

#include "HTile/HTileDialect.h"
#include "HTile/HTilePasses.h"
#include "HTile/HTileTransformExtension.h"
#include "LoopTr/LoopTransformExtension.h"
#include "StableHLO/StableHLOTilingInterfaceImpl.h"
#include "StableHLO/StableHLOTransformExtension.h"
#include "TA/TADialect.h"
#include "TA/TAPasses.h"
#include "TA/TATransformExtension.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

void neptune::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<htile::HTileDialect, ta::TADialect, mlir::stablehlo::StablehloDialect>();
}

void neptune::registerAllExtensions(mlir::DialectRegistry &registry) {
  htile::registerHTileTransformExtension(registry);
  neptune::registerStableHLOTransformExtension(registry);
  neptune::registerStableHLOTilingInterfaceExternalModels(registry);
  ta::registerTATransformExtension(registry);
  loop::registerLoopTransformExtension(registry);
}

void neptune::registerAllPasses() {
  htile::registerHTilePasses();
  ta::registerTAPasses();
  mlir::stablehlo::registerStablehloLinalgTransformsPasses();
}
