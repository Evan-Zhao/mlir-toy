#include "neptune/InitAll.h"

#include "HTile/HTileDialect.h"
#include "HTile/HTilePasses.h"
#include "HTile/HTileTransformExtension.h"
#include "LoopTr/LoopTransformExtension.h"
#include "StableHLO/StableHLOTransformExtension.h"
#include "TA/TADialect.h"
#include "TA/TAPasses.h"
#include "TA/TATransformExtension.h"
#include "Tune/TuneTransformExtension.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

void neptune::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<htile::HTileDialect, ta::TADialect, mlir::stablehlo::StablehloDialect>();
}

void neptune::registerAllExtensions(mlir::DialectRegistry &registry) {
  htile::registerHTileTransformExtension(registry);
  neptune::registerStableHLOTransformExtension(registry);
  ta::registerTATransformExtension(registry);
  loop::registerLoopTransformExtension(registry);
  neptune::registerTuneTransformExtension(registry);
}

void neptune::registerAllPasses() {
  htile::registerHTilePasses();
  ta::registerTAPasses();
  mlir::stablehlo::registerStablehloLinalgTransformsPasses();
}
