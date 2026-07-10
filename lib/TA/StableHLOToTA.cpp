#include "TA/TAOps.h"
#include "TA/TAPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "stablehlo-to-ta"

namespace ta {

using namespace mlir;

namespace {

struct ImportStableHLOToTAPass
    : public PassWrapper<ImportStableHLOToTAPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportStableHLOToTAPass)

  StringRef getArgument() const final { return "stablehlo-to-ta"; }
  StringRef getDescription() const final {
    return "Import supported StableHLO tensor dataflow into the ta dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TADialect, func::FuncDialect, stablehlo::StablehloDialect>();
  }

  void runOnOperation() final {}
};

} // namespace

void registerStableHLOToTAPass() { PassRegistration<ImportStableHLOToTAPass>(); }

} // namespace ta

#undef DEBUG_TYPE
