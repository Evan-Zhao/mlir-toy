#include "LoopTr/LoopTransform.h"
#include "LoopTr/LoopTransformOps.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

namespace loop {

void registerLoopTransformExtension(mlir::DialectRegistry &registry) {
  // MLIR 22.1.4 hangs in TransformDialectExtension::registerTransformOps for
  // this out-of-tree plugin. Register the op directly as a narrow workaround;
  // the ops are still constrained by their TableGen traits and verified when
  // used.
  registry.addExtension(+[](mlir::MLIRContext *, mlir::transform::TransformDialect *dialect) {
    struct TransformDialectAccess : public mlir::transform::TransformDialect {
      using mlir::Dialect::addOperations;
    };
    static_cast<TransformDialectAccess *>(dialect)
        ->addOperations<
            mlir::transform::ScfLocalizeScratchTensorsOp,
            mlir::transform::ScfFoldUnitExtentDimsViaReshapesPatternsOp,
            mlir::transform::LinalgEraseUnusedOperandsAndResultsOp,
            mlir::transform::LinalgGreedyInlineElementwiseOp, mlir::transform::FusionIntoProducerOp,
            mlir::transform::FusionGreedyConsumersIntoProducerOp,
            mlir::transform::ScfFuseReductionIntoForallOp,
            mlir::transform::FusionFindNextReductionOp, mlir::transform::FusionCloneFuseElemwiseOp,
            mlir::transform::FusionRepairReductionFrontierOp,
            mlir::transform::LoopSpecializeDeadTileOp>();
  });
}

} // namespace loop

#define GET_OP_CLASSES
#include "LoopTransformOps.cpp.inc"

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "LoopTransformPlugin", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) { loop::registerLoopTransformExtension(*registry); }};
}
