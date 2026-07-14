#include "TA/TATransformExtension.h"

#include "TA/TADialect.h"
#include "TA/TATransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

void ta::registerTATransformExtension(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, mlir::transform::TransformDialect *dialect) {
    ctx->loadDialect<ta::TADialect>();
    struct TransformDialectAccess : public mlir::transform::TransformDialect {
      using mlir::Dialect::addOperations;
    };
    static_cast<TransformDialectAccess *>(dialect)
        ->addOperations<mlir::transform::StablehloGatherToLinalgConversionPatternsOp,
                        mlir::transform::TAMatchEinsumOp, mlir::transform::TAToLinalgOp,
                        mlir::transform::TASinkDivAfterMatmulPatternsOp,
                        mlir::transform::TASinkRightMulAfterMatmulPatternsOp,
                        mlir::transform::TAReassociateRightMulfPatternsOp,
                        mlir::transform::TAExpToExp2PatternsOp>();
  });
}
