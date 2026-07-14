#include "HTile/HTileTransformExtension.h"

#include "HTile/HTileDialect.h"
#include "HTile/HTileTransformOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

void htile::registerHTileTransformExtension(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, mlir::transform::TransformDialect *dialect) {
    ctx->loadDialect<htile::HTileDialect, mlir::bufferization::BufferizationDialect,
                     mlir::memref::MemRefDialect>();
    struct TransformDialectAccess : public mlir::transform::TransformDialect {
      using mlir::Dialect::addOperations;
    };
    static_cast<TransformDialectAccess *>(dialect)
        ->addOperations<mlir::transform::HTileLinalgToSemanticOp,
                        mlir::transform::HTileOutlineKernelsOp>();
  });
}
