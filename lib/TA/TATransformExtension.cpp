#include "TA/TATransformExtension.h"

#include "TA/TADialect.h"
#include "TA/TATransformOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "stablehlo/dialect/StablehloOps.h"

void ta::registerTATransformExtension(mlir::DialectRegistry &registry) {
  // transform.apply_registered_pass constructs its nested pass manager after
  // execution has started, too late for that manager to load newly discovered
  // dependent dialects safely. Preload the StableHLO-to-Linalg pass
  // dependencies when StableHLO is loaded.
  registry.addExtension(+[](mlir::MLIRContext *context, mlir::stablehlo::StablehloDialect *) {
    context->loadDialect<mlir::bufferization::BufferizationDialect, mlir::complex::ComplexDialect,
                         mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                         mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                         mlir::shape::ShapeDialect, mlir::sparse_tensor::SparseTensorDialect>();
  });

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
