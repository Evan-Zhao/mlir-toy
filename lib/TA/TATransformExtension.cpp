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

namespace {

class TATransformDialectExtension
    : public mlir::transform::TransformDialectExtension<TATransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TATransformDialectExtension)

  using Base::Base;

  void init() {
    declareDependentDialect<ta::TADialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "TATransformOps.cpp.inc"
#undef GET_OP_LIST
        >();
  }
};

} // namespace

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

  registry.addExtensions<TATransformDialectExtension>();
}
