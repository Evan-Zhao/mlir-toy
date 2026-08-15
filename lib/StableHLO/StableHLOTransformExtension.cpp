#include "StableHLO/StableHLOTransformExtension.h"

#include "StableHLO/StableHLOTransformOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace {

class StableHLOTransformDialectExtension
    : public mlir::transform::TransformDialectExtension<StableHLOTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StableHLOTransformDialectExtension)

  using Base::Base;

  void init() {
    // Preload the StableHLO-to-Linalg pass dependencies. A pass invoked by
    // transform.apply_registered_pass cannot load dialects while the transform
    // interpreter's parent pass manager is running in multithreaded mode.
    declareDependentDialect<mlir::arith::ArithDialect>();
    declareDependentDialect<mlir::bufferization::BufferizationDialect>();
    declareDependentDialect<mlir::complex::ComplexDialect>();
    declareDependentDialect<mlir::linalg::LinalgDialect>();
    declareDependentDialect<mlir::math::MathDialect>();
    declareDependentDialect<mlir::memref::MemRefDialect>();
    declareDependentDialect<mlir::scf::SCFDialect>();
    declareDependentDialect<mlir::shape::ShapeDialect>();
    declareDependentDialect<mlir::sparse_tensor::SparseTensorDialect>();
    declareDependentDialect<mlir::stablehlo::StablehloDialect>();
    declareDependentDialect<mlir::tensor::TensorDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "StableHLOTransformOps.cpp.inc"
#undef GET_OP_LIST
        >();
  }
};

} // namespace

void neptune::registerStableHLOTransformExtension(mlir::DialectRegistry &registry) {
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  registry.addExtensions<StableHLOTransformDialectExtension>();
}
