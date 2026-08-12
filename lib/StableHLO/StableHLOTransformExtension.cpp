#include "StableHLO/StableHLOTransformExtension.h"

#include "StableHLO/StableHLOTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    declareDependentDialect<mlir::scf::SCFDialect>();
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
  registry.addExtensions<StableHLOTransformDialectExtension>();
}
