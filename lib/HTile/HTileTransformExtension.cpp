#include "HTile/HTileTransformExtension.h"

#include "HTile/HTileDialect.h"
#include "HTile/HTileTransformOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

namespace {

class HTileTransformDialectExtension
    : public mlir::transform::TransformDialectExtension<HTileTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HTileTransformDialectExtension)

  using Base::Base;

  void init() {
    declareDependentDialect<htile::HTileDialect>();
    declareGeneratedDialect<mlir::bufferization::BufferizationDialect>();
    declareGeneratedDialect<mlir::memref::MemRefDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "HTileTransformOps.cpp.inc"
#undef GET_OP_LIST
        >();
  }
};

} // namespace

void htile::registerHTileTransformExtension(mlir::DialectRegistry &registry) {
  registry.addExtensions<HTileTransformDialectExtension>();
}
