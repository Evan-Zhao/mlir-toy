#include "TA/TATransformExtension.h"

#include "TA/TADialect.h"
#include "TA/TATransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

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
  registry.addExtensions<TATransformDialectExtension>();
}
