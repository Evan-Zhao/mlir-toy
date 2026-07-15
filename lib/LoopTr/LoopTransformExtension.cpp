#include "LoopTr/LoopTransformExtension.h"

#include "LoopTr/LoopTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

namespace {

class LoopTransformDialectExtension
    : public mlir::transform::TransformDialectExtension<LoopTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopTransformDialectExtension)

  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "LoopTransformOps.cpp.inc"
#undef GET_OP_LIST
        >();
  }
};

} // namespace

void loop::registerLoopTransformExtension(mlir::DialectRegistry &registry) {
  registry.addExtensions<LoopTransformDialectExtension>();
}
