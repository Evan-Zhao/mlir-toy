#include "Tune/TuneTransformExtension.h"

#include "Tune/TuneTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

namespace {

class TuneTransformDialectExtension
    : public mlir::transform::TransformDialectExtension<TuneTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TuneTransformDialectExtension)

  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "TuneTransformOps.cpp.inc"
#undef GET_OP_LIST
        >();
  }
};

} // namespace

void neptune::registerTuneTransformExtension(mlir::DialectRegistry &registry) {
  registry.addExtensions<TuneTransformDialectExtension>();
}
