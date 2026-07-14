#include "LoopTr/LoopTransformExtension.h"

#include "LoopTr/LoopTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

void loop::registerLoopTransformExtension(mlir::DialectRegistry &registry) {
  // MLIR 22.1.4 hangs in TransformDialectExtension::registerTransformOps for
  // this out-of-tree plugin. Register the op directly as a narrow workaround;
  // the ops are still constrained by their TableGen traits and verified when
  // used.
  registry.addExtension(+[](mlir::MLIRContext *, mlir::transform::TransformDialect *dialect) {
    struct TransformDialectAccess : public mlir::transform::TransformDialect {
      using mlir::Dialect::addOperations;
    };
    static_cast<TransformDialectAccess *>(dialect)
        ->addOperations<
#define GET_OP_LIST
#include "LoopTransformOps.cpp.inc"
#undef GET_OP_LIST
            >();
  });
}
