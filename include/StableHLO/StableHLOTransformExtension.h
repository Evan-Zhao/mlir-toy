#ifndef STABLEHLO_STABLEHLOTRANSFORMEXTENSION_H
#define STABLEHLO_STABLEHLOTRANSFORMEXTENSION_H

#include "mlir/IR/DialectRegistry.h"

namespace neptune {

void registerStableHLOTransformExtension(mlir::DialectRegistry &registry);

} // namespace neptune

#endif // STABLEHLO_STABLEHLOTRANSFORMEXTENSION_H
