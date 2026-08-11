#ifndef TUNE_TUNETRANSFORMEXTENSION_H
#define TUNE_TUNETRANSFORMEXTENSION_H

#include "mlir/IR/DialectRegistry.h"

namespace neptune {

void registerTuneTransformExtension(mlir::DialectRegistry &registry);

} // namespace neptune

#endif // TUNE_TUNETRANSFORMEXTENSION_H
