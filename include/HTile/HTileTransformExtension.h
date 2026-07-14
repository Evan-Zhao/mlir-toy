#ifndef HTILE_HTILETRANSFORMEXTENSION_H
#define HTILE_HTILETRANSFORMEXTENSION_H

#include "mlir/IR/DialectRegistry.h"

namespace htile {

void registerHTileTransformExtension(mlir::DialectRegistry &registry);

} // namespace htile

#endif // HTILE_HTILETRANSFORMEXTENSION_H
