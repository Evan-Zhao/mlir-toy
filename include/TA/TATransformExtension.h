#ifndef TA_TATRANSFORMEXTENSION_H
#define TA_TATRANSFORMEXTENSION_H

#include "mlir/IR/DialectRegistry.h"

namespace ta {

void registerTATransformExtension(mlir::DialectRegistry &registry);

} // namespace ta

#endif // TA_TATRANSFORMEXTENSION_H
