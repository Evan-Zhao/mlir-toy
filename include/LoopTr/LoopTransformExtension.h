#ifndef LOOPTR_LOOPTRANSFORMEXTENSION_H
#define LOOPTR_LOOPTRANSFORMEXTENSION_H

#include "mlir/IR/DialectRegistry.h"

namespace loop {

void registerLoopTransformExtension(mlir::DialectRegistry &registry);

} // namespace loop

#endif // LOOPTR_LOOPTRANSFORMEXTENSION_H
