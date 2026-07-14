#ifndef NEPTUNE_INITALL_H
#define NEPTUNE_INITALL_H

#include "mlir/IR/DialectRegistry.h"

namespace neptune {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllExtensions(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace neptune

#endif // NEPTUNE_INITALL_H
