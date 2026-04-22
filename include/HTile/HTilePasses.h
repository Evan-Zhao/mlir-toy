#ifndef HTILE_HTILEPASSES_H
#define HTILE_HTILEPASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace htile {

std::unique_ptr<mlir::Pass> createDotTransposeToLoadOrderPass();
void registerHTilePasses();

} // namespace htile

#endif // HTILE_HTILEPASSES_H
