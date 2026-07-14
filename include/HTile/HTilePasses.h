#ifndef HTILE_HTILEPASSES_H
#define HTILE_HTILEPASSES_H

#include "mlir/Pass/Pass.h"

namespace htile {

#define GEN_PASS_DECL
#include "HTilePasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "HTilePasses.h.inc"

} // namespace htile

#endif // HTILE_HTILEPASSES_H
