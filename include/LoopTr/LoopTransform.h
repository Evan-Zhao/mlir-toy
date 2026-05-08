#ifndef LOOPTR_LOOPTRANSFORM_H
#define LOOPTR_LOOPTRANSFORM_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace loop {

void registerLoopTransformExtension(mlir::DialectRegistry &registry);

} // namespace loop

#endif // LOOPTR_LOOPTRANSFORM_H
