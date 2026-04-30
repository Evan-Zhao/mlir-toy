#ifndef LINALGEXT_LINALGEXTTRANSFORM_H
#define LINALGEXT_LINALGEXTTRANSFORM_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace linalg_ext {

void registerLinalgExtTransformExtension(mlir::DialectRegistry &registry);

} // namespace linalg_ext

#endif // LINALGEXT_LINALGEXTTRANSFORM_H
