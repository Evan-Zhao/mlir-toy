#ifndef TA_TAPASSES_H
#define TA_TAPASSES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace ta {

mlir::LogicalResult lowerTAToLinalg(
    mlir::Operation *target, mlir::OpBuilder &builder,
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> *loweredOps = nullptr,
    llvm::function_ref<void(mlir::Operation *,
                            const llvm::DenseMap<mlir::Operation *, mlir::Operation *> &)>
        beforeErase = nullptr);

void registerLinalgToTAPass();
void registerTAToLinalgPass();

} // namespace ta

#endif // TA_TAPASSES_H
