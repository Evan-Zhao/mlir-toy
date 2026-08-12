#ifndef STABLEHLO_STABLEHLOLEGALIZECONTROLFLOW_H
#define STABLEHLO_STABLEHLOLEGALIZECONTROLFLOW_H

#include "mlir/IR/PatternMatch.h"

namespace neptune::stablehlo {

mlir::LogicalResult legalizeControlFlow(mlir::Operation *target,
                                        mlir::RewriterBase::Listener *listener = nullptr);

} // namespace neptune::stablehlo

#endif // STABLEHLO_STABLEHLOLEGALIZECONTROLFLOW_H
