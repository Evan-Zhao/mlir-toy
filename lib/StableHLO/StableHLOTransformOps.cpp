#include "StableHLO/StableHLOTransformOps.h"

#include "StableHLO/StableHLOLegalizeControlFlow.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/conversions/linalg/transforms/Rewriters.h"
#include "stablehlo/conversions/linalg/transforms/TypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace mlir::transform {

void StablehloLegalizeControlFlowOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
StablehloLegalizeControlFlowOp::applyToOne(TransformRewriter &rewriter, Operation *target,
                                           ApplyToEachResultList &results, TransformState &state) {
  (void)results;
  (void)state;
  auto *listener = static_cast<RewriterBase::Listener *>(rewriter.getListener());
  if (failed(neptune::stablehlo::legalizeControlFlow(target, listener)))
    return emitSilenceableError() << "failed to legalize StableHLO control flow";
  return DiagnosedSilenceableFailure::success();
}

namespace {

void appendStablehloConversionPatternsForRoot(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns, StringRef rootName) {
  RewritePatternSet stablehloPatterns(patterns.getContext());
  stablehlo::populateStablehloToLinalgConversionPatterns(
      patterns.getContext(), typeConverter, &stablehloPatterns,
      /*enablePrimitiveOps=*/false, /*enableSparseOps=*/false,
      /*captureScalarInputs=*/true);
  for (std::unique_ptr<RewritePattern> &pattern : stablehloPatterns.getNativePatterns()) {
    std::optional<OperationName> root = pattern->getRootKind();
    if (root && root->getStringRef() == rootName)
      patterns.getNativePatterns().push_back(std::move(pattern));
  }
}

} // namespace

void StablehloSliceToTensorConversionPatternsOp::populatePatterns(TypeConverter &typeConverter,
                                                                  RewritePatternSet &patterns) {
  appendStablehloConversionPatternsForRoot(typeConverter, patterns,
                                           stablehlo::SliceOp::getOperationName());
}

std::unique_ptr<TypeConverter> StablehloSliceToTensorConversionPatternsOp::getTypeConverter() {
  return std::make_unique<stablehlo::LinalgTypeConverter>();
}

} // namespace mlir::transform

#define GET_OP_CLASSES
#include "StableHLOTransformOps.cpp.inc"
