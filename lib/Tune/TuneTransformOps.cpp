#include "Tune/TuneTransformOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace mlir::transform {

void TuneSampleCategoricalOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  producesHandle(getOperation()->getOpResults(), effects);
}

DiagnosedSilenceableFailure
TuneSampleCategoricalOp::apply(TransformRewriter &rewriter, TransformResults &results,
                               TransformState &state) {
  (void)rewriter;
  (void)state;
  results.setParams(cast<OpResult>(getResult()), {getDefaultValueAttr()});
  return DiagnosedSilenceableFailure::success();
}

LogicalResult TuneSampleCategoricalOp::verify() {
  auto paramType = dyn_cast<ParamType>(getResult().getType());
  if (!paramType || !paramType.getType().isInteger(64))
    return emitOpError("requires an !transform.param<i64> result");

  ArrayRef<int64_t> candidates = getCandidates();
  if (candidates.empty())
    return emitOpError("requires at least one candidate");

  llvm::DenseSet<int64_t> seen;
  for (int64_t candidate : candidates) {
    if (!seen.insert(candidate).second)
      return emitOpError() << "has duplicate candidate " << candidate;
  }

  int64_t defaultValue = getDefaultValue();
  if (!seen.contains(defaultValue))
    return emitOpError() << "has default value " << defaultValue
                         << " that is not in the candidate list";
  return success();
}

void TuneChooseSequenceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getInputsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure TuneChooseSequenceOp::apply(TransformRewriter &rewriter,
                                                         TransformResults &results,
                                                         TransformState &state) {
  (void)rewriter;
  (void)results;
  (void)state;
  return emitDefiniteFailure()
         << "requires candidate materialization before transform interpretation";
}

LogicalResult TuneChooseSequenceOp::verify() {
  DictionaryAttr cases = getCases();
  if (cases.empty())
    return emitOpError("requires at least one case");

  for (NamedAttribute namedCase : cases) {
    if (!isa<SymbolRefAttr>(namedCase.getValue()))
      return emitOpError() << "requires case '" << namedCase.getName()
                           << "' to reference a named sequence";
  }

  if (!cases.get(getDefaultCase()))
    return emitOpError() << "has default case '" << getDefaultCase()
                         << "' that is not present in cases";
  return success();
}

LogicalResult TuneChooseSequenceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  for (NamedAttribute namedCase : getCases()) {
    auto calleeAttr = cast<SymbolRefAttr>(namedCase.getValue());
    auto sequence = symbolTable.lookupNearestSymbolFrom<NamedSequenceOp>(*this, calleeAttr);
    if (!sequence)
      return emitOpError() << "case '" << namedCase.getName() << "' references unknown sequence "
                           << calleeAttr;

    FunctionType functionType = sequence.getFunctionType();
    if (!llvm::equal(functionType.getInputs(), getOperandTypes()) ||
        !llvm::equal(functionType.getResults(), getResultTypes()))
      return emitOpError() << "case '" << namedCase.getName()
                           << "' sequence signature does not match "
                           << getOperation()->getName();
  }
  return success();
}

} // namespace mlir::transform

#define GET_OP_CLASSES
#include "TuneTransformOps.cpp.inc"
