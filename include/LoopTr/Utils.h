#ifndef LOOPTR_UTILS_H
#define LOOPTR_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include <optional>

namespace mlir {

namespace transform {
class TransformRewriter;
} // namespace transform

#define CHECK_NON_EMPTY_OPS(state, transform, getter, nameStr, varName)                            \
  SmallVector<Operation *> varName = llvm::to_vector((state).getPayloadOps(getter()));             \
  if ((varName).empty())                                                                           \
    return emitSilenceableFailure(transform, "expected at least one " nameStr " payload op");

#define CHECK_EXTRACT_UNIQUE_OP(state, transform, getter, nameStr, varName)                        \
  Operation *varName;                                                                              \
  {                                                                                                \
    SmallVector<Operation *> varName##Ops = llvm::to_vector((state).getPayloadOps(getter()));      \
    if (!llvm::hasSingleElement(varName##Ops))                                                     \
      return emitSilenceableFailure(transform, "expected exactly one " nameStr                     \
                                               " payload op, got " +                               \
                                                   std::to_string(varName##Ops.size()));           \
    (varName) = varName##Ops.front();                                                              \
  }

#define CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getter, nameStr, varName, Type)             \
  CHECK_EXTRACT_UNIQUE_OP(state, transform, getter, nameStr, varName##1);                          \
  (varName) = dyn_cast<Type>(varName##1);                                                          \
  if (!(varName))                                                                                  \
    return emitSilenceableFailure(transform, "expected " nameStr " to be a " #Type);

/// Verifies that `op` is an elementwise linalg.generic operation with a single output.
LogicalResult isSingleOutputElemwiseLinalgOp(Operation *op);

/// Check that `generic` has a single reduction iterator, and return its index.
FailureOr<uint64_t> getReductionIteratorIndex(linalg::GenericOp generic);

/// Match a single-result linalg.generic reduction with exactly one reduction iterator.
///
/// The output indexing map must be the iteration space with the reduction dimension dropped.
/// This intentionally does not constrain the number of input operands or their indexing maps.
FailureOr<uint64_t> matchOneDimReductionGeneric(linalg::GenericOp generic);

/// Returns the unique tensor.parallel_insert_slice in `loop` that publishes `result`.
FailureOr<tensor::ParallelInsertSliceOp> getParallelInsertSliceForLoopResult(scf::ForallOp loop,
                                                                             OpResult result);

struct LoopResultRelay {
  OpResult inLoopResult;
  OpResult loopReturnResult;
  /// The operation that writes the loop-local result to the loop return tensor.
  /// This is either a tensor.parallel_insert_slice, for an scf.forall loop,
  /// or a (tensor.insert_slice or nullptr) for an scf.for loop.
  Operation *mediator;
};

using LoopResultRelaysT = SmallVector<LoopResultRelay>;

/// Builds relay chains for a loop nest ordered outer-to-inner, where
/// `loops[i + 1]->getParentOp() == loops[i]`.
///
/// The returned map is keyed by results of the outermost loop. Each value contains the full
/// relay chain for that loop result, ordered inner-to-outer, and each relay records the in-loop
/// OpResult, the loop result it feeds, and the mediator op connecting them. Only loop results
/// whose relays continue all the way to the innermost loop are included.
FailureOr<DenseMap<OpResult, LoopResultRelaysT>>
getChainedLoopResultMap(ArrayRef<Operation *> loops);

struct BinaryReductionCombinerMatch {
  BlockArgument accumulatorArg;
  Value yieldedValue;
  Value nonAccumulator;
  Operation *combiner;
};

/// Match a single-result scalar combiner for one result of a linalg.generic reduction.
///
/// Keep this inline because it is shared by multiple mlir-opt plugins. An out-of-line definition in
/// one plugin leaves other plugins with a dynamically-looked-up symbol, which crashes if the
/// defining plugin was not loaded.
inline FailureOr<BinaryReductionCombinerMatch>
matchBinaryReductionCombiner(linalg::GenericOp generic, unsigned resultNumber,
                             bool emitDiagnostics = false) {
  auto fail = [&](const Twine &message) -> FailureOr<BinaryReductionCombinerMatch> {
    if (emitDiagnostics)
      generic.emitError() << message;
    return failure();
  };

  if (resultNumber >= generic.getNumDpsInits())
    return fail("reduction result number is out of bounds for linalg.generic outputs");

  auto yield = cast<linalg::YieldOp>(generic.getBody()->getTerminator());
  if (resultNumber >= yield.getNumOperands())
    return fail("reduction result number is out of bounds for linalg.yield operands");

  Value yieldedValue = yield.getOperand(resultNumber);
  Operation *combiner = yieldedValue.getDefiningOp();
  if (!combiner || combiner->getNumOperands() != 2 || combiner->getNumResults() != 1) {
    if (emitDiagnostics) {
      generic.emitError() << "expected the reduction combiner to have 2 operands and 1 result";
      if (combiner)
        combiner->emitRemark() << "this is the reduction combiner";
    }
    return failure();
  }

  BlockArgument accumulatorArg =
      generic.getBody()->getArgument(generic.getNumDpsInputs() + resultNumber);
  Value lhs = combiner->getOperand(0), rhs = combiner->getOperand(1);
  bool lhsIsAcc = lhs == accumulatorArg, rhsIsAcc = rhs == accumulatorArg;
  if (lhsIsAcc == rhsIsAcc)
    return fail("expected exactly one reduction combiner operand to be the accumulator");

  return BinaryReductionCombinerMatch{
      .accumulatorArg = accumulatorArg,
      .yieldedValue = yieldedValue,
      .nonAccumulator = lhsIsAcc ? rhs : lhs,
      .combiner = combiner,
  };
}

SmallVector<OpFoldResult> getUnitStrides(RewriterBase &rewriter, size_t rank);

/// Returns the sizes of each dimension of `tensor` as a vector of `OpFoldResult`.
/// For dynamic dimensions, creates a `tensor.dim` op to query the size at runtime;
/// for static dimensions, returns the constant integer attribute directly.
SmallVector<OpFoldResult> getMixedTensorSizes(RewriterBase &rewriter, Location loc, Value tensor);

/// Clone the operations in `block` into the current insertion point of `builder`, except for the
/// terminator. Returns a vector of pairs of the original and cloned operations.
SmallVector<std::pair<Operation *, Operation *>>
cloneBlockWithoutTerminator(OpBuilder &builder, Block &block, IRMapping &mapping);

FailureOr<Value> cloneValueDefChainAtInsertionPoint(RewriterBase &rewriter, Value value,
                                                    IRMapping &mapping);

/// A wrapper around `cloneValueDefChainAtInsertionPoint` that applies to all operands of
/// `toMoveOperands`.
LogicalResult recursiveMoveOperandsBeforeOp(Operation &toMoveOperands, RewriterBase &rewriter,
                                            Operation &moveBefore);

/// Run `scf::tileAndFuseConsumerOfSlice` and print internal detailed error when it fails.
FailureOr<scf::SCFFuseConsumerOfSliceResult>
tileAndFuseConsumerWithDebug(RewriterBase &rewriter, Operation &consumer,
                             MutableArrayRef<LoopLikeOpInterface> loops);

/// Listener that follows one operation across rewrite notifications.
///
/// The listener forwards all events to `previous` and updates the tracked
/// operation when it sees the current operation replaced. If the current
/// operation is erased or no unique same-kind replacement can be inferred,
/// `getOperation()` returns nullptr.
struct TrackedOperationListener : public RewriterBase::ForwardingListener {
  TrackedOperationListener(Operation *trackedOp, OpBuilder::Listener *previous);

  Operation *getOperation() const { return trackedOp; }

  void notifyOperationReplaced(Operation *op, Operation *newOp) override;
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;
  void notifyOperationErased(Operation *op) override;

private:
  Operation *trackedOp;
};

struct ElementwiseInlineResult {
  Operation *fusedOp;
  bool applied;
};

/// Greedily inline elementwise producers into a linalg.generic op.
///
/// If `operandNumber` is set, only that DPS input operand is considered.
/// Returns the final rewritten op and whether any inlining happened.
FailureOr<ElementwiseInlineResult>
greedyInlineElementwiseProducers(transform::TransformRewriter &rewriter, linalg::GenericOp target,
                                 std::optional<int64_t> operandNumber = std::nullopt);

/// Tiles and fuses `operation` into a double loop structure, in two steps.
/// Returns the results of these two fusions as a pair.
FailureOr<std::pair<Operation *, Operation *>>
tileAndFuseConsumerIntoDoubleLoops(RewriterBase &rewriter, scf::ForallOp &outerLoop,
                                   scf::ForOp &innerLoop, Operation &operation);

/// Run a narrow local CSE over `op` and its nested regions. This is useful for
/// deduplicating loop-index affine.apply ops introduced by tiling/fusion
/// without running broad canonicalization that may disturb loop-result relays.
void eliminateLocalCommonSubexpressions(RewriterBase &rewriter, Operation *op);

Value createExtractSliceFromState(RewriterBase &rewriter, Location loc, Value fullTensor,
                                  ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides);

void pointRewriterToForallParallel(RewriterBase &rewriter, scf::ForallOp forall);

linalg::GenericOp cloneGenericOnTile(RewriterBase &rewriter, linalg::GenericOp sourceGeneric,
                                     Value inputTile, Value initTile, Location loc);

} // namespace mlir

#endif // LOOPTR_UTILS_H
