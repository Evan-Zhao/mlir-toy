#ifndef LOOPTR_UTILS_H
#define LOOPTR_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
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

struct DefUsePathCollection {
  llvm::SetVector<Operation *> operations;
  SmallVector<Operation *> descendants;
};

/// Find descendant operations by forward BFS from `ancestorValues`, then
/// collect the operations on paths between the ancestors and those descendants.
///
/// Forward traversal stops at every operation accepted by `isDescendant`. If
/// `stopAfterFirstDescendant` is true, the first accepted operation in BFS
/// order is the only descendant. The returned operations are the intersection
/// of the forward and backward slices, in producer-to-consumer topological
/// order. They include defining operations of ancestor values and accepted
/// descendants. Paths through block arguments are not followed.
DefUsePathCollection collectOpsOnDefUsePaths(ValueRange ancestorValues,
                                             llvm::function_ref<bool(Operation *)> isDescendant,
                                             bool stopAfterFirstDescendant = false);

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
inline FailureOr<tensor::ParallelInsertSliceOp>
getParallelInsertSliceForLoopResult(scf::ForallOp loop, OpResult result) {
  if (result.getOwner() != loop.getOperation())
    return failure();
  BlockArgument bbArg = loop.getTiedBlockArgument(result);
  SmallVector<Operation *> combiningOps = loop.getCombiningOps(bbArg);
  if (!llvm::hasSingleElement(combiningOps))
    return failure();
  auto insertSlice = dyn_cast<tensor::ParallelInsertSliceOp>(combiningOps.front());
  if (!insertSlice)
    return failure();
  return insertSlice;
}

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
FailureOr<BinaryReductionCombinerMatch> matchBinaryReductionCombiner(linalg::GenericOp generic,
                                                                     unsigned resultNumber,
                                                                     bool emitDiagnostics = false);

/// Returns the sizes of each dimension of `tensor` as a vector of `OpFoldResult`.
/// For dynamic dimensions, creates a `tensor.dim` op to query the size at runtime;
/// for static dimensions, returns the constant integer attribute directly.
SmallVector<OpFoldResult> getMixedTensorSizes(RewriterBase &rewriter, Location loc, Value tensor);

/// Move the insertion point of `builder` to the combining op region of the `forall` loop.
/// We have a helper for this action because it is error-prone.
void pointBuilderToForallParallel(OpBuilder &builder, scf::ForallOp forall);

/// Clone the operations in `block` into the current insertion point of `builder`, except for the
/// terminator. Returns a vector of pairs of the original and cloned operations.
inline SmallVector<std::pair<Operation *, Operation *>>
cloneBlockWithoutTerminator(OpBuilder &builder, Block &block, IRMapping &mapping) {
  SmallVector<std::pair<Operation *, Operation *>> clonedOps;
  for (Operation &op : block.without_terminator()) {
    Operation *cloned = builder.clone(op, mapping);
    clonedOps.emplace_back(&op, cloned);
  }
  return clonedOps;
}

struct ForallOutputExtension {
  scf::ForallOp forall;
  IRMapping mapping;
  unsigned oldOutputCount;
  SmallVector<std::pair<Operation *, Operation *>> clonedOps;

  ValueRange getAppendedOutputArgs() {
    return forall.getRegionOutArgs().drop_front(oldOutputCount);
  }
  auto getAppendedResults() { return forall.getResults().drop_front(oldOutputCount); }
  auto getPreservedResults() { return forall.getResults().take_front(oldOutputCount); }
};

/// Clone `forall` at the rewriter's current insertion point with the same
/// iteration space and existing outputs, plus `appendedOutputs`. The body and
/// combining ops are cloned, and `mapping` maps old induction variables,
/// output arguments, body operations, and values to their clones. The old
/// forall is left in place for the caller to replace at the appropriate point
/// in its rewrite. All outputs must dominate the insertion point.
ForallOutputExtension cloneForallWithAppendedOutputs(RewriterBase &rewriter, scf::ForallOp forall,
                                                     ValueRange appendedOutputs);

/// Notify the rewriter listener that each cloned operation, and each nested
/// operation at the same preorder position, replaces its original counterpart.
void notifyClonedOpsRecursively(RewriterBase &rewriter,
                                ArrayRef<std::pair<Operation *, Operation *>> clonedOps);

enum class DefChainAction : uint8_t { Clone, Move };

/// Make all of `values` available at the insertion point of `rewriter`, which recursively clones or
/// moves `values` and their defining operations as needed. Every operation-defined value that does
/// not dominate the insertion point is copied; a non-dominating block argument causes failure.
/// In `DefChainAction::Move` mode, cloning is followed by replacing every result of each original
/// operation and erasing it. Replacement is deferred until every chain has been cloned
/// successfully.
///
/// Callers are responsible for establishing that cloning or moving the
/// operations is legal and that the insertion point dominates replaced uses.
FailureOr<SmallVector<Value>>
makeValuesAvailableAtInsertionPoint(RewriterBase &rewriter, ValueRange values, IRMapping &mapping,
                                    DefChainAction action = DefChainAction::Clone);

/// Move all definition chains needed by the operands of `toMoveOperands`.
LogicalResult recursiveMoveOperandsBeforeOp(Operation &toMoveOperands, RewriterBase &rewriter,
                                            Operation &moveBefore);

/// Run `scf::tileAndFuseConsumerOfSlice` and print internal detailed error when it fails.
FailureOr<scf::SCFFuseConsumerOfSliceResult>
tileAndFuseConsumerWithDebug(RewriterBase &rewriter, Operation &consumer,
                             MutableArrayRef<LoopLikeOpInterface> loops);

/// Listener that follows operations across rewrite notifications.
///
/// The listener forwards all events to `previous` and updates `trackedOps`
/// when a tracked operation is replaced. Erased operations, and operations for
/// which no unique same-kind replacement can be inferred, are removed from the
/// vector.
struct TrackedOperationsListener : public RewriterBase::ForwardingListener {
  TrackedOperationsListener(SmallVectorImpl<Operation *> &trackedOps,
                            OpBuilder::Listener *previous);

  void notifyOperationReplaced(Operation *op, Operation *newOp) override;
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;
  void notifyOperationErased(Operation *op) override;

private:
  SmallVectorImpl<Operation *> &trackedOps;
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

} // namespace mlir

#endif // LOOPTR_UTILS_H
