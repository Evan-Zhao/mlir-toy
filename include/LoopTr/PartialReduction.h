#ifndef LOOPTR_PARTIAL_REDUCTION_H
#define LOOPTR_PARTIAL_REDUCTION_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

namespace mlir::transform {

/// Describes how a reduction over a tensor produced by an scf.forall can be split.
///
/// The split dimension is described both in consumer iteration-space coordinates and in the
/// producer tensor coordinates whose tile offset is controlled by one forall induction variable.
/// Partial-reduction fusion removes that producer tensor dimension from the split-local tile and
/// appends a new rfactor dimension indexed by the forall split id.
struct ReductionForallSplitPlan {
  // The result of the loop that is consumed by the reduction.
  OpOperand *loopProducedInput;
  // The initial value of the reduction.
  Value reductionInit;
  // The tensor.parallel_insert_slice that publishes the loop result to the reduction input.
  tensor::ParallelInsertSliceOp producerInsert;
  // The index of the forall induction variable that controls the offset on the reduction dimension.
  unsigned removedIvIndex;
  // The reduction dimension in the consumer's linalg iteration space.
  uint64_t opRedDim;
  // The corresponding dimension in the forall-produced operand tensor.
  uint64_t producerRedDim;
};

/// Result of cloning a reduction into an existing scf.forall as an rfactor partial.
struct PartialReductionForallResult {
  /// The rebuilt forall loop with extras output for the rfactor tensor.
  scf::ForallOp newForall;
  /// The tiled, fused op inside the loop that performs the partial reduction.
  linalg::GenericOp rFactorOp;
  /// The reduction after the loop that merges the rfactor result output.
  linalg::ReduceOp writebackOp;
  /// Old-to-new operation pairs for ops cloned while rebuilding the forall body.
  SmallVector<std::pair<Operation *, Operation *>> clonedOps;
};

/// Detect which forall induction variable tiles the reduction dimension of `consumer`.
///
/// The reduction input must be a result of `loop`, and that result must be published by a
/// tensor.parallel_insert_slice whose offset on the producer tensor dimension corresponding to the
/// consumer reduction iterator is controlled by one forall IV.
FailureOr<ReductionForallSplitPlan>
detectReductionForallSplit(const TransformOpInterface &transform, scf::ForallOp loop,
                           linalg::GenericOp consumer);

/// Decompose the reduction `consumer` which uses `loop` result, over a reduction split plan `plan`,
/// into a in-loop partial (rfactor) reduction, and an out-loop merge (writeback) reduction.
///
/// Rebuilds the forall with an extra rfactor shared output, extracts the split-local rfactor tile,
/// clones the reduction into the loop to run over the tile, and creates the writeback reduction
/// after the loop.
///
/// This op also notifies the rewriter of the op rewrite events.
FailureOr<PartialReductionForallResult> rFactorReductionUnderForall(TransformOpInterface transform,
                                                                    TransformRewriter &rewriter,
                                                                    scf::ForallOp forall,
                                                                    linalg::GenericOp consumer,
                                                                    ReductionForallSplitPlan &plan);

} // namespace mlir::transform

#endif // LOOPTR_PARTIAL_REDUCTION_H
