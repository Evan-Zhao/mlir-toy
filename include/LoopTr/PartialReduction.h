#ifndef LOOPTR_PARTIAL_REDUCTION_H
#define LOOPTR_PARTIAL_REDUCTION_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace mlir::transform {

/// Describes how a reduction over a tensor produced by an scf.forall can be split.
///
/// The split dimension is the producer tensor dimension whose tile offset is controlled by one
/// forall induction variable. Partial-reduction fusion removes that dimension from the split-local
/// tile and appends a new rfactor dimension indexed by the forall split id.
struct ReductionForallSplitPlan {
  // The result of the loop that is consumed by the reduction.
  Value producerResult;
  // The tensor.parallel_insert_slice that publishes the loop result to the reduction input.
  tensor::ParallelInsertSliceOp producerInsert;
  // The index of the forall induction variable that controls the offset on the reduction dimension.
  unsigned removedIvIndex;
  // The reduction dimension in the forall-produced tensor.
  uint64_t reductionDim;
  // The initial value of the reduction.
  Value reductionInit;
};

/// Result of cloning a reduction into an existing scf.forall as an rfactor partial.
///
/// `newForall` is the rebuilt loop with one extra shared output, `partialReduce` is the
/// split-local reduction cloned inside the loop, and the op-pair lists record payload clones so
/// transform handle tracking can be updated by the caller.
struct PartialReductionForallResult {
  scf::ForallOp newForall;
  linalg::GenericOp partialReduce;
  SmallVector<std::pair<Operation *, Operation *>> clonedOps;
  SmallVector<std::pair<Operation *, Operation *>> clonedCombiningOps;
};

/// Detect which forall induction variable tiles the reduction dimension of `consumer`.
///
/// The reduction input must be a result of `loop`, and that result must be published by a
/// tensor.parallel_insert_slice whose offset on `reductionDim` is controlled by one forall IV.
FailureOr<ReductionForallSplitPlan>
detectReductionForallSplit(const TransformOpInterface &transform, scf::ForallOp loop,
                           linalg::GenericOp consumer, uint64_t reductionDim);

/// Clone `consumer` into `loop` as a split-local rfactor reduction.
///
/// Rebuilds the forall with an extra rfactor shared output, clones the original body and combining
/// ops, extracts the split-local rfactor tile, clones the reduction on the in-loop producer tile,
/// and publishes the partial result into the rfactor tensor. The final write-back merge reduction
/// is intentionally left to the caller.
///
FailureOr<PartialReductionForallResult>
fusePartialReductionIntoForall(TransformOpInterface transform, RewriterBase &rewriter,
                               scf::ForallOp loop, linalg::GenericOp consumer,
                               PartialReductionOpInterface consumerPR,
                               ReductionForallSplitPlan &plan);

} // namespace mlir::transform

#endif // LOOPTR_PARTIAL_REDUCTION_H
