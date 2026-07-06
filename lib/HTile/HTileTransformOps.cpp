#include "HTile/HTileTransformOps.h"
#include "HTile/HTileDialect.h"

#include "LoopTr/Utils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>

#define DEBUG_TYPE "htile-transform-ops"

using namespace mlir;
using bufferization::ToTensorOp;

namespace mlir::transform {
namespace {

bool isSingleResultTensorGeneric(linalg::GenericOp op) {
  return op->getNumResults() == 1 && op.getNumDpsInits() == 1 &&
         isa<RankedTensorType>(op->getResult(0).getType());
}

FailureOr<size_t> getDimPosition(AffineExpr expr) {
  auto dim = dyn_cast<AffineDimExpr>(expr);
  if (!dim)
    return failure();
  return dim.getPosition();
}

SmallVector<size_t> getMapDims(AffineMap map) {
  SmallVector<size_t> dims;
  dims.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    auto maybePos = getDimPosition(expr);
    if (failed(maybePos))
      return {};
    dims.push_back(*maybePos);
  }
  return dims;
}

FailureOr<Value> createBroadcastToResultShape(OpBuilder &builder, Location loc, Value input,
                                              AffineMap indexingMap, RankedTensorType resultType,
                                              SmallVectorImpl<Operation *> &createdOps) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();
  auto targetType = RankedTensorType::get(resultType.getShape(), inputType.getElementType(),
                                          resultType.getEncoding());
  if (inputType.getShape() == targetType.getShape() && indexingMap.isIdentity())
    return input;

  SmallVector<size_t> mappedDims = getMapDims(indexingMap);
  size_t inputRank = static_cast<size_t>(inputType.getRank());
  size_t resultRank = static_cast<size_t>(resultType.getRank());
  if (mappedDims.size() != inputRank)
    return failure();

  SmallVector<bool> used(resultRank, false);
  std::optional<size_t> previous;
  for (size_t dim : mappedDims) {
    if (dim >= resultRank || used[dim] || (previous && dim <= *previous))
      return failure();
    used[dim] = true;
    previous = dim;
  }

  SmallVector<int64_t> broadcastDims;
  for (size_t dim = 0; dim < resultRank; ++dim)
    if (!used[dim])
      broadcastDims.push_back(static_cast<int64_t>(dim));

  if (broadcastDims.empty()) {
    if (inputType.getShape() == targetType.getShape())
      return input;
    return failure();
  }

  auto broadcast = htile::BroadcastOp::create(builder, loc, targetType, input,
                                              builder.getDenseI64ArrayAttr(broadcastDims));
  createdOps.push_back(broadcast);
  return broadcast.getResult();
}

FailureOr<Value> materializeScalarAsTile(OpBuilder &builder, Location loc, Value scalar,
                                         RankedTensorType shapeType,
                                         SmallVectorImpl<Operation *> &createdOps) {
  if (isa<RankedTensorType>(scalar.getType()))
    return failure();
  auto resultType =
      RankedTensorType::get(shapeType.getShape(), scalar.getType(), shapeType.getEncoding());
  auto full = htile::FullOp::create(builder, loc, resultType, scalar);
  createdOps.push_back(full);
  return full.getResult();
}

FailureOr<Value> materializeIndexTile(OpBuilder &builder, Location loc, linalg::IndexOp indexOp,
                                      RankedTensorType resultType,
                                      SmallVectorImpl<Operation *> &createdOps) {
  size_t dim = static_cast<size_t>(indexOp.getDim());
  if (dim >= static_cast<size_t>(resultType.getRank()))
    return failure();

  int64_t extent = resultType.getDimSize(static_cast<int64_t>(dim));
  if (extent == ShapedType::kDynamic)
    return failure();

  auto indexType = builder.getIndexType();
  auto arangeType = RankedTensorType::get({extent}, indexType, resultType.getEncoding());
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  createdOps.push_back(zero.getDefiningOp());
  Value end = arith::ConstantIndexOp::create(builder, loc, extent);
  createdOps.push_back(end.getDefiningOp());
  auto arange = htile::ArangeOp::create(builder, loc, arangeType, zero, end);
  createdOps.push_back(arange);

  SmallVector<AffineExpr> exprs;
  exprs.push_back(builder.getAffineDimExpr(static_cast<unsigned>(dim)));
  AffineMap indexingMap =
      AffineMap::get(resultType.getRank(), /*symbolCount=*/0, exprs, builder.getContext());
  return createBroadcastToResultShape(builder, loc, arange.getResult(), indexingMap, resultType,
                                      createdOps);
}

FailureOr<StringRef> getHTileReductionKind(Operation *combiner) {
  if (isa<arith::AddFOp>(combiner))
    return StringRef("sum");
  if (isa<arith::MaximumFOp>(combiner))
    return StringRef("max");
  return failure();
}

FailureOr<Value> createTensorScalarLikeOp(OpBuilder &builder, Location loc, Operation *scalarOp,
                                          ValueRange operands, RankedTensorType resultShape,
                                          SmallVectorImpl<Operation *> &createdOps) {

  // Keep scalar constants scalar. Consumers that need tensor operands materialize
  // them with htile.full, which avoids large dense tensor constants in semantic IR.
  if (auto constant = dyn_cast<arith::ConstantOp>(scalarOp)) {
    if (!operands.empty() || isa<ShapedType>(constant.getType()))
      return failure();
    Operation *created = builder.clone(*constant.getOperation());
    createdOps.push_back(created);
    return created->getResult(0);
  }

  // This is a default case that covers all "element-wise" operations that returns one result.
  // The canonical examples are most arith ops and math ops.
  bool elemwise = scalarOp->hasTrait<OpTrait::Elementwise>(),
       oneResult = scalarOp->getNumResults() == 1, noRegions = scalarOp->getNumRegions() == 0,
       noSuccessors = scalarOp->getNumSuccessors() == 0,
       noMemoryEffects = isMemoryEffectFree(scalarOp);
  LLVM_DEBUG(llvm::dbgs() << "Creating tensor-scalar-like op for: " << *scalarOp << "\n"
                          << "  elemwise = " << elemwise << ", oneResult: " << oneResult
                          << ", noRegions: " << noRegions << ", noSuccessors: " << noSuccessors
                          << ", noMemoryEffects: " << noMemoryEffects << "\n");
  if (elemwise && oneResult && noRegions && noSuccessors && noMemoryEffects) {
    auto resultType = RankedTensorType::get(
        resultShape.getShape(), scalarOp->getResult(0).getType(), resultShape.getEncoding());
    OperationState state(loc, scalarOp->getName());
    state.addOperands(operands);
    state.addTypes(resultType);
    state.addAttributes(scalarOp->getAttrs());
    Operation *created = builder.create(state);
    createdOps.push_back(created);
    return created->getResult(0);
  }
  scalarOp->emitError() << "this scalar operation is not supported";
  return failure();
}

LogicalResult expandAffineApplyOpsInLinalgBody(linalg::GenericOp op) {
  SmallVector<affine::AffineApplyOp> affineApplies;
  op.getBody()->walk([&](affine::AffineApplyOp affineApply) {
    if (affineApply->getParentOp() == op)
      affineApplies.push_back(affineApply);
  });

  for (affine::AffineApplyOp affineApply : affineApplies) {
    OpBuilder builder(affineApply);
    std::optional<SmallVector<Value, 8>> expanded = affine::expandAffineMap(
        builder, affineApply.getLoc(), affineApply.getAffineMap(), affineApply.getOperands());
    if (!expanded || expanded->size() != 1)
      return failure();
    affineApply.getResult().replaceAllUsesWith((*expanded)[0]);
    affineApply->erase();
  }
  return success();
}

LogicalResult rewriteFill(RewriterBase &rewriter, linalg::FillOp op) {
  if (op->getNumResults() != 1)
    return failure();
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return failure();
  rewriter.setInsertionPoint(op);
  auto full = htile::FullOp::create(rewriter, op.getLoc(), resultType, op.getInputs().front());
  rewriter.replaceOp(op, full.getResult());
  return success();
}

LogicalResult rewriteBroadcast(RewriterBase &rewriter, linalg::BroadcastOp op) {
  if (op->getNumResults() != 1)
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !resultType)
    return failure();

  rewriter.setInsertionPoint(op);
  auto broadcast = htile::BroadcastOp::create(rewriter, op.getLoc(), resultType, op.getInput(),
                                              op.getDimensionsAttr());
  rewriter.replaceOp(op, broadcast.getResult());
  return success();
}

LogicalResult rewriteContraction(RewriterBase &rewriter, linalg::GenericOp op) {
  if (!isSingleResultTensorGeneric(op) || op.getInputs().size() != 2)
    return failure();

  SmallVector<unsigned> parallelDims, reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);
  if (reductionDims.size() != 1)
    return failure();

  auto isMapEqualToDims = [](AffineMap map, ArrayRef<unsigned> dims) {
    if (map.getNumResults() != dims.size())
      return false;
    for (auto [expr, dim] : llvm::zip_equal(map.getResults(), dims)) {
      auto maybePos = getDimPosition(expr);
      if (failed(maybePos) || *maybePos != dim)
        return false;
    }
    return true;
  };
  auto detectTranspose = [&](AffineMap map, unsigned dim0, unsigned dim1,
                             bool &result) -> LogicalResult {
    if (isMapEqualToDims(map, {dim0, dim1}))
      result = false;
    else if (isMapEqualToDims(map, {dim1, dim0}))
      result = true;
    else
      return failure();
    return success();
  };

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  AffineMap lhsMap = maps[0], rhsMap = maps[1], outMap = maps[2];
  unsigned k = reductionDims[0];
  // Output map must trivially cover all parallel dims.
  if (!isMapEqualToDims(outMap, parallelDims))
    return failure();
  bool transposeA = false, transposeB = false;
  if (parallelDims.size() == 2) {
    // mat-mat contraction.
    unsigned n = parallelDims[0], m = parallelDims[1];
    if (failed(detectTranspose(lhsMap, n, k, transposeA)) ||
        failed(detectTranspose(rhsMap, k, m, transposeB)))
      return failure();
  } else if (parallelDims.size() == 1 && lhsMap.getNumResults() == 1) {
    // vec-mat contraction, lhs is a vector, rhs is a matrix.
    size_t m = parallelDims[0];
    if (!isMapEqualToDims(lhsMap, {k}))
      return failure();
    if (failed(detectTranspose(rhsMap, k, m, transposeB)))
      return failure();
  } else if (parallelDims.size() == 1 && rhsMap.getNumResults() == 1) {
    // mat-vec contraction, lhs is a matrix, rhs is a vector.
    size_t n = parallelDims[0];
    if (!isMapEqualToDims(rhsMap, {k}))
      return failure();
    if (failed(detectTranspose(lhsMap, n, k, transposeA)))
      return failure();
  } else {
    return failure();
  }

  rewriter.setInsertionPoint(op);
  auto lhsAttr = transposeA ? rewriter.getUnitAttr() : UnitAttr{},
       rhsAttr = transposeB ? rewriter.getUnitAttr() : UnitAttr{};
  auto dot =
      htile::DotOp::create(rewriter, op.getLoc(), op.getResult(0).getType(), op.getInputs()[0],
                           op.getInputs()[1], op.getDpsInits()[0], lhsAttr, rhsAttr, StringAttr{});
  rewriter.replaceOp(op, dot.getResult());
  return success();
}

LogicalResult rewriteReduction(RewriterBase &rewriter, linalg::GenericOp op) {
  if (!isSingleResultTensorGeneric(op) || op.getInputs().size() != 1)
    return failure();

  SmallVector<unsigned> reductionDims;
  op.getReductionDims(reductionDims);
  if (reductionDims.size() != 1)
    return failure();
  unsigned reductionAxis = reductionDims[0];

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (!maps.front().isIdentity())
    return failure();

  SmallVector<size_t> outputDims = getMapDims(maps.back());
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  if (outputDims.size() != static_cast<size_t>(resultType.getRank()))
    return failure();
  if (llvm::is_contained(outputDims, reductionAxis))
    return failure();

  FailureOr<BinaryReductionCombinerMatch> combiner = matchBinaryReductionCombiner(op, 0);
  if (failed(combiner) || combiner->nonAccumulator != op.getBlock()->getArgument(0))
    return failure();
  FailureOr<StringRef> kind = getHTileReductionKind(combiner->combiner);
  if (failed(kind))
    return failure();

  rewriter.setInsertionPoint(op);
  auto reduce =
      htile::ReduceOp::create(rewriter, op.getLoc(), op.getResult(0).getType(), op.getInputs()[0],
                              rewriter.getI64IntegerAttr(static_cast<int64_t>(reductionAxis)),
                              rewriter.getStringAttr(*kind));
  Value combined;
  if (*kind == "sum")
    combined =
        arith::AddFOp::create(rewriter, op.getLoc(), op.getDpsInits()[0], reduce.getResult());
  else if (*kind == "max")
    combined =
        arith::MaximumFOp::create(rewriter, op.getLoc(), op.getDpsInits()[0], reduce.getResult());
  else
    return failure();
  rewriter.replaceOp(op, combined);
  return success();
}

LogicalResult rewriteElementwise(RewriterBase &rewriter, linalg::GenericOp op) {
  if (!isSingleResultTensorGeneric(op))
    return failure();
  SmallVector<unsigned> parallelDims;
  op.getParallelDims(parallelDims);
  if (parallelDims.size() != op.getNumLoops())
    return failure();
  if (failed(expandAffineApplyOpsInLinalgBody(op)))
    return failure();

  RankedTensorType resultType = cast<RankedTensorType>(op.getResult(0).getType());
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  rewriter.setInsertionPoint(op);

  // Track the operations created during the rewrite, and remove them if this function fails.
  SmallVector<Operation *> createdOps;
  bool failedAndRevert = true;
  auto guard = llvm::scope_exit([&]() {
    if (!failedAndRevert)
      return;
    for (Operation *op : llvm::reverse(createdOps))
      op->erase();
    createdOps.clear();
  });

  IRMapping mapping;
  unsigned argIndex = 0;
  auto rewriteOperands = [&](OperandRange range) {
    for (Value v : range) {
      FailureOr<Value> prepared = createBroadcastToResultShape(
          rewriter, op.getLoc(), v, maps[argIndex], resultType, createdOps);
      if (failed(prepared))
        return failure();
      mapping.map(op.getBlock()->getArgument(argIndex), *prepared);
      ++argIndex;
    }
    return success();
  };
  if (failed(rewriteOperands(op.getInputs())))
    return failure();
  if (failed(rewriteOperands(op.getDpsInits())))
    return failure();

  auto yield = dyn_cast<linalg::YieldOp>(op.getBlock()->getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return failure();

  for (Operation &bodyOp : op.getBlock()->without_terminator()) {
    if (auto index = dyn_cast<linalg::IndexOp>(bodyOp)) {
      FailureOr<Value> tensorIndex =
          materializeIndexTile(rewriter, bodyOp.getLoc(), index, resultType, createdOps);
      if (failed(tensorIndex))
        return failure();
      mapping.map(index.getResult(), *tensorIndex);
      continue;
    }

    SmallVector<Value> mappedOperands;
    for (Value operand : bodyOp.getOperands()) {
      Value mapped = mapping.lookupOrNull(operand);
      if (mapped && isa<RankedTensorType>(mapped.getType())) {
        mappedOperands.push_back(mapped);
        continue;
      }
      auto scalarVal = mapped ? mapped : operand;
      FailureOr<Value> tile =
          materializeScalarAsTile(rewriter, bodyOp.getLoc(), scalarVal, resultType, createdOps);
      if (failed(tile))
        return failure();
      mappedOperands.push_back(*tile);
    }
    FailureOr<Value> tensorOp = createTensorScalarLikeOp(rewriter, bodyOp.getLoc(), &bodyOp,
                                                         mappedOperands, resultType, createdOps);
    if (failed(tensorOp) || bodyOp.getNumResults() != 1)
      return failure();
    mapping.map(bodyOp.getResult(0), *tensorOp);
  }

  Value replacement = mapping.lookupOrNull(yield.getValues()[0]);
  if (!replacement)
    return failure();
  rewriter.replaceOp(op, replacement);
  failedAndRevert = false;
  return success();
}

LogicalResult rewriteOriginalLinalgOp(RewriterBase &rewriter, Operation *op) {
  if (!op->getParentOp())
    return success();
  if (auto fill = dyn_cast<linalg::FillOp>(op))
    return rewriteFill(rewriter, fill);
  if (auto broadcast = dyn_cast<linalg::BroadcastOp>(op))
    return rewriteBroadcast(rewriter, broadcast);
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op); linalgOp && !isa<linalg::GenericOp>(op)) {
    rewriter.setInsertionPoint(op);
    FailureOr<linalg::GenericOp> generic = linalg::generalizeNamedOp(rewriter, linalgOp);
    if (failed(generic))
      return failure();
    return rewriteOriginalLinalgOp(rewriter, *generic);
  }
  if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
    if (succeeded(rewriteContraction(rewriter, generic)))
      return success();
    if (succeeded(rewriteReduction(rewriter, generic)))
      return success();
    if (succeeded(rewriteElementwise(rewriter, generic)))
      return success();
    op->emitError() << "unsupported linalg.generic operation";
    return failure();
  }
  op->emitError() << "unsupported linalg operation";
  return failure();
}

LogicalResult foldExpandShapeOfSingleResultForall(RewriterBase &rewriter,
                                                  tensor::ExpandShapeOp expandOp) {
  auto forallOp = expandOp.getSrc().getDefiningOp<scf::ForallOp>();
  if (!forallOp) {
    expandOp.emitError() << "expected tensor.expand_shape source to be an scf.forall result";
    return failure();
  }
  if (forallOp.getNumResults() != 1 || !forallOp.getResult(0).hasOneUse()) {
    forallOp.emitError() << "expected single-result scf.forall with one use by tensor.expand_shape";
    return failure();
  }

  BlockArgument oldOutArg = forallOp.getRegionOutArgs().front();
  SmallVector<tensor::ParallelInsertSliceOp> inserts;
  for (OpOperand &use : oldOutArg.getUses()) {
    auto insert = dyn_cast<tensor::ParallelInsertSliceOp>(use.getOwner());
    if (!insert || insert.getDest() != oldOutArg) {
      use.getOwner()->emitError()
          << "expected scf.forall shared_out use to be tensor.parallel_insert_slice into the "
             "shared output";
      return failure();
    }
    inserts.push_back(insert);
  }
  if (inserts.empty()) {
    forallOp.emitError()
        << "expected scf.forall shared_out to be published by tensor.parallel_insert_slice";
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);
  Value expandedOutput = tensor::ExpandShapeOp::create(
      rewriter, forallOp.getLoc(), expandOp.getResultType(), forallOp.getOutputs().front(),
      expandOp.getReassociationIndices(), expandOp.getMixedOutputShape());
  auto newForallOp = scf::ForallOp::create(
      rewriter, forallOp.getLoc(), forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), expandedOutput, forallOp.getMapping(),
      [&](OpBuilder &, Location, ValueRange bbArgs) {
        SmallVector<Value> replacements = llvm::to_vector(bbArgs);
        rewriter.mergeBlocks(forallOp.getBody(), bbArgs.front().getParentBlock(), replacements);
      });

  auto expandRankReducedSliceParams =
      [&](ArrayRef<OpFoldResult> oldParams,
          int64_t unitValue) -> FailureOr<SmallVector<OpFoldResult>> {
    RankedTensorType expandedType = expandOp.getResultType();
    SmallVector<ReassociationIndices> reassociation = expandOp.getReassociationIndices();
    if (reassociation.size() != oldParams.size()) {
      expandOp.emitError()
          << "expand_shape reassociation rank does not match tensor.parallel_insert_slice params";
      return failure();
    }

    OpFoldResult unit = rewriter.getIndexAttr(unitValue);
    SmallVector<OpFoldResult> expandedParams;
    expandedParams.reserve(expandedType.getRank());
    for (auto [oldDim, group] : llvm::enumerate(reassociation)) {
      std::optional<int64_t> carriedDim;
      for (int64_t expandedDim : group) {
        int64_t dimSize = expandedType.getDimSize(expandedDim);
        if (dimSize == 1)
          continue;
        if (ShapedType::isDynamic(dimSize) || carriedDim) {
          expandOp.emitError()
              << "expected each reassociation group to have at most one static non-unit "
                 "dimension";
          return failure();
        }
        carriedDim = expandedDim;
      }

      if (!carriedDim)
        carriedDim = group.back();
      for (int64_t expandedDim : group)
        expandedParams.push_back(expandedDim == *carriedDim ? oldParams[oldDim] : unit);
    }
    return expandedParams;
  };

  for (tensor::ParallelInsertSliceOp insert : inserts) {
    auto expandedOffsetsR = expandRankReducedSliceParams(insert.getMixedOffsets(), 0),
         expandedSizesR = expandRankReducedSliceParams(insert.getMixedSizes(), 1),
         expandedStridesR = expandRankReducedSliceParams(insert.getMixedStrides(), 1);
    if (failed(expandedOffsetsR) || failed(expandedSizesR) || failed(expandedStridesR)) {
      insert.emitError()
          << "failed to remap tensor.parallel_insert_slice params for expanded forall result";
      return failure();
    }
    rewriter.setInsertionPoint(insert);
    tensor::ParallelInsertSliceOp::create(rewriter, insert.getLoc(), insert.getSource(),
                                          newForallOp.getRegionOutArgs().front(), *expandedOffsetsR,
                                          *expandedSizesR, *expandedStridesR);
    rewriter.eraseOp(insert);
  }

  rewriter.replaceOp(expandOp, newForallOp.getResult(0));
  rewriter.eraseOp(forallOp);
  return success();
}

LogicalResult foldForallResultExpands(RewriterBase &rewriter, Operation *target) {
  SmallVector<tensor::ExpandShapeOp> expandOps;
  target->walk([&](tensor::ExpandShapeOp expandOp) { expandOps.push_back(expandOp); });

  for (tensor::ExpandShapeOp expandOp : expandOps) {
    if (!expandOp->getBlock())
      continue;
    if (failed(foldExpandShapeOfSingleResultForall(rewriter, expandOp))) {
      expandOp.emitError() << "failed to fold tensor.expand_shape into producer scf.forall";
      return failure();
    }
  }
  return success();
}

LogicalResult applyRewritesGreedily(TransformRewriter &rewriter, Operation *target,
                                    const std::function<void(RewritePatternSet &)> &patternSet) {
  RewritePatternSet patterns(target->getContext());
  patternSet(patterns);
  GreedyRewriteConfig config;
  config.setListener(static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  return applyPatternsGreedily(target, std::move(patterns), config);
}

struct OutputStore {
  Value returnedTensor;
  BlockArgument outputMemrefArg;
};

struct KernelAbiRewriteInfo {
  SmallVector<OutputStore> outputStores;
};

FailureOr<KernelAbiRewriteInfo> rewriteFunctionAbi(RewriterBase &rewriter, func::FuncOp funcOp) {
  if (funcOp.isDeclaration())
    return funcOp.emitError() << "cannot rewrite ABI of external function";

  FunctionType oldType = funcOp.getFunctionType();
  KernelAbiRewriteInfo info;
  SmallVector<Type> newInputTypes;
  SmallVector<Type> oldTensorInputTypes;
  SmallVector<unsigned> tensorInputIndices;
  SmallVector<DictionaryAttr> newArgAttrs;
  funcOp.getAllArgAttrs(newArgAttrs);
  newInputTypes.reserve(oldType.getNumInputs() + oldType.getNumResults());

  auto getMemrefAbiType = [&](Type type, StringRef role, size_t index) -> FailureOr<MemRefType> {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType)
      return funcOp.emitError() << "unsupported "
                                << (isa<TensorType>(type) ? "unranked tensor " : "non-tensor ")
                                << role << " " << index;
    if (tensorType.getEncoding())
      return funcOp.emitError() << "unsupported encoded tensor " << role << " " << index;
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  };

  for (auto [index, inputType] : llvm::enumerate(oldType.getInputs())) {
    if (!isa<TensorType>(inputType)) {
      newInputTypes.push_back(inputType);
      continue;
    }
    FailureOr<MemRefType> memrefType = getMemrefAbiType(inputType, "argument", index);
    if (failed(memrefType))
      return failure();
    newInputTypes.push_back(*memrefType);
    oldTensorInputTypes.push_back(inputType);
    tensorInputIndices.push_back(index);
  }
  for (auto [index, resultType] : llvm::enumerate(oldType.getResults())) {
    FailureOr<MemRefType> memrefType = getMemrefAbiType(resultType, "result", index);
    if (failed(memrefType))
      return failure();
    newInputTypes.push_back(*memrefType);
  }

  funcOp.setType(FunctionType::get(funcOp.getContext(), newInputTypes, {}));

  Block &entry = funcOp.getBody().front();
  for (size_t index = 0, e = oldType.getNumInputs(); index < e; ++index)
    entry.getArgument(index).setType(newInputTypes[index]);
  for (size_t index = oldType.getNumInputs(), e = newInputTypes.size(); index < e; ++index)
    entry.addArgument(newInputTypes[index], funcOp.getLoc());

  if (!newArgAttrs.empty()) {
    auto emptyDict = DictionaryAttr::get(funcOp.getContext());
    newArgAttrs.append(oldType.getNumResults(), emptyDict);
    funcOp.setAllArgAttrs(newArgAttrs);
  }
  funcOp.setAllResultAttrs(ArrayRef<DictionaryAttr>{});

  rewriter.setInsertionPointToStart(&entry);
  for (auto [oldTensorType, argIndex] : llvm::zip_equal(oldTensorInputTypes, tensorInputIndices)) {
    BlockArgument arg = entry.getArgument(argIndex);
    auto tensor = ToTensorOp::create(rewriter, arg.getLoc(), oldTensorType, arg,
                                     /*restrict=*/true, /*writable=*/true);
    for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
      if (use.getOwner() == tensor)
        continue;
      use.set(tensor.getResult());
    }
  }

  SmallVector<func::ReturnOp> returns;
  funcOp.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
  if (returns.size() != 1)
    return funcOp.emitError() << "expected exactly one return op";
  for (func::ReturnOp returnOp : returns) {
    if (returnOp.getNumOperands() != oldType.getNumResults())
      return returnOp.emitError() << "return operand count does not match function result count";
    for (auto [index, returned] : llvm::enumerate(returnOp.getOperands())) {
      info.outputStores.push_back(
          {returned, cast<BlockArgument>(entry.getArgument(oldType.getNumInputs() + index))});
    }
    rewriter.setInsertionPoint(returnOp);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp);
  }

  return info;
}

LogicalResult rewriteExtractSliceAsLoad(RewriterBase &rewriter, func::FuncOp funcOp,
                                        tensor::ExtractSliceOp extract) {
  auto toTensor = extract.getSource().getDefiningOp<ToTensorOp>();
  if (!toTensor)
    return extract.emitError()
           << "expected tensor.extract_slice source to be a function memref argument";

  auto memrefArg = dyn_cast<BlockArgument>(toTensor.getBuffer());
  if (!memrefArg || memrefArg.getOwner() != &funcOp.getBody().front() ||
      !isa<MemRefType>(memrefArg.getType()))
    return extract.emitError()
           << "expected tensor.extract_slice source to be a function memref argument";

  if (!extract.hasUnitStride())
    return extract.emitError() << "unsupported non-unit tensor.extract_slice stride";

  if (extract->use_empty())
    return extract.emitError() << "expected tensor.extract_slice result to feed an htile op";

  StringRef htileNamespace = htile::HTileDialect::getDialectNamespace();
  for (OpOperand &use : extract->getUses()) {
    Operation *owner = use.getOwner();
    if (!owner->getDialect() || owner->getDialect()->getNamespace() != htileNamespace)
      return extract.emitError() << "expected tensor.extract_slice result to feed only htile ops";
  }

  rewriter.setInsertionPoint(extract);
  SmallVector<Value> offsets =
      getValueOrCreateConstantIndexOp(rewriter, extract.getLoc(), extract.getMixedOffsets());
  auto load = htile::LoadOp::create(rewriter, extract.getLoc(), extract.getResultType(),
                                    toTensor.getBuffer(), offsets);
  rewriter.replaceOp(extract, load.getResult());
  return success();
}

struct ForallStoreGroup {
  scf::ForallOp forallOp;
  SmallVector<OutputStore> outputStores;
};

LogicalResult materializeStoreForForallResult(RewriterBase &rewriter, OutputStore store) {
  auto result = dyn_cast<OpResult>(store.returnedTensor);
  if (!result)
    return emitError(store.returnedTensor.getLoc())
           << "expected returned tensor to be produced by scf.forall";
  auto forallOp = dyn_cast<scf::ForallOp>(result.getDefiningOp());
  if (!forallOp)
    return result.getDefiningOp()->emitError()
           << "expected returned tensor to be produced by scf.forall";

  BlockArgument outputArg = forallOp.getTiedBlockArgument(result);
  SmallVector<Operation *> combiningOps = forallOp.getCombiningOps(outputArg);
  if (combiningOps.empty())
    return forallOp.emitError() << "expected returned scf.forall result to be published";

  for (Operation *combiningOp : combiningOps) {
    auto insert = dyn_cast<tensor::ParallelInsertSliceOp>(combiningOp);
    if (!insert)
      return combiningOp->emitError()
             << "expected returned scf.forall result to use tensor.parallel_insert_slice";
    if (!insert.hasUnitStride())
      return insert.emitError() << "unsupported non-unit tensor.parallel_insert_slice stride";

    auto inParallel = insert->getParentOfType<scf::InParallelOp>();
    if (!inParallel)
      return insert.emitError() << "expected tensor.parallel_insert_slice under scf.forall";

    rewriter.setInsertionPoint(inParallel);
    SmallVector<Value> offsets =
        getValueOrCreateConstantIndexOp(rewriter, insert.getLoc(), insert.getMixedOffsets());
    htile::StoreOp::create(rewriter, insert.getLoc(), insert.getSource(), store.outputMemrefArg,
                           offsets);
    rewriter.eraseOp(insert);
  }

  return success();
}

LogicalResult rebuildForallWithoutOutputs(RewriterBase &rewriter, scf::ForallOp forallOp) {
  for (BlockArgument outputArg : forallOp.getRegionOutArgs()) {
    if (!outputArg.use_empty())
      return forallOp.emitError() << "unsupported remaining use of scf.forall shared_out";
  }
  for (OpResult result : forallOp->getResults()) {
    if (!result.use_empty())
      return forallOp.emitError()
             << "expected returned scf.forall result to have no remaining uses";
  }

  SmallVector<Value> oldOutputs = llvm::to_vector(forallOp.getOutputs());
  rewriter.setInsertionPoint(forallOp);
  scf::ForallOp::create(
      rewriter, forallOp.getLoc(), forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), ValueRange{}, forallOp.getMapping(),
      [&](OpBuilder &nestedBuilder, Location, ValueRange bbArgs) {
        SmallVector<Value> replacements = llvm::to_vector(bbArgs);
        replacements.append(oldOutputs.begin(), oldOutputs.end());
        rewriter.mergeBlocks(forallOp.getBody(), nestedBuilder.getBlock(), replacements);
      });
  rewriter.eraseOp(forallOp);

  for (Value oldOutput : oldOutputs) {
    Operation *def = oldOutput.getDefiningOp();
    if (def && def->use_empty())
      rewriter.eraseOp(def);
  }
  return success();
}

LogicalResult rewriteOutputStores(RewriterBase &rewriter, KernelAbiRewriteInfo &info) {
  SmallVector<ForallStoreGroup> groups;
  for (OutputStore store : info.outputStores) {
    auto result = dyn_cast<OpResult>(store.returnedTensor);
    if (!result)
      return emitError(store.returnedTensor.getLoc())
             << "expected returned tensor to be produced by scf.forall";
    auto forallOp = dyn_cast<scf::ForallOp>(result.getDefiningOp());
    if (!forallOp)
      return result.getDefiningOp()->emitError()
             << "expected returned tensor to be produced by scf.forall";

    auto existing = llvm::find_if(
        groups, [&](const ForallStoreGroup &group) { return group.forallOp == forallOp; });
    if (existing == groups.end()) {
      groups.push_back({forallOp, {}});
      existing = std::prev(groups.end());
    }
    existing->outputStores.push_back(store);
  }

  for (ForallStoreGroup &group : groups) {
    if (!group.forallOp->getBlock())
      continue;
    for (OutputStore store : group.outputStores) {
      if (failed(materializeStoreForForallResult(rewriter, store))) {
        group.forallOp.emitError() << "failed to materialize htile.store for scf.forall result";
        return failure();
      }
    }
    if (failed(rebuildForallWithoutOutputs(rewriter, group.forallOp))) {
      group.forallOp.emitError() << "failed to rebuild scf.forall without tensor outputs";
      return failure();
    }
  }
  return success();
}

LogicalResult rewriteExtractSlicesAsLoads(RewriterBase &rewriter, func::FuncOp funcOp) {
  SmallVector<tensor::ExtractSliceOp> extracts;
  funcOp.walk([&](tensor::ExtractSliceOp extract) { extracts.push_back(extract); });

  for (tensor::ExtractSliceOp extract : extracts) {
    if (!extract->getBlock())
      continue;
    if (failed(rewriteExtractSliceAsLoad(rewriter, funcOp, extract))) {
      extract.emitError() << "failed to rewrite tensor.extract_slice as htile.load";
      return failure();
    }
  }
  return success();
}

} // namespace

void HTileLinalgToSemanticOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure HTileLinalgToSemanticOp::applyToOne(TransformRewriter &rewriter,
                                                                Operation *target,
                                                                ApplyToEachResultList &results,
                                                                TransformState &state) {
  (void)results;
  (void)state;
  auto transform = cast<TransformOpInterface>(getOperation());

  SmallVector<Operation *> originalLinalgOps;
  target->walk([&](linalg::LinalgOp op) { originalLinalgOps.push_back(op); });

  for (Operation *op : originalLinalgOps) {
    if (succeeded(rewriteOriginalLinalgOp(rewriter, op)))
      continue;
    return emitSilenceableFailure(transform) << "failed to rewrite linalg op: " << *op;
  }

  if (failed(applyRewritesGreedily(rewriter, target, [&](RewritePatternSet &patterns) {
        tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
        tensor::populateBubbleUpExtractSliceOpPatterns(patterns);
        tensor::populateReassociativeReshapeFoldingPatterns(patterns);
        tensor::populateFoldTensorEmptyPatterns(patterns);
      })))
    return emitSilenceableFailure(transform, "failed to apply tensor cleanup patterns");
  if (failed(foldForallResultExpands(rewriter, target)))
    return emitSilenceableFailure(transform, "failed to fold forall result expand_shape ops");
  if (failed(applyRewritesGreedily(rewriter, target, [&](RewritePatternSet &patterns) {
        tensor::populateFoldTensorEmptyPatterns(patterns);
      })))
    return emitSilenceableFailure(transform, "failed to fold tensor.empty ops");

  return DiagnosedSilenceableFailure::success();
}

void HTileSemanticToKernelAbiOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure HTileSemanticToKernelAbiOp::applyToOne(TransformRewriter &rewriter,
                                                                   Operation *target,
                                                                   ApplyToEachResultList &results,
                                                                   TransformState &state) {
  (void)results;
  (void)state;
  auto transform = cast<TransformOpInterface>(getOperation());

  auto funcOp = dyn_cast<func::FuncOp>(target);
  if (!funcOp)
    return emitSilenceableFailure(transform, "expected func.func target");

  FailureOr<KernelAbiRewriteInfo> info = rewriteFunctionAbi(rewriter, funcOp);
  if (failed(info))
    return emitSilenceableFailure(transform, "failed to rewrite function ABI");
  if (failed(rewriteOutputStores(rewriter, *info)))
    return emitSilenceableFailure(transform, "failed to rewrite output stores");
  if (failed(rewriteExtractSlicesAsLoads(rewriter, funcOp)))
    return emitSilenceableFailure(transform, "failed to rewrite extract_slice ops as loads");
  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform

namespace htile {

void registerHTileTransformExtension(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, mlir::transform::TransformDialect *dialect) {
    ctx->loadDialect<htile::HTileDialect, mlir::bufferization::BufferizationDialect>();
    struct TransformDialectAccess : public mlir::transform::TransformDialect {
      using mlir::Dialect::addOperations;
    };
    static_cast<TransformDialectAccess *>(dialect)
        ->addOperations<mlir::transform::HTileLinalgToSemanticOp,
                        mlir::transform::HTileSemanticToKernelAbiOp>();
  });
}

} // namespace htile

#define GET_OP_CLASSES
#include "HTileTransformOps.cpp.inc"
