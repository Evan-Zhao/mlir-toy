#include "HTile/HTileTransformOps.h"

#include "LoopTr/Utils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "htile-transform-ops"

using namespace mlir;
using bufferization::ToTensorOp;

#define BAIL(message) return emitSilenceableFailure(transform, message)

namespace mlir::transform {
namespace {

struct FoldRankReducingExtractOfExpandShape : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = extractOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp)
      return failure();
    if (!extractOp.hasUnitStride())
      return failure();
    if (extractOp.getResultType().getRank() >= expandOp.getResultType().getRank() ||
        extractOp.getResultType().getRank() > expandOp.getSrcType().getRank())
      return failure();

    ArrayRef<int64_t> expandedShape = expandOp.getResultType().getShape();
    SmallVector<OpFoldResult> oldOffsets = extractOp.getMixedOffsets(),
                              oldSizes = extractOp.getMixedSizes(),
                              oldStrides = extractOp.getMixedStrides();
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;

    for (auto &group : expandOp.getReassociationIndices()) {
      SmallVector<int64_t> carriedDims;
      for (int64_t expandedDim : group) {
        int64_t dimSize = expandedShape[expandedDim];
        if (dimSize == 1) {
          if (!isZeroInteger(oldOffsets[expandedDim]) || !isOneInteger(oldSizes[expandedDim]) ||
              !isOneInteger(oldStrides[expandedDim]))
            return failure();
          continue;
        }
        carriedDims.push_back(expandedDim);
      }
      if (carriedDims.size() > 1)
        return failure();
      auto carriedDim = carriedDims.empty() ? std::optional<int64_t>() : carriedDims.front();
      newOffsets.push_back(carriedDim ? oldOffsets[*carriedDim] : rewriter.getIndexAttr(0));
      newSizes.push_back(carriedDim ? oldSizes[*carriedDim] : rewriter.getIndexAttr(1));
      newStrides.push_back(carriedDim ? oldStrides[*carriedDim] : rewriter.getIndexAttr(1));
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        extractOp, extractOp.getResultType(), expandOp.getSrc(), newOffsets, newSizes, newStrides);
    return success();
  }
};

struct ForallResultExpand {
  unsigned resultNumber;
  tensor::ExpandShapeOp expandOp;
  tensor::ParallelInsertSliceOp insertOp;
};

LogicalResult foldForallResultExpandShapes(RewriterBase &rewriter, scf::ForallOp forallOp,
                                           ArrayRef<ForallResultExpand> resultExpands) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);
  SmallVector<Value> newOutputs = llvm::to_vector(forallOp.getOutputs());
  for (auto &expand : resultExpands) {
    auto expandOp = expand.expandOp;
    newOutputs[expand.resultNumber] = tensor::ExpandShapeOp::create(
        rewriter, forallOp.getLoc(), expandOp.getResultType(), newOutputs[expand.resultNumber],
        expandOp.getReassociationIndices(), expandOp.getMixedOutputShape());
  }
  auto newForallOp = scf::ForallOp::create(
      rewriter, forallOp.getLoc(), forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), newOutputs, forallOp.getMapping(),
      [&](OpBuilder &, Location, ValueRange bbArgs) {
        SmallVector<Value> replacements = llvm::to_vector(bbArgs);
        rewriter.mergeBlocks(forallOp.getBody(), bbArgs.front().getParentBlock(), replacements);
      });

  for (auto &expand : resultExpands) {
    auto insertOp = expand.insertOp;
    auto expandOp = expand.expandOp;
    RankedTensorType expandedType = expandOp.getResultType();
    SmallVector<ReassociationIndices> reassociation = expandOp.getReassociationIndices();

    OpFoldResult one = rewriter.getIndexAttr(1), zero = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (auto [oldDim, group] : llvm::enumerate(reassociation)) {
      bool seenCarriedDim = false;
      for (int64_t expandedDim : group) {
        bool isCarriedDim = expandedType.getDimSize(expandedDim) != 1;
        if (isCarriedDim && seenCarriedDim)
          return expandOp.emitError() << "expected each reassociation group to have at most one "
                                         "static non-unit dimension";
        else if (isCarriedDim)
          seenCarriedDim = true;
        offsets.push_back(isCarriedDim ? insertOp.getMixedOffsets()[oldDim] : zero);
        sizes.push_back(isCarriedDim ? insertOp.getMixedSizes()[oldDim] : one);
        strides.push_back(isCarriedDim ? insertOp.getMixedStrides()[oldDim] : one);
      }
    }

    rewriter.setInsertionPoint(insertOp);
    tensor::ParallelInsertSliceOp::create(rewriter, insertOp.getLoc(), insertOp.getSource(),
                                          newForallOp.getRegionOutArgs()[expand.resultNumber],
                                          offsets, sizes, strides);
    rewriter.eraseOp(insertOp);
  }

  for (ForallResultExpand expand : resultExpands)
    rewriter.replaceOp(expand.expandOp, newForallOp.getResult(expand.resultNumber));
  SmallVector<Value> replacements;
  llvm::append_range(replacements, newForallOp->getResults());
  rewriter.replaceOp(forallOp, replacements);
  return success();
}

struct FoldForallResultExpandShape : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp forallOp, PatternRewriter &rewriter) const override {
    SmallVector<ForallResultExpand> resultExpands;
    for (OpResult result : forallOp->getResults()) {
      // For each result, we need (1) its unique expand_shape user, and (2)
      // its parallel_insert_slice op inside the forall body.
      // Once we reduce the rank of the result, we can emit an expand_shape to compensate for that
      // for all users. Except the parallel insert, because it's in the in_parallel region
      // of the loop where expand_shape isn't allowed. So parallel inserts needs special handling.
      if (!result.hasOneUse())
        continue;
      auto expandOp = dyn_cast<tensor::ExpandShapeOp>(*result.user_begin());
      if (!expandOp)
        continue;
      auto insert = getParallelInsertSliceForLoopResult(forallOp, result);
      if (failed(insert))
        continue;
      resultExpands.emplace_back(result.getResultNumber(), expandOp, *insert);
    }
    if (resultExpands.empty())
      return failure();
    if (failed(foldForallResultExpandShapes(rewriter, forallOp, resultExpands))) {
      forallOp.emitError() << "failed to fold tensor.expand_shape users into scf.forall results";
      return failure();
    }
    return success();
  }
};

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

LogicalResult applyRewritesGreedily(TransformRewriter &rewriter, Operation *target,
                                    const std::function<void(RewritePatternSet &)> &patternSet) {
  RewritePatternSet patterns(target->getContext());
  patternSet(patterns);
  GreedyRewriteConfig config;
  config.setListener(static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  return applyPatternsGreedily(target, std::move(patterns), config);
}

FailureOr<scf::ForallOp> rebuildForallWithoutOutputs(RewriterBase &rewriter,
                                                     scf::ForallOp forallOp) {
  for (BlockArgument outputArg : forallOp.getRegionOutArgs()) {
    if (!outputArg.use_empty()) {
      outputArg.user_begin()->emitRemark() << "one of the remaining uses here";
      return forallOp.emitError() << "unsupported remaining use of scf.forall shared_out";
    }
  }
  for (OpResult result : forallOp->getResults()) {
    if (!result.use_empty())
      return forallOp.emitError()
             << "expected returned scf.forall result to have no remaining uses";
  }

  SmallVector<Value> oldOutputs = llvm::to_vector(forallOp.getOutputs());
  rewriter.setInsertionPoint(forallOp);
  auto newForall = scf::ForallOp::create(
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
  return newForall;
}

struct TensorToBufferMap {
  void mapTensorToMemref(Value tensor, Value memref) { tensorToMemref[tensor] = memref; }

  Value getTensorMemref(Value tensor) {
    if (auto it = tensorToMemref.find(tensor); it != tensorToMemref.end()) {
      return it->second;
    }
    return nullptr;
  }

  Value getOrCreateTensorMemrefForRead(RewriterBase &rewriter, scf::ForallOp forall, Value tensor) {
    if (Value buffer = getTensorMemref(tensor)) {
      return buffer;
    }

    auto tensorType = cast<RankedTensorType>(tensor.getType());
    auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forall);
    Value buffer = bufferization::ToBufferOp::create(rewriter, tensor.getLoc(), memrefType, tensor,
                                                     /*readOnly=*/true);
    mapTensorToMemref(tensor, buffer);
    return buffer;
  }

  DenseMap<Value, Value> tensorToMemref;
};

LogicalResult materializeStoreForInsertSlice(RewriterBase &rewriter,
                                             tensor::ParallelInsertSliceOp insert, Value buffer) {
  if (!insert.hasUnitStride())
    return insert.emitError() << "unsupported non-unit tensor.parallel_insert_slice stride";
  SmallVector<Value> offsets =
      getValueOrCreateConstantIndexOp(rewriter, insert.getLoc(), insert.getMixedOffsets());
  htile::StoreOp::create(rewriter, insert.getLoc(), insert.getSource(), buffer, offsets);
  return success();
}

FailureOr<Value> materializeLoadForExtractSlice(OpBuilder &builder, tensor::ExtractSliceOp extract,
                                                Value buffer, bool emitHTileLoad) {
  if (emitHTileLoad) {
    if (!extract.hasUnitStride())
      return extract.emitError() << "unsupported non-unit tensor.extract_slice stride";
    SmallVector<Value> offsets =
        getValueOrCreateConstantIndexOp(builder, extract.getLoc(), extract.getMixedOffsets());
    auto loadOp =
        htile::LoadOp::create(builder, extract.getLoc(), extract.getResultType(), buffer, offsets);
    return loadOp.getResult();
  } else {
    auto sourceMemrefType = cast<MemRefType>(buffer.getType());
    auto resultType = cast<RankedTensorType>(extract.getResultType());
    auto subviewType = memref::SubViewOp::inferRankReducedResultType(
        resultType.getShape(), sourceMemrefType, extract.getMixedOffsets(), extract.getMixedSizes(),
        extract.getMixedStrides());
    Value subview = memref::SubViewOp::create(builder, extract.getLoc(), subviewType, buffer,
                                              extract.getMixedOffsets(), extract.getMixedSizes(),
                                              extract.getMixedStrides());
    auto loadOp =
        ToTensorOp::create(builder, extract.getLoc(), resultType, subview, /*restrict=*/true,
                           /*writable=*/true);
    return loadOp.getResult();
  }
}

Value materializeLoadForWholeTensor(OpBuilder &builder, Location loc, RankedTensorType tensorType,
                                    Value buffer, bool emitHTileLoad) {
  if (emitHTileLoad) {
    SmallVector<Value> offsets;
    offsets.reserve(tensorType.getRank());
    for (int64_t i = 0, e = tensorType.getRank(); i < e; ++i)
      offsets.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
    return htile::LoadOp::create(builder, loc, tensorType, buffer, offsets);
  } else {
    return ToTensorOp::create(builder, loc, tensorType, buffer,
                              /*restrict=*/true, /*writable=*/true);
  }
}

LogicalResult bufferizeForallResults(RewriterBase &rewriter, ArrayRef<scf::ForallOp> forallOps,
                                     TensorToBufferMap &map) {
  OpBuilder::InsertionGuard guard(rewriter);
  for (auto forall : forallOps) {
    // Create a memref buffer for each result tensor and map it to the tensor.
    for (OpResult result : forall->getResults()) {
      auto tensorType = dyn_cast<RankedTensorType>(result.getType());
      if (!tensorType)
        continue;
      if (!tensorType.hasStaticShape())
        return forall.emitError() << "result # " << result.getResultNumber()
                                  << " of this forall is a ranked tensor with dynamic shape";
      // Allocate the buffer before the forall loop.
      rewriter.setInsertionPoint(forall);
      auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      auto buffer = memref::AllocOp::create(rewriter, forall.getLoc(), memrefType);
      map.mapTensorToMemref(result, buffer);
      map.mapTensorToMemref(forall.getTiedBlockArgument(result), buffer);
    }

    // Materialize each tensor.parallel_insert_slice op into a memref store.
    // Insert the memref store ops before the forall terminator (i.e. not in in_parallel region).
    rewriter.setInsertionPoint(forall.getTerminator());
    for (Operation &combiningOp : llvm::make_early_inc_range(forall.getTerminator())) {
      auto insert = dyn_cast<tensor::ParallelInsertSliceOp>(&combiningOp);
      if (!insert)
        return combiningOp.emitError() << "expected forall in_parallel region to only have "
                                          "tensor.parallel_insert_slice ops";
      auto dest = insert.getDest();
      auto buffer = map.getTensorMemref(dest);
      if (!buffer)
        return insert.emitError()
               << "this op doesn't publish to a tensor-typed block argument of the loop";
      if (failed(materializeStoreForInsertSlice(rewriter, insert, buffer)))
        return failure();
      rewriter.eraseOp(insert);
    }
  }
  return success();
}

FailureOr<bool> materializeLoadsForTensorUsers(RewriterBase &rewriter, OpOperand &use,
                                               RankedTensorType tensorType, Value buffer,
                                               bool emitHTileLoad) {
  Operation *owner = use.getOwner();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(owner);
  if (auto extract = dyn_cast<tensor::ExtractSliceOp>(owner)) {
    // If the use is a tensor.extract_slice, materialize a sliced read of the buffer.
    auto loaded = materializeLoadForExtractSlice(rewriter, extract, buffer, emitHTileLoad);
    if (failed(loaded))
      return failure();
    rewriter.replaceOp(extract, *loaded);
    return FailureOr<bool>(true); // Op erased
  } else {
    // Otherwise, fall back to a full read of the buffer.
    Value loaded =
        materializeLoadForWholeTensor(rewriter, owner->getLoc(), tensorType, buffer, emitHTileLoad);
    use.set(loaded);
    return FailureOr<bool>(false); // Op not erased
  }
}

LogicalResult bufferizeTensorReadInForalls(RewriterBase &rewriter,
                                           ArrayRef<scf::ForallOp> forallOps,
                                           TensorToBufferMap &map) {
  for (auto forall : forallOps) {
    DenseSet<Value> blockArgs;
    for (auto arg : forall.getBody()->getArguments()) {
      blockArgs.insert(arg);
    }
    WalkResult walkResult = forall.getBody()->walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        // Skip non-tensors, and tensors defined inside the loop body (not block arguments).
        Value tensor = operand.get();
        auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
        if (!tensorType)
          continue;
        bool isBlockArg = blockArgs.count(tensor);
        bool isInLoop = forall.getRegion().isAncestor(tensor.getParentRegion());
        if (isInLoop && !isBlockArg)
          continue;
        auto buffer = map.getOrCreateTensorMemrefForRead(rewriter, forall, tensor);
        // If this op is an extract_slice, it may be entirely removed.
        FailureOr<bool> erased =
            materializeLoadsForTensorUsers(rewriter, operand, tensorType, buffer,
                                           /*emitHTileLoad=*/true);
        if (failed(erased))
          return WalkResult::interrupt();
        if (*erased)
          return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
  }
  return success();
}

LogicalResult bufferizeForallResultUses(RewriterBase &rewriter, ArrayRef<scf::ForallOp> forallOps,
                                        TensorToBufferMap &map) {
  for (auto forall : forallOps) {
    for (OpOperand &use : llvm::make_early_inc_range(forall->getUses())) {
      Value tensor = use.get();
      Value buffer = map.getTensorMemref(tensor);
      if (!buffer)
        continue;
      auto tensorType = cast<RankedTensorType>(tensor.getType());
      if (failed(materializeLoadsForTensorUsers(rewriter, use, tensorType, buffer,
                                                /*emitHTileLoad=*/false)))
        return failure();
    }
  }
  return success();
}

std::string getRequestedKernelName(ArrayAttr kernelNames, size_t index) {
  if (kernelNames)
    return cast<StringAttr>(kernelNames[index]).getValue().str();
  return ("outlined_kernel_" + Twine(index)).str();
}

std::string getUniqueKernelName(Operation *symbolTableOp, StringRef baseName) {
  if (!SymbolTable::lookupSymbolIn(symbolTableOp, baseName))
    return baseName.str();

  unsigned uniquingCounter = 0;
  SmallString<32> name = SymbolTable::generateSymbolName<32>(
      baseName,
      [&](StringRef candidate) {
        return SymbolTable::lookupSymbolIn(symbolTableOp, candidate) != nullptr;
      },
      uniquingCounter);
  return name.str().str();
}

struct OutlinedKernel {
  scf::ForallOp forall;
  htile::KernelOp kernel;
  SmallVector<Value> operands;
  htile::LaunchFuncOp launch;
};

FailureOr<std::pair<SmallVector<Value>, SmallVector<int64_t>>>
getLoopNormalizedIVsAndTripCounts(RewriterBase &rewriter, scf::ForallOp forall) {
  auto lowerBounds = forall.getMixedLowerBound(), upperBounds = forall.getMixedUpperBound(),
       steps = forall.getMixedStep();
  Location loc = forall.getLoc();
  Type indexType = rewriter.getIndexType();

  size_t nDims = lowerBounds.size();
  SmallVector<int64_t> tripCounts;
  tripCounts.reserve(nDims);
  SmallVector<Value> ids;
  ids.reserve(nDims);
  for (size_t index = 0; index < nDims; ++index) {
    std::optional<int64_t> maybeLower = getConstantIntValue(lowerBounds[index]),
                           maybeUpper = getConstantIntValue(upperBounds[index]),
                           maybeStep = getConstantIntValue(steps[index]);
    if (!maybeLower || !maybeUpper || !maybeStep)
      return forall.emitError() << "expected static lower/upper/step for forall dimension "
                                << index;
    if (*maybeStep <= 0)
      return forall.emitError() << "expected positive static step for forall dimension " << index;
    if (*maybeUpper < *maybeLower)
      return forall.emitError() << "expected upper bound to be >= lower bound for dimension "
                                << index;

    Value id = htile::ProgramIdOp::create(rewriter, loc, indexType, index);
    if (*maybeStep != 1) {
      Value stepValue = arith::ConstantIndexOp::create(rewriter, loc, *maybeStep);
      id = arith::MulIOp::create(rewriter, loc, id, stepValue);
    }
    if (*maybeLower != 0) {
      Value lowerValue = arith::ConstantIndexOp::create(rewriter, loc, *maybeLower);
      id = arith::AddIOp::create(rewriter, loc, id, lowerValue);
    }
    ids.push_back(id);

    int64_t distance = *maybeUpper - *maybeLower;
    tripCounts.push_back((distance + *maybeStep - 1) / *maybeStep);
  }

  return std::make_pair(ids, tripCounts);
}

FailureOr<SmallVector<Value>> legalizeKernelExternalValues(RewriterBase &rewriter,
                                                           htile::KernelOp kernel) {
  Region &region = kernel.getBody();
  Block &entryBlock = region.front();

  llvm::SetVector<Value> captures;
  kernel.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!region.isAncestor(operand.getParentRegion()))
        captures.insert(operand);
    }
  });

  SmallVector<Value> operands;
  OpBuilder::InsertionGuard guard(rewriter);
  // Allow captures to be memrefs or arith.constant. If it's a constant, copy it into the kernel.
  for (Value capture : captures) {
    if (isa<MemRefType>(capture.getType())) {
      BlockArgument arg = entryBlock.addArgument(capture.getType(), capture.getLoc());
      rewriter.replaceUsesWithIf(capture, arg, [&](OpOperand &use) {
        return region.isAncestor(use.getOwner()->getParentRegion());
      });
      operands.push_back(capture);
      continue;
    }

    Operation *def = capture.getDefiningOp();
    if (!isa_and_nonnull<arith::ConstantOp>(def))
      return kernel.emitError() << "unsupported non-memref kernel capture: " << capture;
    rewriter.setInsertionPointToStart(&entryBlock);
    Operation *cloned = rewriter.clone(*def);
    rewriter.replaceUsesWithIf(capture, cloned->getResult(0), [&](OpOperand &use) {
      return region.isAncestor(use.getOwner()->getParentRegion());
    });
  }

  return operands;
}

FailureOr<SmallVector<OutlinedKernel>> createKernelOps(RewriterBase &rewriter, Operation *hostOp,
                                                       ArrayRef<scf::ForallOp> forallOps,
                                                       ArrayAttr kernelNames) {
  Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(hostOp);
  if (!symbolTableOp)
    return hostOp->emitError() << "expected selected forall parent to have a symbol table";

  SmallVector<OutlinedKernel> kernels;
  kernels.reserve(forallOps.size());
  Operation *insertAfter = hostOp;
  for (auto [index, forallValue] : llvm::enumerate(forallOps)) {
    scf::ForallOp forall = forallValue;
    std::string requestedName = getRequestedKernelName(kernelNames, index);
    std::string kernelName = getUniqueKernelName(symbolTableOp, requestedName);

    // Create the kernel after the current hostOp (typically a func.func).
    rewriter.setInsertionPointAfter(insertAfter);
    auto kernel = htile::KernelOp::create(rewriter, forall.getLoc(), kernelName);
    Block *body = new Block();
    kernel.getBody().push_back(body);

    // Map the induction variables to the program IDs, then clone the forall body into the kernel.
    rewriter.setInsertionPointToStart(body);
    auto ivsAndTripCounts = getLoopNormalizedIVsAndTripCounts(rewriter, forall);
    if (failed(ivsAndTripCounts))
      return failure();
    auto [programIds, programBounds] = *ivsAndTripCounts;
    kernel.setProgramBoundsAttr(DenseI64ArrayAttr::get(rewriter.getContext(), programBounds));
    IRMapping mapping;
    mapping.map(forall.getInductionVars(), programIds);
    cloneBlockWithoutTerminator(rewriter, *forall.getBody(), mapping);
    htile::ReturnOp::create(rewriter, forall.getLoc());

    // Check what values are used from the body of the kernel, and list them as operands for the
    // kernel. Only allow memrefs in the operands. Constants are copied into the kernel.
    FailureOr<SmallVector<Value>> operands = legalizeKernelExternalValues(rewriter, kernel);
    if (failed(operands))
      return failure();
    kernels.push_back(OutlinedKernel{.forall = forall,
                                     .kernel = kernel,
                                     .operands = std::move(*operands),
                                     .launch = htile::LaunchFuncOp()});
    insertAfter = kernel.getOperation();
  }

  return kernels;
}

FailureOr<func::FuncOp> validateSameParentFunc(ArrayRef<scf::ForallOp> forallOps) {
  func::FuncOp hostFunc;
  for (scf::ForallOp forall : forallOps) {
    auto parentFunc = dyn_cast<func::FuncOp>(forall->getParentOp());
    if (!parentFunc)
      return forall.emitError()
             << "expected selected scf.forall to be a top-level op directly inside func.func";
    if (!hostFunc)
      hostFunc = parentFunc;
    else if (parentFunc != hostFunc)
      return forall.emitError()
             << "expected all selected scf.forall ops to belong to the same func.func";
  }
  return hostFunc;
}

void createLaunchOpsAndEraseForalls(RewriterBase &rewriter,
                                    MutableArrayRef<OutlinedKernel> kernels) {
  for (OutlinedKernel &outlined : kernels) {
    rewriter.setInsertionPoint(outlined.forall);
    outlined.launch = htile::LaunchFuncOp::create(rewriter, outlined.forall.getLoc(),
                                                  outlined.kernel.getSymName(), outlined.operands);
    if (auto programBounds = outlined.kernel.getProgramBoundsAttr())
      outlined.launch->setAttr("program_bounds", programBounds);
    rewriter.eraseOp(outlined.forall);
  }
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
    if (failed(rewriteOriginalLinalgOp(rewriter, op)))
      BAIL("failed to rewrite linalg op: ") << *op;
  }

  if (failed(applyRewritesGreedily(rewriter, target, [&](RewritePatternSet &patterns) {
        patterns.add<FoldRankReducingExtractOfExpandShape>(patterns.getContext());
        patterns.add<FoldForallResultExpandShape>(patterns.getContext());
        tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
        tensor::populateBubbleUpExtractSliceOpPatterns(patterns);
        tensor::populateReassociativeReshapeFoldingPatterns(patterns);
        tensor::populateFoldTensorEmptyPatterns(patterns);
      })))
    BAIL("failed to apply tensor cleanup patterns");

  return DiagnosedSilenceableFailure::success();
}

namespace {

enum class PackedWindowCallKind : uint8_t { Extract, Insert };

static constexpr llvm::StringLiteral packedWindowExtractTarget = "neptune.packed_window_extract";
static constexpr llvm::StringLiteral packedWindowInsertTarget = "neptune.packed_window_insert";

/// The semantic operands and types shared by packed-window extraction and
/// insertion. `packed` is the source of an extraction and the destination of
/// an insertion; `other` is present only for extraction.
struct PackedWindowCallInfo {
  PackedWindowCallKind kind;
  stablehlo::CustomCallOp call;
  Value packed;
  Value windows;
  Value starts;
  Value lengths;
  Value other;
  RankedTensorType packedType;
  RankedTensorType windowsType;
  RankedTensorType startsType;
};

/// Validate the custom-call ABI and expose it independently of operand order.
static FailureOr<PackedWindowCallInfo> validatePackedWindowCall(stablehlo::CustomCallOp call,
                                                                PackedWindowCallKind kind) {
  StringRef target =
      kind == PackedWindowCallKind::Extract ? packedWindowExtractTarget : packedWindowInsertTarget;
  constexpr unsigned expectedInputs = 4;
  if (call.getCallTargetName() != target)
    return call.emitError() << "expected custom call target @" << target;
  if (call.getHasSideEffect())
    return call.emitError("expected packed-window custom call to be side-effect free");
  if (!call.getCalledComputations().empty())
    return call.emitError("expected packed-window custom call to have no called computations");
  if (!call.getOutputOperandAliases().empty())
    return call.emitError("expected packed-window custom call to have no output aliases");
  if (call.getInputs().size() != expectedInputs || call->getNumResults() != 1)
    return call.emitError() << "expected four inputs and one result";

  for (Type type : llvm::concat<Type>(call.getInputs().getTypes(), call->getResultTypes())) {
    auto rankedType = dyn_cast<RankedTensorType>(type);
    if (!rankedType)
      return call.emitError("expected all inputs and results to be ranked tensors");
    if (!rankedType.hasStaticShape())
      return call.emitError("expected all packed-window shapes to be static");
  }

  ValueRange inputs = call.getInputs();
  Value packed = kind == PackedWindowCallKind::Extract ? inputs[0] : inputs[1];
  Value windows = kind == PackedWindowCallKind::Extract ? call->getResult(0) : inputs[0];
  Value starts = kind == PackedWindowCallKind::Extract ? inputs[1] : inputs[2];
  PackedWindowCallInfo info{
      .kind = kind,
      .call = call,
      .packed = packed,
      .windows = windows,
      .starts = starts,
      .lengths = kind == PackedWindowCallKind::Extract ? inputs[2] : inputs[3],
      .other = kind == PackedWindowCallKind::Extract ? inputs[3] : Value{},
      .packedType = cast<RankedTensorType>(packed.getType()),
      .windowsType = cast<RankedTensorType>(windows.getType()),
      .startsType = cast<RankedTensorType>(starts.getType()),
  };
  RankedTensorType lengthsType = cast<RankedTensorType>(info.lengths.getType());

  if (info.startsType.getRank() != 1 || info.startsType != lengthsType)
    return call.emitError("expected starts and lengths to have the same rank-one tensor type");
  Type indexType = info.startsType.getElementType();
  if (!indexType.isIndex() && !indexType.isSignlessInteger())
    return call.emitError("expected starts and lengths to contain indices or signless integers");
  if (info.windowsType.getRank() < 2 ||
      info.windowsType.getDimSize(0) != info.startsType.getDimSize(0))
    return call.emitError("expected one window per start and length");

  if (kind == PackedWindowCallKind::Extract) {
    RankedTensorType otherType = cast<RankedTensorType>(info.other.getType());
    if (info.packedType.getRank() < 1 ||
        info.windowsType.getRank() != info.packedType.getRank() + 1)
      return call.emitError("expected windows rank to be packed rank plus one");
    if (otherType.getRank() != 0 ||
        otherType.getElementType() != info.packedType.getElementType() ||
        info.windowsType.getElementType() != info.packedType.getElementType())
      return call.emitError("expected packed, windows, and scalar other element types to match");
    if (!llvm::equal(info.packedType.getShape().drop_front(),
                     info.windowsType.getShape().drop_front(2)))
      return call.emitError("expected window element shape to match packed element shape");
  } else {
    RankedTensorType resultType = cast<RankedTensorType>(call->getResult(0).getType());
    if (info.windowsType.getRank() != info.packedType.getRank() + 1)
      return call.emitError("expected windows rank to be destination rank plus one");
    if (info.windowsType.getElementType() != info.packedType.getElementType() ||
        resultType != info.packedType)
      return call.emitError(
          "expected windows element type and destination/result tensor types to match");
    if (!llvm::equal(info.windowsType.getShape().drop_front(2),
                     info.packedType.getShape().drop_front()))
      return call.emitError("expected window element shape to match destination element shape");
  }
  return info;
}

/// An analyzed tile in `[document, token, tail...]` window coordinates. This
/// contains only existing values and attributes, so every tile can be checked
/// before either transform starts mutating the payload IR.
struct PackedWindowTilePlan {
  PackedWindowCallInfo call;
  Operation *tileOp;
  RankedTensorType tileType;
  OpFoldResult documentOffset;
  OpFoldResult tokenOffset;
  OpFoldResult tokenSize;
  SmallVector<OpFoldResult> trailingOffsets;
  SmallVector<OpFoldResult> trailingSizes;
  SmallVector<OpFoldResult> trailingStrides;
  llvm::SmallBitVector rankReducedWindowDims;
};

static FailureOr<PackedWindowTilePlan>
analyzePackedWindowTile(const PackedWindowCallInfo &call,
                        OffsetSizeAndStrideOpInterface publishingOp, RankedTensorType tileType) {
  size_t windowRank = call.windowsType.getRank();
  auto offsets = publishingOp.getMixedOffsets();
  auto sizes = publishingOp.getMixedSizes();
  auto strides = publishingOp.getMixedStrides();
  if (offsets.size() != windowRank || sizes.size() != windowRank || strides.size() != windowRank)
    return publishingOp->emitError("expected a full-rank packed-window tile descriptor");
  if (!tileType.hasStaticShape())
    return publishingOp->emitError("expected the packed-window tile shape to be static");
  if (tileType.getElementType() != call.windowsType.getElementType())
    return publishingOp->emitError("expected the tile and windows element types to match");
  if (!llvm::all_of(strides, isOneInteger))
    return publishingOp->emitError("expected packed-window tile strides to be one");

  SmallVector<int64_t> staticSizes;
  staticSizes.reserve(sizes.size());
  for (OpFoldResult size : sizes) {
    std::optional<int64_t> constant = getConstantIntValue(size);
    if (!constant || *constant <= 0)
      return publishingOp->emitError("expected positive static packed-window tile sizes");
    staticSizes.push_back(*constant);
  }
  if (staticSizes[0] != 1)
    return publishingOp->emitError("expected each packed-window tile to select one document");

  std::optional<llvm::SmallDenseSet<unsigned>> droppedDims =
      computeRankReductionMask(staticSizes, tileType.getShape(), /*matchDynamic=*/false);
  if (!droppedDims)
    return publishingOp->emitError(
        "expected the packed-window tile type to be a rank reduction of its "
        "descriptor sizes");
  llvm::SmallBitVector rankReducedWindowDims(windowRank);
  for (unsigned dim : *droppedDims)
    rankReducedWindowDims.set(dim);

  return PackedWindowTilePlan{
      .call = call,
      .tileOp = publishingOp.getOperation(),
      .tileType = tileType,
      .documentOffset = offsets[0],
      .tokenOffset = offsets[1],
      .tokenSize = sizes[1],
      .trailingOffsets = SmallVector<OpFoldResult>(ArrayRef(offsets).drop_front(2)),
      .trailingSizes = SmallVector<OpFoldResult>(ArrayRef(sizes).drop_front(2)),
      .trailingStrides = SmallVector<OpFoldResult>(ArrayRef(strides).drop_front(2)),
      .rankReducedWindowDims = std::move(rankReducedWindowDims),
  };
}

/// The tile-local address and predicate materialized from a validated plan.
/// Its descriptor is in packed coordinates and therefore omits the logical
/// document dimension of the window tensor.
struct MaterializedPackedWindowTile {
  Value documentIndex;
  Value tokenOffset;
  Value start;
  Value length;
  Value rowOffset;
  Value mask;
  RankedTensorType physicalTileType;
  SmallVector<OpFoldResult> packedOffsets;
  SmallVector<OpFoldResult> packedSizes;
  SmallVector<OpFoldResult> packedStrides;
};

static Value castToIndex(RewriterBase &rewriter, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), value);
}

/// Preserve a tensor tile edge so producer fusion can follow it into the loop,
/// then scalarize only the rank-zero tile.
static Value extractRankOneTensorElement(RewriterBase &rewriter, Location loc, Value tensor,
                                         Value index) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto elementTensorType =
      RankedTensorType::get({}, tensorType.getElementType(), tensorType.getEncoding());
  OpFoldResult one = rewriter.getIndexAttr(1);
  auto slice = tensor::ExtractSliceOp::create(
      rewriter, loc, elementTensorType, tensor, ArrayRef<OpFoldResult>{index},
      ArrayRef<OpFoldResult>{one}, ArrayRef<OpFoldResult>{one});
  return tensor::ExtractOp::create(rewriter, loc, slice, ValueRange{});
}

static MaterializedPackedWindowTile materializePackedWindowTile(RewriterBase &rewriter,
                                                                const PackedWindowTilePlan &plan,
                                                                Operation *insertionPoint) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(insertionPoint);
  Location loc = plan.tileOp->getLoc();

  Value documentIndex = getValueOrCreateConstantIndexOp(rewriter, loc, plan.documentOffset);
  Value tokenOffset = getValueOrCreateConstantIndexOp(rewriter, loc, plan.tokenOffset);
  Value start = extractRankOneTensorElement(rewriter, loc, plan.call.starts, documentIndex);
  Value length = extractRankOneTensorElement(rewriter, loc, plan.call.lengths, documentIndex);
  start = castToIndex(rewriter, loc, start);
  length = castToIndex(rewriter, loc, length);
  Value rowOffset = arith::AddIOp::create(rewriter, loc, start, tokenOffset);

  SmallVector<int64_t> physicalShape;
  for (int64_t windowDim = 1; windowDim < plan.call.windowsType.getRank(); ++windowDim) {
    if (plan.rankReducedWindowDims.test(windowDim))
      continue;
    OpFoldResult size = windowDim == 1 ? plan.tokenSize : plan.trailingSizes[windowDim - 2];
    physicalShape.push_back(*getConstantIntValue(size));
  }
  auto physicalTileType = RankedTensorType::get(physicalShape, plan.tileType.getElementType(),
                                                plan.tileType.getEncoding());

  Value emptyMask = tensor::EmptyOp::create(rewriter, loc, physicalShape, rewriter.getI1Type());
  AffineMap identity = rewriter.getMultiDimIdentityMap(physicalShape.size());
  SmallVector<utils::IteratorType> iteratorTypes(physicalShape.size(),
                                                 utils::IteratorType::parallel);
  bool tokenDimensionDropped = plan.rankReducedWindowDims.test(1);
  auto mask = linalg::GenericOp::create(
      rewriter, loc, TypeRange{emptyMask.getType()}, ValueRange{}, ValueRange{emptyMask},
      ArrayRef<AffineMap>{identity}, iteratorTypes,
      [&](OpBuilder &builder, Location bodyLoc, ValueRange) {
        Value localToken;
        if (tokenDimensionDropped)
          localToken = arith::ConstantIndexOp::create(builder, bodyLoc, 0);
        else
          localToken = linalg::IndexOp::create(builder, bodyLoc, 0);
        Value windowToken = arith::AddIOp::create(builder, bodyLoc, tokenOffset, localToken);
        Value valid =
            arith::CmpIOp::create(builder, bodyLoc, arith::CmpIPredicate::slt, windowToken, length);
        linalg::YieldOp::create(builder, bodyLoc, valid);
      });

  OpFoldResult one = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> packedOffsets{rowOffset};
  llvm::append_range(packedOffsets, plan.trailingOffsets);
  SmallVector<OpFoldResult> packedSizes{plan.tokenSize};
  llvm::append_range(packedSizes, plan.trailingSizes);
  SmallVector<OpFoldResult> packedStrides{one};
  llvm::append_range(packedStrides, plan.trailingStrides);
  return MaterializedPackedWindowTile{
      .documentIndex = documentIndex,
      .tokenOffset = tokenOffset,
      .start = start,
      .length = length,
      .rowOffset = rowOffset,
      .mask = mask.getResult(0),
      .physicalTileType = physicalTileType,
      .packedOffsets = std::move(packedOffsets),
      .packedSizes = std::move(packedSizes),
      .packedStrides = std::move(packedStrides),
  };
}

static void printOpFoldResults(llvm::raw_ostream &os, ArrayRef<OpFoldResult> values) {
  llvm::interleaveComma(values, os, [&](OpFoldResult value) {
    if (auto attribute = dyn_cast<Attribute>(value)) {
      os << attribute;
      return;
    }
    cast<Value>(value).printAsOperand(os, OpPrintingFlags());
  });
}

static Value restoreLogicalWindowTile(RewriterBase &rewriter, Location loc, Value physicalTile,
                                      const PackedWindowTilePlan &plan) {
  if (plan.rankReducedWindowDims.test(0)) {
    assert(physicalTile.getType() == plan.tileType &&
           "a tile that dropped the document dimension is already in logical shape");
    return physicalTile;
  }

  auto physicalType = cast<RankedTensorType>(physicalTile.getType());
  assert(plan.tileType.getRank() == physicalType.getRank() + 1 &&
         plan.tileType.getDimSize(0) == 1 &&
         "the physical tile must omit exactly the unit document dimension");
  if (physicalType.getRank() == 0) {
    Value scalar = tensor::ExtractOp::create(rewriter, loc, physicalTile, ValueRange{});
    return tensor::FromElementsOp::create(rewriter, loc, plan.tileType, ValueRange{scalar});
  }

  SmallVector<ReassociationIndices> reassociation;
  reassociation.push_back({0, 1});
  for (int64_t physicalDim = 1; physicalDim < physicalType.getRank(); ++physicalDim)
    reassociation.push_back({physicalDim + 1});
  return tensor::ExpandShapeOp::create(rewriter, loc, plan.tileType, physicalTile, reassociation);
}

static void printMaterializedPackedWindowTile(llvm::raw_ostream &os,
                                              const MaterializedPackedWindowTile &tile) {
  os << "materialized packed-window tile:\n"
     << "  physical tile type: " << tile.physicalTileType << "\n"
     << "  document index: ";
  tile.documentIndex.printAsOperand(os, OpPrintingFlags());
  os << "\n  token offset: ";
  tile.tokenOffset.printAsOperand(os, OpPrintingFlags());
  os << "\n  start: ";
  tile.start.printAsOperand(os, OpPrintingFlags());
  os << "\n  length: ";
  tile.length.printAsOperand(os, OpPrintingFlags());
  os << "\n  row offset: ";
  tile.rowOffset.printAsOperand(os, OpPrintingFlags());
  os << "\n  packed offsets: [";
  printOpFoldResults(os, tile.packedOffsets);
  os << "]\n  packed sizes: [";
  printOpFoldResults(os, tile.packedSizes);
  os << "]\n  packed strides: [";
  printOpFoldResults(os, tile.packedStrides);
  os << "]\n  mask: ";
  tile.mask.printAsOperand(os, OpPrintingFlags());
  os << " : " << tile.mask.getType() << "\n";
}

static Operation *findSelectedLoop(Operation *user, const DenseSet<Operation *> &selectedLoops) {
  for (Operation *parent = user->getParentOp(); parent; parent = parent->getParentOp())
    if (selectedLoops.contains(parent))
      return parent;
  return nullptr;
}

} // namespace

void HTileFusePackedWindowInsertOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getInsertMutable(), effects);
  onlyReadsHandle(getForallMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure HTileFusePackedWindowInsertOp::apply(TransformRewriter &rewriter,
                                                                 TransformResults &results,
                                                                 TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());

  stablehlo::CustomCallOp insert;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getInsert, "insert", insert,
                               stablehlo::CustomCallOp);
  FailureOr<PackedWindowCallInfo> insertInfo =
      validatePackedWindowCall(insert, PackedWindowCallKind::Insert);
  if (failed(insertInfo))
    BAIL("invalid packed-window insertion custom call");

  scf::ForallOp forall;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForall, "forall", forall, scf::ForallOp);
  auto windows = dyn_cast<OpResult>(insertInfo->windows);
  if (!windows || windows.getOwner() != forall)
    BAIL("expected the selected forall to directly produce the windows input");
  FailureOr<tensor::ParallelInsertSliceOp> publication =
      getParallelInsertSliceForLoopResult(forall, windows);
  if (failed(publication))
    BAIL("expected the windows to be published by one tensor.parallel_insert_slice");

  FailureOr<PackedWindowTilePlan> tilePlan = analyzePackedWindowTile(
      *insertInfo, cast<OffsetSizeAndStrideOpInterface>(publication->getOperation()),
      publication->getSourceType());
  if (failed(tilePlan))
    BAIL("invalid packed-window insertion tile");

  // The packed destination and range metadata become operands of the rebuilt
  // forall. Keep availability handling centralized in the shared loop helper.
  rewriter.setInsertionPoint(forall);
  IRMapping availableMapping;
  FailureOr<SmallVector<Value>> availableValues = makeValuesAvailableAtInsertionPoint(
      rewriter, {insertInfo->packed, insertInfo->starts, insertInfo->lengths}, availableMapping,
      DefChainAction::Move);
  if (failed(availableValues))
    BAIL("failed to make the packed destination, starts, and lengths available before the forall");
  tilePlan->call.packed = (*availableValues)[0];
  tilePlan->call.starts = (*availableValues)[1];
  tilePlan->call.lengths = (*availableValues)[2];

  MaterializedPackedWindowTile materializedTile =
      materializePackedWindowTile(rewriter, *tilePlan, forall.getTerminator());
  LLVM_DEBUG(printMaterializedPackedWindowTile(llvm::dbgs(), materializedTile));

  // The publication source is in logical window coordinates. Remove its unit
  // document dimension before publishing to the packed destination.
  rewriter.setInsertionPoint(forall.getTerminator());
  Value physicalSource = publication->getSource();
  if (!tilePlan->rankReducedWindowDims.test(0)) {
    llvm::SmallBitVector dropDocumentDim(publication->getSourceType().getRank());
    dropDocumentDim.set(0);
    physicalSource =
        tensor::dropGivenUnitDims(rewriter, insert.getLoc(), physicalSource, dropDocumentDim);
  }
  assert(physicalSource.getType() == materializedTile.physicalTileType &&
         "tile analysis must determine the physical publication type");

  rewriter.setInsertionPoint(forall);
  ForallOutputExtension extension =
      cloneForallWithAppendedOutputs(rewriter, forall, ValueRange{tilePlan->call.packed});
  scf::ForallOp newForall = extension.forall;
  notifyClonedOpsRecursively(rewriter, extension.clonedOps);

  auto remapMixedValues = [&](ArrayRef<OpFoldResult> values) {
    return llvm::map_to_vector(values, [&](OpFoldResult value) -> OpFoldResult {
      if (auto dynamic = dyn_cast<Value>(value))
        return extension.mapping.lookupOrDefault(dynamic);
      return value;
    });
  };
  Value clonedSource = extension.mapping.lookupOrDefault(physicalSource);
  Value clonedMask = extension.mapping.lookupOrDefault(materializedTile.mask);
  SmallVector<OpFoldResult> clonedOffsets = remapMixedValues(materializedTile.packedOffsets);
  SmallVector<OpFoldResult> clonedSizes = remapMixedValues(materializedTile.packedSizes);
  SmallVector<OpFoldResult> clonedStrides = remapMixedValues(materializedTile.packedStrides);
  pointBuilderToForallParallel(rewriter, newForall);
  auto maskedInsert = htile::MaskedParallelInsertSliceOp::create(
      rewriter, insert.getLoc(), clonedSource, extension.getAppendedOutputArgs().front(),
      clonedOffsets, clonedSizes, clonedStrides, clonedMask);

  rewriter.replaceOp(insert, extension.getAppendedResults().front());
  if (failed(rewriter.notifyPayloadOperationReplaced(forall, newForall)))
    BAIL("failed to preserve the scf.forall handle");
  rewriter.replaceOp(forall, extension.getPreservedResults());

  LLVM_DEBUG(llvm::dbgs() << "rebuilt forall with packed-window insertion:\n"
                          << maskedInsert << "\n");
  results.set(getOperation()->getResult(0), ArrayRef<Operation *>{maskedInsert.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

void HTileFusePackedWindowExtractOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getExtractsMutable(), effects);
  onlyReadsHandle(getLoopsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure HTileFusePackedWindowExtractOp::apply(TransformRewriter &rewriter,
                                                                  TransformResults &results,
                                                                  TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());

  DenseSet<Operation *> loops;
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (!isa<scf::ForOp, scf::ForallOp>(payload))
      BAIL("expected the loops handle to contain only scf.for or scf.forall operations");
    loops.insert(payload);
  }
  if (loops.empty())
    BAIL("expected at least one selected loop");

  SmallVector<PackedWindowCallInfo> extracts;
  for (Operation *payload : state.getPayloadOps(getExtracts())) {
    auto extract = dyn_cast<stablehlo::CustomCallOp>(payload);
    if (!extract)
      BAIL("expected the extracts handle to contain only stablehlo.custom_call operations");
    FailureOr<PackedWindowCallInfo> extractInfo =
        validatePackedWindowCall(extract, PackedWindowCallKind::Extract);
    if (failed(extractInfo))
      BAIL("invalid packed-window extraction custom call");
    extracts.push_back(*extractInfo);
  }
  if (extracts.empty())
    BAIL("expected at least one packed-window extraction custom call");

  SmallVector<PackedWindowTilePlan, 0> tilePlans;
  for (const PackedWindowCallInfo &extract : extracts) {
    bool foundInLoopTile = false;
    for (OpOperand &use : extract.windows.getUses()) {
      if (!findSelectedLoop(use.getOwner(), loops))
        continue;
      auto slice = dyn_cast<tensor::ExtractSliceOp>(use.getOwner());
      if (!slice || use.getOperandNumber() != 0)
        BAIL("expected each in-loop packed-window extraction user to be tensor.extract_slice");
      FailureOr<PackedWindowTilePlan> tilePlan = analyzePackedWindowTile(
          extract, cast<OffsetSizeAndStrideOpInterface>(slice.getOperation()),
          slice.getResultType());
      if (failed(tilePlan))
        BAIL("invalid packed-window extraction tile");
      tilePlans.push_back(std::move(*tilePlan));
      foundInLoopTile = true;
    }
    if (!foundInLoopTile)
      BAIL("expected each packed-window extraction to have an in-loop tile");
  }

  SmallVector<MaterializedPackedWindowTile, 0> materializedTiles;
  materializedTiles.reserve(tilePlans.size());
  for (const PackedWindowTilePlan &plan : tilePlans) {
    materializedTiles.push_back(materializePackedWindowTile(rewriter, plan, plan.tileOp));
    LLVM_DEBUG(printMaterializedPackedWindowTile(llvm::dbgs(), materializedTiles.back()));
  }

  SmallVector<Operation *> loads;
  for (auto [plan, materialized] : llvm::zip_equal(tilePlans, materializedTiles)) {
    auto slice = cast<tensor::ExtractSliceOp>(plan.tileOp);
    rewriter.setInsertionPoint(slice);
    Location loc = slice.getLoc();
    SmallVector<Value> loadOffsets =
        getValueOrCreateConstantIndexOp(rewriter, loc, materialized.packedOffsets);
    Value other = tensor::ExtractOp::create(rewriter, loc, plan.call.other, ValueRange{});
    auto load = htile::LoadOp::create(rewriter, loc, materialized.physicalTileType,
                                      plan.call.packed, loadOffsets, materialized.mask, other);
    Value replacement = restoreLogicalWindowTile(rewriter, loc, load, plan);
    rewriter.replaceOp(slice, replacement);
    loads.push_back(load);
  }

  for (const PackedWindowCallInfo &extract : extracts)
    if (extract.call->use_empty())
      rewriter.eraseOp(extract.call);

  results.set(getOperation()->getResult(0), loads);
  return DiagnosedSilenceableFailure::success();
}

void HTileOutlineKernelsOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getForallsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure HTileOutlineKernelsOp::apply(TransformRewriter &rewriter,
                                                         TransformResults &results,
                                                         TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());

  SmallVector<scf::ForallOp> forallOps;
  for (Operation *op : state.getPayloadOps(getForalls())) {
    auto forall = dyn_cast<scf::ForallOp>(op);
    if (!forall) {
      op->emitError() << "expected scf.forall payload op";
      BAIL("expected all payload ops to be scf.forall");
    }
    forallOps.push_back(forall);
  }
  if (forallOps.empty())
    BAIL("expected at least one scf.forall payload op");
  FailureOr<func::FuncOp> hostFunc = validateSameParentFunc(forallOps);
  if (failed(hostFunc))
    BAIL("failed to validate selected scf.forall ops");
  Operation *hostOp = hostFunc->getOperation();

  auto kernelNames = (*this)->getAttrOfType<ArrayAttr>("kernel_names");
  if (kernelNames && kernelNames.size() != forallOps.size())
    BAIL("expected kernel_names length to match payload op count");

  TensorToBufferMap bufferMap;
  // Convert forall tensor result to memrefs, and in-loop writes of results to memref writes.
  if (failed(bufferizeForallResults(rewriter, forallOps, bufferMap)))
    BAIL("failed to bufferize forall results");
  // Convert reads of any tensor in foralls to memref reads: get a memref for the tensor being used,
  // and read from the memref instead.
  if (failed(bufferizeTensorReadInForalls(rewriter, forallOps, bufferMap)))
    BAIL("failed to bufferize tensor reads in foralls");
  // Convert any remaining uses of forall result tensors to memref reads, such as func.func return.
  if (failed(bufferizeForallResultUses(rewriter, forallOps, bufferMap)))
    BAIL("failed to bufferize forall result uses");
  // Remove all results and shared out arguments from every forall op.
  for (auto &forall : forallOps) {
    auto newForall = rebuildForallWithoutOutputs(rewriter, forall);
    if (failed(newForall))
      BAIL("failed to rebuild forall without outputs");
    forall = *newForall;
  }
  // Create an htile.kernel op for each forall op, with the boundary memrefs as operands.
  FailureOr<SmallVector<OutlinedKernel>> kernels =
      createKernelOps(rewriter, hostOp, forallOps, kernelNames);
  if (failed(kernels))
    BAIL("failed to create htile.kernel ops");
  createLaunchOpsAndEraseForalls(rewriter, *kernels);

  SmallVector<Operation *> launchOps =
      llvm::map_to_vector(*kernels, [](auto &kernel) { return kernel.launch.getOperation(); });
  SmallVector<Operation *> kernelOps =
      llvm::map_to_vector(*kernels, [](auto &kernel) { return kernel.kernel.getOperation(); });
  results.set(getOperation()->getResult(0), launchOps);
  results.set(getOperation()->getResult(1), kernelOps);

  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform

#define GET_OP_CLASSES
#include "HTileTransformOps.cpp.inc"
