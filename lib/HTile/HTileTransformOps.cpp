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
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
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

struct InLoopGatherUse {
  OpOperand *use;
  Operation *loop;
};

/// Operations prepared to replace one in-loop gather result tile with a
/// rectangular load.
struct PreparedRangedGatherUse {
  stablehlo::PadOp pad;
  tensor::ExtractSliceOp slice;
  htile::LoadOp load;
  Value replacement;
};

static Operation *findInnermostSelectedLoop(Operation *user,
                                            const DenseSet<Operation *> &selectedLoops) {
  for (Operation *parent = user->getParentOp(); parent; parent = parent->getParentOp())
    if (selectedLoops.contains(parent))
      return parent;
  return nullptr;
}

static SmallVector<InLoopGatherUse>
collectInLoopGatherUses(stablehlo::GatherOp gather, const DenseSet<Operation *> &selectedLoops) {
  SmallVector<InLoopGatherUse> uses;
  for (OpOperand &use : gather.getResult().getUses())
    if (Operation *loop = findInnermostSelectedLoop(use.getOwner(), selectedLoops))
      uses.push_back(InLoopGatherUse{&use, loop});
  return uses;
}

/// Check only source bounds represented by positive edge padding. For example,
/// high-only padding on `d0` produces `%offset[0] + i0 < dim(source, 0)`.
static Value buildRangedGatherLoadMask(RewriterBase &rewriter, Location loc, stablehlo::PadOp pad,
                                       ArrayRef<Value> offsets, ArrayRef<OpFoldResult> sizes) {
  Value source = pad.getOperand();
  auto sourceType = cast<RankedTensorType>(source.getType());
  int64_t rank = sourceType.getRank();
  ArrayRef<int64_t> lowPadding = pad.getEdgePaddingLow(), highPadding = pad.getEdgePaddingHigh();

  // Upper bounds are needed only on dimensions where high padding created
  // addresses beyond the source extent.
  SmallVector<Value> sourceDims(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (highPadding[dim] <= 0)
      continue;
    if (sourceType.isDynamicDim(dim))
      sourceDims[dim] = tensor::DimOp::create(rewriter, loc, source, dim);
    else
      sourceDims[dim] = arith::ConstantIndexOp::create(rewriter, loc, sourceType.getDimSize(dim));
  }

  Value empty = tensor::EmptyOp::create(rewriter, loc, sizes, rewriter.getI1Type());
  AffineMap identity = rewriter.getMultiDimIdentityMap(rank);
  SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
  auto mask = linalg::GenericOp::create(
      rewriter, loc, TypeRange{empty.getType()}, ValueRange{}, ValueRange{empty},
      ArrayRef<AffineMap>{identity}, iteratorTypes,
      [&](OpBuilder &builder, Location bodyLoc, ValueRange) {
        Value inBounds;
        auto appendCondition = [&](arith::CmpIPredicate predicate, Value lhs, Value rhs) {
          Value cond = arith::CmpIOp::create(builder, bodyLoc, predicate, lhs, rhs);
          inBounds = inBounds ? arith::AndIOp::create(builder, bodyLoc, inBounds, cond) : cond;
        };
        for (int64_t dim = 0; dim < rank; ++dim) {
          bool checkLower = lowPadding[dim] > 0, checkUpper = highPadding[dim] > 0;
          if (!checkLower && !checkUpper)
            continue;

          Value localIndex = linalg::IndexOp::create(builder, bodyLoc, dim);
          Value sourceIndex = arith::AddIOp::create(builder, bodyLoc, offsets[dim], localIndex);
          if (checkLower) {
            Value zero = arith::ConstantIndexOp::create(builder, bodyLoc, 0);
            appendCondition(arith::CmpIPredicate::sge, sourceIndex, zero);
          }
          if (checkUpper)
            appendCondition(arith::CmpIPredicate::slt, sourceIndex, sourceDims[dim]);
        }
        if (!inBounds)
          inBounds = arith::ConstantOp::create(builder, bodyLoc, builder.getBoolAttr(true));
        linalg::YieldOp::create(builder, bodyLoc, inBounds);
      });
  return mask.getResult(0);
}

static RankedTensorType getTensorTypeFromSizes(ArrayRef<OpFoldResult> sizes, Type elementType,
                                               Attribute encoding = {}) {
  SmallVector<int64_t> shape;
  shape.reserve(sizes.size());
  for (OpFoldResult size : sizes)
    shape.push_back(getConstantIntValue(size).value_or(ShapedType::kDynamic));
  return RankedTensorType::get(shape, elementType, encoding);
}

/// Convert the operand-order load tile to the gather-result order without a
/// permutation: drop collapsed operand dimensions, insert selected batch
/// dimensions, and finally mirror any rank reduction performed by the slice.
static Value reshapeRangedGatherLoad(RewriterBase &rewriter, Location loc, Value load,
                                     ArrayRef<int64_t> operandResultDims,
                                     RankedTensorType gatherResultType,
                                     ArrayRef<OpFoldResult> resultSizes,
                                     RankedTensorType sliceResultType) {
  Value result = load;
  auto loadType = cast<RankedTensorType>(load.getType());

  SmallVector<ReassociationIndices> collapseReassociation;
  SmallVector<int64_t> leadingCollapsedDims;
  SmallVector<int64_t> representedResultDims;
  for (int64_t operandDim = 0; operandDim < loadType.getRank(); ++operandDim) {
    int64_t resultDim = operandResultDims[operandDim];
    if (resultDim < 0) {
      if (collapseReassociation.empty())
        leadingCollapsedDims.push_back(operandDim);
      else
        collapseReassociation.back().push_back(operandDim);
      continue;
    }
    collapseReassociation.emplace_back(leadingCollapsedDims.begin(), leadingCollapsedDims.end());
    leadingCollapsedDims.clear();
    collapseReassociation.back().push_back(operandDim);
    representedResultDims.push_back(resultDim);
  }
  if (!leadingCollapsedDims.empty() && !collapseReassociation.empty())
    collapseReassociation.back().append(leadingCollapsedDims);

  if (representedResultDims.size() != static_cast<size_t>(loadType.getRank())) {
    auto collapsedType =
        tensor::CollapseShapeOp::inferCollapsedType(loadType, collapseReassociation);
    result = tensor::CollapseShapeOp::create(rewriter, loc, collapsedType, result,
                                             collapseReassociation);
  }

  auto fullResultType = getTensorTypeFromSizes(resultSizes, loadType.getElementType(),
                                               gatherResultType.getEncoding());
  if (cast<RankedTensorType>(result.getType()).getRank() != fullResultType.getRank()) {
    SmallVector<ReassociationIndices> expandReassociation;
    SmallVector<int64_t> leadingInsertedDims;
    size_t nextRepresented = 0;
    for (int64_t resultDim = 0; resultDim < fullResultType.getRank(); ++resultDim) {
      bool represented = nextRepresented < representedResultDims.size() &&
                         representedResultDims[nextRepresented] == resultDim;
      if (!represented) {
        if (expandReassociation.empty())
          leadingInsertedDims.push_back(resultDim);
        else
          expandReassociation.back().push_back(resultDim);
        continue;
      }
      expandReassociation.emplace_back(leadingInsertedDims.begin(), leadingInsertedDims.end());
      leadingInsertedDims.clear();
      expandReassociation.back().push_back(resultDim);
      ++nextRepresented;
    }
    if (!leadingInsertedDims.empty() && !expandReassociation.empty())
      expandReassociation.back().append(leadingInsertedDims);
    result = tensor::ExpandShapeOp::create(rewriter, loc, fullResultType, result,
                                           expandReassociation, resultSizes);
  }

  if (result.getType() != sliceResultType) {
    SmallVector<OpFoldResult> zeros(fullResultType.getRank(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> ones(fullResultType.getRank(), rewriter.getIndexAttr(1));
    result = tensor::ExtractSliceOp::create(rewriter, loc, sliceResultType, result, zeros,
                                            resultSizes, ones);
  }
  return result;
}

static FailureOr<PreparedRangedGatherUse> prepareRangedGatherUse(RewriterBase &rewriter,
                                                                 stablehlo::GatherOp gather,
                                                                 stablehlo::PadOp pad,
                                                                 const InLoopGatherUse &inLoopUse) {
#define PLAN_FAIL(message) return inLoopUse.use->getOwner()->emitError() << (message);

  auto slice = dyn_cast<tensor::ExtractSliceOp>(inLoopUse.use->getOwner());
  if (!slice || inLoopUse.use->getOperandNumber() != 0)
    PLAN_FAIL("expected each in-loop gather user to be tensor.extract_slice");
  if (!slice.hasUnitStride())
    PLAN_FAIL("expected in-loop gather slices to have unit strides");

  auto indicesType = gather.getStartIndices().getType();
  int64_t resultRank = gather.getType().getRank();
  int64_t operandRank = gather.getOperand().getType().getRank();
  int64_t indicesRank = indicesType.getRank();
  auto dimNumbers = gather.getDimensionNumbers();
  int64_t indexVectorDim = dimNumbers.getIndexVectorDim();

  DenseSet<int64_t> offsetDims(llvm::from_range, dimNumbers.getOffsetDims());
  auto resultOffsets = slice.getMixedOffsets(), resultSizes = slice.getMixedSizes();
  SmallVector<int64_t> resultBatchDims, startIndicesBatchDims;
  for (int64_t dim = 0; dim < resultRank; ++dim)
    if (!offsetDims.contains(dim))
      resultBatchDims.push_back(dim);
  for (int64_t dim = 0; dim < indicesRank; ++dim)
    if (dim != indexVectorDim)
      startIndicesBatchDims.push_back(dim);
  for (int64_t resultDim : resultBatchDims)
    if (!isOneInteger(resultSizes[resultDim]))
      PLAN_FAIL("expected each gather tile to select exactly one start-index vector");

  SmallVector<int64_t> operandWindowResultDims(operandRank, -1);
  SmallVector<int64_t> operandBatchResultDims(operandRank, -1);
  SmallVector<int64_t> operandStartComponents(operandRank, -1);
  SmallVector<OpFoldResult> loadSizes(operandRank, rewriter.getIndexAttr(1));

  auto insertToSet = [](DenseSet<int64_t> &set, ArrayRef<int64_t> dims) {
    set.insert(dims.begin(), dims.end());
  };
  DenseSet<int64_t> collapsedAndBatchDims;
  insertToSet(collapsedAndBatchDims, dimNumbers.getCollapsedSliceDims());
  insertToSet(collapsedAndBatchDims, dimNumbers.getOperandBatchingDims());
  SmallVector<int64_t> windowOperandDims;
  for (int64_t operandDim = 0, offsetDim = 0; operandDim < operandRank; ++operandDim)
    if (!collapsedAndBatchDims.contains(operandDim)) {
      windowOperandDims.push_back(operandDim);
      auto resultDim = dimNumbers.getOffsetDims()[offsetDim++];
      operandWindowResultDims[operandDim] = resultDim;
      loadSizes[operandDim] = resultSizes[resultDim];
    }

  for (auto [operandDim, indicesDim] : llvm::zip_equal(dimNumbers.getOperandBatchingDims(),
                                                       dimNumbers.getStartIndicesBatchingDims())) {
    int64_t batchPosition = indicesDim - static_cast<int64_t>(indexVectorDim < indicesDim);
    operandBatchResultDims[operandDim] = resultBatchDims[batchPosition];
  }

  for (auto [component, operandDim] : llvm::enumerate(dimNumbers.getStartIndexMap()))
    operandStartComponents[operandDim] = static_cast<int64_t>(component);

  // Collapse/expand can insert or remove unit dimensions, but cannot reorder
  // source dimensions. Reject gathers such as operand d0 -> result d1 and
  // operand d1 -> result d0 before materializing any IR.
  SmallVector<int64_t> operandResultDims(operandRank, -1);
  int64_t previousResultDim = -1;
  for (int64_t operandDim = 0; operandDim < operandRank; ++operandDim) {
    int64_t resultDim = operandWindowResultDims[operandDim] >= 0
                            ? operandWindowResultDims[operandDim]
                            : operandBatchResultDims[operandDim];
    operandResultDims[operandDim] = resultDim;
    if (resultDim < 0)
      continue;
    if (resultDim <= previousResultDim)
      PLAN_FAIL("expected gather result layout not to permute operand dimensions");
    previousResultDim = resultDim;
  }

  // Enforce the anti-clamping padding convention. Nonnegative starts are a
  // semantic precondition; sufficient high padding makes upper clamping
  // unobservable once the padded operand is replaced by a masked source load.
  if (!llvm::all_of(pad.getInteriorPadding(), [](int64_t padding) { return padding == 0; }))
    PLAN_FAIL("expected ranged gather padding to have zero interior padding");
  for (int64_t operandDim : dimNumbers.getStartIndexMap()) {
    if (pad.getEdgePaddingLow()[operandDim] != 0)
      PLAN_FAIL("expected zero low padding on each dynamically indexed operand dimension");
    if (pad.getEdgePaddingHigh()[operandDim] < gather.getSliceSizes()[operandDim])
      PLAN_FAIL("expected high padding to cover the full gather slice on each dynamically indexed "
                "operand dimension");
  }

  // All validation is complete. Materialize the final operand-order load and
  // the value that will replace the gather slice.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(slice);
  Location loc = slice.getLoc();
  SmallVector<Value> materializedResultOffsets =
      getValueOrCreateConstantIndexOp(rewriter, loc, resultOffsets);

  SmallVector<Value> indicesCoord(indicesRank);
  for (auto [indicesDim, resultDim] : llvm::zip_equal(startIndicesBatchDims, resultBatchDims))
    indicesCoord[indicesDim] = materializedResultOffsets[resultDim];

  SmallVector<Value> startComponents(dimNumbers.getStartIndexMap().size());
  for (int64_t component = 0; component < static_cast<int64_t>(startComponents.size());
       ++component) {
    SmallVector<Value> componentCoord = indicesCoord;
    if (indexVectorDim < indicesRank)
      componentCoord[indexVectorDim] = arith::ConstantIndexOp::create(rewriter, loc, component);
    Value start =
        tensor::ExtractOp::create(rewriter, loc, gather.getStartIndices(), componentCoord);
    if (!start.getType().isIndex())
      start = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), start);
    startComponents[component] = start;
  }

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> loadOffsets;
  loadOffsets.reserve(operandRank);
  for (int64_t operandDim = 0; operandDim < operandRank; ++operandDim) {
    Value paddedOffset = zero;
    bool hasBaseOffset = false;
    int64_t batchResultDim = operandBatchResultDims[operandDim];
    int64_t startComponent = operandStartComponents[operandDim];
    if (batchResultDim >= 0) {
      paddedOffset = materializedResultOffsets[batchResultDim];
      hasBaseOffset = true;
    } else if (startComponent >= 0) {
      // Ranged gathers promise nonnegative starts and enough high padding for
      // a complete window. Upper-clamped and raw out-of-source starts both
      // select only padding, so the later source mask makes them equivalent.
      paddedOffset = startComponents[startComponent];
      hasBaseOffset = true;
    }

    int64_t windowResultDim = operandWindowResultDims[operandDim];
    if (windowResultDim >= 0) {
      if (!hasBaseOffset)
        paddedOffset = materializedResultOffsets[windowResultDim];
      else if (!isZeroInteger(resultOffsets[windowResultDim]))
        paddedOffset = arith::AddIOp::create(rewriter, loc, paddedOffset,
                                             materializedResultOffsets[windowResultDim]);
    }

    int64_t lowPadding = pad.getEdgePaddingLow()[operandDim];
    if (lowPadding != 0) {
      Value low = arith::ConstantIndexOp::create(rewriter, loc, lowPadding);
      paddedOffset = arith::SubIOp::create(rewriter, loc, paddedOffset, low);
    }
    loadOffsets.push_back(paddedOffset);
  }

  Value mask = buildRangedGatherLoadMask(rewriter, loc, pad, loadOffsets, loadSizes);
  Value other = tensor::ExtractOp::create(rewriter, loc, pad.getPaddingValue(), ValueRange{});
  auto sourceType = pad.getOperand().getType();
  auto maskType = cast<RankedTensorType>(mask.getType());
  auto loadType = RankedTensorType::get(maskType.getShape(), sourceType.getElementType(),
                                        sourceType.getEncoding());
  auto load =
      htile::LoadOp::create(rewriter, loc, loadType, pad.getOperand(), loadOffsets, mask, other);
  Value replacement = reshapeRangedGatherLoad(rewriter, loc, load, operandResultDims,
                                              gather.getType(), resultSizes, slice.getResultType());
  return PreparedRangedGatherUse{
      .pad = pad, .slice = slice, .load = load, .replacement = replacement};
#undef PLAN_FAIL
}

void HTileFuseRangedGatherIntoLoopsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getGathersMutable(), effects);
  onlyReadsHandle(getLoopsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure HTileFuseRangedGatherIntoLoopsOp::apply(TransformRewriter &rewriter,
                                                                     TransformResults &results,
                                                                     TransformState &state) {
  (void)results;
  auto transform = cast<TransformOpInterface>(getOperation());

  DenseSet<Operation *> selectedLoops;
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (!isa<scf::ForOp, scf::ForallOp>(payload))
      BAIL("expected the loops handle to contain only scf.for or scf.forall operations");
    selectedLoops.insert(payload);
  }
  if (selectedLoops.empty())
    BAIL("expected at least one selected loop");

  SmallVector<stablehlo::GatherOp> gathers;
  for (Operation *payload : state.getPayloadOps(getGathers())) {
    auto gather = dyn_cast<stablehlo::GatherOp>(payload);
    if (!gather)
      BAIL("expected the gathers handle to contain only stablehlo.gather operations");
    gathers.push_back(gather);
  }
  if (gathers.empty())
    BAIL("expected at least one stablehlo.gather");

  SmallVector<Operation *> loads;
  DenseSet<Operation *> pads;
  // Check that each gather describes rectangular windows selected by one index vector.
  // The vector may dynamically position several operand dimensions, and start_indices may have
  // arbitrary batch dimensions; each tiled use must select one such vector rather than a tensor
  // of independently positioned windows.
  for (auto gather : gathers) {
    auto indicesType = gather.getStartIndices().getType();
    auto dimNumbers = gather.getDimensionNumbers();
    size_t numStartComponents = dimNumbers.getStartIndexMap().size();
    if (numStartComponents == 0)
      BAIL("expected ranged gather to have at least one start-index component");

    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();
    if (indexVectorDim == indicesType.getRank()) {
      if (numStartComponents != 1)
        BAIL("expected an implicit index-vector dimension to have exactly one start component");
    } else {
      int64_t indexVectorSize = indicesType.getDimSize(indexVectorDim);
      if (!ShapedType::isDynamic(indexVectorSize) &&
          static_cast<size_t>(indexVectorSize) != numStartComponents)
        BAIL("expected indices.size(dim=index_vector_dim) to match len(start_index_map)");
    }

    Type indexElementType = indicesType.getElementType();
    if (!indexElementType.isIndex() && !indexElementType.isSignlessInteger())
      BAIL("expected ranged gather start indices to have signless integer or index elements");

    auto producingPad = gather.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!producingPad)
      BAIL("expected ranged gather operand to be produced by stablehlo.pad");

    SmallVector<InLoopGatherUse> inLoopUses = collectInLoopGatherUses(gather, selectedLoops);
    LLVM_DEBUG({
      llvm::dbgs() << "ranged gather has " << inLoopUses.size() << " in-loop use(s):\n";
      llvm::dbgs() << gather << "\n";
    });
    for (const InLoopGatherUse &inLoopUse : inLoopUses) {
      FailureOr<PreparedRangedGatherUse> prepared =
          prepareRangedGatherUse(rewriter, gather, producingPad, inLoopUse);
      if (failed(prepared))
        BAIL("failed to prepare a replacement for the in-loop ranged gather use");
      LLVM_DEBUG({
        llvm::dbgs() << "  prepared operand #" << inLoopUse.use->getOperandNumber() << " of ";
        inLoopUse.use->getOwner()->print(llvm::dbgs());
        llvm::dbgs() << "\n    inside " << inLoopUse.loop->getName() << "\n";
      });
      loads.push_back(prepared->load);
      pads.insert(prepared->pad);
      rewriter.replaceOp(prepared->slice, prepared->replacement);
    }
  }

  // A gather can retain uses outside the selected loops. Remove only gathers
  // and pads made dead by replacing all of their planned in-loop slices.
  for (auto *op : llvm::concat<Operation *>(gathers, pads)) {
    if (isOpTriviallyDead(op))
      rewriter.eraseOp(op);
  }
  results.set(getOperation()->getResult(0), loads);
  return DiagnosedSilenceableFailure::success();
}

static LogicalResult verifyCloneableDefChainBefore(Value value, Operation *before,
                                                   DominanceInfo &dominance,
                                                   DenseSet<Operation *> &visited) {
  if (dominance.dominates(value, before))
    return success();

  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getBlock() != before->getBlock())
    return failure();
  if (!visited.insert(definition).second)
    return success();
  if (!isMemoryEffectFree(definition))
    return failure();
  return success(llvm::all_of(definition->getOperands(), [&](Value operand) {
    return succeeded(verifyCloneableDefChainBefore(operand, before, dominance, visited));
  }));
}

struct ParallelScatterIndexing {
  SmallVector<Value> indices;
  SmallVector<int64_t> broadcastDims;
};

/// Convert a tiled StableHLO scatter publication to HTile's mixed advanced
/// indexing form. Scatter start components become sliced tensor indices;
/// update-window dimensions become scalar base offsets and `broadcast_dims`.
static FailureOr<ParallelScatterIndexing>
buildParallelScatterIndexing(RewriterBase &rewriter, stablehlo::ScatterOp scatter,
                             tensor::ParallelInsertSliceOp publication) {
  Location loc = scatter.getLoc();
  RankedTensorType sourceType = publication.getSource().getType(),
                   updatesType = publication.getDest().getType(),
                   inputType = cast<RankedTensorType>(scatter.getInputs().front().getType()),
                   scatterIndicesType = scatter.getScatterIndices().getType();
  int64_t sourceRank = sourceType.getRank(), updateRank = updatesType.getRank(),
          inputRank = inputType.getRank();
  if (inputRank == 0)
    return failure();

  auto dimNums = scatter.getScatterDimensionNumbers();
  ArrayRef<int64_t> updateWindowDims(dimNums.getUpdateWindowDims()),
      insertedWindowDims(dimNums.getInsertedWindowDims()),
      inputBatchingDims(dimNums.getInputBatchingDims()),
      scatterDimsToOperandDims(dimNums.getScatterDimsToOperandDims());
  int64_t indexVectorDim = dimNums.getIndexVectorDim();

  // HTile's compact indexing form currently assumes that batching has already
  // been expanded away and that publication strides are unit.
  if (!inputBatchingDims.empty() || !dimNums.getScatterIndicesBatchingDims().empty() ||
      !llvm::all_of(publication.getMixedStrides(),
                    [](OpFoldResult stride) { return isOneInteger(stride); }))
    return failure();

  SmallVector<int64_t> updateScatterDims;
  for (int64_t dim = 0; dim < updateRank; ++dim)
    if (!llvm::is_contained(updateWindowDims, dim))
      updateScatterDims.push_back(dim);

  SmallVector<int64_t> windowOperandDims;
  for (int64_t dim = 0; dim < inputRank; ++dim)
    if (!llvm::is_contained(insertedWindowDims, dim) && !llvm::is_contained(inputBatchingDims, dim))
      windowOperandDims.push_back(dim);
  if (windowOperandDims.size() != updateWindowDims.size())
    return failure();

  llvm::SmallBitVector droppedUpdateDims = publication.getDroppedDims();
  SmallVector<OpFoldResult> pubOffsets = publication.getMixedOffsets(),
                            pubSizes = publication.getMixedSizes(),
                            pubStrides = publication.getMixedStrides();
  SmallVector<int64_t> updateDimToSourceDim(updateRank, -1);
  for (int64_t updateDim = 0, sourceDim = 0; updateDim < updateRank; ++updateDim)
    if (!droppedUpdateDims.test(updateDim))
      updateDimToSourceDim[updateDim] = sourceDim++;

  int64_t batchRank = 0;
  SmallVector<int64_t> componentShape;
  for (int64_t updateDim : updateScatterDims) {
    int64_t sourceDim = updateDimToSourceDim[updateDim];
    if (sourceDim < 0)
      continue;
    if (sourceDim != batchRank++)
      return failure();
    componentShape.push_back(sourceType.getDimSize(sourceDim));
  }

  ParallelScatterIndexing result;
  result.indices.resize(inputRank);
  SmallVector<bool> assigned(inputRank, false);
  bool hasExplicitIndexVectorDim = indexVectorDim < scatterIndicesType.getRank();
  for (auto [component, operandDim] : llvm::enumerate(scatterDimsToOperandDims)) {
    if (assigned[operandDim])
      return failure();
    SmallVector<OpFoldResult> offsets, sizes, strides;
    int64_t updateScatterPos = 0;
    for (int64_t indicesDim = 0; indicesDim < scatterIndicesType.getRank(); ++indicesDim) {
      if (hasExplicitIndexVectorDim && indicesDim == indexVectorDim) {
        offsets.push_back(rewriter.getIndexAttr(static_cast<int64_t>(component)));
        sizes.push_back(rewriter.getIndexAttr(1));
        strides.push_back(rewriter.getIndexAttr(1));
        continue;
      }
      int64_t updateDim = updateScatterDims[updateScatterPos++];
      offsets.push_back(pubOffsets[updateDim]);
      sizes.push_back(pubSizes[updateDim]);
      strides.push_back(pubStrides[updateDim]);
    }
    auto componentType = RankedTensorType::get(componentShape, scatterIndicesType.getElementType());
    result.indices[operandDim] = tensor::ExtractSliceOp::create(
        rewriter, loc, componentType, scatter.getScatterIndices(), offsets, sizes, strides);
    assigned[operandDim] = true;
  }

  int64_t windowSourceDim = batchRank;
  for (auto [updateDim, inputDim] : llvm::zip_equal(updateWindowDims, windowOperandDims)) {
    if (assigned[inputDim])
      return failure();
    result.indices[inputDim] =
        getValueOrCreateConstantIndexOp(rewriter, loc, pubOffsets[updateDim]);
    assigned[inputDim] = true;
    if (droppedUpdateDims.test(updateDim))
      continue;
    if (updateDimToSourceDim[updateDim] != windowSourceDim++)
      return failure();
    result.broadcastDims.push_back(inputDim);
  }
  if (windowSourceDim != sourceRank)
    return failure();

  Value zero;
  for (size_t inputDim = 0; inputDim < result.indices.size(); ++inputDim) {
    if (assigned[inputDim])
      continue;
    if (!zero)
      zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    result.indices[inputDim] = zero;
  }
  return result;
}

static void notifyClonedOpsRecursively(TransformRewriter &rewriter,
                                       ArrayRef<std::pair<Operation *, Operation *>> clonedOps) {
  for (auto [oldOp, newOp] : clonedOps) {
    SmallVector<Operation *> oldNestedOps, newNestedOps;
    oldOp->walk<WalkOrder::PreOrder>([&](Operation *nested) { oldNestedOps.push_back(nested); });
    newOp->walk<WalkOrder::PreOrder>([&](Operation *nested) { newNestedOps.push_back(nested); });
    assert(oldNestedOps.size() == newNestedOps.size() &&
           "cloning must preserve nested operation structure");
    for (auto [oldNested, newNested] : llvm::zip_equal(oldNestedOps, newNestedOps)) {
      if (succeeded(rewriter.notifyPayloadOperationReplaced(oldNested, newNested)))
        continue;
      // Most cloned operations have no transform handle. In that case there is
      // no mapping to update and the listener failure is expected.
      rewriter.silenceTrackingFailure();
    }
  }
}

void HTileFuseScatterIntoForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getScatterMutable(), effects);
  onlyReadsHandle(getForallMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure HTileFuseScatterIntoForallOp::apply(TransformRewriter &rewriter,
                                                                TransformResults &results,
                                                                TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());

  stablehlo::ScatterOp scatter;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getScatter, "scatter", scatter,
                               stablehlo::ScatterOp);
  scf::ForallOp forall;
  CHECK_EXTRACT_UNIQUE_OP_CAST(state, transform, getForall, "forall", forall, scf::ForallOp);

  if (scatter.getInputs().size() != 1 || scatter.getUpdates().size() != 1 ||
      scatter->getNumResults() != 1)
    BAIL("expected scatter to have exactly one input, update, and result");
  if (!scatter.getUniqueIndices())
    BAIL("expected scatter to have unique_indices = true");

  auto indicesType = dyn_cast<RankedTensorType>(scatter.getScatterIndices().getType());
  if (!indicesType)
    BAIL("expected scatter indices to be a ranked tensor");
  Type indexElementType = indicesType.getElementType();
  auto integerType = dyn_cast<IntegerType>(indexElementType);
  if (!indexElementType.isIndex() && (!integerType || !integerType.isSignless()))
    BAIL("expected scatter indices to have signless integer or index element type");

  Region &updateComputation = scatter.getUpdateComputation();
  if (!updateComputation.hasOneBlock())
    BAIL("expected scatter update computation to have one block");
  Block &updateBlock = updateComputation.front();
  if (updateBlock.getNumArguments() != 2 || !updateBlock.without_terminator().empty())
    BAIL("expected scatter update computation to directly return the update argument");
  auto returnOp = dyn_cast<stablehlo::ReturnOp>(updateBlock.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1 ||
      returnOp.getOperand(0) != updateBlock.getArgument(1))
    BAIL("expected scatter update computation to directly return the update argument");

  auto updateResult = dyn_cast<OpResult>(scatter.getUpdates().front());
  if (!updateResult || updateResult.getOwner() != forall.getOperation())
    BAIL("expected the selected forall to directly produce the scatter update");
  FailureOr<tensor::ParallelInsertSliceOp> updatePublication =
      getParallelInsertSliceForLoopResult(forall, updateResult);
  if (failed(updatePublication))
    BAIL("expected the scatter update to be published by one tensor.parallel_insert_slice");

  Value scatterInput = scatter.getInputs().front();
  Value scatterIndices = scatter.getScatterIndices();
  DominanceInfo dominance(forall->getParentOp());
  DenseSet<Operation *> visited;
  if (failed(verifyCloneableDefChainBefore(scatterInput, forall, dominance, visited)))
    BAIL("expected the scatter input definition chain to be safely clonable before the forall");
  if (failed(verifyCloneableDefChainBefore(scatterIndices, forall, dominance, visited)))
    BAIL("expected the scatter indices definition chain to be safely clonable before the forall");

  rewriter.setInsertionPoint(forall);
  IRMapping movedDefinitions;
  FailureOr<SmallVector<Value>> preparedValues = makeValuesAvailableAtInsertionPoint(
      rewriter, {scatterInput, scatterIndices}, movedDefinitions, DefChainAction::Move);
  if (failed(preparedValues))
    BAIL("failed to make the scatter input and indices available before the forall");
  Value preparedScatterInput = (*preparedValues)[0];

  rewriter.setInsertionPoint(forall.getTerminator());
  FailureOr<ParallelScatterIndexing> parallelScatterIndexing =
      buildParallelScatterIndexing(rewriter, scatter, *updatePublication);
  if (failed(parallelScatterIndexing))
    BAIL("failed to build HTile indexing for the published update tile");

  LLVM_DEBUG({
    Value preparedScatterIndices = (*preparedValues)[1];
    llvm::dbgs() << "validated scatter fusion candidate:\n"
                 << *scatter << "\nupdate publication:\n"
                 << *updatePublication << "\nprepared scatter input: ";
    llvm::dbgs() << preparedScatterInput;
    llvm::dbgs() << "\nprepared scatter indices: " << preparedScatterIndices;
    llvm::dbgs() << "\nparallel scatter indices:";
    for (Value index : parallelScatterIndexing->indices)
      llvm::dbgs() << "\n  " << index;
    llvm::dbgs() << "\nbroadcast dims: [";
    llvm::interleaveComma(parallelScatterIndexing->broadcastDims, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  // Append the scatter destination to the forall outputs. Keep the old update
  // tensor output for now so the existing publication and any DPS scratch uses
  // remain valid while we introduce the direct scatter publication.
  rewriter.setInsertionPoint(forall);
  ForallOutputExtension extension =
      cloneForallWithAppendedOutputs(rewriter, forall, ValueRange{preparedScatterInput});
  scf::ForallOp newForall = extension.forall;
  notifyClonedOpsRecursively(rewriter, extension.clonedOps);
  Value clonedUpdateTile = extension.mapping.lookup(updatePublication->getSource());
  SmallVector<Value> clonedScatterIndices =
      llvm::map_to_vector(parallelScatterIndexing->indices,
                          [&](Value index) { return extension.mapping.lookup(index); });
  pointBuilderToForallParallel(rewriter, newForall);
  auto parallelScatter = htile::ParallelScatterOp::create(
      rewriter, scatter.getLoc(), clonedUpdateTile, extension.getAppendedOutputArgs().front(),
      clonedScatterIndices, parallelScatterIndexing->broadcastDims, /*unique=*/true,
      htile::ScatterOutOfBounds::Discard);

  Value fusedResult = extension.getAppendedResults().front();
  rewriter.replaceAllUsesWith(scatter->getResult(0), fusedResult);
  rewriter.eraseOp(scatter);
  if (failed(rewriter.notifyPayloadOperationReplaced(forall, newForall)))
    BAIL("failed to preserve the scf.forall handle");
  rewriter.replaceOp(forall, extension.getPreservedResults());

  LLVM_DEBUG(llvm::dbgs() << "rebuilt forall with parallel scatter:\n" << parallelScatter << "\n");
  results.set(getOperation()->getResult(0), ArrayRef<Operation *>{parallelScatter.getOperation()});
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
