#include "HTile/HTileTransformOps.h"

#include "HTile/HTileDialect.h"
#include "LoopTr/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;

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

bool isMapEqualToDims(AffineMap map, ArrayRef<size_t> dims) {
  if (map.getNumResults() != dims.size())
    return false;
  for (auto [expr, dim] : llvm::zip_equal(map.getResults(), dims)) {
    auto maybePos = getDimPosition(expr);
    if (failed(maybePos) || *maybePos != dim)
      return false;
  }
  return true;
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

void eraseCreatedOps(SmallVectorImpl<Operation *> &createdOps) {
  for (Operation *op : llvm::reverse(createdOps))
    op->erase();
  createdOps.clear();
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

  Value empty = builder.create<tensor::EmptyOp>(
      loc, targetType.getShape(), targetType.getElementType(), targetType.getEncoding());
  createdOps.push_back(empty.getDefiningOp());
  auto broadcast = builder.create<linalg::BroadcastOp>(loc, input, empty, broadcastDims);
  createdOps.push_back(broadcast);
  return broadcast->getResult(0);
}

FailureOr<Value> materializeScalarAsTile(OpBuilder &builder, Location loc, Value scalar,
                                         RankedTensorType shapeType,
                                         SmallVectorImpl<Operation *> &createdOps) {
  if (isa<RankedTensorType>(scalar.getType()))
    return failure();
  auto resultType =
      RankedTensorType::get(shapeType.getShape(), scalar.getType(), shapeType.getEncoding());
  auto full = builder.create<htile::FullOp>(loc, resultType, scalar);
  createdOps.push_back(full);
  return full.getResult();
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
  Operation *created = nullptr;
  if (isa<arith::AddFOp>(scalarOp)) {
    created = builder.create<arith::AddFOp>(loc, operands[0], operands[1]);
  } else if (isa<arith::MulFOp>(scalarOp)) {
    created = builder.create<arith::MulFOp>(loc, operands[0], operands[1]);
  } else if (isa<arith::SubFOp>(scalarOp)) {
    created = builder.create<arith::SubFOp>(loc, operands[0], operands[1]);
  } else if (isa<arith::DivFOp>(scalarOp)) {
    created = builder.create<arith::DivFOp>(loc, operands[0], operands[1]);
  } else if (isa<arith::MaximumFOp>(scalarOp)) {
    created = builder.create<arith::MaximumFOp>(loc, operands[0], operands[1]);
  } else if (isa<arith::NegFOp>(scalarOp)) {
    created = builder.create<arith::NegFOp>(loc, operands[0]);
  } else if (isa<math::Exp2Op>(scalarOp)) {
    created = builder.create<math::Exp2Op>(loc, operands[0]);
  } else if (auto ext = dyn_cast<arith::ExtFOp>(scalarOp)) {
    auto dstType =
        RankedTensorType::get(resultShape.getShape(), ext.getType(), resultShape.getEncoding());
    created = builder.create<arith::ExtFOp>(loc, dstType, operands[0]);
  } else if (auto trunc = dyn_cast<arith::TruncFOp>(scalarOp)) {
    auto dstType =
        RankedTensorType::get(resultShape.getShape(), trunc.getType(), resultShape.getEncoding());
    created = builder.create<arith::TruncFOp>(loc, dstType, operands[0]);
  }
  if (created) {
    createdOps.push_back(created);
    return created->getResult(0);
  }
  return failure();
}

LogicalResult rewriteFill(RewriterBase &rewriter, linalg::FillOp op) {
  if (op->getNumResults() != 1)
    return failure();
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return failure();
  rewriter.setInsertionPoint(op);
  auto full = rewriter.create<htile::FullOp>(op.getLoc(), resultType, op.getInputs().front());
  rewriter.replaceOp(op, full.getResult());
  return success();
}

LogicalResult rewriteContraction(RewriterBase &rewriter, linalg::GenericOp op) {
  if (!isSingleResultTensorGeneric(op) || op.getInputs().size() != 2)
    return failure();

  SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
  if (iterators.size() != 3)
    return failure();

  SmallVector<size_t> parallelDims;
  SmallVector<size_t> reductionDims;
  for (auto [index, iterator] : llvm::enumerate(iterators)) {
    if (iterator == utils::IteratorType::parallel)
      parallelDims.push_back(index);
    else if (iterator == utils::IteratorType::reduction)
      reductionDims.push_back(index);
    else
      return failure();
  }
  if (parallelDims.size() != 2 || reductionDims.size() != 1)
    return failure();

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  size_t m = parallelDims[0];
  size_t n = parallelDims[1];
  size_t k = reductionDims[0];
  if (!isMapEqualToDims(maps.back(), {m, n}))
    return failure();

  bool transposeA = false;
  if (isMapEqualToDims(maps[0], {m, k}))
    transposeA = false;
  else if (isMapEqualToDims(maps[0], {k, m}))
    transposeA = true;
  else
    return failure();

  bool transposeB = false;
  if (isMapEqualToDims(maps[1], {k, n}))
    transposeB = false;
  else if (isMapEqualToDims(maps[1], {n, k}))
    transposeB = true;
  else
    return failure();

  rewriter.setInsertionPoint(op);
  auto dot = rewriter.create<htile::DotOp>(
      op.getLoc(), op.getResult(0).getType(), op.getInputs()[0], op.getInputs()[1],
      op.getDpsInits()[0], transposeA ? rewriter.getUnitAttr() : UnitAttr{},
      transposeB ? rewriter.getUnitAttr() : UnitAttr{}, StringAttr{});
  rewriter.replaceOp(op, dot.getResult());
  return success();
}

LogicalResult rewriteReduction(RewriterBase &rewriter, linalg::GenericOp op) {
  if (!isSingleResultTensorGeneric(op) || op.getInputs().size() != 1)
    return failure();

  SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
  std::optional<size_t> reductionAxis;
  for (auto [index, iterator] : llvm::enumerate(iterators)) {
    if (iterator == utils::IteratorType::reduction) {
      if (reductionAxis)
        return failure();
      reductionAxis = index;
    } else if (iterator != utils::IteratorType::parallel) {
      return failure();
    }
  }
  if (!reductionAxis)
    return failure();

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (!maps.front().isIdentity())
    return failure();

  SmallVector<size_t> outputDims = getMapDims(maps.back());
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  if (outputDims.size() != static_cast<size_t>(resultType.getRank()))
    return failure();
  if (llvm::is_contained(outputDims, *reductionAxis))
    return failure();

  FailureOr<BinaryReductionCombinerMatch> combiner = matchBinaryReductionCombiner(op, 0);
  if (failed(combiner) || combiner->nonAccumulator != op.getBlock()->getArgument(0))
    return failure();
  FailureOr<StringRef> kind = getHTileReductionKind(combiner->combiner);
  if (failed(kind))
    return failure();

  rewriter.setInsertionPoint(op);
  auto reduce = rewriter.create<htile::ReduceOp>(
      op.getLoc(), op.getResult(0).getType(), op.getInputs()[0],
      rewriter.getI64IntegerAttr(static_cast<int64_t>(*reductionAxis)),
      rewriter.getStringAttr(*kind));
  Value combined;
  if (*kind == "sum")
    combined = rewriter.create<arith::AddFOp>(op.getLoc(), op.getDpsInits()[0], reduce.getResult());
  else if (*kind == "max")
    combined =
        rewriter.create<arith::MaximumFOp>(op.getLoc(), op.getDpsInits()[0], reduce.getResult());
  else
    return failure();
  rewriter.replaceOp(op, combined);
  return success();
}

LogicalResult rewriteElementwise(RewriterBase &rewriter, linalg::GenericOp op) {
  if (!isSingleResultTensorGeneric(op))
    return failure();
  if (!llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType iterator) {
        return iterator == utils::IteratorType::parallel;
      }))
    return failure();

  RankedTensorType resultType = cast<RankedTensorType>(op.getResult(0).getType());
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  rewriter.setInsertionPoint(op);

  SmallVector<Operation *> createdOps;
  IRMapping mapping;
  unsigned argIndex = 0;
  for (Value input : op.getInputs()) {
    FailureOr<Value> prepared = createBroadcastToResultShape(
        rewriter, op.getLoc(), input, maps[argIndex], resultType, createdOps);
    if (failed(prepared)) {
      eraseCreatedOps(createdOps);
      return failure();
    }
    mapping.map(op.getBlock()->getArgument(argIndex), *prepared);
    ++argIndex;
  }
  for (Value init : op.getDpsInits()) {
    FailureOr<Value> prepared = createBroadcastToResultShape(
        rewriter, op.getLoc(), init, maps[argIndex], resultType, createdOps);
    if (failed(prepared)) {
      eraseCreatedOps(createdOps);
      return failure();
    }
    mapping.map(op.getBlock()->getArgument(argIndex), *prepared);
    ++argIndex;
  }

  auto yield = dyn_cast<linalg::YieldOp>(op.getBlock()->getTerminator());
  if (!yield || yield.getValues().size() != 1) {
    eraseCreatedOps(createdOps);
    return failure();
  }

  for (Operation &bodyOp : op.getBlock()->without_terminator()) {
    SmallVector<Value> mappedOperands;
    for (Value operand : bodyOp.getOperands()) {
      Value mapped = mapping.lookupOrNull(operand);
      if (!mapped) {
        FailureOr<Value> tile =
            materializeScalarAsTile(rewriter, bodyOp.getLoc(), operand, resultType, createdOps);
        if (failed(tile)) {
          eraseCreatedOps(createdOps);
          return failure();
        }
        mapped = *tile;
      }
      mappedOperands.push_back(mapped);
    }
    FailureOr<Value> tensorOp = createTensorScalarLikeOp(rewriter, bodyOp.getLoc(), &bodyOp,
                                                         mappedOperands, resultType, createdOps);
    if (failed(tensorOp) || bodyOp.getNumResults() != 1) {
      eraseCreatedOps(createdOps);
      return failure();
    }
    mapping.map(bodyOp.getResult(0), *tensorOp);
  }

  Value replacement = mapping.lookupOrNull(yield.getValues()[0]);
  if (!replacement) {
    eraseCreatedOps(createdOps);
    return failure();
  }
  rewriter.replaceOp(op, replacement);
  return success();
}

LogicalResult rewriteOriginalLinalgOp(RewriterBase &rewriter, Operation *op) {
  if (!op->getParentOp())
    return success();
  if (auto fill = dyn_cast<linalg::FillOp>(op))
    return rewriteFill(rewriter, fill);
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
  if (isa<linalg::BroadcastOp>(op))
    return success();
  op->emitError() << "unsupported linalg operation";
  return failure();
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

  SmallVector<Operation *> originalLinalgOps;
  target->walk([&](linalg::LinalgOp op) { originalLinalgOps.push_back(op); });

  for (Operation *op : originalLinalgOps) {
    if (succeeded(rewriteOriginalLinalgOp(rewriter, op)))
      continue;
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform

namespace htile {

void registerHTileTransformExtension(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, mlir::transform::TransformDialect *dialect) {
    ctx->loadDialect<htile::HTileDialect>();
    struct TransformDialectAccess : public mlir::transform::TransformDialect {
      using mlir::Dialect::addOperations;
    };
    static_cast<TransformDialectAccess *>(dialect)
        ->addOperations<mlir::transform::HTileLinalgToSemanticOp>();
  });
}

} // namespace htile

#define GET_OP_CLASSES
#include "HTileTransformOps.cpp.inc"
