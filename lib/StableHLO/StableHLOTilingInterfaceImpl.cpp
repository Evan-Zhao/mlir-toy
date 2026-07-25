#include "StableHLO/StableHLOTilingInterfaceImpl.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/DenseSet.h"

#include <iterator>

using namespace mlir;

namespace {

/// Return the result dimension corresponding to every start_indices batch
/// dimension. StableHLO places the gather window dimensions at `offset_dims`;
/// the remaining result dimensions correspond, in order, to start_indices
/// dimensions other than `index_vector_dim`.
static FailureOr<SmallVector<int64_t>> getStartIndicesDimToResultDim(stablehlo::GatherOp gatherOp) {
  auto indicesType = dyn_cast<RankedTensorType>(gatherOp.getStartIndices().getType());
  auto resultType = dyn_cast<RankedTensorType>(gatherOp.getResult().getType());
  if (!indicesType || !resultType)
    return failure();

  auto dimNumbers = gatherOp.getDimensionNumbers();
  int64_t indexVectorDim = dimNumbers.getIndexVectorDim();
  int64_t indicesRank = indicesType.getRank();
  if (indexVectorDim < 0 || indexVectorDim > indicesRank)
    return failure();

  llvm::DenseSet<int64_t> offsetDims;
  offsetDims.insert(dimNumbers.getOffsetDims().begin(), dimNumbers.getOffsetDims().end());
  SmallVector<int64_t> batchResultDims;
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim)
    if (!offsetDims.contains(dim))
      batchResultDims.push_back(dim);

  SmallVector<int64_t> mapping(indicesRank, -1);
  size_t nextResultDim = 0;
  for (int64_t dim = 0; dim < indicesRank; ++dim) {
    if (dim == indexVectorDim)
      continue;
    if (nextResultDim == batchResultDims.size())
      return failure();
    mapping[dim] = batchResultDims[nextResultDim++];
  }
  if (nextResultDim != batchResultDims.size())
    return failure();
  return mapping;
}

static SmallVector<int64_t> getWindowOperandDims(stablehlo::GatherOp gatherOp) {
  auto operandType = cast<RankedTensorType>(gatherOp.getOperand().getType());
  auto dimNumbers = gatherOp.getDimensionNumbers();
  llvm::DenseSet<int64_t> collapsedDims;
  collapsedDims.insert(dimNumbers.getCollapsedSliceDims().begin(),
                       dimNumbers.getCollapsedSliceDims().end());
  llvm::DenseSet<int64_t> batchingDims;
  batchingDims.insert(dimNumbers.getOperandBatchingDims().begin(),
                      dimNumbers.getOperandBatchingDims().end());

  SmallVector<int64_t> windowDims;
  for (int64_t dim = 0; dim < operandType.getRank(); ++dim)
    if (!collapsedDims.contains(dim) && !batchingDims.contains(dim))
      windowDims.push_back(dim);
  return windowDims;
}

/// The initial implementation supports the common gather form where every
/// explicitly indexed operand dimension is collapsed. This includes both the
/// packed-varlen and sparse-attention gathers. Tiling a non-collapsed indexed
/// dimension would require adding the result tile offset to start_indices.
static bool isSupportedGather(stablehlo::GatherOp gatherOp) {
  auto operandType = dyn_cast<RankedTensorType>(gatherOp.getOperand().getType());
  auto indicesType = dyn_cast<RankedTensorType>(gatherOp.getStartIndices().getType());
  auto resultType = dyn_cast<RankedTensorType>(gatherOp.getResult().getType());
  if (!operandType || !indicesType || !resultType)
    return false;

  auto dimNumbers = gatherOp.getDimensionNumbers();
  llvm::DenseSet<int64_t> collapsedDims;
  collapsedDims.insert(dimNumbers.getCollapsedSliceDims().begin(),
                       dimNumbers.getCollapsedSliceDims().end());
  for (int64_t dim : dimNumbers.getStartIndexMap())
    if (!collapsedDims.contains(dim))
      return false;

  if (dimNumbers.getOperandBatchingDims().size() != dimNumbers.getStartIndicesBatchingDims().size())
    return false;

  SmallVector<int64_t> windowDims = getWindowOperandDims(gatherOp);
  if (windowDims.size() != dimNumbers.getOffsetDims().size())
    return false;

  return succeeded(getStartIndicesDimToResultDim(gatherOp));
}

static OpFoldResult getDimension(OpBuilder &builder, Location loc, Value value, int64_t dim) {
  auto type = cast<RankedTensorType>(value.getType());
  if (!type.isDynamicDim(dim))
    return builder.getIndexAttr(type.getDimSize(dim));
  return tensor::DimOp::create(builder, loc, value, dim).getResult();
}

static RankedTensorType getTileType(RankedTensorType originalType, ArrayRef<OpFoldResult> sizes) {
  SmallVector<int64_t> shape;
  shape.reserve(sizes.size());
  for (OpFoldResult size : sizes) {
    std::optional<int64_t> constant = getConstantIntValue(size);
    shape.push_back(constant.value_or(ShapedType::kDynamic));
  }
  return RankedTensorType::get(shape, originalType.getElementType(), originalType.getEncoding());
}

struct StableHLOGatherTilingInterface
    : public TilingInterface::ExternalModel<StableHLOGatherTilingInterface, stablehlo::GatherOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto gatherOp = cast<stablehlo::GatherOp>(op);
    auto resultType = cast<RankedTensorType>(gatherOp.getResult().getType());
    return SmallVector<utils::IteratorType>(resultType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &builder) const {
    auto gatherOp = cast<stablehlo::GatherOp>(op);
    auto resultType = cast<RankedTensorType>(gatherOp.getResult().getType());
    FailureOr<SmallVector<int64_t>> indicesToResultDim = getStartIndicesDimToResultDim(gatherOp);
    if (failed(indicesToResultDim))
      return {};

    OpFoldResult zero = builder.getIndexAttr(0);
    OpFoldResult one = builder.getIndexAttr(1);
    SmallVector<Range> ranges;
    ranges.reserve(resultType.getRank());
    for (int64_t resultDim = 0; resultDim < resultType.getRank(); ++resultDim) {
      OpFoldResult size;
      if (!resultType.isDynamicDim(resultDim)) {
        size = builder.getIndexAttr(resultType.getDimSize(resultDim));
      } else {
        auto indicesDim = llvm::find(*indicesToResultDim, resultDim);
        if (indicesDim == indicesToResultDim->end())
          return {};
        size = getDimension(builder, gatherOp.getLoc(), gatherOp.getStartIndices(),
                            std::distance(indicesToResultDim->begin(), indicesDim));
      }
      ranges.push_back(Range{zero, size, one});
    }
    return ranges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &builder,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) const {
    auto gatherOp = cast<stablehlo::GatherOp>(op);
    auto resultType = cast<RankedTensorType>(gatherOp.getResult().getType());
    if (!isSupportedGather(gatherOp) ||
        offsets.size() != static_cast<size_t>(resultType.getRank()) ||
        sizes.size() != static_cast<size_t>(resultType.getRank()))
      return failure();

    Location loc = gatherOp.getLoc();
    OpFoldResult zero = builder.getIndexAttr(0);
    OpFoldResult one = builder.getIndexAttr(1);
    auto dimNumbers = gatherOp.getDimensionNumbers();

    FailureOr<SmallVector<int64_t>> indicesToResultDim = getStartIndicesDimToResultDim(gatherOp);
    if (failed(indicesToResultDim))
      return failure();

    // Slice start_indices along all of its batch dimensions. Keep the index
    // vector dimension intact.
    auto indicesType = cast<RankedTensorType>(gatherOp.getStartIndices().getType());
    SmallVector<OpFoldResult> indicesOffsets(indicesType.getRank(), zero);
    SmallVector<OpFoldResult> indicesSizes;
    indicesSizes.reserve(indicesType.getRank());
    for (int64_t dim = 0; dim < indicesType.getRank(); ++dim) {
      int64_t resultDim = (*indicesToResultDim)[dim];
      if (resultDim < 0) {
        indicesSizes.push_back(getDimension(builder, loc, gatherOp.getStartIndices(), dim));
        continue;
      }
      indicesOffsets[dim] = offsets[resultDim];
      indicesSizes.push_back(sizes[resultDim]);
    }
    SmallVector<OpFoldResult> indicesStrides(indicesType.getRank(), one);
    auto indicesSlice = tensor::ExtractSliceOp::create(
        builder, loc, gatherOp.getStartIndices(), indicesOffsets, indicesSizes, indicesStrides);

    // Keep dynamically indexed/collapsed operand dimensions whole. Slice
    // operand batching dimensions together with their corresponding indices
    // dimensions, and slice the ordinary window dimensions according to the
    // requested result tile.
    auto operandType = cast<RankedTensorType>(gatherOp.getOperand().getType());
    SmallVector<OpFoldResult> operandOffsets(operandType.getRank(), zero);
    SmallVector<OpFoldResult> operandSizes;
    operandSizes.reserve(operandType.getRank());
    for (int64_t dim = 0; dim < operandType.getRank(); ++dim)
      operandSizes.push_back(getDimension(builder, loc, gatherOp.getOperand(), dim));

    for (auto [operandDim, indicesDim] : llvm::zip_equal(
             dimNumbers.getOperandBatchingDims(), dimNumbers.getStartIndicesBatchingDims())) {
      int64_t resultDim = (*indicesToResultDim)[indicesDim];
      if (resultDim < 0)
        return failure();
      operandOffsets[operandDim] = offsets[resultDim];
      operandSizes[operandDim] = sizes[resultDim];
    }

    SmallVector<int64_t> tiledSliceSizes(gatherOp.getSliceSizes().begin(),
                                         gatherOp.getSliceSizes().end());
    SmallVector<int64_t> windowDims = getWindowOperandDims(gatherOp);
    for (auto [operandDim, resultDim] : llvm::zip_equal(windowDims, dimNumbers.getOffsetDims())) {
      std::optional<int64_t> staticSize = getConstantIntValue(sizes[resultDim]);
      if (!staticSize)
        return failure();
      operandOffsets[operandDim] = offsets[resultDim];
      operandSizes[operandDim] = sizes[resultDim];
      tiledSliceSizes[operandDim] = *staticSize;
    }

    SmallVector<OpFoldResult> operandStrides(operandType.getRank(), one);
    auto operandSlice = tensor::ExtractSliceOp::create(
        builder, loc, gatherOp.getOperand(), operandOffsets, operandSizes, operandStrides);

    RankedTensorType tiledResultType = getTileType(resultType, sizes);
    Operation *tiledGather =
        mlir::clone(builder, gatherOp.getOperation(), TypeRange{tiledResultType},
                    ValueRange{operandSlice.getResult(), indicesSlice.getResult()});
    tiledGather->setAttr(gatherOp.getSliceSizesAttrName(),
                         builder.getDenseI64ArrayAttr(tiledSliceSizes));

    return TilingResult{{tiledGather},
                        {tiledGather->getResult(0)},
                        {operandSlice.getOperation(), indicesSlice.getOperation()}};
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &builder, unsigned resultNumber,
                                      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
                                      SmallVector<OpFoldResult> &resultOffsets,
                                      SmallVector<OpFoldResult> &resultSizes) const {
    if (resultNumber != 0)
      return failure();
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &builder,
                                                  unsigned resultNumber,
                                                  ArrayRef<OpFoldResult> offsets,
                                                  ArrayRef<OpFoldResult> sizes) const {
    if (resultNumber != 0)
      return failure();
    return getTiledImplementation(op, builder, offsets, sizes);
  }
};

} // namespace

void neptune::registerStableHLOTilingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, stablehlo::StablehloDialect *dialect) {
    stablehlo::GatherOp::attachInterface<StableHLOGatherTilingInterface>(*ctx);
  });
}
