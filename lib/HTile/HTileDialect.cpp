#include "HTile/HTileDialect.h"
#include "HTile/HTileAttrs.h"
#include "HTile/HTileOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_DIALECT_DEFS
#include "HTileOpsDialect.cpp.inc"

namespace htile {
void HTileDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "HTileAttrs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "HTileOps.cpp.inc"
      >();
}

mlir::ParseResult KernelOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
    return mlir::failure();

  llvm::SmallVector<mlir::OpAsmParser::Argument> args;
  if (parser.parseArgumentList(args, mlir::OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return mlir::failure();

  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, args, /*enableNameShadowing=*/true))
    return mlir::failure();
  KernelOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return mlir::success();
}

void KernelOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer << '(';
  llvm::interleaveComma(getBody().getArguments(), printer,
                        [&](mlir::BlockArgument arg) { printer << arg << " : " << arg.getType(); });
  printer << ")";
  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                           {mlir::SymbolTable::getSymbolAttrName()});
  printer << ' ';
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

mlir::LogicalResult LaunchFuncOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  KernelOp kernelOp = symbolTable.lookupNearestSymbolFrom<KernelOp>(*this, getKernelAttr());
  if (!kernelOp)
    return emitOpError() << "'" << getKernelAttr().getValue()
                         << "' does not reference a valid htile.kernel";
  return mlir::success();
}

mlir::LogicalResult LoadOp::verify() {
  bool hasMask = static_cast<bool>(getMask());
  bool hasOther = static_cast<bool>(getOther());
  if (hasMask != hasOther)
    return emitOpError("requires mask and other to be supplied together");
  if (!hasMask)
    return mlir::success();

  auto resultType = mlir::cast<mlir::RankedTensorType>(getResult().getType());
  auto maskType = mlir::cast<mlir::RankedTensorType>(getMask().getType());
  if (!maskType.getElementType().isInteger(1))
    return emitOpError("requires mask to have i1 element type");
  if (maskType.getShape() != resultType.getShape())
    return emitOpError("requires mask shape to match result shape");

  mlir::Type otherType = getOther().getType();
  if (mlir::isa<mlir::ShapedType>(otherType))
    return emitOpError("requires other to be a scalar");
  if (otherType != resultType.getElementType())
    return emitOpError("requires other type to match the result element type");
  return mlir::success();
}

void MaskedParallelInsertSliceOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                                        mlir::Value source, mlir::Value dest,
                                        llvm::ArrayRef<mlir::OpFoldResult> offsets,
                                        llvm::ArrayRef<mlir::OpFoldResult> sizes,
                                        llvm::ArrayRef<mlir::OpFoldResult> strides, mlir::Value mask,
                                        llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  llvm::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  llvm::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  mlir::dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  mlir::dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  mlir::dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  result.addAttributes(attrs);
  build(builder, result, {}, source, dest, dynamicOffsets, dynamicSizes, dynamicStrides, mask,
        builder.getDenseI64ArrayAttr(staticOffsets), builder.getDenseI64ArrayAttr(staticSizes),
        builder.getDenseI64ArrayAttr(staticStrides));
}

mlir::LogicalResult MaskedParallelInsertSliceOp::verify() {
  if (!mlir::isa<mlir::InParallelOpInterface>(getOperation()->getParentOp()))
    return emitOpError("must be directly nested in an in-parallel operation");

  auto sourceType = getSourceType();
  auto destType = getDestType();
  auto maskType = mlir::cast<mlir::RankedTensorType>(getMask().getType());
  if (sourceType.getElementType() != destType.getElementType())
    return emitOpError("requires source and destination element types to match");
  if (!maskType.getElementType().isInteger(1))
    return emitOpError("requires mask to have i1 element type");
  if (maskType.getShape() != sourceType.getShape())
    return emitOpError("requires mask shape to match source shape");

  if (!mlir::computeRankReductionMask(getStaticSizes(), sourceType.getShape(),
                                      /*matchDynamic=*/true))
    return emitOpError(
        "requires source shape to match slice sizes after dropping static unit dimensions");
  return mlir::success();
}

mlir::MutableOperandRange MaskedParallelInsertSliceOp::getUpdatedDestinations() {
  return getDestMutable();
}

mlir::Operation *MaskedParallelInsertSliceOp::getIteratingParent() {
  if (auto combiningOp =
          mlir::dyn_cast<mlir::InParallelOpInterface>(getOperation()->getParentOp()))
    return combiningOp->getParentOp();
  return nullptr;
}

mlir::MutableOperandRange ParallelScatterOp::getUpdatedDestinations() {
  return getDestMutable();
}

mlir::Operation *ParallelScatterOp::getIteratingParent() {
  if (auto combiningOp =
          mlir::dyn_cast<mlir::InParallelOpInterface>(getOperation()->getParentOp()))
    return combiningOp->getParentOp();
  return nullptr;
}

static bool isValidIndexElementType(mlir::Type type) {
  if (type.isIndex())
    return true;
  auto integerType = mlir::dyn_cast<mlir::IntegerType>(type);
  return integerType && integerType.isSignless();
}

mlir::LogicalResult ParallelScatterOp::verify() {
  if (!mlir::isa<mlir::InParallelOpInterface>(getOperation()->getParentOp()))
    return emitOpError("must be directly nested in an in-parallel operation");
  if (!getUnique())
    return emitOpError("requires the 'unique' keyword");

  auto sourceType = mlir::cast<mlir::RankedTensorType>(getSource().getType());
  auto destType = mlir::cast<mlir::RankedTensorType>(getDest().getType());
  if (sourceType.getElementType() != destType.getElementType())
    return emitOpError("requires source and destination element types to match");
  if (getIndices().size() != static_cast<size_t>(destType.getRank()))
    return emitOpError("requires one index entry per destination dimension");

  llvm::ArrayRef<int64_t> broadcastDims = getBroadcastDims();
  int64_t previous = -1;
  for (int64_t dim : broadcastDims) {
    if (dim < 0 || dim >= destType.getRank())
      return emitOpError("broadcast dimension ") << dim << " is outside destination rank "
                                                  << destType.getRank();
    if (dim <= previous)
      return emitOpError("requires broadcast dimensions to be strictly increasing");
    previous = dim;
  }
  if (broadcastDims.size() > static_cast<size_t>(sourceType.getRank()))
    return emitOpError("has more broadcast dimensions than source dimensions");

  int64_t batchRank = sourceType.getRank() - static_cast<int64_t>(broadcastDims.size());
  for (auto [destDim, index] : llvm::enumerate(getIndices())) {
    mlir::Type indexType = index.getType();
    bool isBroadcastDim =
        llvm::is_contained(broadcastDims, static_cast<int64_t>(destDim));
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(indexType)) {
      if (isBroadcastDim)
        return emitOpError("requires the index for broadcast dimension ")
               << destDim << " to be a scalar base offset";
      if (!isValidIndexElementType(tensorType.getElementType()))
        return emitOpError("requires tensor indices to have signless integer or index elements");
      if (tensorType.getRank() > batchRank)
        return emitOpError("index tensor rank exceeds source batch rank");
      int64_t sourceDim = batchRank - tensorType.getRank();
      for (int64_t indexDim = 0; indexDim < tensorType.getRank(); ++indexDim, ++sourceDim) {
        int64_t indexExtent = tensorType.getDimSize(indexDim);
        int64_t sourceExtent = sourceType.getDimSize(sourceDim);
        if (indexExtent != 1 && indexExtent != mlir::ShapedType::kDynamic &&
            sourceExtent != mlir::ShapedType::kDynamic && indexExtent != sourceExtent)
          return emitOpError("index tensor dimension ")
                 << indexDim << " does not broadcast to source batch dimension " << sourceDim;
      }
      continue;
    }
    if (!isValidIndexElementType(indexType))
      return emitOpError("requires each index to be a scalar or ranked tensor of signless "
                         "integer or index type");
  }

  return mlir::success();
}

mlir::LogicalResult BroadcastOp::verify() {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  if (inputType.getElementType() != resultType.getElementType())
    return emitOpError() << "requires input and result element types to match";

  int64_t inputRank = inputType.getRank();
  int64_t resultRank = resultType.getRank();
  llvm::ArrayRef<int64_t> dimensions = getDimensions();
  if (inputRank + static_cast<int64_t>(dimensions.size()) != resultRank)
    return emitOpError() << "requires input rank plus broadcast dimensions to equal result rank";

  llvm::SmallVector<bool> isBroadcastDim(static_cast<size_t>(resultRank), false);
  std::optional<int64_t> previous;
  for (int64_t dim : dimensions) {
    if (dim < 0 || dim >= resultRank)
      return emitOpError() << "broadcast dimension " << dim << " is outside result rank "
                           << resultRank;
    if (previous && dim <= *previous)
      return emitOpError() << "requires broadcast dimensions to be strictly increasing";
    isBroadcastDim[static_cast<size_t>(dim)] = true;
    previous = dim;
  }

  int64_t inputDim = 0;
  for (int64_t resultDim = 0; resultDim < resultRank; ++resultDim) {
    if (isBroadcastDim[static_cast<size_t>(resultDim)])
      continue;
    int64_t inputExtent = inputType.getDimSize(inputDim);
    int64_t resultExtent = resultType.getDimSize(resultDim);
    if (inputExtent != mlir::ShapedType::kDynamic && resultExtent != mlir::ShapedType::kDynamic &&
        inputExtent != resultExtent)
      return emitOpError() << "input dimension " << inputDim << " has extent " << inputExtent
                           << " but mapped result dimension " << resultDim << " has extent "
                           << resultExtent;
    ++inputDim;
  }
  return mlir::success();
}
} // namespace htile

#include "HTileEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "HTileAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "HTileOps.cpp.inc"
