#include "HTile/HTileDialect.h"
#include "HTile/HTileAttrs.h"
#include "HTile/HTileOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
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

mlir::LogicalResult GatherNdOp::verify() {
  auto sourceType = mlir::dyn_cast<mlir::ShapedType>(getSource().getType());
  if (!sourceType || !sourceType.hasRank())
    return emitOpError("requires a ranked tensor or memref source");
  if (getOperation()->getParentOfType<KernelOp>() &&
      !mlir::isa<mlir::MemRefType>(getSource().getType()))
    return emitOpError("requires source to be a memref inside an htile.kernel");

  auto resultType = getResult().getType();
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("requires source and result element types to match");
  if (getIndices().size() != static_cast<size_t>(sourceType.getRank()))
    return emitOpError() << "requires one index tensor per source dimension; expected "
                         << sourceType.getRank() << " but got " << getIndices().size();

  for (auto [index, value] : llvm::enumerate(getIndices())) {
    auto indexType = mlir::cast<mlir::RankedTensorType>(value.getType());
    if (!indexType.getElementType().isIntOrIndex())
      return emitOpError() << "requires index tensor #" << index
                           << " to have integer or index element type";
    if (indexType.getShape() != resultType.getShape())
      return emitOpError() << "requires index tensor #" << index
                           << " shape to match result shape";
  }
  return mlir::success();
}

mlir::LogicalResult LoadOp::verify() {
  if (getOperation()->getParentOfType<KernelOp>() &&
      !mlir::isa<mlir::MemRefType>(getSource().getType()))
    return emitOpError("requires source to be a memref inside an htile.kernel");

  mlir::Type resultType = getResult().getType();
  auto resultTensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
  if (!resultTensorType && !resultType.isIntOrIndexOrFloat())
    return emitOpError("requires result to be a scalar or ranked tensor");

  bool hasMask = static_cast<bool>(getMask());
  bool hasOther = static_cast<bool>(getOther());
  if (hasMask != hasOther)
    return emitOpError("requires mask and other to be supplied together");
  if (!hasMask)
    return mlir::success();

  auto maskType = mlir::cast<mlir::RankedTensorType>(getMask().getType());
  if (!maskType.getElementType().isInteger(1))
    return emitOpError("requires mask to have i1 element type");
  llvm::ArrayRef<int64_t> resultShape =
      resultTensorType ? resultTensorType.getShape() : llvm::ArrayRef<int64_t>{};
  if (maskType.getShape() != resultShape)
    return emitOpError("requires mask shape to match result shape");

  mlir::Type otherType = getOther().getType();
  if (mlir::isa<mlir::ShapedType>(otherType))
    return emitOpError("requires other to be a scalar");
  mlir::Type resultElementType =
      resultTensorType ? resultTensorType.getElementType() : resultType;
  if (otherType != resultElementType)
    return emitOpError("requires other type to match the result element type");
  return mlir::success();
}

mlir::LogicalResult StoreOp::verify() {
  if (!getMask())
    return mlir::success();

  auto valueType = mlir::cast<mlir::RankedTensorType>(getValue().getType());
  auto maskType = mlir::cast<mlir::RankedTensorType>(getMask().getType());
  if (!maskType.getElementType().isInteger(1))
    return emitOpError("requires mask to have i1 element type");
  if (maskType.getShape() != valueType.getShape())
    return emitOpError("requires mask shape to match value shape");
  return mlir::success();
}

void MaskedParallelInsertSliceOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                                        mlir::Value source, mlir::Value dest,
                                        llvm::ArrayRef<mlir::OpFoldResult> offsets,
                                        llvm::ArrayRef<mlir::OpFoldResult> sizes,
                                        llvm::ArrayRef<mlir::OpFoldResult> strides,
                                        mlir::Value mask,
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
  if (auto combiningOp = mlir::dyn_cast<mlir::InParallelOpInterface>(getOperation()->getParentOp()))
    return combiningOp->getParentOp();
  return nullptr;
}

mlir::LogicalResult UnsqueezeOp::verify() {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  if (inputType.getElementType() != resultType.getElementType())
    return emitOpError() << "requires input and result element types to match";

  llvm::ArrayRef<bool> mask = getMask();
  if (mask.size() != static_cast<size_t>(resultType.getRank()))
    return emitOpError() << "requires one mask entry per result dimension";

  int64_t inputDim = 0;
  for (auto [resultDim, insert] : llvm::enumerate(mask)) {
    int64_t resultExtent = resultType.getDimSize(static_cast<int64_t>(resultDim));
    if (insert) {
      if (resultExtent != 1)
        return emitOpError() << "cannot insert result dimension " << resultDim << " with extent "
                             << resultExtent;
      continue;
    }
    if (inputDim >= inputType.getRank())
      return emitOpError() << "requires input rank plus inserted dimensions to equal result rank";
    int64_t inputExtent = inputType.getDimSize(inputDim);
    if (inputExtent != mlir::ShapedType::kDynamic &&
        resultExtent != mlir::ShapedType::kDynamic && inputExtent != resultExtent)
      return emitOpError() << "input dimension " << inputDim << " has extent " << inputExtent
                           << " but mapped result dimension " << resultDim << " has extent "
                           << resultExtent;
    ++inputDim;
  }
  if (inputDim != inputType.getRank())
    return emitOpError() << "requires input rank plus inserted dimensions to equal result rank";
  return mlir::success();
}

mlir::LogicalResult SqueezeOp::verify() {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  if (inputType.getElementType() != resultType.getElementType())
    return emitOpError() << "requires input and result element types to match";

  llvm::ArrayRef<bool> mask = getMask();
  if (mask.size() != static_cast<size_t>(inputType.getRank()))
    return emitOpError() << "requires one mask entry per input dimension";

  int64_t resultDim = 0;
  for (auto [inputDim, remove] : llvm::enumerate(mask)) {
    int64_t inputExtent = inputType.getDimSize(static_cast<int64_t>(inputDim));
    if (remove) {
      if (inputExtent != 1)
        return emitOpError() << "cannot remove input dimension " << inputDim << " with extent "
                             << inputExtent;
      continue;
    }
    if (resultDim >= resultType.getRank())
      return emitOpError() << "requires input rank minus removed dimensions to equal result rank";
    int64_t resultExtent = resultType.getDimSize(resultDim);
    if (inputExtent != mlir::ShapedType::kDynamic &&
        resultExtent != mlir::ShapedType::kDynamic && inputExtent != resultExtent)
      return emitOpError() << "input dimension " << inputDim << " has extent " << inputExtent
                           << " but mapped result dimension " << resultDim << " has extent "
                           << resultExtent;
    ++resultDim;
  }
  if (resultDim != resultType.getRank())
    return emitOpError() << "requires input rank minus removed dimensions to equal result rank";
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
