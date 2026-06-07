#include "HTile/HTileDialect.h"
#include "HTile/HTileAttrs.h"
#include "HTile/HTileOps.h"
#include "HTile/HTilePasses.h"
#include "HTile/HTileTransformOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
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
      return emitOpError() << "broadcast dimension " << dim
                           << " is outside result rank " << resultRank;
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
    if (inputExtent != mlir::ShapedType::kDynamic &&
        resultExtent != mlir::ShapedType::kDynamic &&
        inputExtent != resultExtent)
      return emitOpError() << "input dimension " << inputDim << " has extent " << inputExtent
                           << " but mapped result dimension " << resultDim << " has extent "
                           << resultExtent;
    ++inputDim;
  }
  return mlir::success();
}
} // namespace htile

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "HTileDialectPlugin", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<htile::HTileDialect>();
            htile::registerHTilePasses();
            htile::registerHTileTransformExtension(*registry);
          }};
}

#include "HTileEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "HTileAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "HTileOps.cpp.inc"
