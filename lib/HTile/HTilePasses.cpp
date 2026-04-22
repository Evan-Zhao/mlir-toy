#include "HTile/HTilePasses.h"
#include "HTile/HTileDialect.h"
#include "HTile/HTileOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

namespace htile {
namespace {

constexpr StringLiteral kDimensionOrderAttrName = "dimension_order";

RankedTensorType permuteTensorType(RankedTensorType type,
                                   ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> shape;
  shape.reserve(permutation.size());
  for (int64_t dim : permutation)
    shape.push_back(type.getShape()[dim]);
  return RankedTensorType::get(shape, type.getElementType(), type.getEncoding());
}

DenseI64ArrayAttr composeDimensionOrder(OpBuilder &builder, LoadOp load,
                                        ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> existing;
  if (auto attr = load->getAttrOfType<DenseI64ArrayAttr>(kDimensionOrderAttrName))
    existing.append(attr.asArrayRef().begin(), attr.asArrayRef().end());
  else
    for (int64_t i = 0, e = permutation.size(); i < e; ++i)
      existing.push_back(i);

  SmallVector<int64_t> composed;
  composed.reserve(permutation.size());
  for (int64_t dim : permutation)
    composed.push_back(existing[dim]);
  return builder.getDenseI64ArrayAttr(composed);
}

bool is2DRankedTensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == 2;
}

struct DotTransposeToLoadOrderPass
    : public PassWrapper<DotTransposeToLoadOrderPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DotTransposeToLoadOrderPass)

  StringRef getArgument() const override {
    return "htile-dot-transpose-to-load-order";
  }

  StringRef getDescription() const override {
    return "Fission htile.dot transpose attributes into htile.permute and "
           "fold load-fed permutes into htile.load dimension_order attributes";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HTileDialect>();
  }

  void runOnOperation() override {
    Operation *root = getOperation();

    SmallVector<DotOp> dots;
    root->walk([&](DotOp dot) {
      if (dot.getTransposeA() || dot.getTransposeB())
        dots.push_back(dot);
    });

    for (DotOp dot : dots)
      if (failed(fissionDotTransposes(dot))) {
        signalPassFailure();
        return;
      }

    SmallVector<PermuteOp> permutes;
    root->walk([&](PermuteOp permute) { permutes.push_back(permute); });

    for (PermuteOp permute : permutes)
      foldPermuteIntoLoad(permute);
  }

  LogicalResult fissionDotTransposes(DotOp dot) {
    OpBuilder builder(dot);
    SmallVector<int64_t> transpose = {1, 0};

    if (dot.getTransposeA()) {
      if (!is2DRankedTensor(dot.getLhs()))
        return dot.emitOpError("transpose_a fission expects a rank-2 lhs");
      auto type = cast<RankedTensorType>(dot.getLhs().getType());
      auto permutedType = permuteTensorType(type, transpose);
      auto permute =
          builder.create<PermuteOp>(dot.getLoc(), permutedType, dot.getLhs(),
                                    builder.getDenseI64ArrayAttr(transpose));
      dot->setOperand(0, permute.getResult());
      dot->removeAttr("transpose_a");
    }

    if (dot.getTransposeB()) {
      if (!is2DRankedTensor(dot.getRhs()))
        return dot.emitOpError("transpose_b fission expects a rank-2 rhs");
      auto type = cast<RankedTensorType>(dot.getRhs().getType());
      auto permutedType = permuteTensorType(type, transpose);
      auto permute =
          builder.create<PermuteOp>(dot.getLoc(), permutedType, dot.getRhs(),
                                    builder.getDenseI64ArrayAttr(transpose));
      dot->setOperand(1, permute.getResult());
      dot->removeAttr("transpose_b");
    }

    return success();
  }

  void foldPermuteIntoLoad(PermuteOp permute) {
    auto load = permute.getInput().getDefiningOp<LoadOp>();
    if (!load)
      return;

    ArrayRef<int64_t> permutation = permute.getPermutation();
    OpBuilder builder(permute);
    auto fusedLoad = builder.create<LoadOp>(
        permute.getLoc(), permute.getResult().getType(), load.getSource(),
        load.getOffsets());
    fusedLoad->setAttrs(load->getAttrDictionary());
    fusedLoad->setAttr(kDimensionOrderAttrName,
                       composeDimensionOrder(builder, load, permutation));

    permute.getResult().replaceAllUsesWith(fusedLoad.getResult());
    permute.erase();
    if (load->use_empty())
      load.erase();
  }
};

} // namespace

std::unique_ptr<Pass> createDotTransposeToLoadOrderPass() {
  return std::make_unique<DotTransposeToLoadOrderPass>();
}

void registerHTilePasses() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;
  PassRegistration<DotTransposeToLoadOrderPass>();
}

} // namespace htile

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "HTilePassPlugin", LLVM_VERSION_STRING,
          []() { htile::registerHTilePasses(); }};
}
