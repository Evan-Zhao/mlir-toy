#include "HTile/HTilePasses.h"
#include "HTile/HTileDialect.h"
#include "HTile/HTileOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace htile {
#define GEN_PASS_DEF_DOTTRANSPOSETOLOADORDERPASS
#include "HTilePasses.h.inc"

namespace {

constexpr StringLiteral kDimensionOrderAttrName = "dimension_order";

RankedTensorType permuteTensorType(RankedTensorType type, ArrayRef<int64_t> permutation) {
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
    : public impl::DotTransposeToLoadOrderPassBase<DotTransposeToLoadOrderPass> {
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
      auto permute = PermuteOp::create(builder, dot.getLoc(), permutedType, dot.getLhs(),
                                       builder.getDenseI64ArrayAttr(transpose));
      dot->setOperand(0, permute.getResult());
      dot->removeAttr("transpose_a");
    }

    if (dot.getTransposeB()) {
      if (!is2DRankedTensor(dot.getRhs()))
        return dot.emitOpError("transpose_b fission expects a rank-2 rhs");
      auto type = cast<RankedTensorType>(dot.getRhs().getType());
      auto permutedType = permuteTensorType(type, transpose);
      auto permute = PermuteOp::create(builder, dot.getLoc(), permutedType, dot.getRhs(),
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
    auto fusedLoad = LoadOp::create(builder, permute.getLoc(), permute.getResult().getType(),
                                    load.getSource(), load.getOffsets());
    fusedLoad->setAttrs(load->getAttrDictionary());
    fusedLoad->setAttr(kDimensionOrderAttrName, composeDimensionOrder(builder, load, permutation));

    permute.getResult().replaceAllUsesWith(fusedLoad.getResult());
    permute.erase();
    if (load->use_empty())
      load.erase();
  }
};

} // namespace
} // namespace htile
