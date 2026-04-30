#include "LinalgExt/LinalgExtTransform.h"
#include "LinalgExt/LinalgExtTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

bool isAncestor(Operation *ancestor, Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent == ancestor)
      return true;
  return false;
}

DiagnosedSilenceableFailure
emitSilenceableFailure(transform::TransformOpInterface transform,
                       const Twine &message) {
  return transform.emitSilenceableError(message);
}

LogicalResult verifyUnaryMap(linalg::MapOp map) {
  if (map.getInputs().size() != 1)
    return failure();
  if (map->getNumResults() != 1)
    return failure();
  return success();
}

linalg::MapOp cloneMapOnTile(RewriterBase &rewriter, linalg::MapOp sourceMap,
                             Value inputTile, Value initTile,
                             Location loc) {
  auto cloned = linalg::MapOp::create(
      rewriter, loc, ValueRange{inputTile}, initTile,
      [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
        Block &oldBlock = sourceMap.getMapper().front();
        IRMapping mapping;
        for (auto [oldArg, newArg] : llvm::zip_equal(oldBlock.getArguments(),
                                                     newArgs))
          mapping.map(oldArg, newArg);

        for (Operation &op : oldBlock.without_terminator())
          builder.clone(op, mapping);

        auto oldYield = cast<linalg::YieldOp>(oldBlock.getTerminator());
        SmallVector<Value> yielded;
        yielded.reserve(oldYield.getValues().size());
        for (Value value : oldYield.getValues())
          yielded.push_back(mapping.lookup(value));
        linalg::YieldOp::create(builder, nestedLoc, yielded);
      });
  return cloned;
}

} // namespace

namespace mlir {
namespace transform {

DiagnosedSilenceableFailure
LinalgExtFuseUnaryElementwiseConsumerIntoLoopOp::apply(
    transform::TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {
  SmallVector<Operation *> consumers =
      llvm::to_vector(state.getPayloadOps(getConsumerOp()));
  SmallVector<Operation *> loops =
      llvm::to_vector(state.getPayloadOps(getContainingLoop()));

  auto transform = cast<TransformOpInterface>(getOperation());
  if (consumers.size() != 1)
    return emitSilenceableFailure(transform,
                                  "expected exactly one consumer payload op");
  if (loops.size() != 1)
    return emitSilenceableFailure(
        transform, "expected exactly one containing loop payload op");

  auto map = dyn_cast<linalg::MapOp>(consumers.front());
  if (!map || failed(verifyUnaryMap(map)))
    return emitSilenceableFailure(
        transform, "expected a unary tensor linalg.map consumer");

  auto loop = dyn_cast<scf::ForOp>(loops.front());
  if (!loop)
    return emitSilenceableFailure(transform,
                                  "expected an scf.for containing loop");

  Value consumerInput = map.getInputs().front();
  Operation *producerContainer = consumerInput.getDefiningOp();
  if (!producerContainer)
    return emitSilenceableFailure(
        transform, "consumer input must be produced by an operation");
  if (!isAncestor(producerContainer, loop))
    return emitSilenceableFailure(
        transform,
        "containing loop must be nested under the consumer input producer");

  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getResults().size() != 1)
    return emitSilenceableFailure(
        transform, "expected containing loop to yield one tensor value");

  auto insert = yield.getResults().front().getDefiningOp<tensor::InsertSliceOp>();
  if (!insert)
    return emitSilenceableFailure(
        transform, "expected loop yield operand to be a tensor.insert_slice");

  rewriter.setInsertionPoint(insert);
  auto initTile = tensor::ExtractSliceOp::create(
      rewriter, insert.getLoc(), insert.getSource().getType(), insert.getDest(),
      insert.getMixedOffsets(), insert.getMixedSizes(),
      insert.getMixedStrides());
  linalg::MapOp fused =
      cloneMapOnTile(rewriter, map, insert.getSource(), initTile.getResult(),
                     map.getLoc());

  rewriter.modifyOpInPlace(insert, [&]() {
    insert.getSourceMutable().assign(fused.getResult().front());
  });

  rewriter.replaceOp(map, consumerInput);

  transformResults.set(getOperation()->getResult(0),
                       ArrayRef<Operation *>{fused.getOperation()});
  transformResults.set(getOperation()->getResult(1),
                       ArrayRef<Operation *>{loop.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace mlir

namespace linalg_ext {

void registerLinalgExtTransformExtension(mlir::DialectRegistry &registry) {
  // MLIR 22.1.4 hangs in TransformDialectExtension::registerTransformOps for
  // this out-of-tree plugin. Register the op directly as a narrow workaround;
  // the op is still constrained by its TableGen traits and verified when used.
  registry.addExtension(+[](mlir::MLIRContext *,
                            mlir::transform::TransformDialect *dialect) {
    struct TransformDialectAccess : public mlir::transform::TransformDialect {
      using mlir::Dialect::addOperations;
    };
    static_cast<TransformDialectAccess *>(dialect)
        ->addOperations<
            mlir::transform::LinalgExtFuseUnaryElementwiseConsumerIntoLoopOp>();
  });
}

} // namespace linalg_ext

#define GET_OP_CLASSES
#include "LinalgExtTransformOps.cpp.inc"

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION,
      "LinalgExtTransformPlugin",
      LLVM_VERSION_STRING,
      [](mlir::DialectRegistry *registry) {
        linalg_ext::registerLinalgExtTransformExtension(*registry);
      }};
}
