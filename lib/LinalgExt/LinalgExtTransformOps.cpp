#include "LinalgExt/LinalgExtTransformOps.h"
#include "LinalgExt/LinalgExtTransform.h"

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

FailureOr<tensor::InsertSliceOp> getInsertSliceForLoopResult(scf::ForOp loop,
                                                             OpResult result) {
  if (result.getOwner() != loop.getOperation())
    return failure();
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield)
    return failure();
  return yield.getOperand(result.getResultNumber())
      .getDefiningOp<tensor::InsertSliceOp>();
}

FailureOr<tensor::ParallelInsertSliceOp>
getParallelInsertSliceForLoopResult(scf::ForallOp loop, OpResult result) {
  if (result.getOwner() != loop.getOperation())
    return failure();
  BlockArgument bbArg = loop.getTiedBlockArgument(result);
  SmallVector<Operation *> combiningOps = loop.getCombiningOps(bbArg);
  if (combiningOps.size() != 1)
    return failure();
  return dyn_cast<tensor::ParallelInsertSliceOp>(combiningOps.front());
}

linalg::MapOp cloneMapOnTile(RewriterBase &rewriter, linalg::MapOp sourceMap,
                             Value inputTile, Value initTile, Location loc) {
  auto cloned = linalg::MapOp::create(
      rewriter, loc, ValueRange{inputTile}, initTile,
      [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
        Block &oldBlock = sourceMap.getMapper().front();
        IRMapping mapping;
        for (auto [oldArg, newArg] :
             llvm::zip_equal(oldBlock.getArguments(), newArgs))
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

  auto loopLike = dyn_cast<LoopLikeOpInterface>(loops.front());
  if (!loopLike)
    return emitSilenceableFailure(
        transform, "expected an scf.for or scf.forall containing loop");

  Value consumerInput = map.getInputs().front();
  auto producerResult = dyn_cast<OpResult>(consumerInput);
  if (!producerResult)
    return emitSilenceableFailure(
        transform, "consumer input must be produced by the containing loop");
  Operation *producerContainer = producerResult.getOwner();
  if (producerContainer != loopLike.getOperation() &&
      !isAncestor(producerContainer, loopLike.getOperation()))
    return emitSilenceableFailure(
        transform,
        "containing loop must be nested under the consumer input producer");

  linalg::MapOp fused;
  if (auto forLoop = dyn_cast<scf::ForOp>(loopLike.getOperation())) {
    FailureOr<tensor::InsertSliceOp> maybeInsert =
        getInsertSliceForLoopResult(forLoop, producerResult);
    if (failed(maybeInsert) || !*maybeInsert)
      return emitSilenceableFailure(
          transform,
          "expected the loop result consumed by the map to be yielded via "
          "tensor.insert_slice");

    auto insert = *maybeInsert;
    rewriter.setInsertionPoint(insert);
    auto initTile = tensor::ExtractSliceOp::create(
        rewriter, insert.getLoc(), insert.getSource().getType(),
        insert.getDest(), insert.getMixedOffsets(), insert.getMixedSizes(),
        insert.getMixedStrides());
    fused = cloneMapOnTile(rewriter, map, insert.getSource(),
                           initTile.getResult(), map.getLoc());

    rewriter.modifyOpInPlace(insert, [&]() {
      insert.getSourceMutable().assign(fused.getResult().front());
    });
  } else {
    auto forallLoop = cast<scf::ForallOp>(loopLike.getOperation());
    FailureOr<tensor::ParallelInsertSliceOp> maybeInsert =
        getParallelInsertSliceForLoopResult(forallLoop, producerResult);
    if (failed(maybeInsert) || !*maybeInsert)
      return emitSilenceableFailure(
          transform,
          "expected the loop result consumed by the map to be combined via "
          "tensor.parallel_insert_slice");

    auto insert = *maybeInsert;
    rewriter.setInsertionPoint(insert->getParentOp());
    auto initTile = tensor::ExtractSliceOp::create(
        rewriter, insert.getLoc(), insert.getSource().getType(),
        insert.getDest(), insert.getMixedOffsets(), insert.getMixedSizes(),
        insert.getMixedStrides());
    fused = cloneMapOnTile(rewriter, map, insert.getSource(),
                           initTile.getResult(), map.getLoc());

    rewriter.modifyOpInPlace(insert, [&]() {
      insert.getSourceMutable().assign(fused.getResult().front());
    });
  }

  rewriter.replaceOp(map, consumerInput);

  transformResults.set(getOperation()->getResult(0),
                       ArrayRef<Operation *>{fused.getOperation()});
  transformResults.set(getOperation()->getResult(1),
                       ArrayRef<Operation *>{loopLike.getOperation()});
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
  return {MLIR_PLUGIN_API_VERSION, "LinalgExtTransformPlugin",
          LLVM_VERSION_STRING, [](mlir::DialectRegistry *registry) {
            linalg_ext::registerLinalgExtTransformExtension(*registry);
          }};
}
