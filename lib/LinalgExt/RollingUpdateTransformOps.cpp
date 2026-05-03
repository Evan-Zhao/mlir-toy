#include "LinalgExt/LinalgExtTransformOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <deque>
#include <llvm/Support/raw_ostream.h>
#include <optional>

using namespace mlir;

namespace {

bool isReductionLike(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  return llvm::any_of(linalgOp.getIteratorTypesArray(), [](auto iteratorType) {
    return iteratorType == utils::IteratorType::reduction;
  });
}

LogicalResult verifyElementwiseMap(linalg::MapOp map) {
  if (map.getNumDpsInits() != 1)
    return failure();
  if (map->getNumResults() != 1)
    return failure();
  return success();
}

LogicalResult verifyElementwiseGeneric(linalg::GenericOp generic) {
  if (generic.getNumDpsInits() != 1)
    return failure();
  if (generic->getNumResults() != 1)
    return failure();
  if (!generic.isAllParallelLoops())
    return failure();
  if (!generic.hasPureTensorSemantics())
    return failure();
  if (!llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isProjectedPermutation(); }))
    return failure();
  if (!generic.getIndexingMapsArray().back().isIdentity())
    return failure();
  return success();
}

LogicalResult verifyForceFusibleElementwiseOp(Operation *op) {
  if (auto map = dyn_cast<linalg::MapOp>(op))
    return verifyElementwiseMap(map);
  if (auto generic = dyn_cast<linalg::GenericOp>(op))
    return verifyElementwiseGeneric(generic);
  return failure();
}

SmallVector<Value> getElementwiseInputs(Operation *op) {
  if (auto map = dyn_cast<linalg::MapOp>(op))
    return llvm::to_vector(map.getInputs());
  return llvm::to_vector(cast<linalg::GenericOp>(op).getInputs());
}

Value getElementwiseInit(Operation *op) {
  if (auto map = dyn_cast<linalg::MapOp>(op))
    return map.getDpsInits().front();
  return cast<linalg::GenericOp>(op).getDpsInits().front();
}

SmallVector<AffineMap> getElementwiseInputMaps(Operation *op) {
  if (auto map = dyn_cast<linalg::MapOp>(op)) {
    auto resultType = cast<RankedTensorType>(map->getResult(0).getType());
    SmallVector<AffineMap> maps(map.getInputs().size(),
                                AffineMap::getMultiDimIdentityMap(
                                    resultType.getRank(), op->getContext()));
    return maps;
  }
  auto generic = cast<linalg::GenericOp>(op);
  SmallVector<AffineMap> maps;
  maps.reserve(generic.getNumDpsInputs());
  for (unsigned i = 0; i < generic.getNumDpsInputs(); ++i)
    maps.push_back(generic.getIndexingMapsArray()[i]);
  return maps;
}

RankedTensorType inferTensorTypeFromMixedSizes(ArrayRef<OpFoldResult> sizes,
                                               Type elementType) {
  SmallVector<int64_t> shape;
  shape.reserve(sizes.size());
  for (OpFoldResult size : sizes) {
    auto maybeConst = getConstantIntValue(size);
    shape.push_back(maybeConst ? *maybeConst : ShapedType::kDynamic);
  }
  return RankedTensorType::get(shape, elementType);
}

void cloneSingleRegionBody(OpBuilder &builder, Location nestedLoc,
                           Block &oldBlock, ValueRange newArgs) {
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
}

Operation *cloneElementwiseOnTiles(RewriterBase &rewriter, Operation *sourceOp,
                                   ValueRange tiledInputs, Value initTile,
                                   Location loc) {
  if (auto map = dyn_cast<linalg::MapOp>(sourceOp)) {
    return linalg::MapOp::create(
               rewriter, loc, tiledInputs, initTile,
               [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
                 cloneSingleRegionBody(builder, nestedLoc,
                                       map.getMapper().front(), newArgs);
               })
        .getOperation();
  }

  auto generic = cast<linalg::GenericOp>(sourceOp);
  return linalg::GenericOp::create(
             rewriter, loc, TypeRange{initTile.getType()}, tiledInputs,
             ValueRange{initTile}, generic.getIndexingMapsArray(),
             generic.getIteratorTypesArray(),
             [&](OpBuilder &builder, Location nestedLoc, ValueRange newArgs) {
               cloneSingleRegionBody(builder, nestedLoc,
                                     generic->getRegion(0).front(), newArgs);
             })
      .getOperation();
}

SmallVector<OpFoldResult> projectByMap(ArrayRef<OpFoldResult> values,
                                       AffineMap map) {
  SmallVector<OpFoldResult> projected;
  projected.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    projected.push_back(values[dimExpr.getPosition()]);
  }
  return projected;
}

using OpsT = SmallVector<Operation *>;

std::optional<std::pair<OpsT, OpsT>> collectRollingFrontierPath(Operation *op) {
  OpsT elementwiseOps, frontierOps;
  SmallPtrSet<Operation *, 8> visitedOps;
  std::deque<std::pair<Operation *, bool>> worklist;
  worklist.push_back({op, false});

  while (!worklist.empty()) {
    auto [currentOp, foundReduction] = worklist.front();
    worklist.pop_front();
    if (visitedOps.contains(currentOp))
      continue;
    visitedOps.insert(currentOp);

    if (foundReduction) {
      frontierOps.push_back(currentOp);
      continue;
    }

    if (currentOp != op)
      elementwiseOps.push_back(currentOp);

    bool hasUsers = false;
    for (Value result : currentOp->getOpResults()) {
      for (auto user : result.getUsers()) {
        worklist.push_back({user, isReductionLike(user)});
        hasUsers = true;
      }
    }
    if (!hasUsers) {
      currentOp->emitRemark("path ends here without seeing a reduction");
      return std::nullopt;
    }
  }
  return {{elementwiseOps, frontierOps}};
}

struct LoopResultLocalValue {
  Value value;
  tensor::InsertSliceOp insert;
};

} // namespace

namespace mlir {
namespace transform {

DiagnosedSilenceableFailure LinalgExtRollingUpdateFwdFrontierOp::apply(
    transform::TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  SmallVector<Operation *> producers =
      llvm::to_vector(state.getPayloadOps(getProducerOp()));
  if (producers.size() != 1)
    return emitSilenceableFailure(
        transform,
        "expected exactly one producer payload op in the transform handle");

  auto frontierResult = collectRollingFrontierPath(producers.front());
  if (!frontierResult)
    return emitSilenceableFailure(
        transform, "target has no closed rolling update frontier");
  auto [elemwiseOps, frontierOps] = *frontierResult;
  transformResults.set(getOperation()->getResult(0), elemwiseOps);
  transformResults.set(getOperation()->getResult(1), frontierOps);
  return DiagnosedSilenceableFailure::success();
}

template <typename T>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const SmallVector<T> &values) {
  bool first = true;
  os << '[';
  for (const T &value : values) {
    if (first)
      first = false;
    else
      os << ", ";
    os << value;
  }
  os << ']';
  return os;
}

DiagnosedSilenceableFailure LinalgExtForceFuseElemwiseChainIntoLoopOp::apply(
    transform::TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());
  SmallVector<Operation *> elemwiseOps =
      llvm::to_vector(state.getPayloadOps(getElemwiseChainOps()));
  SmallVector<Operation *> loops =
      llvm::to_vector(state.getPayloadOps(getStreamingLoop()));

  if (elemwiseOps.empty())
    return emitSilenceableFailure(
        transform, "expected at least one elementwise payload op");
  if (loops.size() != 1)
    return emitSilenceableFailure(
        transform, "expected exactly one streaming loop payload op");

  auto currentLoop = dyn_cast<scf::ForOp>(loops.front());
  if (!currentLoop)
    return emitSilenceableFailure(
        transform, "expected the streaming loop to be an scf.for");

  for (Operation *elemwiseOp : elemwiseOps)
    if (failed(verifyForceFusibleElementwiseOp(elemwiseOp)))
      return emitSilenceableFailure(
          transform, "expected every op in the chain to be an elementwise "
                     "linalg.map or linalg.generic with one tensor result");

  // Step 1. Each elemwise op to fuse may introduce extra loop-carried values.
  // Rebuild the streaming loop once with extra iter_arg needed to hold these
  // values.
  // 1.1. Enlist these values and clone the loop.
  unsigned oldNumResults = currentLoop.getNumResults();
  SmallVector<Value> newInitArgs = llvm::to_vector(currentLoop.getInitArgs());
  for (Operation *elemwiseOp : elemwiseOps) {
    newInitArgs.push_back(getElementwiseInit(elemwiseOp));
  }
  rewriter.setInsertionPoint(currentLoop);
  auto newLoop = scf::ForOp::create(
      rewriter, currentLoop.getLoc(), currentLoop.getLowerBound(),
      currentLoop.getUpperBound(), currentLoop.getStep(), newInitArgs);

  // 1.2. Set up old value to new value mapping for the loop induction variable
  // and region iter args.
  auto *newBody = newLoop.getBody();
  auto defaultYield = cast<scf::YieldOp>(newBody->getTerminator());
  IRMapping mapping;
  mapping.map(currentLoop.getInductionVar(), newLoop.getInductionVar());
  for (auto [index, oldArg] : llvm::enumerate(currentLoop.getRegionIterArgs()))
    mapping.map(oldArg, newLoop.getRegionIterArgs()[index]);

  // 1.3. Clone the loop body except the yield operation.
  rewriter.setInsertionPoint(newBody, Block::iterator(defaultYield));
  for (Operation &op : currentLoop.getBody()->without_terminator())
    rewriter.clone(op, mapping);
  llvm::errs() << "newLoop after cloning body: " << *newLoop << "\n";

  auto oldYield = cast<scf::YieldOp>(currentLoop.getBody()->getTerminator());
  SmallVector<Value> clonedYieldOperands;
  clonedYieldOperands.reserve(oldYield.getNumOperands());
  for (Value operand : oldYield.getOperands())
    clonedYieldOperands.push_back(mapping.lookupOrDefault(operand));

  SmallVector<LoopResultLocalValue> loopLocalValues;
  loopLocalValues.reserve(oldNumResults);
  for (Value yielded : oldYield.getOperands()) {
    LoopResultLocalValue localValue{mapping.lookupOrDefault(yielded), nullptr};
    if (auto oldInsert = yielded.getDefiningOp<tensor::InsertSliceOp>()) {
      auto newInsert = mapping.lookupOrDefault(oldInsert.getResult())
                           .getDefiningOp<tensor::InsertSliceOp>();
      if (!newInsert)
        return emitSilenceableFailure(
            transform,
            "expected cloned tensor.insert_slice to remain available in the "
            "rebuilt loop body");
      localValue.value = newInsert.getSource();
      localValue.insert = newInsert;
    }
    loopLocalValues.push_back(localValue);
  }
  rewriter.setInsertionPoint(newBody, Block::iterator(defaultYield));

  SmallVector<Value> firstInputs = getElementwiseInputs(elemwiseOps.front());
  std::optional<unsigned> anchorLoopResultIndex;
  for (Value input : firstInputs) {
    auto result = dyn_cast<OpResult>(input);
    if (!result || result.getOwner() != currentLoop.getOperation())
      continue;
    if (loopLocalValues[result.getResultNumber()].insert) {
      anchorLoopResultIndex = result.getResultNumber();
      break;
    }
  }
  if (!anchorLoopResultIndex)
    return emitSilenceableFailure(
        transform, "expected the first elementwise op to consume a tensor loop "
                   "result materialized via tensor.insert_slice");

  auto anchorInsert = loopLocalValues[*anchorLoopResultIndex].insert;
  SmallVector<OpFoldResult> anchorOffsets = anchorInsert.getMixedOffsets();
  SmallVector<OpFoldResult> anchorSizes = anchorInsert.getMixedSizes();
  SmallVector<OpFoldResult> anchorStrides = anchorInsert.getMixedStrides();

  // Step 2. Clone each elementwise op under the rebuilt loop in the original
  // producer-to-consumer order, using previously cloned sidecar tiles when the
  // chain depends on earlier elementwise results.
  SmallVector<Operation *> sidecarOps;
  sidecarOps.reserve(elemwiseOps.size());
  llvm::DenseMap<Value, Value> loopResultValues;
  for (auto [index, result] : llvm::enumerate(currentLoop.getResults()))
    loopResultValues[result] = newLoop.getResults()[index];
  llvm::DenseMap<Value, Value> sidecarTileValues;
  SmallVector<Value> sidecarYieldOperands;
  sidecarYieldOperands.reserve(elemwiseOps.size());

  for (auto [index, elemwiseOp] : llvm::enumerate(elemwiseOps)) {
    SmallVector<Value> remappedInputs = getElementwiseInputs(elemwiseOp);
    for (Value &input : remappedInputs) {
      if (auto mapped = sidecarTileValues.find(input);
          mapped != sidecarTileValues.end()) {
        input = mapped->second;
        continue;
      }
      if (auto mapped = loopResultValues.find(input);
          mapped != loopResultValues.end())
        input = mapped->second;
    }

    SmallVector<AffineMap> inputMaps = getElementwiseInputMaps(elemwiseOp);
    SmallVector<Value> tiledInputs;
    tiledInputs.reserve(remappedInputs.size());
    for (auto [input, inputMap] : llvm::zip_equal(remappedInputs, inputMaps)) {
      if (!isa<RankedTensorType>(input.getType())) {
        tiledInputs.push_back(input);
        continue;
      }

      auto projectedOffsets = projectByMap(anchorOffsets, inputMap);
      auto projectedSizes = projectByMap(anchorSizes, inputMap);
      SmallVector<OpFoldResult> projectedStrides(projectedOffsets.size(),
                                                 rewriter.getIndexAttr(1));
      auto projectedType = inferTensorTypeFromMixedSizes(
          projectedSizes, cast<ShapedType>(input.getType()).getElementType());

      if (auto inputResult = dyn_cast<OpResult>(input);
          inputResult && inputResult.getOwner() == newLoop.getOperation()) {
        LoopResultLocalValue localValue =
            loopLocalValues[inputResult.getResultNumber()];
        if (localValue.value.getType() != projectedType)
          return emitSilenceableFailure(
              transform,
              "expected loop-carried operand tile shape to match the "
              "projected elementwise tile shape");
        tiledInputs.push_back(localValue.value);
        continue;
      }

      tiledInputs.push_back(tensor::ExtractSliceOp::create(
          rewriter, elemwiseOp->getLoc(), projectedType, input,
          projectedOffsets, projectedSizes, projectedStrides));
    }

    auto sidecarResultType =
        dyn_cast<RankedTensorType>(elemwiseOp->getResult(0).getType());
    if (!sidecarResultType)
      return emitSilenceableFailure(
          transform, "expected the elementwise op to return a ranked tensor");
    auto sidecarTileType = inferTensorTypeFromMixedSizes(
        anchorSizes, sidecarResultType.getElementType());
    Value sidecarInitTile = tensor::ExtractSliceOp::create(
        rewriter, elemwiseOp->getLoc(), sidecarTileType,
        newLoop.getRegionIterArgs()[oldNumResults + index], anchorOffsets,
        anchorSizes, anchorStrides);

    Operation *sidecarOp =
        cloneElementwiseOnTiles(rewriter, elemwiseOp, tiledInputs,
                                sidecarInitTile, elemwiseOp->getLoc());
    auto sidecarInsert = tensor::InsertSliceOp::create(
        rewriter, elemwiseOp->getLoc(), sidecarOp->getResult(0),
        newLoop.getRegionIterArgs()[oldNumResults + index], anchorOffsets,
        anchorSizes, anchorStrides);

    llvm::errs() << "Sidecar op: " << *sidecarOp << "\n";
    llvm::errs() << "sidecarInsert: " << sidecarInsert << "\n";
    sidecarOps.push_back(sidecarOp);
    sidecarTileValues[elemwiseOp->getResult(0)] = sidecarOp->getResult(0);
    sidecarYieldOperands.push_back(sidecarInsert.getResult());
  }

  // Step 3. Publish all sidecar tensors as extra loop results, then replace
  // the old loop while leaving the original out-of-loop chain untouched.
  SmallVector<Value> newYieldOperands = clonedYieldOperands;
  newYieldOperands.append(sidecarYieldOperands.begin(),
                          sidecarYieldOperands.end());
  llvm::errs() << "Yield operands: " << newYieldOperands << "\n";
  rewriter.setInsertionPoint(defaultYield);
  scf::YieldOp::create(rewriter, currentLoop.getLoc(), newYieldOperands);
  llvm::errs() << "Loop body with yield: " << *newLoop << "\n";

  rewriter.eraseOp(defaultYield);
  rewriter.replaceOp(currentLoop,
                     newLoop.getResults().take_front(oldNumResults));
  transformResults.set(getOperation()->getResult(0), sidecarOps);
  transformResults.set(getOperation()->getResult(1),
                       ArrayRef<Operation *>{newLoop.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace mlir
