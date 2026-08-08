#include "HTile/HTileTransformOps.h"

#include "LoopTr/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CheckedArithmetic.h"

using namespace mlir;
using bufferization::ToTensorOp;

#define BAIL(message) return emitSilenceableFailure(transform, message)

namespace mlir::transform {
namespace {

FailureOr<scf::ForallOp> rebuildForallWithoutOutputs(RewriterBase &rewriter,
                                                     scf::ForallOp forallOp) {
  for (BlockArgument outputArg : forallOp.getRegionOutArgs()) {
    if (!outputArg.use_empty()) {
      outputArg.user_begin()->emitRemark() << "one of the remaining uses here";
      return forallOp.emitError() << "unsupported remaining use of scf.forall shared_out";
    }
  }
  for (OpResult result : forallOp->getResults()) {
    if (!result.use_empty())
      return forallOp.emitError()
             << "expected returned scf.forall result to have no remaining uses";
  }

  SmallVector<Value> oldOutputs = llvm::to_vector(forallOp.getOutputs());
  rewriter.setInsertionPoint(forallOp);
  auto newForall = scf::ForallOp::create(
      rewriter, forallOp.getLoc(), forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), ValueRange{}, forallOp.getMapping(),
      [&](OpBuilder &nestedBuilder, Location, ValueRange bbArgs) {
        SmallVector<Value> replacements = llvm::to_vector(bbArgs);
        replacements.append(oldOutputs.begin(), oldOutputs.end());
        rewriter.mergeBlocks(forallOp.getBody(), nestedBuilder.getBlock(), replacements);
      });
  rewriter.eraseOp(forallOp);

  for (Value oldOutput : oldOutputs) {
    Operation *def = oldOutput.getDefiningOp();
    if (def && def->use_empty())
      rewriter.eraseOp(def);
  }
  return newForall;
}

struct TensorToBufferMap {
  void mapTensorToMemref(Value tensor, Value memref) { tensorToMemref[tensor] = memref; }

  Value getTensorMemref(Value tensor) {
    if (auto it = tensorToMemref.find(tensor); it != tensorToMemref.end()) {
      return it->second;
    }
    return nullptr;
  }

  Value getOrCreateTensorMemrefForRead(RewriterBase &rewriter, scf::ForallOp forall, Value tensor) {
    if (Value buffer = getTensorMemref(tensor)) {
      return buffer;
    }

    auto tensorType = cast<RankedTensorType>(tensor.getType());
    auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forall);
    Value buffer = bufferization::ToBufferOp::create(rewriter, tensor.getLoc(), memrefType, tensor,
                                                     /*readOnly=*/true);
    mapTensorToMemref(tensor, buffer);
    return buffer;
  }

  DenseMap<Value, Value> tensorToMemref;
};

LogicalResult materializeStoreForInsertSlice(RewriterBase &rewriter,
                                             tensor::ParallelInsertSliceOp insert, Value buffer) {
  if (!insert.hasUnitStride())
    return insert.emitError() << "unsupported non-unit tensor.parallel_insert_slice stride";
  SmallVector<Value> offsets =
      getValueOrCreateConstantIndexOp(rewriter, insert.getLoc(), insert.getMixedOffsets());
  htile::StoreOp::create(rewriter, insert.getLoc(), insert.getSource(), buffer, offsets);
  return success();
}

LogicalResult materializeStoreForInsertSlice(RewriterBase &rewriter,
                                             htile::MaskedParallelInsertSliceOp insert,
                                             Value buffer) {
  if (insert.getSourceType().getRank() != insert.getDestType().getRank())
    return insert.emitError() << "unsupported rank-reduced masked publication";
  if (!insert.hasUnitStride())
    return insert.emitError() << "unsupported non-unit htile.masked_parallel_insert_slice stride";
  SmallVector<Value> offsets =
      getValueOrCreateConstantIndexOp(rewriter, insert.getLoc(), insert.getMixedOffsets());
  htile::StoreOp::create(rewriter, insert.getLoc(), insert.getSource(), buffer, offsets,
                         insert.getMask());
  return success();
}

FailureOr<Value> materializeLoadForExtractSlice(OpBuilder &builder, tensor::ExtractSliceOp extract,
                                                Value buffer, bool emitHTileLoad) {
  if (emitHTileLoad) {
    if (!extract.hasUnitStride())
      return extract.emitError() << "unsupported non-unit tensor.extract_slice stride";
    SmallVector<Value> offsets =
        getValueOrCreateConstantIndexOp(builder, extract.getLoc(), extract.getMixedOffsets());
    auto loadOp =
        htile::LoadOp::create(builder, extract.getLoc(), extract.getResultType(), buffer, offsets);
    return loadOp.getResult();
  } else {
    auto sourceMemrefType = cast<MemRefType>(buffer.getType());
    auto resultType = cast<RankedTensorType>(extract.getResultType());
    auto subviewType = memref::SubViewOp::inferRankReducedResultType(
        resultType.getShape(), sourceMemrefType, extract.getMixedOffsets(), extract.getMixedSizes(),
        extract.getMixedStrides());
    Value subview = memref::SubViewOp::create(builder, extract.getLoc(), subviewType, buffer,
                                              extract.getMixedOffsets(), extract.getMixedSizes(),
                                              extract.getMixedStrides());
    auto loadOp =
        ToTensorOp::create(builder, extract.getLoc(), resultType, subview, /*restrict=*/true,
                           /*writable=*/true);
    return loadOp.getResult();
  }
}

Value materializeLoadForWholeTensor(OpBuilder &builder, Location loc, RankedTensorType tensorType,
                                    Value buffer, bool emitHTileLoad) {
  if (emitHTileLoad) {
    SmallVector<Value> offsets;
    offsets.reserve(tensorType.getRank());
    for (int64_t i = 0, e = tensorType.getRank(); i < e; ++i)
      offsets.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
    return htile::LoadOp::create(builder, loc, tensorType, buffer, offsets);
  } else {
    return ToTensorOp::create(builder, loc, tensorType, buffer,
                              /*restrict=*/true, /*writable=*/true);
  }
}

LogicalResult bufferizeForallResults(RewriterBase &rewriter, ArrayRef<scf::ForallOp> forallOps,
                                     TensorToBufferMap &map) {
  OpBuilder::InsertionGuard guard(rewriter);
  for (auto forall : forallOps) {
    // Create a memref buffer for each result tensor and map it to the tensor.
    for (OpResult result : forall->getResults()) {
      size_t resultNum = result.getResultNumber();
      auto tensorType = dyn_cast<RankedTensorType>(result.getType());
      if (!tensorType)
        continue;
      if (!tensorType.hasStaticShape())
        return forall.emitError() << "result # " << resultNum
                                  << " of this forall is a ranked tensor with dynamic shape";
      // Allocate the buffer before the forall loop.
      rewriter.setInsertionPoint(forall);
      auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      auto buffer = memref::AllocOp::create(rewriter, forall.getLoc(), memrefType);
      map.mapTensorToMemref(result, buffer);
      map.mapTensorToMemref(forall.getTiedBlockArgument(result), buffer);

      // Preserve the initial value of the tensor by copying it into the buffer before the forall.
      Value initialTensor = forall.getOutputs()[resultNum];
      if (!initialTensor.getDefiningOp<tensor::EmptyOp>()) {
        Value initialBuffer = map.getOrCreateTensorMemrefForRead(rewriter, forall, initialTensor);
        rewriter.setInsertionPoint(forall);
        memref::CopyOp::create(rewriter, forall.getLoc(), initialBuffer, buffer);
      }
    }

    // Materialize each slice publication into a memref store. Insert the stores
    // before the forall terminator rather than inside the in_parallel region.
    rewriter.setInsertionPoint(forall.getTerminator());
    for (Operation &combiningOp : llvm::make_early_inc_range(forall.getTerminator())) {
      if (auto insert = dyn_cast<tensor::ParallelInsertSliceOp>(&combiningOp)) {
        Value buffer = map.getTensorMemref(insert.getDest());
        if (!buffer)
          return insert.emitError()
                 << "this op doesn't publish to a tensor-typed block argument of the loop";
        if (failed(materializeStoreForInsertSlice(rewriter, insert, buffer)))
          return failure();
        rewriter.eraseOp(insert);
        continue;
      }
      if (auto insert = dyn_cast<htile::MaskedParallelInsertSliceOp>(&combiningOp)) {
        Value buffer = map.getTensorMemref(insert.getDest());
        if (!buffer)
          return insert.emitError()
                 << "this op doesn't publish to a tensor-typed block argument of the loop";
        if (failed(materializeStoreForInsertSlice(rewriter, insert, buffer)))
          return failure();
        rewriter.eraseOp(insert);
        continue;
      }
      return combiningOp.emitError()
             << "expected forall in_parallel region to contain only supported slice publications";
    }
  }
  return success();
}

FailureOr<bool> materializeLoadsForTensorUsers(RewriterBase &rewriter, OpOperand &use,
                                               RankedTensorType tensorType, Value buffer,
                                               bool emitHTileLoad) {
  Operation *owner = use.getOwner();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(owner);
  if (isa<htile::LoadOp>(owner) && use.getOperandNumber() == 0) {
    // Another transform may have already materialized the desired tile load.
    // Retarget that load instead of reading the entire tensor first.
    use.set(buffer);
    return FailureOr<bool>(false); // Op not erased
  } else if (auto extract = dyn_cast<tensor::ExtractSliceOp>(owner)) {
    // If the use is a tensor.extract_slice, materialize a sliced read of the buffer.
    auto loaded = materializeLoadForExtractSlice(rewriter, extract, buffer, emitHTileLoad);
    if (failed(loaded))
      return failure();
    rewriter.replaceOp(extract, *loaded);
    return FailureOr<bool>(true); // Op erased
  } else {
    // Otherwise, fall back to a full read of the buffer.
    Value loaded =
        materializeLoadForWholeTensor(rewriter, owner->getLoc(), tensorType, buffer, emitHTileLoad);
    use.set(loaded);
    return FailureOr<bool>(false); // Op not erased
  }
}

LogicalResult bufferizeTensorReadInForalls(RewriterBase &rewriter,
                                           ArrayRef<scf::ForallOp> forallOps,
                                           TensorToBufferMap &map) {
  for (auto forall : forallOps) {
    DenseSet<Value> blockArgs;
    for (auto arg : forall.getBody()->getArguments()) {
      blockArgs.insert(arg);
    }
    WalkResult walkResult = forall.getBody()->walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        // Skip non-tensors, and tensors defined inside the loop body (not block arguments).
        Value tensor = operand.get();
        auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
        if (!tensorType)
          continue;
        bool isBlockArg = blockArgs.count(tensor);
        bool isInLoop = forall.getRegion().isAncestor(tensor.getParentRegion());
        if (isInLoop && !isBlockArg)
          continue;
        auto buffer = map.getOrCreateTensorMemrefForRead(rewriter, forall, tensor);
        // If this op is an extract_slice, it may be entirely removed.
        FailureOr<bool> erased =
            materializeLoadsForTensorUsers(rewriter, operand, tensorType, buffer,
                                           /*emitHTileLoad=*/true);
        if (failed(erased))
          return WalkResult::interrupt();
        if (*erased)
          return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
  }
  return success();
}

LogicalResult bufferizeForallResultUses(RewriterBase &rewriter, ArrayRef<scf::ForallOp> forallOps,
                                        TensorToBufferMap &map) {
  for (auto forall : forallOps) {
    for (OpOperand &use : llvm::make_early_inc_range(forall->getUses())) {
      Value tensor = use.get();
      Value buffer = map.getTensorMemref(tensor);
      if (!buffer)
        continue;
      auto tensorType = cast<RankedTensorType>(tensor.getType());
      if (failed(materializeLoadsForTensorUsers(rewriter, use, tensorType, buffer,
                                                /*emitHTileLoad=*/false)))
        return failure();
    }
  }
  return success();
}

std::string getRequestedKernelName(ArrayAttr kernelNames, size_t index) {
  if (kernelNames)
    return cast<StringAttr>(kernelNames[index]).getValue().str();
  return ("outlined_kernel_" + Twine(index)).str();
}

std::string getUniqueKernelName(Operation *symbolTableOp, StringRef baseName) {
  if (!SymbolTable::lookupSymbolIn(symbolTableOp, baseName))
    return baseName.str();

  unsigned uniquingCounter = 0;
  SmallString<32> name = SymbolTable::generateSymbolName<32>(
      baseName,
      [&](StringRef candidate) {
        return SymbolTable::lookupSymbolIn(symbolTableOp, candidate) != nullptr;
      },
      uniquingCounter);
  return name.str().str();
}

struct OutlinedKernel {
  scf::ForallOp forall;
  htile::KernelOp kernel;
  SmallVector<Value> operands;
  htile::LaunchFuncOp launch;
};

FailureOr<std::pair<SmallVector<Value>, SmallVector<int64_t>>>
getLoopIVsAndProgramBounds(RewriterBase &rewriter, scf::ForallOp forall) {
  auto lowerBounds = forall.getMixedLowerBound(), upperBounds = forall.getMixedUpperBound(),
       steps = forall.getMixedStep();
  Location loc = forall.getLoc();
  Type indexType = rewriter.getIndexType();

  size_t nDims = lowerBounds.size();
  SmallVector<int64_t> lowers, stepValues, logicalTripCounts;
  lowers.reserve(nDims);
  stepValues.reserve(nDims);
  logicalTripCounts.reserve(nDims);
  for (size_t index = 0; index < nDims; ++index) {
    std::optional<int64_t> maybeLower = getConstantIntValue(lowerBounds[index]),
                           maybeUpper = getConstantIntValue(upperBounds[index]),
                           maybeStep = getConstantIntValue(steps[index]);
    if (!maybeLower || !maybeUpper || !maybeStep)
      return forall.emitError() << "expected static lower/upper/step for forall dimension "
                                << index;
    if (*maybeStep <= 0)
      return forall.emitError() << "expected positive static step for forall dimension " << index;
    if (*maybeUpper < *maybeLower)
      return forall.emitError() << "expected upper bound to be >= lower bound for dimension "
                                << index;

    lowers.push_back(*maybeLower);
    stepValues.push_back(*maybeStep);
    int64_t distance = *maybeUpper - *maybeLower;
    logicalTripCounts.push_back((distance + *maybeStep - 1) / *maybeStep);
  }

  constexpr size_t maxProgramDimensions = 3;
  SmallVector<Value> normalizedIds(nDims);
  SmallVector<int64_t> programBounds;
  if (nDims <= maxProgramDimensions) {
    programBounds = logicalTripCounts;
    for (size_t index = 0; index < nDims; ++index)
      normalizedIds[index] = htile::ProgramIdOp::create(rewriter, loc, indexType, index);
  } else {
    // GPU launch grids have at most three dimensions. Flatten the leading
    // dimensions into axis 0 and recover their row-major logical IDs.
    size_t collapsedDims = nDims - (maxProgramDimensions - 1);
    int64_t collapsedBound = 1;
    for (int64_t tripCount : ArrayRef(logicalTripCounts).take_front(collapsedDims)) {
      std::optional<int64_t> product = llvm::checkedMul(collapsedBound, tripCount);
      if (!product)
        return forall.emitError("collapsed program bound overflows i64");
      collapsedBound = *product;
    }
    programBounds.push_back(collapsedBound);
    programBounds.append(logicalTripCounts.begin() + collapsedDims, logicalTripCounts.end());

    if (collapsedBound == 0) {
      for (size_t index = 0; index < collapsedDims; ++index)
        normalizedIds[index] = arith::ConstantIndexOp::create(rewriter, loc, 0);
    } else {
      Value remaining = htile::ProgramIdOp::create(rewriter, loc, indexType, 0);
      for (size_t index = collapsedDims; index-- > 1;) {
        Value bound = arith::ConstantIndexOp::create(rewriter, loc, logicalTripCounts[index]);
        normalizedIds[index] = arith::RemUIOp::create(rewriter, loc, remaining, bound);
        remaining = arith::DivUIOp::create(rewriter, loc, remaining, bound);
      }
      normalizedIds[0] = remaining;
    }
    for (size_t index = collapsedDims; index < nDims; ++index) {
      size_t programDimension = index - collapsedDims + 1;
      normalizedIds[index] = htile::ProgramIdOp::create(rewriter, loc, indexType, programDimension);
    }
  }

  SmallVector<Value> ids;
  ids.reserve(nDims);
  for (auto [index, normalizedId] : llvm::enumerate(normalizedIds)) {
    Value id = normalizedId;
    if (stepValues[index] != 1) {
      Value step = arith::ConstantIndexOp::create(rewriter, loc, stepValues[index]);
      id = arith::MulIOp::create(rewriter, loc, id, step);
    }
    if (lowers[index] != 0) {
      Value lower = arith::ConstantIndexOp::create(rewriter, loc, lowers[index]);
      id = arith::AddIOp::create(rewriter, loc, id, lower);
    }
    ids.push_back(id);
  }

  return std::make_pair(ids, programBounds);
}

std::optional<unsigned> getSourceFuncArgumentNumber(Value value) {
  if (auto toBuffer = value.getDefiningOp<bufferization::ToBufferOp>())
    value = toBuffer.getTensor();

  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return std::nullopt;
  Block *owner = blockArg.getOwner();
  if (!owner->isEntryBlock() || !isa<func::FuncOp>(owner->getParentOp()))
    return std::nullopt;
  return blockArg.getArgNumber();
}

FailureOr<SmallVector<Value>> legalizeKernelExternalValues(RewriterBase &rewriter,
                                                           htile::KernelOp kernel) {
  Region &region = kernel.getBody();
  Block &entryBlock = region.front();

  llvm::SetVector<Value> captures;
  kernel.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!region.isAncestor(operand.getParentRegion()))
        captures.insert(operand);
    }
  });

  // First-use order inside the kernel is not a stable ABI: e.g. ALiBi is read
  // before V even though it follows V in the host function signature. Prefer
  // source function argument order, then retain first-use order for allocations.
  SmallVector<Value> orderedCaptures(captures.begin(), captures.end());
  llvm::stable_sort(orderedCaptures, [](Value lhs, Value rhs) {
    std::optional<unsigned> lhsArg = getSourceFuncArgumentNumber(lhs);
    std::optional<unsigned> rhsArg = getSourceFuncArgumentNumber(rhs);
    if (lhsArg && rhsArg)
      return *lhsArg < *rhsArg;
    return lhsArg.has_value() && !rhsArg.has_value();
  });

  SmallVector<Value> operands;
  OpBuilder::InsertionGuard guard(rewriter);
  // Allow captures to be memrefs or arith.constant. If it's a constant, copy it into the kernel.
  for (Value capture : orderedCaptures) {
    if (isa<MemRefType>(capture.getType())) {
      BlockArgument arg = entryBlock.addArgument(capture.getType(), capture.getLoc());
      rewriter.replaceUsesWithIf(capture, arg, [&](OpOperand &use) {
        return region.isAncestor(use.getOwner()->getParentRegion());
      });
      operands.push_back(capture);
      continue;
    }

    Operation *def = capture.getDefiningOp();
    if (!isa_and_nonnull<arith::ConstantOp>(def))
      return kernel.emitError() << "unsupported non-memref kernel capture: " << capture;
    rewriter.setInsertionPointToStart(&entryBlock);
    Operation *cloned = rewriter.clone(*def);
    rewriter.replaceUsesWithIf(capture, cloned->getResult(0), [&](OpOperand &use) {
      return region.isAncestor(use.getOwner()->getParentRegion());
    });
  }

  return operands;
}

FailureOr<SmallVector<OutlinedKernel>> createKernelOps(RewriterBase &rewriter, Operation *hostOp,
                                                       ArrayRef<scf::ForallOp> forallOps,
                                                       ArrayAttr kernelNames) {
  Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(hostOp);
  if (!symbolTableOp)
    return hostOp->emitError() << "expected selected forall parent to have a symbol table";

  SmallVector<OutlinedKernel> kernels;
  kernels.reserve(forallOps.size());
  Operation *insertAfter = hostOp;
  for (auto [index, forallValue] : llvm::enumerate(forallOps)) {
    scf::ForallOp forall = forallValue;
    std::string requestedName = getRequestedKernelName(kernelNames, index);
    std::string kernelName = getUniqueKernelName(symbolTableOp, requestedName);

    // Create the kernel after the current hostOp (typically a func.func).
    rewriter.setInsertionPointAfter(insertAfter);
    auto kernel = htile::KernelOp::create(rewriter, forall.getLoc(), kernelName);
    Block *body = new Block();
    kernel.getBody().push_back(body);

    // Map the induction variables to the program IDs, then clone the forall body into the kernel.
    rewriter.setInsertionPointToStart(body);
    auto ivsAndProgramBounds = getLoopIVsAndProgramBounds(rewriter, forall);
    if (failed(ivsAndProgramBounds))
      return failure();
    auto [programIds, programBounds] = *ivsAndProgramBounds;
    kernel.setProgramBoundsAttr(DenseI64ArrayAttr::get(rewriter.getContext(), programBounds));
    IRMapping mapping;
    mapping.map(forall.getInductionVars(), programIds);
    cloneBlockWithoutTerminator(rewriter, *forall.getBody(), mapping);
    htile::ReturnOp::create(rewriter, forall.getLoc());

    // Check what values are used from the body of the kernel, and list them as operands for the
    // kernel. Only allow memrefs in the operands. Constants are copied into the kernel.
    FailureOr<SmallVector<Value>> operands = legalizeKernelExternalValues(rewriter, kernel);
    if (failed(operands))
      return failure();
    kernels.push_back(OutlinedKernel{.forall = forall,
                                     .kernel = kernel,
                                     .operands = std::move(*operands),
                                     .launch = htile::LaunchFuncOp()});
    insertAfter = kernel.getOperation();
  }

  return kernels;
}

FailureOr<func::FuncOp> validateSameParentFunc(ArrayRef<scf::ForallOp> forallOps) {
  func::FuncOp hostFunc;
  for (scf::ForallOp forall : forallOps) {
    auto parentFunc = dyn_cast<func::FuncOp>(forall->getParentOp());
    if (!parentFunc)
      return forall.emitError()
             << "expected selected scf.forall to be a top-level op directly inside func.func";
    if (!hostFunc)
      hostFunc = parentFunc;
    else if (parentFunc != hostFunc)
      return forall.emitError()
             << "expected all selected scf.forall ops to belong to the same func.func";
  }
  return hostFunc;
}

void createLaunchOpsAndEraseForalls(RewriterBase &rewriter,
                                    MutableArrayRef<OutlinedKernel> kernels) {
  for (OutlinedKernel &outlined : kernels) {
    rewriter.setInsertionPoint(outlined.forall);
    outlined.launch = htile::LaunchFuncOp::create(rewriter, outlined.forall.getLoc(),
                                                  outlined.kernel.getSymName(), outlined.operands);
    if (auto programBounds = outlined.kernel.getProgramBoundsAttr())
      outlined.launch->setAttr("program_bounds", programBounds);
    rewriter.eraseOp(outlined.forall);
  }
}

} // namespace

void HTileOutlineKernelsOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getForallsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure HTileOutlineKernelsOp::apply(TransformRewriter &rewriter,
                                                         TransformResults &results,
                                                         TransformState &state) {
  auto transform = cast<TransformOpInterface>(getOperation());

  SmallVector<scf::ForallOp> forallOps;
  for (Operation *op : state.getPayloadOps(getForalls())) {
    auto forall = dyn_cast<scf::ForallOp>(op);
    if (!forall) {
      op->emitError() << "expected scf.forall payload op";
      BAIL("expected all payload ops to be scf.forall");
    }
    forallOps.push_back(forall);
  }
  if (forallOps.empty())
    BAIL("expected at least one scf.forall payload op");
  FailureOr<func::FuncOp> hostFunc = validateSameParentFunc(forallOps);
  if (failed(hostFunc))
    BAIL("failed to validate selected scf.forall ops");
  Operation *hostOp = hostFunc->getOperation();

  auto kernelNames = (*this)->getAttrOfType<ArrayAttr>("kernel_names");
  if (kernelNames && kernelNames.size() != forallOps.size())
    BAIL("expected kernel_names length to match payload op count");

  TensorToBufferMap bufferMap;
  // Convert forall tensor result to memrefs, and in-loop writes of results to memref writes.
  if (failed(bufferizeForallResults(rewriter, forallOps, bufferMap)))
    BAIL("failed to bufferize forall results");
  // Convert reads of any tensor in foralls to memref reads: get a memref for the tensor being used,
  // and read from the memref instead.
  if (failed(bufferizeTensorReadInForalls(rewriter, forallOps, bufferMap)))
    BAIL("failed to bufferize tensor reads in foralls");
  // Convert any remaining uses of forall result tensors to memref reads, such as func.func return.
  if (failed(bufferizeForallResultUses(rewriter, forallOps, bufferMap)))
    BAIL("failed to bufferize forall result uses");
  // Remove all results and shared out arguments from every forall op.
  for (auto &forall : forallOps) {
    auto newForall = rebuildForallWithoutOutputs(rewriter, forall);
    if (failed(newForall))
      BAIL("failed to rebuild forall without outputs");
    forall = *newForall;
  }
  // Create an htile.kernel op for each forall op, with the boundary memrefs as operands.
  FailureOr<SmallVector<OutlinedKernel>> kernels =
      createKernelOps(rewriter, hostOp, forallOps, kernelNames);
  if (failed(kernels))
    BAIL("failed to create htile.kernel ops");
  createLaunchOpsAndEraseForalls(rewriter, *kernels);

  SmallVector<Operation *> launchOps =
      llvm::map_to_vector(*kernels, [](auto &kernel) { return kernel.launch.getOperation(); });
  SmallVector<Operation *> kernelOps =
      llvm::map_to_vector(*kernels, [](auto &kernel) { return kernel.kernel.getOperation(); });
  results.set(getOperation()->getResult(0), launchOps);
  results.set(getOperation()->getResult(1), kernelOps);

  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::transform
