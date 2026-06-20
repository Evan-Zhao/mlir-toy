#include "TA/TADialect.h"
#include "TA/TAPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace ta {

using namespace mlir;

namespace {

struct FunctionAxisInfo {};

class FunctionAxisDiscovery {
public:
  FailureOr<FunctionAxisInfo> run(RankedTensorType resultType) { return FunctionAxisInfo{}; }
};

class FunctionEmitter {
public:
  LogicalResult run() { return success(); }
};

struct ImportLinalgToTAPass
    : public PassWrapper<ImportLinalgToTAPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportLinalgToTAPass)

  StringRef getArgument() const final { return "linalg-to-ta"; }
  StringRef getDescription() const final {
    return "Import supported linalg.generic tensor dataflow into the ta dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TADialect, affine::AffineDialect, func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (func.empty())
      return;

    auto returnOp = dyn_cast<func::ReturnOp>(func.front().getTerminator());
    if (!returnOp || returnOp.getNumOperands() == 0)
      return;
    if (returnOp.getNumOperands() != 1) {
      func.emitOpError("ta importer currently expects one function result");
      signalPassFailure();
      return;
    }

    auto resultType = dyn_cast<RankedTensorType>(func.getResultTypes().front());
    if (!resultType)
      return;

    FunctionAxisDiscovery discovery;
    FailureOr<FunctionAxisInfo> axisInfo = discovery.run(resultType);
    if (failed(axisInfo)) {
      signalPassFailure();
      return;
    }

    FunctionEmitter emitter;
    if (failed(emitter.run())) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

void registerLinalgToTAPass() { PassRegistration<ImportLinalgToTAPass>(); }

} // namespace ta
