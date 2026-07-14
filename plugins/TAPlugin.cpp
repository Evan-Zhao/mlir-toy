#include "TA/TADialect.h"
#include "TA/TAPasses.h"
#include "TA/TATransformExtension.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Compiler.h"

namespace {

void registerTAPasses() {
  ta::registerTAPasses();
  mlir::stablehlo::registerStablehloLinalgTransformsPasses();
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TADialectPlugin", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<ta::TADialect, mlir::stablehlo::StablehloDialect>();
            // transform.apply_registered_pass constructs its nested pass
            // manager after execution has started, too late for that manager
            // to load newly discovered dependent dialects safely. Preload the
            // StableHLO-to-Linalg pass dependencies when StableHLO is loaded.
            registry->addExtension(
                +[](mlir::MLIRContext *context, mlir::stablehlo::StablehloDialect *) {
                  context->loadDialect<mlir::bufferization::BufferizationDialect,
                                       mlir::complex::ComplexDialect, mlir::linalg::LinalgDialect,
                                       mlir::math::MathDialect, mlir::memref::MemRefDialect,
                                       mlir::scf::SCFDialect, mlir::shape::ShapeDialect,
                                       mlir::sparse_tensor::SparseTensorDialect>();
                });
            ta::registerTATransformExtension(*registry);
            registerTAPasses();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TAPassPlugin", LLVM_VERSION_STRING,
          []() { registerTAPasses(); }};
}
