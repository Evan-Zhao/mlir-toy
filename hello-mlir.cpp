#include "HTile/HTileDialect.h"
#include "HTile/HTileOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

namespace {

struct HelloWorldPass
    : public mlir::PassWrapper<HelloWorldPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HelloWorldPass)

  llvm::StringRef getArgument() const final { return "hello-world"; }
  llvm::StringRef getDescription() const final {
    return "Print hello world for each htile.hello operation.";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    module.walk([](htile::HelloOp op) {
      llvm::outs() << "Hello from HelloWorldPass visiting "
                   << op->getName().getStringRef() << "\n";
    });
  }
};

} // namespace

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<htile::HTileDialect>();

  mlir::OpBuilder builder(&context);

  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  module->setAttr("hello.message", builder.getStringAttr("Hello from MLIR 20"));
  builder.setInsertionPointToStart(module.getBody());
  htile::HelloOp::create(builder, builder.getUnknownLoc());

  if (failed(mlir::verify(module))) {
    llvm::errs() << "Generated module failed verification\n";
    return 1;
  }

  mlir::PassManager passManager(&context);
  passManager.addPass(std::make_unique<HelloWorldPass>());
  if (failed(passManager.run(module))) {
    llvm::errs() << "Pass pipeline failed\n";
    return 1;
  }

  llvm::outs() << "\nFinal IR:\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";

  return 0;
}
