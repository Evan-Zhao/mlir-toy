#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/embed.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace py = pybind11;

static std::string toString(const llvm::json::Value &value) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << value;
  return os.str();
}

int main() {
  // Prove this binary can also host normal MLIR C++ code.
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral moduleText = R"mlir(
module {
  func.func @identity(%arg0: f32) -> f32 {
    return %arg0 : f32
  }
}
)mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(moduleText, &context);
  if (!module) {
    std::cerr << "failed to parse MLIR module\n";
    return 1;
  }
  std::cout << "parsed MLIR op: " << module->getOperationName().str() << "\n";

  setenv("PYTHONHOME", PYTHON_BASE_PREFIX, 0);
  py::scoped_interpreter guard{};
  try {
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, PY_DEV_PACKAGE_DIR);
    sys.attr("path").attr("insert")(0, PY_SITE_PACKAGES);

    py::module_ solver = py::module_::import("neptune_mlir.rolling_solver");
    py::object result = solver.attr("solve_rolling_updater")("exp(c - r)", "r", "r_new", "acc");

    std::cout << "python solver result: " << result.cast<std::string>() << "\n";

    llvm::json::Value gExpr = llvm::json::Object{
        {"op", "exp"},
        {"type", "f32"},
        {"arg",
         llvm::json::Object{
             {"op", "sub"},
             {"type", "f32"},
             {"lhs", llvm::json::Object{{"op", "var"}, {"name", "c"}, {"type", "f32"}}},
             {"rhs", llvm::json::Object{{"op", "var"}, {"name", "r"}, {"type", "f32"}}},
         }},
    };

    py::module_ json = py::module_::import("json");
    py::object gExprPy = json.attr("loads")(toString(gExpr));
    py::object jsonResult =
        solver.attr("solve_rolling_updater_json")(gExprPy, "r", "r_new", "acc");
    std::string jsonResultText =
        py::str(json.attr("dumps")(jsonResult, py::arg("sort_keys") = true));
    std::cout << "python solver json result: " << jsonResultText << "\n";
  } catch (const py::error_already_set &e) {
    std::cerr << "python error:\n" << e.what() << "\n";
    return 1;
  }

  return 0;
}
