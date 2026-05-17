#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <pybind11/embed.h>

namespace py = pybind11;

static std::string toString(const llvm::json::Value &value) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << value;
  return storage;
}

static void ensurePythonInterpreter() {
  static std::once_flag initOnce;
  static std::unique_ptr<py::scoped_interpreter> interpreter;
  std::call_once(initOnce, []() { interpreter = std::make_unique<py::scoped_interpreter>(); });
}

void runPythonHello() {
  ensurePythonInterpreter();
  py::gil_scoped_acquire gil;
  py::module_ pkg = py::module_::import("neptune_mlir");
  std::string packageFile = py::str(pkg.attr("__file__"));
  std::cout << "neptune_mlir.__file__ = " << packageFile << "\n";

  py::module_ solver = py::module_::import("neptune_mlir.rolling_solver");
  py::object result = solver.attr("solve_rolling_updater")("exp(c - r)", "r", "r'", "acc");
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
  py::object jsonResult = solver.attr("solve_rolling_updater_json")(gExprPy, "r", "r'", "acc");
  std::string jsonResultText = py::str(json.attr("dumps")(jsonResult, py::arg("sort_keys") = true));
  std::cout << "python solver json result: " << jsonResultText << "\n";
}

int main() {
  try {
    runPythonHello();
    return 0;
  } catch (const py::error_already_set &e) {
    std::cerr << "python error:\n" << e.what() << "\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
