#include <pybind11/embed.h>

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

namespace py = pybind11;

namespace neptune_mlir {

void printNeptuneMlirPackagePath() {
  static std::once_flag initOnce;
  static std::unique_ptr<py::scoped_interpreter> interpreter;

  std::call_once(initOnce, []() { interpreter = std::make_unique<py::scoped_interpreter>(); });
  py::gil_scoped_acquire gil;
  py::module_ pkg = py::module_::import("neptune_mlir");
  std::string packageFile = py::str(pkg.attr("__file__"));
  std::cout << "neptune_mlir.__file__ = " << packageFile << "\n";
}

} // namespace neptune_mlir

int main() {
  neptune_mlir::printNeptuneMlirPackagePath();
  return 0;
}
