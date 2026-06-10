"""Select the MLIR Python binding used by Neptune.

Used to point to TorchMLIR when installed. Now we are using the MLIR Python bindings
from llvm-project consistently. Kept as a separate module so we can switch between
different MLIR Python bindings in the future if needed.
"""

import importlib
from types import ModuleType

__all__ = ["ir", "BINDING_MODULE"]

ir: ModuleType = importlib.import_module("mlir.ir")
BINDING_MODULE = "mlir.ir"
