"""Select the MLIR Python binding used by Neptune.

Torch-MLIR ships its own MLIR Python bindings. Prefer those in a clean process
so exporting through torch-mlir and translating through MLIR share one runtime.
"""

import importlib
from types import ModuleType

__all__ = ["ir", "BINDING_MODULE"]

ir: ModuleType
try:
    ir = importlib.import_module("torch_mlir.ir")
    BINDING_MODULE = "torch_mlir.ir"
except ModuleNotFoundError as e:
    if e.name not in {"torch_mlir", "torch_mlir.ir"}:
        raise
    ir = importlib.import_module("mlir.ir")
    BINDING_MODULE = "mlir.ir"
