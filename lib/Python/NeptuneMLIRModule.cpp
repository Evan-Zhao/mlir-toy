#include "HTile/HTileDialect.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include <Python.h>

#ifndef _PyCFunction_CAST
#define _PyCFunction_CAST(func) reinterpret_cast<PyCFunction>(func)
#endif

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HTile, htile, htile::HTileDialect)

namespace {

bool hasCompatibleTypeIDs(MlirContext context) {
  mlir::MLIRContext *cppContext = unwrap(context);
  auto moduleInfo = mlir::RegisteredOperationName::lookup(
      mlir::ModuleOp::getOperationName(), cppContext);
  return moduleInfo &&
         moduleInfo->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>() &&
         moduleInfo->hasTrait<mlir::OpTrait::SymbolTable>() &&
         moduleInfo->getInterface<mlir::SymbolOpInterface>();
}

MlirContext contextFromPython(PyObject *context) {
  PyObject *capsule = PyObject_GetAttrString(context, "_CAPIPtr");
  if (!capsule)
    return MlirContext{nullptr};
  void *ptr = PyCapsule_GetPointer(capsule, "mlir.ir.Context._CAPIPtr");
  Py_DECREF(capsule);
  if (!ptr) {
    PyErr_SetString(PyExc_TypeError, "expected an mlir.ir.Context object");
    return MlirContext{nullptr};
  }
  return MlirContext{ptr};
}

PyObject *registerHTileDialect(PyObject *, PyObject *args, PyObject *kwargs) {
  PyObject *context = nullptr;
  int load = 1;
  static const char *kwlist[] = {"context", "load", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", const_cast<char **>(kwlist), &context,
                                   &load))
    return nullptr;

  MlirContext mlirContext = contextFromPython(context);
  if (!mlirContext.ptr)
    return nullptr;

  if (!hasCompatibleTypeIDs(mlirContext)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "MLIR TypeID ABI mismatch: the Neptune native extension and the MLIR "
        "Python runtime were built with incompatible C++ toolchains");
    return nullptr;
  }

  MlirDialectHandle handle = mlirGetDialectHandle__htile__();
  mlirDialectHandleRegisterDialect(handle, mlirContext);
  if (load)
    mlirDialectHandleLoadDialect(handle, mlirContext);

  Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"register_htile_dialect", _PyCFunction_CAST(registerHTileDialect),
     METH_VARARGS | METH_KEYWORDS, "Register and optionally load the HTile dialect."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_neptuneMlir",
    "Neptune MLIR Python registration hooks",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

} // namespace

PyMODINIT_FUNC PyInit__neptuneMlir() { return PyModule_Create(&module); }
