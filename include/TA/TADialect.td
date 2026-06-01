include "mlir/IR/OpBase.td"

def TA_Dialect : Dialect {
  let name = "ta";
  let cppNamespace = "::ta";
  let summary = "Tensor algebra rewrite dialect";
  let description = [{
    A tensor algebra dialect for whole-program indexed rewrites over named
    axes. This first implementation provides the core IR shell: axis
    attributes, expression types, and the primitive operations from the
    design note.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}
