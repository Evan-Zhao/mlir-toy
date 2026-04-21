include "mlir/IR/OpBase.td"

def HTile_Dialect : Dialect {
  let name = "htile";
  let cppNamespace = "::htile";
  let summary = "Neptune's tile dialect";
  let description = [{
    Neptune's tile dialect.
  }];
}
