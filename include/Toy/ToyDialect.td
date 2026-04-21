include "mlir/IR/OpBase.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::toy";
  let summary = "Tiny example dialect";
  let description = [{
    A small dialect used by this standalone example.
  }];
}
