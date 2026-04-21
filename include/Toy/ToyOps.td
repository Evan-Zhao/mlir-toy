include "Toy/ToyDialect.td"

def Toy_HelloOp : Op<Toy_Dialect, "hello"> {
  let summary = "trivial hello operation";
  let description = [{
    A zero-operand, zero-result operation used to demonstrate a custom op.
  }];

  let assemblyFormat = "attr-dict";
}
