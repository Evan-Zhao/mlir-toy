include "HTile/HTileDialect.td"

def HTile_HelloOp : Op<HTile_Dialect, "hello"> {
  let summary = "trivial hello operation";
  let description = [{
    A zero-operand, zero-result operation used to demonstrate a custom op.
  }];

  let assemblyFormat = "attr-dict";
}
