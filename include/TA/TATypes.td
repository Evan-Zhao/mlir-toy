include "TA/TAAttrs.td"
include "mlir/IR/AttrTypeBase.td"

class TA_Type<string name, string typeMnemonic> : TypeDef<TA_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def TA_Expr : TA_Type<"Expr", "expr"> {
  let summary = "indexed scalar expression";
  let description = [{
    A scalar-valued expression over a set of logical axes. The axis list is an
    upper bound on the axes the value may depend on.
  }];

  let parameters = (ins "::mlir::Type":$elementType, "::ta::AxesAttr":$axes);
  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}

def TA_ExprType : Type<CPred<"::llvm::isa<::ta::ExprType>($_self)">, "ta expression type">;
