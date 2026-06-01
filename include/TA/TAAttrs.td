include "TA/TADialect.td"
include "mlir/IR/AttrTypeBase.td"

def TA_AxisAttr : AttrDef<TA_Dialect, "Axis"> {
  let mnemonic = "axis";
  let summary = "logical tensor algebra axis";
  let description = [{
    Identifies a logical tensor algebra axis such as `b`, `h`, `i`, or `j`.
    Expression types refer to axes by this attribute rather than by
    SSA value.
  }];

  let parameters = (ins "::mlir::StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";

  let builders = [
    AttrBuilder<(ins "::llvm::StringRef":$name), [{
      return $_get($_ctxt, ::mlir::StringAttr::get($_ctxt, name));
    }]>
  ];

  let genVerifyDecl = 1;
}

def TA_AxesAttr : AttrDef<TA_Dialect, "Axes"> {
  let mnemonic = "axes";
  let summary = "list of logical tensor algebra axes";
  let description = [{
    A compact attribute for axis lists. It prints as `#ta.axes<i, j>` while
    storing an array of `#ta.axis` attributes internally.
  }];

  let parameters = (ins "::mlir::ArrayAttr":$axes);
  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}
