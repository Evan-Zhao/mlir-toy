include "TA/TADialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def TA_ReduceKindAttr
    : I32EnumAttr<"ReduceKind", "built-in tensor algebra reducer",
                  [I32EnumAttrCase<"Add", 0, "add">, I32EnumAttrCase<"Mul", 1, "mul">,
                   I32EnumAttrCase<"Max", 2, "max">, I32EnumAttrCase<"Min", 3, "min">]> {
  let cppNamespace = "::ta";
  let genSpecializedAttr = 0;
}

def TA_ReduceKindEnumAttr : EnumAttr<TA_Dialect, TA_ReduceKindAttr, "reduce_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

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

  let builders = [AttrBuilder<(ins "::llvm::StringRef":$name), [{
      return $_get($_ctxt, ::mlir::StringAttr::get($_ctxt, name));
    }]>];

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
