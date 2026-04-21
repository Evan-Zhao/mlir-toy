include "HTile/HTileDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def HTile_Placement : I32EnumAttr<"Placement", "tile data placement", [
  I32EnumAttrCase<"Shared", 0, "shared">,
  I32EnumAttrCase<"Local", 1, "local">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::htile";
}

def HTile_EncodingAttr : AttrDef<HTile_Dialect, "Encoding"> {
  let mnemonic = "encoding";
  let summary = "HTile tensor encoding";
  let description = [{
    Describes target placement information for an SSA tile value.

    This attribute is intended for use as a ranked tensor encoding, for example:

    ```
    tensor<128x128xf16, #htile.encoding<placement = shared>>
    tensor<128x128xf32, #htile.encoding<placement = local>>
    ```
  }];

  let parameters = (ins EnumParameter<HTile_Placement>:$placement);
  let assemblyFormat = "`<` struct(params) `>`";
}
