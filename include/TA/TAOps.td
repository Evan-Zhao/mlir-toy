include "TA/TATypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TA_Op<string mnemonic, list<Trait> traits = []>
    : Op<TA_Dialect, mnemonic, traits>;

def TA_YieldOp : TA_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "yield from a tensor algebra region";
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

def TA_AtOp : TA_Op<"at", [Pure]> {
  let summary = "observe a tensor at indexed coordinates";
  let arguments = (ins AnyType:$source,
                       Variadic<Index>:$indices,
                       OptionalAttr<TA_AxesAttr>:$axes);
  let results = (outs TA_ExprType:$result);

  let assemblyFormat = "$source `[` $indices `]` attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
}

def TA_EvalOp : TA_Op<"eval", [Pure]> {
  let summary = "evaluate a scoped tensor expression at indexed coordinates";
  let arguments = (ins AnyRankedTensor:$source,
                       Variadic<Index>:$indices,
                       OptionalAttr<TA_AxesAttr>:$axes);
  let results = (outs TA_ExprType:$result);

  let assemblyFormat = "$source `[` $indices `]` attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
}

def TA_MapOp : TA_Op<"map", [
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "pointwise scalar map over expression operands";
  let description = [{
    Applies a scalar region pointwise over the union of operand axis supports.
  }];

  let arguments = (ins Variadic<TA_ExprType>:$inputs);
  let results = (outs TA_ExprType:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$inputs $body attr-dict `:` functional-type($inputs, results)";
  let hasVerifier = 1;
}

def TA_ReduceOp : TA_Op<"reduce", [Pure]> {
  let summary = "mathematical reduction over expression axes";
  let arguments = (ins TA_ExprType:$input,
                       Optional<AnyType>:$identity,
                       StrAttr:$kind,
                       TA_AxesAttr:$axes);
  let results = (outs TA_ExprType:$result);

  let assemblyFormat = "$input (`identity` `(` $identity^ `:` type($identity) `)`)? "
                       "attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def TA_ScopeOp : TA_Op<"scope", [
    SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]> {
  let summary = "materialize a scoped indexed expression";
  let arguments = (ins TA_AxesAttr:$axes);
  let results = (outs AnyRankedTensor:$result);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
