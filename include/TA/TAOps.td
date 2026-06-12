include "TA/TATypes.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TA_Op<string mnemonic, list<Trait> traits = []> : Op<TA_Dialect, mnemonic, traits>;

class TA_ElementwiseOp<string mnemonic, list<Trait> traits = []>
    : TA_Op<mnemonic, [Pure]#traits> {
  let hasVerifier = 1;
}

class TA_UnaryElementwiseOp<string mnemonic, list<Trait> traits = []>
    : TA_ElementwiseOp<mnemonic, [InferTypeOpAdaptor]#traits> {
  let arguments = (ins TA_ExprType:$operand);
  let results = (outs TA_ExprType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class TA_BinaryElementwiseOp<string mnemonic, list<Trait> traits = []>
    : TA_ElementwiseOp<mnemonic, [InferTypeOpAdaptor]#traits> {
  let arguments = (ins TA_ExprType:$lhs, TA_ExprType:$rhs);
  let results = (outs TA_ExprType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class TA_TernaryElementwiseOp<string mnemonic, list<Trait> traits = []>
    : TA_ElementwiseOp<mnemonic, [InferTypeOpAdaptor]#traits> {
  let arguments = (ins TA_ExprType:$a, TA_ExprType:$b, TA_ExprType:$c);
  let results = (outs TA_ExprType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class TA_FloatCastElementwiseOp<string mnemonic, list<Trait> traits = []>
    : TA_ElementwiseOp<mnemonic, traits> {
  let arguments = (ins TA_ExprType:$operand);
  let results = (outs TA_ExprType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def TA_YieldOp : TA_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "yield from a tensor algebra region";
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

def TA_AtOp : TA_Op<"at", [Pure]> {
  let summary = "observe a tensor at indexed coordinates";
  let arguments = (ins AnyType:$source, Variadic<Index>:$indices);
  let results = (outs TA_ExprType:$result);

  let assemblyFormat = "$source `[` $indices `]` attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
}

def TA_MapOp : TA_Op<"map", [SingleBlockImplicitTerminator<"YieldOp">, InferTypeOpAdaptor,
                             Pure]> {
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

def TA_ConstantOp : TA_Op<"constant", [Pure, InferTypeOpAdaptor]> {
  let summary = "scalar constant lifted to a ta.expr";
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs TA_ExprType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TA_NegFOp : TA_UnaryElementwiseOp<"negf">;
def TA_AddFOp : TA_BinaryElementwiseOp<"addf", [Commutative]>;
def TA_SubFOp : TA_BinaryElementwiseOp<"subf">;
def TA_SubIOp : TA_BinaryElementwiseOp<"subi">;
def TA_AndIOp : TA_BinaryElementwiseOp<"andi", [Commutative]>;
def TA_MulFOp : TA_BinaryElementwiseOp<"mulf", [Commutative]>;
def TA_DivFOp : TA_BinaryElementwiseOp<"divf">;
def TA_MaximumFOp : TA_BinaryElementwiseOp<"maximumf", [Commutative]>;
def TA_MinimumFOp : TA_BinaryElementwiseOp<"minimumf", [Commutative]>;
def TA_MaxNumFOp : TA_BinaryElementwiseOp<"maxnumf", [Commutative]>;
def TA_MinNumFOp : TA_BinaryElementwiseOp<"minnumf", [Commutative]>;

def TA_AbsFOp : TA_UnaryElementwiseOp<"absf">;
def TA_CeilOp : TA_UnaryElementwiseOp<"ceil">;
def TA_ExpOp : TA_UnaryElementwiseOp<"exp">;
def TA_Exp2Op : TA_UnaryElementwiseOp<"exp2">;
def TA_FloorOp : TA_UnaryElementwiseOp<"floor">;
def TA_LogOp : TA_UnaryElementwiseOp<"log">;
def TA_Log2Op : TA_UnaryElementwiseOp<"log2">;
def TA_RsqrtOp : TA_UnaryElementwiseOp<"rsqrt">;
def TA_SqrtOp : TA_UnaryElementwiseOp<"sqrt">;
def TA_TanhOp : TA_UnaryElementwiseOp<"tanh">;
def TA_ExtFOp : TA_FloatCastElementwiseOp<"extf">;
def TA_TruncFOp : TA_FloatCastElementwiseOp<"truncf"> {
  let hasCanonicalizer = 1;
}

def TA_PowFOp : TA_BinaryElementwiseOp<"powf">;
def TA_FmaOp : TA_TernaryElementwiseOp<"fma">;

def TA_CmpFOp : TA_ElementwiseOp<"cmpf", [InferTypeOpAdaptor]> {
  let summary = "floating-point comparison lifted over ta.expr axes";
  let arguments = (ins Arith_CmpFPredicateAttr:$predicate, TA_ExprType:$lhs, TA_ExprType:$rhs);
  let results = (outs TA_ExprType:$result);
  let assemblyFormat = "$predicate `,` operands attr-dict `:` functional-type(operands, results)";
}

def TA_IndexOp : TA_Op<"index", [Pure]> {
  let summary = "lift a scope axis coordinate to a ta.expr";
  let arguments = (ins Index:$axis);
  let results = (outs TA_ExprType:$result);
  let assemblyFormat = "$axis attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TA_CmpIOp : TA_ElementwiseOp<"cmpi", [InferTypeOpAdaptor]> {
  let summary = "integer comparison lifted over ta.expr axes";
  let arguments = (ins Arith_CmpIPredicateAttr:$predicate, TA_ExprType:$lhs, TA_ExprType:$rhs);
  let results = (outs TA_ExprType:$result);
  let assemblyFormat = "$predicate `,` operands attr-dict `:` functional-type(operands, results)";
}

def TA_SelectOp : TA_ElementwiseOp<"select", [InferTypeOpAdaptor]> {
  let summary = "select lifted over ta.expr axes";
  let arguments = (ins TA_ExprType:$condition, TA_ExprType:$true_value, TA_ExprType:$false_value);
  let results = (outs TA_ExprType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def TA_ReduceOp : TA_Op<"reduce", [Pure, InferTypeOpAdaptor]> {
  let summary = "mathematical reduction over expression axes";
  let arguments = (ins TA_ReduceKindEnumAttr:$kind, TA_ExprType:$input, Optional<AnyType>:$identity,
      TA_AxesAttr:$axes);
  let results = (outs TA_ExprType:$result);

  let assemblyFormat = "$kind $input (`identity` `(` $identity^ `:` type($identity) `)`)? "
                       "attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def TA_ScopeOp
    : TA_Op<"scope", [SingleBlockImplicitTerminator<"YieldOp">,
                      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]> {
  let summary = "materialize a scoped indexed expression";
  let arguments = (ins Variadic<Index>:$dynamic_extents, TA_AxesAttr:$axes,
      DenseI64ArrayAttr:$static_extents);
  let results = (outs AnyRankedTensor:$result);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
