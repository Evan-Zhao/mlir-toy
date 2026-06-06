include "mlir/IR/OpBase.td"

def TA_ElementwiseOpInterface : OpInterface<"ElementwiseOpInterface"> {
  let description = [{
    Common interface for TA operations that lift scalar elementwise
    computation over ta.expr axis supports.
  }];

  let cppNamespace = "::ta";

  let methods = [InterfaceMethod<
                     /*desc=*/"Return the stable scalar operation kind.",
                     /*returnType=*/"::llvm::StringRef",
                     /*methodName=*/"getScalarOpKind",
                     /*args=*/(ins),
                     /*methodBody=*/[{}],
                     /*defaultImpl=*/[{
        return ConcreteOp::getOperationName();
      }]>,
                 InterfaceMethod<
                     /*desc=*/"Return the axes of one expression operand.",
                     /*returnType=*/"::ta::AxesAttr",
                     /*methodName=*/"getOperandAxes",
                     /*args=*/(ins "unsigned":$index),
                     /*methodBody=*/[{}],
                     /*defaultImpl=*/[{
        auto expr = ::llvm::cast<::ta::ExprType>($_op->getOperand(index).getType());
        return expr.getAxes();
      }]>,
                 InterfaceMethod<
                     /*desc=*/"Return the axes of the single expression result.",
                     /*returnType=*/"::ta::AxesAttr",
                     /*methodName=*/"getResultAxes",
                     /*args=*/(ins),
                     /*methodBody=*/[{}],
                     /*defaultImpl=*/[{
        auto expr = ::llvm::cast<::ta::ExprType>($_op->getResult(0).getType());
        return expr.getAxes();
      }]>];
}
