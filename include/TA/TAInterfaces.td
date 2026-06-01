include "mlir/IR/OpBase.td"

def TA_ElementwiseOpInterface : OpInterface<"ElementwiseOpInterface"> {
  let description = [{
    Common interface for TA operations that lift scalar elementwise
    computation over ta.expr axis supports.
  }];

  let cppNamespace = "::ta";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the stable scalar operation kind.",
      /*returnType=*/"::llvm::StringRef",
      /*methodName=*/"getScalarOpKind",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImpl=*/[{
        return ConcreteOp::getOperationName();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the axes of one expression operand.",
      /*returnType=*/"::ta::AxesAttr",
      /*methodName=*/"getOperandAxes",
      /*args=*/(ins "unsigned":$index),
      /*methodBody=*/[{}],
      /*defaultImpl=*/[{
        auto expr = ::llvm::cast<::ta::ExprType>($_op->getOperand(index).getType());
        return expr.getAxes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the axes of the single expression result.",
      /*returnType=*/"::ta::AxesAttr",
      /*methodName=*/"getResultAxes",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImpl=*/[{
        auto expr = ::llvm::cast<::ta::ExprType>($_op->getResult(0).getType());
        return expr.getAxes();
      }]
    >
  ];
}

def TA_ReduceLikeOpInterface : OpInterface<"ReduceLikeOpInterface"> {
  let description = [{
    Common interface for TA reduction binders.
  }];

  let cppNamespace = "::ta";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the reducer kind attribute.",
      /*returnType=*/"::ta::ReduceKindAttr",
      /*methodName=*/"getReduceKindAttr",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImpl=*/[{
        auto op = ::llvm::cast<ConcreteOp>(this->getOperation());
        return op.getKindAttr();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the axes reduced by this operation.",
      /*returnType=*/"::ta::AxesAttr",
      /*methodName=*/"getReductionAxes",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImpl=*/[{
        auto op = ::llvm::cast<ConcreteOp>(this->getOperation());
        return op.getAxes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the payload expression type.",
      /*returnType=*/"::ta::ExprType",
      /*methodName=*/"getPayloadExprType",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Return the result expression type.",
      /*returnType=*/"::ta::ExprType",
      /*methodName=*/"getResultExprType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImpl=*/[{
        return ::llvm::cast<::ta::ExprType>($_op->getResult(0).getType());
      }]
    >
  ];
}
