#ifndef TA_TAPASSES
#define TA_TAPASSES

include "mlir/Pass/PassBase.td"

def LinalgToTAPass : Pass<"linalg-to-ta", "mlir::func::FuncOp"> {
  let summary = "Import supported linalg.generic tensor dataflow into the ta dialect";
  let dependentDialects = [
    "ta::TADialect",
    "mlir::affine::AffineDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def StableHLOToTAPass : Pass<"stablehlo-to-ta", "mlir::func::FuncOp"> {
  let summary = "Import supported StableHLO tensor dataflow into the ta dialect";
  let dependentDialects = [
    "ta::TADialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::stablehlo::StablehloDialect"
  ];
}

def TAToLinalgPass : Pass<"ta-to-linalg", "mlir::func::FuncOp"> {
  let summary = "Lower supported ta.scope expression graphs to linalg.generic";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // TA_TAPASSES
