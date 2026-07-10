// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s --canonicalize | FileCheck %s

module {
  // CHECK-LABEL: func.func @fold_cast_of_float_constant
  func.func @fold_cast_of_float_constant() -> tensor<1xf32> {
    %0 = ta.scope axes(%i "i" extent 1) {
      // CHECK: ta.constant 0.0883883461 : f32 {ta.import_group = 4 : i64} : !ta.expr<f32, []>
      // CHECK-NOT: ta.cast
      %c = ta.constant 0.088388347648318433 : f64 {ta.import_group = 4 : i64} : !ta.expr<f64, []>
      %scale = ta.cast %c {ta.import_group = 4 : i64}
          : (!ta.expr<f64, []>) -> !ta.expr<f32, []>
      ta.yield %scale : !ta.expr<f32, []>
    } : () -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }

  // CHECK-LABEL: func.func @fold_cast_of_integer_constant
  func.func @fold_cast_of_integer_constant() -> tensor<1xf32> {
    %0 = ta.scope axes(%i "i" extent 1) {
      // CHECK: ta.constant -3.000000e+00 : f32 {ta.import_group = 7 : i64} : !ta.expr<f32, []>
      // CHECK-NOT: ta.cast
      %c = ta.constant -3 : i32 {ta.import_group = 7 : i64} : !ta.expr<i32, []>
      %scale = ta.cast %c {ta.import_group = 7 : i64}
          : (!ta.expr<i32, []>) -> !ta.expr<f32, []>
      ta.yield %scale : !ta.expr<f32, []>
    } : () -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }

  // CHECK-LABEL: func.func @fold_mulf_of_float_constants
  func.func @fold_mulf_of_float_constants() -> tensor<1xf32> {
    %0 = ta.scope axes(%i "i" extent 1) {
      // CHECK: ta.constant 7.500000e+00 : f32
      // CHECK-NOT: ta.mul
      %lhs = ta.constant 2.500000e+00 : f32 : !ta.expr<f32, []>
      %rhs = ta.constant 3.000000e+00 : f32 : !ta.expr<f32, []>
      %mul = ta.mul %lhs, %rhs
          : (!ta.expr<f32, []>, !ta.expr<f32, []>) -> !ta.expr<f32, []>
      ta.yield %mul : !ta.expr<f32, []>
    } : () -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
