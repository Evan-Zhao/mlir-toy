// RUN: neptune-opt %s --canonicalize | FileCheck %s

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

  // CHECK-LABEL: func.func @fold_integer_add_zero
  func.func @fold_integer_add_zero() -> tensor<4xi64> {
    %0 = ta.scope axes(%i "i" extent 4) {
      // CHECK: %[[INDEX:.+]] = ta.index
      // CHECK-NOT: ta.constant
      // CHECK-NOT: ta.add
      // CHECK: ta.yield %[[INDEX]]
      %index = ta.index %i : !ta.expr<i64, [i]>
      %zero = ta.constant 0 : i64 : !ta.expr<i64, []>
      %left = ta.add %zero, %index
          : (!ta.expr<i64, []>, !ta.expr<i64, [i]>) -> !ta.expr<i64, [i]>
      %right = ta.add %left, %zero
          : (!ta.expr<i64, [i]>, !ta.expr<i64, []>) -> !ta.expr<i64, [i]>
      ta.yield %right : !ta.expr<i64, [i]>
    } : () -> tensor<4xi64>
    return %0 : tensor<4xi64>
  }

  // A structural unit axis may remain in expression support after affine canonicalization removes
  // it from a linearized physical index.
  // CHECK-LABEL: func.func @preserve_structural_unit_axis
  func.func @preserve_structural_unit_axis(%arg: tensor<4xf32>) -> tensor<1x4xf32> {
    %0 = ta.scope axes(%h "h" extent 1, %i "i" extent 4) {
      // CHECK-NOT: affine.linearize_index
      // CHECK: %[[VALUE:.+]] = ta.at %{{.*}}[%i] : tensor<4xf32> -> !ta.expr<f32, [h, i]>
      // CHECK: ta.yield %[[VALUE]]
      %index = affine.linearize_index disjoint [%h, %i] by (1, 4) : index
      %value = ta.at %arg[%index] : tensor<4xf32> -> !ta.expr<f32, [h, i]>
      ta.yield %value : !ta.expr<f32, [h, i]>
    } : () -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }

  // CHECK-LABEL: func.func @fold_float_infinity_identities
  func.func @fold_float_infinity_identities(%arg: tensor<4xf32>) -> tensor<4xf32> {
    %0 = ta.scope axes(%i "i" extent 4) {
      // CHECK: %[[X:.+]] = ta.at
      // CHECK-NOT: ta.maximum
      // CHECK-NOT: ta.minimum
      // CHECK: ta.yield %[[X]]
      %x = ta.at %arg[%i] : tensor<4xf32> -> !ta.expr<f32, [i]>
      %neg_inf = ta.constant 0xFF800000 : f32 : !ta.expr<f32, []>
      %max = ta.maximum %neg_inf, %x
          : (!ta.expr<f32, []>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %pos_inf = ta.constant 0x7F800000 : f32 : !ta.expr<f32, []>
      %min = ta.minimum %max, %pos_inf
          : (!ta.expr<f32, [i]>, !ta.expr<f32, []>) -> !ta.expr<f32, [i]>
      ta.yield %min : !ta.expr<f32, [i]>
    } : () -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
