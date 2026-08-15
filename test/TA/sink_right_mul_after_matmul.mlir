// RUN: neptune-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @ta_sink_right_mul_with_fp8_prewiden(
// CHECK: %[[SCORES:.+]] = ta.at %{{.+}}[%i, %j]
// CHECK: %[[SCALE:.+]] = ta.at %{{.+}}[%i]
// CHECK: %[[VALUES:.+]] = ta.at %{{.+}}[%j, %d]
// CHECK: %[[NARROW:.+]] = ta.cast %[[VALUES]]
// CHECK-SAME: -> !ta.expr<f16, [j, d]>
// CHECK: %[[WIDE:.+]] = ta.cast %[[NARROW]]
// CHECK-SAME: -> !ta.expr<f32, [j, d]>
// CHECK: %[[RED:.+]] = ta.reduce <add>
// CHECK: %[[HOISTED:.+]] = ta.mul %[[RED]], %[[SCALE]]
// CHECK: ta.yield %[[HOISTED]]
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_right_mul_after_matmul
    } : !transform.any_op
    transform.yield
  }

  func.func @ta_sink_right_mul_with_fp8_prewiden(
      %scores: tensor<2x3xf32>, %scale: tensor<2xf32>,
      %values: tensor<3x4xf8E4M3FN>) -> tensor<2x4xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3, %d "d" extent 4) {
      %scores_expr = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %scale_expr = ta.at %scale[%i]
          : tensor<2xf32> -> !ta.expr<f32, [i]>
      %values_expr = ta.at %values[%j, %d]
          : tensor<3x4xf8E4M3FN> -> !ta.expr<f8E4M3FN, [j, d]>
      %values_f32 = ta.cast %values_expr {ta.import_group = 4 : i64}
          : (!ta.expr<f8E4M3FN, [j, d]>) -> !ta.expr<f32, [j, d]>
      %scaled_values = ta.mul %values_f32, %scale_expr {ta.import_group = 5 : i64}
          : (!ta.expr<f32, [j, d]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [j, d, i]>
      %values_f16 = ta.cast %scaled_values {ta.import_group = 6 : i64}
          : (!ta.expr<f32, [j, d, i]>) -> !ta.expr<f16, [j, d, i]>
      %rounded_values = ta.cast %values_f16 {ta.import_group = 7 : i64}
          : (!ta.expr<f16, [j, d, i]>) -> !ta.expr<f32, [j, d, i]>
      %prod = ta.mul %scores_expr, %rounded_values {ta.import_group = 11 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [j, d, i]>)
         -> !ta.expr<f32, [i, j, d]>
      %sum = ta.reduce #ta.reduce_kind<add> %prod
          {axes = #ta.axes<j>, ta.import_group = 11 : i64}
          : !ta.expr<f32, [i, j, d]> -> !ta.expr<f32, [i, d]>
      ta.yield %sum : !ta.expr<f32, [i, d]>
    } : () -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
// -----

// CHECK-LABEL: func.func @ta_sink_right_mul_through_f16_reject_reduction_axis(
// CHECK: %[[SCORES:.+]] = ta.at %{{.+}}[%i, %j]
// CHECK: %[[SCALE:.+]] = ta.at %{{.+}}[%j]
// CHECK: %[[VALUES:.+]] = ta.at %{{.+}}[%j, %d]
// CHECK: %[[VALUES_F32:.+]] = ta.cast %[[VALUES]]
// CHECK: %[[SCALED:.+]] = ta.mul %[[VALUES_F32]], %[[SCALE]]
// CHECK: %[[NARROW:.+]] = ta.cast %[[SCALED]]
// CHECK: %[[WIDE:.+]] = ta.cast %[[NARROW]]
// CHECK: %[[PROD:.+]] = ta.mul %[[SCORES]], %[[WIDE]]
// CHECK: ta.reduce <add> %[[PROD]]
// CHECK-SAME: axes = #ta.axes<j>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_right_mul_after_matmul
    } : !transform.any_op
    transform.yield
  }

  func.func @ta_sink_right_mul_through_f16_reject_reduction_axis(
      %scores: tensor<2x3xf32>, %scale: tensor<3xf32>,
      %values: tensor<3x4xf8E4M3FN>) -> tensor<2x4xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3, %d "d" extent 4) {
      %scores_expr = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %scale_expr = ta.at %scale[%j]
          : tensor<3xf32> -> !ta.expr<f32, [j]>
      %values_expr = ta.at %values[%j, %d]
          : tensor<3x4xf8E4M3FN> -> !ta.expr<f8E4M3FN, [j, d]>
      %values_f32 = ta.cast %values_expr {ta.import_group = 4 : i64}
          : (!ta.expr<f8E4M3FN, [j, d]>) -> !ta.expr<f32, [j, d]>
      %scaled_values = ta.mul %values_f32, %scale_expr {ta.import_group = 5 : i64}
          : (!ta.expr<f32, [j, d]>, !ta.expr<f32, [j]>)
         -> !ta.expr<f32, [j, d]>
      %values_f16 = ta.cast %scaled_values {ta.import_group = 6 : i64}
          : (!ta.expr<f32, [j, d]>) -> !ta.expr<f16, [j, d]>
      %rounded_values = ta.cast %values_f16 {ta.import_group = 7 : i64}
          : (!ta.expr<f16, [j, d]>) -> !ta.expr<f32, [j, d]>
      %prod = ta.mul %scores_expr, %rounded_values {ta.import_group = 11 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [j, d]>)
         -> !ta.expr<f32, [i, j, d]>
      %sum = ta.reduce #ta.reduce_kind<add> %prod {axes = #ta.axes<j>, ta.import_group = 11 : i64}
          : !ta.expr<f32, [i, j, d]> -> !ta.expr<f32, [i, d]>
      ta.yield %sum : !ta.expr<f32, [i, d]>
    } : () -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
