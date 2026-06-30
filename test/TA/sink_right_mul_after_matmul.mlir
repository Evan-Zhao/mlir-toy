// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_right_mul_after_matmul
    } : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @ta_sink_right_mul_after_matmul(
  func.func @ta_sink_right_mul_after_matmul(%scores: tensor<2x3xf32>,
                                            %scale: tensor<2xf32>,
                                            %values: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3, %d "d" extent 4) {
      // CHECK: %[[SCORES:.+]] = ta.at %{{.+}}[%i, %j]
      %scores_expr = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      // CHECK: %[[SCALE:.+]] = ta.at %{{.+}}[%i]
      %scale_expr = ta.at %scale[%i]
          : tensor<2xf32> -> !ta.expr<f32, [i]>
      // CHECK: %[[VALUES:.+]] = ta.at %{{.+}}[%j, %d]
      %values_expr = ta.at %values[%j, %d]
          : tensor<3x4xf32> -> !ta.expr<f32, [j, d]>
      %scaled_values = ta.mulf %values_expr, %scale_expr {ta.import_group = 5 : i64}
          : (!ta.expr<f32, [j, d]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [j, d, i]>
      // CHECK: %[[PROD:.+]] = ta.mulf %[[SCORES]], %[[VALUES]]
      %prod = ta.mulf %scores_expr, %scaled_values {ta.import_group = 11 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [j, d, i]>)
         -> !ta.expr<f32, [i, j, d]>
      // CHECK: %[[RED:.+]] = ta.reduce <add> %[[PROD]]
      // CHECK-SAME: axes = #ta.axes<j>
      %sum = ta.reduce #ta.reduce_kind<add> %prod {axes = #ta.axes<j>, ta.import_group = 11 : i64}
          : !ta.expr<f32, [i, j, d]> -> !ta.expr<f32, [i, d]>
      // CHECK: %[[HOISTED:.+]] = ta.mulf %[[RED]], %[[SCALE]]
      // CHECK-SAME: -> !ta.expr<f32, [i, d]>
      // CHECK: ta.yield %[[HOISTED]]
      ta.yield %sum : !ta.expr<f32, [i, d]>
    } : () -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @ta_sink_right_mul_after_matmul_reject_reduction_axis(
  func.func @ta_sink_right_mul_after_matmul_reject_reduction_axis(%scores: tensor<2x3xf32>,
                                                                  %scale: tensor<3xf32>,
                                                                  %values: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3, %d "d" extent 4) {
      // CHECK: %[[SCORES:.+]] = ta.at %{{.+}}[%i, %j]
      %scores_expr = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      // CHECK: %[[SCALE:.+]] = ta.at %{{.+}}[%j]
      %scale_expr = ta.at %scale[%j]
          : tensor<3xf32> -> !ta.expr<f32, [j]>
      // CHECK: %[[VALUES:.+]] = ta.at %{{.+}}[%j, %d]
      %values_expr = ta.at %values[%j, %d]
          : tensor<3x4xf32> -> !ta.expr<f32, [j, d]>
      // CHECK: %[[SCALED:.+]] = ta.mulf %[[VALUES]], %[[SCALE]]
      %scaled_values = ta.mulf %values_expr, %scale_expr {ta.import_group = 5 : i64}
          : (!ta.expr<f32, [j, d]>, !ta.expr<f32, [j]>)
         -> !ta.expr<f32, [j, d]>
      // CHECK: %[[PROD:.+]] = ta.mulf %[[SCORES]], %[[SCALED]]
      %prod = ta.mulf %scores_expr, %scaled_values {ta.import_group = 11 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [j, d]>)
         -> !ta.expr<f32, [i, j, d]>
      // CHECK: ta.reduce <add> %[[PROD]]
      // CHECK-SAME: axes = #ta.axes<j>
      %sum = ta.reduce #ta.reduce_kind<add> %prod {axes = #ta.axes<j>, ta.import_group = 11 : i64}
          : !ta.expr<f32, [i, j, d]> -> !ta.expr<f32, [i, d]>
      ta.yield %sum : !ta.expr<f32, [i, d]>
    } : () -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
