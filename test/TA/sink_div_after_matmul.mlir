// RUN: neptune-opt %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @ta_sink_div_after_matmul(
  func.func @ta_sink_div_after_matmul(%scores: tensor<2x3xf32>, %den: tensor<2xf32>,
                                        %values: tensor<3x4xf32>) -> tensor<2x4xf32> {
    // CHECK: %[[OUT:.+]] = ta.scope
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3, %d "d" extent 4) {
      // CHECK: %[[NUM:.+]] = ta.at %{{.+}}[%i, %j]
      %num = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      // CHECK: %[[DEN:.+]] = ta.at %{{.+}}[%i]
      %den_expr = ta.at %den[%i]
          : tensor<2xf32> -> !ta.expr<f32, [i]>
      %prob = ta.div %num, %den_expr {ta.import_group = 9 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [i, j]>
      // CHECK: %[[V:.+]] = ta.at %{{.+}}[%j, %d]
      %v = ta.at %values[%j, %d]
          : tensor<3x4xf32> -> !ta.expr<f32, [j, d]>
      // CHECK: %[[PROD:.+]] = ta.mul %[[NUM]], %[[V]]
      // CHECK: %[[RED:.+]] = ta.reduce <add> %[[PROD]]
      // CHECK-SAME: axes = #ta.axes<j>
      %prod = ta.mul %prob, %v {ta.import_group = 11 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [j, d]>)
         -> !ta.expr<f32, [i, j, d]>
      %sum = ta.reduce #ta.reduce_kind<add> %prod {axes = #ta.axes<j>, ta.import_group = 11 : i64}
          : !ta.expr<f32, [i, j, d]> -> !ta.expr<f32, [i, d]>
      // CHECK: %[[DIV:.+]] = ta.div %[[RED]], %[[DEN]]
      // CHECK-SAME: -> !ta.expr<f32, [i, d]>
      // CHECK: ta.yield %[[DIV]]
      ta.yield %sum : !ta.expr<f32, [i, d]>
    } : () -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @ta_sink_left_div_through_f16_after_matmul(
  func.func @ta_sink_left_div_through_f16_after_matmul(
      %scores: tensor<2x3xf32>, %den: tensor<2xf32>,
      %values: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3, %d "d" extent 4) {
      // CHECK: %[[NUM:.+]] = ta.at %{{.+}}[%i, %j]
      %num = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      // CHECK: %[[DEN:.+]] = ta.at %{{.+}}[%i]
      %den_expr = ta.at %den[%i]
          : tensor<2xf32> -> !ta.expr<f32, [i]>
      // CHECK: %[[V:.+]] = ta.at %{{.+}}[%j, %d]
      %v = ta.at %values[%j, %d]
          : tensor<3x4xf32> -> !ta.expr<f32, [j, d]>
      %prob = ta.div %num, %den_expr {ta.import_group = 9 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [i, j]>
      // CHECK: %[[NARROW:.+]] = ta.cast %[[NUM]]
      // CHECK-SAME: -> !ta.expr<f16, [i, j]>
      %prob_f16 = ta.cast %prob {ta.import_group = 10 : i64}
          : (!ta.expr<f32, [i, j]>) -> !ta.expr<f16, [i, j]>
      // CHECK: %[[WIDE:.+]] = ta.cast %[[NARROW]]
      // CHECK-SAME: -> !ta.expr<f32, [i, j]>
      %prob_f32 = ta.cast %prob_f16 {ta.import_group = 11 : i64}
          : (!ta.expr<f16, [i, j]>) -> !ta.expr<f32, [i, j]>
      // CHECK: %[[PROD:.+]] = ta.mul %[[WIDE]], %[[V]]
      %prod = ta.mul %prob_f32, %v {ta.import_group = 12 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [j, d]>)
         -> !ta.expr<f32, [i, j, d]>
      // CHECK: %[[RED:.+]] = ta.reduce <add> %[[PROD]]
      // CHECK-SAME: axes = #ta.axes<j>
      %sum = ta.reduce #ta.reduce_kind<add> %prod
          {axes = #ta.axes<j>, ta.import_group = 12 : i64}
          : !ta.expr<f32, [i, j, d]> -> !ta.expr<f32, [i, d]>
      // CHECK: %[[DIV:.+]] = ta.div %[[RED]], %[[DEN]]
      // CHECK-SAME: -> !ta.expr<f32, [i, d]>
      // CHECK: ta.yield %[[DIV]]
      ta.yield %sum : !ta.expr<f32, [i, d]>
    } : () -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @ta_sink_right_div_through_f16_after_matmul(
  func.func @ta_sink_right_div_through_f16_after_matmul(
      %values: tensor<3x4xf32>, %scores: tensor<2x3xf32>,
      %den: tensor<2xf32>) -> tensor<4x2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3, %d "d" extent 4) {
      // CHECK: %[[V:.+]] = ta.at %{{.+}}[%j, %d]
      %v = ta.at %values[%j, %d]
          : tensor<3x4xf32> -> !ta.expr<f32, [j, d]>
      // CHECK: %[[NUM:.+]] = ta.at %{{.+}}[%i, %j]
      %num = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      // CHECK: %[[DEN:.+]] = ta.at %{{.+}}[%i]
      %den_expr = ta.at %den[%i]
          : tensor<2xf32> -> !ta.expr<f32, [i]>
      %prob = ta.div %num, %den_expr {ta.import_group = 9 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [i, j]>
      // CHECK: %[[NARROW:.+]] = ta.cast %[[NUM]]
      // CHECK-SAME: -> !ta.expr<f16, [i, j]>
      %prob_f16 = ta.cast %prob {ta.import_group = 10 : i64}
          : (!ta.expr<f32, [i, j]>) -> !ta.expr<f16, [i, j]>
      // CHECK: %[[WIDE:.+]] = ta.cast %[[NARROW]]
      // CHECK-SAME: -> !ta.expr<f32, [i, j]>
      %prob_f32 = ta.cast %prob_f16 {ta.import_group = 11 : i64}
          : (!ta.expr<f16, [i, j]>) -> !ta.expr<f32, [i, j]>
      // CHECK: %[[PROD:.+]] = ta.mul %[[V]], %[[WIDE]]
      %prod = ta.mul %v, %prob_f32 {ta.import_group = 12 : i64}
          : (!ta.expr<f32, [j, d]>, !ta.expr<f32, [i, j]>)
         -> !ta.expr<f32, [j, d, i]>
      // CHECK: %[[RED:.+]] = ta.reduce <add> %[[PROD]]
      // CHECK-SAME: axes = #ta.axes<j>
      %sum = ta.reduce #ta.reduce_kind<add> %prod
          {axes = #ta.axes<j>, ta.import_group = 12 : i64}
          : !ta.expr<f32, [j, d, i]> -> !ta.expr<f32, [d, i]>
      // CHECK: %[[DIV:.+]] = ta.div %[[RED]], %[[DEN]]
      // CHECK-SAME: -> !ta.expr<f32, [d, i]>
      // CHECK: ta.yield %[[DIV]]
      ta.yield %sum : !ta.expr<f32, [d, i]>
    } : () -> tensor<4x2xf32>
    return %out : tensor<4x2xf32>
  }

  // CHECK-LABEL: func.func @ta_sink_div_after_matmul_reject_reduction_axis(
  func.func @ta_sink_div_after_matmul_reject_reduction_axis(%scores: tensor<2x3xf32>,
                                                             %den: tensor<3xf32>,
                                                             %values: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3, %d "d" extent 4) {
      // CHECK: %[[NUM:.+]] = ta.at %{{.+}}[%i, %j]
      %num = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      // CHECK: %[[DEN:.+]] = ta.at %{{.+}}[%j]
      %den_expr = ta.at %den[%j]
          : tensor<3xf32> -> !ta.expr<f32, [j]>
      // CHECK: %[[PROB:.+]] = ta.div %[[NUM]], %[[DEN]]
      %prob = ta.div %num, %den_expr {ta.import_group = 9 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [j]>)
         -> !ta.expr<f32, [i, j]>
      %v = ta.at %values[%j, %d]
          : tensor<3x4xf32> -> !ta.expr<f32, [j, d]>
      // CHECK: %[[PROD:.+]] = ta.mul %[[PROB]], %{{.+}}
      %prod = ta.mul %prob, %v {ta.import_group = 11 : i64}
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
