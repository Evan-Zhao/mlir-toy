// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.exchange_div_and_matmul
    } : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @ta_exchange_div_and_matmul(
  func.func @ta_exchange_div_and_matmul(%scores: tensor<2x3xf32>, %den: tensor<2xf32>,
                                        %values: tensor<3x4xf32>) -> tensor<2x4xf32> {
    // CHECK: %[[OUT:.+]] = ta.scope
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3, %d "d" extent 4) {
      // CHECK: %[[NUM:.+]] = ta.at %{{.+}}[%i, %j]
      %num = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      // CHECK: %[[DEN:.+]] = ta.at %{{.+}}[%i]
      %den_expr = ta.at %den[%i]
          : tensor<2xf32> -> !ta.expr<f32, [i]>
      %prob = ta.divf %num, %den_expr {ta.import_group = 9 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [i, j]>
      // CHECK: %[[V:.+]] = ta.at %{{.+}}[%j, %d]
      %v = ta.at %values[%j, %d]
          : tensor<3x4xf32> -> !ta.expr<f32, [j, d]>
      // CHECK: %[[PROD:.+]] = ta.mulf %[[NUM]], %[[V]]
      // CHECK: %[[RED:.+]] = ta.reduce <add> %[[PROD]]
      // CHECK-SAME: axes = #ta.axes<j>
      %prod = ta.mulf %prob, %v {ta.import_group = 11 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [j, d]>)
         -> !ta.expr<f32, [i, j, d]>
      %sum = ta.reduce #ta.reduce_kind<add> %prod {axes = #ta.axes<j>, ta.import_group = 11 : i64}
          : !ta.expr<f32, [i, j, d]> -> !ta.expr<f32, [i, d]>
      // CHECK: %[[DIV:.+]] = ta.divf %[[RED]], %[[DEN]]
      // CHECK-SAME: -> !ta.expr<f32, [i, d]>
      // CHECK: ta.yield %[[DIV]]
      ta.yield %sum : !ta.expr<f32, [i, d]>
    } : () -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @ta_exchange_div_and_matmul_reject_reduction_axis(
  func.func @ta_exchange_div_and_matmul_reject_reduction_axis(%scores: tensor<2x3xf32>,
                                                             %den: tensor<3xf32>,
                                                             %values: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3, %d "d" extent 4) {
      // CHECK: %[[NUM:.+]] = ta.at %{{.+}}[%i, %j]
      %num = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      // CHECK: %[[DEN:.+]] = ta.at %{{.+}}[%j]
      %den_expr = ta.at %den[%j]
          : tensor<3xf32> -> !ta.expr<f32, [j]>
      // CHECK: %[[PROB:.+]] = ta.divf %[[NUM]], %[[DEN]]
      %prob = ta.divf %num, %den_expr {ta.import_group = 9 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [j]>)
         -> !ta.expr<f32, [i, j]>
      %v = ta.at %values[%j, %d]
          : tensor<3x4xf32> -> !ta.expr<f32, [j, d]>
      // CHECK: %[[PROD:.+]] = ta.mulf %[[PROB]], %{{.+}}
      %prod = ta.mulf %prob, %v {ta.import_group = 11 : i64}
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
