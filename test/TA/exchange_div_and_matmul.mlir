// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %target = transform.structured.match ops{["ta.reduce"]} attributes {target} in %func
        : (!transform.any_op) -> !transform.any_op
    %reduction, %division = transform.ta.exchange_div_and_matmul %target
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  // CHECK-LABEL: func.func @ta_exchange_div_and_matmul(
  func.func @ta_exchange_div_and_matmul(%scores: tensor<2x3xf32>, %den: tensor<2xf32>,
                                        %values: tensor<3x4xf32>) -> tensor<2x4xf32> {
    // CHECK: %[[OUT:.+]] = ta.scope
    %out = ta.scope axes(%i "i" : index, %j "j" : index, %d "d" : index) {
      // CHECK: %[[NUM:.+]] = ta.at %{{.+}}[%i, %j]
      %num = ta.at %scores[%i, %j] {axes = #ta.axes<i, j>}
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      // CHECK: %[[DEN:.+]] = ta.at %{{.+}}[%i]
      %den_expr = ta.at %den[%i] {axes = #ta.axes<i>}
          : tensor<2xf32> -> !ta.expr<f32, [i]>
      %prob = ta.divf %num, %den_expr
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [i, j]>
      // CHECK: %[[V:.+]] = ta.at %{{.+}}[%j, %d]
      %v = ta.at %values[%j, %d] {axes = #ta.axes<j, d>}
          : tensor<3x4xf32> -> !ta.expr<f32, [j, d]>
      // CHECK: %[[PROD:.+]] = ta.mulf %[[NUM]], %[[V]]
      // CHECK: %[[RED:.+]] = ta.reduce <add> %[[PROD]] {{.*}}{axes = #ta.axes<j>}
      %prod = ta.mulf %prob, %v
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [j, d]>)
         -> !ta.expr<f32, [i, j, d]>
      %sum = ta.reduce #ta.reduce_kind<add> %prod {axes = #ta.axes<j>, target}
          : !ta.expr<f32, [i, j, d]> -> !ta.expr<f32, [i, d]>
      // CHECK: %[[DIV:.+]] = ta.divf %[[RED]], %[[DEN]]
      // CHECK-SAME: -> !ta.expr<f32, [i, d]>
      // CHECK: ta.yield %[[DIV]]
      ta.yield %sum : !ta.expr<f32, [i, d]>
    } : () -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
