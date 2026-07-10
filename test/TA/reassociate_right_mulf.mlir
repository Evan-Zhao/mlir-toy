// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.reassociate_right_mulf
    } : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @ta_reassociate_right_mulf_reject_equal_cost(
  func.func @ta_reassociate_right_mulf_reject_equal_cost(%lhs: tensor<1xf32>,
                                                         %rhs: tensor<3xf32>) -> tensor<1xf32> {
    %out = ta.scope axes(%i "i" extent 1, %j "j" extent 3) {
      // CHECK: %[[LHS:.+]] = ta.at %{{.+}}[%i]
      %lhs_expr = ta.at %lhs[%i]
          : tensor<1xf32> -> !ta.expr<f32, [i]>
      // CHECK: %[[RHS:.+]] = ta.at %{{.+}}[%j]
      %rhs_expr = ta.at %rhs[%j]
          : tensor<3xf32> -> !ta.expr<f32, [j]>
      // CHECK: %[[FACTOR:.+]] = ta.constant
      %factor_expr = ta.constant 2.000000e+00 : f32 : !ta.expr<f32, []>
      // CHECK: %[[INNER:.+]] = ta.mul %[[LHS]], %[[RHS]]
      %inner = ta.mul %lhs_expr, %rhs_expr {ta.import_group = 5 : i64}
          : (!ta.expr<f32, [i]>, !ta.expr<f32, [j]>)
         -> !ta.expr<f32, [i, j]>
      // CHECK: %[[OUTER:.+]] = ta.mul %[[FACTOR]], %[[INNER]]
      %outer = ta.mul %factor_expr, %inner {ta.import_group = 11 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i, j]>)
         -> !ta.expr<f32, [i, j]>
      // CHECK: ta.reduce <add> %[[OUTER]]
      // CHECK-SAME: axes = #ta.axes<j>
      %sum = ta.reduce #ta.reduce_kind<add> %outer {axes = #ta.axes<j>, ta.import_group = 11 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %sum : !ta.expr<f32, [i]>
    } : () -> tensor<1xf32>
    return %out : tensor<1xf32>
  }

  // CHECK-LABEL: func.func @ta_reassociate_right_mulf_uses_element_count(
  func.func @ta_reassociate_right_mulf_uses_element_count(%lhs: tensor<100xf32>,
                                                          %rhs: tensor<2xf32>) -> tensor<100x2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 100) {
      // CHECK: %[[LHS:.+]] = ta.at %{{.+}}[%j]
      %lhs_expr = ta.at %lhs[%j]
          : tensor<100xf32> -> !ta.expr<f32, [j]>
      // CHECK: %[[RHS:.+]] = ta.at %{{.+}}[%i]
      %rhs_expr = ta.at %rhs[%i]
          : tensor<2xf32> -> !ta.expr<f32, [i]>
      // CHECK: %[[FACTOR:.+]] = ta.constant
      %factor_expr = ta.constant 2.000000e+00 : f32 : !ta.expr<f32, []>
      %inner = ta.mul %lhs_expr, %rhs_expr {ta.import_group = 5 : i64}
          : (!ta.expr<f32, [j]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [j, i]>
      // CHECK: %[[NEW_INNER:.+]] = ta.mul %[[RHS]], %[[FACTOR]]
      // CHECK-SAME: -> !ta.expr<f32, [i]>
      %outer = ta.mul %factor_expr, %inner {ta.import_group = 11 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [j, i]>)
         -> !ta.expr<f32, [j, i]>
      // CHECK: %[[NEW_OUTER:.+]] = ta.mul %[[LHS]], %[[NEW_INNER]]
      // CHECK-SAME: -> !ta.expr<f32, [j, i]>
      // CHECK: ta.yield %[[NEW_OUTER]]
      ta.yield %outer : !ta.expr<f32, [j, i]>
    } : () -> tensor<100x2xf32>
    return %out : tensor<100x2xf32>
  }
}
