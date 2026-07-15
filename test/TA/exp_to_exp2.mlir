// RUN: neptune-opt %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.exp_to_exp2
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @ta_exp_to_exp2_softmax_core(
  func.func @ta_exp_to_exp2_softmax_core(%scores: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      // CHECK: %[[X:.+]] = ta.at %{{.+}}[%i, %j]
      %x = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %scale = ta.constant 5.000000e-01 : f32 : !ta.expr<f32, []>
      %s = ta.mul %scale, %x {ta.import_group = 1 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      %m = ta.reduce #ta.reduce_kind<max> %s {axes = #ta.axes<j>, ta.import_group = 2 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %centered = ta.sub %s, %m {ta.import_group = 3 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [i, j]>
      %p = ta.exp %centered {ta.import_group = 4 : i64}
          : (!ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      // CHECK: %[[NEW_SCALE:.+]] = ta.constant 0.72134751 : f32
      // CHECK: %[[SCALED_X:.+]] = ta.mul %[[NEW_SCALE]], %[[X]]
      // CHECK: %[[MAX:.+]] = ta.reduce <max> %[[SCALED_X]]
      // CHECK-SAME: axes = #ta.axes<j>
      // CHECK: %[[CENTERED:.+]] = ta.sub %[[SCALED_X]], %[[MAX]]
      // CHECK: %[[P:.+]] = ta.exp2 %[[CENTERED]]
      // CHECK: ta.yield %[[P]]
      ta.yield %p : !ta.expr<f32, [i, j]>
    } : () -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }

  // CHECK-LABEL: func.func @ta_exp_to_exp2_masked_softmax_core(
  func.func @ta_exp_to_exp2_masked_softmax_core(%scores: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      // CHECK: %[[X:.+]] = ta.at %{{.+}}[%i, %j]
      %x = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %mask = ta.constant true : !ta.expr<i1, []>
      %neg_inf = ta.constant 0xFF800000 : f32 : !ta.expr<f32, []>
      %scale = ta.constant 5.000000e-01 : f32 : !ta.expr<f32, []>
      %s = ta.mul %scale, %x {ta.import_group = 5 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      %masked = ta.select %mask, %s, %neg_inf {ta.import_group = 6 : i64}
          : (!ta.expr<i1, []>, !ta.expr<f32, [i, j]>, !ta.expr<f32, []>)
         -> !ta.expr<f32, [i, j]>
      %m = ta.reduce #ta.reduce_kind<max> %masked {axes = #ta.axes<j>, ta.import_group = 7 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %centered = ta.sub %masked, %m {ta.import_group = 8 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [i, j]>
      %p = ta.exp %centered {ta.import_group = 9 : i64}
          : (!ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      // CHECK: %[[NEW_SCALE:.+]] = ta.constant 0.72134751 : f32
      // CHECK: %[[SCALED_X:.+]] = ta.mul %[[NEW_SCALE]], %[[X]]
      // CHECK: %[[SCALED_NEG_INF:.+]] = ta.constant 0xFF800000 : f32
      // CHECK: %[[MASKED:.+]] = ta.select %{{.+}}, %[[SCALED_X]], %[[SCALED_NEG_INF]]
      // CHECK: %[[MAX:.+]] = ta.reduce <max> %[[MASKED]]
      // CHECK-SAME: axes = #ta.axes<j>
      // CHECK: %[[CENTERED:.+]] = ta.sub %[[MASKED]], %[[MAX]]
      // CHECK: %[[P:.+]] = ta.exp2 %[[CENTERED]]
      // CHECK: ta.yield %[[P]]
      ta.yield %p : !ta.expr<f32, [i, j]>
    } : () -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
