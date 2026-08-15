// RUN: neptune-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @sink_left_positive_scale_after_max_reduce(
// CHECK: %[[INPUT:.+]] = ta.at
// CHECK: %[[SCALE:.+]] = ta.constant 5.000000e-01
// CHECK-NEXT: %[[MAX:.+]] = ta.reduce <max> %[[INPUT]]
// CHECK-NEXT: %[[RESULT:.+]] = ta.mul %[[SCALE]], %[[MAX]]
// CHECK: ta.yield %[[RESULT]]
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !transform.any_op
    transform.yield
  }

  func.func @sink_left_positive_scale_after_max_reduce(%input: tensor<2x3xf32>) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %scale = ta.constant 5.000000e-01 : f32 : !ta.expr<f32, []>
      %scaled = ta.mul %scale, %x {ta.import_group = 1 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      %max = ta.reduce #ta.reduce_kind<max> %scaled
          {axes = #ta.axes<j>, ta.import_group = 2 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %max : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }
}
// -----

// CHECK-LABEL: func.func @sink_right_positive_scale_after_max_reduce(
// CHECK: %[[INPUT:.+]] = ta.at
// CHECK: %[[SCALE:.+]] = ta.constant 5.000000e-01
// CHECK-NEXT: %[[MAX:.+]] = ta.reduce <max> %[[INPUT]]
// CHECK-NEXT: %[[RESULT:.+]] = ta.mul %[[MAX]], %[[SCALE]]
// CHECK: ta.yield %[[RESULT]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !transform.any_op
    transform.yield
  }

  func.func @sink_right_positive_scale_after_max_reduce(%input: tensor<2x3xf32>) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %scale = ta.constant 5.000000e-01 : f32 : !ta.expr<f32, []>
      %scaled = ta.mul %x, %scale {ta.import_group = 3 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, []>) -> !ta.expr<f32, [i, j]>
      %max = ta.reduce #ta.reduce_kind<max> %scaled
          {axes = #ta.axes<j>, ta.import_group = 4 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %max : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }
}
// -----

// CHECK-LABEL: func.func @sink_left_positive_scale_after_masked_select(
// CHECK: %[[INPUT:.+]] = ta.at
// CHECK: %[[PRED:.+]] = ta.at
// CHECK: %[[NEG_INF:.+]] = ta.constant 0xFF800000
// CHECK: %[[SCALE:.+]] = ta.constant 5.000000e-01
// CHECK: %[[MASKED:.+]] = ta.select %[[PRED]], %[[INPUT]], %[[NEG_INF]]
// CHECK: %[[MAX:.+]] = ta.reduce <max> %[[MASKED]]
// CHECK-NEXT: %[[RESULT:.+]] = ta.mul %[[SCALE]], %[[MAX]]
// CHECK: ta.yield %[[RESULT]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !transform.any_op
    transform.yield
  }

  func.func @sink_left_positive_scale_after_masked_select(
      %input: tensor<2x3xf32>, %pred: tensor<2x3xi1>) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %p = ta.at %pred[%i, %j]
          : tensor<2x3xi1> -> !ta.expr<i1, [i, j]>
      %neg_inf = ta.constant 0xFF800000 : f32 : !ta.expr<f32, []>
      %scale = ta.constant 5.000000e-01 : f32 : !ta.expr<f32, []>
      %scaled = ta.mul %scale, %x {ta.import_group = 5 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      %masked = ta.select %p, %scaled, %neg_inf {ta.import_group = 6 : i64}
          : (!ta.expr<i1, [i, j]>, !ta.expr<f32, [i, j]>, !ta.expr<f32, []>)
         -> !ta.expr<f32, [i, j]>
      %max = ta.reduce #ta.reduce_kind<max> %masked
          {axes = #ta.axes<j>, ta.import_group = 7 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %max : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }
}
// -----

// CHECK-LABEL: func.func @sink_right_positive_scale_after_masked_select(
// CHECK: %[[INPUT:.+]] = ta.at
// CHECK: %[[PRED:.+]] = ta.at
// CHECK: %[[NEG_INF:.+]] = ta.constant 0xFF800000
// CHECK: %[[SCALE:.+]] = ta.constant 5.000000e-01
// CHECK: %[[MASKED:.+]] = ta.select %[[PRED]], %[[INPUT]], %[[NEG_INF]]
// CHECK: %[[MAX:.+]] = ta.reduce <max> %[[MASKED]]
// CHECK-NEXT: %[[RESULT:.+]] = ta.mul %[[MAX]], %[[SCALE]]
// CHECK: ta.yield %[[RESULT]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !transform.any_op
    transform.yield
  }

  func.func @sink_right_positive_scale_after_masked_select(
      %input: tensor<2x3xf32>, %pred: tensor<2x3xi1>) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %p = ta.at %pred[%i, %j]
          : tensor<2x3xi1> -> !ta.expr<i1, [i, j]>
      %neg_inf = ta.constant 0xFF800000 : f32 : !ta.expr<f32, []>
      %scale = ta.constant 5.000000e-01 : f32 : !ta.expr<f32, []>
      %scaled = ta.mul %x, %scale {ta.import_group = 8 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, []>) -> !ta.expr<f32, [i, j]>
      %masked = ta.select %p, %scaled, %neg_inf {ta.import_group = 9 : i64}
          : (!ta.expr<i1, [i, j]>, !ta.expr<f32, [i, j]>, !ta.expr<f32, []>)
         -> !ta.expr<f32, [i, j]>
      %max = ta.reduce #ta.reduce_kind<max> %masked
          {axes = #ta.axes<j>, ta.import_group = 10 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %max : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }
}
// -----

// CHECK-LABEL: func.func @do_not_sink_scale_after_masked_max_with_finite_fill(
// CHECK: %[[INPUT:.+]] = ta.at
// CHECK: %[[PRED:.+]] = ta.at
// CHECK: %[[FILL:.+]] = ta.constant -1.000000e+03
// CHECK: %[[SCALE:.+]] = ta.constant 5.000000e-01
// CHECK: %[[SCALED:.+]] = ta.mul %[[SCALE]], %[[INPUT]]
// CHECK: %[[MASKED:.+]] = ta.select %[[PRED]], %[[SCALED]], %[[FILL]]
// CHECK: %[[MAX:.+]] = ta.reduce <max> %[[MASKED]]
// CHECK: ta.yield %[[MAX]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !transform.any_op
    transform.yield
  }

  func.func @do_not_sink_scale_after_masked_max_with_finite_fill(
      %input: tensor<2x3xf32>, %pred: tensor<2x3xi1>) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %p = ta.at %pred[%i, %j]
          : tensor<2x3xi1> -> !ta.expr<i1, [i, j]>
      %fill = ta.constant -1.000000e+03 : f32 : !ta.expr<f32, []>
      %scale = ta.constant 5.000000e-01 : f32 : !ta.expr<f32, []>
      %scaled = ta.mul %scale, %x {ta.import_group = 11 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      %masked = ta.select %p, %scaled, %fill {ta.import_group = 12 : i64}
          : (!ta.expr<i1, [i, j]>, !ta.expr<f32, [i, j]>, !ta.expr<f32, []>)
         -> !ta.expr<f32, [i, j]>
      %max = ta.reduce #ta.reduce_kind<max> %masked
          {axes = #ta.axes<j>, ta.import_group = 13 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %max : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }
}
// -----

// CHECK-LABEL: func.func @do_not_sink_negative_scale_after_max_reduce(
// CHECK: %[[INPUT:.+]] = ta.at
// CHECK: %[[SCALE:.+]] = ta.constant -5.000000e-01
// CHECK-NEXT: %[[SCALED:.+]] = ta.mul %[[SCALE]], %[[INPUT]]
// CHECK-NEXT: %[[MAX:.+]] = ta.reduce <max> %[[SCALED]]
// CHECK: ta.yield %[[MAX]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !transform.any_op
    transform.yield
  }

  func.func @do_not_sink_negative_scale_after_max_reduce(%input: tensor<2x3xf32>) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %scale = ta.constant -5.000000e-01 : f32 : !ta.expr<f32, []>
      %scaled = ta.mul %scale, %x {ta.import_group = 5 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      %max = ta.reduce #ta.reduce_kind<max> %scaled
          {axes = #ta.axes<j>, ta.import_group = 6 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %max : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }
}
