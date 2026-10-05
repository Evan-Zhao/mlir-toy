// RUN: %neptune-opt %s --transform-interpreter --split-input-file | FileCheck %s

// Scale hoisting preserves the finite-extremum initializer.
// CHECK-LABEL: func.func @hoist_positive_scale_before_max_reduce(
// CHECK: %[[X:.+]] = ta.at %{{.+}}[%i, %j]
// CHECK: %[[NEW_SCALE:.+]] = ta.constant 0.72134751 : f32
// CHECK: %[[SCALED_X:.+]] = ta.mul %[[NEW_SCALE]], %[[X]]
// CHECK: %[[MAX:.+]] = ta.reduce <max> %[[SCALED_X]] init(-3.40282347E+38 : f32)
// CHECK-SAME: axes = #ta.axes<j>
// CHECK: %[[CENTERED:.+]] = ta.sub %[[SCALED_X]], %[[MAX]]
// CHECK: %[[P:.+]] = ta.exp2 %[[CENTERED]]
// CHECK: ta.yield %[[P]]
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.exp_to_exp2
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  func.func @hoist_positive_scale_before_max_reduce(%scores: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %scores[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %scale = ta.constant 5.000000e-01 : f32 : !ta.expr<f32, []>
      %s = ta.mul %scale, %x {ta.import_group = 1 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      %m = ta.reduce #ta.reduce_kind<max> %s init(0xFF7FFFFF : f32) {axes = #ta.axes<j>, ta.import_group = 2 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %centered = ta.sub %s, %m {ta.import_group = 3 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [i, j]>
      %p = ta.exp %centered {ta.import_group = 4 : i64}
          : (!ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      ta.yield %p : !ta.expr<f32, [i, j]>
    } : () -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
// -----

// Both infinities and a constant SSA +max_float are accepted as sentinels.
// CHECK-LABEL: func.func @hoist_with_extreme_initializers
// CHECK: %[[SEED:.+]] = arith.constant 3.40282347E+38 : f32
// CHECK: %[[X:.+]] = ta.at
// CHECK: %[[C:.+]] = ta.constant 2.000000e+00 : f32
// CHECK: %[[S:.+]] = ta.mul %[[C]], %[[X]]
// CHECK-DAG: ta.reduce <max> %[[S]] init(0xFF800000 : f32)
// CHECK-DAG: ta.reduce <max> %[[S]] init(0x7F800000 : f32)
// CHECK-DAG: ta.reduce <max> %[[S]] init(%[[SEED]] : f32)
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.exp_to_exp2
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  func.func @hoist_with_extreme_initializers(%input: tensor<2x3xf32>) -> tensor<2xf32> {
    %seed = arith.constant 0x7F7FFFFF : f32
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %c = ta.constant 2.0 : f32 : !ta.expr<f32, []>
      %a = ta.reduce <max> %x init(0xFF800000 : f32) {axes = #ta.axes<j>, ta.import_group = 1 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %b = ta.reduce <max> %x init(0x7F800000 : f32) {axes = #ta.axes<j>, ta.import_group = 1 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %d = ta.reduce <max> %x init(%seed : f32) {axes = #ta.axes<j>, ta.import_group = 1 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %sa = ta.mul %c, %a {ta.import_group = 2 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %sb = ta.mul %c, %b {ta.import_group = 2 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %sd = ta.mul %c, %d {ta.import_group = 2 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %ab = ta.add %sa, %sb : (!ta.expr<f32, [i]>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %result = ta.add %ab, %sd : (!ta.expr<f32, [i]>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      ta.yield %result : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }
}
// -----

// Other finite bounds (attribute or SSA), NaNs, and unknown seeds block hoisting.
// CHECK-LABEL: func.func @do_not_hoist_with_other_initializers
// CHECK: %[[SEED:.+]] = arith.constant -1.000000e+03 : f32
// CHECK: %[[X:.+]] = ta.at
// CHECK: %[[C:.+]] = ta.constant 2.000000e+00 : f32
// CHECK: %[[A:.+]] = ta.reduce <max> %[[X]] init(-1.000000e+03 : f32)
// CHECK: %[[B:.+]] = ta.reduce <max> %[[X]] init(%[[SEED]] : f32)
// CHECK: %[[D:.+]] = ta.reduce <max> %[[X]] init(%arg1 : f32)
// CHECK: %[[N:.+]] = ta.reduce <max> %[[X]] init(0x7FC00000 : f32)
// CHECK: %[[SA:.+]] = ta.mul %[[C]], %[[A]]
// CHECK: %[[SB:.+]] = ta.mul %[[C]], %[[B]]
// CHECK: %[[SD:.+]] = ta.mul %[[C]], %[[D]]
// CHECK: %[[SN:.+]] = ta.mul %[[C]], %[[N]]
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.exp_to_exp2
    } : !transform.any_op
    transform.yield
  }

  func.func @do_not_hoist_with_other_initializers(%input: tensor<2x3xf32>, %init: f32) -> tensor<2xf32> {
    %seed = arith.constant -1000.0 : f32
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %c = ta.constant 2.0 : f32 : !ta.expr<f32, []>
      %a = ta.reduce <max> %x init(-1000.0 : f32) {axes = #ta.axes<j>, ta.import_group = 1 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %b = ta.reduce <max> %x init(%seed : f32) {axes = #ta.axes<j>, ta.import_group = 1 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %d = ta.reduce <max> %x init(%init : f32) {axes = #ta.axes<j>, ta.import_group = 1 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %n = ta.reduce <max> %x init(0x7FC00000 : f32) {axes = #ta.axes<j>, ta.import_group = 1 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %sa = ta.mul %c, %a {ta.import_group = 2 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %sb = ta.mul %c, %b {ta.import_group = 2 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %sd = ta.mul %c, %d {ta.import_group = 2 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %sn = ta.mul %c, %n {ta.import_group = 2 : i64}
          : (!ta.expr<f32, []>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %ab = ta.add %sa, %sb : (!ta.expr<f32, [i]>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %dn = ta.add %sd, %sn : (!ta.expr<f32, [i]>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      %result = ta.add %ab, %dn : (!ta.expr<f32, [i]>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
      ta.yield %result : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }
}
// -----

// CHECK-LABEL: func.func @preserve_log2e_around_add(
// CHECK: %[[ADD:.+]] = ta.add
// CHECK: %[[LOG2E:.+]] = ta.constant 1.44269502 : f32
// CHECK: %[[SCALED_ADD:.+]] = ta.mul %[[LOG2E]], %[[ADD]]
// CHECK: %[[EXP2:.+]] = ta.exp2 %[[SCALED_ADD]]
// CHECK: ta.yield %[[EXP2]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.exp_to_exp2
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  func.func @preserve_log2e_around_add(
      %lhs: tensor<2x3xf32>, %rhs: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %lhs[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %y = ta.at %rhs[%i, %j]
          : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %sum = ta.add %x, %y {ta.import_group = 10 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      %exp = ta.exp %sum {ta.import_group = 11 : i64}
          : (!ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      ta.yield %exp : !ta.expr<f32, [i, j]>
    } : () -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
// -----

// CHECK-LABEL: func.func @hoist_positive_scale_before_masked_max_reduce(
// CHECK: %[[X:.+]] = ta.at %{{.+}}[%i, %j]
// CHECK: %[[NEW_SCALE:.+]] = ta.constant 0.72134751 : f32
// CHECK: %[[SCALED_X:.+]] = ta.mul %[[NEW_SCALE]], %[[X]]
// CHECK: %[[SCALED_NEG_INF:.+]] = ta.constant 0xFF800000 : f32
// CHECK: %[[MASKED:.+]] = ta.select %{{.+}}, %[[SCALED_X]], %[[SCALED_NEG_INF]]
// CHECK: %[[MAX:.+]] = ta.reduce <max> %[[MASKED]] init(-3.40282347E+38 : f32)
// CHECK-SAME: axes = #ta.axes<j>
// CHECK: %[[CENTERED:.+]] = ta.sub %[[MASKED]], %[[MAX]]
// CHECK: %[[P:.+]] = ta.exp2 %[[CENTERED]]
// CHECK: ta.yield %[[P]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.exp_to_exp2
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  func.func @hoist_positive_scale_before_masked_max_reduce(%scores: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
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
      %m = ta.reduce #ta.reduce_kind<max> %masked init(0xFF7FFFFF : f32) {axes = #ta.axes<j>, ta.import_group = 7 : i64}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      %centered = ta.sub %masked, %m {ta.import_group = 8 : i64}
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i]>)
         -> !ta.expr<f32, [i, j]>
      %p = ta.exp %centered {ta.import_group = 9 : i64}
          : (!ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      ta.yield %p : !ta.expr<f32, [i, j]>
    } : () -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
