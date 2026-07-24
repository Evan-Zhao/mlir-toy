// RUN: neptune-opt %s --transform-interpreter | FileCheck %s

// Exercise the TA portion of the attention schedule on exact StableHLO exporter output: import the
// scalar expression graph, rewrite exp to exp2, and sink softmax normalization after P @ V.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %ta_func = transform.apply_registered_pass "stablehlo-to-ta" to %func
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %ta_func {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.ta.exp_to_exp2
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !transform.any_op
    transform.apply_cse to %ta_func : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @attention(
  // CHECK: %[[SCOPE:.+]] = ta.scope axes(%i0 "i0" extent 1, %i1 "i1" extent 2, %i2 "i2" extent 4, %i3 "i3" extent 3, %j0 "j0" extent 3, %j1 "j1" extent 4) {
  // CHECK: %[[Q:.+]] = ta.cast
  // CHECK-SAME: -> !ta.expr<f32, [i0, i1, i2, j0]>
  // CHECK: %[[K:.+]] = ta.cast
  // CHECK-SAME: -> !ta.expr<f32, [i0, i1, j1, j0]>
  // CHECK: %[[QK:.+]] = ta.mul %[[Q]], %[[K]]
  // CHECK: %[[DOT:.+]] = ta.reduce <add> %[[QK]] {axes = #ta.axes<j0>
  // CHECK: %[[LOG2_SCALE:.+]] = ta.constant 0.83294034 : f32
  // CHECK: %[[SCORES:.+]] = ta.mul %[[LOG2_SCALE]], %[[DOT]]
  // CHECK: %[[MAX:.+]] = ta.reduce <max> %[[SCORES]] {axes = #ta.axes<j1>
  // CHECK: %[[CENTERED:.+]] = ta.sub %[[SCORES]], %[[MAX]]
  // CHECK: %[[EXP:.+]] = ta.exp2 %[[CENTERED]]
  // CHECK: %[[DEN:.+]] = ta.reduce <add> %[[EXP]] {axes = #ta.axes<j1>
  // CHECK: %[[V:.+]] = ta.cast
  // CHECK-SAME: -> !ta.expr<f32, [i0, i1, j1, i3]>
  // CHECK: %[[PV:.+]] = ta.mul %[[EXP]], %[[V]]
  // CHECK: %[[NUM:.+]] = ta.reduce <add> %[[PV]] {axes = #ta.axes<j1>
  // CHECK: %[[NORMALIZED:.+]] = ta.div %[[NUM]], %[[DEN]]
  // CHECK: %[[OUT:.+]] = ta.cast %[[NORMALIZED]]
  // CHECK-SAME: -> !ta.expr<f16, [i0, i1, i2, i3]>
  // CHECK: ta.yield %[[OUT]]
  // CHECK: return %[[SCOPE]] : tensor<1x2x4x3xf16>
  func.func @attention(%arg0: tensor<1x2x4x3xf16>, %arg1: tensor<1x2x4x3xf16>,
                       %arg2: tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf16> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<0.57735026918962584> : tensor<1xf64>
    %0 = stablehlo.convert %arg0 : (tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf32>
    %1 = stablehlo.convert %arg1 : (tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1, 3, 2] : (tensor<1x2x4x3xf32>) -> tensor<1x2x3x4xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
    %4 = stablehlo.dot_general %0, %3, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x2x4x3xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x4x4xf32>
    %5 = stablehlo.convert %cst_1 : (tensor<1xf64>) -> tensor<1xf32>
    %6 = stablehlo.reshape %5 : (tensor<1xf32>) -> tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<1x2x4x4xf32>
    %8 = stablehlo.multiply %4, %7 : tensor<1x2x4x4xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x4x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
    %10 = stablehlo.reshape %9 : (tensor<1x2x4xf32>) -> tensor<1x2x4x1xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2, 3] : (tensor<1x2x4x1xf32>) -> tensor<1x2x4x4xf32>
    %12 = stablehlo.subtract %8, %11 : tensor<1x2x4x4xf32>
    %13 = stablehlo.exponential %12 : tensor<1x2x4x4xf32>
    %14 = stablehlo.reduce(%13 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x2x4x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x2x4xf32>) -> tensor<1x2x4x1xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2, 3] : (tensor<1x2x4x1xf32>) -> tensor<1x2x4x4xf32>
    %17 = stablehlo.divide %13, %16 : tensor<1x2x4x4xf32>
    %18 = stablehlo.convert %arg2 : (tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2, 3] : (tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
    %20 = stablehlo.dot_general %17, %19, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x2x4x4xf32>, tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
    %21 = stablehlo.convert %20 : (tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf16>
    return %21 : tensor<1x2x4x3xf16>
  }
}
