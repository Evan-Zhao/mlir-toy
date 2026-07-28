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
  // CHECK: %[[SCOPE:.+]] = ta.scope axes(%i0 "i0" extent 1, %i1 "i1" extent 2, %i2 "i2" extent 4, %i3 "i3" extent 3, %j0 "j0" extent 4, %j1 "j1" extent 3) {
  // CHECK: %[[Q:.+]] = ta.cast
  // CHECK-SAME: -> !ta.expr<f32, [i0, i1, i2, j1]>
  // CHECK: %[[K:.+]] = ta.cast
  // CHECK-SAME: -> !ta.expr<f32, [i0, i1, j0, j1]>
  // CHECK: %[[QK:.+]] = ta.mul %[[Q]], %[[K]]
  // CHECK: %[[DOT:.+]] = ta.reduce <add> %[[QK]] {axes = #ta.axes<j1>
  // CHECK: %[[LOG2_SCALE:.+]] = ta.constant 0.83294034 : f32
  // CHECK: %[[SCORES:.+]] = ta.mul %[[LOG2_SCALE]], %[[DOT]]
  // CHECK: %[[MAX:.+]] = ta.reduce <max> %[[SCORES]] {axes = #ta.axes<j0>
  // CHECK: %[[CENTERED:.+]] = ta.sub %[[SCORES]], %[[MAX]]
  // CHECK: %[[EXP:.+]] = ta.exp2 %[[CENTERED]]
  // CHECK: %[[DEN:.+]] = ta.reduce <add> %[[EXP]] {axes = #ta.axes<j0>
  // CHECK: %[[V:.+]] = ta.cast
  // CHECK-SAME: -> !ta.expr<f32, [i0, i1, j0, i3]>
  // CHECK: %[[P_F16:.+]] = ta.cast %[[EXP]]
  // CHECK-SAME: -> !ta.expr<f16, [i0, i1, i2, j0]>
  // CHECK: %[[P_F32:.+]] = ta.cast %[[P_F16]]
  // CHECK-SAME: -> !ta.expr<f32, [i0, i1, i2, j0]>
  // CHECK: %[[PV:.+]] = ta.mul %[[P_F32]], %[[V]]
  // CHECK: %[[NUM:.+]] = ta.reduce <add> %[[PV]] {axes = #ta.axes<j0>
  // CHECK: %[[NORMALIZED:.+]] = ta.div %[[NUM]], %[[DEN]]
  // CHECK: %[[OUT:.+]] = ta.cast %[[NORMALIZED]]
  // CHECK-SAME: -> !ta.expr<f16, [i0, i1, i2, i3]>
  // CHECK: ta.yield %[[OUT]]
  // CHECK: return %[[SCOPE]] : tensor<1x2x4x3xf16>
  func.func @attention(%arg0: tensor<1x2x4x3xf16>, %arg1: tensor<1x2x4x3xf16>, %arg2: tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf16> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<0.57735026918962584> : tensor<1xf64>
    %0 = stablehlo.transpose %arg1, dims = [0, 1, 3, 2] : (tensor<1x2x4x3xf16>) -> tensor<1x2x3x4xf16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x2x4x3xf16>, tensor<1x2x3x4xf16>) -> tensor<1x2x4x4xf32>
    %3 = stablehlo.convert %cst_1 : (tensor<1xf64>) -> tensor<1xf32>
    %4 = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1x2x4x4xf32>
    %6 = stablehlo.multiply %2, %5 : tensor<1x2x4x4xf32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x4x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
    %8 = stablehlo.reshape %7 : (tensor<1x2x4xf32>) -> tensor<1x2x4x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3] : (tensor<1x2x4x1xf32>) -> tensor<1x2x4x4xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<1x2x4x4xf32>
    %11 = stablehlo.exponential %10 : tensor<1x2x4x4xf32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x2x4x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
    %13 = stablehlo.reshape %12 : (tensor<1x2x4xf32>) -> tensor<1x2x4x1xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x2x4x1xf32>) -> tensor<1x2x4x4xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x2x4x4xf32>
    %16 = stablehlo.convert %15 : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf16>
    %17 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 3] : (tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf16>
    %18 = stablehlo.dot_general %16, %17, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x2x4x4xf16>, tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf32>
    %19 = stablehlo.convert %18 : (tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf16>
    return %19 : tensor<1x2x4x3xf16>
  }
}
