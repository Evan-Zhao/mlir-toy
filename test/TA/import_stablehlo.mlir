// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin --load-pass-plugin=%neptune_ta_plugin --pass-pipeline='builtin.module(func.func(stablehlo-to-ta))' %s | FileCheck %s

// CHECK-LABEL: func.func @attention
// CHECK-NEXT: %[[SCOPE:.+]] = ta.scope axes(%i0 "i0" extent 1, %i1 "i1" extent 2, %i2 "i2" extent 4, %i3 "i3" extent 3, %j0 "j0" extent 3, %j1 "j1" extent 4) {
// CHECK: %[[Q16:.+]] = ta.at %{{.+}}[%i0, %i1, %i2, %j0] {{.*}} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [i0, i1, i2, j0]>
// CHECK: %[[Q:.+]] = ta.cast %[[Q16]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j0]>
// CHECK: %[[K16:.+]] = ta.at %{{.+}}[%i0, %i1, %j1, %j0] {{.*}} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [i0, i1, j1, j0]>
// CHECK: %[[K:.+]] = ta.cast %[[K16]] {{.*}} -> !ta.expr<f32, [i0, i1, j1, j0]>
// CHECK: %[[QK:.+]] = ta.mulf %[[Q]], %[[K]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j0, j1]>
// CHECK: %[[DOT:.+]] = ta.reduce <add> %[[QK]] {axes = #ta.axes<j0>{{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[SCALE64:.+]] = ta.constant 0.5773502691896257{{.*}} : !ta.expr<f64, []>
// CHECK: %[[SCALE:.+]] = ta.cast %[[SCALE64]] {{.*}} -> !ta.expr<f32, []>
// CHECK: %[[SCORES:.+]] = ta.mulf %[[DOT]], %[[SCALE]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[MAX:.+]] = ta.reduce <max> %[[SCORES]] {axes = #ta.axes<j1>{{.*}} -> !ta.expr<f32, [i0, i1, i2]>
// CHECK: %[[CENTERED:.+]] = ta.subf %[[SCORES]], %[[MAX]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[EXP:.+]] = ta.exp %[[CENTERED]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[DEN:.+]] = ta.reduce <add> %[[EXP]] {axes = #ta.axes<j1>{{.*}} -> !ta.expr<f32, [i0, i1, i2]>
// CHECK: %[[PROB:.+]] = ta.divf %[[EXP]], %[[DEN]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[V16:.+]] = ta.at %{{.+}}[%i0, %i1, %j1, %i3] {{.*}} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [i0, i1, j1, i3]>
// CHECK: %[[V:.+]] = ta.cast %[[V16]] {{.*}} -> !ta.expr<f32, [i0, i1, j1, i3]>
// CHECK: %[[PV:.+]] = ta.mulf %[[PROB]], %[[V]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j1, i3]>
// CHECK: %[[NUM:.+]] = ta.reduce <add> %[[PV]] {axes = #ta.axes<j1>{{.*}} -> !ta.expr<f32, [i0, i1, i2, i3]>
// CHECK: %[[OUT:.+]] = ta.cast %[[NUM]] {{.*}} -> !ta.expr<f16, [i0, i1, i2, i3]>
// CHECK: ta.yield %[[OUT]] : !ta.expr<f16, [i0, i1, i2, i3]>
// CHECK: return %[[SCOPE]] : tensor<1x2x4x3xf16>
func.func @attention(%q: tensor<1x2x4x3xf16>, %k: tensor<1x2x4x3xf16>,
                     %v: tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf16> {
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %scale64 = arith.constant dense<0.5773502691896257> : tensor<1xf64>
  %q32 = stablehlo.convert %q : (tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf32>
  %k32 = stablehlo.convert %k : (tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf32>
  %kt = stablehlo.transpose %k32, dims = [0, 1, 3, 2]
      : (tensor<1x2x4x3xf32>) -> tensor<1x2x3x4xf32>
  %kt_bcast = stablehlo.broadcast_in_dim %kt, dims = [0, 1, 2, 3]
      : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %qk = stablehlo.dot_general %q32, %kt_bcast,
      batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2]
      : (tensor<1x2x4x3xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x4x4xf32>
  %scale1 = stablehlo.convert %scale64 : (tensor<1xf64>) -> tensor<1xf32>
  %scale0 = stablehlo.reshape %scale1 : (tensor<1xf32>) -> tensor<f32>
  %scale = stablehlo.broadcast_in_dim %scale0, dims = []
      : (tensor<f32>) -> tensor<1x2x4x4xf32>
  %scores = stablehlo.multiply %qk, %scale : tensor<1x2x4x4xf32>
  %max = stablehlo.reduce(%scores init: %neg_inf) applies stablehlo.maximum
      across dimensions = [3]
      : (tensor<1x2x4x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
  %max1 = stablehlo.reshape %max : (tensor<1x2x4xf32>) -> tensor<1x2x4x1xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max1, dims = [0, 1, 2, 3]
      : (tensor<1x2x4x1xf32>) -> tensor<1x2x4x4xf32>
  %centered = stablehlo.subtract %scores, %max_bcast : tensor<1x2x4x4xf32>
  %exp = stablehlo.exponential %centered : tensor<1x2x4x4xf32>
  %sum = stablehlo.reduce(%exp init: %zero) applies stablehlo.add
      across dimensions = [3]
      : (tensor<1x2x4x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
  %sum1 = stablehlo.reshape %sum : (tensor<1x2x4xf32>) -> tensor<1x2x4x1xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum1, dims = [0, 1, 2, 3]
      : (tensor<1x2x4x1xf32>) -> tensor<1x2x4x4xf32>
  %prob = stablehlo.divide %exp, %sum_bcast : tensor<1x2x4x4xf32>
  %v32 = stablehlo.convert %v : (tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf32>
  %v_bcast = stablehlo.broadcast_in_dim %v32, dims = [0, 1, 2, 3]
      : (tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
  %pv = stablehlo.dot_general %prob, %v_bcast,
      batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2]
      : (tensor<1x2x4x4xf32>, tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
  %out = stablehlo.convert %pv : (tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf16>
  return %out : tensor<1x2x4x3xf16>
}
