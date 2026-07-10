// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin --load-pass-plugin=%neptune_ta_plugin --pass-pipeline='builtin.module(func.func(stablehlo-to-ta))' %s | FileCheck %s

// CHECK-LABEL: func.func @attention
// CHECK-NEXT: %[[SCOPE:.+]] = ta.scope axes(%i0 "i0" extent 1, %i1 "i1" extent 2, %i2 "i2" extent 4, %i3 "i3" extent 3, %j0 "j0" extent 3, %j1 "j1" extent 4) {
// CHECK: %[[Q16:.+]] = ta.at %{{.+}}[%i0, %i1, %i2, %j0] {{.*}} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [i0, i1, i2, j0]>
// CHECK: %[[Q:.+]] = ta.cast %[[Q16]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j0]>
// CHECK: %[[K16:.+]] = ta.at %{{.+}}[%i0, %i1, %j1, %j0] {{.*}} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [i0, i1, j1, j0]>
// CHECK: %[[K:.+]] = ta.cast %[[K16]] {{.*}} -> !ta.expr<f32, [i0, i1, j1, j0]>
// CHECK: %[[QK:.+]] = ta.mul %[[Q]], %[[K]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j0, j1]>
// CHECK: %[[DOT:.+]] = ta.reduce <add> %[[QK]] {axes = #ta.axes<j0>{{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[SCALE64:.+]] = ta.constant 0.5773502691896257{{.*}} : !ta.expr<f64, []>
// CHECK: %[[SCALE:.+]] = ta.cast %[[SCALE64]] {{.*}} -> !ta.expr<f32, []>
// CHECK: %[[SCORES:.+]] = ta.mul %[[DOT]], %[[SCALE]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[MAX:.+]] = ta.reduce <max> %[[SCORES]] {axes = #ta.axes<j1>{{.*}} -> !ta.expr<f32, [i0, i1, i2]>
// CHECK: %[[CENTERED:.+]] = ta.sub %[[SCORES]], %[[MAX]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[EXP:.+]] = ta.exp %[[CENTERED]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[DEN:.+]] = ta.reduce <add> %[[EXP]] {axes = #ta.axes<j1>{{.*}} -> !ta.expr<f32, [i0, i1, i2]>
// CHECK: %[[PROB:.+]] = ta.div %[[EXP]], %[[DEN]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[V16:.+]] = ta.at %{{.+}}[%i0, %i1, %j1, %i3] {{.*}} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [i0, i1, j1, i3]>
// CHECK: %[[V:.+]] = ta.cast %[[V16]] {{.*}} -> !ta.expr<f32, [i0, i1, j1, i3]>
// CHECK: %[[PV:.+]] = ta.mul %[[PROB]], %[[V]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j1, i3]>
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

// CHECK-LABEL: func.func @partition_around_unsupported_ops
// CHECK: %[[LHS:.+]] = stablehlo.slice
// CHECK: %[[RHS:.+]] = stablehlo.slice
// CHECK: %[[SUM:.+]] = ta.scope axes(%i0 "i0" extent 4, %i1 "i1" extent 3) {
// CHECK:   %[[L:.+]] = ta.at %[[LHS]][%i0, %i1]
// CHECK:   %[[R:.+]] = ta.at %[[RHS]][%i0, %i1]
// CHECK:   %[[ADD:.+]] = ta.add %[[L]], %[[R]]
// CHECK:   ta.yield %[[ADD]]
// CHECK: stablehlo.concatenate %[[SUM]], %[[LHS]], dim = 0
func.func @partition_around_unsupported_ops(%arg: tensor<5x3xf32>) -> tensor<8x3xf32> {
  %lhs = stablehlo.slice %arg [0:4, 0:3]
      : (tensor<5x3xf32>) -> tensor<4x3xf32>
  %rhs = stablehlo.slice %arg [1:5, 0:3]
      : (tensor<5x3xf32>) -> tensor<4x3xf32>
  %sum = stablehlo.add %lhs, %rhs : tensor<4x3xf32>
  %result = stablehlo.concatenate %sum, %lhs, dim = 0
      : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<8x3xf32>
  return %result : tensor<8x3xf32>
}

// CHECK-LABEL: func.func @partition_permuted_dot
// CHECK: %[[DOT:.+]] = ta.scope axes(%i0 "i0" extent 2, %i1 "i1" extent 3, %i2 "i2" extent 4, %i3 "i3" extent 3, %j0 "j0" extent 5)
// CHECK: %[[ADAPTED:.+]] = stablehlo.broadcast_in_dim %[[DOT]], dims = [0, 2, 1, 3] : (tensor<2x3x4x3xf32>) -> tensor<2x4x3x3xf32>
// CHECK: %[[SLICE:.+]] = stablehlo.slice %[[ADAPTED]]
// CHECK: %[[EXP_SCOPE:.+]] = ta.scope axes(%i0 "i0" extent 2, %i1 "i1" extent 3, %i2 "i2" extent 4, %i3 "i3" extent 3)
// CHECK: ta.at %[[DOT]]{{.*}} {ta.import_group = 1 : i64}
// CHECK: ta.exp {{.*}} {ta.import_group = 2 : i64}
// CHECK: %[[EXP_ADAPTED:.+]] = stablehlo.broadcast_in_dim %[[EXP_SCOPE]], dims = [0, 2, 1, 3]
// CHECK: stablehlo.concatenate %[[SLICE]], %[[EXP_ADAPTED]]
func.func @partition_permuted_dot(%q: tensor<2x3x4x5xf16>,
                                  %k: tensor<2x3x4x5xf16>)
    -> tensor<2x4x6x3xf32> {
  %dot = stablehlo.dot_general %q, %k,
      batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3]
      : (tensor<2x3x4x5xf16>, tensor<2x3x4x5xf16>) -> tensor<2x4x3x3xf32>
  %slice = stablehlo.slice %dot [0:2, 0:4, 0:3, 0:3]
      : (tensor<2x4x3x3xf32>) -> tensor<2x4x3x3xf32>
  %exp = stablehlo.exponential %dot : tensor<2x4x3x3xf32>
  %result = stablehlo.concatenate %slice, %exp, dim = 2
      : (tensor<2x4x3x3xf32>, tensor<2x4x3x3xf32>) -> tensor<2x4x6x3xf32>
  return %result : tensor<2x4x6x3xf32>
}

// CHECK-LABEL: func.func @select
// CHECK: %[[SCOPE:.+]] = ta.scope axes(%i0 "i0" extent 2, %i1 "i1" extent 3) {
// CHECK: %[[PRED:.+]] = ta.at %{{.+}}[%i0, %i1]{{.*}}!ta.expr<i1, [i0, i1]>
// CHECK: %[[TRUE:.+]] = ta.at %{{.+}}[%i0, %i1]{{.*}}!ta.expr<f32, [i0, i1]>
// CHECK: %[[FALSE:.+]] = ta.at %{{.+}}[%i0, %i1]{{.*}}!ta.expr<f32, [i0, i1]>
// CHECK: %[[SELECTED:.+]] = ta.select %[[PRED]], %[[TRUE]], %[[FALSE]]{{.*}}!ta.expr<f32, [i0, i1]>
// CHECK: %[[SCALAR_PRED:.+]] = ta.at %{{.+}}[]{{.*}}!ta.expr<i1, []>
// CHECK: %[[RESULT:.+]] = ta.select %[[SCALAR_PRED]], %[[SELECTED]], %{{.+}}{{.*}}!ta.expr<f32, [i0, i1]>
// CHECK: ta.yield %[[RESULT]]
// CHECK: return %[[SCOPE]] : tensor<2x3xf32>
func.func @select(%pred: tensor<2x3xi1>, %scalar_pred: tensor<i1>,
                  %on_true: tensor<2x3xf32>, %on_false: tensor<2x3xf32>)
    -> tensor<2x3xf32> {
  %selected = stablehlo.select %pred, %on_true, %on_false
      : tensor<2x3xi1>, tensor<2x3xf32>
  %result = stablehlo.select %scalar_pred, %selected, %on_false
      : tensor<i1>, tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @compare
// CHECK: ta.cmpf une,
// CHECK: ta.cmpi slt,
// CHECK: ta.cmpi uge,
// CHECK: ta.select
// CHECK: return %{{.+}} : tensor<2x3xf32>
func.func @compare(%flhs: tensor<2x3xf32>, %frhs: tensor<2x3xf32>,
                   %ilhs: tensor<2x3xi32>, %irhs: tensor<2x3xi32>)
    -> tensor<2x3xf32> {
  %float_pred = stablehlo.compare NE, %flhs, %frhs, FLOAT
      : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
  %signed_pred = stablehlo.compare LT, %ilhs, %irhs, SIGNED
      : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
  %unsigned_pred = stablehlo.compare GE, %ilhs, %irhs, UNSIGNED
      : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
  %float_selected = stablehlo.select %float_pred, %flhs, %frhs
      : tensor<2x3xi1>, tensor<2x3xf32>
  %signed_selected = stablehlo.select %signed_pred, %float_selected, %frhs
      : tensor<2x3xi1>, tensor<2x3xf32>
  %result = stablehlo.select %unsigned_pred, %signed_selected, %frhs
      : tensor<2x3xi1>, tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @integer_iota
// CHECK: %[[SCOPE:.+]] = ta.scope axes(%i0 "i0" extent 3) {
// CHECK: %[[INDEX:.+]] = ta.index %i0{{.*}}!ta.expr<i32, [i0]>
// CHECK: ta.yield %[[INDEX]]
// CHECK: stablehlo.broadcast_in_dim %[[SCOPE]], dims = [1] : (tensor<3xi32>) -> tensor<2x3xi32>
func.func @integer_iota() -> tensor<2x3xi32> {
  %result = stablehlo.iota dim = 1 : tensor<2x3xi32>
  return %result : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @float_iota
// CHECK: %[[SCOPE:.+]] = ta.scope axes(%i0 "i0" extent 4) {
// CHECK: %[[INDEX:.+]] = ta.index %i0{{.*}}!ta.expr<i64, [i0]>
// CHECK: %[[CAST:.+]] = ta.cast %[[INDEX]]{{.*}}!ta.expr<f32, [i0]>
// CHECK: ta.yield %[[CAST]]
func.func @float_iota() -> tensor<4xf32> {
  %result = stablehlo.iota dim = 0 : tensor<4xf32>
  return %result : tensor<4xf32>
}
