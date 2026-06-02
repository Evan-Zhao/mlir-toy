// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin --load-pass-plugin=%neptune_ta_plugin --pass-pipeline='builtin.module(func.func(ta-import-linalg))' %s | FileCheck %s

#matmul_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#matmul_rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#matmul_out = affine_map<(d0, d1, d2) -> (d0, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map8 = affine_map<(d0, d1, d2, d3) -> ()>

// CHECK-LABEL: func.func @matmul
// CHECK-NEXT: %[[SCOPE:.+]] = ta.scope axes(%a0 "a0" : index, %a1 "a1" : index, %r2 "r2" : index) {
// CHECK-NEXT:   %[[LHS:.+]] = ta.at %{{.+}}[%a0, %r2] {axes = #ta.axes<a0, r2>, ta.import_group = 0 : i64} : tensor<4x8xf32> -> !ta.expr<f32, [a0, r2]>
// CHECK-NEXT:   %[[RHS:.+]] = ta.at %{{.+}}[%r2, %a1] {axes = #ta.axes<r2, a1>, ta.import_group = 0 : i64} : tensor<8x16xf32> -> !ta.expr<f32, [r2, a1]>
// CHECK-NEXT:   %[[MUL:.+]] = ta.mulf %[[LHS]], %[[RHS]] {ta.import_group = 0 : i64} : (!ta.expr<f32, [a0, r2]>, !ta.expr<f32, [r2, a1]>) -> !ta.expr<f32, [a0, a1, r2]>
// CHECK-NEXT:   %[[DOT:.+]] = ta.reduce <add> %[[MUL]] {axes = #ta.axes<r2>, ta.import_group = 0 : i64} : !ta.expr<f32, [a0, a1, r2]> -> !ta.expr<f32, [a0, a1]>
// CHECK-NEXT:   ta.yield %[[DOT]] : !ta.expr<f32, [a0, a1]>
// CHECK-NEXT: } : () -> tensor<4x16xf32>
// CHECK-NEXT: return %[[SCOPE]] : tensor<4x16xf32>
func.func @matmul(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x16xf32>
  %fill = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%cst : f32) outs(%empty : tensor<4x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x16xf32>
  %0 = linalg.generic {
      indexing_maps = [#matmul_lhs, #matmul_rhs, #matmul_out],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%fill : tensor<4x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @attention
// CHECK-NEXT: %[[SCOPE:.+]] = ta.scope axes(%a0 "a0" : index, %a1 "a1" : index, %a2 "a2" : index, %a3 "a3" : index, %r4 "r4" : index, %r7 "r7" : index) {
// CHECK-NEXT:   %[[Q16:.+]] = ta.at %{{.+}}[%a0, %a1, %a2, %r7] {axes = #ta.axes<a0, a1, a2, r7>, ta.import_group = 0 : i64} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [a0, a1, a2, r7]>
// CHECK-NEXT:   %[[Q:.+]] = ta.extf %[[Q16]] {ta.import_group = 0 : i64} : (!ta.expr<f16, [a0, a1, a2, r7]>) -> !ta.expr<f32, [a0, a1, a2, r7]>
// CHECK-NEXT:   %[[K16:.+]] = ta.at %{{.+}}[%a0, %a1, %r4, %r7] {axes = #ta.axes<a0, a1, r4, r7>, ta.import_group = 1 : i64} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [a0, a1, r4, r7]>
// CHECK-NEXT:   %[[K:.+]] = ta.extf %[[K16]] {ta.import_group = 1 : i64} : (!ta.expr<f16, [a0, a1, r4, r7]>) -> !ta.expr<f32, [a0, a1, r4, r7]>
// CHECK-NEXT:   %[[QK:.+]] = ta.mulf %[[Q]], %[[K]] {ta.import_group = 3 : i64} : (!ta.expr<f32, [a0, a1, a2, r7]>, !ta.expr<f32, [a0, a1, r4, r7]>) -> !ta.expr<f32, [a0, a1, a2, r4, r7]>
// CHECK-NEXT:   %[[DOT:.+]] = ta.reduce <add> %[[QK]] {axes = #ta.axes<r7>, ta.import_group = 3 : i64} : !ta.expr<f32, [a0, a1, a2, r4, r7]> -> !ta.expr<f32, [a0, a1, a2, r4]>
// CHECK-NEXT:   %[[SCALE:.+]] = ta.constant 0.577350259 : f32 {ta.import_group = 4 : i64} : !ta.expr<f32, []>
// CHECK-NEXT:   %[[SCORES:.+]] = ta.mulf %[[DOT]], %[[SCALE]] {ta.import_group = 4 : i64} : (!ta.expr<f32, [a0, a1, a2, r4]>, !ta.expr<f32, []>) -> !ta.expr<f32, [a0, a1, a2, r4]>
// CHECK-NEXT:   %[[MAX:.+]] = ta.reduce <max> %[[SCORES]] {axes = #ta.axes<r4>, ta.import_group = 5 : i64} : !ta.expr<f32, [a0, a1, a2, r4]> -> !ta.expr<f32, [a0, a1, a2]>
// CHECK-NEXT:   %[[CENTERED:.+]] = ta.subf %[[SCORES]], %[[MAX]] {ta.import_group = 6 : i64} : (!ta.expr<f32, [a0, a1, a2, r4]>, !ta.expr<f32, [a0, a1, a2]>) -> !ta.expr<f32, [a0, a1, a2, r4]>
// CHECK-NEXT:   %[[EXP:.+]] = ta.exp %[[CENTERED]] {ta.import_group = 7 : i64} : (!ta.expr<f32, [a0, a1, a2, r4]>) -> !ta.expr<f32, [a0, a1, a2, r4]>
// CHECK-NEXT:   %[[DEN:.+]] = ta.reduce <add> %[[EXP]] {axes = #ta.axes<r4>, ta.import_group = 8 : i64} : !ta.expr<f32, [a0, a1, a2, r4]> -> !ta.expr<f32, [a0, a1, a2]>
// CHECK-NEXT:   %[[PROB:.+]] = ta.divf %[[EXP]], %[[DEN]] {ta.import_group = 9 : i64} : (!ta.expr<f32, [a0, a1, a2, r4]>, !ta.expr<f32, [a0, a1, a2]>) -> !ta.expr<f32, [a0, a1, a2, r4]>
// CHECK-NEXT:   %[[V16:.+]] = ta.at %{{.+}}[%a0, %a1, %r4, %a3] {axes = #ta.axes<a0, a1, r4, a3>, ta.import_group = 10 : i64} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [a0, a1, r4, a3]>
// CHECK-NEXT:   %[[V:.+]] = ta.extf %[[V16]] {ta.import_group = 10 : i64} : (!ta.expr<f16, [a0, a1, r4, a3]>) -> !ta.expr<f32, [a0, a1, a3, r4]>
// CHECK-NEXT:   %[[PV:.+]] = ta.mulf %[[PROB]], %[[V]] {ta.import_group = 11 : i64} : (!ta.expr<f32, [a0, a1, a2, r4]>, !ta.expr<f32, [a0, a1, a3, r4]>) -> !ta.expr<f32, [a0, a1, a2, a3, r4]>
// CHECK-NEXT:   %[[NUM:.+]] = ta.reduce <add> %[[PV]] {axes = #ta.axes<r4>, ta.import_group = 11 : i64} : !ta.expr<f32, [a0, a1, a2, a3, r4]> -> !ta.expr<f32, [a0, a1, a2, a3]>
// CHECK-NEXT:   %[[OUT:.+]] = ta.truncf %[[NUM]] {ta.import_group = 12 : i64} : (!ta.expr<f32, [a0, a1, a2, a3]>) -> !ta.expr<f16, [a0, a1, a2, a3]>
// CHECK-NEXT:   ta.yield %[[OUT]] : !ta.expr<f16, [a0, a1, a2, a3]>
// CHECK-NEXT: } : () -> tensor<1x2x4x3xf16>
// CHECK-NEXT: return %[[SCOPE]] : tensor<1x2x4x3xf16>
func.func @attention(%arg0: tensor<1x2x4x3xf16>,
                     %arg1: tensor<1x2x4x3xf16>,
                     %arg2: tensor<1x2x4x3xf16>) -> tensor<1x2x4x3xf16> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %cst_1 = arith.constant 5.773502691896258e-01 : f32
  %0 = tensor.empty() : tensor<1x2x4x3xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<1x2x4x3xf16>) outs(%0 : tensor<1x2x4x3xf32>) {
  ^bb0(%in: f16, %out: f32):
    %25 = arith.extf %in : f16 to f32
    linalg.yield %25 : f32
  } -> tensor<1x2x4x3xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg1 : tensor<1x2x4x3xf16>) outs(%0 : tensor<1x2x4x3xf32>) {
  ^bb0(%in: f16, %out: f32):
    %25 = arith.extf %in : f16 to f32
    linalg.yield %25 : f32
  } -> tensor<1x2x4x3xf32>
  %3 = tensor.empty() : tensor<1x2x3x4xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%2 : tensor<1x2x4x3xf32>) outs(%3 : tensor<1x2x3x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x2x3x4xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]]
      : tensor<1x2x4x3xf32> into tensor<2x4x3xf32>
  %collapsed_2 = tensor.collapse_shape %4 [[0, 1], [2], [3]]
      : tensor<1x2x3x4xf32> into tensor<2x3x4xf32>
  %5 = tensor.empty() : tensor<2x4x4xf32>
  %6 = linalg.generic {indexing_maps = [#map2, #map3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%cst : f32) outs(%5 : tensor<2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x4xf32>
  %7 = linalg.generic {indexing_maps = [#map4, #map5, #map6],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%collapsed, %collapsed_2 : tensor<2x4x3xf32>, tensor<2x3x4xf32>)
      outs(%6 : tensor<2x4x4xf32>) {
  ^bb0(%in: f32, %in_7: f32, %out: f32):
    %25 = arith.mulf %in, %in_7 : f32
    %26 = arith.addf %out, %25 : f32
    linalg.yield %26 : f32
  } -> tensor<2x4x4xf32>
  %expanded = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 2, 4, 4]
      : tensor<2x4x4xf32> into tensor<1x2x4x4xf32>
  %8 = tensor.empty() : tensor<1x2x4x4xf32>
  %9 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<1x2x4x4xf32>) outs(%8 : tensor<1x2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %25 = arith.mulf %in, %cst_1 : f32
    linalg.yield %25 : f32
  } -> tensor<1x2x4x4xf32>
  %10 = tensor.empty() : tensor<1x2x4xf32>
  %11 = linalg.generic {indexing_maps = [#map2, #map3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%cst_0 : f32) outs(%10 : tensor<1x2x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x2x4xf32>
  %12 = linalg.generic {indexing_maps = [#map, #map6],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%9 : tensor<1x2x4x4xf32>) outs(%11 : tensor<1x2x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %25 = arith.maximumf %in, %out : f32
    linalg.yield %25 : f32
  } -> tensor<1x2x4xf32>
  %expanded_3 = tensor.expand_shape %12 [[0], [1], [2, 3]] output_shape [1, 2, 4, 1]
      : tensor<1x2x4xf32> into tensor<1x2x4x1xf32>
  %13 = linalg.generic {indexing_maps = [#map, #map7, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%9, %expanded_3 : tensor<1x2x4x4xf32>, tensor<1x2x4x1xf32>)
      outs(%8 : tensor<1x2x4x4xf32>) {
  ^bb0(%in: f32, %in_7: f32, %out: f32):
    %25 = arith.subf %in, %in_7 : f32
    linalg.yield %25 : f32
  } -> tensor<1x2x4x4xf32>
  %14 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%13 : tensor<1x2x4x4xf32>) outs(%8 : tensor<1x2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %25 = math.exp %in : f32
    linalg.yield %25 : f32
  } -> tensor<1x2x4x4xf32>
  %15 = tensor.empty() : tensor<1x2x4x1xf32>
  %16 = linalg.generic {indexing_maps = [#map8, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%cst : f32) outs(%15 : tensor<1x2x4x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x2x4x1xf32>
  %17 = linalg.generic {indexing_maps = [#map, #map7],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%14 : tensor<1x2x4x4xf32>) outs(%16 : tensor<1x2x4x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %25 = arith.addf %in, %out : f32
    linalg.yield %25 : f32
  } -> tensor<1x2x4x1xf32>
  %18 = linalg.generic {indexing_maps = [#map, #map7, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%14, %17 : tensor<1x2x4x4xf32>, tensor<1x2x4x1xf32>)
      outs(%8 : tensor<1x2x4x4xf32>) {
  ^bb0(%in: f32, %in_7: f32, %out: f32):
    %25 = arith.divf %in, %in_7 : f32
    linalg.yield %25 : f32
  } -> tensor<1x2x4x4xf32>
  %19 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg2 : tensor<1x2x4x3xf16>) outs(%0 : tensor<1x2x4x3xf32>) {
  ^bb0(%in: f16, %out: f32):
    %25 = arith.extf %in : f16 to f32
    linalg.yield %25 : f32
  } -> tensor<1x2x4x3xf32>
  %collapsed_4 = tensor.collapse_shape %18 [[0, 1], [2], [3]]
      : tensor<1x2x4x4xf32> into tensor<2x4x4xf32>
  %collapsed_5 = tensor.collapse_shape %19 [[0, 1], [2], [3]]
      : tensor<1x2x4x3xf32> into tensor<2x4x3xf32>
  %20 = tensor.empty() : tensor<2x4x3xf32>
  %21 = linalg.generic {indexing_maps = [#map2, #map3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%cst : f32) outs(%20 : tensor<2x4x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x3xf32>
  %22 = linalg.generic {indexing_maps = [#map4, #map5, #map6],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%collapsed_4, %collapsed_5 : tensor<2x4x4xf32>, tensor<2x4x3xf32>)
      outs(%21 : tensor<2x4x3xf32>) {
  ^bb0(%in: f32, %in_7: f32, %out: f32):
    %25 = arith.mulf %in, %in_7 : f32
    %26 = arith.addf %out, %25 : f32
    linalg.yield %26 : f32
  } -> tensor<2x4x3xf32>
  %expanded_6 = tensor.expand_shape %22 [[0, 1], [2], [3]] output_shape [1, 2, 4, 3]
      : tensor<2x4x3xf32> into tensor<1x2x4x3xf32>
  %23 = tensor.empty() : tensor<1x2x4x3xf16>
  %24 = linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded_6 : tensor<1x2x4x3xf32>) outs(%23 : tensor<1x2x4x3xf16>) {
  ^bb0(%in: f32, %out: f16):
    %25 = arith.truncf %in : f32 to f16
    linalg.yield %25 : f16
  } -> tensor<1x2x4x3xf16>
  return %24 : tensor<1x2x4x3xf16>
}
