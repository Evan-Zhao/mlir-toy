// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin --load-pass-plugin=%neptune_ta_plugin --pass-pipeline='builtin.module(func.func(linalg-to-ta))' %s | FileCheck %s

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
#mask = affine_map<(d0, d1) -> (d0, d1)>
#scalar = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @matmul
// CHECK-NEXT: %[[SCOPE:.+]] = ta.scope axes(%i0 "i0" extent 4, %i1 "i1" extent 16, %j0 "j0" extent 8) {
// CHECK-NEXT:   {{.*}} = ta.constant 0.000000e+00 : f32
// CHECK-NEXT:   %[[LHS:.+]] = ta.at %{{.+}}[%i0, %j0] {{.*}} : tensor<4x8xf32> -> !ta.expr<f32, [i0, j0]>
// CHECK-NEXT:   %[[RHS:.+]] = ta.at %{{.+}}[%j0, %i1] {{.*}} : tensor<8x16xf32> -> !ta.expr<f32, [j0, i1]>
// CHECK-NEXT:   %[[MUL:.+]] = ta.mulf %[[LHS]], %[[RHS]] {{.*}} : (!ta.expr<f32, [i0, j0]>, !ta.expr<f32, [j0, i1]>) -> !ta.expr<f32, [i0, j0, i1]>
// CHECK-NEXT:   %[[DOT:.+]] = ta.reduce <add> %[[MUL]] {axes = #ta.axes<j0>{{.*}} : !ta.expr<f32, [i0, j0, i1]> -> !ta.expr<f32, [i0, i1]>
// CHECK-NEXT:   ta.yield %[[DOT]] : !ta.expr<f32, [i0, i1]>
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

// CHECK-LABEL: func.func @index_sitofp
// CHECK: ta.scope axes(%i0 "i0" extent 4)
// CHECK: %[[IDX:.+]] = ta.index %i0{{.*}}!ta.expr<i64, [i0]>
// CHECK: %[[CAST:.+]] = ta.cast %[[IDX]] {{.*}} : (!ta.expr<i64, [i0]>) -> !ta.expr<f32, [i0]>
// CHECK: ta.yield %[[CAST]] : !ta.expr<f32, [i0]>
// CHECK: return {{.*}} : tensor<4xf32>
func.func @index_sitofp() -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%empty : tensor<4xf32>) {
  ^bb0(%out: f32):
    %i = linalg.index 0 : index
    %ii = arith.index_cast %i : index to i64
    %f = arith.sitofp %ii : i64 to f32
    linalg.yield %f : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @attention
// CHECK: %[[SCOPE:.+]] = ta.scope axes(%i0 "i0" extent 1, %i1 "i1" extent 2, %i2 "i2" extent 4, %i3 "i3" extent 3, %j0 "j0" extent 3, %j1 "j1" extent 4) {
// CHECK: %[[Q16:.+]] = ta.at %{{.+}}[%i0, %i1, %i2, %j0] {{.*}} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [i0, i1, i2, j0]>
// CHECK: %[[Q:.+]] = ta.cast %[[Q16]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j0]>
// CHECK: %[[K16:.+]] = ta.at %{{.+}}[%i0, %i1, %j1, %j0] {{.*}} : tensor<1x2x4x3xf16> -> !ta.expr<f16, [i0, i1, j1, j0]>
// CHECK: %[[K:.+]] = ta.cast %[[K16]] {{.*}} -> !ta.expr<f32, [i0, i1, j1, j0]>
// CHECK: %[[QK:.+]] = ta.mulf %[[Q]], %[[K]] {{.*}} -> !ta.expr<f32, [i0, i1, i2, j0, j1]>
// CHECK: %[[DOT:.+]] = ta.reduce <add> %[[QK]] {axes = #ta.axes<j0>{{.*}} -> !ta.expr<f32, [i0, i1, i2, j1]>
// CHECK: %[[SCALE:.+]] = ta.constant 0.577350259 : f32
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

// CHECK-LABEL: func.func @causal_mask
// CHECK: ta.scope axes(%i0 "i0" extent 4, %i1 "i1" extent 4)
// CHECK: ta.at %{{.+}}[%i0, %i1] {{.*}} : tensor<4x4xi1> -> !ta.expr<i1, [i0, i1]>
// CHECK: ta.at %{{.+}}[] {{.*}} : tensor<f32> -> !ta.expr<f32, []>
// CHECK: ta.index %i1{{.*}}!ta.expr<i64, [i1]>
// CHECK: ta.index %i0{{.*}}!ta.expr<i64, [i0]>
// CHECK: ta.cmpi sle{{.*}}-> !ta.expr<i1, [i1, i0]>
// CHECK: ta.select
// CHECK: ta.select{{.*}}-> !ta.expr<f32, [i0, i1]>
// CHECK: return {{.*}} : tensor<4x4xf32>
func.func @causal_mask(%scores: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %false = arith.constant false
  %true = arith.constant dense<true> : tensor<4x4xi1>
  %neg_inf = arith.constant dense<0xFF800000> : tensor<f32>
  %empty = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.generic {
      indexing_maps = [#mask, #mask, #scalar, #mask],
      iterator_types = ["parallel", "parallel"]}
      ins(%true, %scores, %neg_inf : tensor<4x4xi1>, tensor<4x4xf32>, tensor<f32>)
      outs(%empty : tensor<4x4xf32>) {
  ^bb0(%in: i1, %score: f32, %masked: f32, %out: f32):
    %i = linalg.index 0 : index
    %ii = arith.index_cast %i : index to i64
    %j = linalg.index 1 : index
    %jj = arith.index_cast %j : index to i64
    %live = arith.cmpi sle, %jj, %ii : i64
    %pred = arith.select %live, %in, %false : i1
    %selected = arith.select %pred, %score, %masked : f32
    linalg.yield %selected : f32
  } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @packed_unit_linalg_index_mask
// CHECK: ta.scope axes(%i0 "i0" extent 1, %i1 "i1" extent 4, %i2 "i2" extent 4, %j0 "j0" extent 4)
// CHECK: ta.index %i2{{.*}}!ta.expr<i64, [i2]>
// CHECK: ta.index %j0{{.*}}!ta.expr<i64, [j0]>
// CHECK-NOT: ta.index %i0
// CHECK: ta.subst {{.*}}from_axes = #ta.axes<j0, i2>{{.*}}to_axes = #ta.axes<i1, i2>
// CHECK: ta.select{{.*}}-> !ta.expr<f32, [i0, i1, i2]>
// CHECK: return {{.*}} : tensor<1x4x4xf32>
func.func @packed_unit_linalg_index_mask(%scores: tensor<1x4x4xf32>)
    -> tensor<1x4x4xf32> {
  %false = arith.constant false
  %true = arith.constant dense<true> : tensor<4x4xi1>
  %neg_inf = arith.constant dense<0xFF800000> : tensor<f32>
  %mask_empty = tensor.empty() : tensor<4x4xi1>
  %mask = linalg.generic {
      indexing_maps = [#mask, #mask],
      iterator_types = ["parallel", "parallel"]}
      ins(%true : tensor<4x4xi1>) outs(%mask_empty : tensor<4x4xi1>) {
  ^bb0(%in: i1, %out: i1):
    %i = linalg.index 0 : index
    %ii = arith.index_cast %i : index to i64
    %j = linalg.index 1 : index
    %jj = arith.index_cast %j : index to i64
    %live = arith.cmpi sle, %jj, %ii : i64
    %pred = arith.select %live, %in, %false : i1
    linalg.yield %pred : i1
  } -> tensor<4x4xi1>
  %expanded = tensor.expand_shape %mask [[0, 1], [2]] output_shape [1, 4, 4]
      : tensor<4x4xi1> into tensor<1x4x4xi1>
  %empty = tensor.empty() : tensor<1x4x4xf32>
  %0 = linalg.generic {
      indexing_maps = [#map3, #map3, #map2, #map3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%expanded, %scores, %neg_inf
          : tensor<1x4x4xi1>, tensor<1x4x4xf32>, tensor<f32>)
      outs(%empty : tensor<1x4x4xf32>) {
  ^bb0(%pred: i1, %score: f32, %masked: f32, %out: f32):
    %selected = arith.select %pred, %score, %masked : f32
    linalg.yield %selected : f32
  } -> tensor<1x4x4xf32>
  return %0 : tensor<1x4x4xf32>
}
