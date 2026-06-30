// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin --load-pass-plugin=%neptune_ta_plugin %s --transform-interpreter | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map8 = affine_map<(d0, d1, d2, d3) -> ()>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %ta_func = transform.apply_registered_pass "linalg-to-ta" to %func
        : (!transform.any_op) -> !transform.any_op
    transform.ta.rewrite_exp_to_exp2 %ta_func : !transform.any_op
    transform.apply_patterns to %ta_func {
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !transform.any_op
    %linalg_func = transform.apply_registered_pass "ta-to-linalg" to %ta_func
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @attention(
  // CHECK-NOT: ta.scope
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK: arith.extf
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK: arith.extf
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: arith.constant 0.83294034 : f32
  // CHECK: arith.mulf
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK: arith.maximumf
  // CHECK: arith.subf
  // CHECK: math.exp2
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: arith.extf
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: arith.divf
  // CHECK: arith.truncf
  // CHECK: return {{.*}} : tensor<1x2x4x3xf16>
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
}
