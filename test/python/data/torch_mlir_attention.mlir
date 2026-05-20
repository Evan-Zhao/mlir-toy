#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @attention(%arg0: tensor<1x4x128x64xf16>, %arg1: tensor<1x4x128x64xf16>, %arg2: tensor<1x4x128x64xf16>) -> tensor<1x4x128x64xf16> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<1x4x128x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x4x128x64xf16>) outs(%0 : tensor<1x4x128x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x4x128x64xf16>) outs(%0 : tensor<1x4x128x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x64xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x4x128x64xf16>) outs(%0 : tensor<1x4x128x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x64xf32>
    %4 = tensor.empty() : tensor<1x4x64x128xf32>
    %transposed = linalg.transpose ins(%2 : tensor<1x4x128x64xf32>) outs(%4 : tensor<1x4x64x128xf32>) permutation = [0, 1, 3, 2]
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x4x128x64xf32> into tensor<4x128x64xf32>
    %collapsed_2 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x4x64x128xf32> into tensor<4x64x128xf32>
    %5 = tensor.empty() : tensor<4x128x128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %7 = linalg.batch_matmul ins(%collapsed, %collapsed_2 : tensor<4x128x64xf32>, tensor<4x64x128xf32>) outs(%6 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %expanded = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 4, 128, 128] : tensor<4x128x128xf32> into tensor<1x4x128x128xf32>
    %8 = tensor.empty() : tensor<1x4x128x128xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x4x128x128xf32>) outs(%8 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.mulf %in, %cst_1 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x128xf32>
    %10 = tensor.empty() : tensor<1x4x128xi64>
    %11 = linalg.fill ins(%c0_i64 : i64) outs(%10 : tensor<1x4x128xi64>) -> tensor<1x4x128xi64>
    %12 = tensor.empty() : tensor<1x4x128xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<1x4x128xf32>) -> tensor<1x4x128xf32>
    %14:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9 : tensor<1x4x128x128xf32>) outs(%13, %11 : tensor<1x4x128xf32>, tensor<1x4x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_7: i64):
      %29 = linalg.index 3 : index
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.maximumf %in, %out : f32
      %32 = arith.cmpf ogt, %in, %out : f32
      %33 = arith.select %32, %30, %out_7 : i64
      linalg.yield %31, %33 : f32, i64
    } -> (tensor<1x4x128xf32>, tensor<1x4x128xi64>)
    %expanded_3 = tensor.expand_shape %14#0 [[0], [1], [2, 3]] output_shape [1, 4, 128, 1] : tensor<1x4x128xf32> into tensor<1x4x128x1xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %expanded_3 : tensor<1x4x128x128xf32>, tensor<1x4x128x1xf32>) outs(%8 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.subf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x128xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<1x4x128x128xf32>) outs(%8 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.exp %in : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x128xf32>
    %17 = tensor.empty() : tensor<1x4x128x1xf32>
    %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<1x4x128x1xf32>) -> tensor<1x4x128x1xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%16 : tensor<1x4x128x128xf32>) outs(%18 : tensor<1x4x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.addf %in, %out : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x1xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %19 : tensor<1x4x128x128xf32>, tensor<1x4x128x1xf32>) outs(%8 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.divf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x128xf32>
    %21 = tensor.empty() : tensor<1x4x128x128xf16>
    %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20 : tensor<1x4x128x128xf32>) outs(%21 : tensor<1x4x128x128xf16>) {
    ^bb0(%in: f32, %out: f16):
      %29 = arith.truncf %in : f32 to f16
      linalg.yield %29 : f16
    } -> tensor<1x4x128x128xf16>
    %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22 : tensor<1x4x128x128xf16>) outs(%8 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x128xf32>
    %collapsed_4 = tensor.collapse_shape %23 [[0, 1], [2], [3]] : tensor<1x4x128x128xf32> into tensor<4x128x128xf32>
    %collapsed_5 = tensor.collapse_shape %3 [[0, 1], [2], [3]] : tensor<1x4x128x64xf32> into tensor<4x128x64xf32>
    %24 = tensor.empty() : tensor<4x128x64xf32>
    %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %26 = linalg.batch_matmul ins(%collapsed_4, %collapsed_5 : tensor<4x128x128xf32>, tensor<4x128x64xf32>) outs(%25 : tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %expanded_6 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 4, 128, 64] : tensor<4x128x64xf32> into tensor<1x4x128x64xf32>
    %27 = tensor.empty() : tensor<1x4x128x64xf16>
    %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<1x4x128x64xf32>) outs(%27 : tensor<1x4x128x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %29 = arith.truncf %in : f32 to f16
      linalg.yield %29 : f16
    } -> tensor<1x4x128x64xf16>
    return %28 : tensor<1x4x128x64xf16>
  }
}
