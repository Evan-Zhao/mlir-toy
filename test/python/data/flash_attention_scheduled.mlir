#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
module {
  func.func @attention(%arg0: tensor<1x32x4096x128xf16>, %arg1: tensor<1x32x4096x128xf16>, %arg2: tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16> {
    %cst = arith.constant 1.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0xFFC00000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.0883883461 : f32
    %0 = tensor.empty() : tensor<1x32x4096x4096xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %2 = tensor.empty() : tensor<1x32x4096xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %5 = tensor.empty() : tensor<1x32x4096x4096xf16>
    %6 = tensor.empty() : tensor<1x32x4096x128xf32>
    %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<1x32x4096x128xf32>) -> tensor<1x32x4096x128xf32>
    %8 = tensor.empty() : tensor<1x32x4096x128xf16>
    %9 = scf.forall (%arg3, %arg4) in (32, 32) shared_outs(%arg5 = %8) -> (tensor<1x32x4096x128xf16>) {
      %10 = affine.apply #map(%arg4)
      %extracted_slice = tensor.extract_slice %0[0, %arg3, %10, 0] [1, 1, 128, 4096] [1, 1, 1, 1] : tensor<1x32x4096x4096xf32> to tensor<1x1x128x4096xf32>
      %extracted_slice_3 = tensor.extract_slice %3[0, %arg3, %10] [1, 1, 128] [1, 1, 1] : tensor<1x32x4096xf32> to tensor<1x1x128xf32>
      %extracted_slice_4 = tensor.extract_slice %4[0, %arg3, %10] [1, 1, 128] [1, 1, 1] : tensor<1x32x4096xf32> to tensor<1x1x128xf32>
      %extracted_slice_5 = tensor.extract_slice %5[0, %arg3, %10, 0] [1, 1, 128, 4096] [1, 1, 1, 1] : tensor<1x32x4096x4096xf16> to tensor<1x1x128x4096xf16>
      %extracted_slice_6 = tensor.extract_slice %arg2[0, %arg3, 0, 0] [1, 1, 4096, 128] [1, 1, 1, 1] : tensor<1x32x4096x128xf16> to tensor<1x1x4096x128xf16>
      %extracted_slice_7 = tensor.extract_slice %7[0, %arg3, %10, 0] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x32x4096x128xf32> to tensor<1x1x128x128xf32>
      %11:8 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %extracted_slice, %arg8 = %extracted_slice_3, %arg9 = %extracted_slice, %arg10 = %extracted_slice_4, %arg11 = %extracted_slice, %arg12 = %extracted_slice, %arg13 = %extracted_slice_5, %arg14 = %extracted_slice_7) -> (tensor<1x1x128x4096xf32>, tensor<1x1x128xf32>, tensor<1x1x128x4096xf32>, tensor<1x1x128xf32>, tensor<1x1x128x4096xf32>, tensor<1x1x128x4096xf32>, tensor<1x1x128x4096xf16>, tensor<1x1x128x128xf32>) {
        %13 = affine.apply #map(%arg6)
        %extracted_slice_9 = tensor.extract_slice %arg0[0, %arg3, %10, 0] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x32x4096x128xf16> to tensor<1x1x128x128xf16>
        %extracted_slice_10 = tensor.extract_slice %arg1[0, %arg3, %13, 0] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x32x4096x128xf16> to tensor<1x1x128x128xf16>
        %extracted_slice_11 = tensor.extract_slice %1[0, %arg3, %10, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x32x4096x4096xf32> to tensor<1x1x128x128xf32>
        %14 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) outs(%extracted_slice_11 : tensor<1x1x128x128xf32>) {
        ^bb0(%in: f16, %in_22: f16, %out: f32):
          %24 = arith.extf %in : f16 to f32
          %25 = arith.extf %in_22 : f16 to f32
          %26 = arith.mulf %24, %25 : f32
          %27 = arith.addf %out, %26 : f32
          linalg.yield %27 : f32
        } -> tensor<1x1x128x128xf32>
        %extracted_slice_12 = tensor.extract_slice %0[0, %arg3, %10, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x32x4096x4096xf32> to tensor<1x1x128x128xf32>
        %mapped = linalg.map ins(%14 : tensor<1x1x128x128xf32>) outs(%extracted_slice_12 : tensor<1x1x128x128xf32>)
          (%in: f32, %init: f32) {
            %24 = arith.mulf %in, %cst_2 : f32
            linalg.yield %24 : f32
          }
        %15 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%mapped : tensor<1x1x128x128xf32>) outs(%arg8 : tensor<1x1x128xf32>) {
        ^bb0(%in: f32, %out: f32):
          %24 = arith.maxnumf %in, %out : f32
          linalg.yield %24 : f32
        } -> tensor<1x1x128xf32>
        %extracted_slice_13 = tensor.extract_slice %arg9[0, 0, 0, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x4096xf32> to tensor<1x1x128x128xf32>
        %16 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mapped, %15 : tensor<1x1x128x128xf32>, tensor<1x1x128xf32>) outs(%extracted_slice_13 : tensor<1x1x128x128xf32>) {
        ^bb0(%in: f32, %in_22: f32, %out: f32):
          %24 = arith.subf %in, %in_22 : f32
          %25 = math.exp %24 : f32
          linalg.yield %25 : f32
        } -> tensor<1x1x128x128xf32>
        %17 = linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %arg8, %15 : tensor<1x1x128xf32>, tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%arg10 : tensor<1x1x128xf32>) {
        ^bb0(%in: f32, %in_22: f32, %in_23: f32, %out: f32):
          %24 = arith.subf %in_22, %in_23 : f32
          %25 = math.exp %24 : f32
          %26 = arith.mulf %in, %25 : f32
          linalg.yield %26 : f32
        } -> tensor<1x1x128xf32>
        %18 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%16 : tensor<1x1x128x128xf32>) outs(%17 : tensor<1x1x128xf32>) {
        ^bb0(%in: f32, %out: f32):
          %24 = arith.addf %in, %out : f32
          linalg.yield %24 : f32
        } -> tensor<1x1x128xf32>
        %extracted_slice_14 = tensor.extract_slice %arg11[0, 0, 0, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x4096xf32> to tensor<1x1x128x128xf32>
        %19 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mapped, %15 : tensor<1x1x128x128xf32>, tensor<1x1x128xf32>) outs(%extracted_slice_14 : tensor<1x1x128x128xf32>) {
        ^bb0(%in: f32, %in_22: f32, %out: f32):
          %24 = arith.subf %in, %in_22 : f32
          %25 = math.exp %24 : f32
          linalg.yield %25 : f32
        } -> tensor<1x1x128x128xf32>
        %extracted_slice_15 = tensor.extract_slice %arg12[0, 0, 0, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x4096xf32> to tensor<1x1x128x128xf32>
        %20 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19, %18 : tensor<1x1x128x128xf32>, tensor<1x1x128xf32>) outs(%extracted_slice_15 : tensor<1x1x128x128xf32>) {
        ^bb0(%in: f32, %in_22: f32, %out: f32):
          %24 = arith.divf %in, %in_22 : f32
          linalg.yield %24 : f32
        } -> tensor<1x1x128x128xf32>
        %extracted_slice_16 = tensor.extract_slice %arg13[0, 0, 0, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x4096xf16> to tensor<1x1x128x128xf16>
        %21 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20 : tensor<1x1x128x128xf32>) outs(%extracted_slice_16 : tensor<1x1x128x128xf16>) {
        ^bb0(%in: f32, %out: f16):
          %24 = arith.truncf %in : f32 to f16
          linalg.yield %24 : f16
        } -> tensor<1x1x128x128xf16>
        %inserted_slice = tensor.insert_slice %mapped into %arg7[0, 0, 0, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x128xf32> into tensor<1x1x128x4096xf32>
        %inserted_slice_17 = tensor.insert_slice %16 into %arg9[0, 0, 0, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x128xf32> into tensor<1x1x128x4096xf32>
        %inserted_slice_18 = tensor.insert_slice %19 into %arg11[0, 0, 0, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x128xf32> into tensor<1x1x128x4096xf32>
        %inserted_slice_19 = tensor.insert_slice %20 into %arg12[0, 0, 0, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x128xf32> into tensor<1x1x128x4096xf32>
        %extracted_slice_20 = tensor.extract_slice %extracted_slice_6[0, 0, %13, 0] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x4096x128xf16> to tensor<1x1x128x128xf16>
        %22 = linalg.generic {indexing_maps = [#map4, #map5, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg14, %17, %18 : tensor<1x1x128x128xf32>, tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%arg14 : tensor<1x1x128x128xf32>) {
        ^bb0(%in: f32, %in_22: f32, %in_23: f32, %out: f32):
          %24 = arith.mulf %in, %in_22 : f32
          %25 = arith.divf %cst, %in_23 : f32
          %26 = arith.mulf %24, %25 : f32
          linalg.yield %26 : f32
        } -> tensor<1x1x128x128xf32>
        %23 = linalg.generic {indexing_maps = [#map1, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%21, %extracted_slice_20 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) outs(%22 : tensor<1x1x128x128xf32>) {
        ^bb0(%in: f16, %in_22: f16, %out: f32):
          %24 = arith.extf %in : f16 to f32
          %25 = arith.extf %in_22 : f16 to f32
          %26 = arith.mulf %24, %25 : f32
          %27 = arith.addf %out, %26 : f32
          linalg.yield %27 : f32
        } -> tensor<1x1x128x128xf32>
        %inserted_slice_21 = tensor.insert_slice %21 into %arg13[0, 0, 0, %13] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x128xf16> into tensor<1x1x128x4096xf16>
        scf.yield %inserted_slice, %15, %inserted_slice_17, %18, %inserted_slice_18, %inserted_slice_19, %inserted_slice_21, %23 : tensor<1x1x128x4096xf32>, tensor<1x1x128xf32>, tensor<1x1x128x4096xf32>, tensor<1x1x128xf32>, tensor<1x1x128x4096xf32>, tensor<1x1x128x4096xf32>, tensor<1x1x128x4096xf16>, tensor<1x1x128x128xf32>
      }
      %extracted_slice_8 = tensor.extract_slice %arg5[0, %arg3, %10, 0] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x32x4096x128xf16> to tensor<1x1x128x128xf16>
      %12 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11#7 : tensor<1x1x128x128xf32>) outs(%extracted_slice_8 : tensor<1x1x128x128xf16>) {
      ^bb0(%in: f32, %out: f16):
        %13 = arith.truncf %in : f32 to f16
        linalg.yield %13 : f16
      } -> tensor<1x1x128x128xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %12 into %arg5[0, %arg3, %10, 0] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x128xf16> into tensor<1x32x4096x128xf16>
      }
    }
    return %9 : tensor<1x32x4096x128xf16>
  }
}
