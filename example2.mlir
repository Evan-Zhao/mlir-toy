func.func @attention(%arg0: tensor<1x32x4096x128xf16>, %arg1: tensor<1x32x4096x128xf16>, %arg2: tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16> {
  %cst = arith.constant 0xFFC00000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 0.0883883461 : f32
  %0 = tensor.empty() : tensor<1x32x4096x4096xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
  %2 = tensor.empty() : tensor<1x32x4096x4096xf32>
  %3 = tensor.empty() : tensor<1x32x4096xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
  %5 = tensor.empty() : tensor<1x32x4096x4096xf32>
  %6:3 = scf.forall (%arg3, %arg4) in (32, 32) shared_outs(%arg5 = %0, %arg6 = %4, %arg7 = %5) -> (tensor<1x32x4096x4096xf32>, tensor<1x32x4096xf32>, tensor<1x32x4096x4096xf32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %16 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
    %17 = affine.apply affine_map<(d0) -> (d0 * 128)>(%c0)
    %extracted_slice = tensor.extract_slice %arg5[0, %arg3, %16, %17] [1, 1, 128, 4096] [1, 1, 1, 1] : tensor<1x32x4096x4096xf32> to tensor<1x1x128x4096xf32>
    %extracted_slice_2 = tensor.extract_slice %arg6[0, %arg3, %16] [1, 1, 128] [1, 1, 1] : tensor<1x32x4096xf32> to tensor<1x1x128xf32>
    %18 = tensor.empty() : tensor<1x1x128x4096xf32>
    %19:3 = scf.for %arg8 = %c0 to %c32 step %c1 iter_args(%arg9 = %extracted_slice, %arg10 = %extracted_slice_2, %arg11 = %18) -> (tensor<1x1x128x4096xf32>, tensor<1x1x128xf32>, tensor<1x1x128x4096xf32>) {
      %20 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %21 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg8)
      %extracted_slice_3 = tensor.extract_slice %arg0[0, %arg3, %20, 0] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x32x4096x128xf16> to tensor<1x1x128x128xf16>
      %extracted_slice_4 = tensor.extract_slice %arg1[0, %arg3, %21, 0] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x32x4096x128xf16> to tensor<1x1x128x128xf16>
      %extracted_slice_5 = tensor.extract_slice %1[0, %arg3, %20, %21] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x32x4096x4096xf32> to tensor<1x1x128x128xf32>
      %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice_3, %extracted_slice_4 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) outs(%extracted_slice_5 : tensor<1x1x128x128xf32>) {
      ^bb0(%in: f16, %in_11: f16, %out: f32):
        %30 = arith.extf %in : f16 to f32
        %31 = arith.extf %in_11 : f16 to f32
        %32 = arith.mulf %30, %31 : f32
        %33 = arith.addf %out, %32 : f32
        linalg.yield %33 : f32
      } -> tensor<1x1x128x128xf32>
      %23 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %24 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg8)
      %extracted_slice_6 = tensor.extract_slice %arg5[0, %arg3, %23, %24] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x32x4096x4096xf32> to tensor<1x1x128x128xf32>
      %mapped = linalg.map ins(%22 : tensor<1x1x128x128xf32>) outs(%extracted_slice_6 : tensor<1x1x128x128xf32>)
        (%in: f32, %init: f32) {
          %30 = arith.mulf %in, %cst_1 : f32
          linalg.yield %30 : f32
        }
      %inserted_slice = tensor.insert_slice %mapped into %arg9[0, 0, 0, %24] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x128xf32> into tensor<1x1x128x4096xf32>
      %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%mapped : tensor<1x1x128x128xf32>) outs(%arg10 : tensor<1x1x128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %30 = arith.maxnumf %in, %out : f32
        linalg.yield %30 : f32
      } -> tensor<1x1x128xf32>
      %26 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg8)
      %27 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg8)
      %extracted_slice_7 = tensor.extract_slice %arg9[0, 0, 0, %26] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x4096xf32> to tensor<1x1x128x128xf32>
      %extracted_slice_8 = tensor.extract_slice %arg10[0, 0, 0] [1, 1, 128] [1, 1, 1] : tensor<1x1x128xf32> to tensor<1x1x128xf32>
      %extracted_slice_9 = tensor.extract_slice %arg11[0, 0, 0, %27] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x4096xf32> to tensor<1x1x128x128xf32>
      %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mapped, %extracted_slice_8 : tensor<1x1x128x128xf32>, tensor<1x1x128xf32>) outs(%extracted_slice_9 : tensor<1x1x128x128xf32>) {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %30 = arith.subf %in, %in_11 : f32
        %31 = math.exp %30 : f32
        linalg.yield %31 : f32
      } -> tensor<1x1x128x128xf32>
      %29 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg8)
      %inserted_slice_10 = tensor.insert_slice %28 into %arg11[0, 0, 0, %29] [1, 1, 128, 128] [1, 1, 1, 1] : tensor<1x1x128x128xf32> into tensor<1x1x128x4096xf32>
      scf.yield %inserted_slice, %25, %inserted_slice_10 : tensor<1x1x128x4096xf32>, tensor<1x1x128xf32>, tensor<1x1x128x4096xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %19#0 into %arg5[0, %arg3, %16, %17] [1, 1, 128, 4096] [1, 1, 1, 1] : tensor<1x1x128x4096xf32> into tensor<1x32x4096x4096xf32>
      tensor.parallel_insert_slice %19#1 into %arg6[0, %arg3, %16] [1, 1, 128] [1, 1, 1] : tensor<1x1x128xf32> into tensor<1x32x4096xf32>
      tensor.parallel_insert_slice %19#2 into %arg7[0, %arg3, %16, %17] [1, 1, 128, 4096] [1, 1, 1, 1] : tensor<1x1x128x4096xf32> into tensor<1x32x4096x4096xf32>
    }
  }
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6#0, %6#1 : tensor<1x32x4096x4096xf32>, tensor<1x32x4096xf32>) outs(%2 : tensor<1x32x4096x4096xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %16 = arith.subf %in, %in_2 : f32
    %17 = math.exp %16 : f32
    linalg.yield %17 : f32
  } -> tensor<1x32x4096x4096xf32>
  %8 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%7 : tensor<1x32x4096x4096xf32>) outs(%8 : tensor<1x32x4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    %16 = arith.addf %in, %out : f32
    linalg.yield %16 : f32
  } -> tensor<1x32x4096xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %9 : tensor<1x32x4096x4096xf32>, tensor<1x32x4096xf32>) outs(%2 : tensor<1x32x4096x4096xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %16 = arith.divf %in, %in_2 : f32
    linalg.yield %16 : f32
  } -> tensor<1x32x4096x4096xf32>
  %11 = arith.truncf %10 : tensor<1x32x4096x4096xf32> to tensor<1x32x4096x4096xf16>
  %12 = tensor.empty() : tensor<1x32x4096x128xf32>
  %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<1x32x4096x128xf32>) -> tensor<1x32x4096x128xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%11, %arg2 : tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) outs(%13 : tensor<1x32x4096x128xf32>) {
  ^bb0(%in: f16, %in_2: f16, %out: f32):
    %16 = arith.extf %in : f16 to f32
    %17 = arith.extf %in_2 : f16 to f32
    %18 = arith.mulf %16, %17 : f32
    %19 = arith.addf %out, %18 : f32
    linalg.yield %19 : f32
  } -> tensor<1x32x4096x128xf32>
  %15 = arith.truncf %14 : tensor<1x32x4096x128xf32> to tensor<1x32x4096x128xf16>
  return %15 : tensor<1x32x4096x128xf16>
}
