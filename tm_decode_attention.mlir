!any = !transform.any_op
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_4d_matmul_transb(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b h i d, b h j d -> b h i j"} : (!any) -> !any
    transform.yield %matched : !any
  }

  transform.named_sequence @__transform_main(%module: !any) {
    %func0 = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %func1 = transform.apply_registered_pass "linalg-generalize-named-ops" to %func0 : (!any) -> !any
    %func = transform.apply_registered_pass "linalg-to-ta" to %func1 : (!any) -> !any
    transform.ta.rewrite_exp_to_exp2 %func : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !any
    %bmm0 = transform.collect_matching @match_4d_matmul_transb in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any

    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 1, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bmax, %_2 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %prefix = transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %bmax { inline_elementwise } : (!any, !any) -> !any
    transform.linalg.erase_unused_operands_and_results %prefix : !any

    // Use fuse_partial_reduction_into_forall here (similar to RFactor in TVM).
    // It splits bmax into a local reduction and a global one (bmax_wb),
    // and fuses the local one under the forall loop (becomes bmax_rf).
    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %bmax_rf, %bmax_wb = transform.scf.fuse_partial_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    // Find the next reduction forward from the loop, but only look for users of the 0th result of the loop.
    // %bmax_rf would be using the 1st result of the loop, and we don't want that one.
    %bmm1, %elemwise = transform.fusion.find_next_reduction %forall_loop[0] : (!any) -> (!any, !any)
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %sidecars = transform.fusion.clone_fuse_rfactor_elemwise
        %elemwise into %forall_loop substituting (%bmax_wb -> %bmax_rf)
        : (!any, !any, !any, !any) -> !any
    %fused_bmm1, %bmm1_writeback = transform.fusion.repair_rfactor_reduction_frontier
        %bmm1 substituting reduce %bmax_wb -> %bmax_rf elemwise %elemwise -> %sidecars
        into %forall_loop : (!any, !any, !any, !any, !any, !any) -> (!any, !any)

    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x1x64xf16>, %arg1: tensor<1x4x1024x64xf16>, %arg2: tensor<1x4x1024x64xf16>) -> tensor<1x4x1x64xf16> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<1x4x1x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x4x1x64xf16>) outs(%0 : tensor<1x4x1x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %27 = arith.extf %in : f16 to f32
      linalg.yield %27 : f32
    } -> tensor<1x4x1x64xf32>
    %2 = tensor.empty() : tensor<1x4x1024x64xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x4x1024x64xf16>) outs(%2 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %27 = arith.extf %in : f16 to f32
      linalg.yield %27 : f32
    } -> tensor<1x4x1024x64xf32>
    %4 = tensor.empty() : tensor<1x4x64x1024xf32>
    %transposed = linalg.transpose ins(%3 : tensor<1x4x1024x64xf32>) outs(%4 : tensor<1x4x64x1024xf32>) permutation = [0, 1, 3, 2]
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x4x1x64xf32> into tensor<4x1x64xf32>
    %collapsed_2 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x4x64x1024xf32> into tensor<4x64x1024xf32>
    %5 = tensor.empty() : tensor<4x1x1024xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x1x1024xf32>) -> tensor<4x1x1024xf32>
    %7 = linalg.batch_matmul ins(%collapsed, %collapsed_2 : tensor<4x1x64xf32>, tensor<4x64x1024xf32>) outs(%6 : tensor<4x1x1024xf32>) -> tensor<4x1x1024xf32>
    %expanded = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 4, 1, 1024] : tensor<4x1x1024xf32> into tensor<1x4x1x1024xf32>
    %8 = tensor.empty() : tensor<1x4x1x1024xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x4x1x1024xf32>) outs(%8 : tensor<1x4x1x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %27 = arith.mulf %in, %cst_1 : f32
      linalg.yield %27 : f32
    } -> tensor<1x4x1x1024xf32>
    %10 = tensor.empty() : tensor<1x4x1xi64>
    %11 = linalg.fill ins(%c0_i64 : i64) outs(%10 : tensor<1x4x1xi64>) -> tensor<1x4x1xi64>
    %12 = tensor.empty() : tensor<1x4x1xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<1x4x1xf32>) -> tensor<1x4x1xf32>
    %14:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9 : tensor<1x4x1x1024xf32>) outs(%13, %11 : tensor<1x4x1xf32>, tensor<1x4x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_7: i64):
      %27 = linalg.index 3 : index
      %28 = arith.index_cast %27 : index to i64
      %29 = arith.maximumf %in, %out : f32
      %30 = arith.cmpf ogt, %in, %out : f32
      %31 = arith.select %30, %28, %out_7 : i64
      linalg.yield %29, %31 : f32, i64
    } -> (tensor<1x4x1xf32>, tensor<1x4x1xi64>)
    %expanded_3 = tensor.expand_shape %14#0 [[0], [1], [2, 3]] output_shape [1, 4, 1, 1] : tensor<1x4x1xf32> into tensor<1x4x1x1xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %expanded_3 : tensor<1x4x1x1024xf32>, tensor<1x4x1x1xf32>) outs(%8 : tensor<1x4x1x1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %27 = arith.subf %in, %in_7 : f32
      linalg.yield %27 : f32
    } -> tensor<1x4x1x1024xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<1x4x1x1024xf32>) outs(%8 : tensor<1x4x1x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %27 = math.exp %in : f32
      linalg.yield %27 : f32
    } -> tensor<1x4x1x1024xf32>
    %17 = tensor.empty() : tensor<1x4x1x1xf32>
    %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<1x4x1x1xf32>) -> tensor<1x4x1x1xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%16 : tensor<1x4x1x1024xf32>) outs(%18 : tensor<1x4x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %27 = arith.addf %in, %out : f32
      linalg.yield %27 : f32
    } -> tensor<1x4x1x1xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %19 : tensor<1x4x1x1024xf32>, tensor<1x4x1x1xf32>) outs(%8 : tensor<1x4x1x1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %27 = arith.divf %in, %in_7 : f32
      linalg.yield %27 : f32
    } -> tensor<1x4x1x1024xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x4x1024x64xf16>) outs(%2 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %27 = arith.extf %in : f16 to f32
      linalg.yield %27 : f32
    } -> tensor<1x4x1024x64xf32>
    %collapsed_4 = tensor.collapse_shape %20 [[0, 1], [2], [3]] : tensor<1x4x1x1024xf32> into tensor<4x1x1024xf32>
    %collapsed_5 = tensor.collapse_shape %21 [[0, 1], [2], [3]] : tensor<1x4x1024x64xf32> into tensor<4x1024x64xf32>
    %22 = tensor.empty() : tensor<4x1x64xf32>
    %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<4x1x64xf32>) -> tensor<4x1x64xf32>
    %24 = linalg.batch_matmul ins(%collapsed_4, %collapsed_5 : tensor<4x1x1024xf32>, tensor<4x1024x64xf32>) outs(%23 : tensor<4x1x64xf32>) -> tensor<4x1x64xf32>
    %expanded_6 = tensor.expand_shape %24 [[0, 1], [2], [3]] output_shape [1, 4, 1, 64] : tensor<4x1x64xf32> into tensor<1x4x1x64xf32>
    %25 = tensor.empty() : tensor<1x4x1x64xf16>
    %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<1x4x1x64xf32>) outs(%25 : tensor<1x4x1x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %27 = arith.truncf %in : f32 to f16
      linalg.yield %27 : f16
    } -> tensor<1x4x1x64xf16>
    return %26 : tensor<1x4x1x64xf16>
  }
}
