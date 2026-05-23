!any = !transform.any_op

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> ()>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_2d_1d_reduction(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.structured %candidate : (!any) -> !any {
    ^bb0(%op: !any):
      transform.match.structured.dim %op[0, 1] {parallel} : !any
      transform.match.structured.dim %op[2] {reduction} : !any
      transform.match.structured.yield %op : !any
    }
    transform.yield %matched : !any
  }

  transform.named_sequence @return_matched(%arg: !any {transform.readonly}) -> !any {
    transform.yield %arg : !any
  }

  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.linalg.fold_zero_indexed_unit_dims %func : !any

    %transposes = transform.structured.match ops{["linalg.transpose"]} in %func : (!any) -> !any
    %transposes_lg = transform.structured.generalize %transposes : (!any) -> !any
    %bmms = transform.structured.match ops{["linalg.batch_matmul"]} in %func : (!any) -> !any
    %bmms_lg = transform.structured.generalize %bmms : (!any) -> !any

    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_expanding_reshape
      transform.apply_patterns.tensor.reassociative_reshape_folding
    } : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.canonicalization
    } : !any

    %bmm0, %_0 = transform.split_handle %bmms_lg : (!any) -> (!any, !any)
    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 128, 64, 0] : (!any) -> (!any, !any)

    %bscale = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    transform.fusion.into_producer %bscale into %forall_loop : (!any, !any) -> !any
    %bmask = transform.get_consumers_of_result %forall_loop[1] : (!any) -> !any
    transform.linalg.greedy_inline_elementwise %bmask : !any
    transform.fusion.into_producer %bmask into %forall_loop : (!any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %consumers = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %_2, %bmax = transform.foreach_match restrict_root in %consumers
        @match_2d_1d_reduction -> @return_matched : (!any) -> (!any, !any)
    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.scf.fuse_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    %bsum, %elemwise = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %fused_bsum = transform.fusion.repair_reduction_frontier
        (%fused_bmax, %bsum) and (%elemwise, %elemwise_sidecars) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    %bmm1, %elemwise_1 = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_3 = transform.fusion.repair_reduction_frontier
        (%fused_bsum, %bmm1) and (%elemwise_1, %elemwise_sidecars_1) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %trunc = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %fused_trunc = transform.fusion.into_producer %trunc into %forall_loop : (!any, !any) -> !any

    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.tensor.bubble_up_extract_slice
      transform.apply_patterns.canonicalization
    } : !any
    transform.apply_cse to %func : !any

    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf16>, %arg2: tensor<1x4x1024x64xf16>) -> tensor<1x4x1024x64xf16> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant dense<0xFF800000> : tensor<f32>
    %cst_2 = arith.constant 1.250000e-01 : f32
    %true = arith.constant true
    %0 = tensor.empty() : tensor<1x4x1024x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x4x1024x64xf16>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %35 = arith.extf %in : f16 to f32
      linalg.yield %35 : f32
    } -> tensor<1x4x1024x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x4x1024x64xf16>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %35 = arith.extf %in : f16 to f32
      linalg.yield %35 : f32
    } -> tensor<1x4x1024x64xf32>
    %3 = tensor.empty() : tensor<1x4x64x1024xf32>
    %transposed = linalg.transpose ins(%2 : tensor<1x4x1024x64xf32>) outs(%3 : tensor<1x4x64x1024xf32>) permutation = [0, 1, 3, 2]
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x4x1024x64xf32> into tensor<4x1024x64xf32>
    %collapsed_3 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x4x64x1024xf32> into tensor<4x64x1024xf32>
    %4 = tensor.empty() : tensor<4x1024x1024xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %6 = linalg.batch_matmul ins(%collapsed, %collapsed_3 : tensor<4x1024x64xf32>, tensor<4x64x1024xf32>) outs(%5 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %expanded = tensor.expand_shape %6 [[0, 1], [2], [3]] output_shape [1, 4, 1024, 1024] : tensor<4x1024x1024xf32> into tensor<1x4x1024x1024xf32>
    %7 = tensor.empty() : tensor<1x4x1024x1024xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x4x1024x1024xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.mulf %in, %cst_2 : f32
      linalg.yield %35 : f32
    } -> tensor<1x4x1024x1024xf32>
    %9 = tensor.empty() : tensor<1024x1024xi1>
    %10 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<1024x1024xi1>) outs(%9 : tensor<1024x1024xi1>) {
    ^bb0(%in: i1, %out: i1):
      linalg.yield %true : i1
    } -> tensor<1024x1024xi1>
    %11 = tensor.empty() : tensor<1024xi64>
    %12 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%11 : tensor<1024xi64>) {
    ^bb0(%out: i64):
      %35 = linalg.index 0 : index
      %36 = arith.index_cast %35 : index to i64
      linalg.yield %36 : i64
    } -> tensor<1024xi64>
    %expanded_4 = tensor.expand_shape %12 [[0, 1]] output_shape [1, 1024] : tensor<1024xi64> into tensor<1x1024xi64>
    %expanded_5 = tensor.expand_shape %12 [[0, 1]] output_shape [1024, 1] : tensor<1024xi64> into tensor<1024x1xi64>
    %13 = tensor.empty() : tensor<1024x1024xi64>
    %14 = linalg.generic {indexing_maps = [#map3, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_4, %expanded_5 : tensor<1x1024xi64>, tensor<1024x1xi64>) outs(%13 : tensor<1024x1024xi64>) {
    ^bb0(%in: i64, %in_11: i64, %out: i64):
      %35 = arith.subi %in, %in_11 : i64
      linalg.yield %35 : i64
    } -> tensor<1024x1024xi64>
    %15 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<1024x1024xi64>) outs(%9 : tensor<1024x1024xi1>) {
    ^bb0(%in: i64, %out: i1):
      %35 = arith.cmpi sle, %in, %c0_i64 : i64
      linalg.yield %35 : i1
    } -> tensor<1024x1024xi1>
    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%15, %10 : tensor<1024x1024xi1>, tensor<1024x1024xi1>) outs(%9 : tensor<1024x1024xi1>) {
    ^bb0(%in: i1, %in_11: i1, %out: i1):
      %35 = arith.andi %in, %in_11 : i1
      linalg.yield %35 : i1
    } -> tensor<1024x1024xi1>
    %expanded_6 = tensor.expand_shape %16 [[0, 1, 2], [3]] output_shape [1, 1, 1024, 1024] : tensor<1024x1024xi1> into tensor<1x1x1024x1024xi1>
    %17 = linalg.generic {indexing_maps = [#map5, #map, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6, %8, %cst_1 : tensor<1x1x1024x1024xi1>, tensor<1x4x1024x1024xf32>, tensor<f32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: i1, %in_11: f32, %in_12: f32, %out: f32):
      %35 = arith.select %in, %in_11, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<1x4x1024x1024xf32>
    %18 = tensor.empty() : tensor<1x4x1024xi64>
    %19 = linalg.fill ins(%c0_i64 : i64) outs(%18 : tensor<1x4x1024xi64>) -> tensor<1x4x1024xi64>
    %20 = tensor.empty() : tensor<1x4x1024xf32>
    %21 = linalg.fill ins(%cst_0 : f32) outs(%20 : tensor<1x4x1024xf32>) -> tensor<1x4x1024xf32>
    %22:2 = linalg.generic {indexing_maps = [#map, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%17 : tensor<1x4x1024x1024xf32>) outs(%21, %19 : tensor<1x4x1024xf32>, tensor<1x4x1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_11: i64):
      %35 = linalg.index 3 : index
      %36 = arith.index_cast %35 : index to i64
      %37 = arith.maximumf %in, %out : f32
      %38 = arith.cmpf ogt, %in, %out : f32
      %39 = arith.select %38, %36, %out_11 : i64
      linalg.yield %37, %39 : f32, i64
    } -> (tensor<1x4x1024xf32>, tensor<1x4x1024xi64>)
    %expanded_7 = tensor.expand_shape %22#0 [[0], [1], [2, 3]] output_shape [1, 4, 1024, 1] : tensor<1x4x1024xf32> into tensor<1x4x1024x1xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map8, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %expanded_7 : tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %35 = arith.subf %in, %in_11 : f32
      linalg.yield %35 : f32
    } -> tensor<1x4x1024x1024xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<1x4x1024x1024xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = math.exp %in : f32
      linalg.yield %35 : f32
    } -> tensor<1x4x1024x1024xf32>
    %25 = tensor.empty() : tensor<1x4x1024x1xf32>
    %26 = linalg.fill ins(%cst : f32) outs(%25 : tensor<1x4x1024x1xf32>) -> tensor<1x4x1024x1xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%24 : tensor<1x4x1024x1024xf32>) outs(%26 : tensor<1x4x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    } -> tensor<1x4x1024x1xf32>
    %28 = linalg.generic {indexing_maps = [#map, #map8, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %27 : tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %35 = arith.divf %in, %in_11 : f32
      linalg.yield %35 : f32
    } -> tensor<1x4x1024x1024xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x4x1024x64xf16>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %35 = arith.extf %in : f16 to f32
      linalg.yield %35 : f32
    } -> tensor<1x4x1024x64xf32>
    %collapsed_8 = tensor.collapse_shape %28 [[0, 1], [2], [3]] : tensor<1x4x1024x1024xf32> into tensor<4x1024x1024xf32>
    %collapsed_9 = tensor.collapse_shape %29 [[0, 1], [2], [3]] : tensor<1x4x1024x64xf32> into tensor<4x1024x64xf32>
    %30 = tensor.empty() : tensor<4x1024x64xf32>
    %31 = linalg.fill ins(%cst : f32) outs(%30 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
    %32 = linalg.batch_matmul ins(%collapsed_8, %collapsed_9 : tensor<4x1024x1024xf32>, tensor<4x1024x64xf32>) outs(%31 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
    %expanded_10 = tensor.expand_shape %32 [[0, 1], [2], [3]] output_shape [1, 4, 1024, 64] : tensor<4x1024x64xf32> into tensor<1x4x1024x64xf32>
    %33 = tensor.empty() : tensor<1x4x1024x64xf16>
    %34 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_10 : tensor<1x4x1024x64xf32>) outs(%33 : tensor<1x4x1024x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %35 = arith.truncf %in : f32 to f16
      linalg.yield %35 : f16
    } -> tensor<1x4x1024x64xf16>
    return %34 : tensor<1x4x1024x64xf16>
  }
}
