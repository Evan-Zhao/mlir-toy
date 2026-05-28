// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter | FileCheck %s --check-prefix=MATCH
//
// Transform-dialect schedule for a Torch-MLIR causal-attention payload. This
// extends the basic FlashAttention-style schedule with
// `transform.loop.specialize_dead_tile`, which splits the streaming K/V loop
// into a fully-live prefix and a mixed suffix.

!any = !transform.any_op

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> ()>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>

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

    %bmm0, %bmm1 = transform.split_handle %bmms_lg : (!any) -> (!any, !any)
    %bmm1_1, %_0 = transform.linalg.exchange_div_and_matmul %bmm1 : (!any) -> (!any, !any)
    transform.linalg.greedy_inline_elementwise %bmm1_1 { operand_number = 1 } : !any

    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 128, 64, 0] : (!any) -> (!any, !any)

    %bscale = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    transform.fusion.into_producer %bscale into %forall_loop : (!any, !any) -> !any
    %bmask = transform.get_consumers_of_result %forall_loop[1] : (!any) -> !any
    transform.linalg.greedy_inline_elementwise %bmask : !any
    %fused_bmask = transform.fusion.into_producer %bmask into %forall_loop : (!any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %consumers = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %_2, %bmax = transform.foreach_match restrict_root in %consumers
        @match_2d_1d_reduction -> @return_matched : (!any) -> (!any, !any)
    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.scf.fuse_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    %bmm1_2, %elemwise = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_3 = transform.fusion.repair_reduction_frontier
        (%fused_bmax, %bmm1_2) and (%elemwise, %elemwise_sidecars) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    %bsum, %elemwise_1 = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_4 = transform.fusion.repair_reduction_frontier
        (%fused_bmax, %bsum) and (%elemwise_1, %elemwise_sidecars_1) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    // CSE removes duplicate affine values and helps fusion
    // (fusion compares offset equal by comparing pointers to SSA value).
    transform.apply_cse to %func : !any
    %div = transform.get_consumers_of_result %forall_loop[1] : (!any) -> !any
    transform.fusion.into_producer %div into %forall_loop : (!any, !any) -> !any
    %trunc = transform.get_consumers_of_result %forall_loop[2] : (!any) -> !any
    transform.fusion.into_producer %trunc into %forall_loop : (!any, !any) -> !any

    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.apply_cse to %func : !any
    transform.scf.localize_scratch_tensors %func : !any

    // 0xFF800000: -inf in f32
    %live_loop, %mixed_loop = transform.loop.specialize_dead_tile %fused_bmask in %j0_loop
        {dead_value = 0xFF800000 : f32} : !any, !any -> !any, !any

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
    %false = arith.constant false
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant dense<true> : tensor<1024x1024xi1>
    %cst_2 = arith.constant dense<0xFF800000> : tensor<f32>
    %cst_3 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<1x4x1024x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x4x1024x64xf16>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x1024x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x4x1024x64xf16>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x1024x64xf32>
    %3 = tensor.empty() : tensor<1x4x64x1024xf32>
    %transposed = linalg.transpose ins(%2 : tensor<1x4x1024x64xf32>) outs(%3 : tensor<1x4x64x1024xf32>) permutation = [0, 1, 3, 2]
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x4x1024x64xf32> into tensor<4x1024x64xf32>
    %collapsed_4 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x4x64x1024xf32> into tensor<4x64x1024xf32>
    %4 = tensor.empty() : tensor<4x1024x1024xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %6 = linalg.batch_matmul ins(%collapsed, %collapsed_4 : tensor<4x1024x64xf32>, tensor<4x64x1024xf32>) outs(%5 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %expanded = tensor.expand_shape %6 [[0, 1], [2], [3]] output_shape [1, 4, 1024, 1024] : tensor<4x1024x1024xf32> into tensor<1x4x1024x1024xf32>
    %7 = tensor.empty() : tensor<1x4x1024x1024xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x4x1024x1024xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.mulf %in, %cst_3 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x1024x1024xf32>
    %9 = tensor.empty() : tensor<1024x1024xi1>
    %10 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<1024x1024xi1>) outs(%9 : tensor<1024x1024xi1>) {
    ^bb0(%in: i1, %out: i1):
      %29 = linalg.index 0 : index
      %30 = arith.index_cast %29 : index to i64
      %31 = linalg.index 1 : index
      %32 = arith.index_cast %31 : index to i64
      %33 = arith.cmpi sle, %32, %30 : i64
      %34 = arith.select %33, %in, %false : i1
      linalg.yield %34 : i1
    } -> tensor<1024x1024xi1>
    %expanded_5 = tensor.expand_shape %10 [[0, 1, 2], [3]] output_shape [1, 1, 1024, 1024] : tensor<1024x1024xi1> into tensor<1x1x1024x1024xi1>
    %11 = linalg.generic {indexing_maps = [#map2, #map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_5, %8, %cst_2 : tensor<1x1x1024x1024xi1>, tensor<1x4x1024x1024xf32>, tensor<f32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: i1, %in_10: f32, %in_11: f32, %out: f32):
      %29 = arith.select %in, %in_10, %in_11 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x1024x1024xf32>
    %12 = tensor.empty() : tensor<1x4x1024xi64>
    %13 = linalg.fill ins(%c0_i64 : i64) outs(%12 : tensor<1x4x1024xi64>) -> tensor<1x4x1024xi64>
    %14 = tensor.empty() : tensor<1x4x1024xf32>
    %15 = linalg.fill ins(%cst_0 : f32) outs(%14 : tensor<1x4x1024xf32>) -> tensor<1x4x1024xf32>
    %16:2 = linalg.generic {indexing_maps = [#map, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%11 : tensor<1x4x1024x1024xf32>) outs(%15, %13 : tensor<1x4x1024xf32>, tensor<1x4x1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_10: i64):
      %29 = linalg.index 3 : index
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.maximumf %in, %out : f32
      %32 = arith.cmpf ogt, %in, %out : f32
      %33 = arith.select %32, %30, %out_10 : i64
      linalg.yield %31, %33 : f32, i64
    } -> (tensor<1x4x1024xf32>, tensor<1x4x1024xi64>)
    %expanded_6 = tensor.expand_shape %16#0 [[0], [1], [2, 3]] output_shape [1, 4, 1024, 1] : tensor<1x4x1024xf32> into tensor<1x4x1024x1xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11, %expanded_6 : tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %29 = arith.subf %in, %in_10 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x1024x1024xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x4x1024x1024xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.exp %in : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x1024x1024xf32>
    %19 = tensor.empty() : tensor<1x4x1024x1xf32>
    %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x4x1024x1xf32>) -> tensor<1x4x1024x1xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%18 : tensor<1x4x1024x1024xf32>) outs(%20 : tensor<1x4x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.addf %in, %out : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x1024x1xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18, %21 : tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %29 = arith.divf %in, %in_10 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x1024x1024xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x4x1024x64xf16>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x1024x64xf32>
    %collapsed_7 = tensor.collapse_shape %22 [[0, 1], [2], [3]] : tensor<1x4x1024x1024xf32> into tensor<4x1024x1024xf32>
    %collapsed_8 = tensor.collapse_shape %23 [[0, 1], [2], [3]] : tensor<1x4x1024x64xf32> into tensor<4x1024x64xf32>
    %24 = tensor.empty() : tensor<4x1024x64xf32>
    %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
    %26 = linalg.batch_matmul ins(%collapsed_7, %collapsed_8 : tensor<4x1024x1024xf32>, tensor<4x1024x64xf32>) outs(%25 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
    %expanded_9 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 4, 1024, 64] : tensor<4x1024x64xf32> into tensor<1x4x1024x64xf32>
    %27 = tensor.empty() : tensor<1x4x1024x64xf16>
    %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_9 : tensor<1x4x1024x64xf32>) outs(%27 : tensor<1x4x1024x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %29 = arith.truncf %in : f32 to f16
      linalg.yield %29 : f16
    } -> tensor<1x4x1024x64xf16>
    return %28 : tensor<1x4x1024x64xf16>
  }
}

// MATCH-LABEL: func.func @attention(
// MATCH-NOT: tensor.empty() : tensor<4x1024x64xf32>
// MATCH: %[[FORALL:[0-9]+]] = scf.forall (%{{.*}}, %{{.*}}) in (4, 8) shared_outs(%{{.*}} = %{{.*}}) -> (tensor<4x1024x64xf16>)
// MATCH: tensor.empty() : tensor<1x128xf32>
// MATCH: tensor.empty() : tensor<1x128x64xf32>
// MATCH: linalg.fill ins(%cst_0 : f32) outs(%{{.*}} : tensor<1x128x64xf32>)
// MATCH: %[[LIVE_BOUND:.+]] = arith.select %{{.*}}, %{{.*}}, %c16 : index
// MATCH: %[[DEAD_BOUND:.+]] = arith.select %{{.*}}, %{{.*}}, %c16 : index
// MATCH: %[[LIVE:.+]]:3 = scf.for %{{.*}} = %c0 to %[[LIVE_BOUND]] step %c1 iter_args(
// MATCH: %[[MIXED:.+]]:3 = scf.for %{{.*}} = %[[LIVE_BOUND]] to %[[DEAD_BOUND]] step %c1 iter_args(%{{.*}} = %[[LIVE]]#0, %{{.*}} = %[[LIVE]]#1, %{{.*}} = %[[LIVE]]#2) -> (tensor<1x128xf32>, tensor<1x128x64xf32>, tensor<1x128xf32>)
// MATCH: linalg.generic {indexing_maps = [#map{{[0-9]+}}, #map{{[0-9]+}}, #map{{[0-9]+}}, #map{{[0-9]+}}], iterator_types = ["parallel", "parallel", "parallel"]}
// MATCH: math.exp
// MATCH: linalg.generic {indexing_maps = [#map{{[0-9]+}}, #map{{[0-9]+}}, #map{{[0-9]+}}], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[MIXED]]#1, %[[MIXED]]#2 : tensor<1x128x64xf32>, tensor<1x128xf32>)
// MATCH: arith.divf %{{.*}}, %{{.*}} : f32
// MATCH: linalg.generic {indexing_maps = [#map{{[0-9]+}}, #map{{[0-9]+}}], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}} : tensor<1x128x64xf32>) outs(%{{.*}} : tensor<1x128x64xf16>)
// MATCH: tensor.parallel_insert_slice %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf16> into tensor<4x1024x64xf16>
// MATCH: tensor.expand_shape %[[FORALL]] {{\[}}[0, 1], [2], [3]] output_shape [1, 4, 1024, 64]
