// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s --check-prefix=MATCH
//
// Transform-dialect schedule that transforms the Torch-MLIR GQA payload below
// into a FlashAttention-like fused program with a `(group, head)` outer loop shape.

!any = !transform.any_op
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, 0)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_4d_1d_reduction(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.structured %candidate : (!any) -> !any {
    ^bb0(%op: !any):
      transform.match.structured.dim %op[0, 1, 2, 3] {parallel} : !any
      transform.match.structured.dim %op[4] {reduction} : !any
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

    %bmm0, %_0 = transform.split_handle %bmms_lg : (!any) -> (!any, !any)
    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    transform.linalg.erase_unused_operands_and_results %bmm0 : !any
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 1, 128, 64, 0] : (!any) -> (!any, !any)

    %bscale = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %fused_bscale = transform.fusion.into_producer %bscale into %forall_loop : (!any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %consumers = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %_2, %bmax = transform.foreach_match restrict_root in %consumers
        @match_4d_1d_reduction -> @return_matched : (!any) -> (!any, !any)
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
    transform.linalg.erase_unused_operands_and_results %bmm1 : !any
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_3 = transform.fusion.repair_reduction_frontier
        (%fused_bsum, %bmm1) and (%elemwise_1, %elemwise_sidecars_1) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    %trunc = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %fused_trunc = transform.fusion.into_producer %trunc into %forall_loop : (!any, !any) -> !any

    transform.apply_patterns to %func {
      transform.apply_patterns.tensor.bubble_up_extract_slice
      transform.apply_patterns.canonicalization
    } : !any
    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.apply_cse to %func : !any

    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x128x64xf16>, %arg1: tensor<1x2x128x64xf16>,
      %arg2: tensor<1x2x128x64xf16>) -> tensor<1x2x2x128x64xf16> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4]] output_shape [1, 2, 2, 128, 64] : tensor<1x4x128x64xf16> into tensor<1x2x2x128x64xf16>
    %0 = tensor.empty() : tensor<1x2x2x128x64xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x2x128x64xf16>) outs(%0 : tensor<1x2x2x128x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x2x2x128x64xf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x2x128x64xf16>) outs(%0 : tensor<1x2x2x128x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x2x2x128x64xf16>
    %3 = tensor.empty() : tensor<1x2x2x128x64xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x2x2x128x64xf16>) outs(%3 : tensor<1x2x2x128x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %28 = arith.extf %in : f16 to f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x128x64xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x2x2x128x64xf16>) outs(%3 : tensor<1x2x2x128x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %28 = arith.extf %in : f16 to f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x128x64xf32>
    %6 = tensor.empty() : tensor<1x2x2x64x128xf32>
    %transposed = linalg.transpose ins(%5 : tensor<1x2x2x128x64xf32>) outs(%6 : tensor<1x2x2x64x128xf32>) permutation = [0, 1, 2, 4, 3]
    %collapsed = tensor.collapse_shape %4 [[0, 1, 2], [3], [4]] : tensor<1x2x2x128x64xf32> into tensor<4x128x64xf32>
    %collapsed_2 = tensor.collapse_shape %transposed [[0, 1, 2], [3], [4]] : tensor<1x2x2x64x128xf32> into tensor<4x64x128xf32>
    %7 = tensor.empty() : tensor<4x128x128xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %9 = linalg.batch_matmul ins(%collapsed, %collapsed_2 : tensor<4x128x64xf32>, tensor<4x64x128xf32>) outs(%8 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %expanded_3 = tensor.expand_shape %9 [[0, 1, 2], [3], [4]] output_shape [1, 2, 2, 128, 128] : tensor<4x128x128xf32> into tensor<1x2x2x128x128xf32>
    %10 = tensor.empty() : tensor<1x2x2x128x128xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<1x2x2x128x128xf32>) outs(%10 : tensor<1x2x2x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %28 = arith.mulf %in, %cst_1 : f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x128x128xf32>
    %12 = tensor.empty() : tensor<1x2x2x128xi64>
    %13 = linalg.fill ins(%c0_i64 : i64) outs(%12 : tensor<1x2x2x128xi64>) -> tensor<1x2x2x128xi64>
    %14 = tensor.empty() : tensor<1x2x2x128xf32>
    %15 = linalg.fill ins(%cst_0 : f32) outs(%14 : tensor<1x2x2x128xf32>) -> tensor<1x2x2x128xf32>
    %16:2 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%11 : tensor<1x2x2x128x128xf32>) outs(%15, %13 : tensor<1x2x2x128xf32>, tensor<1x2x2x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_8: i64):
      %28 = linalg.index 4 : index
      %29 = arith.index_cast %28 : index to i64
      %30 = arith.maximumf %in, %out : f32
      %31 = arith.cmpf ogt, %in, %out : f32
      %32 = arith.select %31, %29, %out_8 : i64
      linalg.yield %30, %32 : f32, i64
    } -> (tensor<1x2x2x128xf32>, tensor<1x2x2x128xi64>)
    %expanded_4 = tensor.expand_shape %16#0 [[0], [1], [2], [3, 4]] output_shape [1, 2, 2, 128, 1] : tensor<1x2x2x128xf32> into tensor<1x2x2x128x1xf32>
    %17 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%11, %expanded_4 : tensor<1x2x2x128x128xf32>, tensor<1x2x2x128x1xf32>) outs(%10 : tensor<1x2x2x128x128xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %28 = arith.subf %in, %in_8 : f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x128x128xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x2x2x128x128xf32>) outs(%10 : tensor<1x2x2x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %28 = math.exp %in : f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x128x128xf32>
    %19 = tensor.empty() : tensor<1x2x2x128x1xf32>
    %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x2x2x128x1xf32>) -> tensor<1x2x2x128x1xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%18 : tensor<1x2x2x128x128xf32>) outs(%20 : tensor<1x2x2x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %28 = arith.addf %in, %out : f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x128x1xf32>
    %22 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%18, %21 : tensor<1x2x2x128x128xf32>, tensor<1x2x2x128x1xf32>) outs(%10 : tensor<1x2x2x128x128xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %28 = arith.divf %in, %in_8 : f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x128x128xf32>
    %23 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x2x2x128x64xf16>) outs(%3 : tensor<1x2x2x128x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %28 = arith.extf %in : f16 to f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x128x64xf32>
    %collapsed_5 = tensor.collapse_shape %22 [[0, 1, 2], [3], [4]] : tensor<1x2x2x128x128xf32> into tensor<4x128x128xf32>
    %collapsed_6 = tensor.collapse_shape %23 [[0, 1, 2], [3], [4]] : tensor<1x2x2x128x64xf32> into tensor<4x128x64xf32>
    %24 = tensor.empty() : tensor<4x128x64xf32>
    %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %26 = linalg.batch_matmul ins(%collapsed_5, %collapsed_6 : tensor<4x128x128xf32>, tensor<4x128x64xf32>) outs(%25 : tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %expanded_7 = tensor.expand_shape %26 [[0, 1, 2], [3], [4]] output_shape [1, 2, 2, 128, 64] : tensor<4x128x64xf32> into tensor<1x2x2x128x64xf32>
    %27 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7 : tensor<1x2x2x128x64xf32>) outs(%0 : tensor<1x2x2x128x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %28 = arith.truncf %in : f32 to f16
      linalg.yield %28 : f16
    } -> tensor<1x2x2x128x64xf16>
    return %27 : tensor<1x2x2x128x64xf16>
  }
}

// MATCH-LABEL: func.func @attention(
// MATCH-NOT: linalg.batch_matmul
// MATCH-NOT: linalg.transpose
// MATCH-NOT: tensor.collapse_shape
// MATCH-NOT: tensor.expand_shape %arg0
// MATCH: %[[OUT:.+]] = scf.forall (%{{.*}}, %{{.*}}) in (2, 2) shared_outs(%{{.*}} = %{{.*}}) -> (tensor<1x2x2x128x64xf32>)
// MATCH: %{{.*}}:3 = scf.for %{{.*}} = %c0 to %c2 step %c1 iter_args(
// MATCH: affine.linearize_index disjoint [%{{.*}}, %{{.*}}] by (2, 2) : index
// MATCH: tensor.extract_slice %arg0[0, %{{.*}}, 0, 0] [1, 1, 128, 64] [1, 1, 1, 1]
// MATCH: tensor.expand_shape %{{.*}} output_shape [1, 1, 1, 128, 64]
// MATCH: tensor.empty() : tensor<1x1x1x128x64xf32>
// MATCH: linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]}
// MATCH: math.exp
// MATCH: arith.divf %cst, %{{.*}} : f32
// MATCH: linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%{{.*}} : tensor<1x2x2x128x64xf32>) outs(%{{.*}} : tensor<1x2x2x128x64xf16>)
