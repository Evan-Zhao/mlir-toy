// RUN: neptune-opt --transform-interpreter %s 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf8E4M3FN>, %arg2: tensor<1x4x1024x64xf8E4M3FN>, %arg3: tensor<1x4x1x1xf32>, %arg4: tensor<1x4x1x1xf32>) -> tensor<1x4x1024x64xf16>
// CHECK: htile.launch_func @attention_kernel
// CHECK-SAME: {program_bounds = array<i64: 4, 8>}
// CHECK-NOT: scf.forall
// CHECK: return

// CHECK-LABEL: htile.kernel @attention_kernel
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<1x4x1024x64xf8E4M3FN>
// CHECK-SAME: memref<1x4x1x1xf32>
// CHECK-SAME: memref<1x4x1024x64xf8E4M3FN>
// CHECK-SAME: memref<1x4x1x1xf32>
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: attributes {program_bounds = array<i64: 4, 8>}
// CHECK: htile.program_id 0
// CHECK: htile.program_id 1
// CHECK-NOT: scf.forall
// CHECK: htile.full %{{.*}} : f32 -> tensor<128xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<128x64xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<128xf32>
// CHECK: %[[LIVE_BOUND_RAW:.+]] = affine.apply
// CHECK: %[[LIVE_BOUND_MAX:.+]] = arith.maxsi %[[LIVE_BOUND_RAW]], %c0 : index
// CHECK: %[[LIVE_BOUND:.+]] = arith.minsi %[[LIVE_BOUND_MAX]], %c16 : index
// CHECK: %[[LIVE_LOOP:.+]]:3 = scf.for %{{.*}} = %c0 to %[[LIVE_BOUND]] step %c1 iter_args(
// CHECK: scf.yield
// CHECK: %[[DEAD_BOUND_RAW:.+]] = affine.apply
// CHECK: %[[DEAD_BOUND_MAX:.+]] = arith.maxsi %[[DEAD_BOUND_RAW]], %c0 : index
// CHECK: %[[DEAD_BOUND:.+]] = arith.minsi %[[DEAD_BOUND_MAX]], %c16 : index
// CHECK: %{{.*}}:3 = scf.for %{{.*}} = %[[LIVE_BOUND]] to %[[DEAD_BOUND]] step %c1 iter_args(%{{.*}} = %[[LIVE_LOOP]]#0, %{{.*}} = %[[LIVE_LOOP]]#1, %{{.*}} = %[[LIVE_LOOP]]#2)
// CHECK: htile.store
// CHECK: htile.return

!any = !transform.any_op

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> ()>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>

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
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.exp_to_exp2
      transform.apply_patterns.ta.sink_div_after_matmul
      transform.apply_patterns.ta.sink_right_mul_after_matmul
      transform.apply_patterns.ta.reassociate_right_mulf
    } : !any
    transform.apply_cse to %func : !any
    %bmm0 = transform.collect_matching @match_4d_matmul_transb in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any

    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 128, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bmax, %_2 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %prefix = transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %bmax { inline_elementwise } : (!any, !any) -> !any
    transform.linalg.erase_unused_operands_and_results %prefix : !any

    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.scf.fuse_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    %bmm1, %elemwise = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    %_3 = transform.fusion.repair_reduction_frontier
        %bmm1 reduce_producer %fused_bmax
        substituting elemwise %elemwise -> %elemwise_sidecars
        into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    %bsum, %elemwise_1 = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_4 = transform.fusion.repair_reduction_frontier
        %bsum reduce_producer %fused_bmax
        substituting elemwise %elemwise_1 -> %elemwise_sidecars_1
        into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %ret = transform.structured.match ops{["func.return"]} in %func : (!any) -> !any
    transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %ret : (!any, !any) -> !any

    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !any
    transform.apply_cse to %func : !any

    // 0xFF800000: -inf in f32
    %live_loop, %mixed_loop = transform.loop.specialize_dead_tile in %j0_loop
        {dead_value = 0xFF800000 : f32} : (!any) -> (!any, !any)

    // --- HTile lowering begins ---
    transform.htile.linalg_to_semantic %func : !any
    %launches, %kernels = transform.htile.outline_kernels %forall_loop
        {kernel_names = ["attention_kernel"]} : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.verify %func : !any
    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf8E4M3FN>, %arg2: tensor<1x4x1024x64xf8E4M3FN>, %arg3: tensor<1x4x1x1xf32>, %arg4: tensor<1x4x1x1xf32>) -> tensor<1x4x1024x64xf16> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %false = arith.constant false
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant dense<true> : tensor<1024x1024xi1>
    %cst_2 = arith.constant dense<0xFF800000> : tensor<f32>
    %cst_3 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<1x4x1024x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x4x1024x64xf8E4M3FN>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f8E4M3FN, %out: f32):
      %31 = arith.extf %in : f8E4M3FN to f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg3 : tensor<1x4x1024x64xf32>, tensor<1x4x1x1xf32>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %31 = arith.mulf %in, %in_9 : f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x64xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x4x1024x64xf8E4M3FN>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f8E4M3FN, %out: f32):
      %31 = arith.extf %in : f8E4M3FN to f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x64xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg4 : tensor<1x4x1024x64xf32>, tensor<1x4x1x1xf32>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %31 = arith.mulf %in, %in_9 : f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x64xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x4x1024x64xf16>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %31 = arith.extf %in : f16 to f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x64xf32>
    %6 = tensor.empty() : tensor<1x4x64x1024xf32>
    %transposed = linalg.transpose ins(%2 : tensor<1x4x1024x64xf32>) outs(%6 : tensor<1x4x64x1024xf32>) permutation = [0, 1, 3, 2]
    %collapsed = tensor.collapse_shape %5 [[0, 1], [2], [3]] : tensor<1x4x1024x64xf32> into tensor<4x1024x64xf32>
    %collapsed_4 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x4x64x1024xf32> into tensor<4x64x1024xf32>
    %7 = tensor.empty() : tensor<4x1024x1024xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %9 = linalg.batch_matmul ins(%collapsed, %collapsed_4 : tensor<4x1024x64xf32>, tensor<4x64x1024xf32>) outs(%8 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %expanded = tensor.expand_shape %9 [[0, 1], [2], [3]] output_shape [1, 4, 1024, 1024] : tensor<4x1024x1024xf32> into tensor<1x4x1024x1024xf32>
    %10 = tensor.empty() : tensor<1x4x1024x1024xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x4x1024x1024xf32>) outs(%10 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %31 = arith.mulf %in, %cst_3 : f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x1024xf32>
    %12 = tensor.empty() : tensor<1024x1024xi1>
    %13 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<1024x1024xi1>) outs(%12 : tensor<1024x1024xi1>) {
    ^bb0(%in: i1, %out: i1):
      %31 = linalg.index 0 : index
      %32 = arith.index_cast %31 : index to i64
      %33 = linalg.index 1 : index
      %34 = arith.index_cast %33 : index to i64
      %35 = arith.cmpi sle, %34, %32 : i64
      %36 = arith.select %35, %in, %false : i1
      linalg.yield %36 : i1
    } -> tensor<1024x1024xi1>
    %14 = linalg.generic {indexing_maps = [#map3, #map, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %11, %cst_2 : tensor<1024x1024xi1>, tensor<1x4x1024x1024xf32>, tensor<f32>) outs(%10 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: i1, %in_9: f32, %in_10: f32, %out: f32):
      %31 = arith.select %in, %in_9, %in_10 : f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x1024xf32>
    %15 = tensor.empty() : tensor<1x4x1024xi64>
    %16 = linalg.fill ins(%c0_i64 : i64) outs(%15 : tensor<1x4x1024xi64>) -> tensor<1x4x1024xi64>
    %17 = tensor.empty() : tensor<1x4x1024xf32>
    %18 = linalg.fill ins(%cst_0 : f32) outs(%17 : tensor<1x4x1024xf32>) -> tensor<1x4x1024xf32>
    %19:2 = linalg.generic {indexing_maps = [#map, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%14 : tensor<1x4x1024x1024xf32>) outs(%18, %16 : tensor<1x4x1024xf32>, tensor<1x4x1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_9: i64):
      %31 = linalg.index 3 : index
      %32 = arith.index_cast %31 : index to i64
      %33 = arith.maximumf %in, %out : f32
      %34 = arith.cmpf ogt, %in, %out : f32
      %35 = arith.select %34, %32, %out_9 : i64
      linalg.yield %33, %35 : f32, i64
    } -> (tensor<1x4x1024xf32>, tensor<1x4x1024xi64>)
    %expanded_5 = tensor.expand_shape %19#0 [[0], [1], [2, 3]] output_shape [1, 4, 1024, 1] : tensor<1x4x1024xf32> into tensor<1x4x1024x1xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %expanded_5 : tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1xf32>) outs(%10 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %31 = arith.subf %in, %in_9 : f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x1024xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20 : tensor<1x4x1024x1024xf32>) outs(%10 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %31 = math.exp %in : f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x1024xf32>
    %22 = tensor.empty() : tensor<1x4x1024x1xf32>
    %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<1x4x1024x1xf32>) -> tensor<1x4x1024x1xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%21 : tensor<1x4x1024x1024xf32>) outs(%23 : tensor<1x4x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %31 = arith.addf %in, %out : f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x1xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %24 : tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1xf32>) outs(%10 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %31 = arith.divf %in, %in_9 : f32
      linalg.yield %31 : f32
    } -> tensor<1x4x1024x1024xf32>
    %collapsed_6 = tensor.collapse_shape %25 [[0, 1], [2], [3]] : tensor<1x4x1024x1024xf32> into tensor<4x1024x1024xf32>
    %collapsed_7 = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<1x4x1024x64xf32> into tensor<4x1024x64xf32>
    %26 = tensor.empty() : tensor<4x1024x64xf32>
    %27 = linalg.fill ins(%cst : f32) outs(%26 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
    %28 = linalg.batch_matmul ins(%collapsed_6, %collapsed_7 : tensor<4x1024x1024xf32>, tensor<4x1024x64xf32>) outs(%27 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
    %expanded_8 = tensor.expand_shape %28 [[0, 1], [2], [3]] output_shape [1, 4, 1024, 64] : tensor<4x1024x64xf32> into tensor<1x4x1024x64xf32>
    %29 = tensor.empty() : tensor<1x4x1024x64xf16>
    %30 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_8 : tensor<1x4x1024x64xf32>) outs(%29 : tensor<1x4x1024x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %31 = arith.truncf %in : f32 to f16
      linalg.yield %31 : f16
    } -> tensor<1x4x1024x64xf16>
    return %30 : tensor<1x4x1024x64xf16>
  }
}
