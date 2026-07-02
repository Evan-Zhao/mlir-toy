// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin --load-dialect-plugin=%neptune_ta_plugin --load-dialect-plugin=%neptune_htile_plugin --load-pass-plugin=%neptune_ta_plugin %s --transform-interpreter 2>&1 | FileCheck %s
//
// Transform-dialect schedule for the windowed-attention payload through HTile.

!any = !transform.any_op

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> ()>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_4d_matmul_transb(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b h i d, b h j d -> b h i j"} : (!any) -> !any
    transform.yield %matched : !any
  }

  transform.named_sequence @match_4d_matmul(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b h i j, b h j d -> b h i d"} : (!any) -> !any
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
    %bmm1 = transform.collect_matching @match_4d_matmul in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 64, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bmax, %_2 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %prefix = transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %bmax { inline_elementwise } : (!any, !any) -> !any
    transform.linalg.erase_unused_operands_and_results %prefix : !any

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

    transform.apply_cse to %func : !any
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
    transform.htile.semantic_to_kernel_abi %func : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    transform.yield
  }

  func.func @attention(%arg0: tensor<2x4x1024x64xf16>, %arg1: tensor<2x4x1024x64xf16>, %arg2: tensor<2x4x1024x64xf16>) -> tensor<2x4x1024x64xf16> {
    %c-127_i64 = arith.constant -127 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %false = arith.constant false
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant dense<true> : tensor<1024x1024xi1>
    %cst_2 = arith.constant dense<0xFF800000> : tensor<f32>
    %cst_3 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<2x4x1024x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x4x1024x64xf16>) outs(%0 : tensor<2x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %35 = arith.extf %in : f16 to f32
      linalg.yield %35 : f32
    } -> tensor<2x4x1024x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<2x4x1024x64xf16>) outs(%0 : tensor<2x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %35 = arith.extf %in : f16 to f32
      linalg.yield %35 : f32
    } -> tensor<2x4x1024x64xf32>
    %3 = tensor.empty() : tensor<2x4x64x1024xf32>
    %transposed = linalg.transpose ins(%2 : tensor<2x4x1024x64xf32>) outs(%3 : tensor<2x4x64x1024xf32>) permutation = [0, 1, 3, 2]
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<2x4x1024x64xf32> into tensor<8x1024x64xf32>
    %collapsed_4 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<2x4x64x1024xf32> into tensor<8x64x1024xf32>
    %4 = tensor.empty() : tensor<8x1024x1024xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<8x1024x1024xf32>) -> tensor<8x1024x1024xf32>
    %6 = linalg.batch_matmul ins(%collapsed, %collapsed_4 : tensor<8x1024x64xf32>, tensor<8x64x1024xf32>) outs(%5 : tensor<8x1024x1024xf32>) -> tensor<8x1024x1024xf32>
    %expanded = tensor.expand_shape %6 [[0, 1], [2], [3]] output_shape [2, 4, 1024, 1024] : tensor<8x1024x1024xf32> into tensor<2x4x1024x1024xf32>
    %7 = tensor.empty() : tensor<2x4x1024x1024xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<2x4x1024x1024xf32>) outs(%7 : tensor<2x4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.mulf %in, %cst_3 : f32
      linalg.yield %35 : f32
    } -> tensor<2x4x1024x1024xf32>
    %9 = tensor.empty() : tensor<1024x1024xi1>
    %10 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<1024x1024xi1>) outs(%9 : tensor<1024x1024xi1>) {
    ^bb0(%in: i1, %out: i1):
      %35 = linalg.index 0 : index
      %36 = arith.index_cast %35 : index to i64
      %37 = linalg.index 1 : index
      %38 = arith.index_cast %37 : index to i64
      %39 = arith.cmpi sle, %38, %36 : i64
      %40 = arith.select %39, %in, %false : i1
      linalg.yield %40 : i1
    } -> tensor<1024x1024xi1>
    %11 = tensor.empty() : tensor<1024xi64>
    %12 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%11 : tensor<1024xi64>) {
    ^bb0(%out: i64):
      %35 = linalg.index 0 : index
      %36 = arith.index_cast %35 : index to i64
      linalg.yield %36 : i64
    } -> tensor<1024xi64>
    %expanded_5 = tensor.expand_shape %12 [[0, 1]] output_shape [1, 1024] : tensor<1024xi64> into tensor<1x1024xi64>
    %expanded_6 = tensor.expand_shape %12 [[0, 1]] output_shape [1024, 1] : tensor<1024xi64> into tensor<1024x1xi64>
    %13 = tensor.empty() : tensor<1024x1024xi64>
    %14 = linalg.generic {indexing_maps = [#map3, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_5, %expanded_6 : tensor<1x1024xi64>, tensor<1024x1xi64>) outs(%13 : tensor<1024x1024xi64>) {
    ^bb0(%in: i64, %in_12: i64, %out: i64):
      %35 = arith.subi %in, %in_12 : i64
      linalg.yield %35 : i64
    } -> tensor<1024x1024xi64>
    %15 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<1024x1024xi64>) outs(%9 : tensor<1024x1024xi1>) {
    ^bb0(%in: i64, %out: i1):
      %35 = arith.cmpi sge, %in, %c-127_i64 : i64
      linalg.yield %35 : i1
    } -> tensor<1024x1024xi1>
    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%15, %10 : tensor<1024x1024xi1>, tensor<1024x1024xi1>) outs(%9 : tensor<1024x1024xi1>) {
    ^bb0(%in: i1, %in_12: i1, %out: i1):
      %35 = arith.andi %in, %in_12 : i1
      linalg.yield %35 : i1
    } -> tensor<1024x1024xi1>
    %expanded_7 = tensor.expand_shape %16 [[0, 1, 2], [3]] output_shape [1, 1, 1024, 1024] : tensor<1024x1024xi1> into tensor<1x1x1024x1024xi1>
    %17 = linalg.generic {indexing_maps = [#map5, #map, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7, %8, %cst_2 : tensor<1x1x1024x1024xi1>, tensor<2x4x1024x1024xf32>, tensor<f32>) outs(%7 : tensor<2x4x1024x1024xf32>) {
    ^bb0(%in: i1, %in_12: f32, %in_13: f32, %out: f32):
      %35 = arith.select %in, %in_12, %in_13 : f32
      linalg.yield %35 : f32
    } -> tensor<2x4x1024x1024xf32>
    %18 = tensor.empty() : tensor<2x4x1024xi64>
    %19 = linalg.fill ins(%c0_i64 : i64) outs(%18 : tensor<2x4x1024xi64>) -> tensor<2x4x1024xi64>
    %20 = tensor.empty() : tensor<2x4x1024xf32>
    %21 = linalg.fill ins(%cst_0 : f32) outs(%20 : tensor<2x4x1024xf32>) -> tensor<2x4x1024xf32>
    %22:2 = linalg.generic {indexing_maps = [#map, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%17 : tensor<2x4x1024x1024xf32>) outs(%21, %19 : tensor<2x4x1024xf32>, tensor<2x4x1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_12: i64):
      %35 = linalg.index 3 : index
      %36 = arith.index_cast %35 : index to i64
      %37 = arith.maximumf %in, %out : f32
      %38 = arith.cmpf ogt, %in, %out : f32
      %39 = arith.select %38, %36, %out_12 : i64
      linalg.yield %37, %39 : f32, i64
    } -> (tensor<2x4x1024xf32>, tensor<2x4x1024xi64>)
    %expanded_8 = tensor.expand_shape %22#0 [[0], [1], [2, 3]] output_shape [2, 4, 1024, 1] : tensor<2x4x1024xf32> into tensor<2x4x1024x1xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map8, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %expanded_8 : tensor<2x4x1024x1024xf32>, tensor<2x4x1024x1xf32>) outs(%7 : tensor<2x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.subf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<2x4x1024x1024xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<2x4x1024x1024xf32>) outs(%7 : tensor<2x4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = math.exp %in : f32
      linalg.yield %35 : f32
    } -> tensor<2x4x1024x1024xf32>
    %25 = tensor.empty() : tensor<2x4x1024x1xf32>
    %26 = linalg.fill ins(%cst : f32) outs(%25 : tensor<2x4x1024x1xf32>) -> tensor<2x4x1024x1xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%24 : tensor<2x4x1024x1024xf32>) outs(%26 : tensor<2x4x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    } -> tensor<2x4x1024x1xf32>
    %28 = linalg.generic {indexing_maps = [#map, #map8, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %27 : tensor<2x4x1024x1024xf32>, tensor<2x4x1024x1xf32>) outs(%7 : tensor<2x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.divf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<2x4x1024x1024xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<2x4x1024x64xf16>) outs(%0 : tensor<2x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %35 = arith.extf %in : f16 to f32
      linalg.yield %35 : f32
    } -> tensor<2x4x1024x64xf32>
    %collapsed_9 = tensor.collapse_shape %28 [[0, 1], [2], [3]] : tensor<2x4x1024x1024xf32> into tensor<8x1024x1024xf32>
    %collapsed_10 = tensor.collapse_shape %29 [[0, 1], [2], [3]] : tensor<2x4x1024x64xf32> into tensor<8x1024x64xf32>
    %30 = tensor.empty() : tensor<8x1024x64xf32>
    %31 = linalg.fill ins(%cst : f32) outs(%30 : tensor<8x1024x64xf32>) -> tensor<8x1024x64xf32>
    %32 = linalg.batch_matmul ins(%collapsed_9, %collapsed_10 : tensor<8x1024x1024xf32>, tensor<8x1024x64xf32>) outs(%31 : tensor<8x1024x64xf32>) -> tensor<8x1024x64xf32>
    %expanded_11 = tensor.expand_shape %32 [[0, 1], [2], [3]] output_shape [2, 4, 1024, 64] : tensor<8x1024x64xf32> into tensor<2x4x1024x64xf32>
    %33 = tensor.empty() : tensor<2x4x1024x64xf16>
    %34 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_11 : tensor<2x4x1024x64xf32>) outs(%33 : tensor<2x4x1024x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %35 = arith.truncf %in : f32 to f16
      linalg.yield %35 : f16
    } -> tensor<2x4x1024x64xf16>
    return %34 : tensor<2x4x1024x64xf16>
  }
}

// CHECK-LABEL: func.func @attention(
// CHECK-SAME: %arg0: memref<2x4x1024x64xf16>, %arg1: memref<2x4x1024x64xf16>, %arg2: memref<2x4x1024x64xf16>, %arg3: memref<2x4x1024x64xf16>)
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf32>
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf16>
// CHECK: scf.forall (%[[B:.*]], %[[H:.*]], %[[I_TILE:.*]]) in (2, 4, 16) {
// CHECK: htile.full %{{.*}} : f32 -> tensor<64xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<64x64xf32>
// CHECK: %[[LIVE_LOWER_RAW:.*]] = affine.apply
// CHECK: %[[LIVE_LOWER_CLAMP_LO:.*]] = arith.maxsi %[[LIVE_LOWER_RAW]], %c0 : index
// CHECK: %[[LIVE_LOWER:.*]] = arith.minsi %[[LIVE_LOWER_CLAMP_LO]], %c16 : index
// CHECK: %[[LIVE_UPPER_CLAMP_LO:.*]] = arith.maxsi %[[I_TILE]], %c0 : index
// CHECK: %[[LIVE_UPPER:.*]] = arith.minsi %[[LIVE_UPPER_CLAMP_LO]], %c16 : index
// CHECK: %[[LIVE_LOOP:.*]]:3 = scf.for %{{.*}} = %[[LIVE_LOWER]] to %[[LIVE_UPPER]] step %c1 iter_args(%{{.*}}, %{{.*}}, %{{.*}})
// CHECK: htile.load %arg0{{\[}}%[[B]], %{{.*}}, %{{.*}}{{\]}} : memref<2x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.load %arg1{{\[}}%[[B]], %{{.*}}, %{{.*}}{{\]}} : memref<2x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} {transpose_b} : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "max" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: math.exp2
// CHECK: htile.load %arg2{{\[}}%[[B]], %{{.*}}, %{{.*}}{{\]}} : memref<2x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: scf.yield
// CHECK: %[[DEAD_BOUND_RAW:.*]] = affine.apply
// CHECK: %[[DEAD_BOUND_MAX:.*]] = arith.maxsi %[[DEAD_BOUND_RAW]], %c0 : index
// CHECK: %[[DEAD_BOUND:.*]] = arith.minsi %[[DEAD_BOUND_MAX]], %c16 : index
// CHECK: %[[MIXED_LOOP:.*]]:3 = scf.for %{{.*}} = %[[LIVE_UPPER]] to %[[DEAD_BOUND]] step %c1 iter_args(%{{.*}} = %[[LIVE_LOOP]]#0, %{{.*}} = %[[LIVE_LOOP]]#1, %{{.*}} = %[[LIVE_LOOP]]#2)
// CHECK: htile.load %arg0{{\[}}%[[B]], %{{.*}}, %{{.*}}{{\]}} : memref<2x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.load %arg1{{\[}}%[[B]], %{{.*}}, %{{.*}}{{\]}} : memref<2x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} {transpose_b} : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.arange %c0 to %c64 : tensor<64xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [1] : tensor<64xindex> -> tensor<64x64xindex>
// CHECK: arith.subi
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "max" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: math.exp2
// CHECK: htile.load %arg2{{\[}}%[[B]], %{{.*}}, %{{.*}}{{\]}} : memref<2x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK: htile.store %{{.*}}, %arg3{{\[}}%[[B]], %{{.*}}, %{{.*}}{{\]}} : tensor<64x64xf16>, memref<2x4x1024x64xf16>
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK-NOT: ta.
// CHECK: return
