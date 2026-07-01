// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin --load-dialect-plugin=%neptune_ta_plugin --load-dialect-plugin=%neptune_htile_plugin --transform-interpreter %s 2>&1 | FileCheck %s
//
// Transform-dialect schedule for Torch-MLIR ALiBi causal attention.
// This intentionally starts from the masked-attention schedule because ALiBi
// is a score-side bias before the causal mask and online softmax.

!any = !transform.any_op

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1, d2, d3) -> ()>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_3d_1d_reduction(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.structured %candidate : (!any) -> !any {
    ^bb0(%op: !any):
      transform.match.structured.dim %op[0, 1, 2] {parallel} : !any
      transform.match.structured.dim %op[3] {reduction} : !any
      transform.match.structured.yield %op : !any
    }
    transform.yield %matched : !any
  }

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

  transform.named_sequence @return_matched(%arg: !any {transform.readonly}) -> !any {
    transform.yield %arg : !any
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

    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 64, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bscale = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    transform.fusion.into_producer %bscale into %forall_loop : (!any, !any) -> !any
    %balibi = transform.get_consumers_of_result %forall_loop[1] : (!any) -> !any
    transform.linalg.greedy_inline_elementwise %balibi : !any
    transform.fusion.into_producer %balibi into %forall_loop : (!any, !any) -> !any
    %bmask = transform.get_consumers_of_result %forall_loop[2] : (!any) -> !any
    transform.linalg.greedy_inline_elementwise %bmask : !any
    %fused_bmask = transform.fusion.into_producer %bmask into %forall_loop : (!any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %consumers = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %_2, %bmax = transform.foreach_match restrict_root in %consumers
        @match_3d_1d_reduction -> @return_matched : (!any) -> (!any, !any)
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
    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !any
    transform.apply_cse to %func : !any

    // 0xFF800000: -inf in f32
    %live_loop, %mixed_loop = transform.loop.specialize_dead_tile %fused_bmask in %j0_loop
        {dead_value = 0xFF800000 : f32} : (!any, !any) -> (!any, !any)

    // --- HTile lowering begins ---
    transform.htile.linalg_to_semantic %func : !any
    transform.htile.semantic_to_kernel_abi %func : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf16>, %arg2: tensor<1x4x1024x64xf16>, %arg3: tensor<4xf32>) -> tensor<1x4x1024x64xf16> {
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
      %37 = arith.extf %in : f16 to f32
      linalg.yield %37 : f32
    } -> tensor<1x4x1024x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x4x1024x64xf16>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %37 = arith.extf %in : f16 to f32
      linalg.yield %37 : f32
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
      %37 = arith.mulf %in, %cst_3 : f32
      linalg.yield %37 : f32
    } -> tensor<1x4x1024x1024xf32>
    %9 = tensor.empty() : tensor<1024xi64>
    %10 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%9 : tensor<1024xi64>) {
    ^bb0(%out: i64):
      %37 = linalg.index 0 : index
      %38 = arith.index_cast %37 : index to i64
      linalg.yield %38 : i64
    } -> tensor<1024xi64>
    %11 = tensor.empty() : tensor<1024xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%10 : tensor<1024xi64>) outs(%11 : tensor<1024xf32>) {
    ^bb0(%in: i64, %out: f32):
      %37 = arith.sitofp %in : i64 to f32
      linalg.yield %37 : f32
    } -> tensor<1024xf32>
    %expanded_5 = tensor.expand_shape %12 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1024] : tensor<1024xf32> into tensor<1x1x1x1024xf32>
    %expanded_6 = tensor.expand_shape %12 [[0, 1, 2, 3]] output_shape [1, 1, 1024, 1] : tensor<1024xf32> into tensor<1x1x1024x1xf32>
    %13 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_5, %expanded_6 : tensor<1x1x1x1024xf32>, tensor<1x1x1024x1xf32>) outs(%13 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: f32, %in_13: f32, %out: f32):
      %37 = arith.subf %in, %in_13 : f32
      linalg.yield %37 : f32
    } -> tensor<1x1x1024x1024xf32>
    %expanded_7 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 4, 1, 1] : tensor<4xf32> into tensor<1x4x1x1xf32>
    %15 = linalg.generic {indexing_maps = [#map4, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %expanded_7 : tensor<1x1x1024x1024xf32>, tensor<1x4x1x1xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_13: f32, %out: f32):
      %37 = arith.mulf %in, %in_13 : f32
      linalg.yield %37 : f32
    } -> tensor<1x4x1024x1024xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %15 : tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1024xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_13: f32, %out: f32):
      %37 = arith.addf %in, %in_13 : f32
      linalg.yield %37 : f32
    } -> tensor<1x4x1024x1024xf32>
    %17 = tensor.empty() : tensor<1024x1024xi1>
    %18 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<1024x1024xi1>) outs(%17 : tensor<1024x1024xi1>) {
    ^bb0(%in: i1, %out: i1):
      %37 = linalg.index 0 : index
      %38 = arith.index_cast %37 : index to i64
      %39 = linalg.index 1 : index
      %40 = arith.index_cast %39 : index to i64
      %41 = arith.cmpi sle, %40, %38 : i64
      %42 = arith.select %41, %in, %false : i1
      linalg.yield %42 : i1
    } -> tensor<1024x1024xi1>
    %expanded_8 = tensor.expand_shape %18 [[0, 1, 2], [3]] output_shape [1, 1, 1024, 1024] : tensor<1024x1024xi1> into tensor<1x1x1024x1024xi1>
    %19 = linalg.generic {indexing_maps = [#map4, #map, #map7, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_8, %16, %cst_2 : tensor<1x1x1024x1024xi1>, tensor<1x4x1024x1024xf32>, tensor<f32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: i1, %in_13: f32, %in_14: f32, %out: f32):
      %37 = arith.select %in, %in_13, %in_14 : f32
      linalg.yield %37 : f32
    } -> tensor<1x4x1024x1024xf32>
    %20 = tensor.empty() : tensor<1x4x1024xi64>
    %21 = linalg.fill ins(%c0_i64 : i64) outs(%20 : tensor<1x4x1024xi64>) -> tensor<1x4x1024xi64>
    %22 = tensor.empty() : tensor<1x4x1024xf32>
    %23 = linalg.fill ins(%cst_0 : f32) outs(%22 : tensor<1x4x1024xf32>) -> tensor<1x4x1024xf32>
    %24:2 = linalg.generic {indexing_maps = [#map, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%19 : tensor<1x4x1024x1024xf32>) outs(%23, %21 : tensor<1x4x1024xf32>, tensor<1x4x1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_13: i64):
      %37 = linalg.index 3 : index
      %38 = arith.index_cast %37 : index to i64
      %39 = arith.maximumf %in, %out : f32
      %40 = arith.cmpf ogt, %in, %out : f32
      %41 = arith.select %40, %38, %out_13 : i64
      linalg.yield %39, %41 : f32, i64
    } -> (tensor<1x4x1024xf32>, tensor<1x4x1024xi64>)
    %expanded_9 = tensor.expand_shape %24#0 [[0], [1], [2, 3]] output_shape [1, 4, 1024, 1] : tensor<1x4x1024xf32> into tensor<1x4x1024x1xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19, %expanded_9 : tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_13: f32, %out: f32):
      %37 = arith.subf %in, %in_13 : f32
      linalg.yield %37 : f32
    } -> tensor<1x4x1024x1024xf32>
    %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25 : tensor<1x4x1024x1024xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %37 = math.exp %in : f32
      linalg.yield %37 : f32
    } -> tensor<1x4x1024x1024xf32>
    %27 = tensor.empty() : tensor<1x4x1024x1xf32>
    %28 = linalg.fill ins(%cst : f32) outs(%27 : tensor<1x4x1024x1xf32>) -> tensor<1x4x1024x1xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%26 : tensor<1x4x1024x1024xf32>) outs(%28 : tensor<1x4x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %37 = arith.addf %in, %out : f32
      linalg.yield %37 : f32
    } -> tensor<1x4x1024x1xf32>
    %30 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %29 : tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1xf32>) outs(%7 : tensor<1x4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_13: f32, %out: f32):
      %37 = arith.divf %in, %in_13 : f32
      linalg.yield %37 : f32
    } -> tensor<1x4x1024x1024xf32>
    %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x4x1024x64xf16>) outs(%0 : tensor<1x4x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %37 = arith.extf %in : f16 to f32
      linalg.yield %37 : f32
    } -> tensor<1x4x1024x64xf32>
    %collapsed_10 = tensor.collapse_shape %30 [[0, 1], [2], [3]] : tensor<1x4x1024x1024xf32> into tensor<4x1024x1024xf32>
    %collapsed_11 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<1x4x1024x64xf32> into tensor<4x1024x64xf32>
    %32 = tensor.empty() : tensor<4x1024x64xf32>
    %33 = linalg.fill ins(%cst : f32) outs(%32 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
    %34 = linalg.batch_matmul ins(%collapsed_10, %collapsed_11 : tensor<4x1024x1024xf32>, tensor<4x1024x64xf32>) outs(%33 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
    %expanded_12 = tensor.expand_shape %34 [[0, 1], [2], [3]] output_shape [1, 4, 1024, 64] : tensor<4x1024x64xf32> into tensor<1x4x1024x64xf32>
    %35 = tensor.empty() : tensor<1x4x1024x64xf16>
    %36 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_12 : tensor<1x4x1024x64xf32>) outs(%35 : tensor<1x4x1024x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %37 = arith.truncf %in : f32 to f16
      linalg.yield %37 : f16
    } -> tensor<1x4x1024x64xf16>
    return %36 : tensor<1x4x1024x64xf16>
  }
}

// CHECK-LABEL: func.func @attention(
// CHECK-SAME: %arg0: memref<1x4x1024x64xf16>, %arg1: memref<1x4x1024x64xf16>, %arg2: memref<1x4x1024x64xf16>, %arg3: memref<4xf32>, %arg4: memref<1x4x1024x64xf16>)
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf32>
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf16>
// CHECK: scf.forall (%[[H:.*]], %[[I_TILE:.*]]) in (4, 16) {
// CHECK: htile.full %{{.*}} : f32 -> tensor<64xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<64x64xf32>
// CHECK: %[[RAW_BOUND:.+]] = arith.maxsi %[[I_TILE]], %c0 : index
// CHECK: %[[CAPPED_BOUND:.+]] = arith.minsi %[[RAW_BOUND]], %c16 : index
// CHECK: %[[LIVE:.+]]:3 = scf.for %[[J_TILE:.*]] = %c0 to %[[CAPPED_BOUND]] step %c1 iter_args(
// CHECK: htile.load %arg0{{\[}}%{{.*}}, %[[H]], %{{.*}}, %{{.*}}{{\]}} : memref<1x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.load %arg1{{\[}}%{{.*}}, %[[H]], %{{.*}}, %{{.*}}{{\]}} : memref<1x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} {transpose_b} : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.load %arg3{{\[}}%[[H]]{{\]}} : memref<4xf32> -> tensor<f32>
// CHECK: htile.broadcast %{{.*}} dimensions = [0, 1] : tensor<f32> -> tensor<64x64xf32>
// CHECK: htile.arange %c0 to %c64 : tensor<64xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [1] : tensor<64xindex> -> tensor<64x64xindex>
// CHECK: arith.muli %{{.*}}, %{{.*}} {{.*}} : tensor<64x64xindex>
// CHECK: arith.addi %{{.*}}, %{{.*}} : tensor<64x64xindex>
// CHECK: htile.arange %c0 to %c64 : tensor<64xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [0] : tensor<64xindex> -> tensor<64x64xindex>
// CHECK: arith.subf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<64x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: %[[LIVE_SCORES:.+]] = arith.addf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK-NOT: arith.select
// CHECK: htile.reduce %[[LIVE_SCORES]] axis 1 kind "max" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: math.exp2
// CHECK: htile.load %arg2{{\[}}%{{.*}}, %[[H]], %{{.*}}, %{{.*}}{{\]}} : memref<1x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: %[[DEAD_BOUND_RAW:.+]] = affine.apply
// CHECK: %[[DEAD_BOUND_MAX:.+]] = arith.maxsi %[[DEAD_BOUND_RAW]], %c0 : index
// CHECK: %[[DEAD_BOUND:.+]] = arith.minsi %[[DEAD_BOUND_MAX]], %c16 : index
// CHECK: %[[MIXED:.+]]:3 = scf.for %{{.*}} = %[[CAPPED_BOUND]] to %[[DEAD_BOUND]] step %c1 iter_args(%{{.*}} = %[[LIVE]]#0, %{{.*}} = %[[LIVE]]#1, %{{.*}} = %[[LIVE]]#2)
// CHECK: htile.load %arg0{{\[}}%{{.*}}, %[[H]], %{{.*}}, %{{.*}}{{\]}} : memref<1x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.load %arg1{{\[}}%{{.*}}, %[[H]], %{{.*}}, %{{.*}}{{\]}} : memref<1x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} {transpose_b} : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.load %arg3{{\[}}%[[H]]{{\]}} : memref<4xf32> -> tensor<f32>
// CHECK: htile.broadcast %{{.*}} dimensions = [0, 1] : tensor<f32> -> tensor<64x64xf32>
// CHECK: arith.subf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<64x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.addf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.cmpi sle, %{{.*}}, %{{.*}} : tensor<64x64xi64>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "max" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: math.exp2
// CHECK: htile.load %arg2{{\[}}%{{.*}}, %[[H]], %{{.*}}, %{{.*}}{{\]}} : memref<1x4x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK: htile.store %{{.*}}, %arg4{{\[}}%{{.*}}, %[[H]], %{{.*}}, %{{.*}}{{\]}} : tensor<64x64xf16>, memref<1x4x1024x64xf16>
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK-NOT: ta.
// CHECK: return
