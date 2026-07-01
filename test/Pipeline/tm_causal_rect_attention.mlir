// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin --load-dialect-plugin=%neptune_ta_plugin --load-dialect-plugin=%neptune_htile_plugin %s --transform-interpreter 2>&1 | FileCheck %s
//
// Transform-dialect schedule for a causal-rect masked-attention payload. This
// models cross-attention shape pressure: Q has a shorter sequence than K/V,
// and a non-causal K/V prefix mask leaves a mixed tile plus a dead suffix.

!any = !transform.any_op

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> ()>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>

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
        %bmm0 tile_sizes [1, 1, 128, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bscale = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    transform.fusion.into_producer %bscale into %forall_loop : (!any, !any) -> !any
    %bmask = transform.get_consumers_of_result %forall_loop[1] : (!any) -> !any
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

  func.func @attention(%arg0: tensor<1x4x128x64xf16>, %arg1: tensor<1x4x512x64xf16>, %arg2: tensor<1x4x512x64xf16>) -> tensor<1x4x128x64xf16> {
    %c0_i64 = arith.constant 0 : i64
    %c400_i64 = arith.constant 400 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %false = arith.constant false
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant dense<true> : tensor<128x512xi1>
    %cst_2 = arith.constant dense<0xFF800000> : tensor<f32>
    %cst_3 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<1x4x128x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x4x128x64xf16>) outs(%0 : tensor<1x4x128x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %30 = arith.extf %in : f16 to f32
      linalg.yield %30 : f32
    } -> tensor<1x4x128x64xf32>
    %2 = tensor.empty() : tensor<1x4x512x64xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x4x512x64xf16>) outs(%2 : tensor<1x4x512x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %30 = arith.extf %in : f16 to f32
      linalg.yield %30 : f32
    } -> tensor<1x4x512x64xf32>
    %4 = tensor.empty() : tensor<1x4x64x512xf32>
    %transposed = linalg.transpose ins(%3 : tensor<1x4x512x64xf32>) outs(%4 : tensor<1x4x64x512xf32>) permutation = [0, 1, 3, 2]
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x4x128x64xf32> into tensor<4x128x64xf32>
    %collapsed_4 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x4x64x512xf32> into tensor<4x64x512xf32>
    %5 = tensor.empty() : tensor<4x128x512xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x128x512xf32>) -> tensor<4x128x512xf32>
    %7 = linalg.batch_matmul ins(%collapsed, %collapsed_4 : tensor<4x128x64xf32>, tensor<4x64x512xf32>) outs(%6 : tensor<4x128x512xf32>) -> tensor<4x128x512xf32>
    %expanded = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 4, 128, 512] : tensor<4x128x512xf32> into tensor<1x4x128x512xf32>
    %8 = tensor.empty() : tensor<1x4x128x512xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x4x128x512xf32>) outs(%8 : tensor<1x4x128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.mulf %in, %cst_3 : f32
      linalg.yield %30 : f32
    } -> tensor<1x4x128x512xf32>
    %10 = tensor.empty() : tensor<128x512xi1>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<128x512xi1>) outs(%10 : tensor<128x512xi1>) {
    ^bb0(%in: i1, %out: i1):
      %30 = linalg.index 1 : index
      %31 = arith.index_cast %30 : index to i64
      %32 = arith.cmpi slt, %31, %c400_i64 : i64
      %33 = arith.select %32, %in, %false : i1
      linalg.yield %33 : i1
    } -> tensor<128x512xi1>
    %12 = linalg.generic {indexing_maps = [#map2, #map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11, %9, %cst_2 : tensor<128x512xi1>, tensor<1x4x128x512xf32>, tensor<f32>) outs(%8 : tensor<1x4x128x512xf32>) {
    ^bb0(%in: i1, %in_10: f32, %in_11: f32, %out: f32):
      %30 = arith.select %in, %in_10, %in_11 : f32
      linalg.yield %30 : f32
    } -> tensor<1x4x128x512xf32>
    %13 = tensor.empty() : tensor<1x4x128xi64>
    %14 = linalg.fill ins(%c0_i64 : i64) outs(%13 : tensor<1x4x128xi64>) -> tensor<1x4x128xi64>
    %15 = tensor.empty() : tensor<1x4x128xf32>
    %16 = linalg.fill ins(%cst_0 : f32) outs(%15 : tensor<1x4x128xf32>) -> tensor<1x4x128xf32>
    %17:2 = linalg.generic {indexing_maps = [#map, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%12 : tensor<1x4x128x512xf32>) outs(%16, %14 : tensor<1x4x128xf32>, tensor<1x4x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_10: i64):
      %30 = linalg.index 3 : index
      %31 = arith.index_cast %30 : index to i64
      %32 = arith.maximumf %in, %out : f32
      %33 = arith.cmpf ogt, %in, %out : f32
      %34 = arith.select %33, %31, %out_10 : i64
      linalg.yield %32, %34 : f32, i64
    } -> (tensor<1x4x128xf32>, tensor<1x4x128xi64>)
    %expanded_5 = tensor.expand_shape %17#0 [[0], [1], [2, 3]] output_shape [1, 4, 128, 1] : tensor<1x4x128xf32> into tensor<1x4x128x1xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %expanded_5 : tensor<1x4x128x512xf32>, tensor<1x4x128x1xf32>) outs(%8 : tensor<1x4x128x512xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %30 = arith.subf %in, %in_10 : f32
      linalg.yield %30 : f32
    } -> tensor<1x4x128x512xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18 : tensor<1x4x128x512xf32>) outs(%8 : tensor<1x4x128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = math.exp %in : f32
      linalg.yield %30 : f32
    } -> tensor<1x4x128x512xf32>
    %20 = tensor.empty() : tensor<1x4x128x1xf32>
    %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x4x128x1xf32>) -> tensor<1x4x128x1xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%19 : tensor<1x4x128x512xf32>) outs(%21 : tensor<1x4x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.addf %in, %out : f32
      linalg.yield %30 : f32
    } -> tensor<1x4x128x1xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19, %22 : tensor<1x4x128x512xf32>, tensor<1x4x128x1xf32>) outs(%8 : tensor<1x4x128x512xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %30 = arith.divf %in, %in_10 : f32
      linalg.yield %30 : f32
    } -> tensor<1x4x128x512xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x4x512x64xf16>) outs(%2 : tensor<1x4x512x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %30 = arith.extf %in : f16 to f32
      linalg.yield %30 : f32
    } -> tensor<1x4x512x64xf32>
    %collapsed_6 = tensor.collapse_shape %23 [[0, 1], [2], [3]] : tensor<1x4x128x512xf32> into tensor<4x128x512xf32>
    %collapsed_7 = tensor.collapse_shape %24 [[0, 1], [2], [3]] : tensor<1x4x512x64xf32> into tensor<4x512x64xf32>
    %25 = tensor.empty() : tensor<4x128x64xf32>
    %26 = linalg.fill ins(%cst : f32) outs(%25 : tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %27 = linalg.batch_matmul ins(%collapsed_6, %collapsed_7 : tensor<4x128x512xf32>, tensor<4x512x64xf32>) outs(%26 : tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %expanded_8 = tensor.expand_shape %27 [[0, 1], [2], [3]] output_shape [1, 4, 128, 64] : tensor<4x128x64xf32> into tensor<1x4x128x64xf32>
    %28 = tensor.empty() : tensor<1x4x128x64xf16>
    %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_8 : tensor<1x4x128x64xf32>) outs(%28 : tensor<1x4x128x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %30 = arith.truncf %in : f32 to f16
      linalg.yield %30 : f16
    } -> tensor<1x4x128x64xf16>
    return %29 : tensor<1x4x128x64xf16>
  }
}

// CHECK-LABEL: func.func @attention(
// CHECK-SAME: %arg0: memref<1x4x128x64xf16>, %arg1: memref<1x4x512x64xf16>, %arg2: memref<1x4x512x64xf16>, %arg3: memref<1x4x128x64xf16>)
// CHECK-NOT: tensor.empty() : tensor<4x128x64xf32>
// CHECK-NOT: tensor.empty() : tensor<4x128x64xf16>
// CHECK: scf.forall (%{{.*}}) in (4) {
// CHECK: htile.full %cst{{.*}} : f32 -> tensor<128xf32>
// CHECK: htile.full %cst{{.*}} : f32 -> tensor<128x64xf32>
// CHECK: %[[LIVE:.+]]:3 = scf.for %{{.*}} = %c0 to %c6 step %c1 iter_args(
// CHECK: htile.load %arg0{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : memref<1x4x128x64xf16> -> tensor<128x64xf16>
// CHECK: htile.load %arg1{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : memref<1x4x512x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} {transpose_b}
// CHECK: scf.yield
// CHECK: htile.load %arg0{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : memref<1x4x128x64xf16> -> tensor<128x64xf16>
// CHECK: htile.load %arg1{{\[}}%{{.*}}, %{{.*}}, %c384, %{{.*}}{{\]}} : memref<1x4x512x64xf16> -> tensor<64x64xf16>
// CHECK: htile.arange %c0 to %c64 : tensor<64xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [0] : tensor<64xindex> -> tensor<128x64xindex>
// CHECK: htile.full %c6 : index -> tensor<128x64xindex>
// CHECK: arith.muli %{{.*}}, %{{.*}} {{.*}} : tensor<128x64xindex>
// CHECK: arith.addi %{{.*}}, %{{.*}} : tensor<128x64xindex>
// CHECK: htile.full %c400_i64 : i64 -> tensor<128x64xi64>
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : tensor<128x64xi64>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64xi1>, tensor<128x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "max" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: math.exp2
// CHECK: htile.load %arg2{{\[}}%{{.*}}, %{{.*}}, %c384, %{{.*}}{{\]}} : memref<1x4x512x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64xf32>, tensor<64x64xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<128x64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK: htile.store %{{.*}}, %arg3{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : tensor<128x64xf16>, memref<1x4x128x64xf16>
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK: return
