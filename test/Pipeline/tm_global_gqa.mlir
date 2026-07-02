// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin --load-dialect-plugin=%neptune_ta_plugin --load-dialect-plugin=%neptune_htile_plugin %s --transform-interpreter 2>&1 | FileCheck %s
//
// Transform-dialect schedule that transforms the Torch-MLIR GQA payload below
// into a FlashAttention-like fused program with a `(group, head)` outer loop shape.

!any = !transform.any_op
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, 0)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_4d_matmul_transb(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b g h i d, b h j d -> b g h i j"} : (!any) -> !any
    transform.yield %matched : !any
  }

  transform.named_sequence @match_4d_matmul(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b g h i j, b h j d -> b g h i d"} : (!any) -> !any
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

    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 1, 128, 64, 0] : (!any) -> (!any, !any)
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

    // --- HTile lowering begins ---
    transform.htile.linalg_to_semantic %func : !any
    transform.htile.semantic_to_kernel_abi %func : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x2x1024x64xf16>, %arg2: tensor<1x2x1024x64xf16>) -> tensor<1x2x2x1024x64xf16> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4]] output_shape [1, 2, 2, 1024, 64] : tensor<1x4x1024x64xf16> into tensor<1x2x2x1024x64xf16>
    %0 = tensor.empty() : tensor<1x2x2x1024x64xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x2x1024x64xf16>) outs(%0 : tensor<1x2x2x1024x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x2x2x1024x64xf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x2x1024x64xf16>) outs(%0 : tensor<1x2x2x1024x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x2x2x1024x64xf16>
    %3 = tensor.empty() : tensor<1x2x2x1024x64xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x2x2x1024x64xf16>) outs(%3 : tensor<1x2x2x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %28 = arith.extf %in : f16 to f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x1024x64xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x2x2x1024x64xf16>) outs(%3 : tensor<1x2x2x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %28 = arith.extf %in : f16 to f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x1024x64xf32>
    %6 = tensor.empty() : tensor<1x2x2x64x1024xf32>
    %transposed = linalg.transpose ins(%5 : tensor<1x2x2x1024x64xf32>) outs(%6 : tensor<1x2x2x64x1024xf32>) permutation = [0, 1, 2, 4, 3]
    %collapsed = tensor.collapse_shape %4 [[0, 1, 2], [3], [4]] : tensor<1x2x2x1024x64xf32> into tensor<4x1024x64xf32>
    %collapsed_2 = tensor.collapse_shape %transposed [[0, 1, 2], [3], [4]] : tensor<1x2x2x64x1024xf32> into tensor<4x64x1024xf32>
    %7 = tensor.empty() : tensor<4x1024x1024xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %9 = linalg.batch_matmul ins(%collapsed, %collapsed_2 : tensor<4x1024x64xf32>, tensor<4x64x1024xf32>) outs(%8 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %expanded_3 = tensor.expand_shape %9 [[0, 1, 2], [3], [4]] output_shape [1, 2, 2, 1024, 1024] : tensor<4x1024x1024xf32> into tensor<1x2x2x1024x1024xf32>
    %10 = tensor.empty() : tensor<1x2x2x1024x1024xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<1x2x2x1024x1024xf32>) outs(%10 : tensor<1x2x2x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %28 = arith.mulf %in, %cst_1 : f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x1024x1024xf32>
    %12 = tensor.empty() : tensor<1x2x2x1024xi64>
    %13 = linalg.fill ins(%c0_i64 : i64) outs(%12 : tensor<1x2x2x1024xi64>) -> tensor<1x2x2x1024xi64>
    %14 = tensor.empty() : tensor<1x2x2x1024xf32>
    %15 = linalg.fill ins(%cst_0 : f32) outs(%14 : tensor<1x2x2x1024xf32>) -> tensor<1x2x2x1024xf32>
    %16:2 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%11 : tensor<1x2x2x1024x1024xf32>) outs(%15, %13 : tensor<1x2x2x1024xf32>, tensor<1x2x2x1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_8: i64):
      %28 = linalg.index 4 : index
      %29 = arith.index_cast %28 : index to i64
      %30 = arith.maximumf %in, %out : f32
      %31 = arith.cmpf ogt, %in, %out : f32
      %32 = arith.select %31, %29, %out_8 : i64
      linalg.yield %30, %32 : f32, i64
    } -> (tensor<1x2x2x1024xf32>, tensor<1x2x2x1024xi64>)
    %expanded_4 = tensor.expand_shape %16#0 [[0], [1], [2], [3, 4]] output_shape [1, 2, 2, 1024, 1] : tensor<1x2x2x1024xf32> into tensor<1x2x2x1024x1xf32>
    %17 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%11, %expanded_4 : tensor<1x2x2x1024x1024xf32>, tensor<1x2x2x1024x1xf32>) outs(%10 : tensor<1x2x2x1024x1024xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %28 = arith.subf %in, %in_8 : f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x1024x1024xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x2x2x1024x1024xf32>) outs(%10 : tensor<1x2x2x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %28 = math.exp %in : f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x1024x1024xf32>
    %19 = tensor.empty() : tensor<1x2x2x1024x1xf32>
    %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x2x2x1024x1xf32>) -> tensor<1x2x2x1024x1xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%18 : tensor<1x2x2x1024x1024xf32>) outs(%20 : tensor<1x2x2x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %28 = arith.addf %in, %out : f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x1024x1xf32>
    %22 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%18, %21 : tensor<1x2x2x1024x1024xf32>, tensor<1x2x2x1024x1xf32>) outs(%10 : tensor<1x2x2x1024x1024xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %28 = arith.divf %in, %in_8 : f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x1024x1024xf32>
    %23 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x2x2x1024x64xf16>) outs(%3 : tensor<1x2x2x1024x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %28 = arith.extf %in : f16 to f32
      linalg.yield %28 : f32
    } -> tensor<1x2x2x1024x64xf32>
    %collapsed_5 = tensor.collapse_shape %22 [[0, 1, 2], [3], [4]] : tensor<1x2x2x1024x1024xf32> into tensor<4x1024x1024xf32>
    %collapsed_6 = tensor.collapse_shape %23 [[0, 1, 2], [3], [4]] : tensor<1x2x2x1024x64xf32> into tensor<4x1024x64xf32>
    %24 = tensor.empty() : tensor<4x1024x64xf32>
    %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
    %26 = linalg.batch_matmul ins(%collapsed_5, %collapsed_6 : tensor<4x1024x1024xf32>, tensor<4x1024x64xf32>) outs(%25 : tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32>
    %expanded_7 = tensor.expand_shape %26 [[0, 1, 2], [3], [4]] output_shape [1, 2, 2, 1024, 64] : tensor<4x1024x64xf32> into tensor<1x2x2x1024x64xf32>
    %27 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7 : tensor<1x2x2x1024x64xf32>) outs(%0 : tensor<1x2x2x1024x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %28 = arith.truncf %in : f32 to f16
      linalg.yield %28 : f16
    } -> tensor<1x2x2x1024x64xf16>
    return %27 : tensor<1x2x2x1024x64xf16>
  }
}

// CHECK-LABEL: func.func @attention(
// CHECK-SAME: %arg0: memref<1x4x1024x64xf16>, %arg1: memref<1x2x1024x64xf16>, %arg2: memref<1x2x1024x64xf16>, %arg3: memref<1x2x2x1024x64xf16>)
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: linalg.transpose
// CHECK-NOT: linalg.generic
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf32>
// CHECK-NOT: tensor.empty() : tensor<2x2x1024x64xf16>
// CHECK: scf.forall (%{{.*}}, %{{.*}}, %{{.*}}) in (2, 2, 8) {
// CHECK: htile.full %{{.*}} : f32 -> tensor<128xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<128x64xf32>
// CHECK: %{{.*}}:3 = scf.for %{{.*}} = %c0 to %c16 step %c1 iter_args(
// CHECK: affine.apply
// CHECK: affine.apply
// CHECK: htile.load %arg0{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : memref<1x4x1024x64xf16> -> tensor<128x64xf16>
// CHECK: htile.load %arg1{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : memref<1x2x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} {transpose_b} : tensor<128x64xf16>, tensor<64x64xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "max" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: htile.broadcast %{{.*}} dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<128x64xf32>
// CHECK: htile.load %arg2{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : memref<1x2x1024x64xf16> -> tensor<64x64xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64xf32>, tensor<64x64xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<128x64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK: htile.store %{{.*}}, %arg3{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : tensor<128x64xf16>, memref<1x2x2x1024x64xf16>
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK: return
