// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin --load-dialect-plugin=%neptune_ta_plugin --load-dialect-plugin=%neptune_htile_plugin %s --transform-interpreter 2>&1 | FileCheck %s
//
// Transform-dialect schedule that transforms the Torch-MLIR attention payload
// below into a FlashAttention-like fused program.
// Mirrors `_schedule_attention_flash` from Neptune.

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

  transform.named_sequence @match_4d_matmul(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b h i j, b h j d -> b h i d"} : (!any) -> !any
    transform.yield %matched : !any
  }

  transform.named_sequence @__transform_main(%module: !any) {
    // Prepass. Translate the input function from linalg to `ta` dialect, which enables
    // more flexible expression rewrites. Apply rewrites, then translate back to linalg.
    %func0 = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %func1 = transform.apply_registered_pass "linalg-generalize-named-ops" to %func0 : (!any) -> !any
    %func = transform.apply_registered_pass "linalg-to-ta" to %func1 : (!any) -> !any
    // This canonicalization step folds trunc(const(f64), f32) into a constant in f32.
    // This is only used in this test case, because only this test case has an f64 constant.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    // Replace `exp(x)` with `exp2(x * log2(e))`, then push `log2(e)` constant around
    // until it folds with other multiplicative constants.
    transform.ta.rewrite_exp_to_exp2 %func : !any
    // Replace `matmul(P_ij / s_i, V_jd)` with `matmul(P_ij, V_jd) / s_i`.
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !any
    // This is a special idiom: use einsum to match matmuls in `ta` dialect is easy.
    // Then the `to_linalg` translator keeps these handles alive even after the translation,
    // so you get %bmm0 to point to the first matmul in linalg.
    %bmm0 = transform.collect_matching @match_4d_matmul_transb in %func : (!any) -> !any
    %bmm1 = transform.collect_matching @match_4d_matmul in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any

    // Start working out a full loop nest over the first batch matmul `bmm0`.
    // Inline elemwise ops (in this case, should be F16->F32 casts) into bmm0.
    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    // Also inline elementwise ops into bmm1.
    // `operand_number = 1` says only inline producers of the RHS of the matmul.
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    // Tile all parallel dimensions of bmm0 (b, h, i, j) into a scf.forall loop.
    // We'll fuse everything else into this loop nest.
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 128, 64, 0] : (!any) -> (!any, !any)
    // Canonicalization removes trivial (size-1) dimensions. This copies the loop and invalidates
    // all handles pointing to ops inside the loop body.
    // So we want to do this before we start fusion.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    // Fusion 1. Match element-wise ops that consumes mm0. Keep going until we reach a reduction
    // (`find_next_reduction` finds the nearest reduction).
    // In this case, the nearest reduction is the row-wise max of the softmax,
    // and we only have one elementwise op (the scale op) to fuse.
    //   TVM: sch.reverse_compute_at(bscale, j0)
    %bmax, %_2 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %bmax : (!any, !any) -> !any

    // Fusion 2. Fuse row-max into forall, splitting a serial `for` loop from forall in the process.
    // Because of how MLIR scf.for works, this fusion implicitly also r-factors the reduction.
    //   TVM: sch.reverse_compute_at(bmax, j0); sch.rfactor(...)
    // The "row-max" in the input program has two outputs: the max value and the argmax.
    // The subsequent fusion only supports single-output ops, so we remove the unused argmax
    // output before fusion.
    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.scf.fuse_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    // Fusion 3 (rolling update). First find the nearest reduction reachable from the loop's
    // output value, together with the ordered elementwise chain between them.
    // This could find either the row-sum of softmax, or the second matmul,
    // since they both depend on the loop's output. In this schedule
    %bmm1_2, %elemwise = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    // Clone and fuse that elementwise chain under %forall_loop and %j0_loop,
    // publishing the "sidecar" tensors as extra loop results.
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    // Repair the first reduction frontier by turning it into loop-carried state
    // driven by the relayed sidecar value.
    %_3 = transform.fusion.repair_reduction_frontier
        (%fused_bmax, %bmm1_2) and (%elemwise, %elemwise_sidecars) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    // Fusion 4. Apply rolling update again, this time with the second matmul being the reduction.
    %bsum, %elemwise_1 = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_4 = transform.fusion.repair_reduction_frontier
        (%fused_bmax, %bsum) and (%elemwise_1, %elemwise_sidecars_1) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    // Fusion 5. Fuse the trailing elemwise ops into the forall loop (but outside the for loop):
    // elemwise division, then FP32->FP16 cast.
    // CSE removes duplicate affine values and helps fusion
    // (fusion compares offset equal by comparing pointers to SSA value).
    transform.apply_cse to %func : !any
    %div = transform.get_consumers_of_result %forall_loop[1] : (!any) -> !any
    transform.fusion.into_producer %div into %forall_loop : (!any, !any) -> !any
    %trunc = transform.get_consumers_of_result %forall_loop[2] : (!any) -> !any
    transform.fusion.into_producer %trunc into %forall_loop : (!any, !any) -> !any

    // Post-pass: pushes lingering init tensor (see destination-passing style)
    // before and outside the loops into the loop body.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.scf.localize_scratch_tensors %func : !any
    // Remove unit-size dims from the loops and the linalg ops in the loop.
    // This is useful when we lower to HTile, because HTile requires (for example) dot to be in 2D.
    transform.apply_patterns to %func {
      transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !any
    transform.apply_cse to %func : !any

    // --- HTile lowering begins ---
    // Use the translator to lower the tiled linalg program into HTile.
    transform.htile.linalg_to_semantic %func : !any
    transform.htile.semantic_to_kernel_abi %func : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x128x128xf16>, %arg1: tensor<1x4x128x128xf16>, %arg2: tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.088388347648318433 : f64
    %0 = tensor.empty() : tensor<1x4x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x4x128x128xf16>) outs(%0 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %22 = arith.extf %in : f16 to f32
      linalg.yield %22 : f32
    } -> tensor<1x4x128x128xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x4x128x128xf16>) outs(%0 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %22 = arith.extf %in : f16 to f32
      linalg.yield %22 : f32
    } -> tensor<1x4x128x128xf32>
    %transposed = linalg.transpose ins(%2 : tensor<1x4x128x128xf32>) outs(%0 : tensor<1x4x128x128xf32>) permutation = [0, 1, 3, 2]
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x4x128x128xf32> into tensor<4x128x128xf32>
    %collapsed_2 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x4x128x128xf32> into tensor<4x128x128xf32>
    %3 = tensor.empty() : tensor<4x128x128xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %5 = linalg.batch_matmul ins(%collapsed, %collapsed_2 : tensor<4x128x128xf32>, tensor<4x128x128xf32>) outs(%4 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %expanded = tensor.expand_shape %5 [[0, 1], [2], [3]] output_shape [1, 4, 128, 128] : tensor<4x128x128xf32> into tensor<1x4x128x128xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x4x128x128xf32>) outs(%0 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = arith.truncf %cst_1 : f64 to f32
      %23 = arith.mulf %in, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<1x4x128x128xf32>
    %7 = tensor.empty() : tensor<1x4x128xi64>
    %8 = linalg.fill ins(%c0_i64 : i64) outs(%7 : tensor<1x4x128xi64>) -> tensor<1x4x128xi64>
    %9 = tensor.empty() : tensor<1x4x128xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<1x4x128xf32>) -> tensor<1x4x128xf32>
    %11:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%6 : tensor<1x4x128x128xf32>) outs(%10, %8 : tensor<1x4x128xf32>, tensor<1x4x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_7: i64):
      %22 = linalg.index 3 : index
      %23 = arith.index_cast %22 : index to i64
      %24 = arith.maximumf %in, %out : f32
      %25 = arith.cmpf ogt, %in, %out : f32
      %26 = arith.select %25, %23, %out_7 : i64
      linalg.yield %24, %26 : f32, i64
    } -> (tensor<1x4x128xf32>, tensor<1x4x128xi64>)
    %expanded_3 = tensor.expand_shape %11#0 [[0], [1], [2, 3]] output_shape [1, 4, 128, 1] : tensor<1x4x128xf32> into tensor<1x4x128x1xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %expanded_3 : tensor<1x4x128x128xf32>, tensor<1x4x128x1xf32>) outs(%0 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %22 = arith.subf %in, %in_7 : f32
      linalg.yield %22 : f32
    } -> tensor<1x4x128x128xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<1x4x128x128xf32>) outs(%0 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = math.exp %in : f32
      linalg.yield %22 : f32
    } -> tensor<1x4x128x128xf32>
    %14 = tensor.empty() : tensor<1x4x128x1xf32>
    %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<1x4x128x1xf32>) -> tensor<1x4x128x1xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%13 : tensor<1x4x128x128xf32>) outs(%15 : tensor<1x4x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = arith.addf %in, %out : f32
      linalg.yield %22 : f32
    } -> tensor<1x4x128x1xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %16 : tensor<1x4x128x128xf32>, tensor<1x4x128x1xf32>) outs(%0 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %22 = arith.divf %in, %in_7 : f32
      linalg.yield %22 : f32
    } -> tensor<1x4x128x128xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x4x128x128xf16>) outs(%0 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %22 = arith.extf %in : f16 to f32
      linalg.yield %22 : f32
    } -> tensor<1x4x128x128xf32>
    %collapsed_4 = tensor.collapse_shape %17 [[0, 1], [2], [3]] : tensor<1x4x128x128xf32> into tensor<4x128x128xf32>
    %collapsed_5 = tensor.collapse_shape %18 [[0, 1], [2], [3]] : tensor<1x4x128x128xf32> into tensor<4x128x128xf32>
    %19 = linalg.batch_matmul ins(%collapsed_4, %collapsed_5 : tensor<4x128x128xf32>, tensor<4x128x128xf32>) outs(%4 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %expanded_6 = tensor.expand_shape %19 [[0, 1], [2], [3]] output_shape [1, 4, 128, 128] : tensor<4x128x128xf32> into tensor<1x4x128x128xf32>
    %20 = tensor.empty() : tensor<1x4x128x128xf16>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<1x4x128x128xf32>) outs(%20 : tensor<1x4x128x128xf16>) {
    ^bb0(%in: f32, %out: f16):
      %22 = arith.truncf %in : f32 to f16
      linalg.yield %22 : f16
    } -> tensor<1x4x128x128xf16>
    return %21 : tensor<1x4x128x128xf16>
  }
}

// CHECK-LABEL: func.func @attention(%arg0: memref<1x4x128x128xf16>, %arg1: memref<1x4x128x128xf16>, %arg2: memref<1x4x128x128xf16>, %arg3: memref<1x4x128x128xf16>)
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: linalg.transpose
// CHECK-NOT: linalg.generic
// CHECK-NOT: tensor.empty() : tensor<4x128x128xf32>
// CHECK-NOT: tensor.empty() : tensor<4x128x128xf16>
// CHECK: scf.forall (%{{.*}}) in (4) {
// CHECK: htile.full %{{.*}} : f32 -> tensor<128xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<128x128xf32>
// CHECK: %{{.*}}:3 = scf.for %{{.*}} = %c0 to %c2 step %c1 iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>)
// CHECK: htile.load %arg0{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : memref<1x4x128x128xf16> -> tensor<128x128xf16>
// CHECK: htile.load %arg1{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : memref<1x4x128x128xf16> -> tensor<64x128xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} {transpose_b} : tensor<128x128xf16>, tensor<64x128xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "max" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: htile.broadcast %{{.*}} dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<128x64xf32>
// CHECK: htile.load %arg2{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : memref<1x4x128x128xf16> -> tensor<64x128xf16>
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64xf32>, tensor<64x128xf16>, tensor<128x128xf32> -> tensor<128x128xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<128x128xf32>
// CHECK: arith.truncf %{{.*}} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK: htile.store %{{.*}}, %arg3{{\[}}%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}{{\]}} : tensor<128x128xf16>, memref<1x4x128x128xf16>
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK: return
