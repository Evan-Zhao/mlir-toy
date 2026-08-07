// RUN: neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s
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

  transform.named_sequence @__transform_main(%module: !any) {
    // Prepass. Translate the input function from stablehlo to `ta` dialect, which enables
    // more flexible expression rewrites.
    %func0 = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %func = transform.apply_registered_pass "stablehlo-to-ta" to %func0 : (!any) -> !any
    transform.apply_patterns to %func {
      // This canonicalization step folds trunc(const(f64), f32) into a constant in f32.
      // This is only used in this test case, because only this test case has an f64 constant.
      transform.apply_patterns.canonicalization
      // Replace `exp(x)` with `exp2(x * log2(e))`, push `log2(e)` into scalar factors.
      transform.apply_patterns.ta.exp_to_exp2
      // Replace `matmul(P_ij / s_i, V_jd)` with `matmul(P_ij, V_jd) / s_i`.
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !any
    // Rewrites `max(QK^T * k) => max(QK^T) * k` (here `k` is a scalar `1/sqrt(d)`).
    // The latter is better because it combines with latter operations into a hardware FMA instruction.
    // Keep this separate from exp-to-exp2's opposite scale-motion patterns.
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !any
    transform.apply_cse to %func : !any
    // This is a special idiom: use einsum to match matmuls in `ta` dialect is easy.
    // Then the `to_linalg` translator keeps these handles alive even after the translation,
    // so you get %bmm0 to point to the first matmul in linalg.
    %bmm0 = transform.collect_matching @match_4d_matmul_transb in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any

    // Start working out a full loop nest over the first batch matmul `bmm0`.
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
    %prefix = transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %bmax { inline_elementwise } : (!any, !any) -> !any
    transform.linalg.erase_unused_operands_and_results %prefix : !any

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
    // This could find either the row-sum of softmax, or the second matmul, since they both
    // depend on the loop's output. Currently in this schedule it finds the row-sum first,
    // but just keep in mind that this can change (and doesn't matter).
    %bsum, %elemwise = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    // Clone and fuse that elementwise chain under %forall_loop and %j0_loop,
    // publishing the "sidecar" tensors as extra loop results.
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    // Repair the first reduction frontier by turning it into loop-carried state
    // driven by the relayed sidecar value.
    %_3 = transform.fusion.repair_reduction_frontier
        %bsum reduce_producer %fused_bmax
        substituting elemwise %elemwise -> %elemwise_sidecars
        into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    // Fusion 4. Apply rolling update again, this time with the second matmul being the reduction.
    %bmm1, %elemwise_1 = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_4 = transform.fusion.repair_reduction_frontier
        %bmm1 reduce_producer %fused_bmax
        substituting elemwise %elemwise_1 -> %elemwise_sidecars_1
        into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    // Fusion 5. Fuse the trailing elemwise ops into the forall loop (but outside the for loop):
    // elemwise division, then FP32->FP16 cast. Keep going until we see the return op.
    %ret = transform.structured.match ops{["func.return"]} in %func : (!any) -> !any
    transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %ret : (!any, !any) -> !any

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
    %launches, %kernels = transform.htile.outline_kernels %forall_loop
        {kernel_names = ["attention_kernel"]} : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.verify %func : !any
    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x128x128xf16>, %arg1: tensor<1x4x128x128xf16>, %arg2: tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<0.088388347648318433> : tensor<1xf64>
    %0 = stablehlo.transpose %arg1, dims = [0, 1, 3, 2] : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x128x128xf16>, tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf32>
    %3 = stablehlo.convert %cst_1 : (tensor<1xf64>) -> tensor<1xf32>
    %4 = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %6 = stablehlo.multiply %2, %5 : tensor<1x4x128x128xf32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %8 = stablehlo.reshape %7 : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<1x4x128x128xf32>
    %11 = stablehlo.exponential %10 : tensor<1x4x128x128xf32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %13 = stablehlo.reshape %12 : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x4x128x128xf32>
    %16 = stablehlo.convert %15 : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf16>
    %17 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 3] : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %18 = stablehlo.dot_general %16, %17, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x128x128xf16>, tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf32>
    %19 = stablehlo.convert %18 : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf16>
    return %19 : tensor<1x4x128x128xf16>
  }
}

// CHECK-LABEL: func.func @attention(%arg0: tensor<1x4x128x128xf16>, %arg1: tensor<1x4x128x128xf16>, %arg2: tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: linalg.transpose
// CHECK-NOT: linalg.generic
// CHECK: %[[OUT:.*]] = memref.alloc() : memref<1x4x128x128xf16>
// CHECK: %[[Q:.*]] = bufferization.to_buffer %arg0 read_only : tensor<1x4x128x128xf16> to memref<1x4x128x128xf16>
// CHECK: %[[K:.*]] = bufferization.to_buffer %arg1 read_only : tensor<1x4x128x128xf16> to memref<1x4x128x128xf16>
// CHECK: %[[V:.*]] = bufferization.to_buffer %arg2 read_only : tensor<1x4x128x128xf16> to memref<1x4x128x128xf16>
// CHECK: htile.launch_func @attention_kernel(%[[Q]], %[[K]], %[[V]], %[[OUT]]) {program_bounds = array<i64: 4>} : memref<1x4x128x128xf16>, memref<1x4x128x128xf16>, memref<1x4x128x128xf16>, memref<1x4x128x128xf16>
// CHECK-NOT: scf.forall
// CHECK: bufferization.to_tensor %[[OUT]]
// CHECK: return

// CHECK-LABEL: htile.kernel @attention_kernel
// CHECK-SAME: memref<1x4x128x128xf16>
// CHECK-SAME: memref<1x4x128x128xf16>
// CHECK-SAME: memref<1x4x128x128xf16>
// CHECK-SAME: memref<1x4x128x128xf16>
// CHECK-SAME: attributes {program_bounds = array<i64: 4>}
// CHECK: htile.program_id 0
// CHECK-NOT: scf.forall
// CHECK: htile.full %{{.*}} : f32 -> tensor<128xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<128x128xf32>
// CHECK: %{{.*}}:5 = scf.for %{{.*}} = %c0 to %c2 step %c1
// CHECK-SAME: -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
// CHECK-SAME: tensor<128xf32>, tensor<128x128xf32>)
// CHECK: htile.load %arg0
// CHECK: htile.load %arg1
// CHECK: %[[RAW_SCORES:.+]] = htile.dot %{{.*}}, %{{.*}} {transpose_b} : tensor<128x128xf16>, tensor<64x128xf16> -> tensor<128x64xf32>
// CHECK: htile.reduce %[[RAW_SCORES]] axis 1 kind "max" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: arith.mulf %[[RAW_SCORES]], %{{.*}} : tensor<128x64xf32>
// CHECK: htile.broadcast %{{.*}} dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<128x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: htile.load %arg2
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64xf16>, tensor<64x128xf16>, tensor<128x128xf32> -> tensor<128x128xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<128x128xf32>
// CHECK: arith.truncf %{{.*}} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK: htile.store %{{.*}}, %arg3
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK: htile.return
