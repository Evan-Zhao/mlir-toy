// RUN: neptune-opt --transform-interpreter %s 2>&1 | FileCheck %s

// CHECK-NOT: dead-tile propagation could not prove
// CHECK-LABEL: func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf8E4M3FN>, %arg2: tensor<1x4x1024x64xf8E4M3FN>, %arg3: tensor<1x4x1x1xf32>, %arg4: tensor<1x4x1x1xf32>) -> tensor<1x4x1024x64xf16>
// CHECK-DAG: %[[Q:.+]] = bufferization.to_buffer %arg0 read_only
// CHECK-DAG: %[[K:.+]] = bufferization.to_buffer %arg1 read_only
// CHECK-DAG: %[[V:.+]] = bufferization.to_buffer %arg2 read_only
// CHECK-DAG: %[[K_SCALE:.+]] = bufferization.to_buffer %arg3 read_only
// CHECK-DAG: %[[V_SCALE:.+]] = bufferization.to_buffer %arg4 read_only
// CHECK: htile.launch_func @attention_kernel(%[[Q]], %[[K]], %[[V]], %[[K_SCALE]], %[[V_SCALE]], %{{.+}})
// CHECK-SAME: {program_bounds = array<i64: 4, 8>}
// CHECK-NOT: scf.forall
// CHECK: return

// CHECK-LABEL: htile.kernel @attention_kernel
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<1x4x1024x64xf8E4M3FN>
// CHECK-SAME: memref<1x4x1024x64xf8E4M3FN>
// CHECK-SAME: memref<1x4x1x1xf32>
// CHECK-SAME: memref<1x4x1x1xf32>
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: attributes {program_bounds = array<i64: 4, 8>}
// CHECK: htile.program_id 0
// CHECK: htile.program_id 1
// CHECK-NOT: scf.forall
// CHECK: htile.full %{{.*}} : f32 -> tensor<128xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<128xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<128x64xf32>
// CHECK: %[[LIVE_BOUND_RAW:.+]] = affine.apply
// CHECK: %[[LIVE_BOUND_MAX:.+]] = arith.maxsi %[[LIVE_BOUND_RAW]], %c0 : index
// CHECK: %[[LIVE_BOUND:.+]] = arith.minsi %[[LIVE_BOUND_MAX]], %c16 : index
// CHECK: %[[LIVE_LOOP:.+]]:5 = scf.for %{{.*}} = %c0 to %[[LIVE_BOUND]] step %c1 iter_args(
// CHECK: scf.yield
// CHECK: %[[DEAD_BOUND_RAW:.+]] = affine.apply
// CHECK: %[[DEAD_BOUND_MAX:.+]] = arith.maxsi %[[DEAD_BOUND_RAW]], %c0 : index
// CHECK: %[[DEAD_BOUND:.+]] = arith.minsi %[[DEAD_BOUND_MAX]], %c16 : index
// CHECK: %{{.*}}:5 = scf.for %{{.*}} = %[[LIVE_BOUND]] to %[[DEAD_BOUND]] step %c1
// CHECK-SAME: iter_args(%{{.*}} = %[[LIVE_LOOP]]#0, %{{.*}} = %[[LIVE_LOOP]]#1,
// CHECK-SAME: %{{.*}} = %[[LIVE_LOOP]]#2, %{{.*}} = %[[LIVE_LOOP]]#3,
// CHECK-SAME: %{{.*}} = %[[LIVE_LOOP]]#4)
// CHECK: %[[MASKED_RAW:.+]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64xi1>, tensor<128x64xf32>
// CHECK: htile.reduce %[[MASKED_RAW]] axis 1 kind "max" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: arith.mulf %[[MASKED_RAW]], %{{.*}} : tensor<128x64xf32>
// CHECK: htile.store
// CHECK: htile.return

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_4d_matmul_transb(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b h i d, b h j d -> b h i j"} : (!any) -> !any
    transform.yield %matched : !any
  }

  transform.named_sequence @__transform_main(%module: !any) {
    %func0 = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %func = transform.apply_registered_pass "stablehlo-to-ta" to %func0 : (!any) -> !any
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.exp_to_exp2
      transform.apply_patterns.ta.sink_div_after_matmul
      transform.apply_patterns.ta.sink_right_mul_after_matmul
    } : !any
    // Factor a positive scale out of masked scores, then move it after the max reduction.
    // Keep this separate from exp-to-exp2's opposite scale-motion patterns.
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !any
    transform.apply_cse to %func : !any
    %bmm0 = transform.collect_matching @match_4d_matmul_transb in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any

    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 128, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bmax, %_2 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %prefix = transform.fusion.greedy_consumers_into_producer %forall_loop until %bmax { inline_elementwise } : (!any, !any) -> !any
    transform.linalg.erase_unused_operands_and_results %prefix : !any

    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.scf.fuse_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    %bsum, %elemwise = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_3 = transform.fusion.repair_reduction_frontier
        %bsum reduce_producer %fused_bmax
        substituting elemwise %elemwise -> %elemwise_sidecars
        into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

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

    %ret = transform.structured.match ops{["func.return"]} in %func : (!any) -> !any
    transform.fusion.greedy_consumers_into_producer %forall_loop until %ret : (!any, !any) -> !any

    // Fuse KV scale multiplication downwards into the loops.
    %consumer_loops = transform.merge_handles %forall_loop, %j0_loop : !any
    %_5 = transform.fusion.greedy_input_producers_into_consumer %consumer_loops : (!any) -> !any

    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !any
    transform.apply_cse to %func : !any

    // 0xFF800000: -inf in f32
    %live_loop, %mixed_loop, %unchanged_loop =
        transform.loop.specialize_dead_tile in %j0_loop
        {dead_value = 0xFF800000 : f32} : (!any) -> (!any, !any, !any)

    // --- HTile lowering begins ---
    transform.htile.linalg_to_semantic %func : !any
    %launches, %kernels = transform.htile.outline_kernels %forall_loop
        {kernel_names = ["attention_kernel"]} : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.verify %func : !any
    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf8E4M3FN>, %arg2: tensor<1x4x1024x64xf8E4M3FN>, %arg3: tensor<1x4x1x1xf32>, %arg4: tensor<1x4x1x1xf32>) -> tensor<1x4x1024x64xf16> {
    %c = stablehlo.constant dense<0> : tensor<1024x1024xi64>
    %c_0 = stablehlo.constant dense<true> : tensor<1024x1024xi1>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_1 = stablehlo.constant dense<false> : tensor<1024x1024xi1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<1.250000e-01> : tensor<1xf64>
    %0 = stablehlo.convert %arg1 : (tensor<1x4x1024x64xf8E4M3FN>) -> tensor<1x4x1024x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1, 2, 3] : (tensor<1x4x1x1xf32>) -> tensor<1x4x1024x64xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x4x1024x64xf32>
    %3 = stablehlo.convert %2 : (tensor<1x4x1024x64xf32>) -> tensor<1x4x1024x64xf16>
    %4 = stablehlo.convert %arg2 : (tensor<1x4x1024x64xf8E4M3FN>) -> tensor<1x4x1024x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg4, dims = [0, 1, 2, 3] : (tensor<1x4x1x1xf32>) -> tensor<1x4x1024x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1x4x1024x64xf32>
    %7 = stablehlo.convert %6 : (tensor<1x4x1024x64xf32>) -> tensor<1x4x1024x64xf16>
    %8 = stablehlo.transpose %3, dims = [0, 1, 3, 2] : (tensor<1x4x1024x64xf16>) -> tensor<1x4x64x1024xf16>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3] : (tensor<1x4x64x1024xf16>) -> tensor<1x4x64x1024xf16>
    %10 = stablehlo.dot_general %arg0, %9, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x1024x64xf16>, tensor<1x4x64x1024xf16>) -> tensor<1x4x1024x1024xf32>
    %11 = stablehlo.convert %cst_3 : (tensor<1xf64>) -> tensor<1xf32>
    %12 = stablehlo.reshape %11 : (tensor<1xf32>) -> tensor<f32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<1x4x1024x1024xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<1x4x1024x1024xf32>
    %15 = stablehlo.iota dim = 1 : tensor<1024x1024xi64>
    %16 = stablehlo.iota dim = 0 : tensor<1024x1024xi64>
    %17 = stablehlo.add %16, %c : tensor<1024x1024xi64>
    %18 = stablehlo.compare LE, %15, %17, SIGNED : (tensor<1024x1024xi64>, tensor<1024x1024xi64>) -> tensor<1024x1024xi1>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1024x1024xi1>) -> tensor<1024x1024xi1>
    %20 = stablehlo.select %19, %c_0, %c_1 : tensor<1024x1024xi1>, tensor<1024x1024xi1>
    %21 = stablehlo.reshape %20 : (tensor<1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2, 3] : (tensor<1x1x1024x1024xi1>) -> tensor<1x4x1024x1024xi1>
    %23 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3] : (tensor<1x4x1024x1024xf32>) -> tensor<1x4x1024x1024xf32>
    %24 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x1024x1024xf32>
    %25 = stablehlo.select %22, %23, %24 : tensor<1x4x1024x1024xi1>, tensor<1x4x1024x1024xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x1024x1024xf32>, tensor<f32>) -> tensor<1x4x1024xf32>
    %27 = stablehlo.reshape %26 : (tensor<1x4x1024xf32>) -> tensor<1x4x1024x1xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 3] : (tensor<1x4x1024x1xf32>) -> tensor<1x4x1024x1024xf32>
    %29 = stablehlo.subtract %25, %28 : tensor<1x4x1024x1024xf32>
    %30 = stablehlo.exponential %29 : tensor<1x4x1024x1024xf32>
    %31 = stablehlo.reduce(%30 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<1x4x1024x1024xf32>, tensor<f32>) -> tensor<1x4x1024xf32>
    %32 = stablehlo.reshape %31 : (tensor<1x4x1024xf32>) -> tensor<1x4x1024x1xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2, 3] : (tensor<1x4x1024x1xf32>) -> tensor<1x4x1024x1024xf32>
    %34 = stablehlo.divide %30, %33 : tensor<1x4x1024x1024xf32>
    %35 = stablehlo.convert %34 : (tensor<1x4x1024x1024xf32>) -> tensor<1x4x1024x1024xf16>
    %36 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x4x1024x64xf16>) -> tensor<1x4x1024x64xf16>
    %37 = stablehlo.dot_general %35, %36, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x1024x1024xf16>, tensor<1x4x1024x64xf16>) -> tensor<1x4x1024x64xf32>
    %38 = stablehlo.convert %37 : (tensor<1x4x1024x64xf32>) -> tensor<1x4x1024x64xf16>
    return %38 : tensor<1x4x1024x64xf16>
  }
}
