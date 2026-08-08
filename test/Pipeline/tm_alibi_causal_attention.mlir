// RUN: neptune-opt --transform-interpreter %s 2>&1 | FileCheck %s
//
// Transform-dialect schedule for Torch-MLIR ALiBi causal attention.
// This intentionally starts from the masked-attention schedule because ALiBi
// is a score-side bias before the causal mask and online softmax.

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
      transform.apply_patterns.canonicalization
      transform.apply_patterns.ta.exp_to_exp2
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !any
    // Preserve log2(e) around the complete biased logit, then move it after max.
    // Keep this separate from exp-to-exp2's opposite scale-motion patterns.
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !any
    transform.apply_cse to %func : !any
    %bmm0 = transform.collect_matching @match_4d_matmul_transb in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any

    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 64, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bmax, %_2 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %prefix = transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %bmax { inline_elementwise } : (!any, !any) -> !any
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

  func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf16>, %arg2: tensor<1x4x1024x64xf16>, %arg3: tensor<4xf32>) -> tensor<1x4x1024x64xf16> {
    %c = stablehlo.constant dense<0> : tensor<1024x1024xi64>
    %c_0 = stablehlo.constant dense<0> : tensor<1024xi64>
    %c_1 = stablehlo.constant dense<1> : tensor<1024xi64>
    %c_2 = stablehlo.constant dense<1024> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %c_4 = stablehlo.constant dense<true> : tensor<1024x1024xi1>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_5 = stablehlo.constant dense<false> : tensor<1024x1024xi1>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_7 = arith.constant dense<1.250000e-01> : tensor<1xf64>
    %0 = stablehlo.transpose %arg1, dims = [0, 1, 3, 2] : (tensor<1x4x1024x64xf16>) -> tensor<1x4x64x1024xf16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x4x64x1024xf16>) -> tensor<1x4x64x1024xf16>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x1024x64xf16>, tensor<1x4x64x1024xf16>) -> tensor<1x4x1024x1024xf32>
    %3 = stablehlo.convert %cst_7 : (tensor<1xf64>) -> tensor<1xf32>
    %4 = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1x4x1024x1024xf32>
    %6 = stablehlo.multiply %2, %5 : tensor<1x4x1024x1024xf32>
    %7 = stablehlo.convert %c_2 : (tensor<i64>) -> tensor<f64>
    %8 = stablehlo.convert %c_3 : (tensor<i64>) -> tensor<f64>
    %9 = stablehlo.divide %7, %8 : tensor<f64>
    %10 = stablehlo.ceil %9 : tensor<f64>
    %11 = stablehlo.convert %10 : (tensor<f64>) -> tensor<i64>
    %12 = stablehlo.reshape %11 : (tensor<i64>) -> tensor<1xi64>
    %13 = stablehlo.dynamic_iota %12, dim = 0 : (tensor<1xi64>) -> tensor<1024xi64>
    %14 = stablehlo.multiply %13, %c_1 : tensor<1024xi64>
    %15 = stablehlo.add %14, %c_0 : tensor<1024xi64>
    %16 = stablehlo.convert %15 : (tensor<1024xi64>) -> tensor<1024xf32>
    %17 = stablehlo.reshape %16 : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %18 = stablehlo.reshape %16 : (tensor<1024xf32>) -> tensor<1024x1xf32>
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<1024x1024xf32>
    %20 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1024x1xf32>) -> tensor<1024x1024xf32>
    %21 = stablehlo.subtract %19, %20 : tensor<1024x1024xf32>
    %22 = stablehlo.reshape %arg3 : (tensor<4xf32>) -> tensor<4x1xf32>
    %23 = stablehlo.reshape %22 : (tensor<4x1xf32>) -> tensor<4x1x1xf32>
    %24 = stablehlo.broadcast_in_dim %21, dims = [1, 2] : (tensor<1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %25 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2] : (tensor<4x1x1xf32>) -> tensor<4x1024x1024xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<4x1024x1024xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [1, 2, 3] : (tensor<4x1024x1024xf32>) -> tensor<1x4x1024x1024xf32>
    %28 = stablehlo.add %6, %27 : tensor<1x4x1024x1024xf32>
    %29 = stablehlo.iota dim = 1 : tensor<1024x1024xi64>
    %30 = stablehlo.iota dim = 0 : tensor<1024x1024xi64>
    %31 = stablehlo.add %30, %c : tensor<1024x1024xi64>
    %32 = stablehlo.compare LE, %29, %31, SIGNED : (tensor<1024x1024xi64>, tensor<1024x1024xi64>) -> tensor<1024x1024xi1>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1] : (tensor<1024x1024xi1>) -> tensor<1024x1024xi1>
    %34 = stablehlo.select %33, %c_4, %c_5 : tensor<1024x1024xi1>, tensor<1024x1024xi1>
    %35 = stablehlo.reshape %34 : (tensor<1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x1x1024x1024xi1>) -> tensor<1x4x1024x1024xi1>
    %37 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x4x1024x1024xf32>) -> tensor<1x4x1024x1024xf32>
    %38 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x1024x1024xf32>
    %39 = stablehlo.select %36, %37, %38 : tensor<1x4x1024x1024xi1>, tensor<1x4x1024x1024xf32>
    %40 = stablehlo.reduce(%39 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x1024x1024xf32>, tensor<f32>) -> tensor<1x4x1024xf32>
    %41 = stablehlo.reshape %40 : (tensor<1x4x1024xf32>) -> tensor<1x4x1024x1xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2, 3] : (tensor<1x4x1024x1xf32>) -> tensor<1x4x1024x1024xf32>
    %43 = stablehlo.subtract %39, %42 : tensor<1x4x1024x1024xf32>
    %44 = stablehlo.exponential %43 : tensor<1x4x1024x1024xf32>
    %45 = stablehlo.reduce(%44 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x4x1024x1024xf32>, tensor<f32>) -> tensor<1x4x1024xf32>
    %46 = stablehlo.reshape %45 : (tensor<1x4x1024xf32>) -> tensor<1x4x1024x1xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2, 3] : (tensor<1x4x1024x1xf32>) -> tensor<1x4x1024x1024xf32>
    %48 = stablehlo.divide %44, %47 : tensor<1x4x1024x1024xf32>
    %49 = stablehlo.convert %48 : (tensor<1x4x1024x1024xf32>) -> tensor<1x4x1024x1024xf16>
    %50 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 3] : (tensor<1x4x1024x64xf16>) -> tensor<1x4x1024x64xf16>
    %51 = stablehlo.dot_general %49, %50, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x1024x1024xf16>, tensor<1x4x1024x64xf16>) -> tensor<1x4x1024x64xf32>
    %52 = stablehlo.convert %51 : (tensor<1x4x1024x64xf32>) -> tensor<1x4x1024x64xf16>
    return %52 : tensor<1x4x1024x64xf16>
  }
}

// CHECK-NOT: dead-tile propagation could not prove
// CHECK-LABEL: func.func @attention(
// CHECK-SAME: %arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf16>, %arg2: tensor<1x4x1024x64xf16>, %arg3: tensor<4xf32>) -> tensor<1x4x1024x64xf16>
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf32>
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf16>
// CHECK-DAG: %[[OUT:.+]] = memref.alloc() : memref<1x4x1024x64xf16>
// CHECK-DAG: %[[Q:.+]] = bufferization.to_buffer %arg0 read_only
// CHECK-DAG: %[[K:.+]] = bufferization.to_buffer %arg1 read_only
// CHECK-DAG: %[[V:.+]] = bufferization.to_buffer %arg2 read_only
// CHECK-DAG: %[[ALIBI:.+]] = bufferization.to_buffer %arg3 read_only
// CHECK: htile.launch_func @attention_kernel(%[[Q]], %[[K]], %[[V]], %[[ALIBI]], %[[OUT]])
// CHECK-SAME: {program_bounds = array<i64: 4, 16>}
// CHECK-NOT: scf.forall
// CHECK: return

// CHECK-LABEL: htile.kernel @attention_kernel
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<4xf32>
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: attributes {program_bounds = array<i64: 4, 16>}
// CHECK: htile.program_id 0
// CHECK: htile.program_id 1
// CHECK-NOT: scf.forall
// CHECK: htile.full %{{.*}} : f32 -> tensor<64xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<64x64xf32>
// CHECK: %[[RAW_BOUND:.+]] = arith.maxsi %{{.*}}, %c0 : index
// CHECK: %[[CAPPED_BOUND:.+]] = arith.minsi %[[RAW_BOUND]], %c16 : index
// CHECK: %[[LIVE:.+]]:5 = scf.for %[[J_TILE:.*]] = %c0 to %[[CAPPED_BOUND]] step %c1 iter_args(
// CHECK: htile.load %arg0
// CHECK: htile.load %arg1
// CHECK: htile.dot %{{.*}}, %{{.*}} {transpose_b} : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf32>
// CHECK: htile.load %arg3
// CHECK: arith.muli %{{.*}}, %{{.*}} {{.*}} : index
// CHECK: arith.muli %{{.*}}, %{{.*}} {{.*}} : index
// CHECK: htile.broadcast %{{.*}} dimensions = [0, 1] : tensor<f32> -> tensor<64x64xf32>
// CHECK: htile.arange %c0 to %c64 : tensor<64xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [1] : tensor<64xindex> -> tensor<64x64xindex>
// CHECK: htile.full %{{.*}} : index -> tensor<64x64xindex>
// CHECK-NOT: arith.muli %{{.*}}, %{{.*}} {{.*}} : tensor<64x64xindex>
// CHECK: arith.addi %{{.*}}, %{{.*}} : tensor<64x64xindex>
// CHECK: htile.arange %c0 to %c64 : tensor<64xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [0] : tensor<64xindex> -> tensor<64x64xindex>
// CHECK: htile.full %{{.*}} : index -> tensor<64x64xindex>
// CHECK: arith.addi %{{.*}}, %{{.*}} : tensor<64x64xindex>
// CHECK: arith.subf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: %[[LIVE_SCORES:.+]] = arith.addf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK-NOT: arith.select
// CHECK: htile.reduce %[[LIVE_SCORES]] axis 1 kind "max" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: arith.mulf %[[LIVE_SCORES]], %{{.*}} : tensor<64x64xf32>
// CHECK: math.exp2
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: htile.load %arg2
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: %[[DEAD_BOUND_RAW:.+]] = affine.apply
// CHECK: %[[DEAD_BOUND_MAX:.+]] = arith.maxsi %[[DEAD_BOUND_RAW]], %c0 : index
// CHECK: %[[DEAD_BOUND:.+]] = arith.minsi %[[DEAD_BOUND_MAX]], %c16 : index
// CHECK: %[[MIXED:.+]]:5 = scf.for %{{.*}} = %[[CAPPED_BOUND]] to %[[DEAD_BOUND]] step %c1
// CHECK-SAME: iter_args(%{{.*}} = %[[LIVE]]#0, %{{.*}} = %[[LIVE]]#1,
// CHECK-SAME: %{{.*}} = %[[LIVE]]#2, %{{.*}} = %[[LIVE]]#3, %{{.*}} = %[[LIVE]]#4)
// CHECK: htile.load %arg0
// CHECK: htile.load %arg1
// CHECK: htile.dot %{{.*}}, %{{.*}} {transpose_b} : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf32>
// CHECK: htile.load %arg3
// CHECK: htile.broadcast %{{.*}} dimensions = [0, 1] : tensor<f32> -> tensor<64x64xf32>
// CHECK: arith.subf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: %[[MIXED_SCORES:.+]] = arith.addf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.cmpi sle, %{{.*}}, %{{.*}} : tensor<64x64xi64>
// CHECK: %[[MASKED_RAW:.+]] = arith.select %{{.*}}, %[[MIXED_SCORES]], %{{.*}} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: htile.reduce %[[MASKED_RAW]] axis 1 kind "max" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: arith.mulf %[[MASKED_RAW]], %{{.*}} : tensor<64x64xf32>
// CHECK: math.exp2
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: htile.load %arg2
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK: htile.store %{{.*}}, %arg4
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK-NOT: ta.
// CHECK: htile.return
