// RUN: neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s
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

  transform.named_sequence @__transform_main(%module: !any) {
    %func0 = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %func = transform.apply_registered_pass "stablehlo-to-ta" to %func0 : (!any) -> !any
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.ta.exp_to_exp2
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !any
    // Factor a positive scale out of masked scores, then move it after the max reduction.
    // Keep this separate from exp-to-exp2's opposite scale-motion patterns.
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !any
    transform.apply_cse to %func : !any
    %bmm0 = transform.collect_matching @match_4d_matmul_transb in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 64, 64, 0] : (!any) -> (!any, !any)
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

  func.func @attention(%arg0: tensor<2x4x1024x64xf16>, %arg1: tensor<2x4x1024x64xf16>, %arg2: tensor<2x4x1024x64xf16>) -> tensor<2x4x1024x64xf16> {
    %c = stablehlo.constant dense<-127> : tensor<1024x1024xi64>
    %c_0 = stablehlo.constant dense<0> : tensor<1024xi64>
    %c_1 = stablehlo.constant dense<1> : tensor<1024xi64>
    %c_2 = stablehlo.constant dense<1024> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %c_4 = stablehlo.constant dense<0> : tensor<1024x1024xi64>
    %c_5 = stablehlo.constant dense<true> : tensor<1024x1024xi1>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_6 = stablehlo.constant dense<false> : tensor<1024x1024xi1>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_8 = arith.constant dense<1.250000e-01> : tensor<1xf64>
    %0 = stablehlo.transpose %arg1, dims = [0, 1, 3, 2] : (tensor<2x4x1024x64xf16>) -> tensor<2x4x64x1024xf16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<2x4x64x1024xf16>) -> tensor<2x4x64x1024xf16>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x4x1024x64xf16>, tensor<2x4x64x1024xf16>) -> tensor<2x4x1024x1024xf32>
    %3 = stablehlo.convert %cst_8 : (tensor<1xf64>) -> tensor<1xf32>
    %4 = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<2x4x1024x1024xf32>
    %6 = stablehlo.multiply %2, %5 : tensor<2x4x1024x1024xf32>
    %7 = stablehlo.iota dim = 1 : tensor<1024x1024xi64>
    %8 = stablehlo.iota dim = 0 : tensor<1024x1024xi64>
    %9 = stablehlo.add %8, %c_4 : tensor<1024x1024xi64>
    %10 = stablehlo.compare LE, %7, %9, SIGNED : (tensor<1024x1024xi64>, tensor<1024x1024xi64>) -> tensor<1024x1024xi1>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x1024xi1>) -> tensor<1024x1024xi1>
    %12 = stablehlo.select %11, %c_5, %c_6 : tensor<1024x1024xi1>, tensor<1024x1024xi1>
    %13 = stablehlo.convert %c_2 : (tensor<i64>) -> tensor<f64>
    %14 = stablehlo.convert %c_3 : (tensor<i64>) -> tensor<f64>
    %15 = stablehlo.divide %13, %14 : tensor<f64>
    %16 = stablehlo.ceil %15 : tensor<f64>
    %17 = stablehlo.convert %16 : (tensor<f64>) -> tensor<i64>
    %18 = stablehlo.reshape %17 : (tensor<i64>) -> tensor<1xi64>
    %19 = stablehlo.dynamic_iota %18, dim = 0 : (tensor<1xi64>) -> tensor<1024xi64>
    %20 = stablehlo.multiply %19, %c_1 : tensor<1024xi64>
    %21 = stablehlo.add %20, %c_0 : tensor<1024xi64>
    %22 = stablehlo.reshape %21 : (tensor<1024xi64>) -> tensor<1x1024xi64>
    %23 = stablehlo.reshape %21 : (tensor<1024xi64>) -> tensor<1024x1xi64>
    %24 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<1x1024xi64>) -> tensor<1024x1024xi64>
    %25 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<1024x1xi64>) -> tensor<1024x1024xi64>
    %26 = stablehlo.subtract %24, %25 : tensor<1024x1024xi64>
    %27 = stablehlo.compare GE, %26, %c, SIGNED : (tensor<1024x1024xi64>, tensor<1024x1024xi64>) -> tensor<1024x1024xi1>
    %28 = stablehlo.and %27, %12 : tensor<1024x1024xi1>
    %29 = stablehlo.reshape %28 : (tensor<1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2, 3] : (tensor<1x1x1024x1024xi1>) -> tensor<2x4x1024x1024xi1>
    %31 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3] : (tensor<2x4x1024x1024xf32>) -> tensor<2x4x1024x1024xf32>
    %32 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x1024x1024xf32>
    %33 = stablehlo.select %30, %31, %32 : tensor<2x4x1024x1024xi1>, tensor<2x4x1024x1024xf32>
    %34 = stablehlo.reduce(%33 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<2x4x1024x1024xf32>, tensor<f32>) -> tensor<2x4x1024xf32>
    %35 = stablehlo.reshape %34 : (tensor<2x4x1024xf32>) -> tensor<2x4x1024x1xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<2x4x1024x1xf32>) -> tensor<2x4x1024x1024xf32>
    %37 = stablehlo.subtract %33, %36 : tensor<2x4x1024x1024xf32>
    %38 = stablehlo.exponential %37 : tensor<2x4x1024x1024xf32>
    %39 = stablehlo.reduce(%38 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<2x4x1024x1024xf32>, tensor<f32>) -> tensor<2x4x1024xf32>
    %40 = stablehlo.reshape %39 : (tensor<2x4x1024xf32>) -> tensor<2x4x1024x1xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2, 3] : (tensor<2x4x1024x1xf32>) -> tensor<2x4x1024x1024xf32>
    %42 = stablehlo.divide %38, %41 : tensor<2x4x1024x1024xf32>
    %43 = stablehlo.convert %42 : (tensor<2x4x1024x1024xf32>) -> tensor<2x4x1024x1024xf16>
    %44 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 3] : (tensor<2x4x1024x64xf16>) -> tensor<2x4x1024x64xf16>
    %45 = stablehlo.dot_general %43, %44, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x4x1024x1024xf16>, tensor<2x4x1024x64xf16>) -> tensor<2x4x1024x64xf32>
    %46 = stablehlo.convert %45 : (tensor<2x4x1024x64xf32>) -> tensor<2x4x1024x64xf16>
    return %46 : tensor<2x4x1024x64xf16>
  }
}

// CHECK-NOT: dead-tile propagation could not prove
// CHECK-LABEL: func.func @attention(
// CHECK-SAME: %arg0: tensor<2x4x1024x64xf16>, %arg1: tensor<2x4x1024x64xf16>, %arg2: tensor<2x4x1024x64xf16>) -> tensor<2x4x1024x64xf16>
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf32>
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf16>
// CHECK: htile.launch_func @attention_kernel
// CHECK-SAME: {program_bounds = array<i64: 2, 4, 16>}
// CHECK-NOT: scf.forall
// CHECK: return

// CHECK-LABEL: htile.kernel @attention_kernel
// CHECK-SAME: memref<2x4x1024x64xf16>
// CHECK-SAME: memref<2x4x1024x64xf16>
// CHECK-SAME: memref<2x4x1024x64xf16>
// CHECK-SAME: memref<2x4x1024x64xf16>
// CHECK-SAME: attributes {program_bounds = array<i64: 2, 4, 16>}
// CHECK: htile.program_id 0
// CHECK: htile.program_id 1
// CHECK: htile.program_id 2
// CHECK-NOT: scf.forall
// CHECK: htile.full %{{.*}} : f32 -> tensor<64xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<64x64xf32>
// CHECK: %[[LIVE_LOWER_RAW:.*]] = affine.apply
// CHECK: %[[LIVE_LOWER_CLAMP_LO:.*]] = arith.maxsi %[[LIVE_LOWER_RAW]], %c0 : index
// CHECK: %[[LIVE_LOWER:.*]] = arith.minsi %[[LIVE_LOWER_CLAMP_LO]], %c16 : index
// CHECK: %[[LIVE_UPPER_CLAMP_LO:.*]] = arith.maxsi %{{.*}}, %c0 : index
// CHECK: %[[LIVE_UPPER:.*]] = arith.minsi %[[LIVE_UPPER_CLAMP_LO]], %c16 : index
// CHECK: %[[LIVE_LOOP:.*]]:5 = scf.for %{{.*}} = %[[LIVE_LOWER]] to %[[LIVE_UPPER]] step %c1
// CHECK: htile.load %arg0
// CHECK: htile.load %arg1
// CHECK: %[[LIVE_RAW:.+]] = htile.dot %{{.*}}, %{{.*}} {transpose_b} : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf32>
// CHECK: htile.reduce %[[LIVE_RAW]] axis 1 kind "max" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: arith.mulf %[[LIVE_RAW]], %{{.*}} : tensor<64x64xf32>
// CHECK: math.exp2
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: htile.load %arg2
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: scf.yield
// CHECK: %[[DEAD_BOUND_RAW:.*]] = affine.apply
// CHECK: %[[DEAD_BOUND_MAX:.*]] = arith.maxsi %[[DEAD_BOUND_RAW]], %c0 : index
// CHECK: %[[DEAD_BOUND:.*]] = arith.minsi %[[DEAD_BOUND_MAX]], %c16 : index
// CHECK: %[[MIXED_LOOP:.*]]:5 = scf.for %{{.*}} = %[[LIVE_UPPER]] to %[[DEAD_BOUND]] step %c1
// CHECK-SAME: iter_args(%{{.*}} = %[[LIVE_LOOP]]#0, %{{.*}} = %[[LIVE_LOOP]]#1,
// CHECK-SAME: %{{.*}} = %[[LIVE_LOOP]]#2, %{{.*}} = %[[LIVE_LOOP]]#3,
// CHECK-SAME: %{{.*}} = %[[LIVE_LOOP]]#4)
// CHECK: htile.load %arg0
// CHECK: htile.load %arg1
// CHECK: %[[MIXED_RAW:.+]] = htile.dot %{{.*}}, %{{.*}} {transpose_b} : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf32>
// CHECK: htile.arange %c0 to %c64 : tensor<64xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [1] : tensor<64xindex> -> tensor<64x64xindex>
// CHECK: arith.subi
// CHECK: %[[MASKED_RAW:.+]] = arith.select %{{.*}}, %[[MIXED_RAW]], %{{.*}} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: htile.reduce %[[MASKED_RAW]] axis 1 kind "max" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: arith.mulf %[[MASKED_RAW]], %{{.*}} : tensor<64x64xf32>
// CHECK: math.exp2
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: htile.load %arg2
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK: htile.store %{{.*}}, %arg3
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK-NOT: ta.
// CHECK: htile.return
