// RUN: neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s
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
    transform.apply_cse to %func : !any
    %bmm0 = transform.collect_matching @match_4d_matmul_transb in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any

    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 128, 64, 0] : (!any) -> (!any, !any)
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

  func.func @attention(%arg0: tensor<1x4x128x64xf16>, %arg1: tensor<1x4x512x64xf16>, %arg2: tensor<1x4x512x64xf16>) -> tensor<1x4x128x64xf16> {
    %c = stablehlo.constant dense<0> : tensor<128x512xi64>
    %c_0 = stablehlo.constant dense<true> : tensor<128x512xi1>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_1 = stablehlo.constant dense<false> : tensor<128x512xi1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<1.250000e-01> : tensor<1xf64>
    %0 = stablehlo.transpose %arg1, dims = [0, 1, 3, 2] : (tensor<1x4x512x64xf16>) -> tensor<1x4x64x512xf16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x4x64x512xf16>) -> tensor<1x4x64x512xf16>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x128x64xf16>, tensor<1x4x64x512xf16>) -> tensor<1x4x128x512xf32>
    %3 = stablehlo.convert %cst_3 : (tensor<1xf64>) -> tensor<1xf32>
    %4 = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1x4x128x512xf32>
    %6 = stablehlo.multiply %2, %5 : tensor<1x4x128x512xf32>
    %7 = stablehlo.iota dim = 1 : tensor<128x512xi64>
    %8 = stablehlo.iota dim = 0 : tensor<128x512xi64>
    %9 = stablehlo.add %8, %c : tensor<128x512xi64>
    %10 = stablehlo.compare LE, %7, %9, SIGNED : (tensor<128x512xi64>, tensor<128x512xi64>) -> tensor<128x512xi1>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<128x512xi1>) -> tensor<128x512xi1>
    %12 = stablehlo.select %11, %c_0, %c_1 : tensor<128x512xi1>, tensor<128x512xi1>
    %13 = stablehlo.reshape %12 : (tensor<128x512xi1>) -> tensor<1x1x128x512xi1>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x1x128x512xi1>) -> tensor<1x4x128x512xi1>
    %15 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2, 3] : (tensor<1x4x128x512xf32>) -> tensor<1x4x128x512xf32>
    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x128x512xf32>
    %17 = stablehlo.select %14, %15, %16 : tensor<1x4x128x512xi1>, tensor<1x4x128x512xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x512xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %19 = stablehlo.reshape %18 : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x512xf32>
    %21 = stablehlo.subtract %17, %20 : tensor<1x4x128x512xf32>
    %22 = stablehlo.exponential %21 : tensor<1x4x128x512xf32>
    %23 = stablehlo.reduce(%22 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x512xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %24 = stablehlo.reshape %23 : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x512xf32>
    %26 = stablehlo.divide %22, %25 : tensor<1x4x128x512xf32>
    %27 = stablehlo.convert %26 : (tensor<1x4x128x512xf32>) -> tensor<1x4x128x512xf16>
    %28 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 3] : (tensor<1x4x512x64xf16>) -> tensor<1x4x512x64xf16>
    %29 = stablehlo.dot_general %27, %28, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x128x512xf16>, tensor<1x4x512x64xf16>) -> tensor<1x4x128x64xf32>
    %30 = stablehlo.convert %29 : (tensor<1x4x128x64xf32>) -> tensor<1x4x128x64xf16>
    return %30 : tensor<1x4x128x64xf16>
  }
}

// CHECK-LABEL: func.func @attention(
// CHECK-SAME: %arg0: tensor<1x4x128x64xf16>, %arg1: tensor<1x4x512x64xf16>, %arg2: tensor<1x4x512x64xf16>) -> tensor<1x4x128x64xf16>
// CHECK-NOT: tensor.empty() : tensor<4x128x64xf32>
// CHECK-NOT: tensor.empty() : tensor<4x128x64xf16>
// CHECK: htile.launch_func @attention_kernel
// CHECK-SAME: {program_bounds = array<i64: 4>}
// CHECK-NOT: scf.forall
// CHECK: return

// CHECK-LABEL: htile.kernel @attention_kernel
// CHECK-SAME: memref<1x4x128x64xf16>
// CHECK-SAME: memref<1x4x512x64xf16>
// CHECK-SAME: memref<1x4x512x64xf16>
// CHECK-SAME: memref<1x4x128x64xf16>
// CHECK-SAME: attributes {program_bounds = array<i64: 4>}
// CHECK: htile.program_id 0
// CHECK-NOT: scf.forall
// CHECK: htile.full %cst{{.*}} : f32 -> tensor<128xf32>
// CHECK: htile.full %cst{{.*}} : f32 -> tensor<128x64xf32>
// CHECK: %[[LIVE:.+]]:3 = scf.for %{{.*}} = %c0 to %c0 step %c1 iter_args(
// CHECK: scf.yield
// CHECK: %{{.*}}:3 = scf.for %{{.*}} = %c0 to %c2 step %c1 iter_args(%{{.*}} = %[[LIVE]]#0, %{{.*}} = %[[LIVE]]#1, %{{.*}} = %[[LIVE]]#2)
// CHECK: htile.load %arg0
// CHECK: htile.load %arg1
// CHECK: htile.arange %c0 to %c128 : tensor<128xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [1] : tensor<128xindex> -> tensor<128x64xindex>
// CHECK: htile.arange %c0 to %c64 : tensor<64xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [0] : tensor<64xindex> -> tensor<128x64xindex>
// CHECK: htile.full %{{.*}} : index -> tensor<128x64xindex>
// CHECK: arith.muli %{{.*}}, %{{.*}} {{.*}} : tensor<128x64xindex>
// CHECK: arith.addi %{{.*}}, %{{.*}} : tensor<128x64xindex>
// CHECK: arith.index_cast %{{.*}} : tensor<128x64xindex> to tensor<128x64xi64>
// CHECK: arith.index_cast %{{.*}} : tensor<128x64xindex> to tensor<128x64xi64>
// CHECK: arith.cmpi sle, %{{.*}}, %{{.*}} : tensor<128x64xi64>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64xi1>, tensor<128x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "max" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: math.exp2
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: htile.load %arg2
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64xf16>, tensor<64x64xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<128x64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK: htile.store %{{.*}}, %arg3
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK: htile.return
