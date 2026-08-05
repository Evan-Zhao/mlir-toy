// RUN: neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s

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
        %bmm0 tile_sizes [1, 1, 1, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bmax, %_2 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %prefix = transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %bmax { inline_elementwise } : (!any, !any) -> !any
    transform.linalg.erase_unused_operands_and_results %prefix : !any

    // Use fuse_partial_reduction_into_forall here (similar to RFactor in TVM).
    // It splits bmax into a local reduction and a global one (bmax_wb),
    // and fuses the local one under the forall loop (becomes bmax_rf).
    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %bmax_rf, %bmax_wb = transform.scf.fuse_partial_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    // Find the next reduction forward from the loop, but only look for users of the 0th result of the loop.
    // %bmax_rf would be using the 1st result of the loop, and we don't want that one.
    %bsum, %elemwise = transform.fusion.find_next_reduction %forall_loop[0] : (!any) -> (!any, !any)
    %sidecars = transform.fusion.clone_fuse_rfactor_elemwise
        %elemwise into %forall_loop substituting (%bmax_wb -> %bmax_rf)
        : (!any, !any, !any, !any) -> !any
    %fused_bsum, %bsum_writeback = transform.fusion.repair_rfactor_reduction_frontier
        %bsum substituting reduce %bmax_wb -> %bmax_rf elemwise %elemwise -> %sidecars
        into %forall_loop : (!any, !any, !any, !any, !any, !any) -> (!any, !any)

    // Apply SplitK repair again for the remaining second-matmul reduction frontier.
    %bmm1, %elemwise_1 = transform.fusion.find_next_reduction %forall_loop[0] : (!any) -> (!any, !any)
    %sidecars_1 = transform.fusion.clone_fuse_rfactor_elemwise
        %elemwise_1 into %forall_loop substituting (%bmax_wb -> %bmax_rf)
        : (!any, !any, !any, !any) -> !any
    %fused_bmm1, %bmm1_writeback = transform.fusion.repair_rfactor_reduction_frontier
        %bmm1 substituting reduce %bmax_wb -> %bmax_rf elemwise %elemwise_1 -> %sidecars_1
        into %forall_loop : (!any, !any, !any, !any, !any, !any) -> (!any, !any)

    %_3, %writeback_loop = transform.structured.tile_using_forall
        %bmax_wb tile_sizes [1, 1, 1] : (!any) -> (!any, !any)
    // Remove superseded frontier users before fusing the writeback consumer chain.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    %ret = transform.structured.match ops{["func.return"]} in %func : (!any) -> !any
    transform.fusion.greedy_consumers_into_producer
        %writeback_loop[0] until %ret { inline_elementwise } : (!any, !any) -> !any

    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !any
    transform.apply_cse to %func : !any

    transform.htile.linalg_to_semantic %func : !any
    %kernel_loops = transform.merge_handles %forall_loop, %writeback_loop : !any
    %launches, %kernels = transform.htile.outline_kernels %kernel_loops
        {kernel_names = ["decode_partial", "decode_merge"]}
        : (!any) -> (!any, !any)
    transform.verify %func : !any

    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x1x64xf16>, %arg1: tensor<1x4x1024x64xf16>, %arg2: tensor<1x4x1024x64xf16>) -> tensor<1x4x1x64xf16> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<1.250000e-01> : tensor<1xf64>
    %0 = stablehlo.transpose %arg1, dims = [0, 1, 3, 2] : (tensor<1x4x1024x64xf16>) -> tensor<1x4x64x1024xf16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x4x64x1024xf16>) -> tensor<1x4x64x1024xf16>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x1x64xf16>, tensor<1x4x64x1024xf16>) -> tensor<1x4x1x1024xf32>
    %3 = stablehlo.convert %cst_1 : (tensor<1xf64>) -> tensor<1xf32>
    %4 = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1x4x1x1024xf32>
    %6 = stablehlo.multiply %2, %5 : tensor<1x4x1x1024xf32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x1x1024xf32>, tensor<f32>) -> tensor<1x4x1xf32>
    %8 = stablehlo.reshape %7 : (tensor<1x4x1xf32>) -> tensor<1x4x1x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3] : (tensor<1x4x1x1xf32>) -> tensor<1x4x1x1024xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<1x4x1x1024xf32>
    %11 = stablehlo.exponential %10 : tensor<1x4x1x1024xf32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x4x1x1024xf32>, tensor<f32>) -> tensor<1x4x1xf32>
    %13 = stablehlo.reshape %12 : (tensor<1x4x1xf32>) -> tensor<1x4x1x1xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x4x1x1xf32>) -> tensor<1x4x1x1024xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x4x1x1024xf32>
    %16 = stablehlo.convert %15 : (tensor<1x4x1x1024xf32>) -> tensor<1x4x1x1024xf16>
    %17 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 3] : (tensor<1x4x1024x64xf16>) -> tensor<1x4x1024x64xf16>
    %18 = stablehlo.dot_general %16, %17, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x1x1024xf16>, tensor<1x4x1024x64xf16>) -> tensor<1x4x1x64xf32>
    %19 = stablehlo.convert %18 : (tensor<1x4x1x64xf32>) -> tensor<1x4x1x64xf16>
    return %19 : tensor<1x4x1x64xf16>
  }
}

// CHECK-LABEL: func.func @attention(
// CHECK-SAME: %arg0: tensor<1x4x1x64xf16>, %arg1: tensor<1x4x1024x64xf16>, %arg2: tensor<1x4x1024x64xf16>) -> tensor<1x4x1x64xf16>
// CHECK: memref.alloc() : memref<4x16xf32>
// CHECK: memref.alloc() : memref<4x16xf32>
// CHECK: memref.alloc() : memref<4x64x16xf32>
// CHECK: htile.launch_func @decode_partial
// CHECK-SAME: {program_bounds = array<i64: 4, 16>}
// CHECK: htile.launch_func @decode_merge
// CHECK-SAME: {program_bounds = array<i64: 4>}
// CHECK: bufferization.to_tensor
// CHECK: return

// CHECK-LABEL: htile.kernel @decode_partial
// CHECK-SAME: memref<1x4x1x64xf16>
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<4x16xf32>
// CHECK-SAME: memref<4x16xf32>
// CHECK-SAME: memref<4x64x16xf32>
// CHECK-SAME: attributes {program_bounds = array<i64: 4, 16>}
// CHECK: htile.program_id 0
// CHECK: htile.program_id 1
// CHECK: htile.load %arg0
// CHECK: htile.load %arg1
// CHECK: htile.dot %{{.*}}, %{{.*}} {transpose_b}
// CHECK: htile.load %arg3
// CHECK: htile.reduce %{{.*}} axis 0 kind "max" : tensor<64xf32> -> tensor<f32>
// CHECK: math.exp2
// CHECK: htile.load %arg4
// CHECK: htile.reduce %{{.*}} axis 0 kind "sum" : tensor<64xf32> -> tensor<f32>
// CHECK: htile.load %arg2
// CHECK: htile.load %arg5
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64xf16>, tensor<64x64xf16>, tensor<64xf32> -> tensor<64xf32>
// CHECK: htile.store %{{.*}}, %arg3
// CHECK: htile.store %{{.*}}, %arg4
// CHECK: htile.store %{{.*}}, %arg5
// CHECK: htile.return

// CHECK-LABEL: htile.kernel @decode_merge
// CHECK-SAME: memref<4x16xf32>
// CHECK-SAME: memref<4x64x16xf32>
// CHECK-SAME: memref<4x16xf32>
// CHECK-SAME: memref<1x4x1x64xf16>
// CHECK-SAME: attributes {program_bounds = array<i64: 4>}
// CHECK: htile.program_id 0
// CHECK: htile.load %arg0
// CHECK: htile.reduce %{{.*}} axis 3 kind "max"
// CHECK: htile.load %arg1
// CHECK: math.exp2
// CHECK: htile.load %arg2
// CHECK: htile.reduce %{{.*}} axis 4 kind "sum"
// CHECK: htile.reduce %{{.*}} axis 3 kind "sum"
// CHECK: arith.divf
// CHECK: arith.truncf
// CHECK: htile.store %{{.*}}, %arg3
// CHECK: htile.return
