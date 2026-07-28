// RUN: neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s
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
        {equation = "b ... h i d, b h j d -> b ... h i j"} : (!any) -> !any
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
        %bmm0 tile_sizes [1, 1, 1, 128, 64, 0] : (!any) -> (!any, !any)
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

    // --- HTile lowering begins ---
    transform.htile.linalg_to_semantic %func : !any
    %launches, %kernels = transform.htile.outline_kernels %forall_loop
        {kernel_names = ["attention_kernel"]} : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.verify %func : !any
    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x2x1024x64xf16>, %arg2: tensor<1x2x1024x64xf16>) -> tensor<1x2x2x1024x64xf16> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<1.250000e-01> : tensor<1xf64>
    %0 = stablehlo.reshape %arg0 : (tensor<1x4x1024x64xf16>) -> tensor<1x2x2x1024x64xf16>
    %1 = stablehlo.reshape %arg1 : (tensor<1x2x1024x64xf16>) -> tensor<1x1x2x1024x64xf16>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3, 4] : (tensor<1x1x2x1024x64xf16>) -> tensor<1x2x2x1024x64xf16>
    %3 = stablehlo.reshape %arg2 : (tensor<1x2x1024x64xf16>) -> tensor<1x1x2x1024x64xf16>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2, 3, 4] : (tensor<1x1x2x1024x64xf16>) -> tensor<1x2x2x1024x64xf16>
    %5 = stablehlo.transpose %2, dims = [0, 1, 2, 4, 3] : (tensor<1x2x2x1024x64xf16>) -> tensor<1x2x2x64x1024xf16>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2, 3, 4] : (tensor<1x2x2x64x1024xf16>) -> tensor<1x2x2x64x1024xf16>
    %7 = stablehlo.dot_general %0, %6, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3] : (tensor<1x2x2x1024x64xf16>, tensor<1x2x2x64x1024xf16>) -> tensor<1x2x2x1024x1024xf32>
    %8 = stablehlo.convert %cst_1 : (tensor<1xf64>) -> tensor<1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<1x2x2x1024x1024xf32>
    %11 = stablehlo.multiply %7, %10 : tensor<1x2x2x1024x1024xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x2x1024x1024xf32>, tensor<f32>) -> tensor<1x2x2x1024xf32>
    %13 = stablehlo.reshape %12 : (tensor<1x2x2x1024xf32>) -> tensor<1x2x2x1024x1xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3, 4] : (tensor<1x2x2x1024x1xf32>) -> tensor<1x2x2x1024x1024xf32>
    %15 = stablehlo.subtract %11, %14 : tensor<1x2x2x1024x1024xf32>
    %16 = stablehlo.exponential %15 : tensor<1x2x2x1024x1024xf32>
    %17 = stablehlo.reduce(%16 init: %cst_0) applies stablehlo.add across dimensions = [4] : (tensor<1x2x2x1024x1024xf32>, tensor<f32>) -> tensor<1x2x2x1024xf32>
    %18 = stablehlo.reshape %17 : (tensor<1x2x2x1024xf32>) -> tensor<1x2x2x1024x1xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2, 3, 4] : (tensor<1x2x2x1024x1xf32>) -> tensor<1x2x2x1024x1024xf32>
    %20 = stablehlo.divide %16, %19 : tensor<1x2x2x1024x1024xf32>
    %21 = stablehlo.convert %20 : (tensor<1x2x2x1024x1024xf32>) -> tensor<1x2x2x1024x1024xf16>
    %22 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3, 4] : (tensor<1x2x2x1024x64xf16>) -> tensor<1x2x2x1024x64xf16>
    %23 = stablehlo.dot_general %21, %22, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3] : (tensor<1x2x2x1024x1024xf16>, tensor<1x2x2x1024x64xf16>) -> tensor<1x2x2x1024x64xf32>
    %24 = stablehlo.convert %23 : (tensor<1x2x2x1024x64xf32>) -> tensor<1x2x2x1024x64xf16>
    %25 = stablehlo.reshape %24 : (tensor<1x2x2x1024x64xf16>) -> tensor<1x2x2x1024x64xf16>
    return %25 : tensor<1x2x2x1024x64xf16>
  }
}

// CHECK-LABEL: func.func @attention(
// CHECK-SAME: %arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x2x1024x64xf16>, %arg2: tensor<1x2x1024x64xf16>) -> tensor<1x2x2x1024x64xf16>
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: linalg.transpose
// CHECK-NOT: linalg.generic
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf32>
// CHECK-NOT: tensor.empty() : tensor<2x2x1024x64xf16>
// CHECK: %[[OUT:.*]] = memref.alloc() : memref<1x2x2x1024x64xf16>
// CHECK: %[[Q:.*]] = bufferization.to_buffer %arg0 read_only : tensor<1x4x1024x64xf16> to memref<1x4x1024x64xf16>
// CHECK: %[[K:.*]] = bufferization.to_buffer %arg1 read_only : tensor<1x2x1024x64xf16> to memref<1x2x1024x64xf16>
// CHECK: %[[V:.*]] = bufferization.to_buffer %arg2 read_only : tensor<1x2x1024x64xf16> to memref<1x2x1024x64xf16>
// CHECK: htile.launch_func @attention_kernel(%[[Q]], %[[K]], %[[V]], %[[OUT]]) {program_bounds = array<i64: 2, 2, 8>}
// CHECK-NOT: scf.forall
// CHECK: bufferization.to_tensor %[[OUT]]
// CHECK: return

// CHECK-LABEL: htile.kernel @attention_kernel
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<1x2x1024x64xf16>
// CHECK-SAME: memref<1x2x1024x64xf16>
// CHECK-SAME: memref<1x2x2x1024x64xf16>
// CHECK-SAME: attributes {program_bounds = array<i64: 2, 2, 8>}
// CHECK: htile.program_id 0
// CHECK: htile.program_id 1
// CHECK: htile.program_id 2
// CHECK-NOT: scf.forall
// CHECK: htile.full %{{.*}} : f32 -> tensor<128xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<128x64xf32>
// CHECK: %{{.*}}:3 = scf.for %{{.*}} = %c0 to %c16 step %c1 iter_args(
// CHECK: affine.apply
// CHECK: affine.apply
// CHECK: htile.load %arg0
// CHECK: htile.load %arg1
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} {transpose_b} : tensor<128x64xf16>, tensor<64x64xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "max" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: htile.broadcast %{{.*}} dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<128x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<128x64xf32> -> tensor<128xf32>
// CHECK: htile.load %arg2
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64xf16>, tensor<64x64xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<128x64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK: htile.store %{{.*}}, %arg3
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK: htile.return
