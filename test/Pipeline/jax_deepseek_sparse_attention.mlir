// RUN: not neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s
//
// Sparse-attention schedule for the default output of:
//
//   python examples/jax_deepseek_sparse_attention.py

!any = !transform.any_op

module @jit_deepseek_sparse_attention attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, transform.with_named_sequence} {
  transform.named_sequence @match_sparse_qk(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b q h d, b q t d -> b q h t"} : (!any) -> !any
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
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.sink_scale_after_max
    } : !any
    transform.apply_cse to %func : !any
    %bmm0 = transform.collect_matching @match_sparse_qk in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any

    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 16, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bmax, %_2 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %prefix = transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %bmax { inline_elementwise } : (!any, !any) -> !any
    transform.linalg.erase_unused_operands_and_results %prefix : !any
    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %t_loop = transform.scf.fuse_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    %bsum, %elemwise = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %t_loop : (!any, !any, !any) -> !any
    %_3 = transform.fusion.repair_reduction_frontier
        %bsum reduce_producer %fused_bmax
        substituting elemwise %elemwise -> %elemwise_sidecars
        into %forall_loop, %t_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    %bmm1, %elemwise_1 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %t_loop : (!any, !any, !any) -> !any
    %_4 = transform.fusion.repair_reduction_frontier
        %bmm1 reduce_producer %fused_bmax
        substituting elemwise %elemwise_1 -> %elemwise_sidecars_1
        into %forall_loop, %t_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %ret = transform.structured.match ops{["func.return"]} in %func : (!any) -> !any
    transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %ret : (!any, !any) -> !any

    // Pull both gathers and their shared selected-index producer into the
    // streaming selected-token loop. Each gather is retiled from
    // [1, 128, 2048, D] to [1, 1, 64, D].
    %consumer_loops = transform.merge_handles %forall_loop, %t_loop : !any
    %fused_producers, %_5 = transform.fusion.greedy_input_producers_into_consumer %consumer_loops
        : (!any) -> (!any, !any)

    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !any
    transform.apply_cse to %func : !any
    transform.apply_licm to %t_loop : !any

    transform.htile.linalg_to_semantic %func : !any
    %launches, %kernels = transform.htile.outline_kernels %forall_loop
        {kernel_names = ["deepseek_sparse_attention_kernel"]} : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.verify %func : !any
    transform.yield
  }

  func.func public @main(%arg0: tensor<1x128x128x576xbf16>, %arg1: tensor<1x16384x576xbf16>, %arg2: tensor<1x16384x512xbf16>, %arg3: tensor<1x128x2048xi32>) -> (tensor<1x128x128x512xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1, 2] : (tensor<1x128x2048xi32>) -> tensor<1x128x2048x1xi32>
    %1 = "stablehlo.gather"(%arg1, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [3], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 576>}> : (tensor<1x16384x576xbf16>, tensor<1x128x2048x1xi32>) -> tensor<1x128x2048x576xbf16>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1, 2] : (tensor<1x128x2048xi32>) -> tensor<1x128x2048x1xi32>
    %3 = "stablehlo.gather"(%arg2, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [3], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 512>}> : (tensor<1x16384x512xbf16>, tensor<1x128x2048x1xi32>) -> tensor<1x128x2048x512xbf16>
    %4 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x128x128x576xbf16>, tensor<1x128x2048x576xbf16>) -> tensor<1x128x128x2048xf32>
    %cst = stablehlo.constant dense<0.0416666679> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128x2048xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1x128x128x2048xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x128x128x2048xf32>, tensor<f32>) -> tensor<1x128x128xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
    %9 = stablehlo.maximum %8, %7 : tensor<1x128x128xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1x128x128xf32>) -> tensor<1x128x128x1xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2, 3] : (tensor<1x128x128x1xf32>) -> tensor<1x128x128x2048xf32>
    %12 = stablehlo.subtract %6, %11 : tensor<1x128x128x2048xf32>
    %13 = stablehlo.exponential %12 : tensor<1x128x128x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%13 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<1x128x128x2048xf32>, tensor<f32>) -> tensor<1x128x128xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2] : (tensor<1x128x128xf32>) -> tensor<1x128x128x1xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2, 3] : (tensor<1x128x128x1xf32>) -> tensor<1x128x128x2048xf32>
    %17 = stablehlo.divide %13, %16 : tensor<1x128x128x2048xf32>
    %18 = stablehlo.convert %17 : (tensor<1x128x128x2048xf32>) -> tensor<1x128x128x2048xbf16>
    %19 = stablehlo.dot_general %18, %3, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x128x128x2048xbf16>, tensor<1x128x2048x512xbf16>) -> tensor<1x128x128x512xf32>
    %20 = stablehlo.convert %19 : (tensor<1x128x128x512xf32>) -> tensor<1x128x128x512xbf16>
    return %20 : tensor<1x128x128x512xbf16>
  }
}

// HTile kernel outlining does not recognize stablehlo.gather operations (yet).
// A failure is preferable here.
// CHECK: error: unsupported tensor read by 'stablehlo.gather' during kernel outlining
// CHECK: note: see current operation:
// CHECK-SAME: "stablehlo.gather"
// CHECK-SAME: tensor<1x1x64x576xbf16>
// CHECK: error: failed to bufferize tensor reads in foralls
