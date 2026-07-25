!any = !transform.any_op

module @jit_doc_offset_attention attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, transform.with_named_sequence} {
  transform.named_sequence @match_4d_matmul_transb(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b i h d, b j h d -> b i h j"} : (!any) -> !any
    transform.yield %matched : !any
  }

  transform.named_sequence @__transform_main(%module: !any) {
    // Expecting the payload to be in stablehlo and have been "cleaned up": `inline, canonicalize, cse` have been run.
    // In particular we can't run `inline` here because it applies to the whole module,
    // and transform ops cannot update the module itself.
    %func0 = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %func = transform.apply_registered_pass "stablehlo-to-ta" to %func0 : (!any) -> !any
    transform.apply_patterns to %func {
      transform.apply_patterns.ta.exp_to_exp2
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !any
    transform.apply_cse to %func : !any
    %bmm0 = transform.collect_matching @match_4d_matmul_transb in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any

    // Nothing out of ordinary here: a regular schedule for dense attention.
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 128, 1, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bmax, %_2 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %prefix = transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %bmax { inline_elementwise } : (!any, !any) -> !any
    transform.linalg.erase_unused_operands_and_results %prefix : !any

    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.scf.fuse_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    %bmm1, %elemwise = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    %_3 = transform.fusion.repair_reduction_frontier
        %bmm1 reduce_producer %fused_bmax
        substituting elemwise %elemwise -> %elemwise_sidecars
        into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    %bsum, %elemwise_1 = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_4 = transform.fusion.repair_reduction_frontier
        %bsum reduce_producer %fused_bmax
        substituting elemwise %elemwise_1 -> %elemwise_sidecars_1
        into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    // Difference from dense attention (1): fuse gathers, tensor slices, etc. downwards into the loop nest.
    // Only do this after we've scheduled the dense attention itself
    // (specifically after the PV matmul is in the loop).
    // This step will fail to fuse the stablehlo.slice ops themselves, and that is fine.
    %slices = transform.structured.match ops{["stablehlo.slice"]} in %func : (!any) -> !any
    %consumer_loops = transform.merge_handles %forall_loop, %j0_loop : !any
    transform.fusion.greedy_producers_into_consumer %slices into %consumer_loops : (!any, !any) -> (!any, !any)

    // Instead of fusing stablehlo.slice, convert them to tensor.extract_slice, and merge with existing extract_slice ops in the loop.
    transform.apply_conversion_patterns to %func {
      transform.apply_conversion_patterns.stablehlo.slice_to_tensor
    } {illegal_ops = ["stablehlo.slice"], legal_dialects = ["tensor"], partial_conversion, preserve_handles} : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } : !any

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

    transform.yield
  }

  func.func public @main(%arg0: tensor<1024x4x64xf16>, %arg1: tensor<1024x4x64xf16>, %arg2: tensor<1024x4x64xf16>, %arg3: tensor<9xi32>) -> (tensor<1024x4x64xf16> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %c = stablehlo.constant dense<1024> : tensor<i32>
    %c_0 = stablehlo.constant dense<128> : tensor<i32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_3 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %0 = stablehlo.slice %arg3 [0:8] : (tensor<9xi32>) -> tensor<8xi32>
    %1 = stablehlo.slice %arg3 [1:9] : (tensor<9xi32>) -> tensor<8xi32>
    %2 = stablehlo.subtract %1, %0 : tensor<8xi32>
    %3 = stablehlo.iota dim = 0 : tensor<128xi32>
    %4 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %5 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<128xi32>) -> tensor<1x128xi32>
    %6 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x128xi32>
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<8x128xi32>
    %8 = stablehlo.add %6, %7 : tensor<8x128xi32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<8x128xi32>) -> tensor<8x128x1xi32>
    %10 = "stablehlo.gather"(%arg0, %9) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4, 64>}> : (tensor<1024x4x64xf16>, tensor<8x128x1xi32>) -> tensor<8x128x4x64xf16>
    %11 = "stablehlo.gather"(%arg1, %9) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4, 64>}> : (tensor<1024x4x64xf16>, tensor<8x128x1xi32>) -> tensor<8x128x4x64xf16>
    %12 = "stablehlo.gather"(%arg2, %9) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4, 64>}> : (tensor<1024x4x64xf16>, tensor<8x128x1xi32>) -> tensor<8x128x4x64xf16>
    %13 = stablehlo.dot_general %10, %11, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x128x4x64xf16>, tensor<8x128x4x64xf16>) -> tensor<8x4x128x128xf32>
    %14 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x4x128x128xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<8x4x128x128xf32>
    %16 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x128xi32>
    %18 = stablehlo.compare LT, %7, %17, SIGNED : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi1>
    %19 = stablehlo.reshape %18 : (tensor<8x128xi1>) -> tensor<8x1x1x128xi1>
    %20 = stablehlo.convert %cst_2 : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2, 3] : (tensor<8x1x1x128xi1>) -> tensor<8x4x128x128xi1>
    %22 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<4x128x128xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [1, 2, 3] : (tensor<4x128x128xf32>) -> tensor<8x4x128x128xf32>
    %24 = stablehlo.select %21, %15, %23 : tensor<8x4x128x128xi1>, tensor<8x4x128x128xf32>
    %25 = stablehlo.reduce(%24 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<8x4x128x128xf32>, tensor<f32>) -> tensor<8x4x128xf32>
    %26 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4x128xf32>
    %27 = stablehlo.maximum %26, %25 : tensor<8x4x128xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<8x4x128xf32>) -> tensor<8x4x128x1xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<8x4x128x1xf32>) -> tensor<8x4x128x128xf32>
    %30 = stablehlo.subtract %24, %29 : tensor<8x4x128x128xf32>
    %31 = stablehlo.exponential %30 : tensor<8x4x128x128xf32>
    %32 = stablehlo.reduce(%31 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<8x4x128x128xf32>, tensor<f32>) -> tensor<8x4x128xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<8x4x128xf32>) -> tensor<8x4x128x1xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2, 3] : (tensor<8x4x128x1xf32>) -> tensor<8x4x128x128xf32>
    %35 = stablehlo.divide %31, %34 : tensor<8x4x128x128xf32>
    %36 = stablehlo.convert %12 : (tensor<8x128x4x64xf16>) -> tensor<8x128x4x64xf32>
    %37 = stablehlo.convert %35 : tensor<8x4x128x128xf32>
    %38 = stablehlo.dot_general %36, %37, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x128x4x64xf32>, tensor<8x4x128x128xf32>) -> tensor<8x4x64x128xf32>
    %39 = stablehlo.transpose %38, dims = [0, 3, 1, 2] : (tensor<8x4x64x128xf32>) -> tensor<8x128x4x64xf32>
    %40 = stablehlo.convert %39 : (tensor<8x128x4x64xf32>) -> tensor<8x128x4x64xf16>
    %41 = stablehlo.iota dim = 0 : tensor<8xi32>
    %42 = stablehlo.compare GT, %17, %7, SIGNED : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi1>
    %43 = stablehlo.broadcast_in_dim %41, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %44 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<8x1xi32>
    %45 = stablehlo.multiply %43, %44 : tensor<8x1xi32>
    %46 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x1xi32>
    %47 = stablehlo.add %46, %45 : tensor<8x1xi32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x128xi32>
    %49 = stablehlo.add %48, %7 : tensor<8x128xi32>
    %50 = stablehlo.select %42, %8, %49 : tensor<8x128xi1>, tensor<8x128xi32>
    %51 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<1024x4x64xf16>
    %52 = stablehlo.broadcast_in_dim %50, dims = [0, 1] : (tensor<8x128xi32>) -> tensor<8x128x1xi32>
    %53 = "stablehlo.scatter"(%51, %52, %40) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg4: tensor<f16>, %arg5: tensor<f16>):
      stablehlo.return %arg5 : tensor<f16>
    }) : (tensor<1024x4x64xf16>, tensor<8x128x1xi32>, tensor<8x128x4x64xf16>) -> tensor<1024x4x64xf16>
    return %53 : tensor<1024x4x64xf16>
  }
}
