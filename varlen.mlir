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

    // Difference from dense attention (1): fuse linalg ops post-loop upwards into the loop,
    // then fuse our custom insert op neptune.packed_window_insert into the loop nest
    // as a masked slice publication.
    %insert = transform.structured.match
        ops{["stablehlo.custom_call"]} attributes {call_target_name = "neptune.packed_window_insert"}
        in %func : (!any) -> !any
    transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %insert : (!any, !any) -> !any
    transform.htile.fuse_packed_window_insert %insert into %forall_loop : (!any, !any) -> !any

    // Difference from dense attention (2): first fuse ordinary producer chains so our custom
    // packed_window_extract become the immediate producers of in-loop slices.
    // Then fuse the packed_window_extract into the loop nest as a masked slice consumption.
    %consumer_loops = transform.merge_handles %forall_loop, %j0_loop : !any
    %_5, %consumer_loops_1 =
        transform.fusion.greedy_input_producers_into_consumer %consumer_loops
        : (!any) -> (!any, !any)
    %extracts = transform.structured.match
        ops{["stablehlo.custom_call"]} attributes {call_target_name = "neptune.packed_window_extract"}
        in %func : (!any) -> !any
    transform.htile.fuse_packed_window_extract %extracts into %consumer_loops_1 : (!any, !any) -> !any
    %_6, %consumer_loops_2 =
        transform.fusion.greedy_input_producers_into_consumer %consumer_loops
        : (!any) -> (!any, !any)

    // StableHLO slices were not fused because they are not fusable either. However, we can convert them to
    // tensor.extract_slice ops, then combine them with existing tensor.extract_slice ops in the loop.
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

    // Difference from dense attention (3): run LICM before specialize-dead-tile.
    // LICM is needed to move `doc_len[b]` and `doc_len[b+1]` out of the inner loop,
    // so the reduced loop range can be computed over these values.
    transform.apply_licm to %j0_loop : !any
    %live_loop, %mixed_loop = transform.loop.specialize_dead_tile in %j0_loop
        {dead_value = 0xFF800000 : f32} : (!any) -> (!any, !any)
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !any

    transform.yield
  }

  func.func public @main(%arg0: tensor<1024x4x64xf16>, %arg1: tensor<1024x4x64xf16>, %arg2: tensor<1024x4x64xf16>, %arg3: tensor<9xi32>) -> (tensor<1024x4x64xf16> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.slice %arg3 [0:8] : (tensor<9xi32>) -> tensor<8xi32>
    %1 = stablehlo.slice %arg3 [1:9] : (tensor<9xi32>) -> tensor<8xi32>
    %2 = stablehlo.subtract %1, %0 : tensor<8xi32>
    %3 = stablehlo.iota dim = 0 : tensor<512xi32>
    %4 = stablehlo.custom_call @neptune.packed_window_extract(%arg0, %0, %2, %cst_2) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>]} : (tensor<1024x4x64xf16>, tensor<8xi32>, tensor<8xi32>, tensor<f16>) -> tensor<8x512x4x64xf16>
    %5 = stablehlo.custom_call @neptune.packed_window_extract(%arg1, %0, %2, %cst_2) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>]} : (tensor<1024x4x64xf16>, tensor<8xi32>, tensor<8xi32>, tensor<f16>) -> tensor<8x512x4x64xf16>
    %6 = stablehlo.custom_call @neptune.packed_window_extract(%arg2, %0, %2, %cst_2) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>]} : (tensor<1024x4x64xf16>, tensor<8xi32>, tensor<8xi32>, tensor<f16>) -> tensor<8x512x4x64xf16>
    %7 = stablehlo.dot_general %4, %5, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] :(tensor<8x512x4x64xf16>, tensor<8x512x4x64xf16>) -> tensor<8x4x512x512xf32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x4x512x512xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<8x4x512x512xf32>
    %10 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<512xi32>) -> tensor<1x512xi32>
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x512xi32>) -> tensor<8x512xi32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x512xi32>
    %14 = stablehlo.compare LT, %12, %13, SIGNED : (tensor<8x512xi32>, tensor<8x512xi32>) -> tensor<8x512xi1>
    %15 = stablehlo.reshape %14 : (tensor<8x512xi1>) -> tensor<8x1x512x1xi1>
    %16 = stablehlo.reshape %14 : (tensor<8x512xi1>) -> tensor<8x1x1x512xi1>
    %17 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2, 3] : (tensor<8x1x512x1xi1>) -> tensor<8x1x512x512xi1>
    %18 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<8x1x1x512xi1>) -> tensor<8x1x512x512xi1>
    %19 = stablehlo.and %17, %18 : tensor<8x1x512x512xi1>
    %20 = stablehlo.convert %cst_0 : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2, 3] : (tensor<8x1x512x512xi1>) -> tensor<8x4x512x512xi1>
    %22 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<4x512x512xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [1, 2, 3] : (tensor<4x512x512xf32>) -> tensor<8x4x512x512xf32>
    %24 = stablehlo.select %21, %9, %23 : tensor<8x4x512x512xi1>, tensor<8x4x512x512xf32>
    %25 = stablehlo.reduce(%24 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<8x4x512x512xf32>, tensor<f32>) -> tensor<8x4x512xf32>
    %26 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x4x512xf32>
    %27 = stablehlo.maximum %26, %25 : tensor<8x4x512xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<8x4x512xf32>) -> tensor<8x4x512x1xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<8x4x512x1xf32>) -> tensor<8x4x512x512xf32>
    %30 = stablehlo.subtract %24, %29 : tensor<8x4x512x512xf32>
    %31 = stablehlo.exponential %30 : tensor<8x4x512x512xf32>
    %32 = stablehlo.reduce(%31 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x4x512x512xf32>, tensor<f32>) -> tensor<8x4x512xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<8x4x512xf32>) -> tensor<8x4x512x1xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2, 3] : (tensor<8x4x512x1xf32>) -> tensor<8x4x512x512xf32>
    %35 = stablehlo.divide %31, %34 : tensor<8x4x512x512xf32>
    %36 = stablehlo.convert %35 : (tensor<8x4x512x512xf32>) -> tensor<8x4x512x512xf16>
    %37 = stablehlo.dot_general %36, %6, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT]: (tensor<8x4x512x512xf16>, tensor<8x512x4x64xf16>) -> tensor<8x4x512x64xf32>
    %38 = stablehlo.transpose %37, dims = [0, 2, 1, 3] : (tensor<8x4x512x64xf32>) -> tensor<8x512x4x64xf32>
    %39 = stablehlo.convert %38 : (tensor<8x512x4x64xf32>) -> tensor<8x512x4x64xf16>
    %40 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f16>) -> tensor<1024x4x64xf16>
    %41 = stablehlo.custom_call @neptune.packed_window_insert(%39, %40, %0, %2) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<8x512x4x64xf16>, tensor<1024x4x64xf16>, tensor<8xi32>, tensor<8xi32>) -> tensor<1024x4x64xf16>
    return %41 : tensor<1024x4x64xf16>
  }
}
