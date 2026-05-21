!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_3d_1d_reduction(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.structured %candidate : (!any) -> !any {
    ^bb0(%op: !any):
      transform.match.structured.dim %op[0, 1, 2] {parallel} : !any
      transform.match.structured.dim %op[3] {reduction} : !any
      transform.match.structured.yield %op : !any
    }
    transform.yield %matched : !any
  }

  transform.named_sequence @return_matched(%arg: !any {transform.readonly}) -> !any {
    transform.yield %arg : !any
  }

  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.loop.fold_zero_indexed_unit_dims %func : !transform.any_op

    // TorchMLIR attention describes 4D matmuls with linalg.batch_matmul, which only supports
    // exactly one batch dimension, by fusing the batch dims together.
    // We restore the batch dims by "generalizing" the batch matmul (and its producers,
    // like linalg.transpose) to a linalg.generic, then fuse the reshaping operation into the generic op.
    // Do that for the transpose op first (there should be exactly one for the QK^T matmul).
    %transposes = transform.structured.match ops{["linalg.transpose"]} in %func : (!any) -> !any
    %transposes_lg = transform.structured.generalize %transposes : (!any) -> !any
    transform.loop.fold_expanding_reshape %transposes_lg : !any
    // Then do batch matmul ops.
    %bmms = transform.structured.match ops{["linalg.batch_matmul"]} in %func : (!any) -> !any
    %bmms_lg = transform.structured.generalize %bmms : (!any) -> !any
    transform.loop.fold_expanding_reshape %bmms_lg : !any

    // Take the first batch matmul `bmm0`.
    // Inline elementwise ops before bmm0 (in this case, should be F16->F32 casts) into it.
    %bmm0, %_0 = transform.split_handle %bmms_lg : (!any) -> (!any, !any)
    %bmm0_1 = transform.loop.inline_elementwise %bmm0: (!any) -> !any
    transform.loop.erase_unused_operands_and_results %bmm0_1 : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    // Tile all parallel dimensions of mm0 (b, h, i, j) into a scf.forall loop.
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0_1 tile_sizes [1, 1, 128, 64, 0] : (!any) -> (!any, !any)

    // Match an element-wise op that is a consumer of mm0, and fuse it into mm0.
    %bscale = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %fused_bscale = transform.loop.fuse_into_producer_op %bscale into %forall_loop : (!any, !any) -> !any
    // Fusion can create redundant loop-carried values and canonicalization removes them.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    // Fuse row-max into forall, splitting a serial `for` loop from forall in the process.
    %consumers = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %_2, %bmax = transform.foreach_match restrict_root in %consumers
        @match_3d_1d_reduction -> @return_matched : (!any) -> (!any, !any)
    // The "row-max" in the input program has two outputs: the max value and the argmax.
    // The subsequent fusion only supports single-output ops, so we remove the unused argmax
    // output before fusion.
    transform.loop.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.loop.fuse_reduction_consumer_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    %bsum, %elemwise = transform.match.loop_ru.rolling_update_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars = transform.loop_ru.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %fused_bsum = transform.loop_ru.repair_reduction_frontier
        (%fused_bmax, %bsum) and (%elemwise, %elemwise_sidecars) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    %bmm1, %elemwise_1 = transform.match.loop_ru.rolling_update_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %bmm1_1 = transform.loop.inline_elementwise %bmm1 { operand_number = 1 }: (!any) -> !any
    transform.loop.erase_unused_operands_and_results %bmm1_1 : !any
    %elemwise_sidecars_1 = transform.loop_ru.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_3 = transform.loop_ru.repair_reduction_frontier
        (%fused_bsum, %bmm1_1) and (%elemwise_1, %elemwise_sidecars_1) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %trunc = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %fused_trunc = transform.loop.fuse_into_producer_op %trunc into %forall_loop : (!any, !any) -> !any

    %func_1 = transform.apply_registered_pass "remove-dead-values" to %func : (!any) -> !any
    transform.apply_patterns to %func_1 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.fold_tensor_empty
    } : !any

    transform.yield
  }
}
