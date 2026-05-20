!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_2d_1d_reduction(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.structured %candidate : (!any) -> !any {
    ^bb0(%op: !any):
      transform.match.structured.dim %op[0, 1] {parallel} : !any
      transform.match.structured.dim %op[2] {reduction} : !any
      transform.match.structured.yield %op : !any
    }
    transform.yield %matched : !any
  }

  transform.named_sequence @return_matched(%arg: !any {transform.readonly}) -> !any {
    transform.yield %arg : !any
  }

  transform.named_sequence @__transform_main(%module: !any) {
    %func0 = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    // This pass removes unit-extent dimensions in tensors, which are often seen in reduction outputs
    // in Torch-MLIR code, because it applies `keepdim=True` on reductions.
    // The resulted code is difficult to work with in MLIR, because no one likes index maps
    // that have constants on the RHS (like `(d0, d1) -> (d0, 0)`).
    // By removing these dimensions, this pass creates actual broadcasting maps
    // (`(d0, d1) -> (d0, 0)`) that is more widely accepted.
    %func = transform.apply_registered_pass "linalg-fold-unit-extent-dims" to %func0 : (!any) -> !any

    // Grab all linalg.transpose and spell them out so they can be fused with other ops.
    // For attention there should be exactly 1 transpose for the QK^T matmul.
    %transposes = transform.structured.match ops{["linalg.transpose"]} in %func : (!any) -> !any
    %_0 = transform.structured.generalize %transposes : (!any) -> !any

    // Take the first batch matmul in the program, which we call `mm0`.
    // linalg.batch_matmul has exactly one batch dimension. In our case,
    // it is the fusion of the batch and head dimensions.
    %bmms = transform.structured.match ops{["linalg.batch_matmul"]} in %func : (!any) -> !any
    %bmm0, %_rest = transform.split_handle %bmms {overflow_result = 1} : (!any) -> (!any, !any)
    // Tile all parallel dimensions of mm0 (bh, i, j) into a scf.forall loop.
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 128, 64, 0] : (!any) -> (!any, !any)

    // Match an element-wise op that is a consumer of mm0, and fuse it into mm0.
    %bscale = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %fused_bscale = transform.loop.fuse_into_producer_op %bscale into %forall_loop : (!any, !any) -> !any
    // Fusion can create redundant loop-carried values and canonicalization removes them.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    // Fuse row-max into forall, splitting a serial `for` loop from forall in the process.
    %consumers = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %_2, %bmax = transform.foreach_match restrict_root in %consumers
        @match_2d_1d_reduction -> @return_matched : (!any) -> (!any, !any)
    // The "row-max" in the input program has two outputs: the max value and the argmax.
    // The subsequent fusion only supports single-output ops, so we remove the unused argmax
    // output before fusion.
    transform.loop.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.loop.fuse_reduction_consumer_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    transform.yield
  }
}
