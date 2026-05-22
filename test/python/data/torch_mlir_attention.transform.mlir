// Transform-dialect schedule that transforms torch_mlir_attention.mlir into a FlashAttention-like fused program.
// Mirrors `_schedule_attention_flash` from Neptune, with some extra dimension reshaping capabilities.
//
// Usage:
//   mlir-opt test/python/data/torch_mlir_attention.mlir \
//     --load-dialect-plugin=build/libLoopTransform.so --transform-interpreter \
//     --transform-preload-library=transform-library-paths=test/python/data/torch_mlir_attention.transform.mlir \

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  // Checks if an op is a linalg operation with exactly 4D of an iteration space,
  // where the first 3 dims are parallel and the last dim is reduction.
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
    // Prepass 1. The input program from TorchMLIR has some "keepdim" reductions with an indexing map
    // like `(d0, d1, d2, d3) -> (d0, d1, d2, 0)`, and many MLIR transformations will reject these maps.
    // This step converts these keepdim reductions to non-keepdim ones, and remove the `0` dim
    // in these indexing maps.
    transform.linalg.fold_zero_indexed_unit_dims %func : !any

    // Prepass 2. TorchMLIR attention describes 4D matmuls with linalg.batch_matmul,
    // which requires exactly one "batch" dimension, by fusing the batch and head dims together.
    // We separate these dims by "generalizing" the batch matmul (and its producers,
    // like linalg.transpose) to a linalg.generic, then fuse the reshaping operation into the generic op.
    // Do that for the transpose op first (there should be exactly one for the QK^T matmul).
    %transposes = transform.structured.match ops{["linalg.transpose"]} in %func : (!any) -> !any
    %transposes_lg = transform.structured.generalize %transposes : (!any) -> !any
    transform.linalg.fold_expanding_reshape %transposes_lg : !any
    // Then do batch matmul ops.
    %bmms = transform.structured.match ops{["linalg.batch_matmul"]} in %func : (!any) -> !any
    %bmms_lg = transform.structured.generalize %bmms : (!any) -> !any
    transform.linalg.fold_expanding_reshape %bmms_lg : !any
    // This pattern cancels out back-to-back expand+collapse pairs.
    transform.apply_patterns to %func { transform.apply_patterns.tensor.reassociative_reshape_folding } : !any

    // Take the first batch matmul `bmm0`.
    // Inline elementwise ops before bmm0 (in this case, should be F16->F32 casts) into it.
    %bmm0, %_0 = transform.split_handle %bmms_lg : (!any) -> (!any, !any)
    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    transform.linalg.erase_unused_operands_and_results %bmm0 : !any
    // Tile all parallel dimensions of bmm0 (b, h, i, j) into a scf.forall loop.
    // We'll fuse everything else into this loop nest.
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 128, 64, 0] : (!any) -> (!any, !any)

    // Fusion 1. Match an element-wise op that is a consumer of mm0, and fuse it into mm0.
    //   TVM: sch.reverse_compute_at(bscale, j0)
    %bscale = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %fused_bscale = transform.fusion.into_producer %bscale into %forall_loop : (!any, !any) -> !any
    // Fusion can create redundant loop-carried values, and canonicalization removes them.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    // Fusion 2. Fuse row-max into forall, splitting a serial `for` loop from forall in the process.
    // Because of how MLIR scf.for works, this fusion implicitly also r-factors the reduction.
    //   TVM: sch.reverse_compute_at(bmax, j0); sch.rfactor(...)
    %consumers = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %_2, %bmax = transform.foreach_match restrict_root in %consumers
        @match_3d_1d_reduction -> @return_matched : (!any) -> (!any, !any)
    // The "row-max" in the input program has two outputs: the max value and the argmax.
    // The subsequent fusion only supports single-output ops, so we remove the unused argmax
    // output before fusion.
    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.scf.fuse_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    // Fusion 3 (rolling update). First find the nearest reduction reachable from the loop's
    // output value, together with the ordered elementwise chain between them.
    %bsum, %elemwise = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    // Clone and fuse that elementwise chain under %forall_loop and %j0_loop,
    // publishing the "sidecar" tensors as extra loop results.
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    // Repair the first reduction frontier by turning it into loop-carried state
    // driven by the relayed sidecar value.
    %fused_bsum = transform.fusion.repair_reduction_frontier
        (%fused_bmax, %bsum) and (%elemwise, %elemwise_sidecars) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    // Fusion 4. Apply rolling update again, this time with the second matmul being the reduction.
    %bmm1, %elemwise_1 = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    // Also inline (F16->F32 casts) into the second matmul. `operand_number = 1` says only
    // inline producers of the RHS of the matmul.
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    transform.linalg.erase_unused_operands_and_results %bmm1 : !any
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_3 = transform.fusion.repair_reduction_frontier
        (%fused_bsum, %bmm1) and (%elemwise_1, %elemwise_sidecars_1) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    // Fusion 5. Fuse the trailing FP32->FP16 cast into the forall loop (but outside the for loop).
    %trunc = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %fused_trunc = transform.fusion.into_producer %trunc into %forall_loop : (!any, !any) -> !any

    // Post-pass: pushes lingering init tensor (see destination-passing style)
    // before and outside the loops into the loop body.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.apply_cse to %func : !any

    transform.yield
  }
}
