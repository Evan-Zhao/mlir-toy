// Transform-dialect schedule that lowers attention_l0.mlir into the shape of
// flash_attention_l1.mlir. Mirrors `_schedule_attention_flash` from Neptune.
//
// Usage:
//   mlir-opt test/python/data/attention_l0.mlir \
//     --load-dialect-plugin=build/libLoopTransform.so \
//     --transform-preload-library=transform-library-paths=test/python/data/attention_l0_to_l1.transform.mlir \
//     --transform-interpreter

module attributes {transform.with_named_sequence} {
  // Checks if an op `op` is an element-wise operation, including linalg.generic and linalg.map.
  transform.named_sequence @match_elemwise(
      %candidate: !transform.any_op {transform.readonly}
  ) -> !transform.any_op {
    %matched = transform.match.structured %candidate
        : (!transform.any_op) -> !transform.any_op {
    ^bb0(%op: !transform.any_op):
      transform.match.structured.body %op {elementwise} : !transform.any_op
      transform.match.structured.yield %op : !transform.any_op
    }
    transform.yield %matched : !transform.any_op
  }

  transform.named_sequence @match_unary_reduction(
      %candidate: !transform.any_op {transform.readonly}
  ) -> !transform.any_op {
    %matched = transform.match.structured %candidate
        : (!transform.any_op) -> !transform.any_op {
    ^bb0(%op: !transform.any_op):
      %num_inputs = transform.match.structured.num_inputs %op
          : (!transform.any_op) -> !transform.param<i64>
      %one = transform.param.constant 1 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %num_inputs, %one : !transform.param<i64>
      transform.match.structured.dim %op[0, 1, 2] {parallel} : !transform.any_op
      transform.match.structured.dim %op[3] {reduction} : !transform.any_op
      transform.match.structured.yield %op : !transform.any_op
    }
    transform.yield %matched : !transform.any_op
  }

  transform.named_sequence @return_matched(
      %arg: !transform.any_op {transform.readonly}
  ) -> !transform.any_op {
    transform.yield %arg : !transform.any_op
  }

  // Fuses all elementwise consumers of a given producer op into the producer.
  transform.named_sequence @fuse_elemwise_consumer_of_result0(
      %producer: !transform.any_op {transform.readonly}
  ) -> (!transform.any_op) {
    %consumers = transform.get_consumers_of_result %producer[0]
        : (!transform.any_op) -> !transform.any_op
    %_1, %elemwise_consumers = transform.foreach_match restrict_root in %consumers
        @match_elemwise -> @return_matched
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %elemwise_consumers : !transform.any_op
    %fused_consumers =
      transform.fusion.into_producer %elemwise_consumers into %producer
        : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield %fused_consumers : !transform.any_op
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    // Step 0. Decompose softmax into linalg ops, and non-linalg elemwise ops
    // (like arith.truncf) to linalg ops too.
    %func0 = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %func = transform.apply_registered_pass "convert-elementwise-to-linalg" to %func0
        : (!transform.any_op) -> !transform.any_op
    %softmax = transform.structured.match ops{["linalg.softmax"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %decomposed = transform.structured.decompose_interface %softmax
        : (!transform.any_op) -> !transform.any_op

    // Step 1. Pattern match a matmul to find "matmul1" in attention,
    // then tile its outer iteration space.
    // == Pattern match:
    // Match the QK^T contraction by compute structure, not by tensor shape.
    // This is a batched matmul-NT generic:
    //   S[b, h, i, j] += Q[b, h, i, k] * K[b, h, j, k]
    %qk_candidate = transform.structured.match ops{["linalg.generic"]}
        attributes {
          indexing_maps = [
            affine_map<(b, h, i, j, k) -> (b, h, i, k)>,
            affine_map<(b, h, i, j, k) -> (b, h, j, k)>,
            affine_map<(b, h, i, j, k) -> (b, h, i, j)>]
        } in %func
        : (!transform.any_op) -> !transform.any_op
    %qk_one = transform.split_handle %qk_candidate
        : (!transform.any_op) -> !transform.any_op
    %b0 = transform.match.structured %qk_one
        : (!transform.any_op) -> !transform.any_op {
    ^bb0(%candidate: !transform.any_op):
      transform.match.structured.dim %candidate[0, 1, 2, 3] {parallel}
          : !transform.any_op
      transform.match.structured.dim %candidate[4] {reduction}
          : !transform.any_op
      transform.match.structured.body %candidate
          {contraction = ["arith.mulf", "arith.addf"]} : !transform.any_op
      transform.match.structured.yield %candidate : !transform.any_op
    }
    // == Tiling
    // We'll tile all parallel dimensions of b0 (b, h, i, j) into a scf.forall loop,
    // which has a parallel execution intent.
    // Soon the `j` dimension will become sequential as fusion happens,
    // but we'll worry about that later.
    %_0, %forall_loop =
      transform.structured.tile_using_forall %b0 tile_sizes [1, 1, 128, 128, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 2. Match an element-wise op that is a consumer of b0, and fuse it into b0
    // For attention, this op would be the score-scaling op, which we call `bscale`.
    // `bscale` will be fused under `forall_loop` (the outer loop nest we created by tiling).
    //   TVM: sch.reverse_compute_at(bscale, j0)
    %fused_bscale = transform.include @fuse_elemwise_consumer_of_result0 failures(propagate)
        (%forall_loop) : (!transform.any_op) -> (!transform.any_op)
    // Fusion can create redundant loop-carried values and canonicalization removes them.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !transform.any_op

    // Step 3. Match a reduction op that is a consumer of bscale.
    // This would be the row-max op in attention softmax.
    // Because of how MLIR scf.for and linalg work, this fusion implicitly also rfactors the reduction.
    //   TVM: sch.reverse_compute_at(bmax, j0); sch.rfactor(...)
    %consumers_1 = transform.get_consumers_of_result %forall_loop[0]
        : (!transform.any_op) -> !transform.any_op
    %_2, %bmax = transform.foreach_match restrict_root in %consumers_1
        @match_unary_reduction -> @return_matched
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_bmax, %j0_loop =
      transform.scf.fuse_reduction_into_forall %bmax into %forall_loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 4. Prepare for rolling update. First find the nearest reduction
    // frontier reachable from the loop-local value, together with the ordered
    // elementwise chain between them.
    %bsum, %elemwise =
      transform.fusion.find_next_reduction %forall_loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Clone and fuse that elementwise chain under %forall_loop and %j0_loop,
    // publishing the sidecar tensors as extra loop results.
    %elemwise_sidecars =
      transform.fusion.clone_fuse_elemwise %elemwise into %forall_loop, %j0_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    // Repair the first reduction frontier by turning it into loop-carried state
    // driven by the relayed sidecar value.
    %fused_bsum = transform.fusion.repair_reduction_frontier
        (%fused_bmax, %bsum) and (%elemwise, %elemwise_sidecars) into %forall_loop, %j0_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op,
           !transform.any_op, !transform.any_op) -> !transform.any_op

    // Step 5. Apply rolling update again, this time with the second matmul being the reduction.
    %mm2, %elemwise_1 =
      transform.fusion.find_next_reduction %forall_loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %elemwise_sidecars_1 =
      transform.fusion.clone_fuse_elemwise %elemwise_1 into %forall_loop, %j0_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    %reduce_r = transform.fusion.repair_reduction_frontier
        (%fused_bsum, %mm2) and (%elemwise_1, %elemwise_sidecars_1) into %forall_loop, %j0_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op,
           !transform.any_op, !transform.any_op) -> !transform.any_op

    // Step 6. Fuse the trailing FP32->FP16 cast into the forall loop (but outside the for loop).
    // Canonicalize first to reduce the number of outputs from the forall loop.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !transform.any_op
    %fused_trunc = transform.include @fuse_elemwise_consumer_of_result0 failures(propagate)
        (%forall_loop) : (!transform.any_op) -> (!transform.any_op)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}
