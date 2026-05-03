// Transform-dialect schedule that lowers attention_l0.mlir into the shape of
// flash_attention_l1.mlir. Mirrors `_schedule_attention_flash` from Neptune.
//
// Usage:
//   mlir-opt test/python/data/attention_l0.mlir \
//     --load-dialect-plugin=build/libLinalgExtTransform.so \
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

  // Fuse the consumer `c` of an op `op` if `c` only reads from `op` and nothing else.
  // Fails if there are zero or multiple such consumers.
  transform.named_sequence @fuse_up_elemwise_consumer(
      %producer_loop: !transform.any_op {transform.consumed}
  ) -> (!transform.any_op, !transform.any_op) {
    %consumers = transform.get_consumers_of_result %producer_loop[0]
        : (!transform.any_op) -> !transform.any_op
    %consumers_1 = transform.collect_matching @match_elemwise in %consumers
        : (!transform.any_op) -> !transform.any_op
    // Fail if there are zero or multiple such consumers.
    %fused_consumer, %updated_loop =
      transform.linalg_ext.fuse_elemwise_into_producer %consumers_1 into %producer_loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield %fused_consumer, %updated_loop : !transform.any_op, !transform.any_op
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

  transform.named_sequence @fuse_up_reduction_consumer(
      %producer_loop: !transform.any_op {transform.consumed}
  ) -> (!transform.any_op, !transform.any_op, !transform.any_op) {
    %consumers = transform.get_consumers_of_result %producer_loop[0]
        : (!transform.any_op) -> !transform.any_op
    %consumers_1 = transform.collect_matching @match_unary_reduction in %consumers
        : (!transform.any_op) -> !transform.any_op
    %fused_consumer, %updated_loop, %inner_loop =
      transform.linalg_ext.fuse_reduction_consumer_into_forall %consumers_1 into %producer_loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield %fused_consumer, %updated_loop, %inner_loop
      : !transform.any_op, !transform.any_op, !transform.any_op
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op

    // Step 0. Decompose softmax into linalg ops.
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
    %bscale, %forall_loop_1 = transform.include @fuse_up_elemwise_consumer
        failures(propagate) (%forall_loop) : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    // Fusion can create redundant loop-carried values, so we apply canonicalization
    // to remove them.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !transform.any_op

    // Step 3. Match a reduction op that is a consumer of bscale.
    // This would be the row-max op in attention softmax.
    // Because of how MLIR scf.for and linalg work, this fusion implicitly also rfactors the reduction.
    //   TVM: sch.reverse_compute_at(bsum, j0); sch.rfactor(...)
    %bsum, %forall_loop_2, %j0_loop = transform.include @fuse_up_reduction_consumer
        failures(propagate) (%forall_loop_1) : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !transform.any_op

    // Step 4. Prepare for rolling update. Rolling update can fuse two reductions together,
    // including all element-wise ops between them.
    // There is an analysis that returns which elemwise ops and reduction ops will be affected in this process.
    %elemwises, %reduces =
      transform.match.linalg_ext.rolling_update_fwd_frontier %forall_loop_2
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %first_reduce, %_1 = transform.split_handle %reduces { overflow_result = 1 : i64 }
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // TODO: implement rolling update transforms and use them here.

    // Step 7. Canonicalize + CSE.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}
