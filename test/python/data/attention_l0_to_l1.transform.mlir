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
    %consumers = transform.get_consumers_of_result %forall_loop[0]
        : (!transform.any_op) -> !transform.any_op
    %_1, %bscale = transform.foreach_match restrict_root in %consumers
        @match_elemwise -> @return_matched
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_bscale =
      transform.loop.fuse_into_producer_op %bscale into %forall_loop
        : (!transform.any_op, !transform.any_op) -> !transform.any_op
    // Fusion can create redundant loop-carried values and canonicalization removes them.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !transform.any_op

    // Step 3. Match a reduction op that is a consumer of bscale.
    // This would be the row-max op in attention softmax.
    // Because of how MLIR scf.for and linalg work, this fusion implicitly also rfactors the reduction.
    //   TVM: sch.reverse_compute_at(bsum, j0); sch.rfactor(...)
    %consumers_1 = transform.get_consumers_of_result %forall_loop[0]
        : (!transform.any_op) -> !transform.any_op
    %_2, %bsum = transform.foreach_match restrict_root in %consumers_1
        @match_unary_reduction -> @return_matched
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_bsum, %j0_loop =
      transform.loop.fuse_reduction_consumer_into_forall %bsum into %forall_loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    // This CSE is actually required here so the next `fuse_elemwise_into_producer` can work
    // because it unifies identical affine map operations.
    transform.apply_cse to %func : !transform.any_op

    // Step 4. Prepare for rolling update. Rolling update can fuse two reductions together,
    // including all element-wise ops between them.
    // First use rolling_update_next_reduction to find the next reduction op.
    %reduce, %elemwise =
      transform.match.loop_ru.rolling_update_next_reduction %forall_loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // rolling_update_force_fuse_elemwise fuses all element-wise ops under %forall_loop,
    // then under %j0_loop.
    %elemwise_sidecars =
      transform.loop_ru.clone_fuse_elemwise %elemwise into %forall_loop, %j0_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    // Don't do canonicalization here -- we have intentionally unused results.
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}
