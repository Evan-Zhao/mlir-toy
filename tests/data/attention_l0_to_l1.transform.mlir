// Transform-dialect schedule that lowers attention_l0.mlir into the shape of
// flash_attention_l1.mlir. Mirrors `_schedule_attention_flash` from Neptune.
//
// Usage:
//   mlir-opt tests/data/attention_l0.mlir \
//     --load-dialect-plugin=build/libLinalgExtTransformPlugin.dylib \
//     --transform-preload-library=transform-library-paths=tests/data/attention_l0_to_l1.transform.mlir \
//     --transform-interpreter

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_unary_elementwise(
      %candidate: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %matched = transform.match.structured %candidate
        : (!transform.any_op) -> !transform.any_op {
    ^bb0(%op: !transform.any_op):
      transform.match.structured.body %op {elementwise} : !transform.any_op

      %num_inputs = transform.match.structured.num_inputs %op
          : (!transform.any_op) -> !transform.param<i64>
      %one = transform.param.constant 1 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %num_inputs, %one : !transform.param<i64>

      transform.match.structured.yield %op : !transform.any_op
    }
    transform.yield %matched : !transform.any_op
  }

  transform.named_sequence @__transform_main(
      %module: !transform.any_op) {

    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op

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

    // ============================================================
    // Step 1. Tile matmul1 outer iteration space.
    //   TVM: i0, j0 = tile_loops(sch, [i, j], decisions=[128, 128])
    //        bind_block_idx(sch, [*axes, i0])
    //
    //   Iteration order on b0 is (b, h, i, j, k).
    //   - (b, h, i) -> outer parallel scf.forall  (the eventual block grid;
    //                                              no #gpu mapping at L1).
    //   - j        -> sequential scf.for          (the K/V-block streaming
    //                                              loop).
    //   - k        -> stays as the matmul's inner reduction (un-tiled;
    //                  head dim = 128 fits).
    // ============================================================
    %b0_outer, %forall_loop =
      transform.structured.tile_using_forall %b0
          tile_sizes [1, 1, 128, 0, 0]
        : (!transform.any_op)
       -> (!transform.any_op, !transform.any_op)
    %b0_inner, %j0_loop =
      transform.structured.tile_using_for %b0_outer
          tile_sizes [0, 0, 0, 128, 0]
        : (!transform.any_op)
       -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 2. Match element-wise ops that consume b0, and fuse them into the j0 loop.
    // (For attention, this step would only match the score-scaling op.)
    //   TVM: sch.reverse_compute_at(b1, j0)
    // ============================================================
    %consumers = transform.get_consumers_of_result %forall_loop[0]
        : (!transform.any_op) -> !transform.any_op
    %score_ops = transform.collect_matching @match_unary_elementwise in %consumers
        : (!transform.any_op) -> !transform.any_op
    %b1_fused, %j0_loop_1 =
      transform.linalg_ext.fuse_unary_elementwise_consumer_into_loop %score_ops into %j0_loop
        : (!transform.any_op, !transform.any_op)
       -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 7. Canonicalize + CSE.
    // ============================================================
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}
