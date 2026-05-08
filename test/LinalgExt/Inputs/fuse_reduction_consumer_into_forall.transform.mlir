module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op

    %scale = transform.structured.match ops{["linalg.map"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %scale_tiled, %forall_loop =
      transform.structured.tile_using_forall %scale tile_sizes [64, 64]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %row_max = transform.structured.match ops{["linalg.generic"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %fused, %new_forall, %new_for =
      transform.loop.fuse_reduction_consumer_into_forall %row_max into %forall_loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}
