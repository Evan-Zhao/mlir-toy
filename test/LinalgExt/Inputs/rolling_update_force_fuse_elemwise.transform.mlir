module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %reduce, %elemwise =
      transform.match.linalg_ext.rolling_update_next_reduction %loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %loop_2 = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %sidecar, %new_loop =
      transform.linalg_ext.rolling_update_force_fuse_elemwise %elemwise into %loop_2
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
