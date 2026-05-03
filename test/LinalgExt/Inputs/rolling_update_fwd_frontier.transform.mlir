module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %elemwises, %reduces =
      transform.match.linalg_ext.rolling_update_fwd_frontier %loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %elemwises : !transform.any_op
    transform.print %reduces : !transform.any_op
    transform.yield
  }
}
