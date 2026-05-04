module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %f1 = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "linear_chain"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop1 = transform.structured.match ops{["scf.for"]} in %f1
        : (!transform.any_op) -> !transform.any_op
    %r1, %e1 = transform.match.linalg_ext.rolling_update_next_reduction %loop1
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %r1 : !transform.any_op
    transform.print %e1 : !transform.any_op

    %f2 = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "direct_reduce"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop2 = transform.structured.match ops{["scf.for"]} in %f2
        : (!transform.any_op) -> !transform.any_op
    %r2, %e2 = transform.match.linalg_ext.rolling_update_next_reduction %loop2
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %r2 : !transform.any_op
    transform.print %e2 : !transform.any_op

    %f3 = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "two_elemwise"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop3 = transform.structured.match ops{["scf.for"]} in %f3
        : (!transform.any_op) -> !transform.any_op
    %r3, %e3 = transform.match.linalg_ext.rolling_update_next_reduction %loop3
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %r3 : !transform.any_op
    transform.print %e3 : !transform.any_op

    %f4 = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "two_reductions"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop4 = transform.structured.match ops{["scf.for"]} in %f4
        : (!transform.any_op) -> !transform.any_op
    %r4, %e4 = transform.match.linalg_ext.rolling_update_next_reduction %loop4
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %r4 : !transform.any_op
    transform.print %e4 : !transform.any_op

    %f5 = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "attention_like"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop5 = transform.structured.match ops{["scf.for"]} in %f5
        : (!transform.any_op) -> !transform.any_op
    %r5, %e5 = transform.match.linalg_ext.rolling_update_next_reduction %loop5
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %r5 : !transform.any_op
    transform.print %e5 : !transform.any_op

    transform.yield
  }
}
