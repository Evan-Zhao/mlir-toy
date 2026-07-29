// RUN: neptune-opt %s -o /dev/null

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %insert = transform.structured.match
        ops{["stablehlo.custom_call"]}
        attributes {call_target_name = "neptune.packed_window_insert"}
        in %module : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %masked_insert = transform.htile.fuse_packed_window_insert %insert into %forall
        : (!transform.any_op, !transform.any_op) -> !transform.any_op

    %extracts = transform.structured.match
        ops{["stablehlo.custom_call"]}
        attributes {call_target_name = "neptune.packed_window_extract"}
        in %module : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.for", "scf.forall"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loads = transform.htile.fuse_packed_window_extract %extracts into %loops
        : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
