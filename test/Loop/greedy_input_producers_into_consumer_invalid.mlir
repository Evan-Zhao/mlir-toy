// RUN: not neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s

// CHECK: error: expected consumer loops in strictly nested outer-to-inner order

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %for = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %inner_to_outer = transform.merge_handles %for, %forall
        : !transform.any_op
    %fused = transform.fusion.greedy_input_producers_into_consumer %inner_to_outer
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @wrong_loop_order() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.forall (%i) in (1) {
      scf.for %j = %c0 to %c1 step %c1 {
      }
    }
    return
  }
}
