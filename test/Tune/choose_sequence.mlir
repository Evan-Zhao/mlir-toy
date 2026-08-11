// RUN: neptune-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @configure_a(%target: !transform.any_op) {
    transform.yield
  }

  transform.named_sequence @configure_b(%target: !transform.any_op) {
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.tune.choose_sequence "strategy" %root
        default = "a" cases = {a = @configure_a, b = @configure_b}
        : (!transform.any_op) -> ()
    transform.yield
  }
}

// CHECK: transform.tune.choose_sequence "strategy" %{{[^ ]+}}
// CHECK-SAME: default = "a"
// CHECK-SAME: cases = {a = @configure_a, b = @configure_b}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @configure(%target: !transform.any_op) {
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // expected-error@+1 {{has default case 'missing' that is not present in cases}}
    transform.tune.choose_sequence "backend" %root
        default = "missing" cases = {triton = @configure}
        : (!transform.any_op) -> ()
    transform.yield
  }
}
