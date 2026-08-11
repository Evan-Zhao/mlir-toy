// RUN: neptune-opt %s --transform-interpreter --split-input-file --verify-diagnostics | FileCheck %s

module attributes {transform.with_named_sequence} {
  func.func @payload() {
    return
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %choice = transform.tune.sample_categorical "test.tile_size"
        candidates = [32, 64, 128] default = 64 : !transform.param<i64>
    transform.annotate %root "test.tile_size" = %choice
        : !transform.any_op, !transform.param<i64>
    transform.yield
  }
}

// CHECK: module attributes
// CHECK-SAME: test.tile_size = 64 : i64

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // expected-error@+1 {{has default value 64 that is not in the candidate list}}
    %choice = transform.tune.sample_categorical "tile"
        candidates = [32, 128] default = 64 : !transform.param<i64>
    transform.yield
  }
}
