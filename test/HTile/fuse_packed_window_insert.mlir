// RUN: neptune-opt %s --transform-interpreter --split-input-file --verify-diagnostics | FileCheck %s


// CHECK-LABEL: func.func @fuse_packed_window_insert(
// CHECK-NOT: stablehlo.custom_call
// CHECK: scf.forall
// CHECK: %[[START_SLICE:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}] [1] [1]
// CHECK-SAME: tensor<2xi32> to tensor<i32>
// CHECK: tensor.extract %[[START_SLICE]][] : tensor<i32>
// CHECK: %[[LENGTH_SLICE:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}] [1] [1]
// CHECK-SAME: tensor<2xi32> to tensor<i32>
// CHECK: tensor.extract %[[LENGTH_SLICE]][] : tensor<i32>
// CHECK: arith.addi
// CHECK: %[[MASK:.*]] = linalg.generic
// CHECK: arith.cmpi slt
// CHECK: %[[SOURCE:.*]] = tensor.collapse_shape
// CHECK: htile.masked_parallel_insert_slice %[[SOURCE]] into %{{.*}}[%{{.*}}, 0]
// CHECK-SAME: [4, 3] [1, 1]
// CHECK-SAME: mask(%[[MASK]] : tensor<4x3xi1>)
// CHECK-SAME: tensor<4x3xf32> into tensor<8x3xf32>

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
    transform.yield
  }

  func.func @fuse_packed_window_insert(
      %tile: tensor<1x4x3xf32>, %destination: tensor<8x3xf32>,
      %starts: tensor<2xi32>, %lengths: tensor<2xi32>) -> tensor<8x3xf32> {
    %empty = tensor.empty() : tensor<2x4x3xf32>
    %windows = scf.forall (%document) in (2) shared_outs(%out = %empty)
        -> tensor<2x4x3xf32> {
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%document, 0, 0] [1, 4, 3] [1, 1, 1]
            : tensor<1x4x3xf32> into tensor<2x4x3xf32>
      }
    }
    %result = stablehlo.custom_call @neptune.packed_window_insert(
        %windows, %destination, %starts, %lengths)
        : (tensor<2x4x3xf32>, tensor<8x3xf32>, tensor<2xi32>, tensor<2xi32>)
        -> tensor<8x3xf32>
    return %result : tensor<8x3xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %insert = transform.structured.match
        ops{["stablehlo.custom_call"]}
        attributes {call_target_name = "neptune.packed_window_insert"}
        in %module : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %module
        : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{invalid packed-window insertion custom call}}
    %masked_insert = transform.htile.fuse_packed_window_insert %insert into %forall
        : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @invalid_insert_shape(%windows: tensor<2x3x5xf16>,
                                  %destination: tensor<8x4xf16>,
                                  %starts: tensor<2xi32>,
                                  %lengths: tensor<2xi32>) -> tensor<8x4xf16> {
    // expected-error @below {{expected window element shape to match destination element shape}}
    %result = stablehlo.custom_call @neptune.packed_window_insert(
        %windows, %destination, %starts, %lengths)
        : (tensor<2x3x5xf16>, tensor<8x4xf16>, tensor<2xi32>, tensor<2xi32>)
        -> tensor<8x4xf16>
    return %result : tensor<8x4xf16>
  }
}
