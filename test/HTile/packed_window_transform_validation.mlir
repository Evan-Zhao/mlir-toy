// RUN: not neptune-opt --transform-interpreter --split-input-file %s 2>&1 | FileCheck %s

// CHECK: expected window element shape to match destination element shape
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

  func.func @invalid_insert_shape(%windows: tensor<2x3x5xf16>,
                                  %destination: tensor<8x4xf16>,
                                  %starts: tensor<2xi32>,
                                  %lengths: tensor<2xi32>) -> tensor<8x4xf16> {
    %result = stablehlo.custom_call @neptune.packed_window_insert(
        %windows, %destination, %starts, %lengths)
        : (tensor<2x3x5xf16>, tensor<8x4xf16>, tensor<2xi32>, tensor<2xi32>)
        -> tensor<8x4xf16>
    return %result : tensor<8x4xf16>
  }
}

// -----

// CHECK: expected packed, windows, and scalar other element types to match
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %extracts = transform.structured.match
        ops{["stablehlo.custom_call"]}
        attributes {call_target_name = "neptune.packed_window_extract"}
        in %module : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.for"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loads = transform.htile.fuse_packed_window_extract %extracts into %loops
        : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @invalid_extract_other(%packed: tensor<8x4xf16>,
                                   %starts: tensor<2xi32>,
                                   %lengths: tensor<2xi32>,
                                   %other: tensor<f32>) -> tensor<2x3x4xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
    }
    %windows = stablehlo.custom_call @neptune.packed_window_extract(
        %packed, %starts, %lengths, %other)
        : (tensor<8x4xf16>, tensor<2xi32>, tensor<2xi32>, tensor<f32>)
        -> tensor<2x3x4xf16>
    return %windows : tensor<2x3x4xf16>
  }
}
