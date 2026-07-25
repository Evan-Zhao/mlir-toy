// RUN: neptune-opt %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %slice = transform.structured.match ops{["stablehlo.slice"]} in %func
      : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %func {
      transform.apply_conversion_patterns.stablehlo.slice_to_tensor
    } {illegal_ops = ["stablehlo.slice"], legal_dialects = ["tensor"],
       partial_conversion, preserve_handles} : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @slice
  // CHECK-SAME: %[[ARG:.*]]: tensor<9xi32>
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG]][0] [8] [1]
  // CHECK: return %[[SLICE]]
  func.func @slice(%arg: tensor<9xi32>) -> tensor<8xi32> {
    %result = stablehlo.slice %arg [0:8]
        : (tensor<9xi32>) -> tensor<8xi32>
    return %result : tensor<8xi32>
  }
}
