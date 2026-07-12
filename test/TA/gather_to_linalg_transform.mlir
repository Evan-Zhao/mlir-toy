// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin --transform-interpreter %s | FileCheck %s

// Verify that the fine-grained conversion lowers only gather and updates its
// pre-existing transform handle to the replacement linalg.generic.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %gather = transform.structured.match ops{["stablehlo.gather"]} in %func
      : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %func {
      transform.apply_conversion_patterns.stablehlo.gather_to_linalg
    } {illegal_ops = ["stablehlo.gather"], legal_dialects = ["arith", "linalg", "tensor"],
       partial_conversion, preserve_handles} : !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall
      %gather tile_sizes [1, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  // CHECK-LABEL: func.func @gather
  // CHECK: scf.forall
  // CHECK: linalg.generic
  // CHECK: tensor.extract %[[SOURCE:.*]][
  func.func @gather(%source: tensor<8x4xf32>, %indices: tensor<2x1xi32>)
      -> tensor<2x4xf32> {
    %result = "stablehlo.gather"(%source, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1], collapsed_slice_dims = [0],
        start_index_map = [0], index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 4>
    }> : (tensor<8x4xf32>, tensor<2x1xi32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}
