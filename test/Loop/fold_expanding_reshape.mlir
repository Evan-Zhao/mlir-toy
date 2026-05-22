// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @restore_batch_dims_on_transpose
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK: ins(%arg0 : tensor<1x4x8x16xf32>)
// CHECK: tensor<1x4x16x8xf32>
// CHECK-NOT: linalg.transpose

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %transpose = transform.structured.match ops{["linalg.transpose"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.generalize %transpose : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func { transform.apply_patterns.linalg.fold_expanding_reshape } : !transform.any_op
    transform.yield
  }

  func.func @restore_batch_dims_on_transpose(%arg0: tensor<1x4x8x16xf32>) -> tensor<1x4x16x8xf32> {
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]]
        : tensor<1x4x8x16xf32> into tensor<4x8x16xf32>
    %out = tensor.empty() : tensor<4x16x8xf32>
    %transpose = linalg.transpose
        ins(%collapsed : tensor<4x8x16xf32>)
        outs(%out : tensor<4x16x8xf32>)
        permutation = [0, 2, 1]
    %expanded = tensor.expand_shape %transpose [[0, 1], [2], [3]] output_shape [1, 4, 16, 8]
        : tensor<4x16x8xf32> into tensor<1x4x16x8xf32>
    return %expanded : tensor<1x4x16x8xf32>
  }
}
