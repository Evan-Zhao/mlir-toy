// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @keep_batch_drop_keepdim
// CHECK: %[[REDUCE:.*]] = linalg.generic
// CHECK-SAME: tensor<1x4x128x128xf32>
// CHECK-SAME: tensor<1x4x128xf32>
// CHECK: linalg.generic
// CHECK-SAME: tensor<1x4x128x128xf32>, tensor<1x4x128xf32>
// CHECK-NOT: -> (d0, d1, d2, 0)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.loop.fold_zero_indexed_unit_dims %func : !transform.any_op
    transform.yield
  }

  func.func @keep_batch_drop_keepdim(%arg0: tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32> {
    %f0 = arith.constant 0.0 : f32

    %sum_empty = tensor.empty() : tensor<1x4x128x1xf32>
    %sum_seed = linalg.fill ins(%f0 : f32) outs(%sum_empty : tensor<1x4x128x1xf32>)
        -> tensor<1x4x128x1xf32>
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0 : tensor<1x4x128x128xf32>)
        outs(%sum_seed : tensor<1x4x128x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %acc = arith.addf %in, %out : f32
        linalg.yield %acc : f32
    } -> tensor<1x4x128x1xf32>

    %out_empty = tensor.empty() : tensor<1x4x128x128xf32>
    %out = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%arg0, %sum : tensor<1x4x128x128xf32>, tensor<1x4x128x1xf32>)
        outs(%out_empty : tensor<1x4x128x128xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %out_elt: f32):
        %val = arith.divf %lhs, %rhs : f32
        linalg.yield %val : f32
    } -> tensor<1x4x128x128xf32>

    return %out : tensor<1x4x128x128xf32>
  }
}
