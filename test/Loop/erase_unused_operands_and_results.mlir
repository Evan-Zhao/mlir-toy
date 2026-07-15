// RUN: neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @drop_unused_argmax_like_result
// CHECK: %[[REDUCE:.*]] = linalg.generic
// CHECK-SAME: outs(%{{.*}} : tensor<4xf32>)
// CHECK: } -> tensor<4xf32>
// CHECK-NOT: -> (tensor<4xf32>, tensor<4xi64>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %reduce = transform.structured.match ops{["linalg.generic"]} attributes {target} in %func
        : (!transform.any_op) -> !transform.any_op
    transform.linalg.erase_unused_operands_and_results %reduce : !transform.any_op
    transform.yield
  }

  func.func @drop_unused_argmax_like_result(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
    %neg_inf = arith.constant -3.40282347E+38 : f32
    %c0_i64 = arith.constant 0 : i64

    %max_empty = tensor.empty() : tensor<4xf32>
    %max_init = linalg.fill ins(%neg_inf : f32) outs(%max_empty : tensor<4xf32>) -> tensor<4xf32>
    %idx_empty = tensor.empty() : tensor<4xi64>
    %idx_init = linalg.fill ins(%c0_i64 : i64) outs(%idx_empty : tensor<4xi64>) -> tensor<4xi64>

    %reduce:2 = linalg.generic {target,
        indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i)>,
          affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%arg0 : tensor<4x8xf32>)
        outs(%max_init, %idx_init : tensor<4xf32>, tensor<4xi64>) {
      ^bb0(%in: f32, %out: f32, %out_idx: i64):
        %j = linalg.index 1 : index
        %j_i64 = arith.index_cast %j : index to i64
        %new_max = arith.maximumf %in, %out : f32
        %take_new = arith.cmpf ogt, %in, %out : f32
        %new_idx = arith.select %take_new, %j_i64, %out_idx : i64
        linalg.yield %new_max, %new_idx : f32, i64
    } -> (tensor<4xf32>, tensor<4xi64>)

    return %reduce#0 : tensor<4xf32>
  }
}
