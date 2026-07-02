// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter | FileCheck %s

// CHECK: IR printer
// CHECK: arith.maxnumf
// CHECK: IR printer
// CHECK: arith.addf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op

    %max, %max_elemwise = transform.fusion.find_next_reduction %loop[0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %max : !transform.any_op

    %sum, %sum_elemwise = transform.fusion.find_next_reduction %loop[1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %sum : !transform.any_op

    transform.yield
  }

  func.func @two_result_loop(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %empty0 = tensor.empty() : tensor<4x4xf32>
    %empty1 = tensor.empty() : tensor<4x4xf32>
    %scores:2 = scf.for %iv = %c0 to %c2 step %c1
        iter_args(%acc0 = %empty0, %acc1 = %empty1)
        -> (tensor<4x4xf32>, tensor<4x4xf32>) {
      %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%iv)
      %tile0 = tensor.extract_slice %arg0[0, %j] [4, 2] [1, 1]
          : tensor<4x4xf32> to tensor<4x2xf32>
      %inserted0 = tensor.insert_slice %tile0 into %acc0[0, %j] [4, 2] [1, 1]
          : tensor<4x2xf32> into tensor<4x4xf32>
      %tile1 = tensor.extract_slice %arg1[0, %j] [4, 2] [1, 1]
          : tensor<4x4xf32> to tensor<4x2xf32>
      %inserted1 = tensor.insert_slice %tile1 into %acc1[0, %j] [4, 2] [1, 1]
          : tensor<4x2xf32> into tensor<4x4xf32>
      scf.yield %inserted0, %inserted1 : tensor<4x4xf32>, tensor<4x4xf32>
    }

    %neg_inf = arith.constant -3.40282347E+38 : f32
    %max_empty = tensor.empty() : tensor<4xf32>
    %max_init = linalg.fill ins(%neg_inf : f32)
        outs(%max_empty : tensor<4xf32>) -> tensor<4xf32>
    %max = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%scores#0 : tensor<4x4xf32>) outs(%max_init : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %v = arith.maxnumf %in, %out : f32
        linalg.yield %v : f32
    } -> tensor<4xf32>

    %zero = arith.constant 0.0 : f32
    %sum_empty = tensor.empty() : tensor<4xf32>
    %sum_init = linalg.fill ins(%zero : f32)
        outs(%sum_empty : tensor<4xf32>) -> tensor<4xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%scores#1 : tensor<4x4xf32>) outs(%sum_init : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %v = arith.addf %in, %out : f32
        linalg.yield %v : f32
    } -> tensor<4xf32>

    return %max, %sum : tensor<4xf32>, tensor<4xf32>
  }
}
