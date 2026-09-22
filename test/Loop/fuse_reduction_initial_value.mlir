// RUN: %neptune-opt %s --transform-interpreter | FileCheck %s

// A finite running maximum prevents inf-inf in an online softmax whose early
// tiles mask every key for some rows. The mask's -inf value must not change.
// CHECK-LABEL: func.func @finite_running_max
// CHECK: arith.constant 0xFF800000 : f32
// CHECK: scf.forall
// CHECK: %[[FINITE:.*]] = arith.constant -3.40282347E+38 : f32
// CHECK: %[[INIT:.*]] = linalg.fill ins(%[[FINITE]] : f32)
// CHECK: scf.for {{.*}} iter_args({{.*}} = %[[INIT]])
// CHECK: arith.maximumf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %max = transform.structured.match ops{["linalg.generic"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %fused, %inner = transform.scf.fuse_reduction_into_forall %max into %loop
        {initial_value = 0xFF7FFFFF : f32}
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.verify %func : !transform.any_op
    transform.yield
  }

  func.func @finite_running_max(%input: tensor<4x8xf32>) -> tensor<4xf32> {
    %neg_inf = arith.constant 0xFF800000 : f32
    %empty = tensor.empty() : tensor<4x8xf32>
    %scores = scf.forall (%i, %j) in (2, 2) shared_outs(%out = %empty) -> tensor<4x8xf32> {
      %row = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
      %col = affine.apply affine_map<(d0) -> (d0 * 4)>(%j)
      %tile = tensor.extract_slice %input[%row, %col] [2, 4] [1, 1]
          : tensor<4x8xf32> to tensor<2x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%row, %col] [2, 4] [1, 1]
            : tensor<2x4xf32> into tensor<4x8xf32>
      }
    }
    %row_empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%neg_inf : f32) outs(%row_empty : tensor<4xf32>) -> tensor<4xf32>
    %max = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%scores : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%x: f32, %acc: f32):
        %m = arith.maximumf %x, %acc : f32
        linalg.yield %m : f32
    } -> tensor<4xf32>
    return %max : tensor<4xf32>
  }
}
