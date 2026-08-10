// RUN: neptune-opt %s --split-input-file --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    transform.fusion.greedy_consumers_into_producer %loop[0]
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @collapse_forall_result(%init: tensor<2x2x8xf32>,
                                    %tiles: tensor<2x2x4xf32>) -> tensor<4x8xf32> {
    %result = scf.forall (%group, %head, %tile) in (2, 2, 2)
        shared_outs(%out = %init) -> tensor<2x2x8xf32> {
      %offset = affine.apply affine_map<(d0) -> (d0 * 4)>(%tile)
      %slice = tensor.extract_slice %tiles[%group, %head, 0] [1, 1, 4] [1, 1, 1]
          : tensor<2x2x4xf32> to tensor<1x1x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice into %out[%group, %head, %offset] [1, 1, 4]
            [1, 1, 1] : tensor<1x1x4xf32> into tensor<2x2x8xf32>
      }
    }
    %collapsed = tensor.collapse_shape %result [[0, 1], [2]]
        : tensor<2x2x8xf32> into tensor<4x8xf32>
    return %collapsed : tensor<4x8xf32>
  }
}

// CHECK-LABEL: func.func @collapse_forall_result
// CHECK: %[[INIT:.*]] = tensor.collapse_shape %arg0 {{.*}}
// CHECK: %[[RESULT:.*]] = scf.forall (%[[GROUP:.*]], %[[HEAD:.*]], %[[TILE:.*]]) in (2, 2, 2)
// CHECK-SAME: shared_outs(%[[OUT:.*]] = %[[INIT]]) -> (tensor<4x8xf32>)
// CHECK: %[[FLAT:.*]] = affine.linearize_index disjoint [%[[GROUP]], %[[HEAD]]] by (2, 2)
// CHECK: %[[TILE4:.*]] = tensor.collapse_shape %{{.*}} {{.*}} : tensor<1x1x4xf32> into tensor<1x4xf32>
// CHECK: tensor.parallel_insert_slice %[[TILE4]] into %[[OUT]][%[[FLAT]], %{{.*}}] [1, 4]
// CHECK: return %[[RESULT]] : tensor<4x8xf32>

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    transform.fusion.greedy_consumers_into_producer %loop[0]
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @collapse_contiguous_non_unit_tile(%init: tensor<2x8xf32>,
                                                %tiles: tensor<2x8xf32>) -> tensor<16xf32> {
    %result = scf.forall (%row, %tile) in (2, 2)
        shared_outs(%out = %init) -> tensor<2x8xf32> {
      %offset = affine.apply affine_map<(d0) -> (d0 * 4)>(%tile)
      %slice = tensor.extract_slice %tiles[%row, %offset] [1, 4] [1, 1]
          : tensor<2x8xf32> to tensor<1x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice into %out[%row, %offset] [1, 4] [1, 1]
            : tensor<1x4xf32> into tensor<2x8xf32>
      }
    }
    %collapsed = tensor.collapse_shape %result [[0, 1]]
        : tensor<2x8xf32> into tensor<16xf32>
    return %collapsed : tensor<16xf32>
  }
}

// CHECK-LABEL: func.func @collapse_contiguous_non_unit_tile
// CHECK: %[[INIT:.*]] = tensor.collapse_shape %arg0 {{.*}}
// CHECK: %[[RESULT:.*]] = scf.forall (%[[ROW:.*]], %[[TILE:.*]]) in (2, 2)
// CHECK-SAME: shared_outs(%[[OUT:.*]] = %[[INIT]]) -> (tensor<16xf32>)
// CHECK: %[[OFFSET:.*]] = affine.apply
// CHECK: %[[FLAT:.*]] = affine.linearize_index disjoint [%[[ROW]], %[[OFFSET]]] by (2, 8)
// CHECK: %[[TILE4:.*]] = tensor.collapse_shape %{{.*}} {{.*}} : tensor<1x4xf32> into tensor<4xf32>
// CHECK: tensor.parallel_insert_slice %[[TILE4]] into %[[OUT]][%[[FLAT]]] [4] [1]
// CHECK: return %[[RESULT]] : tensor<16xf32>
