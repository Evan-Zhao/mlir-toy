// RUN: neptune-opt %s --split-input-file --transform-interpreter --verify-diagnostics | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %scatter = transform.structured.match ops{["stablehlo.scatter"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %new_forall, %parallel_scatter =
        transform.htile.fuse_scatter_into_forall %scatter into %forall
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  // CHECK-LABEL: func.func @candidate
  // CHECK: %[[PREPARED_INIT:.+]] = tensor.cast %arg0
  // CHECK: %[[PREPARED_INDICES:.+]] = tensor.cast %arg2
  // CHECK: scf.forall
  // CHECK-SAME: shared_outs({{.*}} = %[[PREPARED_INIT]])
  // CHECK: %[[INDEX_TILE:.+]] = tensor.extract_slice %[[PREPARED_INDICES]]
  // CHECK-NOT: linalg.generic
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: htile.parallel_scatter %{{.*}} into %{{.*}}[%[[INDEX_TILE]], %[[C0]]] broadcast_dims([1]) unique out_of_bounds <discard>
  // CHECK-NOT: "stablehlo.scatter"
  // CHECK-NOT: tensor.cast
  func.func @candidate(
      %init: tensor<8x4xf32>, %update_values: tensor<2x4xf32>,
      %indices: tensor<2x1xi32>) -> tensor<8x4xf32> {
    %empty = tensor.empty() : tensor<2x4xf32>
    %updates = scf.forall (%i) in (2) shared_outs(%out = %empty) -> tensor<2x4xf32> {
      %tile = tensor.extract_slice %update_values[%i, 0] [1, 4] [1, 1]
          : tensor<2x4xf32> to tensor<1x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%i, 0] [1, 4] [1, 1]
            : tensor<1x4xf32> into tensor<2x4xf32>
      }
    }
    // These definitions intentionally follow the forall. The transform clones
    // their pure definition chains before the loop.
    %late_init = tensor.cast %init : tensor<8x4xf32> to tensor<8x4xf32>
    %late_indices = tensor.cast %indices : tensor<2x1xi32> to tensor<2x1xi32>
    %result = "stablehlo.scatter"(%late_init, %late_indices, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
          update_window_dims = [1],
          inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0],
          index_vector_dim = 1>,
      unique_indices = true
    }> ({
    ^bb0(%old: tensor<f32>, %update: tensor<f32>):
      stablehlo.return %update : tensor<f32>
    }) : (tensor<8x4xf32>, tensor<2x1xi32>, tensor<2x4xf32>) -> tensor<8x4xf32>
    return %result : tensor<8x4xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %scatter = transform.structured.match ops{["stablehlo.scatter"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %new_forall, %parallel_scatter =
        // expected-error @below {{expected scatter to have unique_indices = true}}
        transform.htile.fuse_scatter_into_forall %scatter into %forall
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @non_unique(
      %init: tensor<8xf32>, %update_values: tensor<2xf32>,
      %indices: tensor<2x1xi32>) -> tensor<8xf32> {
    %empty = tensor.empty() : tensor<2xf32>
    %updates = scf.forall (%i) in (2) shared_outs(%out = %empty) -> tensor<2xf32> {
      %tile = tensor.extract_slice %update_values[%i] [1] [1]
          : tensor<2xf32> to tensor<1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%i] [1] [1]
            : tensor<1xf32> into tensor<2xf32>
      }
    }
    %result = "stablehlo.scatter"(%init, %indices, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
          inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0],
          index_vector_dim = 1>,
      unique_indices = false
    }> ({
    ^bb0(%old: tensor<f32>, %update: tensor<f32>):
      stablehlo.return %update : tensor<f32>
    }) : (tensor<8xf32>, tensor<2x1xi32>, tensor<2xf32>) -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %scatter = transform.structured.match ops{["stablehlo.scatter"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %new_forall, %parallel_scatter =
        // expected-error @below {{expected scatter update computation to directly return the update argument}}
        transform.htile.fuse_scatter_into_forall %scatter into %forall
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @combining_update(
      %init: tensor<8xf32>, %update_values: tensor<2xf32>,
      %indices: tensor<2x1xi32>) -> tensor<8xf32> {
    %empty = tensor.empty() : tensor<2xf32>
    %updates = scf.forall (%i) in (2) shared_outs(%out = %empty) -> tensor<2xf32> {
      %tile = tensor.extract_slice %update_values[%i] [1] [1]
          : tensor<2xf32> to tensor<1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%i] [1] [1]
            : tensor<1xf32> into tensor<2xf32>
      }
    }
    %result = "stablehlo.scatter"(%init, %indices, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
          inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0],
          index_vector_dim = 1>,
      unique_indices = true
    }> ({
    ^bb0(%old: tensor<f32>, %update: tensor<f32>):
      %sum = stablehlo.add %old, %update : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) : (tensor<8xf32>, tensor<2x1xi32>, tensor<2xf32>) -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
