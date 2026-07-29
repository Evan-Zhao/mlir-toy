// RUN: neptune-opt %s --split-input-file --transform-interpreter --verify-diagnostics | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %scatter = transform.structured.match ops{["stablehlo.scatter"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %update_tile = transform.structured.match ops{["tensor.extract_slice"]} in %forall
        : (!transform.any_op) -> !transform.any_op
    %masked_insert_slice =
        transform.htile.fuse_oob_sink_scatter_as_masked_insert_slice %scatter into %forall
        : (!transform.any_op, !transform.any_op) -> !transform.any_op
    // Handles to the original loop and its cloned body remain valid.
    transform.match.operation_name %forall ["scf.forall"] : !transform.any_op
    transform.match.operation_name %update_tile ["tensor.extract_slice"]
        : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @candidate
  // CHECK: %[[PREPARED_INIT:.+]] = tensor.cast %arg0
  // CHECK: %[[MASK:.+]] = arith.constant dense<[true, false]>
  // CHECK: %[[VALID_ROWS:.+]] = linalg.generic
  // CHECK-NOT: arith.select
  // CHECK-NOT: linalg.broadcast
  // CHECK: scf.forall
  // CHECK-SAME: shared_outs({{.*}} = %[[PREPARED_INIT]])
  // CHECK: %[[SOURCE:.+]] = tensor.extract_slice %arg1
  // CHECK: %[[MASK_TILE:.+]] = tensor.extract_slice %[[MASK]][%{{.*}}] [1] [1]
  // CHECK: %[[VALID_ROWS_TILE:.+]] = tensor.extract_slice %[[VALID_ROWS]][%{{.*}}] [1] [1]
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[ROW_I32:.+]] = tensor.extract %[[VALID_ROWS_TILE]][%[[C0]]]
  // CHECK: %[[ROW:.+]] = arith.index_cast %[[ROW_I32]] : i32 to index
  // CHECK: %[[MASK_INIT:.+]] = tensor.empty() : tensor<1x4xi1>
  // CHECK: %[[SOURCE_MASK:.+]] = linalg.broadcast ins(%[[MASK_TILE]] : tensor<1xi1>) outs(%[[MASK_INIT]] : tensor<1x4xi1>) dimensions = [1]
  // CHECK: htile.masked_parallel_insert_slice %[[SOURCE]] into %{{.*}}[%[[ROW]], 0] [1, 4] [1, 1] mask(%[[SOURCE_MASK]] : tensor<1x4xi1>)
  // CHECK-NOT: "stablehlo.scatter"
  // CHECK-NOT: tensor.cast
  func.func @candidate(
      %init: tensor<8x4xf32>, %update_values: tensor<2x4xf32>,
      %starts: tensor<1xi32>) -> tensor<8x4xf32> {
    %empty = tensor.empty() : tensor<2x4xf32>
    %c9_i32 = arith.constant 9 : i32
    %updates = scf.forall (%i) in (2) shared_outs(%out = %empty) -> tensor<2x4xf32> {
      %tile = tensor.extract_slice %update_values[%i, 0] [1, 4] [1, 1]
          : tensor<2x4xf32> to tensor<1x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%i, 0] [1, 4] [1, 1]
            : tensor<1x4xf32> into tensor<2x4xf32>
      }
    }
    // These definitions intentionally follow the forall. The transform moves
    // their pure definition chains before the loop.
    %late_init = tensor.cast %init : tensor<8x4xf32> to tensor<8x4xf32>
    %mask = arith.constant dense<[true, false]> : tensor<2xi1>
    %valid_empty = tensor.empty() : tensor<2xi32>
    %valid_rows = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%starts : tensor<1xi32>) outs(%valid_empty : tensor<2xi32>) {
    ^bb0(%start: i32, %out: i32):
      %i = linalg.index 0 : index
      %i_i32 = arith.index_cast %i : index to i32
      %row = arith.addi %start, %i_i32 : i32
      linalg.yield %row : i32
    } -> tensor<2xi32>
    %sink_empty = tensor.empty() : tensor<2xi32>
    %sink_rows = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        outs(%sink_empty : tensor<2xi32>) {
    ^bb0(%out: i32):
      %i = linalg.index 0 : index
      %i_i32 = arith.index_cast %i : index to i32
      // Exercise MLIR's integer-range model rather than a transform-local
      // whitelist of arithmetic operations. This produces sink rows [9, 8].
      %row = arith.subi %c9_i32, %i_i32 : i32
      linalg.yield %row : i32
    } -> tensor<2xi32>
    %selected_empty = tensor.empty() : tensor<2xi32>
    %selected_rows = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%mask, %valid_rows, %sink_rows
            : tensor<2xi1>, tensor<2xi32>, tensor<2xi32>)
        outs(%selected_empty : tensor<2xi32>) {
    ^bb0(%valid: i1, %row: i32, %sink: i32, %out: i32):
      %selected = arith.select %valid, %row, %sink : i32
      linalg.yield %selected : i32
    } -> tensor<2xi32>
    %indices_empty = tensor.empty() : tensor<2x1xi32>
    %indices = linalg.broadcast ins(%selected_rows : tensor<2xi32>)
        outs(%indices_empty : tensor<2x1xi32>) dimensions = [1]
    %result = "stablehlo.scatter"(%late_init, %indices, %updates) <{
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
    %masked_insert_slice =
        // expected-error @below {{expected scatter indices to select contiguous rows or provably OOB sink rows}}
        transform.htile.fuse_oob_sink_scatter_as_masked_insert_slice %scatter into %forall
        : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @not_ranged(
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
    %masked_insert_slice =
        // expected-error @below {{expected scatter to have unique_indices = true}}
        transform.htile.fuse_oob_sink_scatter_as_masked_insert_slice %scatter into %forall
        : (!transform.any_op, !transform.any_op) -> !transform.any_op
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
    %masked_insert_slice =
        // expected-error @below {{expected scatter update computation to directly return the update argument}}
        transform.htile.fuse_oob_sink_scatter_as_masked_insert_slice %scatter into %forall
        : (!transform.any_op, !transform.any_op) -> !transform.any_op
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
