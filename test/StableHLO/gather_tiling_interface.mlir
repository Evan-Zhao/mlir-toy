// RUN: neptune-opt --transform-interpreter --split-input-file %s | FileCheck %s

// Verify that stablehlo.gather can be tile-and-fused directly into a tiled
// consumer without first lowering the gather to linalg.generic.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %gather = transform.structured.match ops{["stablehlo.gather"]} in %func
      : (!transform.any_op) -> !transform.any_op
    %consumer = transform.structured.match ops{["linalg.generic"]} in %func
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall
      %consumer tile_sizes [1, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused, %new_loop = transform.structured.fuse_into_containing_op
      %gather into %loop
      : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  // CHECK-LABEL: func.func @fuse_gather
  // CHECK: scf.forall
  // CHECK: tensor.extract_slice %[[INDICES:.*]][{{.*}}] [1, 1]
  // CHECK: tensor.extract_slice %[[SOURCE:.*]][0, {{.*}}] [8, 4]
  // CHECK: stablehlo.gather
  // CHECK-SAME: slice_sizes = array<i64: 1, 4>
  func.func @fuse_gather(%source: tensor<8x8xf32>,
                         %indices: tensor<2x1xi32>) -> tensor<2x8xf32> {
    %gathered = "stablehlo.gather"(%source, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1], collapsed_slice_dims = [0],
        start_index_map = [0], index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 8>
    }> : (tensor<8x8xf32>, tensor<2x1xi32>) -> tensor<2x8xf32>
    %empty = tensor.empty() : tensor<2x8xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%gathered : tensor<2x8xf32>)
        outs(%empty : tensor<2x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<2x8xf32>
    return %result : tensor<2x8xf32>
  }
}

// -----

// Verify operand/start-index batching dimensions as used by sparse attention.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %gather = transform.structured.match ops{["stablehlo.gather"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall
      %gather tile_sizes [1, 2, 3, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  // CHECK-LABEL: func.func @batched_gather
  // CHECK: scf.forall
  // CHECK: tensor.extract_slice %[[INDICES:.*]][{{.*}}] [1, 2, 3, 1]
  // CHECK: tensor.extract_slice %[[SOURCE:.*]][{{.*}}] [1, 16, 4]
  // CHECK: stablehlo.gather
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: slice_sizes = array<i64: 1, 1, 4>
  // CHECK-SAME: -> tensor<1x2x3x4xf32>
  func.func @batched_gather(%source: tensor<2x16x8xf32>,
                            %indices: tensor<2x4x6x1xi32>)
      -> tensor<2x4x6x8xf32> {
    %result = "stablehlo.gather"(%source, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [3], collapsed_slice_dims = [1],
        operand_batching_dims = [0], start_indices_batching_dims = [0],
        start_index_map = [1], index_vector_dim = 3>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1, 8>
    }> : (tensor<2x16x8xf32>, tensor<2x4x6x1xi32>)
        -> tensor<2x4x6x8xf32>
    return %result : tensor<2x4x6x8xf32>
  }
}
