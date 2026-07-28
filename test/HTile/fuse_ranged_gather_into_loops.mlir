// RUN: neptune-opt --transform-interpreter --verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_ranged_gather
// CHECK-NOT: stablehlo.pad
// CHECK-NOT: stablehlo.gather
// CHECK: scf.forall
// CHECK: %[[START_I32:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK: %[[START:.*]] = arith.index_cast %[[START_I32]] : i32 to index
// CHECK: %[[MASK:.*]] = linalg.generic
// CHECK: linalg.index 0
// CHECK: arith.cmpi slt
// CHECK: %[[LOAD:.*]] = htile.load %[[SOURCE:.*]][%[[START]], %{{.*}}]
// CHECK-SAME: mask(%[[MASK]] : tensor<4x4xi1>)
// CHECK-SAME: other(%{{.*}} : f32)
// CHECK-SAME: tensor<16x4xf32> -> tensor<4x4xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[LOAD]]
// CHECK-SAME: tensor<4x4xf32> into tensor<1x4x4xf32>
// CHECK: tensor.parallel_insert_slice %[[EXPANDED]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %gathers = transform.structured.match ops{["stablehlo.gather"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.forall"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %loads = transform.htile.fuse_ranged_gather_into_loops %gathers into %loops
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @fuse_ranged_gather(%source: tensor<16x4xf32>,
                                %starts: tensor<2x1xi32>) -> tensor<2x4x4xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %padded = stablehlo.pad %source, %zero,
        low = [0, 0], high = [4, 0], interior = [0, 0]
        : (tensor<16x4xf32>, tensor<f32>) -> tensor<20x4xf32>
    %gather = "stablehlo.gather"(%padded, %starts) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 4, 4>
    }> : (tensor<20x4xf32>, tensor<2x1xi32>) -> tensor<2x4x4xf32>

    %empty = tensor.empty() : tensor<2x4x4xf32>
    %result = scf.forall (%i) in (2) shared_outs(%out = %empty) -> tensor<2x4x4xf32> {
      %tile = tensor.extract_slice %gather[%i, 0, 0] [1, 4, 4] [1, 1, 1]
          : tensor<2x4x4xf32> to tensor<1x4x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%i, 0, 0] [1, 4, 4] [1, 1, 1]
            : tensor<1x4x4xf32> into tensor<2x4x4xf32>
      }
    }
    return %result : tensor<2x4x4xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %gathers = transform.structured.match ops{["stablehlo.gather"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.forall"]} in %module
      : (!transform.any_op) -> !transform.any_op
    // Multiple dynamic start components still describe one rectangular box.
    %loads = transform.htile.fuse_ranged_gather_into_loops %gathers into %loops
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @multiple_start_components(%source: tensor<16x4x8xf32>,
                                       %indices: tensor<2x2xi32>)
      -> tensor<2x4x2x8xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %padded = stablehlo.pad %source, %zero,
        low = [0, 0, 0], high = [4, 2, 0], interior = [0, 0, 0]
        : (tensor<16x4x8xf32>, tensor<f32>) -> tensor<20x6x8xf32>
    %result = "stablehlo.gather"(%padded, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1, 2, 3], start_index_map = [0, 1], index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 4, 2, 8>
    }> : (tensor<20x6x8xf32>, tensor<2x2xi32>) -> tensor<2x4x2x8xf32>
    scf.forall (%i) in (2) {
      %tile = tensor.extract_slice %result[%i, 0, 0, 0] [1, 4, 2, 8] [1, 1, 1, 1]
          : tensor<2x4x2x8xf32> to tensor<1x4x2x8xf32>
      scf.forall.in_parallel {
      }
    }
    return %result : tensor<2x4x2x8xf32>
  }

  // Exercise dropping a collapsed operand dimension before inserting the
  // selected start-index batch dimension.
  func.func @collapsed_indexed_dimension(%source: tensor<16x8xf32>,
                                         %indices: tensor<2x1xi32>)
      -> tensor<2x8xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %padded = stablehlo.pad %source, %zero,
        low = [0, 0], high = [1, 0], interior = [0, 0]
        : (tensor<16x8xf32>, tensor<f32>) -> tensor<17x8xf32>
    %result = "stablehlo.gather"(%padded, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1], collapsed_slice_dims = [0],
        start_index_map = [0], index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 8>
    }> : (tensor<17x8xf32>, tensor<2x1xi32>) -> tensor<2x8xf32>
    scf.forall (%i) in (2) {
      %tile = tensor.extract_slice %result[%i, 0] [1, 8] [1, 1]
          : tensor<2x8xf32> to tensor<1x8xf32>
      scf.forall.in_parallel {
      }
    }
    return %result : tensor<2x8xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %gathers = transform.structured.match ops{["stablehlo.gather"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.forall"]} in %module
      : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{expected ranged gather operand to be produced by stablehlo.pad}}
    %loads = transform.htile.fuse_ranged_gather_into_loops %gathers into %loops
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @selected_loop() {
    scf.forall (%i) in (1) {
      scf.forall.in_parallel {
      }
    }
    return
  }

  // Exercise a non-leading indexed dimension, a non-trailing index-vector
  // dimension, and more than one start_indices batch dimension. These are
  // supported as long as each individual start index has one component.
  func.func @missing_pad(%source: tensor<16x4xf32>,
                         %indices: tensor<1x2x3xi32>) -> tensor<2x16x3x2xf32> {
    %result = "stablehlo.gather"(%source, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1, 3], start_index_map = [1], index_vector_dim = 0>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 16, 2>
    }> : (tensor<16x4xf32>, tensor<1x2x3xi32>) -> tensor<2x16x3x2xf32>
    return %result : tensor<2x16x3x2xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %gathers = transform.structured.match ops{["stablehlo.gather"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.forall"]} in %module
      : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{expected ranged gather to have at least one start-index component}}
    %loads = transform.htile.fuse_ranged_gather_into_loops %gathers into %loops
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @selected_loop() {
    scf.forall (%i) in (1) {
      scf.forall.in_parallel {
      }
    }
    return
  }

  func.func @no_start_components(%source: tensor<16x4xf32>,
                                 %indices: tensor<2x0xi32>) -> tensor<2x4x2xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %padded = stablehlo.pad %source, %zero,
        low = [0, 0], high = [4, 0], interior = [0, 0]
        : (tensor<16x4xf32>, tensor<f32>) -> tensor<20x4xf32>
    %result = "stablehlo.gather"(%padded, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1, 2], start_index_map = [], index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 4, 2>
    }> : (tensor<20x4xf32>, tensor<2x0xi32>) -> tensor<2x4x2xf32>
    return %result : tensor<2x4x2xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %gathers = transform.structured.match ops{["stablehlo.gather"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.forall"]} in %module
      : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{failed to prepare a replacement for the in-loop ranged gather use}}
    %loads = transform.htile.fuse_ranged_gather_into_loops %gathers into %loops
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @multiple_windows(%source: tensor<16xf32>,
                              %indices: tensor<2x1xi32>) -> tensor<2x4xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %padded = stablehlo.pad %source, %zero,
        low = [0], high = [4], interior = [0]
        : (tensor<16xf32>, tensor<f32>) -> tensor<20xf32>
    %result = "stablehlo.gather"(%padded, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1], start_index_map = [0], index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 4>
    }> : (tensor<20xf32>, tensor<2x1xi32>) -> tensor<2x4xf32>
    scf.forall (%i) in (1) {
      // expected-error@+1 {{expected each gather tile to select exactly one start-index vector}}
      %two_windows = tensor.extract_slice %result[0, 0] [2, 4] [1, 1]
          : tensor<2x4xf32> to tensor<2x4xf32>
      scf.forall.in_parallel {
      }
    }
    return %result : tensor<2x4xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %gathers = transform.structured.match ops{["stablehlo.gather"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.forall"]} in %module
      : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{failed to prepare a replacement for the in-loop ranged gather use}}
    %loads = transform.htile.fuse_ranged_gather_into_loops %gathers into %loops
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @indexed_low_padding(%source: tensor<16xf32>,
                                 %indices: tensor<2x1xi32>) -> tensor<2x4xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %padded = stablehlo.pad %source, %zero,
        low = [1], high = [4], interior = [0]
        : (tensor<16xf32>, tensor<f32>) -> tensor<21xf32>
    %result = "stablehlo.gather"(%padded, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1], start_index_map = [0], index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 4>
    }> : (tensor<21xf32>, tensor<2x1xi32>) -> tensor<2x4xf32>
    scf.forall (%i) in (2) {
      // expected-error@+1 {{expected zero low padding on each dynamically indexed operand dimension}}
      %tile = tensor.extract_slice %result[%i, 0] [1, 4] [1, 1]
          : tensor<2x4xf32> to tensor<1x4xf32>
      scf.forall.in_parallel {
      }
    }
    return %result : tensor<2x4xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %gathers = transform.structured.match ops{["stablehlo.gather"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.forall"]} in %module
      : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{failed to prepare a replacement for the in-loop ranged gather use}}
    %loads = transform.htile.fuse_ranged_gather_into_loops %gathers into %loops
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @insufficient_indexed_high_padding(
      %source: tensor<16xf32>, %indices: tensor<2x1xi32>) -> tensor<2x4xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %padded = stablehlo.pad %source, %zero,
        low = [0], high = [3], interior = [0]
        : (tensor<16xf32>, tensor<f32>) -> tensor<19xf32>
    %result = "stablehlo.gather"(%padded, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1], start_index_map = [0], index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 4>
    }> : (tensor<19xf32>, tensor<2x1xi32>) -> tensor<2x4xf32>
    scf.forall (%i) in (2) {
      // expected-error@+1 {{expected high padding to cover the full gather slice on each dynamically indexed operand dimension}}
      %tile = tensor.extract_slice %result[%i, 0] [1, 4] [1, 1]
          : tensor<2x4xf32> to tensor<1x4xf32>
      scf.forall.in_parallel {
      }
    }
    return %result : tensor<2x4xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %gathers = transform.structured.match ops{["stablehlo.gather"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.forall"]} in %module
      : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{failed to prepare a replacement for the in-loop ranged gather use}}
    %loads = transform.htile.fuse_ranged_gather_into_loops %gathers into %loops
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @permuted_result_layout(%source: tensor<2x16xf32>,
                                    %indices: tensor<2x1xi32>) -> tensor<4x2xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %padded = stablehlo.pad %source, %zero,
        low = [0, 0], high = [0, 4], interior = [0, 0]
        : (tensor<2x16xf32>, tensor<f32>) -> tensor<2x20xf32>
    %result = "stablehlo.gather"(%padded, %indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [0], operand_batching_dims = [0],
        start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 4>
    }> : (tensor<2x20xf32>, tensor<2x1xi32>) -> tensor<4x2xf32>
    scf.forall (%i) in (2) {
      // expected-error@+1 {{expected gather result layout not to permute operand dimensions}}
      %tile = tensor.extract_slice %result[0, %i] [4, 1] [1, 1]
          : tensor<4x2xf32> to tensor<4x1xf32>
      scf.forall.in_parallel {
      }
    }
    return %result : tensor<4x2xf32>
  }
}
