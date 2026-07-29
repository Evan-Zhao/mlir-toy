// RUN: neptune-opt --transform-interpreter %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_packed_window_extract(
// CHECK-SAME: %[[PACKED:[^,]+]]: tensor<8x3xf32>
// CHECK-SAME: %[[STARTS:[^,]+]]: tensor<2xi32>
// CHECK-SAME: %[[LENGTHS:[^,]+]]: tensor<2xi32>
// CHECK-SAME: %[[OTHER_TENSOR:[^)]+]]: tensor<f32>
// CHECK-NOT: stablehlo.custom_call
// CHECK: tensor.extract %[[STARTS]][%{{.*}}] : tensor<2xi32>
// CHECK: tensor.extract %[[LENGTHS]][%{{.*}}] : tensor<2xi32>
// CHECK: %[[MASK:.*]] = linalg.generic
// CHECK: arith.cmpi slt
// CHECK: %[[OTHER:.*]] = tensor.extract %[[OTHER_TENSOR]][] : tensor<f32>
// CHECK: %[[LOAD:.*]] = htile.load %[[PACKED]][%{{.*}}, %{{.*}}]
// CHECK-SAME: mask(%[[MASK]] : tensor<4x3xi1>) other(%[[OTHER]] : f32)
// CHECK-SAME: tensor<8x3xf32> -> tensor<4x3xf32>
// CHECK: %[[WINDOW:.*]] = tensor.expand_shape %[[LOAD]]
// CHECK-SAME: tensor<4x3xf32> into tensor<1x4x3xf32>
// CHECK: tensor.parallel_insert_slice %[[WINDOW]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %extracts = transform.structured.match
        ops{["stablehlo.custom_call"]}
        attributes {call_target_name = "neptune.packed_window_extract"}
        in %module : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.forall"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loads = transform.htile.fuse_packed_window_extract %extracts into %loops
        : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @fuse_packed_window_extract(
      %packed: tensor<8x3xf32>, %starts: tensor<2xi32>,
      %lengths: tensor<2xi32>, %other: tensor<f32>) -> tensor<2x4x3xf32> {
    %windows = stablehlo.custom_call @neptune.packed_window_extract(
        %packed, %starts, %lengths, %other)
        : (tensor<8x3xf32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>)
        -> tensor<2x4x3xf32>
    %empty = tensor.empty() : tensor<2x4x3xf32>
    %result = scf.forall (%document) in (2) shared_outs(%out = %empty)
        -> tensor<2x4x3xf32> {
      %window = tensor.extract_slice %windows[%document, 0, 0] [1, 4, 3] [1, 1, 1]
          : tensor<2x4x3xf32> to tensor<1x4x3xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %window into %out[%document, 0, 0] [1, 4, 3] [1, 1, 1]
            : tensor<1x4x3xf32> into tensor<2x4x3xf32>
      }
    }
    return %result : tensor<2x4x3xf32>
  }
}
