// RUN: mlir-opt --load-dialect-plugin=%neptune_mlir_test_plugin %s --transform-interpreter | FileCheck %s

func.func @extract_after_insert(%src: tensor<4xf32>, %dest: tensor<8xf32>) -> tensor<4xf32> {
  %0 = tensor.insert_slice %src into %dest[2] [4] [1] : tensor<4xf32> into tensor<8xf32>
  %1 = tensor.extract_slice %0[2] [4] [1] : tensor<8xf32> to tensor<4xf32>
  return %1 : tensor<4xf32>
}

func.func @insert_after_extract(%src: tensor<8xf32>) -> tensor<8xf32> {
  %0 = tensor.extract_slice %src[2] [4] [1] : tensor<8xf32> to tensor<4xf32>
  %1 = tensor.insert_slice %0 into %src[2] [4] [1] : tensor<4xf32> into tensor<8xf32>
  return %1 : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.loop.tensor_subset_canonicalization
    } : !transform.op<"func.func">
    transform.yield
  }
}

// CHECK-LABEL: func.func @extract_after_insert(
// CHECK-SAME: %[[SRC:.*]]: tensor<4xf32>, %[[DEST:.*]]: tensor<8xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: tensor.extract_slice
// CHECK: return %[[SRC]] : tensor<4xf32>

// CHECK-LABEL: func.func @insert_after_extract(
// CHECK-SAME: %[[SRC:.*]]: tensor<8xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: tensor.extract_slice
// CHECK: return %[[SRC]] : tensor<8xf32>
