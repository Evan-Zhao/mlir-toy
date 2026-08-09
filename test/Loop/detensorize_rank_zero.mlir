// RUN: neptune-opt %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.linalg.detensorize_rank_zero
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %funcs : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @rank_zero_linalg(
  func.func @rank_zero_linalg(%source: tensor<9xi32>, %i: index, %j: index)
      -> (tensor<i32>, i32) {
    // CHECK-NOT: tensor.extract_slice
    // CHECK-DAG: %[[LHS:.+]] = tensor.extract %arg0[%arg1] : tensor<9xi32>
    // CHECK-DAG: %[[RHS:.+]] = tensor.extract %arg0[%arg2] : tensor<9xi32>
    // CHECK: %[[SUB:.+]] = arith.subi %[[LHS]], %[[RHS]] : i32
    // CHECK: %[[TENSOR:.+]] = tensor.from_elements %[[SUB]] : tensor<i32>
    // CHECK-NOT: tensor.extract
    // CHECK: return %[[TENSOR]], %[[SUB]] : tensor<i32>, i32
    %lhs = tensor.extract_slice %source[%i] [1] [1]
        : tensor<9xi32> to tensor<i32>
    %rhs = tensor.extract_slice %source[%j] [1] [1]
        : tensor<9xi32> to tensor<i32>
    %empty = tensor.empty() : tensor<i32>
    %difference = linalg.generic {
        indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>],
        iterator_types = []}
        ins(%lhs, %rhs : tensor<i32>, tensor<i32>)
        outs(%empty : tensor<i32>) {
    ^bb0(%lhs_element: i32, %rhs_element: i32, %unused: i32):
      %sub = arith.subi %lhs_element, %rhs_element : i32
      linalg.yield %sub : i32
    } -> tensor<i32>
    %scalar = tensor.extract %difference[] : tensor<i32>
    return %difference, %scalar : tensor<i32>, i32
  }

  // CHECK-LABEL: func.func @rank_zero_arith(
  func.func @rank_zero_arith(%lhs: i32, %rhs: i32) -> (tensor<i32>, i32) {
    // CHECK-NOT: tensor.from_elements %arg0
    // CHECK-NOT: tensor.from_elements %arg1
    // CHECK: %[[SUM:.+]] = arith.addi %arg0, %arg1 : i32
    // CHECK: %[[TENSOR:.+]] = tensor.from_elements %[[SUM]] : tensor<i32>
    // CHECK-NOT: tensor.extract
    // CHECK: return %[[TENSOR]], %[[SUM]] : tensor<i32>, i32
    %lhs_tensor = tensor.from_elements %lhs : tensor<i32>
    %rhs_tensor = tensor.from_elements %rhs : tensor<i32>
    %sum = arith.addi %lhs_tensor, %rhs_tensor : tensor<i32>
    %scalar = tensor.extract %sum[] : tensor<i32>
    return %sum, %scalar : tensor<i32>, i32
  }

  // CHECK-LABEL: func.func @rank_zero_inputs_to_tiled_linalg(
  func.func @rank_zero_inputs_to_tiled_linalg(
      %lhs: i32, %rhs: i32, %input: tensor<4x8xf32>, %output: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %c0_i32 = arith.constant 0 : i32
    // CHECK-NOT: tensor.from_elements
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%arg2 : tensor<4x8xf32>)
    // CHECK-SAME: outs(%arg3 : tensor<4x8xf32>)
    // CHECK: arith.subi %arg0, %arg1 : i32
    %lhs_tensor = tensor.from_elements %lhs : tensor<i32>
    %rhs_tensor = tensor.from_elements %rhs : tensor<i32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs_tensor, %rhs_tensor, %input
            : tensor<i32>, tensor<i32>, tensor<4x8xf32>)
        outs(%output : tensor<4x8xf32>) {
    ^bb0(%lhs_element: i32, %rhs_element: i32, %input_element: f32, %out: f32):
      %difference = arith.subi %lhs_element, %rhs_element : i32
      %positive = arith.cmpi sgt, %difference, %c0_i32 : i32
      %selected = arith.select %positive, %input_element, %out : f32
      linalg.yield %selected : f32
    } -> tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }
}
