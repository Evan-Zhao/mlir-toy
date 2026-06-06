// RUN: mlir-opt --load-dialect-plugin=%neptune_htile_plugin %s --transform-interpreter | FileCheck %s

#mat_lhs = affine_map<(m, n, k) -> (m, k)>
#mat_rhs_t = affine_map<(m, n, k) -> (n, k)>
#mat_rhs = affine_map<(m, n, k) -> (k, n)>
#mat_out = affine_map<(m, n, k) -> (m, n)>
#id2 = affine_map<(m, n) -> (m, n)>
#row = affine_map<(m, n) -> (m)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.htile.linalg_to_semantic %funcs : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @matmul_transb(
  func.func @matmul_transb(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<2x8xf16>,
      %acc: tensor<4x2xf32>) -> tensor<4x2xf32> {
    // CHECK: htile.dot %arg0, %arg1, %arg2 {transpose_b}
    // CHECK-SAME: tensor<4x8xf16>, tensor<2x8xf16>, tensor<4x2xf32> -> tensor<4x2xf32>
    %0 = linalg.generic {
        indexing_maps = [#mat_lhs, #mat_rhs_t, #mat_out],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<2x8xf16>)
        outs(%acc : tensor<4x2xf32>) {
    ^bb0(%in_lhs: f16, %in_rhs: f16, %out: f32):
      %rhs_f32 = arith.extf %in_rhs : f16 to f32
      %lhs_f32 = arith.extf %in_lhs : f16 to f32
      %mul = arith.mulf %lhs_f32, %rhs_f32 : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
    } -> tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
  }

  // CHECK-LABEL: func.func @matmul(
  func.func @matmul(
      %lhs: tensor<4x8xf32>,
      %rhs: tensor<8x2xf32>,
      %acc: tensor<4x2xf32>) -> tensor<4x2xf32> {
    // CHECK: htile.dot %arg0, %arg1, %arg2
    // CHECK-NOT: transpose_b
    %0 = linalg.generic {
        indexing_maps = [#mat_lhs, #mat_rhs, #mat_out],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x2xf32>)
        outs(%acc : tensor<4x2xf32>) {
    ^bb0(%in_lhs: f32, %in_rhs: f32, %out: f32):
      %mul = arith.mulf %in_lhs, %in_rhs : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
    } -> tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
  }

  // CHECK-LABEL: func.func @row_max(
  func.func @row_max(%input: tensor<4x8xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: %[[REDUCE:.+]] = htile.reduce %arg0 axis 1 kind "max"
    // CHECK: arith.maximumf %arg1, %[[REDUCE]]
    %0 = linalg.generic {
        indexing_maps = [#id2, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>)
        outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %out, %in : f32
      linalg.yield %max : f32
    } -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @elemwise(
  func.func @elemwise(
      %input: tensor<4x8xf32>,
      %row_value: tensor<4xf32>,
      %out: tensor<4x8xf32>,
      %scale: f32) -> tensor<4x8xf32> {
    // CHECK-DAG: %[[BCAST_EMPTY:.+]] = tensor.empty() : tensor<4x8xf32>
    // CHECK-DAG: %[[BCAST:.+]] = linalg.broadcast ins(%arg1 : tensor<4xf32>) outs(%[[BCAST_EMPTY]] : tensor<4x8xf32>) dimensions = [1]
    // CHECK-DAG: %[[SCALE:.+]] = htile.full %arg3 : f32 -> tensor<4x8xf32>
    // CHECK: %[[MUL:.+]] = arith.mulf %arg0, %[[SCALE]] : tensor<4x8xf32>
    // CHECK: arith.subf %[[MUL]], %[[BCAST]] : tensor<4x8xf32>
    // CHECK-NOT: linalg.generic
    %0 = linalg.generic {
        indexing_maps = [#id2, #row, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%input, %row_value : tensor<4x8xf32>, tensor<4xf32>)
        outs(%out : tensor<4x8xf32>) {
    ^bb0(%in: f32, %row: f32, %unused: f32):
      %scaled = arith.mulf %in, %scale : f32
      %shifted = arith.subf %scaled, %row : f32
      linalg.yield %shifted : f32
    } -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func @fill(
  func.func @fill(%value: f32, %out: tensor<4x8xf32>) -> tensor<4x8xf32> {
    // CHECK: htile.full %arg0 : f32 -> tensor<4x8xf32>
    %0 = linalg.fill ins(%value : f32) outs(%out : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}
