// RUN: neptune-opt %s --transform-interpreter --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @matmul_transb(
// CHECK: htile.dot %arg0, %arg1, %arg2 {transpose_b}
// CHECK-SAME: tensor<4x8xf16>, tensor<2x8xf16>, tensor<4x2xf32> -> tensor<4x2xf32>

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

  func.func @matmul_transb(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<2x8xf16>,
      %acc: tensor<4x2xf32>) -> tensor<4x2xf32> {
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
}
// -----

// CHECK-LABEL: func.func @matmul(
// CHECK: htile.dot %arg0, %arg1, %arg2
// CHECK-NOT: transpose_b

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

  func.func @matmul(
      %lhs: tensor<4x8xf32>,
      %rhs: tensor<8x2xf32>,
      %acc: tensor<4x2xf32>) -> tensor<4x2xf32> {
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
}
// -----

// CHECK-LABEL: func.func @matmul_zero_init(
// CHECK-NOT: htile.full
// CHECK: htile.dot %arg0, %arg1 : tensor<4x8xf32>, tensor<8x2xf32> -> tensor<4x2xf32>

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

  func.func @matmul_zero_init(
      %lhs: tensor<4x8xf32>, %rhs: tensor<8x2xf32>) -> tensor<4x2xf32> {
    %empty = tensor.empty() : tensor<4x2xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x2xf32>)
        -> tensor<4x2xf32>
    %0 = linalg.generic {
        indexing_maps = [#mat_lhs, #mat_rhs, #mat_out],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x2xf32>)
        outs(%init : tensor<4x2xf32>) {
    ^bb0(%in_lhs: f32, %in_rhs: f32, %out: f32):
      %mul = arith.mulf %in_lhs, %in_rhs : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
    } -> tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
  }
}
// -----

// CHECK-LABEL: func.func @row_max(
// CHECK: %[[REDUCE:.+]] = htile.reduce %arg0 axis 1 kind "max"
// CHECK: arith.maximumf %arg1, %[[REDUCE]]

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

  func.func @row_max(%input: tensor<4x8xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
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
}
// -----

// CHECK-LABEL: func.func @elemwise(
// CHECK-DAG: %[[BCAST:.+]] = htile.broadcast %arg1 dimensions = [1] : tensor<4xf32> -> tensor<4x8xf32>
// CHECK-DAG: %[[SCALE:.+]] = htile.full %arg3 : f32 -> tensor<4x8xf32>
// CHECK: %[[MUL:.+]] = arith.mulf %arg0, %[[SCALE]] : tensor<4x8xf32>
// CHECK: arith.subf %[[MUL]], %[[BCAST]] : tensor<4x8xf32>
// CHECK-NOT: linalg.generic

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

  func.func @elemwise(
      %input: tensor<4x8xf32>,
      %row_value: tensor<4xf32>,
      %out: tensor<4x8xf32>,
      %scale: f32) -> tensor<4x8xf32> {
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
}
// -----

// CHECK-LABEL: func.func @index_mask(
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[I:.+]] = htile.arange %{{.*}} to %[[C4]] : tensor<4xindex>
// CHECK-DAG: %[[J:.+]] = htile.arange %{{.*}} to %[[C8]] : tensor<8xindex>
// CHECK-DAG: %[[IB:.+]] = htile.broadcast %[[I]] dimensions = [1] : tensor<4xindex> -> tensor<4x8xindex>
// CHECK-DAG: %[[JB:.+]] = htile.broadcast %[[J]] dimensions = [0] : tensor<8xindex> -> tensor<4x8xindex>
// CHECK-DAG: %[[II:.+]] = arith.index_cast %[[IB]] : tensor<4x8xindex> to tensor<4x8xi64>
// CHECK-DAG: %[[JI:.+]] = arith.index_cast %[[JB]] : tensor<4x8xindex> to tensor<4x8xi64>
// CHECK-DAG: %[[CMP:.+]] = arith.cmpi sle, %[[JI]], %[[II]] : tensor<4x8xi64>
// CHECK-DAG: %[[FALSE:.+]] = htile.full %false : i1 -> tensor<4x8xi1>
// CHECK: arith.select %[[CMP]], %arg0, %[[FALSE]] : tensor<4x8xi1>
// CHECK-NOT: linalg.generic

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

  func.func @index_mask(
      %true_tile: tensor<4x8xi1>,
      %out: tensor<4x8xi1>) -> tensor<4x8xi1> {
    %false = arith.constant false
    %0 = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%true_tile : tensor<4x8xi1>)
        outs(%out : tensor<4x8xi1>) {
    ^bb0(%in: i1, %unused: i1):
      %i = linalg.index 0 : index
      %ii = arith.index_cast %i : index to i64
      %j = linalg.index 1 : index
      %ji = arith.index_cast %j : index to i64
      %live = arith.cmpi sle, %ji, %ii : i64
      %selected = arith.select %live, %in, %false : i1
      linalg.yield %selected : i1
    } -> tensor<4x8xi1>
    return %0 : tensor<4x8xi1>
  }
}
// -----

// CHECK-LABEL: func.func @fill(
// CHECK: htile.full %arg0 : f32 -> tensor<4x8xf32>

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

  func.func @fill(%value: f32, %out: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = linalg.fill ins(%value : f32) outs(%out : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}
// -----

// CHECK-LABEL: func.func @unit_dim_expand_shape(
// CHECK: %[[TILE:.+]] = htile.unsqueeze %arg0 mask [false, true, false]
// CHECK-SAME: tensor<4x8xf32> -> tensor<4x1x8xf32>
// CHECK: %[[SCALAR:.+]] = htile.unsqueeze %arg1 mask [true]
// CHECK-SAME: tensor<f32> -> tensor<1xf32>

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

  func.func @unit_dim_expand_shape(%tile: tensor<4x8xf32>, %scalar: tensor<f32>)
      -> (tensor<4x1x8xf32>, tensor<1xf32>) {
    %expanded = tensor.expand_shape %tile [[0, 1], [2]] output_shape [4, 1, 8]
        : tensor<4x8xf32> into tensor<4x1x8xf32>
    %expanded_scalar = tensor.expand_shape %scalar [] output_shape [1]
        : tensor<f32> into tensor<1xf32>
    return %expanded, %expanded_scalar : tensor<4x1x8xf32>, tensor<1xf32>
  }
}
// -----

// CHECK-LABEL: func.func @unit_dim_collapse_shape(
// CHECK: %[[TILE:.+]] = htile.squeeze %arg0 mask [false, true, false]
// CHECK-SAME: tensor<4x1x8xf32> -> tensor<4x8xf32>
// CHECK: %[[SCALAR:.+]] = htile.squeeze %arg1 mask [true]
// CHECK-SAME: tensor<1xf32> -> tensor<f32>

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

  func.func @unit_dim_collapse_shape(%tile: tensor<4x1x8xf32>, %scalar: tensor<1xf32>)
      -> (tensor<4x8xf32>, tensor<f32>) {
    %collapsed = tensor.collapse_shape %tile [[0, 1], [2]]
        : tensor<4x1x8xf32> into tensor<4x8xf32>
    %collapsed_scalar = tensor.collapse_shape %scalar []
        : tensor<1xf32> into tensor<f32>
    return %collapsed, %collapsed_scalar : tensor<4x8xf32>, tensor<f32>
  }
}
// -----

#id = affine_map<(d0) -> (d0)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to rewrite a linalg op}}
    transform.htile.linalg_to_semantic %funcs : !transform.any_op
    transform.yield
  }

  func.func @unsupported(
      %input: tensor<4xf32>,
      %out0: tensor<4xf32>,
      %out1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    // expected-error @below {{unsupported linalg.generic operation}}
    %0:2 = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>)
        outs(%out0, %out1 : tensor<4xf32>, tensor<4xf32>) {
    ^bb0(%in: f32, %unused0: f32, %unused1: f32):
      linalg.yield %in, %in : f32, f32
    } -> (tensor<4xf32>, tensor<4xf32>)
    return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }
}
