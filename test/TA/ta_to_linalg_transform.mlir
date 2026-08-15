// RUN: neptune-opt %s --transform-interpreter --split-input-file 2>&1 | FileCheck %s

// CHECK: IR printer
// CHECK-NEXT: %{{.*}} = linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.addf
// CHECK-LABEL: func.func @ta_matmul_to_linalg_transform(
// CHECK: arith.mulf
// CHECK: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.addf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %ta_matmul = transform.structured.match ops{["ta.reduce"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.ta.to_linalg %func : !transform.any_op
    transform.print %ta_matmul : !transform.any_op
    transform.yield
  }

  func.func @ta_matmul_to_linalg_transform(%lhs: tensor<4x8xf32>,
                                           %rhs: tensor<8x16xf32>)
      -> tensor<4x16xf32> {
    %out = ta.scope axes(%i "i" extent 4, %j "j" extent 16, %k "k" extent 8) {
      %x = ta.at %lhs[%i, %k] {axes = #ta.axes<i, k>}
          : tensor<4x8xf32> -> !ta.expr<f32, [i, k]>
      %y = ta.at %rhs[%k, %j] {axes = #ta.axes<k, j>}
          : tensor<8x16xf32> -> !ta.expr<f32, [k, j]>
      %xy = ta.mul %x, %y
          : (!ta.expr<f32, [i, k]>, !ta.expr<f32, [k, j]>) -> !ta.expr<f32, [i, k, j]>
      %dot = ta.reduce #ta.reduce_kind<add> %xy {axes = #ta.axes<k>}
          : !ta.expr<f32, [i, k, j]> -> !ta.expr<f32, [i, j]>
      ta.yield %dot : !ta.expr<f32, [i, j]>
    } : () -> tensor<4x16xf32>
    return %out : tensor<4x16xf32>
  }
}

// -----

// CHECK-LABEL: func.func @subst_to_linalg(
// CHECK-NOT: ta.subst
// CHECK: linalg.generic {indexing_maps = [#{{.*}}], iterator_types = ["parallel", "parallel"]}
// CHECK-NEXT: ^bb0(%{{.*}}: i64):
// CHECK-NEXT: %[[J_INDEX:.*]] = linalg.index 0 : index
// CHECK-NEXT: %[[J:.*]] = arith.index_cast %[[J_INDEX]] : index to i64
// CHECK-NEXT: %[[I_INDEX:.*]] = linalg.index 1 : index
// CHECK-NEXT: %[[I:.*]] = arith.index_cast %[[I_INDEX]] : index to i64
// CHECK-NEXT: arith.subi %[[J]], %[[I]] : i64

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %ta_matmul = transform.structured.match ops{["ta.reduce"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.ta.to_linalg %func : !transform.any_op
    transform.print %ta_matmul : !transform.any_op
    transform.yield
  }

  func.func @subst_to_linalg() -> tensor<4x4xi64> {
    %out = ta.scope axes(%i "i" extent 4, %j "j" extent 4) {
      %idx = ta.index %i : !ta.expr<i64, [i]>
      %idx_j = ta.subst %idx {from_axes = #ta.axes<i>, to_axes = #ta.axes<j>}
          : !ta.expr<i64, [i]> -> !ta.expr<i64, [j]>
      %diff = ta.sub %idx_j, %idx
          : (!ta.expr<i64, [j]>, !ta.expr<i64, [i]>) -> !ta.expr<i64, [j, i]>
      ta.yield %diff : !ta.expr<i64, [j, i]>
    } : () -> tensor<4x4xi64>
    return %out : tensor<4x4xi64>
  }
}

// -----

// CHECK-LABEL: func.func @cast_to_linalg(
// CHECK: linalg.generic
// CHECK: %[[INDEX:.*]] = linalg.index 0 : index
// CHECK: %[[INDEX_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i64
// CHECK: arith.sitofp %[[INDEX_CAST]] : i64 to f32

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %ta_matmul = transform.structured.match ops{["ta.reduce"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.ta.to_linalg %func : !transform.any_op
    transform.print %ta_matmul : !transform.any_op
    transform.yield
  }

  func.func @cast_to_linalg() -> tensor<4xf32> {
    %out = ta.scope axes(%i "i" extent 4) {
      %idx = ta.index %i : !ta.expr<i64, [i]>
      %cast = ta.cast %idx : (!ta.expr<i64, [i]>) -> !ta.expr<f32, [i]>
      ta.yield %cast : !ta.expr<f32, [i]>
    } : () -> tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
// -----

// CHECK-LABEL: func.func @subst_at_to_linalg(
// CHECK-NOT: ta.subst
// CHECK: linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg0
// CHECK: arith.addf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %ta_matmul = transform.structured.match ops{["ta.reduce"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.ta.to_linalg %func : !transform.any_op
    transform.print %ta_matmul : !transform.any_op
    transform.yield
  }

  func.func @subst_at_to_linalg(%tensor: tensor<4xf32>) -> tensor<4x4xf32> {
    %out = ta.scope axes(%i "i" extent 4, %j "j" extent 4) {
      %x = ta.at %tensor[%i]
          : tensor<4xf32> -> !ta.expr<f32, [i]>
      %y = ta.subst %x {from_axes = #ta.axes<i>, to_axes = #ta.axes<j>}
          : !ta.expr<f32, [i]> -> !ta.expr<f32, [j]>
      %sum = ta.add %y, %x
          : (!ta.expr<f32, [j]>, !ta.expr<f32, [i]>) -> !ta.expr<f32, [j, i]>
      ta.yield %sum : !ta.expr<f32, [j, i]>
    } : () -> tensor<4x4xf32>
    return %out : tensor<4x4xf32>
  }
}
