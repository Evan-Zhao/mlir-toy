// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin --load-pass-plugin=%neptune_ta_plugin %s --transform-interpreter 2>&1 | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_ta_matmul(%candidate: !transform.any_op {transform.readonly})
      -> !transform.any_op {
    %matched = transform.match.ta.einsum %candidate
        {equation = "i k, k j -> i j"}
        : (!transform.any_op) -> !transform.any_op
    transform.yield %matched : !transform.any_op
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %ta_matmul = transform.collect_matching @match_ta_matmul in %func
        : (!transform.any_op) -> !transform.any_op
    transform.ta.to_linalg %func : !transform.any_op
    transform.print %ta_matmul : !transform.any_op
    transform.yield
  }

// CHECK: IR printer
// CHECK-NEXT: %{{.*}} = linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.addf
// CHECK-LABEL: func.func @ta_matmul_to_linalg_transform(
  func.func @ta_matmul_to_linalg_transform(%lhs: tensor<4x8xf32>,
                                           %rhs: tensor<8x16xf32>)
      -> tensor<4x16xf32> {
    %out = ta.scope axes(%i "i" extent 4, %j "j" extent 16, %k "k" extent 8) {
      %x = ta.at %lhs[%i, %k] {axes = #ta.axes<i, k>}
          : tensor<4x8xf32> -> !ta.expr<f32, [i, k]>
      %y = ta.at %rhs[%k, %j] {axes = #ta.axes<k, j>}
          : tensor<8x16xf32> -> !ta.expr<f32, [k, j]>
      %xy = ta.mulf %x, %y
          : (!ta.expr<f32, [i, k]>, !ta.expr<f32, [k, j]>)
         -> !ta.expr<f32, [i, j, k]>
      %dot = ta.reduce #ta.reduce_kind<add> %xy {axes = #ta.axes<k>}
          : !ta.expr<f32, [i, j, k]> -> !ta.expr<f32, [i, j]>
      ta.yield %dot : !ta.expr<f32, [i, j]>
    } : () -> tensor<4x16xf32>
    return %out : tensor<4x16xf32>
  }
}

// CHECK: arith.mulf
// CHECK: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.addf
