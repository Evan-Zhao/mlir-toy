// RUN: %neptune-opt %s --transform-interpreter --split-input-file 2>&1 | FileCheck %s

// CHECK: IR printer
// CHECK-NEXT: %{{.*}} = linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.addf
// CHECK-LABEL: func.func @ta_matmul_to_linalg_transform(
// CHECK: arith.mulf
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[INIT:.*]] = linalg.fill ins(%[[ZERO]] : f32)
// CHECK: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction"]{{.*}}outs(%[[INIT]] : tensor<4x16xf32>)
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
      %dot = ta.reduce #ta.reduce_kind<add> %xy init(0.0 : f32) {axes = #ta.axes<k>}
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

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.ta.to_linalg %funcs : !transform.any_op
    transform.yield
  }

  // CHECK-LABEL: func.func @finite_max_init
  // CHECK: %[[SEED:.*]] = arith.constant -3.40282347E+38 : f32
  // CHECK: %[[INIT:.*]] = linalg.fill ins(%[[SEED]] : f32)
  // CHECK: linalg.generic {{.*}}outs(%[[INIT]] : tensor<2xf32>)
  // CHECK: arith.maximumf
  func.func @finite_max_init(%input: tensor<2x3xf32>) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %max = ta.reduce <max> %x init(0xFF7FFFFF : f32) {axes = #ta.axes<j>}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %max : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @operand_max_init
  // CHECK: %[[INIT:.*]] = linalg.fill ins(%arg1 : f32)
  // CHECK: linalg.generic {{.*}}outs(%[[INIT]] : tensor<2xf32>)
  // CHECK: arith.maximumf
  func.func @operand_max_init(%input: tensor<2x3xf32>, %init: f32) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %max = ta.reduce <max> %x init(%init : f32) {axes = #ta.axes<j>}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %max : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }

  // The initializer seeds each output accumulator once.
  // CHECK-LABEL: func.func @operand_sum_init
  // CHECK: %[[INIT:.*]] = linalg.fill ins(%arg1 : i32)
  // CHECK: linalg.generic {{.*}}ins(%arg0 : tensor<2x3xi32>) outs(%[[INIT]] : tensor<2xi32>)
  // CHECK: ^bb0(%[[X:.*]]: i32, %[[ACC:.*]]: i32):
  // CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[ACC]], %[[X]] : i32
  // CHECK-NEXT: linalg.yield %[[SUM]] : i32
  func.func @operand_sum_init(%input: tensor<2x3xi32>, %init: i32) -> tensor<2xi32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xi32> -> !ta.expr<i32, [i, j]>
      %sum = ta.reduce <add> %x init(%init : i32) {axes = #ta.axes<j>}
          : !ta.expr<i32, [i, j]> -> !ta.expr<i32, [i]>
      ta.yield %sum : !ta.expr<i32, [i]>
    } : () -> tensor<2xi32>
    return %out : tensor<2xi32>
  }

  // CHECK-LABEL: func.func @nonzero_float_sum_init
  // CHECK: %[[SEED:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK: linalg.fill ins(%[[SEED]] : f32)
  // CHECK: arith.addf
  func.func @nonzero_float_sum_init(%input: tensor<2x3xf32>) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %sum = ta.reduce <add> %x init(7.0 : f32) {axes = #ta.axes<j>}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %sum : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @nonzero_integer_sum_init
  // CHECK: %[[SEED:.*]] = arith.constant 7 : i32
  // CHECK: linalg.fill ins(%[[SEED]] : i32)
  // CHECK: arith.addi
  func.func @nonzero_integer_sum_init(%input: tensor<2x3xi32>) -> tensor<2xi32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xi32> -> !ta.expr<i32, [i, j]>
      %sum = ta.reduce <add> %x init(7 : i32) {axes = #ta.axes<j>}
          : !ta.expr<i32, [i, j]> -> !ta.expr<i32, [i]>
      ta.yield %sum : !ta.expr<i32, [i]>
    } : () -> tensor<2xi32>
    return %out : tensor<2xi32>
  }

  // Lowering preserves the initializer extracted from an earlier scope.
  // CHECK-LABEL: func.func @computed_init
  // CHECK: %[[SCALAR:.*]] = linalg.generic
  // CHECK: arith.addf
  // CHECK: %[[SEED:.*]] = tensor.extract %[[SCALAR]][] : tensor<f32>
  // CHECK: %[[INIT:.*]] = linalg.fill ins(%[[SEED]] : f32)
  // CHECK: linalg.generic {{.*}}outs(%[[INIT]] : tensor<2xf32>)
  // CHECK: arith.maximumf
  func.func @computed_init(%input: tensor<2x3xf32>, %a: tensor<f32>, %b: tensor<f32>) -> tensor<2xf32> {
    %scalar = ta.scope axes() {
      %x = ta.at %a[] : tensor<f32> -> !ta.expr<f32, []>
      %y = ta.at %b[] : tensor<f32> -> !ta.expr<f32, []>
      %sum = ta.add %x, %y : (!ta.expr<f32, []>, !ta.expr<f32, []>) -> !ta.expr<f32, []>
      ta.yield %sum : !ta.expr<f32, []>
    } : () -> tensor<f32>
    %init = tensor.extract %scalar[] : tensor<f32>
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %max = ta.reduce <max> %x init(%init : f32) {axes = #ta.axes<j>}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %max : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @negative_zero_init
  // CHECK: %[[SEED:.*]] = arith.constant -0.000000e+00 : f32
  // CHECK: linalg.fill ins(%[[SEED]] : f32)
  // CHECK: arith.addf
  func.func @negative_zero_init(%input: tensor<2x3xf32>) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %sum = ta.reduce <add> %x init(-0.0 : f32) {axes = #ta.axes<j>}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %sum : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }

  // An empty reduction returns its initializer.
  // CHECK-LABEL: func.func @empty_product_init
  // CHECK: %[[SEED:.*]] = arith.constant 3.000000e+00 : f64
  // CHECK: %[[INIT:.*]] = linalg.fill ins(%[[SEED]] : f64)
  // CHECK: linalg.generic {{.*}}ins(%arg0 : tensor<2x0xf64>) outs(%[[INIT]] : tensor<2xf64>)
  // CHECK: arith.mulf
  func.func @empty_product_init(%input: tensor<2x0xf64>) -> tensor<2xf64> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 0) {
      %x = ta.at %input[%i, %j] : tensor<2x0xf64> -> !ta.expr<f64, [i, j]>
      %prod = ta.reduce <mul> %x init(3.0 : f64) {axes = #ta.axes<j>}
          : !ta.expr<f64, [i, j]> -> !ta.expr<f64, [i]>
      ta.yield %prod : !ta.expr<f64, [i]>
    } : () -> tensor<2xf64>
    return %out : tensor<2xf64>
  }

  // CHECK-LABEL: func.func @min_init
  // CHECK: %[[SEED:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[INIT:.*]] = linalg.fill ins(%[[SEED]] : f16)
  // CHECK: linalg.generic {{.*}}outs(%[[INIT]] : tensor<2xf16>)
  // CHECK: arith.minimumf
  func.func @min_init(%input: tensor<2x3xf16>) -> tensor<2xf16> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xf16> -> !ta.expr<f16, [i, j]>
      %min = ta.reduce <min> %x init(0.0 : f16) {axes = #ta.axes<j>}
          : !ta.expr<f16, [i, j]> -> !ta.expr<f16, [i]>
      ta.yield %min : !ta.expr<f16, [i]>
    } : () -> tensor<2xf16>
    return %out : tensor<2xf16>
  }

  // CHECK-LABEL: func.func @canonical_max_init
  // CHECK: %[[SEED:.*]] = arith.constant 0xFF800000 : f32
  // CHECK: %[[INIT:.*]] = linalg.fill ins(%[[SEED]] : f32)
  // CHECK: linalg.generic {{.*}}outs(%[[INIT]] : tensor<2xf32>)
  // CHECK: arith.maximumf
  func.func @canonical_max_init(%input: tensor<2x3xf32>) -> tensor<2xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %x = ta.at %input[%i, %j] : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %max = ta.reduce <max> %x init(0xFF800000 : f32) {axes = #ta.axes<j>}
          : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
      ta.yield %max : !ta.expr<f32, [i]>
    } : () -> tensor<2xf32>
    return %out : tensor<2xf32>
  }

  // Regression: ta.cmpf used to reach an unsupported-scalar-op assertion.
  // CHECK-LABEL: func.func @compare_to_linalg
  // CHECK: arith.cmpf une,
  // CHECK: arith.cmpi slt,
  // CHECK: arith.cmpi uge,
  // CHECK: arith.select
  func.func @compare_to_linalg(%flhs: tensor<2x3xf32>, %frhs: tensor<2x3xf32>,
                               %ilhs: tensor<2x3xi32>, %irhs: tensor<2x3xi32>) -> tensor<2x3xf32> {
    %out = ta.scope axes(%i "i" extent 2, %j "j" extent 3) {
      %fl = ta.at %flhs[%i, %j] : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %fr = ta.at %frhs[%i, %j] : tensor<2x3xf32> -> !ta.expr<f32, [i, j]>
      %il = ta.at %ilhs[%i, %j] : tensor<2x3xi32> -> !ta.expr<i32, [i, j]>
      %ir = ta.at %irhs[%i, %j] : tensor<2x3xi32> -> !ta.expr<i32, [i, j]>
      %fp = ta.cmpf une, %fl, %fr
          : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i, j]>) -> !ta.expr<i1, [i, j]>
      %sp = ta.cmpi slt, %il, %ir
          : (!ta.expr<i32, [i, j]>, !ta.expr<i32, [i, j]>) -> !ta.expr<i1, [i, j]>
      %up = ta.cmpi uge, %il, %ir
          : (!ta.expr<i32, [i, j]>, !ta.expr<i32, [i, j]>) -> !ta.expr<i1, [i, j]>
      %fs = ta.select %fp, %fl, %fr
          : (!ta.expr<i1, [i, j]>, !ta.expr<f32, [i, j]>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      %ss = ta.select %sp, %fs, %fr
          : (!ta.expr<i1, [i, j]>, !ta.expr<f32, [i, j]>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      %us = ta.select %up, %ss, %fr
          : (!ta.expr<i1, [i, j]>, !ta.expr<f32, [i, j]>, !ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
      ta.yield %us : !ta.expr<f32, [i, j]>
    } : () -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
