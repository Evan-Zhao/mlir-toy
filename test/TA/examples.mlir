// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s | FileCheck %s

// CHECK-LABEL: func.func @minimal_ta
func.func @minimal_ta(%tensor: tensor<16xf32>) -> tensor<16xf32> {
  %0 = ta.scope axes(%coord "i" : index) {
    %1 = ta.at %tensor[%coord] {axes = #ta.axes<i>}
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %2 = ta.reduce #ta.reduce_kind<add> %1 {axes = #ta.axes<i>}
        : !ta.expr<f32, [i]> -> !ta.expr<f32, []>
    ta.yield %2 : !ta.expr<f32, []>
  } : () -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @two_axis_ta
func.func @two_axis_ta(%tensor: tensor<16x32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" : index, %col "j" : index) {
    %1 = ta.at %tensor[%row, %col] {axes = #ta.axes<i, j>}
        : tensor<16x32xf32> -> !ta.expr<f32, [i, j]>
    %2 = ta.reduce #ta.reduce_kind<add> %1 {axes = #ta.axes<j>}
        : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
    ta.yield %2 : !ta.expr<f32, [i]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func.func @matmul_ta
func.func @matmul_ta(%lhs: tensor<16x64xf32>, %rhs: tensor<64x32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" : index, %col "j" : index, %red "k" : index) {
    %dot = ta.map_reduce #ta.reduce_kind<add> {
      %x = ta.at %lhs[%row, %red] {axes = #ta.axes<i, k>}
          : tensor<16x64xf32> -> !ta.expr<f32, [i, k]>
      %y = ta.at %rhs[%red, %col] {axes = #ta.axes<k, j>}
          : tensor<64x32xf32> -> !ta.expr<f32, [k, j]>
      %xy = ta.mulf %x, %y
          : (!ta.expr<f32, [i, k]>, !ta.expr<f32, [k, j]>)
         -> !ta.expr<f32, [i, j, k]>
      ta.yield %xy : !ta.expr<f32, [i, j, k]>
    } {axes = #ta.axes<k>} : !ta.expr<f32, [i, j]>
    ta.yield %dot : !ta.expr<f32, [i, j]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func.func @elementwise_ta
func.func @elementwise_ta(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" : index, %col "j" : index) {
    %x = ta.at %rows[%row] {axes = #ta.axes<i>}
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %y = ta.at %cols[%col] {axes = #ta.axes<j>}
        : tensor<32xf32> -> !ta.expr<f32, [j]>
    %sum = ta.addf %x, %y
        : (!ta.expr<f32, [i]>, !ta.expr<f32, [j]>)
       -> !ta.expr<f32, [i, j]>
    %scale = ta.constant 2.000000e+00 : f32 : !ta.expr<f32, []>
    %scaled = ta.mulf %sum, %scale
        : (!ta.expr<f32, [i, j]>, !ta.expr<f32, []>)
       -> !ta.expr<f32, [i, j]>
    %limit = ta.exp2 %scaled
        : (!ta.expr<f32, [i, j]>) -> !ta.expr<f32, [i, j]>
    %pred = ta.cmpf olt, %sum, %limit
        : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i, j]>)
       -> !ta.expr<i1, [i, j]>
    %selected = ta.select %pred, %sum, %limit
        : (!ta.expr<i1, [i, j]>, !ta.expr<f32, [i, j]>, !ta.expr<f32, [i, j]>)
       -> !ta.expr<f32, [i, j]>
    ta.yield %selected : !ta.expr<f32, [i, j]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func.func @map_ta
func.func @map_ta(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" : index, %col "j" : index) {
    %x = ta.at %rows[%row] {axes = #ta.axes<i>}
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %y = ta.at %cols[%col] {axes = #ta.axes<j>}
        : tensor<32xf32> -> !ta.expr<f32, [j]>
    %diff = ta.map %x, %y {
    ^bb0(%sx : f32, %sy : f32):
      %r = arith.subf %sx, %sy : f32
      ta.yield %r : f32
    } : (!ta.expr<f32, [i]>, !ta.expr<f32, [j]>)
     -> !ta.expr<f32, [i, j]>
    %smooth = ta.fma %diff, %diff, %x
        : (!ta.expr<f32, [i, j]>, !ta.expr<f32, [i, j]>, !ta.expr<f32, [i]>)
       -> !ta.expr<f32, [i, j]>
    ta.yield %smooth : !ta.expr<f32, [i, j]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}
