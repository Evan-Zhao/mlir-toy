// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s | FileCheck %s

// CHECK-LABEL: func.func @minimal_ta
func.func @minimal_ta(%tensor: tensor<16xf32>) -> tensor<16xf32> {
  %0 = ta.scope axes(%coord "i" extent 16) {
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
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
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
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32, %red "k" extent 64) {
    %x = ta.at %lhs[%row, %red] {axes = #ta.axes<i, k>}
        : tensor<16x64xf32> -> !ta.expr<f32, [i, k]>
    %y = ta.at %rhs[%red, %col] {axes = #ta.axes<k, j>}
        : tensor<64x32xf32> -> !ta.expr<f32, [k, j]>
    %xy = ta.mulf %x, %y
        : (!ta.expr<f32, [i, k]>, !ta.expr<f32, [k, j]>)
       -> !ta.expr<f32, [i, k, j]>
    %dot = ta.reduce #ta.reduce_kind<add> %xy {axes = #ta.axes<k>}
        : !ta.expr<f32, [i, k, j]> -> !ta.expr<f32, [i, j]>
    ta.yield %dot : !ta.expr<f32, [i, j]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func.func @elementwise_ta
func.func @elementwise_ta(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
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

// CHECK-LABEL: func.func @float_cast_ta
func.func @float_cast_ta(%tensor: tensor<16xf16>) -> tensor<16xf16> {
  %0 = ta.scope axes(%coord "i" extent 16) {
    %x = ta.at %tensor[%coord] {axes = #ta.axes<i>}
        : tensor<16xf16> -> !ta.expr<f16, [i]>
    %wide = ta.extf %x
        : (!ta.expr<f16, [i]>) -> !ta.expr<f32, [i]>
    %narrow = ta.truncf %wide
        : (!ta.expr<f32, [i]>) -> !ta.expr<f16, [i]>
    ta.yield %narrow : !ta.expr<f16, [i]>
  } : () -> tensor<16xf16>
  return %0 : tensor<16xf16>
}

// CHECK-LABEL: func.func @dynamic_extent_ta
func.func @dynamic_extent_ta(%tensor: tensor<?xf32>, %n: index) -> tensor<?xf32> {
  // CHECK: ta.scope axes(%i "i" extent %{{.+}})
  %0 = ta.scope axes(%coord "i" extent %n) {
    %x = ta.at %tensor[%coord] {axes = #ta.axes<i>}
        : tensor<?xf32> -> !ta.expr<f32, [i]>
    ta.yield %x : !ta.expr<f32, [i]>
  } : () -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func.func @map_ta
func.func @map_ta(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
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

// CHECK-LABEL: func.func @attention_ta
func.func @attention_ta(%Q: tensor<2x3x4x6xf32>,
                        %K: tensor<2x3x5x6xf32>,
                        %V: tensor<2x3x5x7xf32>)
    -> tensor<2x3x4x7xf32> {
  %O = ta.scope axes(%b "b" extent 2, %h "h" extent 3,
                     %i "i" extent 4, %j "j" extent 5,
                     %d "d" extent 6, %e "e" extent 7) {
    %q = ta.at %Q[%b, %h, %i, %d] {axes = #ta.axes<b, h, i, d>}
        : tensor<2x3x4x6xf32> -> !ta.expr<f32, [b, h, i, d]>
    %k = ta.at %K[%b, %h, %j, %d] {axes = #ta.axes<b, h, j, d>}
        : tensor<2x3x5x6xf32> -> !ta.expr<f32, [b, h, j, d]>
    %qk = ta.mulf %q, %k
        : (!ta.expr<f32, [b, h, i, d]>, !ta.expr<f32, [b, h, j, d]>)
       -> !ta.expr<f32, [b, h, i, d, j]>
    %dot = ta.reduce #ta.reduce_kind<add> %qk {axes = #ta.axes<d>}
        : !ta.expr<f32, [b, h, i, d, j]> -> !ta.expr<f32, [b, h, i, j]>

    %scale = ta.constant 4.082482904638630e-01 : f32 : !ta.expr<f32, []>
    %s = ta.mulf %scale, %dot
        : (!ta.expr<f32, []>, !ta.expr<f32, [b, h, i, j]>)
       -> !ta.expr<f32, [b, h, i, j]>

    %m = ta.reduce #ta.reduce_kind<max> %s {axes = #ta.axes<j>}
        : !ta.expr<f32, [b, h, i, j]> -> !ta.expr<f32, [b, h, i]>

    %centered = ta.subf %s, %m
        : (!ta.expr<f32, [b, h, i, j]>, !ta.expr<f32, [b, h, i]>)
       -> !ta.expr<f32, [b, h, i, j]>
    %p = ta.exp %centered
        : (!ta.expr<f32, [b, h, i, j]>) -> !ta.expr<f32, [b, h, i, j]>

    %l = ta.reduce #ta.reduce_kind<add> %p {axes = #ta.axes<j>}
        : !ta.expr<f32, [b, h, i, j]> -> !ta.expr<f32, [b, h, i]>

    %v = ta.at %V[%b, %h, %j, %e] {axes = #ta.axes<b, h, j, e>}
        : tensor<2x3x5x7xf32> -> !ta.expr<f32, [b, h, j, e]>
    %pv = ta.mulf %p, %v
        : (!ta.expr<f32, [b, h, i, j]>, !ta.expr<f32, [b, h, j, e]>)
       -> !ta.expr<f32, [b, h, i, j, e]>
    %num = ta.reduce #ta.reduce_kind<add> %pv {axes = #ta.axes<j>}
        : !ta.expr<f32, [b, h, i, j, e]> -> !ta.expr<f32, [b, h, i, e]>

    %o = ta.divf %num, %l
        : (!ta.expr<f32, [b, h, i, e]>, !ta.expr<f32, [b, h, i]>)
       -> !ta.expr<f32, [b, h, i, e]>
    ta.yield %o : !ta.expr<f32, [b, h, i, e]>
  } : () -> tensor<2x3x4x7xf32>

  return %O : tensor<2x3x4x7xf32>
}
