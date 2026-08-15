// RUN: neptune-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @minimal_ta
func.func @minimal_ta(%tensor: tensor<16xf32>) -> tensor<16xf32> {
  %0 = ta.scope axes(%coord "i" extent 16) {
    %1 = ta.at %tensor[%coord]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %2 = ta.reduce #ta.reduce_kind<add> %1 {axes = #ta.axes<i>}
        : !ta.expr<f32, [i]> -> !ta.expr<f32, []>
    ta.yield %2 : !ta.expr<f32, []>
  } : () -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @two_axis_ta
func.func @two_axis_ta(%tensor: tensor<16x32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
    %1 = ta.at %tensor[%row, %col]
        : tensor<16x32xf32> -> !ta.expr<f32, [i, j]>
    %2 = ta.reduce #ta.reduce_kind<add> %1 {axes = #ta.axes<j>}
        : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
    ta.yield %2 : !ta.expr<f32, [i]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_ta
func.func @matmul_ta(%lhs: tensor<16x64xf32>, %rhs: tensor<64x32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32, %red "k" extent 64) {
    %x = ta.at %lhs[%row, %red]
        : tensor<16x64xf32> -> !ta.expr<f32, [i, k]>
    %y = ta.at %rhs[%red, %col]
        : tensor<64x32xf32> -> !ta.expr<f32, [k, j]>
    %xy = ta.mul %x, %y
        : (!ta.expr<f32, [i, k]>, !ta.expr<f32, [k, j]>)
       -> !ta.expr<f32, [i, k, j]>
    %dot = ta.reduce #ta.reduce_kind<add> %xy {axes = #ta.axes<k>}
        : !ta.expr<f32, [i, k, j]> -> !ta.expr<f32, [i, j]>
    ta.yield %dot : !ta.expr<f32, [i, j]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @elementwise_ta
func.func @elementwise_ta(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
    %x = ta.at %rows[%row]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %y = ta.at %cols[%col]
        : tensor<32xf32> -> !ta.expr<f32, [j]>
    %sum = ta.add %x, %y
        : (!ta.expr<f32, [i]>, !ta.expr<f32, [j]>)
       -> !ta.expr<f32, [i, j]>
    %scale = ta.constant 2.000000e+00 : f32 : !ta.expr<f32, []>
    %scaled = ta.mul %sum, %scale
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

// -----

// CHECK-LABEL: func.func @float_cast_ta
func.func @float_cast_ta(%tensor: tensor<16xf16>) -> tensor<16xf16> {
  %0 = ta.scope axes(%coord "i" extent 16) {
    %x = ta.at %tensor[%coord]
        : tensor<16xf16> -> !ta.expr<f16, [i]>
    %wide = ta.cast %x
        : (!ta.expr<f16, [i]>) -> !ta.expr<f32, [i]>
    %narrow = ta.cast %wide
        : (!ta.expr<f32, [i]>) -> !ta.expr<f16, [i]>
    ta.yield %narrow : !ta.expr<f16, [i]>
  } : () -> tensor<16xf16>
  return %0 : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @subst_ta
func.func @subst_ta(%tensor: tensor<16xf32>) -> tensor<16xf32> {
  %0 = ta.scope axes(%i "i" extent 16, %j "j" extent 16) {
    %x = ta.at %tensor[%i]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    // CHECK: ta.subst %{{.+}} {from_axes = #ta.axes<i>, to_axes = #ta.axes<j>} : !ta.expr<f32, [i]> -> !ta.expr<f32, [j]>
    %y = ta.subst %x {from_axes = #ta.axes<i>, to_axes = #ta.axes<j>}
        : !ta.expr<f32, [i]> -> !ta.expr<f32, [j]>
    ta.yield %y : !ta.expr<f32, [j]>
  } : () -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @dynamic_extent_ta
func.func @dynamic_extent_ta(%tensor: tensor<?xf32>, %n: index) -> tensor<?xf32> {
  // CHECK: ta.scope axes(%i "i" extent %{{.+}})
  %0 = ta.scope axes(%coord "i" extent %n) {
    %x = ta.at %tensor[%coord]
        : tensor<?xf32> -> !ta.expr<f32, [i]>
    ta.yield %x : !ta.expr<f32, [i]>
  } : () -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func.func @map_ta
func.func @map_ta(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
    %x = ta.at %rows[%row]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %y = ta.at %cols[%col]
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

// -----

// CHECK-LABEL: func.func @attention_ta
func.func @attention_ta(%Q: tensor<2x3x4x6xf32>,
                        %K: tensor<2x3x5x6xf32>,
                        %V: tensor<2x3x5x7xf32>)
    -> tensor<2x3x4x7xf32> {
  %O = ta.scope axes(%b "b" extent 2, %h "h" extent 3,
                     %i "i" extent 4, %j "j" extent 5,
                     %d "d" extent 6, %e "e" extent 7) {
    %q = ta.at %Q[%b, %h, %i, %d]
        : tensor<2x3x4x6xf32> -> !ta.expr<f32, [b, h, i, d]>
    %k = ta.at %K[%b, %h, %j, %d]
        : tensor<2x3x5x6xf32> -> !ta.expr<f32, [b, h, j, d]>
    %qk = ta.mul %q, %k
        : (!ta.expr<f32, [b, h, i, d]>, !ta.expr<f32, [b, h, j, d]>)
       -> !ta.expr<f32, [b, h, i, d, j]>
    %dot = ta.reduce #ta.reduce_kind<add> %qk {axes = #ta.axes<d>}
        : !ta.expr<f32, [b, h, i, d, j]> -> !ta.expr<f32, [b, h, i, j]>

    %scale = ta.constant 4.082482904638630e-01 : f32 : !ta.expr<f32, []>
    %s = ta.mul %scale, %dot
        : (!ta.expr<f32, []>, !ta.expr<f32, [b, h, i, j]>)
       -> !ta.expr<f32, [b, h, i, j]>

    %m = ta.reduce #ta.reduce_kind<max> %s {axes = #ta.axes<j>}
        : !ta.expr<f32, [b, h, i, j]> -> !ta.expr<f32, [b, h, i]>

    %centered = ta.sub %s, %m
        : (!ta.expr<f32, [b, h, i, j]>, !ta.expr<f32, [b, h, i]>)
       -> !ta.expr<f32, [b, h, i, j]>
    %p = ta.exp %centered
        : (!ta.expr<f32, [b, h, i, j]>) -> !ta.expr<f32, [b, h, i, j]>

    %l = ta.reduce #ta.reduce_kind<add> %p {axes = #ta.axes<j>}
        : !ta.expr<f32, [b, h, i, j]> -> !ta.expr<f32, [b, h, i]>

    %v = ta.at %V[%b, %h, %j, %e]
        : tensor<2x3x5x7xf32> -> !ta.expr<f32, [b, h, j, e]>
    %pv = ta.mul %p, %v
        : (!ta.expr<f32, [b, h, i, j]>, !ta.expr<f32, [b, h, j, e]>)
       -> !ta.expr<f32, [b, h, i, j, e]>
    %num = ta.reduce #ta.reduce_kind<add> %pv {axes = #ta.axes<j>}
        : !ta.expr<f32, [b, h, i, j, e]> -> !ta.expr<f32, [b, h, i, e]>

    %o = ta.div %num, %l
        : (!ta.expr<f32, [b, h, i, e]>, !ta.expr<f32, [b, h, i]>)
       -> !ta.expr<f32, [b, h, i, e]>
    ta.yield %o : !ta.expr<f32, [b, h, i, e]>
  } : () -> tensor<2x3x4x7xf32>

  return %O : tensor<2x3x4x7xf32>
}

// -----

func.func @at_outside_scope(%tensor: tensor<16xf32>, %i: index) {
  // expected-error @+1 {{'ta.at' op must be nested inside a ta.scope}}
  %0 = ta.at %tensor[%i]
      : tensor<16xf32> -> !ta.expr<f32, [i]>
  return
}

// -----

func.func @axis_outside_scope(%tensor: tensor<16x32xf32>, %j: index)
    -> tensor<16x32xf32> {
  // expected-error @+1 {{'ta.scope' op yielded expression uses axis 'j' outside enclosing ta.scope axes}}
  %0 = ta.scope axes(%i "i" extent 16) {
    %1 = ta.at %tensor[%i, %j]
        : tensor<16x32xf32> -> !ta.expr<f32, [i, j]>
    ta.yield %1 : !ta.expr<f32, [i, j]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

func.func @bad_elementwise_result_axes(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
    %x = ta.at %rows[%row]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %y = ta.at %cols[%col]
        : tensor<32xf32> -> !ta.expr<f32, [j]>
    // expected-error @+1 {{'ta.add' op result axes must be the ordered union of operand axes; expected #ta.axes<i, j>}}
    %bad = ta.add %x, %y
        : (!ta.expr<f32, [i]>, !ta.expr<f32, [j]>)
       -> !ta.expr<f32, [i]>
    ta.yield %bad : !ta.expr<f32, [i]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

func.func @bad_at_result_axes(%tensor: tensor<16x32xf32>)
    -> tensor<32x16xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
    // expected-error @+1 {{'ta.at' op result axes must match scope-axis indices, except for structural unit axes; expected #ta.axes<i, j>}}
    %x = ta.at %tensor[%row, %col]
        : tensor<16x32xf32> -> !ta.expr<f32, [j, i]>
    ta.yield %x : !ta.expr<f32, [j, i]>
  } : () -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

// -----

func.func @bad_at_non_scope_index(%tensor: tensor<16xf32>, %j: index)
    -> tensor<f32> {
  %0 = ta.scope axes(%coord "i" extent 16) {
    // expected-error @+1 {{'ta.at' op index operands must be ta.scope axes or constant indices}}
    %x = ta.at %tensor[%j]
        : tensor<16xf32> -> !ta.expr<f32, []>
    ta.yield %x : !ta.expr<f32, []>
  } : () -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @bad_map_body_type(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
    %x = ta.at %rows[%row]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %y = ta.at %cols[%col]
        : tensor<32xf32> -> !ta.expr<f32, [j]>
    // expected-error @+1 {{'ta.map' op body yield type must match result expression element type}}
    %bad = ta.map %x, %y {
    ^bb0(%sx : f32, %sy : f32):
      %r = arith.cmpf olt, %sx, %sy : f32
      ta.yield %r : i1
    } : (!ta.expr<f32, [i]>, !ta.expr<f32, [j]>)
     -> !ta.expr<f32, [i, j]>
    ta.yield %bad : !ta.expr<f32, [i, j]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

func.func @bad_reduce_result_axes(%tensor: tensor<16x32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
    %x = ta.at %tensor[%row, %col]
        : tensor<16x32xf32> -> !ta.expr<f32, [i, j]>
    // expected-error @+1 {{'ta.reduce' op result axes must be payload axes minus reduction axes; expected #ta.axes<i>}}
    %bad = ta.reduce #ta.reduce_kind<add> %x {axes = #ta.axes<j>}
        : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [j]>
    ta.yield %bad : !ta.expr<f32, [j]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

func.func @bad_identity_cast(%tensor: tensor<16xf32>) -> tensor<16xf32> {
  %0 = ta.scope axes(%coord "i" extent 16) {
    %x = ta.at %tensor[%coord]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    // expected-error @+1 {{'ta.cast' op requires a non-identity element type conversion}}
    %bad = ta.cast %x
        : (!ta.expr<f32, [i]>) -> !ta.expr<f32, [i]>
    ta.yield %bad : !ta.expr<f32, [i]>
  } : () -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

func.func @bad_subst_missing_source_axis(%tensor: tensor<16xf32>)
    -> tensor<16xf32> {
  %0 = ta.scope axes(%i "i" extent 16, %j "j" extent 32) {
    %x = ta.at %tensor[%i]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    // expected-error @+1 {{'ta.subst' op source axis 'j' is not present in the input axes}}
    %bad = ta.subst %x {from_axes = #ta.axes<j>, to_axes = #ta.axes<i>}
        : !ta.expr<f32, [i]> -> !ta.expr<f32, [i]>
    ta.yield %bad : !ta.expr<f32, [i]>
  } : () -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

func.func @bad_subst_result_axes(%tensor: tensor<16xf32>)
    -> tensor<16xf32> {
  %0 = ta.scope axes(%i "i" extent 16, %j "j" extent 32) {
    %x = ta.at %tensor[%i]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    // expected-error @+1 {{'ta.subst' op result axes must be input axes after substitution; expected #ta.axes<j>}}
    %bad = ta.subst %x {from_axes = #ta.axes<i>, to_axes = #ta.axes<j>}
        : !ta.expr<f32, [i]> -> !ta.expr<f32, [i]>
    ta.yield %bad : !ta.expr<f32, [i]>
  } : () -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

func.func @bad_subst_duplicate_result_axis(%tensor: tensor<16x32xf32>)
    -> tensor<32x16xf32> {
  %0 = ta.scope axes(%i "i" extent 16, %j "j" extent 32) {
    %x = ta.at %tensor[%i, %j]
        : tensor<16x32xf32> -> !ta.expr<f32, [i, j]>
    // expected-error @+1 {{'ta.subst' op substitution produces duplicate axis 'j'}}
    %bad = ta.subst %x {from_axes = #ta.axes<i>, to_axes = #ta.axes<j>}
        : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [j, i]>
    ta.yield %bad : !ta.expr<f32, [j, i]>
  } : () -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

// -----

func.func @bad_subst_extent_mismatch(%tensor: tensor<16xf32>)
    -> tensor<32xf32> {
  %0 = ta.scope axes(%i "i" extent 16, %j "j" extent 32) {
    %x = ta.at %tensor[%i]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    // expected-error @+1 {{'ta.subst' op substituted axes must have equal extents}}
    %bad = ta.subst %x {from_axes = #ta.axes<i>, to_axes = #ta.axes<j>}
        : !ta.expr<f32, [i]> -> !ta.expr<f32, [j]>
    ta.yield %bad : !ta.expr<f32, [j]>
  } : () -> tensor<32xf32>
  return %0 : tensor<32xf32>
}
