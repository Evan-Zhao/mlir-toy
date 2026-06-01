// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s --verify-diagnostics

func.func @at_outside_scope(%tensor: tensor<16xf32>, %i: index) {
  // expected-error @+1 {{'ta.at' op must be nested inside a ta.scope}}
  %0 = ta.at %tensor[%i] {axes = #ta.axes<i>}
      : tensor<16xf32> -> !ta.expr<f32, [i]>
  return
}

func.func @axis_outside_scope(%tensor: tensor<16x32xf32>, %j: index)
    -> tensor<16x32xf32> {
  // expected-error @+1 {{'ta.scope' op yielded expression uses axis 'j' outside enclosing ta.scope axes}}
  %0 = ta.scope axes(%i "i" : index) {
    %1 = ta.at %tensor[%i, %j] {axes = #ta.axes<i, j>}
        : tensor<16x32xf32> -> !ta.expr<f32, [i, j]>
    ta.yield %1 : !ta.expr<f32, [i, j]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

func.func @bad_elementwise_result_axes(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" : index, %col "j" : index) {
    %x = ta.at %rows[%row] {axes = #ta.axes<i>}
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %y = ta.at %cols[%col] {axes = #ta.axes<j>}
        : tensor<32xf32> -> !ta.expr<f32, [j]>
    // expected-error @+1 {{'ta.addf' op result axes must be the union of operand axes in enclosing ta.scope order; expected #ta.axes<i, j>}}
    %bad = ta.addf %x, %y
        : (!ta.expr<f32, [i]>, !ta.expr<f32, [j]>)
       -> !ta.expr<f32, [i]>
    ta.yield %bad : !ta.expr<f32, [i]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

func.func @bad_map_body_type(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" : index, %col "j" : index) {
    %x = ta.at %rows[%row] {axes = #ta.axes<i>}
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %y = ta.at %cols[%col] {axes = #ta.axes<j>}
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

func.func @bad_reduce_result_axes(%tensor: tensor<16x32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" : index, %col "j" : index) {
    %x = ta.at %tensor[%row, %col] {axes = #ta.axes<i, j>}
        : tensor<16x32xf32> -> !ta.expr<f32, [i, j]>
    // expected-error @+1 {{'ta.map_reduce' op result axes must be payload axes minus reduction axes; expected #ta.axes<i>}}
    %bad = ta.map_reduce #ta.reduce_kind<add> {
      ta.yield %x : !ta.expr<f32, [i, j]>
    } {axes = #ta.axes<j>} : !ta.expr<f32, [j]>
    ta.yield %bad : !ta.expr<f32, [j]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}
