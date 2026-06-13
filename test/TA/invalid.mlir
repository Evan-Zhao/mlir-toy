// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s --verify-diagnostics

func.func @at_outside_scope(%tensor: tensor<16xf32>, %i: index) {
  // expected-error @+1 {{'ta.at' op must be nested inside a ta.scope}}
  %0 = ta.at %tensor[%i]
      : tensor<16xf32> -> !ta.expr<f32, [i]>
  return
}

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

func.func @bad_elementwise_result_axes(%rows: tensor<16xf32>, %cols: tensor<32xf32>)
    -> tensor<16x32xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
    %x = ta.at %rows[%row]
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %y = ta.at %cols[%col]
        : tensor<32xf32> -> !ta.expr<f32, [j]>
    // expected-error @+1 {{'ta.addf' op result axes must be the ordered union of operand axes; expected #ta.axes<i, j>}}
    %bad = ta.addf %x, %y
        : (!ta.expr<f32, [i]>, !ta.expr<f32, [j]>)
       -> !ta.expr<f32, [i]>
    ta.yield %bad : !ta.expr<f32, [i]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

func.func @bad_at_result_axes(%tensor: tensor<16x32xf32>)
    -> tensor<32x16xf32> {
  %0 = ta.scope axes(%row "i" extent 16, %col "j" extent 32) {
    // expected-error @+1 {{'ta.at' op result axes must match scope-axis indices; expected #ta.axes<i, j>}}
    %x = ta.at %tensor[%row, %col]
        : tensor<16x32xf32> -> !ta.expr<f32, [j, i]>
    ta.yield %x : !ta.expr<f32, [j, i]>
  } : () -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

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
