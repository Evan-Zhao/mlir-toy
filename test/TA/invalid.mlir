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
