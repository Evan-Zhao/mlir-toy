// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s | FileCheck %s

// CHECK-LABEL: func.func @minimal_ta
func.func @minimal_ta(%tensor: tensor<16xf32>) -> tensor<16xf32> {
  %0 = ta.scope axes(%coord "i" : index) {
    %1 = ta.at %tensor[%coord] {axes = #ta.axes<i>}
        : tensor<16xf32> -> !ta.expr<f32, [i]>
    %2 = ta.reduce %1 {axes = #ta.axes<i>, kind = "add"}
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
    %2 = ta.reduce %1 {axes = #ta.axes<j>, kind = "add"}
        : !ta.expr<f32, [i, j]> -> !ta.expr<f32, [i]>
    ta.yield %2 : !ta.expr<f32, [i]>
  } : () -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}
