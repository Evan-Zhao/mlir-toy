// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s | FileCheck %s

// CHECK-LABEL: func.func @attention_ta
func.func @attention_ta(%Q: tensor<2x3x4x6xf32>,
                        %K: tensor<2x3x5x6xf32>,
                        %V: tensor<2x3x5x7xf32>)
    -> tensor<2x3x4x7xf32> {
  %O = ta.scope axes(%b "b" : index, %h "h" : index,
                     %i "i" : index, %j "j" : index,
                     %d "d" : index, %e "e" : index) {
    %dot = ta.reduce <add> {
      %q = ta.at %Q[%b, %h, %i, %d] {axes = #ta.axes<b, h, i, d>}
          : tensor<2x3x4x6xf32> -> !ta.expr<f32, [b, h, i, d]>
      %k = ta.at %K[%b, %h, %j, %d] {axes = #ta.axes<b, h, j, d>}
          : tensor<2x3x5x6xf32> -> !ta.expr<f32, [b, h, j, d]>
      %qk = ta.mulf %q, %k
          : (!ta.expr<f32, [b, h, i, d]>, !ta.expr<f32, [b, h, j, d]>)
         -> !ta.expr<f32, [b, h, i, j, d]>
      ta.yield %qk : !ta.expr<f32, [b, h, i, j, d]>
    } {axes = #ta.axes<d>} : !ta.expr<f32, [b, h, i, j]>

    %scale = ta.constant 4.082482904638630e-01 : f32 : !ta.expr<f32, []>
    %s = ta.mulf %scale, %dot
        : (!ta.expr<f32, []>, !ta.expr<f32, [b, h, i, j]>)
       -> !ta.expr<f32, [b, h, i, j]>

    %m = ta.reduce <max> {
      ta.yield %s : !ta.expr<f32, [b, h, i, j]>
    } {axes = #ta.axes<j>} : !ta.expr<f32, [b, h, i]>

    %centered = ta.subf %s, %m
        : (!ta.expr<f32, [b, h, i, j]>, !ta.expr<f32, [b, h, i]>)
       -> !ta.expr<f32, [b, h, i, j]>
    %p = ta.exp %centered
        : (!ta.expr<f32, [b, h, i, j]>) -> !ta.expr<f32, [b, h, i, j]>

    %l = ta.reduce <add> {
      ta.yield %p : !ta.expr<f32, [b, h, i, j]>
    } {axes = #ta.axes<j>} : !ta.expr<f32, [b, h, i]>

    %num = ta.reduce <add> {
      %v = ta.at %V[%b, %h, %j, %e] {axes = #ta.axes<b, h, j, e>}
          : tensor<2x3x5x7xf32> -> !ta.expr<f32, [b, h, j, e]>
      %pv = ta.mulf %p, %v
          : (!ta.expr<f32, [b, h, i, j]>, !ta.expr<f32, [b, h, j, e]>)
         -> !ta.expr<f32, [b, h, i, j, e]>
      ta.yield %pv : !ta.expr<f32, [b, h, i, j, e]>
    } {axes = #ta.axes<j>} : !ta.expr<f32, [b, h, i, e]>

    %o = ta.divf %num, %l
        : (!ta.expr<f32, [b, h, i, e]>, !ta.expr<f32, [b, h, i]>)
       -> !ta.expr<f32, [b, h, i, e]>
    ta.yield %o : !ta.expr<f32, [b, h, i, e]>
  } : () -> tensor<2x3x4x7xf32>

  return %O : tensor<2x3x4x7xf32>
}
