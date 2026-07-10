// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin --load-pass-plugin=%neptune_ta_plugin %s --transform-interpreter 2>&1 | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_var_leading(%candidate: !transform.any_op {transform.readonly})
      -> !transform.any_op {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b ... h i d, b h j d -> b ... h i j"}
        : (!transform.any_op) -> !transform.any_op
    transform.yield %matched : !transform.any_op
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %matches = transform.collect_matching @match_var_leading in %module
        : (!transform.any_op) -> !transform.any_op
    transform.print %matches : !transform.any_op
    transform.yield
  }

  // CHECK: IR printer
  // CHECK: ta.reduce <add> {{.*}} -> !ta.expr<f32, [b, h, i, j]>
  func.func @zero_variadic_leading(%q: tensor<2x3x4x6xf32>,
                                   %k: tensor<2x3x5x6xf32>)
      -> tensor<2x3x4x5xf32> {
    %out = ta.scope axes(%b "b" extent 2, %h "h" extent 3,
                         %i "i" extent 4, %j "j" extent 5,
                         %d "d" extent 6) {
      %q_expr = ta.at %q[%b, %h, %i, %d]
          : tensor<2x3x4x6xf32> -> !ta.expr<f32, [b, h, i, d]>
      %k_expr = ta.at %k[%b, %h, %j, %d]
          : tensor<2x3x5x6xf32> -> !ta.expr<f32, [b, h, j, d]>
      %prod = ta.mul %q_expr, %k_expr
          : (!ta.expr<f32, [b, h, i, d]>, !ta.expr<f32, [b, h, j, d]>)
         -> !ta.expr<f32, [b, h, i, d, j]>
      %dot = ta.reduce #ta.reduce_kind<add> %prod {axes = #ta.axes<d>}
          : !ta.expr<f32, [b, h, i, d, j]> -> !ta.expr<f32, [b, h, i, j]>
      ta.yield %dot : !ta.expr<f32, [b, h, i, j]>
    } : () -> tensor<2x3x4x5xf32>
    return %out : tensor<2x3x4x5xf32>
  }

  // CHECK: ta.reduce <add> {{.*}} -> !ta.expr<f32, [b, g, h, i, j]>
  func.func @one_variadic_leading(%q: tensor<2x7x3x4x6xf32>,
                                  %k: tensor<2x3x5x6xf32>)
      -> tensor<2x7x3x4x5xf32> {
    %out = ta.scope axes(%b "b" extent 2, %g "g" extent 7,
                         %h "h" extent 3, %i "i" extent 4,
                         %j "j" extent 5, %d "d" extent 6) {
      %q_expr = ta.at %q[%b, %g, %h, %i, %d]
          : tensor<2x7x3x4x6xf32> -> !ta.expr<f32, [b, g, h, i, d]>
      %k_expr = ta.at %k[%b, %h, %j, %d]
          : tensor<2x3x5x6xf32> -> !ta.expr<f32, [b, h, j, d]>
      %prod = ta.mul %q_expr, %k_expr
          : (!ta.expr<f32, [b, g, h, i, d]>, !ta.expr<f32, [b, h, j, d]>)
         -> !ta.expr<f32, [b, g, h, i, d, j]>
      %dot = ta.reduce #ta.reduce_kind<add> %prod {axes = #ta.axes<d>}
          : !ta.expr<f32, [b, g, h, i, d, j]> -> !ta.expr<f32, [b, g, h, i, j]>
      ta.yield %dot : !ta.expr<f32, [b, g, h, i, j]>
    } : () -> tensor<2x7x3x4x5xf32>
    return %out : tensor<2x7x3x4x5xf32>
  }
}
