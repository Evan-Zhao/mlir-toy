// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin %s --canonicalize | FileCheck %s

module {
  // CHECK-LABEL: func.func @fold_truncf_of_constant
  func.func @fold_truncf_of_constant() -> tensor<1xf32> {
    %0 = ta.scope axes(%i "i" extent 1) {
      // CHECK: ta.constant 0.0883883461 : f32 {ta.import_group = 4 : i64} : !ta.expr<f32, []>
      // CHECK-NOT: ta.truncf
      %c = ta.constant 0.088388347648318433 : f64 {ta.import_group = 4 : i64} : !ta.expr<f64, []>
      %scale = ta.truncf %c {ta.import_group = 4 : i64}
          : (!ta.expr<f64, []>) -> !ta.expr<f32, []>
      ta.yield %scale : !ta.expr<f32, []>
    } : () -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
