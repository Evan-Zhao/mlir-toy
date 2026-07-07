// RUN: mlir-opt --load-dialect-plugin=%neptune_htile_plugin %s | FileCheck %s

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %launches, %kernels = transform.htile.outline_kernels %module
        {kernel_names = ["decode_partial", "decode_merge"]}
        : (!any) -> (!any, !any)
    transform.yield
  }
}

// CHECK: transform.htile.outline_kernels
// CHECK-SAME: kernel_names = ["decode_partial", "decode_merge"]
