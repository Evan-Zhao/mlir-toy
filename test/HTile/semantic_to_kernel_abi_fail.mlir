// RUN: not mlir-opt --load-dialect-plugin=%neptune_htile_plugin %s --transform-interpreter 2>&1 | FileCheck %s

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.htile.semantic_to_kernel_abi %func : !any
    transform.yield
  }

  func.func @unsupported_view_chain(%input: tensor<4x8xf16>) {
    // CHECK: expected tensor.extract_slice result to feed only htile ops
    %slice = tensor.extract_slice %input[0, 0] [4, 8] [1, 1]
        : tensor<4x8xf16> to tensor<4x8xf16>
    %collapsed = tensor.collapse_shape %slice [[0, 1]]
        : tensor<4x8xf16> into tensor<32xf16>
    %copy = htile.copy %collapsed : tensor<32xf16> -> tensor<32xf16>
    return
  }
}
