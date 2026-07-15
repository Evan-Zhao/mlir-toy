// RUN: not neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s

#id = affine_map<(d0) -> (d0)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.htile.linalg_to_semantic %funcs : !transform.any_op
    transform.yield
  }

  func.func @unsupported(
      %input: tensor<4xf32>,
      %out0: tensor<4xf32>,
      %out1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    // CHECK: unsupported linalg.generic operation
    %0:2 = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>)
        outs(%out0, %out1 : tensor<4xf32>, tensor<4xf32>) {
    ^bb0(%in: f32, %unused0: f32, %unused1: f32):
      linalg.yield %in, %in : f32, f32
    } -> (tensor<4xf32>, tensor<4xf32>)
    return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }
}
