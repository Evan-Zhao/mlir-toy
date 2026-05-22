// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @inline_then_tile
// CHECK: scf.forall
// CHECK: linalg.generic
// CHECK: arith.extf
// CHECK: arith.mulf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %consumer = transform.structured.match ops{["linalg.generic"]} attributes {consumer} in %func
        : (!transform.any_op) -> !transform.any_op

    transform.linalg.greedy_inline_elementwise %consumer : !transform.any_op
    %tiled, %forall = transform.structured.tile_using_forall %consumer tile_sizes [4, 4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }

  func.func @inline_then_tile(%arg0: tensor<8x8xf16>) -> tensor<8x8xf32> {
    %cst = arith.constant 2.0 : f32

    %cast_init = tensor.empty() : tensor<8x8xf32>
    %casted = linalg.generic {producer,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0 : tensor<8x8xf16>)
        outs(%cast_init : tensor<8x8xf32>) {
      ^bb0(%in: f16, %out: f32):
        %ext = arith.extf %in : f16 to f32
        linalg.yield %ext : f32
    } -> tensor<8x8xf32>

    %scale_init = tensor.empty() : tensor<8x8xf32>
    %scaled = linalg.generic {consumer,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%casted : tensor<8x8xf32>)
        outs(%scale_init : tensor<8x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %mul = arith.mulf %in, %cst : f32
        linalg.yield %mul : f32
    } -> tensor<8x8xf32>

    return %scaled : tensor<8x8xf32>
  }
}
