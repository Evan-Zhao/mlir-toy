// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @scale_then_bias
// CHECK: %[[FORALL:.*]]:2 = scf.forall
// CHECK-SAME: shared_outs(%[[OUT0:.*]] = %{{.*}}, %[[OUT1:.*]] = %{{.*}}) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
// CHECK: %[[MAPPED:.*]] = linalg.generic
// CHECK: %[[BIASED:.*]] = linalg.generic
// CHECK: scf.forall.in_parallel {
// CHECK: tensor.parallel_insert_slice %[[MAPPED]] into %[[OUT0]]
// CHECK: tensor.parallel_insert_slice %[[BIASED]] into %[[OUT1]]
// CHECK: return %[[FORALL]]#0, %[[FORALL]]#1 : tensor<128x128xf32>, tensor<128x128xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op

    %scale = transform.structured.match ops{["linalg.generic"]} attributes {scale} in %func
        : (!transform.any_op) -> !transform.any_op
    %scale_tiled, %forall_loop =
      transform.structured.tile_using_forall %scale tile_sizes [64, 64]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %bias = transform.structured.match ops{["linalg.generic"]} attributes {bias} in %func
        : (!transform.any_op) -> !transform.any_op
    %fused =
      transform.loop.fuse_into_producer_op %bias into %forall_loop
        : (!transform.any_op, !transform.any_op) -> !transform.any_op

    transform.yield
  }

  func.func @scale_then_bias(%arg0: tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %scale = arith.constant 5.000000e-01 : f32
    %bias = arith.constant 1.000000e+00 : f32

    %scaled_e = tensor.empty() : tensor<128x128xf32>
    %scaled = linalg.generic {scale,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0 : tensor<128x128xf32>)
        outs(%scaled_e : tensor<128x128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %y = arith.mulf %in, %scale : f32
        linalg.yield %y : f32
    } -> tensor<128x128xf32>

    %biased_e = tensor.empty() : tensor<128x128xf32>
    %biased = linalg.generic {bias,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%scaled : tensor<128x128xf32>)
        outs(%biased_e : tensor<128x128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %y = arith.addf %in, %bias : f32
        linalg.yield %y : f32
    } -> tensor<128x128xf32>

    return %scaled, %biased : tensor<128x128xf32>, tensor<128x128xf32>
  }
}
