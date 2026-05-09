// RUN: mlir-opt --load-dialect-plugin=%neptune_linalg_ext_plugin %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: func.func @row_max_after_scale
// CHECK: %[[FORALL:.*]]:2 = scf.forall (%[[I:.*]]) in (2) shared_outs(%[[SCORES:.*]] = %{{.*}}, %[[ROWS:.*]] = %{{.*}}) -> (tensor<128x128xf32>, tensor<128xf32>) {
// CHECK: %[[I_OFF:.*]] = affine.apply #map(%[[I]])
// CHECK: %[[PANEL_INIT:.*]] = tensor.extract_slice %[[SCORES]][%[[I_OFF]], %{{.*}}] [64, 128] [1, 1] : tensor<128x128xf32> to tensor<64x128xf32>
// CHECK: %[[ROW_INIT:.*]] = tensor.extract_slice %[[ROWS]][%{{.*}}] [64] [1] : tensor<128xf32> to tensor<64xf32>
// CHECK: %[[FOR:.*]]:2 = scf.for %[[J:.*]] = %c0 to %c2 step %c1 iter_args(%[[PANEL_ARG:.*]] = %[[PANEL_INIT]], %[[ROW_ARG:.*]] = %[[ROW_INIT]]) -> (tensor<64x128xf32>, tensor<64xf32>) {
// CHECK: %[[J_OFF:.*]] = affine.apply #map(%[[J]])
// CHECK: %[[MAPPED:.*]] = linalg.map
// CHECK: %[[INSERTED:.*]] = tensor.insert_slice %[[MAPPED]] into %[[PANEL_ARG]][0, %{{.*}}] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<64x128xf32>
// CHECK: %[[RED:.*]] = linalg.generic
// CHECK: scf.yield %[[INSERTED]], %[[RED]] : tensor<64x128xf32>, tensor<64xf32>
// CHECK: scf.forall.in_parallel {
// CHECK: tensor.parallel_insert_slice %[[FOR]]#0 into %[[SCORES]][%[[I_OFF]], %{{.*}}] [64, 128] [1, 1] : tensor<64x128xf32> into tensor<128x128xf32>
// CHECK: tensor.parallel_insert_slice %[[FOR]]#1 into %[[ROWS]][%{{.*}}] [64] [1] : tensor<64xf32> into tensor<128xf32>
// CHECK: return %[[FORALL]]#0, %[[FORALL]]#1 : tensor<128x128xf32>, tensor<128xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op

    %scale = transform.structured.match ops{["linalg.map"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %scale_tiled, %forall_loop =
      transform.structured.tile_using_forall %scale tile_sizes [64, 64]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %row_max = transform.structured.match ops{["linalg.generic"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %fused, %new_forall, %new_for =
      transform.loop.fuse_reduction_consumer_into_forall %row_max into %forall_loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }

  func.func @row_max_after_scale(%arg0: tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128xf32>) {
    %scale = arith.constant 5.000000e-01 : f32
    %neg_inf = arith.constant -3.40282347E+38 : f32

    %scaled_e = tensor.empty() : tensor<128x128xf32>
    %scaled = linalg.map
        ins(%arg0 : tensor<128x128xf32>)
        outs(%scaled_e : tensor<128x128xf32>)
        (%in: f32, %init: f32) {
          %y = arith.mulf %in, %scale : f32
          linalg.yield %y : f32
    }

    %row_max_e = tensor.empty() : tensor<128xf32>
    %row_max_init = linalg.fill ins(%neg_inf : f32)
        outs(%row_max_e : tensor<128xf32>) -> tensor<128xf32>
    %row_max = linalg.generic {
        indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%scaled : tensor<128x128xf32>)
        outs(%row_max_init : tensor<128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %m = arith.maxnumf %in, %out : f32
        linalg.yield %m : f32
    } -> tensor<128xf32>

    return %scaled, %row_max : tensor<128x128xf32>, tensor<128xf32>
  }
}
