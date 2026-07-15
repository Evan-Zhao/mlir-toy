// RUN: neptune-opt %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: func.func @row_max_after_scale
// CHECK: %[[FORALL:.*]]:2 = scf.forall (%[[I:.*]], %[[J:.*]]) in (2, 2) shared_outs(%[[SCORES:.*]] = %{{.*}}, %[[PARTIALS:.*]] = %{{.*}}) -> (tensor<128x128xf32>, tensor<128x2xf32>) {
// CHECK: %[[I_OFF:.*]] = affine.apply #map(%[[I]])
// CHECK: %[[J_OFF:.*]] = affine.apply #map(%[[J]])
// CHECK: %[[MAPPED:.*]] = linalg.map
// CHECK: %[[PARTIAL_INIT:.*]] = tensor.extract_slice %[[PARTIALS]][%[[I_OFF]], %[[J]]] [64, 1] [1, 1] : tensor<128x2xf32> to tensor<64xf32>
// CHECK: %[[PARTIAL_REDUCE:.*]] = linalg.generic
// CHECK: scf.forall.in_parallel {
// CHECK: tensor.parallel_insert_slice %[[MAPPED]] into %[[SCORES]][%[[I_OFF]], %[[J_OFF]]] [64, 64] [1, 1]
// CHECK: tensor.parallel_insert_slice %[[PARTIAL_REDUCE]] into %[[PARTIALS]][%[[I_OFF]], %[[J]]] [64, 1] [1, 1] : tensor<64xf32> into tensor<128x2xf32>
// CHECK: %[[WRITEBACK:.*]] = linalg.reduce ins(%[[FORALL]]#1 : tensor<128x2xf32>) outs(%{{.*}} : tensor<128xf32>) dimensions = [1]
// CHECK: return %[[FORALL]]#0, %[[WRITEBACK]] : tensor<128x128xf32>, tensor<128xf32>

#map = affine_map<(d0) -> (d0 * 64)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op

    %forall_loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op

    %row_max = transform.structured.match ops{["linalg.generic"]} in %func
        : (!transform.any_op) -> !transform.any_op

    %partial, %writeback =
      transform.scf.fuse_partial_reduction_into_forall %row_max into %forall_loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }

  func.func @row_max_after_scale(%arg0: tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128xf32>) {
    %scale = arith.constant 5.000000e-01 : f32
    %neg_inf = arith.constant -3.40282347E+38 : f32

    %scaled_e = tensor.empty() : tensor<128x128xf32>
    %scaled = scf.forall (%arg1, %arg2) in (2, 2) shared_outs(%arg3 = %scaled_e) -> (tensor<128x128xf32>) {
      %row_offset = affine.apply #map(%arg1)
      %col_offset = affine.apply #map(%arg2)
      %src = tensor.extract_slice %arg0[%row_offset, %col_offset] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
      %dst = tensor.extract_slice %arg3[%row_offset, %col_offset] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
      %mapped = linalg.map ins(%src : tensor<64x64xf32>) outs(%dst : tensor<64x64xf32>)
        (%in: f32, %init: f32) {
          %y = arith.mulf %in, %scale : f32
          linalg.yield %y : f32
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %mapped into %arg3[%row_offset, %col_offset] [64, 64] [1, 1]
          : tensor<64x64xf32> into tensor<128x128xf32>
      }
    }

    %row_max_e = tensor.empty() : tensor<128xf32>
    %row_max_init = linalg.fill ins(%neg_inf : f32)
        outs(%row_max_e : tensor<128xf32>) -> tensor<128xf32>
    %row_max = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]
    } ins(%scaled#0 : tensor<128x128xf32>) outs(%row_max_init : tensor<128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %m = arith.maxnumf %in, %out : f32
        linalg.yield %m : f32
    } -> tensor<128xf32>

    return %scaled#0, %row_max : tensor<128x128xf32>, tensor<128xf32>
  }
}
