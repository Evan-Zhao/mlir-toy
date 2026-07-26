// RUN: neptune-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @parallel_scatter
  func.func @parallel_scatter(
      %source: tensor<1x4x1x8xf32>, %rows: tensor<1x4x1xi32>,
      %init: tensor<16x2x8xf32>) -> tensor<16x2x8xf32> {
    %c0 = arith.constant 0 : index
    %result = scf.forall (%head) in (2)
        shared_outs(%out = %init) -> tensor<16x2x8xf32> {
      scf.forall.in_parallel {
        // CHECK: htile.parallel_scatter %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] broadcast_dims([2]) unique out_of_bounds <discard>
        htile.parallel_scatter %source into %out[%rows, %head, %c0]
            broadcast_dims([2]) unique out_of_bounds <discard>
            : (tensor<1x4x1x8xf32>, tensor<1x4x1xi32>, index, index) -> (tensor<16x2x8xf32>)
      }
    }
    return %result : tensor<16x2x8xf32>
  }
}
