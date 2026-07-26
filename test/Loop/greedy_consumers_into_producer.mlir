// RUN: neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s

// CHECK: remark: failed to fuse this consumer into the producer loop
// CHECK: func.return
// CHECK: remark: did not fuse all discovered consumers into the producer loop
// CHECK: transform.fusion.greedy_consumers_into_producer

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %fused = transform.fusion.greedy_consumers_into_producer %loop[0]
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @unfusable_consumer(%arg: tensor<8xf32>) -> tensor<8xf32> {
    %result = scf.forall (%iv) = (0) to (2) step (1)
        shared_outs(%out = %arg) -> tensor<8xf32> {
      %offset = affine.apply affine_map<(d0) -> (d0 * 4)>(%iv)
      %slice = tensor.extract_slice %arg[%offset] [4] [1]
          : tensor<8xf32> to tensor<4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice into %out[%offset] [4] [1]
            : tensor<4xf32> into tensor<8xf32>
      }
    }
    return %result : tensor<8xf32>
  }
}
