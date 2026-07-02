// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter | FileCheck %s

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "no_dead_tile"} in %module : (!any) -> !any
    %loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %live_loop, %mixed_loop =
      transform.loop.specialize_dead_tile in %loop
        {dead_value = 0xFF800000 : f32} : (!any) -> (!any, !any)
    transform.yield
  }

  // CHECK-LABEL: func.func @no_dead_tile(
  // CHECK: scf.for
  // CHECK-NOT: linalg.generic
  func.func @no_dead_tile(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %result = scf.for %i = %c0 to %c16 step %c1
        iter_args(%acc = %arg0) -> tensor<16xf32> {
      scf.yield %acc : tensor<16xf32>
    }
    return %result : tensor<16xf32>
  }
}
