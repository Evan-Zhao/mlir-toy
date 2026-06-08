// RUN: mlir-opt --load-dialect-plugin=%neptune_htile_plugin %s --transform-interpreter 2>&1 | FileCheck %s

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.htile.semantic_to_kernel_abi %func : !any
    transform.yield
  }

  // CHECK-LABEL: func.func @abi(
  // CHECK-SAME: %arg0: memref<4x8xf16>, %arg1: f32, %arg2: memref<4x8xf16>)
  // CHECK-NOT: ->
  func.func @abi(%input: tensor<4x8xf16>, %scale: f32) -> tensor<4x8xf16> {
    // CHECK-DAG: %[[INPUT:.+]] = bufferization.to_tensor %arg0 restrict writable : memref<4x8xf16> to tensor<4x8xf16>
    // CHECK: scf.forall
    // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
    // CHECK-DAG: %[[C0_0:.+]] = arith.constant 0 : index
    // CHECK: %[[LOAD:.+]] = htile.load %arg0[%[[C0]], %[[C0_0]]] : memref<4x8xf16> -> tensor<4x8xf16>
    // CHECK-NOT: tensor.extract_slice
    // CHECK: %[[COPY:.+]] = htile.copy %[[LOAD]] : tensor<4x8xf16> -> tensor<4x8xf16>
    // CHECK: htile.store %[[COPY]], %arg2[%{{.*}}, %{{.*}}] : tensor<4x8xf16>, memref<4x8xf16>
    // CHECK-NOT: tensor.parallel_insert_slice
    %init = tensor.empty() : tensor<4x8xf16>
    %result = scf.forall (%i) in (1) shared_outs(%out = %init) -> (tensor<4x8xf16>) {
      %slice = tensor.extract_slice %input[0, 0] [4, 8] [1, 1]
          : tensor<4x8xf16> to tensor<4x8xf16>
      %copy = htile.copy %slice : tensor<4x8xf16> -> tensor<4x8xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %copy into %out[0, 0] [4, 8] [1, 1]
            : tensor<4x8xf16> into tensor<4x8xf16>
      }
    }
    // CHECK: return
    return %result : tensor<4x8xf16>
  }
}
