// RUN: mlir-opt --load-dialect-plugin=%neptune_htile_plugin %s --transform-interpreter --split-input-file | FileCheck %s

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["producer", "consumer"]}
        : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @selected_forall_consumer(%input: tensor<4xf32>) -> tensor<4xf32> {
    %empty0 = tensor.empty() : tensor<4xf32>
    %producer = scf.forall (%i) in (4) shared_outs(%out = %empty0) -> tensor<4xf32> {
      %slice = tensor.extract_slice %input[%i] [1] [1]
          : tensor<4xf32> to tensor<1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice into %out[%i] [1] [1]
            : tensor<1xf32> into tensor<4xf32>
      }
    }

    %empty1 = tensor.empty() : tensor<4xf32>
    %consumer = scf.forall (%i) in (4) shared_outs(%out = %empty1) -> tensor<4xf32> {
      %slice = tensor.extract_slice %producer[%i] [1] [1]
          : tensor<4xf32> to tensor<1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice into %out[%i] [1] [1]
            : tensor<1xf32> into tensor<4xf32>
      }
    }

    return %consumer : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @selected_forall_consumer
// CHECK: bufferization.to_buffer
// CHECK-NOT: scf.forall
// CHECK: htile.launch_func @producer
// CHECK-SAME: {program_bounds = array<i64: 4>}
// CHECK-NOT: scf.forall
// CHECK: htile.launch_func @consumer
// CHECK-SAME: {program_bounds = array<i64: 4>}
// CHECK-NOT: scf.forall
// CHECK: return

// CHECK-LABEL: htile.kernel @producer
// CHECK-SAME: memref<4xf32>
// CHECK-SAME: attributes {program_bounds = array<i64: 4>}
// CHECK: htile.program_id 0
// CHECK-NOT: scf.forall
// CHECK: htile.store
// CHECK: htile.return

// CHECK-LABEL: htile.kernel @consumer
// CHECK-SAME: memref<4xf32>
// CHECK-SAME: attributes {program_bounds = array<i64: 4>}
// CHECK: htile.program_id 0
// CHECK-NOT: scf.forall
// CHECK: htile.store
// CHECK: htile.return

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["multi_result"]}
        : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @multi_result_publications(%input0: tensor<4xf32>, %input1: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %empty0 = tensor.empty() : tensor<4xf32>
    %empty1 = tensor.empty() : tensor<4xf32>
    %result:2 = scf.forall (%i) in (4)
        shared_outs(%out0 = %empty0, %out1 = %empty1)
        -> (tensor<4xf32>, tensor<4xf32>) {
      %slice0 = tensor.extract_slice %input0[%i] [1] [1]
          : tensor<4xf32> to tensor<1xf32>
      %slice1 = tensor.extract_slice %input1[%i] [1] [1]
          : tensor<4xf32> to tensor<1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice1 into %out1[%i] [1] [1]
            : tensor<1xf32> into tensor<4xf32>
        tensor.parallel_insert_slice %slice0 into %out0[%i] [1] [1]
            : tensor<1xf32> into tensor<4xf32>
      }
    }
    return %result#0, %result#1 : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @multi_result_publications
// CHECK: bufferization.to_buffer
// CHECK: bufferization.to_buffer
// CHECK-NOT: scf.forall
// CHECK: htile.launch_func @multi_result
// CHECK-SAME: {program_bounds = array<i64: 4>}
// CHECK-NOT: scf.forall
// CHECK: return

// CHECK-LABEL: htile.kernel @multi_result
// CHECK-SAME: memref<4xf32>
// CHECK-SAME: memref<4xf32>
// CHECK-SAME: attributes {program_bounds = array<i64: 4>}
// CHECK: htile.program_id 0
// CHECK-NOT: scf.forall
// CHECK: htile.store
// CHECK: htile.store
// CHECK: htile.return
