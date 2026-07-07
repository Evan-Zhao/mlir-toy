// RUN: not mlir-opt --load-dialect-plugin=%neptune_htile_plugin %s --transform-interpreter 2>&1 | FileCheck %s

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %producer, %rest = transform.split_handle %foralls {overflow_result = 1}
        : (!any) -> (!any, !any)
    %launches, %kernels = transform.htile.outline_kernels %producer
        {kernel_names = ["producer"]}
        : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @unselected_forall_consumer(%input: tensor<4xf32>) -> tensor<4xf32> {
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
      // CHECK: Current function:
      // CHECK: bufferization.to_tensor
      // CHECK: transform.htile.outline_kernels is not implemented yet
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
