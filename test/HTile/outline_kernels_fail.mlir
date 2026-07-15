// RUN: neptune-opt %s --transform-interpreter --split-input-file --verify-diagnostics

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    // expected-error @below {{failed to validate selected scf.forall ops}}
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["nested"]} : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @nested_forall(%flag: i1) {
    scf.if %flag {
      // expected-error @below {{expected selected scf.forall to be a top-level op directly inside func.func}}
      scf.forall (%i) in (4) {
        scf.forall.in_parallel {
        }
      }
    }
    return
  }
}

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    // expected-error @below {{failed to create htile.kernel ops}}
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["scalar_capture"]} : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @unsupported_scalar_capture(%scale: f32) -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    // expected-error @below {{unsupported non-memref kernel capture}}
    %result = scf.forall (%i) in (4) shared_outs(%out = %empty) -> tensor<4xf32> {
      %tile_empty = tensor.empty() : tensor<1xf32>
      %tile = linalg.fill ins(%scale : f32) outs(%tile_empty : tensor<1xf32>)
          -> tensor<1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%i] [1] [1]
            : tensor<1xf32> into tensor<4xf32>
      }
    }
    return %result : tensor<4xf32>
  }
}

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    // expected-error @below {{failed to create htile.kernel ops}}
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["dynamic_bounds"]} : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @dynamic_bounds(%n: index) {
    // expected-error @below {{expected static lower/upper/step for forall dimension 0}}
    scf.forall (%i) in (%n) {
      scf.forall.in_parallel {
      }
    }
    return
  }
}
