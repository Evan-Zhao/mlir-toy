// RUN: neptune-opt %s --transform-interpreter --split-input-file --verify-diagnostics | FileCheck %s

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

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["preserve_initializer"]}
        : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @preserve_shared_out_initializer() -> tensor<4xf32> {
    %init = arith.constant dense<1.0> : tensor<4xf32>
    %result = scf.forall (%i) in (2) shared_outs(%out = %init) -> tensor<4xf32> {
      %tile = arith.constant dense<2.0> : tensor<1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%i] [1] [1]
            : tensor<1xf32> into tensor<4xf32>
      }
    }
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @preserve_shared_out_initializer
// CHECK: %[[INIT:.+]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK: %[[RESULT_BUFFER:.+]] = memref.alloc() : memref<4xf32>
// CHECK: %[[INIT_BUFFER:.+]] = bufferization.to_buffer %[[INIT]]
// CHECK: memref.copy %[[INIT_BUFFER]], %[[RESULT_BUFFER]]
// CHECK: htile.launch_func @preserve_initializer
// CHECK: %[[RESULT:.+]] = bufferization.to_tensor %[[RESULT_BUFFER]]
// CHECK: return %[[RESULT]] : tensor<4xf32>

// CHECK-LABEL: htile.kernel @preserve_initializer
// CHECK: htile.store
// CHECK: htile.return

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["masked_publication"]}
        : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @masked_publication() -> tensor<8x2x4xf32> {
    %empty = tensor.empty() : tensor<8x2x4xf32>
    %result = scf.forall (%head) in (2) shared_outs(%out = %empty)
        -> tensor<8x2x4xf32> {
      %source = arith.constant dense<1.0> : tensor<4x1x4xf32>
      %mask = arith.constant dense<true> : tensor<4x1x4xi1>
      scf.forall.in_parallel {
        htile.masked_parallel_insert_slice
            %source into %out[0, %head, 0] [4, 1, 4] [1, 1, 1]
            mask(%mask : tensor<4x1x4xi1>)
            : tensor<4x1x4xf32> into tensor<8x2x4xf32>
      }
    }
    return %result : tensor<8x2x4xf32>
  }
}

// CHECK-LABEL: func.func @masked_publication
// CHECK: %[[RESULT_BUFFER:.+]] = memref.alloc() : memref<8x2x4xf32>
// CHECK: htile.launch_func @masked_publication_0(%[[RESULT_BUFFER]])
// CHECK: %[[RESULT:.+]] = bufferization.to_tensor %[[RESULT_BUFFER]]
// CHECK: return %[[RESULT]] : tensor<8x2x4xf32>

// CHECK-LABEL: htile.kernel @masked_publication_0
// CHECK-SAME: %[[BUFFER:[^ ]+]] : memref<8x2x4xf32>
// CHECK: %[[HEAD:[^ ]+]] = htile.program_id 0
// CHECK: %[[SOURCE:.+]] = arith.constant dense<1.000000e+00> : tensor<4x1x4xf32>
// CHECK: %[[MASK:.+]] = arith.constant dense<true> : tensor<4x1x4xi1>
// CHECK: htile.store %[[SOURCE]], %[[BUFFER]][%{{.*}}, %[[HEAD]], %{{.*}}]
// CHECK-SAME: mask(%[[MASK]] : tensor<4x1x4xi1>)
// CHECK-SAME: tensor<4x1x4xf32>, memref<8x2x4xf32>
// CHECK: htile.return

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["four_dimensions"]} : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @four_dimensions() {
    scf.forall (%i, %j, %k, %l) in (2, 3, 4, 5) {
      %ij = arith.addi %i, %j : index
      %kl = arith.addi %k, %l : index
      %all = arith.addi %ij, %kl : index
      scf.forall.in_parallel {
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @four_dimensions
// CHECK: htile.launch_func @four_dimensions_0
// CHECK-SAME: {program_bounds = array<i64: 6, 4, 5>}

// CHECK-LABEL: htile.kernel @four_dimensions_0
// CHECK-SAME: attributes {program_bounds = array<i64: 6, 4, 5>}
// CHECK: %[[FLAT:.+]] = htile.program_id 0
// CHECK: %[[THREE:.+]] = arith.constant 3 : index
// CHECK: %[[J:.+]] = arith.remui %[[FLAT]], %[[THREE]] : index
// CHECK: %[[I:.+]] = arith.divui %[[FLAT]], %[[THREE]] : index
// CHECK: %[[K:.+]] = htile.program_id 1
// CHECK: %[[L:.+]] = htile.program_id 2
// CHECK-NOT: htile.program_id 3
// CHECK: arith.addi %[[I]], %[[J]] : index
// CHECK: arith.addi %[[K]], %[[L]] : index
// CHECK: htile.return

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["load_from_tensor"]} : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @load_from_tensor(%input: tensor<16x8xf32>) {
    scf.forall (%i) in (1) {
      %c0 = arith.constant 0 : index
      %tile = htile.load %input[%i, %c0]
          : tensor<16x8xf32> -> tensor<4x8xf32>
      scf.forall.in_parallel {
      }
    }
    return
  }
}

// CHECK-LABEL: htile.kernel @load_from_tensor
// CHECK-SAME: %[[INPUT:.*]] : memref<16x8xf32>
// CHECK-NOT: -> tensor<16x8xf32>
// CHECK: htile.load %[[INPUT]][%{{.*}}, %{{.*}}]
// CHECK-SAME: memref<16x8xf32> -> tensor<4x8xf32>
// CHECK-NOT: htile.load
// CHECK: htile.return

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["extract_from_tensor"]} : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @extract_from_tensor(%input: tensor<16x8xi32>) {
    scf.forall (%i) in (1) {
      %c0 = arith.constant 0 : index
      %scalar = tensor.extract %input[%i, %c0] : tensor<16x8xi32>
      scf.forall.in_parallel {
      }
    }
    return
  }
}

// CHECK-LABEL: htile.kernel @extract_from_tensor
// CHECK-SAME: %[[INPUT:.*]] : memref<16x8xi32>
// CHECK-NOT: tensor.extract
// CHECK: htile.load %[[INPUT]][%{{.*}}, %{{.*}}] : memref<16x8xi32> -> i32
// CHECK-NOT: tensor.extract
// CHECK: htile.return

// -----

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

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    // expected-error @below {{failed to bufferize forall results}}
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["non_unit_masked_stride"]} : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @non_unit_masked_stride() -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %result = scf.forall (%i) in (1) shared_outs(%out = %empty) -> tensor<4xf32> {
      %source = arith.constant dense<1.0> : tensor<2xf32>
      %mask = arith.constant dense<true> : tensor<2xi1>
      scf.forall.in_parallel {
        // expected-error @below {{unsupported non-unit htile.masked_parallel_insert_slice stride}}
        htile.masked_parallel_insert_slice %source into %out[0] [2] [2]
            mask(%mask : tensor<2xi1>) : tensor<2xf32> into tensor<4xf32>
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
    // expected-error @below {{failed to bufferize forall results}}
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["rank_reduced_masked_publication"]} : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @rank_reduced_masked_publication() -> tensor<4x1xf32> {
    %empty = tensor.empty() : tensor<4x1xf32>
    %result = scf.forall (%i) in (1) shared_outs(%out = %empty) -> tensor<4x1xf32> {
      %source = arith.constant dense<1.0> : tensor<4xf32>
      %mask = arith.constant dense<true> : tensor<4xi1>
      scf.forall.in_parallel {
        // expected-error @below {{unsupported rank-reduced masked publication}}
        htile.masked_parallel_insert_slice %source into %out[0, 0] [4, 1] [1, 1]
            mask(%mask : tensor<4xi1>) : tensor<4xf32> into tensor<4x1xf32>
      }
    }
    return %result : tensor<4x1xf32>
  }
}

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["rank_zero_kernel"]} : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @rank_zero_tensor_read(%input: tensor<f32>) {
    scf.forall (%i) in (1) {
      %unused = htile.broadcast %input dimensions = [0] : tensor<f32> -> tensor<4xf32>
      scf.forall.in_parallel {
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @rank_zero_tensor_read
// CHECK: bufferization.to_buffer %arg0 read_only : tensor<f32> to memref<f32>
// CHECK: htile.launch_func @rank_zero_kernel
// CHECK-LABEL: htile.kernel @rank_zero_kernel
// CHECK: %[[SCALAR:.*]] = htile.load %arg0[] : memref<f32> -> tensor<f32>
// CHECK: htile.broadcast %[[SCALAR]] dimensions = [0] : tensor<f32> -> tensor<4xf32>

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %foralls = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    // expected-error @below {{failed to bufferize tensor reads in foralls}}
    %launches, %kernels = transform.htile.outline_kernels %foralls
        {kernel_names = ["unknown_tensor_user"]} : (!any) -> (!any, !any)
    transform.yield
  }

  func.func @reject_unknown_tensor_user(%input: tensor<4xf32>) {
    scf.forall (%i) in (1) {
      // Do not silently materialize the whole `%input` tensor for an unknown
      // tensor operation inside a kernel.
      // expected-error @below {{unsupported tensor read by 'arith.addf' during kernel outlining; expected htile.load, tensor.extract, or tensor.extract_slice}}
      %unused = arith.addf %input, %input : tensor<4xf32>
      scf.forall.in_parallel {
      }
    }
    return
  }
}
