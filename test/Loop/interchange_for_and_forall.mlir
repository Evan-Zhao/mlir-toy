// RUN: neptune-opt %s -o /dev/null
// RUN: neptune-opt %s --split-input-file --transform-interpreter --verify-diagnostics | FileCheck %s
// CHECK-LABEL: func.func @perfectly_nested(
// CHECK-SAME: %[[INIT0:.+]]: tensor<4xf32>, %[[INIT1:.+]]: tensor<4xf32>)
// CHECK: %[[LB:.+]] = arith.constant 0 : index
// CHECK: %[[STEP:.+]] = arith.constant 2 : index
// CHECK: scf.forall
// CHECK-SAME: shared_outs(%{{.+}} = %[[INIT0]], %{{.+}} = %[[INIT1]])
// CHECK: %[[WORKER0:.+]] = tensor.extract_slice %[[INIT0]][%[[LB]]] [2] [%[[STEP]]]
// CHECK: %[[WORKER_OFFSET:.+]] = affine.apply {{.*}}(%{{.+}})
// CHECK: %[[WORKER1:.+]] = tensor.extract_slice %[[INIT1]][%[[WORKER_OFFSET]]] [1] [1]
// CHECK: %[[INNER:.+]]:2 = scf.for %[[NORMALIZED_IV:[^ ]+]] =
// CHECK-SAME: iter_args(%[[CARRY0:.+]] = %[[WORKER0]], %[[CARRY1:.+]] = %[[WORKER1]])
// CHECK: affine.apply {{.*}}(%[[NORMALIZED_IV]])
// CHECK: %[[POINT:.+]] = tensor.extract_slice %[[CARRY0]][%[[NORMALIZED_IV]]] [1] [1]
// CHECK: %[[UPDATED:.+]] = arith.addf %[[POINT]], %[[POINT]] : tensor<1xf32>
// CHECK: %[[NEXT0:.+]] = tensor.insert_slice %[[UPDATED]] into %[[CARRY0]][%[[NORMALIZED_IV]]]
// CHECK: scf.yield %[[NEXT0]], %[[CARRY1]]
// CHECK: scf.forall.in_parallel
// CHECK: tensor.parallel_insert_slice %[[INNER]]#0
// CHECK: tensor.parallel_insert_slice %[[INNER]]#1

#identity = affine_map<(d0) -> (d0)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @perfectly_nested(%init0: tensor<4xf32>, %init1: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %result:2 = scf.for %i = %c0 to %c4 step %c2
        iter_args(%carried0 = %init0, %carried1 = %init1)
        -> (tensor<4xf32>, tensor<4xf32>) {
      %next:2 = scf.forall (%j) in (1)
          shared_outs(%out0 = %carried0, %out1 = %carried1)
          -> (tensor<4xf32>, tensor<4xf32>) {
        %tile0 = tensor.extract_slice %carried0[%i] [1] [1]
            : tensor<4xf32> to tensor<1xf32>
        %offset = affine.apply #identity(%j)
        %tile1 = tensor.extract_slice %carried1[%offset] [1] [1]
            : tensor<4xf32> to tensor<1xf32>
        %updated0 = arith.addf %tile0, %tile0 : tensor<1xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %updated0 into %out0[%i] [1] [1]
              : tensor<1xf32> into tensor<4xf32>
          tensor.parallel_insert_slice %tile1 into %out1[%offset] [1] [1]
              : tensor<1xf32> into tensor<4xf32>
        }
      }
      scf.yield %next#0, %next#1 : tensor<4xf32>, tensor<4xf32>
    }
    return %result#0, %result#1 : tensor<4xf32>, tensor<4xf32>
  }
}
// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected the scf.forall to be directly nested in the scf.for}}
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @not_nested() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
    }
    scf.forall (%j) in (1) {
      scf.forall.in_parallel {}
    }
    return
  }
}
// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected a perfect nest with the scf.forall as the only non-terminator operation in the scf.for}}
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @not_perfectly_nested() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
      %unused = arith.addi %i, %c1 : index
      scf.forall (%j) in (1) {
        scf.forall.in_parallel {}
      }
    }
    return
  }
}
// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected the scf.for induction variable to have index type}}
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @non_index_induction() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c1 step %c1 : i32 {
      scf.forall (%j) in (1) {
        scf.forall.in_parallel {}
      }
    }
    return
  }
}
// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected the scf.for step to be a positive constant}}
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @dynamic_step(%step: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %step {
      scf.forall (%j) in (1) {
        scf.forall.in_parallel {}
      }
    }
    return
  }
}
// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to build a localizable loop-carried result plan}}
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @two_varying_dimensions(
      %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %c1 step %c1
        iter_args(%carried = %init) -> (tensor<4x4xf32>) {
      %next = scf.forall (%j) in (1) shared_outs(%out = %carried)
          -> (tensor<4x4xf32>) {
        %tile = tensor.extract_slice %carried[%i, %i] [1, 1] [1, 1]
            : tensor<4x4xf32> to tensor<1x1xf32>
        scf.forall.in_parallel {
          // expected-error @below {{expected at most one publication dimension to vary with the scf.for}}
          tensor.parallel_insert_slice %tile into %out[%i, %i] [1, 1] [1, 1]
              : tensor<1x1xf32> into tensor<4x4xf32>
        }
      }
      scf.yield %next : tensor<4x4xf32>
    }
    return %result : tensor<4x4xf32>
  }
}
// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to build a localizable loop-carried result plan}}
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @offset_expression(%init: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %c1 step %c1
        iter_args(%carried = %init) -> (tensor<4xf32>) {
      %next = scf.forall (%j) in (1) shared_outs(%out = %carried)
          -> (tensor<4xf32>) {
        // expected-error @below {{expected a varying publication offset to be exactly the scf.for induction variable}}
        %offset = arith.addi %i, %c0 : index
        %tile = tensor.extract_slice %carried[%offset] [1] [1]
            : tensor<4xf32> to tensor<1xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %tile into %out[%offset] [1] [1]
              : tensor<1xf32> into tensor<4xf32>
        }
      }
      scf.yield %next : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}
// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to build a localizable loop-carried result plan}}
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @varying_size_two(%init: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %c1 step %c1
        iter_args(%carried = %init) -> (tensor<4xf32>) {
      %next = scf.forall (%j) in (1) shared_outs(%out = %carried)
          -> (tensor<4xf32>) {
        %tile = tensor.extract_slice %carried[%i] [2] [1]
            : tensor<4xf32> to tensor<2xf32>
        scf.forall.in_parallel {
          // expected-error @below {{expected a publication varying with the scf.for to have size one in that dimension}}
          tensor.parallel_insert_slice %tile into %out[%i] [2] [1]
              : tensor<2xf32> into tensor<4xf32>
        }
      }
      scf.yield %next : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}
// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to build a localizable loop-carried result plan}}
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @iv_dependent_size(%init: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %c1 step %c1
        iter_args(%carried = %init) -> (tensor<4xf32>) {
      %next = scf.forall (%j) in (1) shared_outs(%out = %carried)
          -> (tensor<4xf32>) {
        // expected-error @below {{expected publication size to be invariant under the scf.for}}
        %size = arith.addi %i, %c1 : index
        %tile = tensor.extract_slice %carried[0] [%size] [1]
            : tensor<4xf32> to tensor<?xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %tile into %out[0] [%size] [1]
              : tensor<?xf32> into tensor<4xf32>
        }
      }
      scf.yield %next : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}
// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to build a localizable loop-carried result plan}}
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @missing_publication(%init: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %c1 step %c1
        iter_args(%carried = %init) -> (tensor<4xf32>) {
      // expected-error @below {{expected every relayed scf.forall result to have one tensor.parallel_insert_slice combining op}}
      %next = scf.forall (%j) in (1) shared_outs(%out = %carried)
          -> (tensor<4xf32>) {
        scf.forall.in_parallel {}
      }
      scf.yield %next : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}
// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %for = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected every scf.forall result to relay to exactly one scf.for result}}
    %outer_forall, %inner_for = transform.scf.interchange_for_and_forall %for with %forall
        : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @duplicate_relay(%init0: tensor<4xf32>, %init1: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result:2 = scf.for %i = %c0 to %c1 step %c1
        iter_args(%carried0 = %init0, %carried1 = %init1)
        -> (tensor<4xf32>, tensor<4xf32>) {
      %next = scf.forall (%j) in (1) shared_outs(%out = %carried0)
          -> (tensor<4xf32>) {
        %tile = tensor.extract_slice %carried0[0] [1] [1]
            : tensor<4xf32> to tensor<1xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %tile into %out[0] [1] [1]
              : tensor<1xf32> into tensor<4xf32>
        }
      }
      scf.yield %next, %next : tensor<4xf32>, tensor<4xf32>
    }
    return %result#0, %result#1 : tensor<4xf32>, tensor<4xf32>
  }
}
