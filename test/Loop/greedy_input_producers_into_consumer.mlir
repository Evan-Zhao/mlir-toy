// RUN: neptune-opt %s --split-input-file --transform-interpreter | FileCheck %s

// CHECK-LABEL: func.func @producer_path
// CHECK: %[[INIT:.*]] = linalg.fill {unrelated_init}
// CHECK: scf.forall
// CHECK-SAME: shared_outs(%{{.*}} = %[[INIT]])
// CHECK-NOT: unrelated_init
// CHECK: %[[ROOT:.*]] = linalg.generic
// CHECK-SAME: {fusion_root}
// CHECK: %[[STEP:.*]] = linalg.generic
// CHECK-SAME: ins(%[[ROOT]]
// CHECK-SAME: {fusion_step}
// CHECK: tensor.parallel_insert_slice %[[STEP]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %candidates, %new_loop =
        transform.fusion.greedy_input_producers_into_consumer %loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @producer_path(%arg: tensor<8xf32>) -> tensor<8xf32> {
    %root_init = tensor.empty() : tensor<8xf32>
    %root = linalg.generic {
        fusion_root,
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%arg : tensor<8xf32>) outs(%root_init : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %one = arith.constant 1.0 : f32
        %sum = arith.addf %in, %one : f32
        linalg.yield %sum : f32
    } -> tensor<8xf32>

    %step_init = tensor.empty() : tensor<8xf32>
    %step = linalg.generic {
        fusion_step,
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%root : tensor<8xf32>) outs(%step_init : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %two = arith.constant 2.0 : f32
        %product = arith.mulf %in, %two : f32
        linalg.yield %product : f32
    } -> tensor<8xf32>

    %loop_empty = tensor.empty() : tensor<8xf32>
    %zero = arith.constant 0.0 : f32
    %loop_init = linalg.fill {unrelated_init} ins(%zero : f32)
        outs(%loop_empty : tensor<8xf32>) -> tensor<8xf32>
    %result = scf.forall (%iv) = (0) to (2) step (1)
        shared_outs(%out = %loop_init) -> tensor<8xf32> {
      %offset = affine.apply affine_map<(d0) -> (d0 * 4)>(%iv)
      %slice = tensor.extract_slice %step[%offset] [4] [1]
          : tensor<8xf32> to tensor<4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice into %out[%offset] [4] [1]
            : tensor<4xf32> into tensor<8xf32>
      }
    }
    return %result : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @nested_loop_path
// CHECK: linalg.generic
// CHECK-SAME: {fusion_root}
// CHECK: scf.forall
// The outer use is fused into the forall, not into the inner loop.
// CHECK: %[[OUTER_ROOT:.*]] = linalg.generic
// CHECK-SAME: {fusion_root}
// CHECK: scf.for
// The independent inner use is fused again at its deeper placement level.
// CHECK: %[[INNER_ROOT:.*]] = linalg.generic
// CHECK-SAME: {fusion_root}
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[INNER_ROOT]]
// CHECK-SAME: {inner_consumer}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %for = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    // Deliberately merge in inner-to-outer order. The fusion op infers the
    // loop order from payload nesting.
    %loop_nest = transform.merge_handles %for, %forall : !transform.any_op
    %candidates, %new_loop_nest =
        transform.fusion.greedy_input_producers_into_consumer %loop_nest
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @nested_loop_path(%arg: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %root_init = tensor.empty() : tensor<8x8xf32>
    %root = linalg.generic {
        fusion_root,
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg : tensor<8x8xf32>) outs(%root_init : tensor<8x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %one = arith.constant 1.0 : f32
        %sum = arith.addf %in, %one : f32
        linalg.yield %sum : f32
    } -> tensor<8x8xf32>

    %result = scf.forall (%i) = (0) to (2) step (1)
        shared_outs(%out = %root_init) -> tensor<8x8xf32> {
      %i_offset = affine.apply affine_map<(d0) -> (d0 * 4)>(%i)
      %outer_slice = tensor.extract_slice %root[%i_offset, 0] [4, 8] [1, 1]
          : tensor<8x8xf32> to tensor<4x8xf32>
      scf.for %j = %c0 to %c8 step %c4 {
        %inner_slice = tensor.extract_slice %outer_slice[0, %j] [4, 4] [1, 1]
            : tensor<4x8xf32> to tensor<4x4xf32>
        %inner_init = tensor.empty() : tensor<4x4xf32>
        %inner = linalg.generic {
            inner_consumer,
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                             affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%inner_slice : tensor<4x4xf32>)
            outs(%inner_init : tensor<4x4xf32>) {
          ^bb0(%in: f32, %inner_out: f32):
            linalg.yield %in : f32
        } -> tensor<4x4xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %outer_slice into %out[%i_offset, 0] [4, 8] [1, 1]
            : tensor<4x8xf32> into tensor<8x8xf32>
      }
    }
    return %result : tensor<8x8xf32>
  }
}

// -----

// A failed fusion attempt must not leave its speculative producer and slice
// clones behind.
// CHECK-LABEL: func.func @failed_fusion_cleanup
// CHECK-COUNT-1: tensor.from_elements
// CHECK-COUNT-1: tensor.extract_slice

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %candidates, %new_loop =
        transform.fusion.greedy_input_producers_into_consumer %loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @failed_fusion_cleanup() -> tensor<8xf32> {
    %zero = arith.constant 0.0 : f32
    %source = tensor.from_elements %zero, %zero, %zero, %zero,
        %zero, %zero, %zero, %zero : tensor<8xf32>
    %empty = tensor.empty() : tensor<8xf32>
    %result = scf.forall (%iv) = (0) to (2) step (1)
        shared_outs(%out = %empty) -> tensor<8xf32> {
      %offset = affine.apply affine_map<(d0) -> (d0 * 4)>(%iv)
      %slice = tensor.extract_slice %source[%offset] [4] [1]
          : tensor<8xf32> to tensor<4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice into %out[%offset] [4] [1]
            : tensor<4xf32> into tensor<8xf32>
      }
    }
    return %result : tensor<8xf32>
  }
}
