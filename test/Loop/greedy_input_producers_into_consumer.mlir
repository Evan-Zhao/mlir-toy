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
    %candidates = transform.fusion.greedy_input_producers_into_consumer %loop
        : (!transform.any_op) -> !transform.any_op
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
// CHECK-NOT: {fusion_root}
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
    %loop_nest = transform.merge_handles %forall, %for : !transform.any_op
    %candidates = transform.fusion.greedy_input_producers_into_consumer %loop_nest
        : (!transform.any_op) -> !transform.any_op
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
    %candidates = transform.fusion.greedy_input_producers_into_consumer %loop
        : (!transform.any_op) -> !transform.any_op
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

// -----

// A producer that is also used after the forall is reconstructed as another
// shared_out. Its escaping use reads the new forall result, so the untiled
// producer becomes dead.
// CHECK-LABEL: func.func @reconstruct_escaping_producer
// CHECK-NOT: {escaping_producer}
// CHECK: %[[LOOP:.*]]:2 = scf.forall
// CHECK-SAME: shared_outs(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}})
// CHECK: %[[TILED:.*]] = linalg.generic
// CHECK-SAME: {escaping_producer}
// CHECK: tensor.parallel_insert_slice %[[TILED]] into %{{.*}}
// CHECK: tensor.parallel_insert_slice %{{.*}} into %{{.*}}
// CHECK: %[[AFTER:.*]] = linalg.generic
// CHECK-SAME: ins(%[[LOOP]]#0, %[[LOOP]]#1
// CHECK-SAME: {after_loop}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %candidates = transform.fusion.greedy_input_producers_into_consumer %loop
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @reconstruct_escaping_producer(%arg: tensor<8xf32>) -> tensor<8xf32> {
    %producer_empty = tensor.empty() : tensor<8xf32>
    %producer = linalg.generic {
        escaping_producer,
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%arg : tensor<8xf32>) outs(%producer_empty : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %one = arith.constant 1.0 : f32
        %sum = arith.addf %in, %one : f32
        linalg.yield %sum : f32
    } -> tensor<8xf32>

    %loop_empty = tensor.empty() : tensor<8xf32>
    %loop_result = scf.forall (%iv) = (0) to (2) step (1)
        shared_outs(%out = %loop_empty) -> tensor<8xf32> {
      %offset = affine.apply affine_map<(d0) -> (d0 * 4)>(%iv)
      %slice = tensor.extract_slice %producer[%offset] [4] [1]
          : tensor<8xf32> to tensor<4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice into %out[%offset] [4] [1]
            : tensor<4xf32> into tensor<8xf32>
      }
    }

    %after_empty = tensor.empty() : tensor<8xf32>
    %after = linalg.generic {
        after_loop,
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%loop_result, %producer : tensor<8xf32>, tensor<8xf32>)
        outs(%after_empty : tensor<8xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %out: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %after : tensor<8xf32>
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    transform.fusion.greedy_input_producers_into_consumer %loop
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @extract_slice(%arg: tensor<12x4xf32>) -> tensor<8x4xf32> {
    %empty = tensor.empty() : tensor<12x4xf32>
    %producer = linalg.map { arith.negf } ins(%arg : tensor<12x4xf32>)
        outs(%empty : tensor<12x4xf32>)
    %outer = tensor.extract_slice %producer[4, 0] [8, 4] [1, 1]
        : tensor<12x4xf32> to tensor<8x4xf32>
    %init = tensor.empty() : tensor<8x4xf32>
    %result = scf.forall (%row) in (2) shared_outs(%out = %init) -> tensor<8x4xf32> {
      %offset = affine.apply affine_map<(d0) -> (d0 * 4)>(%row)
      %tile = tensor.extract_slice %outer[%offset, 0] [4, 4] [1, 1]
          : tensor<8x4xf32> to tensor<4x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%offset, 0] [4, 4] [1, 1]
            : tensor<4x4xf32> into tensor<8x4xf32>
      }
    }
    return %result : tensor<8x4xf32>
  }
}

// CHECK-LABEL: func.func @extract_slice
// CHECK: scf.forall
// CHECK: tensor.extract_slice %arg0
// CHECK: linalg.map
// CHECK-NOT: tensor.extract_slice %{{.*}} : tensor<12x4xf32> to tensor<8x4xf32>

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %candidates = transform.fusion.greedy_input_producers_into_consumer %loop
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @collapse_shape(%arg: tensor<2x4xf32>) -> tensor<8xf32> {
    %empty = tensor.empty() : tensor<2x4xf32>
    %producer = linalg.map { arith.negf } ins(%arg : tensor<2x4xf32>)
        outs(%empty : tensor<2x4xf32>)
    %collapsed = tensor.collapse_shape %producer [[0, 1]]
        : tensor<2x4xf32> into tensor<8xf32>
    %init = tensor.empty() : tensor<8xf32>
    %result = scf.forall (%row) in (2) shared_outs(%out = %init) -> tensor<8xf32> {
      %offset = affine.apply affine_map<(d0) -> (d0 * 4)>(%row)
      %tile = tensor.extract_slice %collapsed[%offset] [4] [1]
          : tensor<8xf32> to tensor<4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%offset] [4] [1]
            : tensor<4xf32> into tensor<8xf32>
      }
    }
    return %result : tensor<8xf32>
  }
}

// CHECK-LABEL: func.func @collapse_shape
// CHECK: scf.forall
// CHECK: %[[TILED:.*]] = linalg.map
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[TILED]]
// CHECK: tensor.parallel_insert_slice %[[COLLAPSED]]

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    transform.fusion.greedy_input_producers_into_consumer %loop
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @expand_shape(%arg: tensor<8xf32>) -> tensor<2x4xf32> {
    %empty = tensor.empty() : tensor<8xf32>
    %producer = linalg.map { arith.negf } ins(%arg : tensor<8xf32>)
        outs(%empty : tensor<8xf32>)
    %expanded = tensor.expand_shape %producer [[0, 1]] output_shape [2, 4]
        : tensor<8xf32> into tensor<2x4xf32>
    %init = tensor.empty() : tensor<2x4xf32>
    %result = scf.forall (%row) in (2) shared_outs(%out = %init) -> tensor<2x4xf32> {
      %tile = tensor.extract_slice %expanded[%row, 0] [1, 4] [1, 1]
          : tensor<2x4xf32> to tensor<1x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %out[%row, 0] [1, 4] [1, 1]
            : tensor<1x4xf32> into tensor<2x4xf32>
      }
    }
    return %result : tensor<2x4xf32>
  }
}

// CHECK-LABEL: func.func @expand_shape
// CHECK: scf.forall
// CHECK: %[[TILED:.*]] = linalg.map
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[TILED]]
// CHECK: tensor.parallel_insert_slice %[[EXPANDED]]
