// RUN: neptune-opt %s --split-input-file --transform-interpreter | FileCheck %s
// RUN: neptune-opt %s --split-input-file --transform-interpreter 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "no_dead_tile"} in %module : (!any) -> !any
    %loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %live_loop, %mixed_loop, %unchanged_loop =
      transform.loop.specialize_dead_tile in %loop
        {dead_value = 0xFF800000 : f32} : (!any) -> (!any, !any, !any)
    transform.annotate %unchanged_loop "unchanged" : !any
    transform.yield
  }

  // CHECK-LABEL: func.func @no_dead_tile(
  // CHECK: scf.for
  // CHECK: } {unchanged}
  // CHECK-NOT: linalg.generic
  // CHECK: return
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

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "dead_tile"} in %module : (!any) -> !any
    %loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %producer = transform.structured.match ops{["linalg.generic"]} in %func : (!any) -> !any
    %live_loop, %mixed_loop, %unchanged_loop =
      transform.loop.specialize_dead_tile %producer in %loop
        {dead_value = 0xFF800000 : f32} : (!any, !any) -> (!any, !any, !any)
    transform.yield
  }

  // DIAG: remark: dead-tile propagation could not prove loop-carried result #0 is unchanged;
  // DIAG-SAME: the fully-dead suffix will not be truncated
  // CHECK-LABEL: func.func @dead_tile(
  // CHECK: arith.maxsi
  // CHECK: arith.minsi
  // CHECK: %[[LIVE_LOOP:.*]] = scf.for %{{.*}} = %[[LIVE_LOWER:.*]] to %[[LIVE_UPPER:.*]] step %c1 iter_args(%{{.*}} = %0) -> (tensor<1x128x64xf32>) {
  // CHECK-NEXT: scf.yield %arg1 : tensor<1x128x64xf32>
  // CHECK: %[[MIXED_LOOP:.*]] = scf.for %{{.*}} = %[[LIVE_UPPER]] to %c16 step %c1 iter_args(%{{.*}} = %[[LIVE_LOOP]]) -> (tensor<1x128x64xf32>) {
  // CHECK: linalg.generic
  func.func @dead_tile(%q_block: index, %live: tensor<1x128x64xf32>)
      -> tensor<1x128x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %dead = arith.constant dense<0xFF800000> : tensor<f32>
    %empty = tensor.empty() : tensor<1x128x64xf32>
    %result = scf.for %j = %c0 to %c16 step %c1
        iter_args(%acc = %empty) -> tensor<1x128x64xf32> {
      %tile = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2) -> ()>,
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%live, %dead : tensor<1x128x64xf32>, tensor<f32>)
          outs(%empty : tensor<1x128x64xf32>) {
      ^bb0(%in: f32, %dead_scalar: f32, %out: f32):
        %row = linalg.index 1 : index
        %q_abs = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0)>(%q_block)[%row]
        %col = linalg.index 2 : index
        %k_abs = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%j)[%col]
        %q_i64 = arith.index_cast %q_abs : index to i64
        %k_i64 = arith.index_cast %k_abs : index to i64
        %is_live = arith.cmpi sle, %k_i64, %q_i64 : i64
        %selected = arith.select %is_live, %in, %dead_scalar : f32
        linalg.yield %selected : f32
      } -> tensor<1x128x64xf32>
      scf.yield %tile : tensor<1x128x64xf32>
    }
    return %result : tensor<1x128x64xf32>
  }
}

// -----

!any = !transform.any_op

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map0 = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "propagate_dead_tile"} in %module : (!any) -> !any
    %loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %producer = transform.structured.match ops{["linalg.generic"]}
        attributes {tag = "producer"} in %func : (!any) -> !any
    %live_loop, %mixed_loop, %unchanged_loop =
      transform.loop.specialize_dead_tile %producer in %loop
        {dead_value = 0xFF800000 : f32} : (!any, !any) -> (!any, !any, !any)
    transform.yield
  }

  // CHECK-LABEL: func.func @propagate_dead_tile(
  // CHECK: arith.maxsi
  // CHECK: arith.minsi
  // CHECK: %[[LIVE_LOOP:.*]]:3 = scf.for %{{.*}} = %[[LIVE_LOWER:.*]] to %[[LIVE_UPPER:.*]] step %c1 iter_args(%{{.*}} = %arg3, %{{.*}} = %arg4, %{{.*}} = %arg5) -> (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128x32xf32>) {
  // CHECK-NOT: tag = "producer"
  // CHECK: ins(%arg1 : tensor<1x128x64xf32>) outs(%{{.*}} : tensor<1x128xf32>)
  // CHECK: ins(%arg1 : tensor<1x128x64xf32>) outs(%0 : tensor<1x128x64xf32>)
  // CHECK: %[[DEAD_RAW:.*]] = affine.apply
  // CHECK: %[[DEAD_BOUND_MAX:.*]] = arith.maxsi %[[DEAD_RAW]], %c0 : index
  // CHECK: %[[DEAD_BOUND:.*]] = arith.minsi %[[DEAD_BOUND_MAX]], %c16 : index
  // CHECK: %[[MIXED_LOOP:.*]]:3 = scf.for %{{.*}} = %[[LIVE_UPPER]] to %[[DEAD_BOUND]] step %c1 iter_args(%{{.*}} = %[[LIVE_LOOP]]#0, %{{.*}} = %[[LIVE_LOOP]]#1, %{{.*}} = %[[LIVE_LOOP]]#2) -> (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128x32xf32>) {
  // CHECK: tag = "producer"
  func.func @propagate_dead_tile(
      %q_block: index,
      %live: tensor<1x128x64xf32>,
      %rhs: tensor<1x64x32xf32>,
      %rowmax_init: tensor<1x128xf32>,
      %rowsum_init: tensor<1x128xf32>,
      %acc_init: tensor<1x128x32xf32>)
      -> (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %dead = arith.constant dense<0xFF800000> : tensor<f32>
    %tmp = tensor.empty() : tensor<1x128x64xf32>
    %tmp_acc = tensor.empty() : tensor<1x128x32xf32>
    %result:3 = scf.for %j = %c0 to %c16 step %c1
        iter_args(%rowmax = %rowmax_init, %rowsum = %rowsum_init, %acc = %acc_init)
        -> (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128x32xf32>) {
      %producer = linalg.generic {tag = "producer",
          indexing_maps = [#map, #map0, #map], iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%live, %dead : tensor<1x128x64xf32>, tensor<f32>)
          outs(%tmp : tensor<1x128x64xf32>) {
      ^bb0(%in: f32, %dead_scalar: f32, %out: f32):
        %row = linalg.index 1 : index
        %q_abs = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0)>(%q_block)[%row]
        %col = linalg.index 2 : index
        %k_abs = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%j)[%col]
        %q_i64 = arith.index_cast %q_abs : index to i64
        %k_i64 = arith.index_cast %k_abs : index to i64
        %is_live = arith.cmpi sle, %k_i64, %q_i64 : i64
        %selected = arith.select %is_live, %in, %dead_scalar : f32
        linalg.yield %selected : f32
      } -> tensor<1x128x64xf32>
      %rowmax_next = linalg.generic {
          indexing_maps = [#map, #map1],
          iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%producer : tensor<1x128x64xf32>)
          outs(%rowmax : tensor<1x128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %max = arith.maximumf %in, %out : f32
        linalg.yield %max : f32
      } -> tensor<1x128xf32>
      %zeros = linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%producer : tensor<1x128x64xf32>)
          outs(%tmp : tensor<1x128x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %exp = math.exp2 %in : f32
        linalg.yield %exp : f32
      } -> tensor<1x128x64xf32>
      %rowsum_next = linalg.generic {
          indexing_maps = [#map, #map1],
          iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%zeros : tensor<1x128x64xf32>)
          outs(%rowsum : tensor<1x128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      } -> tensor<1x128xf32>
      %acc_next = linalg.generic {
          indexing_maps = [#map2, #map3, #map4],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%zeros, %rhs : tensor<1x128x64xf32>, tensor<1x64x32xf32>)
          outs(%acc : tensor<1x128x32xf32>) {
      ^bb0(%lhs: f32, %rhs_elem: f32, %out: f32):
        %mul = arith.mulf %lhs, %rhs_elem : f32
        %sum = arith.addf %out, %mul : f32
        linalg.yield %sum : f32
      } -> tensor<1x128x32xf32>
      scf.yield %rowmax_next, %rowsum_next, %acc_next
          : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128x32xf32>
    }
    return %result#0, %result#1, %result#2
        : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128x32xf32>
  }
}

// -----

!any = !transform.any_op

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "guarded_dead_tile"} in %module : (!any) -> !any
    %loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %live_loop, %mixed_loop, %unchanged_loop =
      transform.loop.specialize_dead_tile in %loop
        {dead_value = 0xFF800000 : f32} : (!any) -> (!any, !any, !any)
    transform.yield
  }

  // CHECK-LABEL: func.func @guarded_dead_tile(
  // CHECK: %[[LIVE_RAW:.*]] = affine.apply
  // CHECK: %[[LIVE_MAX:.*]] = arith.maxsi %[[LIVE_RAW]], %c0
  // CHECK: %[[LIVE_CAPPED:.*]] = arith.minsi %[[LIVE_MAX]], %c8
  // CHECK: %[[Q_MAX:.*]] = affine.apply
  // CHECK: %[[LIVE_LENGTH:.*]] = affine.apply
  // CHECK: %[[ALL_QUERIES_LIVE:.*]] = arith.cmpi slt, %[[Q_MAX]], %[[LIVE_LENGTH]]
  // CHECK: %[[LIVE_UB:.*]] = arith.select %[[ALL_QUERIES_LIVE]], %[[LIVE_CAPPED]], %{{.*}}
  // CHECK: %[[LIVE_LOOP:.*]] = scf.for %{{.*}} = %{{.*}} to %[[LIVE_UB]] step %c1
  // CHECK-NOT: tag = "producer"
  // CHECK: %[[POSSIBLE_RAW:.*]] = affine.apply
  // CHECK: %[[POSSIBLE_MAX:.*]] = arith.maxsi %[[POSSIBLE_RAW]], %c0
  // CHECK: %[[POSSIBLE_CAPPED:.*]] = arith.minsi %[[POSSIBLE_MAX]], %c8
  // CHECK: %[[Q_MIN:.*]] = affine.apply
  // CHECK: %[[POSSIBLE_LENGTH:.*]] = affine.apply
  // CHECK: %[[ANY_QUERY_LIVE:.*]] = arith.cmpi slt, %[[Q_MIN]], %[[POSSIBLE_LENGTH]]
  // CHECK: %[[POSSIBLE_UB:.*]] = arith.select %[[ANY_QUERY_LIVE]], %[[POSSIBLE_CAPPED]], %[[LIVE_UB]]
  // CHECK: scf.for %{{.*}} = %[[LIVE_UB]] to %[[POSSIBLE_UB]] step %c1
  // CHECK: tag = "producer"
  func.func @guarded_dead_tile(%query_block: index, %length: index,
      %scores: tensor<128x64xf32>, %rowmax_init: tensor<128xf32>)
      -> tensor<128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %dead = arith.constant dense<0xFF800000> : tensor<f32>
    %tile_init = tensor.empty() : tensor<128x64xf32>
    %result = scf.for %j = %c0 to %c8 step %c1
        iter_args(%rowmax = %rowmax_init) -> tensor<128xf32> {
      %tile = linalg.generic {tag = "producer",
          indexing_maps = [#map, affine_map<(d0, d1) -> ()>, #map],
          iterator_types = ["parallel", "parallel"]}
          ins(%scores, %dead : tensor<128x64xf32>, tensor<f32>)
          outs(%tile_init : tensor<128x64xf32>) {
      ^bb0(%score: f32, %dead_scalar: f32, %out: f32):
        %row_index = linalg.index 0 : index
        %query = affine.apply
            affine_map<(d0)[s0] -> (d0 * 128 + s0)>(%query_block)[%row_index]
        %query_live = arith.cmpi slt, %query, %length : index
        %column_index = linalg.index 1 : index
        %key = affine.apply
            affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%j)[%column_index]
        %key_live = arith.cmpi slt, %key, %length : index
        %live = arith.andi %query_live, %key_live : i1
        %selected = arith.select %live, %score, %dead_scalar : f32
        linalg.yield %selected : f32
      } -> tensor<128x64xf32>
      %next = linalg.generic {
          indexing_maps = [#map, #row],
          iterator_types = ["parallel", "reduction"]}
          ins(%tile : tensor<128x64xf32>) outs(%rowmax : tensor<128xf32>) {
      ^bb0(%value: f32, %acc: f32):
        %max = arith.maximumf %value, %acc : f32
        linalg.yield %max : f32
      } -> tensor<128xf32>
      scf.yield %next : tensor<128xf32>
    }
    return %result : tensor<128xf32>
  }
}

// -----

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "windowed_dead_tile_prefix"} in %module : (!any) -> !any
    %loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %producer = transform.structured.match ops{["linalg.generic"]} in %func : (!any) -> !any
    %live_loop, %mixed_loop, %unchanged_loop =
      transform.loop.specialize_dead_tile %producer in %loop
        {dead_value = 0xFF800000 : f32} : (!any, !any) -> (!any, !any, !any)
    transform.yield
  }

  // CHECK-DAG: #[[$LOWER_MAP:.*]] = affine_map<() -> (0)>
  // CHECK-DAG: #[[$UPPER_MAP:.*]] = affine_map<() -> (1)>
  // CHECK-LABEL: func.func @windowed_dead_tile_prefix(
  // CHECK: %[[LOWER_RAW:.*]] = affine.apply #[[$LOWER_MAP]]()
  // CHECK: %[[UPPER_RAW:.*]] = affine.apply #[[$UPPER_MAP]]()
  // CHECK: %[[LOWER_NONNEGATIVE:.*]] = arith.maxsi %[[LOWER_RAW]], %c0 : index
  // CHECK: %[[LIVE_LOWER:.*]] = arith.minsi %[[LOWER_NONNEGATIVE]], %c16 : index
  // CHECK: %[[UPPER_NONNEGATIVE:.*]] = arith.maxsi %[[UPPER_RAW]], %c0 : index
  // CHECK: %[[LIVE_UPPER:.*]] = arith.minsi %[[UPPER_NONNEGATIVE]], %c16 : index
  // CHECK: scf.for %{{.*}} = %[[LIVE_LOWER]] to %[[LIVE_UPPER]] step %c1
  // CHECK-NEXT: scf.yield %arg0 : tensor<64x64xf32>
  // CHECK: scf.for %{{.*}} = %[[LIVE_UPPER]] to %c16 step %c1
  // CHECK: linalg.generic
  func.func @windowed_dead_tile_prefix(%live: tensor<64x64xf32>)
      -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c-127_i64 = arith.constant -127 : i64
    %dead = arith.constant dense<0xFF800000> : tensor<f32>
    %empty = tensor.empty() : tensor<64x64xf32>
    %result = scf.for %j = %c0 to %c16 step %c1
        iter_args(%acc = %empty) -> tensor<64x64xf32> {
      %tile = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> ()>,
            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%live, %dead : tensor<64x64xf32>, tensor<f32>)
          outs(%empty : tensor<64x64xf32>) {
      ^bb0(%in: f32, %dead_scalar: f32, %out: f32):
        %row = linalg.index 0 : index
        %q_abs = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%c1)[%row]
        %col = linalg.index 1 : index
        %k_abs = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%j)[%col]
        %q_i64 = arith.index_cast %q_abs : index to i64
        %k_i64 = arith.index_cast %k_abs : index to i64
        %not_future = arith.cmpi sle, %k_i64, %q_i64 : i64
        %window_offset = arith.subi %k_i64, %q_i64 : i64
        %within_window = arith.cmpi sge, %window_offset, %c-127_i64 : i64
        %is_live = arith.andi %within_window, %not_future : i1
        %selected = arith.select %is_live, %in, %dead_scalar : f32
        linalg.yield %selected : f32
      } -> tensor<64x64xf32>
      scf.yield %tile : tensor<64x64xf32>
    }
    return %result : tensor<64x64xf32>
  }
}
