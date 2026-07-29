// RUN: neptune-opt %s --transform-interpreter | FileCheck %s

!any = !transform.any_op

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "guarded_dead_tile"} in %module : (!any) -> !any
    %loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %live_loop, %mixed_loop =
      transform.loop.specialize_dead_tile in %loop
        {dead_value = 0xFF800000 : f32} : (!any) -> (!any, !any)
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
