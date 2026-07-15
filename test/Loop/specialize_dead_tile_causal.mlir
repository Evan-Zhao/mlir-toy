// RUN: neptune-opt %s --transform-interpreter | FileCheck %s

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
    %live_loop, %mixed_loop =
      transform.loop.specialize_dead_tile %producer in %loop
        {dead_value = 0xFF800000 : f32} : (!any, !any) -> (!any, !any)
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
