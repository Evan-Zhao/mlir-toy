// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter | FileCheck %s

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "windowed_dead_tile_prefix"} in %module : (!any) -> !any
    %loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %producer = transform.structured.match ops{["linalg.generic"]} in %func : (!any) -> !any
    %live_loop, %mixed_loop =
      transform.loop.specialize_dead_tile %producer in %loop
        {dead_value = 0xFF800000 : f32} : !any, !any -> !any, !any
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
