// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter | FileCheck %s

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "dead_tile"} in %module : (!any) -> !any
    %loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %producer = transform.structured.match ops{["linalg.generic"]} in %func : (!any) -> !any
    %live_loop, %mixed_loop =
      transform.loop.specialize_dead_tile %producer in %loop
        {dead_value = 0xFF800000 : f32} : !any, !any -> !any, !any
    transform.yield
  }

  // CHECK-LABEL: func.func @dead_tile(
  // CHECK: %[[LIVE_BOUND:.*]] = arith.select %{{.*}}, %{{.*}}, %c16 : index
  // CHECK: %[[LIVE_LOOP:.*]] = scf.for %{{.*}} = %c0 to %[[LIVE_BOUND]] step %c1 iter_args(%{{.*}} = %0) -> (tensor<1x128x64xf32>) {
  // CHECK-NEXT: scf.yield %arg1 : tensor<1x128x64xf32>
  // CHECK: %[[MIXED_LOOP:.*]] = scf.for %{{.*}} = %[[LIVE_BOUND]] to %c16 step %c1 iter_args(%{{.*}} = %[[LIVE_LOOP]]) -> (tensor<1x128x64xf32>) {
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
