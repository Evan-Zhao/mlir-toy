// RUN: neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s

!any = !transform.any_op
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %generic = transform.structured.match ops{["linalg.generic"]} in %func : (!any) -> !any
    transform.apply_patterns to %func {
      transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
    } : !any
    transform.print %loop : !any
    transform.print %generic : !any
    transform.yield
  }

  func.func @preserve_handles(%init: tensor<1x1x8xf32>) -> tensor<1x1x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %empty = tensor.empty() : tensor<1x1x8xf32>
    %result = scf.for %i = %c0 to %c4 step %c1
        iter_args(%acc = %init) -> tensor<1x1x8xf32> {
      %next = linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%acc : tensor<1x1x8xf32>)
          outs(%empty : tensor<1x1x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x1x8xf32>
      scf.yield %next : tensor<1x1x8xf32>
    }
    return %result : tensor<1x1x8xf32>
  }
}

// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<8xf32>)
// CHECK: linalg.generic
