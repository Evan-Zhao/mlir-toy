// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.apply_patterns to %funcs {
      transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
    } : !any
    transform.print %funcs : !any
    transform.yield
  }

  func.func @fold_for(%init: tensor<1x1x8xf32>) -> tensor<1x1x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %result = scf.for %i = %c0 to %c4 step %c1
        iter_args(%acc = %init) -> tensor<1x1x8xf32> {
      %collapsed = tensor.collapse_shape %acc [[0, 1, 2]]
          : tensor<1x1x8xf32> into tensor<8xf32>
      %expanded = tensor.expand_shape %collapsed [[0, 1, 2]]
          output_shape [1, 1, 8] : tensor<8xf32> into tensor<1x1x8xf32>
      scf.yield %expanded : tensor<1x1x8xf32>
    }
    return %result : tensor<1x1x8xf32>
  }

  func.func @fold_forall(%out: tensor<1x4x8xf32>, %tile: tensor<1x1x8xf32>)
      -> tensor<1x4x8xf32> {
    %result = scf.forall (%i) in (4) shared_outs(%arg = %out)
        -> tensor<1x4x8xf32> {
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %arg[0, %i, 0] [1, 1, 8] [1, 1, 1]
            : tensor<1x1x8xf32> into tensor<1x4x8xf32>
      }
    }
    return %result : tensor<1x4x8xf32>
  }

  func.func @fold_forall_rank_reduced_source(%out: tensor<1x4x8xf32>, %tile: tensor<1x8xf32>)
      -> tensor<1x4x8xf32> {
    %result = scf.forall (%i) in (4) shared_outs(%arg = %out)
        -> tensor<1x4x8xf32> {
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile into %arg[0, %i, 0] [1, 1, 8] [1, 1, 1]
            : tensor<1x8xf32> into tensor<1x4x8xf32>
      }
    }
    return %result : tensor<1x4x8xf32>
  }

  func.func @fold_empty_init() -> tensor<1x1x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %init = tensor.empty() : tensor<1x1x8xf32>
    %result = scf.for %i = %c0 to %c4 step %c1
        iter_args(%acc = %init) -> tensor<1x1x8xf32> {
      %filled = linalg.fill ins(%cst : f32) outs(%acc : tensor<1x1x8xf32>)
          -> tensor<1x1x8xf32>
      scf.yield %filled : tensor<1x1x8xf32>
    }
    return %result : tensor<1x1x8xf32>
  }
}

// CHECK-LABEL: func.func @fold_for(
// CHECK: return %{{.*}} : tensor<1x1x8xf32>

// CHECK-LABEL: func.func @fold_forall(
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1], [2]]
// CHECK: scf.forall (%{{.*}}) in (4) shared_outs(%{{.*}} = %{{.*}}) -> (tensor<4x8xf32>)
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2]]
// CHECK: tensor.parallel_insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [1, 8] [1, 1] : tensor<8xf32> into tensor<4x8xf32>
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1], [2]]
// CHECK: return %{{.*}} : tensor<1x4x8xf32>

// CHECK-LABEL: func.func @fold_forall_rank_reduced_source(
// CHECK: scf.forall (%{{.*}}) in (4) shared_outs(%{{.*}} = %{{.*}}) -> (tensor<4x8xf32>)
// CHECK-NOT: tensor.collapse_shape
// CHECK: tensor.parallel_insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [1, 8] [1, 1] : tensor<1x8xf32> into tensor<4x8xf32>
// CHECK: return %{{.*}} : tensor<1x4x8xf32>

// CHECK-LABEL: func.func @fold_empty_init(
// CHECK: tensor.empty() : tensor<8xf32>
// CHECK-NOT: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2]] : tensor<1x1x8xf32> into tensor<8xf32>
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<8xf32>)
// CHECK: linalg.fill
// CHECK: return %{{.*}} : tensor<1x1x8xf32>
