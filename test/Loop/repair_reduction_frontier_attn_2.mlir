// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @repair_out_like
// CHECK: scf.forall
// CHECK-SAME: shared_outs({{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK: %{{.*}} = linalg.generic {{.*}}iterator_types = ["parallel", "parallel"]{{.*}}outs(%{{.*}} : tensor<2x2xf32>)
// CHECK: ^bb0(%[[IN:.+]]: f32, %[[OLDSUM:.+]]: f32, %[[NEWSUM:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK: %[[MUL0:.+]] = arith.mulf %[[IN]], %[[OLDSUM]] : f32
// CHECK: %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[INV:.+]] = arith.divf %[[ONE]], %[[NEWSUM]] : f32
// CHECK: %[[MUL1:.+]] = arith.mulf %[[MUL0]], %[[INV]] : f32
// CHECK: linalg.yield %[[MUL1]] : f32

!any = !transform.any_op
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} attributes {case_out} in %module
        : (!any) -> !any
    %forall_loop = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %inner_loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %producer_rowsum = transform.structured.match ops{["linalg.generic"]} attributes {producer_rowsum} in %func
        : (!any) -> !any
    %normalize = transform.structured.match ops{["linalg.generic"]} attributes {orig_normalize} in %func
        : (!any) -> !any
    %out_reduce = transform.structured.match ops{["linalg.generic"]} attributes {out_reduce} in %func
        : (!any) -> !any
    %sidecars =
      transform.fusion.clone_fuse_elemwise %normalize into %forall_loop, %inner_loop
        : (!any, !any, !any) -> !any
    %_ = transform.fusion.repair_reduction_frontier
        (%producer_rowsum, %out_reduce) and (%normalize, %sidecars) into %forall_loop, %inner_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.yield
  }

  func.func @repair_out_like(%exp_in: tensor<4x4xf32>, %v: tensor<4x2xf32>) -> tensor<4x2xf32> attributes {case_out} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.0 : f32

    %exp_empty = tensor.empty() : tensor<4x4xf32>
    %rowsum_empty = tensor.empty() : tensor<4xf32>
    %rowsum_init = linalg.fill ins(%zero : f32)
        outs(%rowsum_empty : tensor<4xf32>) -> tensor<4xf32>
    %exped, %rowsum = scf.forall (%i) in (2)
        shared_outs(%out0 = %exp_empty, %out1 = %rowsum_init)
        -> (tensor<4x4xf32>, tensor<4xf32>) {
      %row = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
      %panel = tensor.extract_slice %out0[%row, 0] [2, 4] [1, 1]
          : tensor<4x4xf32> to tensor<2x4xf32>
      %sum0 = tensor.extract_slice %out1[%row] [2] [1]
          : tensor<4xf32> to tensor<2xf32>
      %panel_upd, %sum_upd = scf.for %j = %c0 to %c2 step %c1
          iter_args(%acc0 = %panel, %acc1 = %sum0)
          -> (tensor<2x4xf32>, tensor<2xf32>) {
        %col = affine.apply affine_map<(d0) -> (d0 * 2)>(%j)
        %tile = tensor.extract_slice %exp_in[%row, %col] [2, 2] [1, 1]
            : tensor<4x4xf32> to tensor<2x2xf32>
        %new_panel = tensor.insert_slice %tile into %acc0[0, %col] [2, 2] [1, 1]
            : tensor<2x2xf32> into tensor<2x4xf32>
        %new_sum = linalg.generic {producer_rowsum,
            indexing_maps = [affine_map<(i, j) -> (i, j)>,
                             affine_map<(i, j) -> (i)>],
            iterator_types = ["parallel", "reduction"]}
            ins(%tile : tensor<2x2xf32>) outs(%acc1 : tensor<2xf32>) {
          ^bb0(%in: f32, %out: f32):
            %x = arith.addf %in, %out : f32
            linalg.yield %x : f32
        } -> tensor<2xf32>
        %sum_relay = tensor.insert_slice %new_sum into %acc1[0] [2] [1]
            : tensor<2xf32> into tensor<2xf32>
        scf.yield %new_panel, %sum_relay : tensor<2x4xf32>, tensor<2xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %panel_upd into %out0[%row, 0] [2, 4] [1, 1]
            : tensor<2x4xf32> into tensor<4x4xf32>
        tensor.parallel_insert_slice %sum_upd into %out1[%row] [2] [1]
            : tensor<2xf32> into tensor<4xf32>
      }
    }

    %norm_empty = tensor.empty() : tensor<4x4xf32>
    %norm = linalg.generic {orig_normalize,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%exped, %rowsum : tensor<4x4xf32>, tensor<4xf32>)
        outs(%norm_empty : tensor<4x4xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %out: f32):
        %x = arith.divf %lhs, %rhs : f32
        linalg.yield %x : f32
    } -> tensor<4x4xf32>

    %out_empty = tensor.empty() : tensor<4x2xf32>
    %out_init = linalg.fill ins(%zero : f32) outs(%out_empty : tensor<4x2xf32>) -> tensor<4x2xf32>
    %out = linalg.generic {out_reduce,
        indexing_maps = [affine_map<(i, d, j) -> (i, j)>,
                         affine_map<(i, d, j) -> (j, d)>,
                         affine_map<(i, d, j) -> (i, d)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%norm, %v : tensor<4x4xf32>, tensor<4x2xf32>)
        outs(%out_init : tensor<4x2xf32>) {
      ^bb0(%p: f32, %val: f32, %acc: f32):
        %prod = arith.mulf %p, %val : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<4x2xf32>
    return %out : tensor<4x2xf32>
  }
}
