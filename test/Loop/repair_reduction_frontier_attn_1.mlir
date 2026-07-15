// RUN: neptune-opt %s --transform-interpreter 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @repair_rowsum_like
// CHECK: scf.forall
// CHECK-SAME: shared_outs({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK: %{{.*}} = linalg.generic {{.*}}iterator_types = ["parallel"]{{.*}}outs(%{{.*}} : tensor<2xf32>)
// CHECK: ^bb0(%[[OLDMAX:.+]]: f32, %[[NEWMAX:.+]]: f32, %[[ACC:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK: %[[DELTA:.+]] = arith.subf %[[OLDMAX]], %[[NEWMAX]] : f32
// CHECK: %[[EXP:.+]] = math.exp %[[DELTA]] : f32
// CHECK: %[[SCALED:.+]] = arith.mulf %[[ACC]], %[[EXP]] : f32
// CHECK: linalg.yield %[[SCALED]] : f32

!any = !transform.any_op
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} attributes {case_rowsum} in %module
        : (!any) -> !any
    %forall_loop = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %inner_loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %producer_rowmax = transform.structured.match ops{["linalg.generic"]} attributes {producer_rowmax} in %func
        : (!any) -> !any
    %shift = transform.structured.match ops{["linalg.generic"]} attributes {orig_shift} in %func
        : (!any) -> !any
    %exp = transform.structured.match ops{["linalg.generic"]} attributes {orig_exp} in %func
        : (!any) -> !any
    %rowsum_reduce = transform.structured.match ops{["linalg.generic"]} attributes {rowsum_reduce} in %func
        : (!any) -> !any
    %elemwise = transform.merge_handles %shift, %exp : !any
    %sidecars =
      transform.fusion.clone_fuse_elemwise %elemwise into %forall_loop, %inner_loop
        : (!any, !any, !any) -> !any
    %_ = transform.fusion.repair_reduction_frontier
        %rowsum_reduce reduce_producer %producer_rowmax
        substituting elemwise %elemwise -> %sidecars
        into %forall_loop, %inner_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.yield
  }

  func.func @repair_rowsum_like(%scores_in: tensor<4x4xf32>) -> tensor<4xf32> attributes {case_rowsum} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.0 : f32
    %neg_inf = arith.constant -3.40282347E+38 : f32
    %scale = arith.constant 5.000000e-01 : f32

    %scaled_empty = tensor.empty() : tensor<4x4xf32>
    %rowmax_empty = tensor.empty() : tensor<4xf32>
    %rowmax_init = linalg.fill ins(%neg_inf : f32)
        outs(%rowmax_empty : tensor<4xf32>) -> tensor<4xf32>
    %scaled, %rowmax = scf.forall (%i) in (2)
        shared_outs(%out0 = %scaled_empty, %out1 = %rowmax_init)
        -> (tensor<4x4xf32>, tensor<4xf32>) {
      %row = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
      %panel = tensor.extract_slice %out0[%row, 0] [2, 4] [1, 1]
          : tensor<4x4xf32> to tensor<2x4xf32>
      %max0 = tensor.extract_slice %out1[%row] [2] [1]
          : tensor<4xf32> to tensor<2xf32>
      %panel_upd, %max_upd = scf.for %j = %c0 to %c2 step %c1
          iter_args(%acc0 = %panel, %acc1 = %max0)
          -> (tensor<2x4xf32>, tensor<2xf32>) {
        %col = affine.apply affine_map<(d0) -> (d0 * 2)>(%j)
        %tile = tensor.extract_slice %scores_in[%row, %col] [2, 2] [1, 1]
            : tensor<4x4xf32> to tensor<2x2xf32>
        %tile_init = tensor.extract_slice %acc0[0, %col] [2, 2] [1, 1]
            : tensor<2x4xf32> to tensor<2x2xf32>
        %scaled_tile = linalg.generic {
            indexing_maps = [affine_map<(i, j) -> (i, j)>,
                             affine_map<(i, j) -> (i, j)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%tile : tensor<2x2xf32>) outs(%tile_init : tensor<2x2xf32>) {
          ^bb0(%in: f32, %out: f32):
            %x = arith.mulf %in, %scale : f32
            linalg.yield %x : f32
        } -> tensor<2x2xf32>
        %new_panel = tensor.insert_slice %scaled_tile into %acc0[0, %col] [2, 2] [1, 1]
            : tensor<2x2xf32> into tensor<2x4xf32>
        %new_max = linalg.generic {producer_rowmax,
            indexing_maps = [affine_map<(i, j) -> (i, j)>,
                             affine_map<(i, j) -> (i)>],
            iterator_types = ["parallel", "reduction"]}
            ins(%scaled_tile : tensor<2x2xf32>) outs(%acc1 : tensor<2xf32>) {
          ^bb0(%in: f32, %out: f32):
            %x = arith.maximumf %in, %out : f32
            linalg.yield %x : f32
        } -> tensor<2xf32>
        %max_relay = tensor.insert_slice %new_max into %acc1[0] [2] [1]
            : tensor<2xf32> into tensor<2xf32>
        scf.yield %new_panel, %max_relay : tensor<2x4xf32>, tensor<2xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %panel_upd into %out0[%row, 0] [2, 4] [1, 1]
            : tensor<2x4xf32> into tensor<4x4xf32>
        tensor.parallel_insert_slice %max_upd into %out1[%row] [2] [1]
            : tensor<2xf32> into tensor<4xf32>
      }
    }

    %shift_empty = tensor.empty() : tensor<4x4xf32>
    %shift = linalg.generic {orig_shift,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%scaled, %rowmax : tensor<4x4xf32>, tensor<4xf32>)
        outs(%shift_empty : tensor<4x4xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %out: f32):
        %x = arith.subf %lhs, %rhs : f32
        linalg.yield %x : f32
    } -> tensor<4x4xf32>

    %exp_empty = tensor.empty() : tensor<4x4xf32>
    %exped = linalg.generic {orig_exp,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%shift : tensor<4x4xf32>) outs(%exp_empty : tensor<4x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %x = math.exp %in : f32
        linalg.yield %x : f32
    } -> tensor<4x4xf32>

    %rowsum_empty = tensor.empty() : tensor<4xf32>
    %rowsum_init = linalg.fill ins(%zero : f32)
        outs(%rowsum_empty : tensor<4xf32>) -> tensor<4xf32>
    %rowsum = linalg.generic {rowsum_reduce,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%exped : tensor<4x4xf32>) outs(%rowsum_init : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %x = arith.addf %in, %out : f32
        linalg.yield %x : f32
    } -> tensor<4xf32>
    return %rowsum : tensor<4xf32>
  }
}
