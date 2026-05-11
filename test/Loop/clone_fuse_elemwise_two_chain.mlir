// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s

// Two elemwise consumers where B consumes A. This exercises sidecar chaining.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %forall_loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %inner_loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %a = transform.structured.match ops{["linalg.generic"]} attributes {fuse0} in %func
        : (!transform.any_op) -> !transform.any_op
    %b = transform.structured.match ops{["linalg.generic"]} attributes {fuse1} in %func
        : (!transform.any_op) -> !transform.any_op
    %elemwise = transform.merge_handles %a, %b : !transform.any_op
    %sidecar =
      transform.loop_ru.clone_fuse_elemwise %elemwise into %forall_loop, %inner_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @toy(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 5.000000e-01 : f32

    %scores_init = tensor.empty() : tensor<8x8xf32>
    %rm_init_e = tensor.empty() : tensor<8xf32>
    %neg_inf = arith.constant -3.40282347E+38 : f32
    %rm_fill = linalg.fill ins(%neg_inf : f32)
        outs(%rm_init_e : tensor<8xf32>) -> tensor<8xf32>

    %scores, %rmax = scf.forall (%arg1) in (2)
        shared_outs(%arg2 = %scores_init, %arg3 = %rm_fill)
        -> (tensor<8x8xf32>, tensor<8xf32>) {
      %off = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1)
      %scores_panel = tensor.extract_slice %arg2[%off, 0] [4, 8] [1, 1]
          : tensor<8x8xf32> to tensor<4x8xf32>
      %rm_panel = tensor.extract_slice %arg3[%off] [4] [1]
          : tensor<8xf32> to tensor<4xf32>

      %panel_upd:2 = scf.for %iv = %c0 to %c2 step %c1
          iter_args(%acc = %scores_panel, %rm_acc = %rm_panel)
          -> (tensor<4x8xf32>, tensor<4xf32>) {
        %j = affine.apply affine_map<(d0) -> (d0 * 4)>(%iv)
        %tile = tensor.extract_slice %arg0[0, %j] [4, 4] [1, 1]
            : tensor<8x8xf32> to tensor<4x4xf32>
        %init = tensor.extract_slice %acc[0, %j] [4, 4] [1, 1]
            : tensor<4x8xf32> to tensor<4x4xf32>
        %score_tile = linalg.generic {
            indexing_maps = [affine_map<(i, j) -> (i, j)>,
                             affine_map<(i, j) -> (i, j)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%tile : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) {
          ^bb0(%in: f32, %out: f32):
            %scaled = arith.addf %in, %cst : f32
            linalg.yield %scaled : f32
        } -> tensor<4x4xf32>
        %updated = tensor.insert_slice %score_tile into %acc[0, %j] [4, 4] [1, 1]
            : tensor<4x4xf32> into tensor<4x8xf32>
        scf.yield %updated, %rm_acc : tensor<4x8xf32>, tensor<4xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %panel_upd#0 into %arg2[%off, 0] [4, 8] [1, 1]
            : tensor<4x8xf32> into tensor<8x8xf32>
        tensor.parallel_insert_slice %panel_upd#1 into %arg3[%off] [4] [1]
            : tensor<4xf32> into tensor<8xf32>
      }
    }

    %sub_init = tensor.empty() : tensor<8x8xf32>
    %sub = linalg.generic {fuse0,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%scores, %rmax : tensor<8x8xf32>, tensor<8xf32>)
        outs(%sub_init : tensor<8x8xf32>) {
      ^bb0(%a: f32, %b: f32, %out: f32):
        %x = arith.subf %a, %b : f32
        linalg.yield %x : f32
    } -> tensor<8x8xf32>

    %mul_init = tensor.empty() : tensor<8x8xf32>
    %mul = linalg.generic {fuse1,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%sub : tensor<8x8xf32>) outs(%mul_init : tensor<8x8xf32>) {
      ^bb0(%a: f32, %out: f32):
        %x = arith.mulf %a, %cst : f32
        linalg.yield %x : f32
    } -> tensor<8x8xf32>
    return %mul : tensor<8x8xf32>
  }
}

// CHECK: %{{.*}}:4 = scf.forall
// CHECK-SAME: shared_outs({{[^)]*}}, {{[^)]*}}, {{[^)]*}}, {{[^)]*}}) -> (tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}) -> (tensor<4x8xf32>, tensor<4xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: scf.yield
