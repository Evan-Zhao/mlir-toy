// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s

// Regression: when a cloned sidecar consumer reads a seeded loop result that is
// updated earlier in the same inner loop, it should use the freshly computed
// value directly instead of extracting from the incoming iter_arg.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %forall_loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %inner_loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %elemwise = transform.structured.match ops{["linalg.generic"]} attributes {fuse}
        in %func : (!transform.any_op) -> !transform.any_op
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
    %neg_inf = arith.constant -3.40282347E+38 : f32

    %scores_init = tensor.empty() : tensor<8x8xf32>
    %rmax_init_e = tensor.empty() : tensor<8xf32>
    %rmax_init = linalg.fill ins(%neg_inf : f32)
        outs(%rmax_init_e : tensor<8xf32>) -> tensor<8xf32>

    %scores, %rmax = scf.forall (%arg1) in (2)
        shared_outs(%arg2 = %scores_init, %arg3 = %rmax_init)
        -> (tensor<8x8xf32>, tensor<8xf32>) {
      %off = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1)
      %scores_panel = tensor.extract_slice %arg2[%off, 0] [4, 8] [1, 1]
          : tensor<8x8xf32> to tensor<4x8xf32>
      %rmax_panel = tensor.extract_slice %arg3[%off] [4] [1]
          : tensor<8xf32> to tensor<4xf32>

      %panel_upd:2 = scf.for %iv = %c0 to %c2 step %c1
          iter_args(%acc = %scores_panel, %rm_acc = %rmax_panel)
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
        %rmax_new = linalg.generic {
            indexing_maps = [affine_map<(i, j) -> (i, j)>,
                             affine_map<(i, j) -> (i)>],
            iterator_types = ["parallel", "reduction"]}
            ins(%score_tile : tensor<4x4xf32>) outs(%rm_acc : tensor<4xf32>) {
          ^bb0(%in: f32, %out: f32):
            %mx = arith.maxnumf %in, %out : f32
            linalg.yield %mx : f32
        } -> tensor<4xf32>
        scf.yield %updated, %rmax_new : tensor<4x8xf32>, tensor<4xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %panel_upd#0 into %arg2[%off, 0] [4, 8] [1, 1]
            : tensor<4x8xf32> into tensor<8x8xf32>
        tensor.parallel_insert_slice %panel_upd#1 into %arg3[%off] [4] [1]
            : tensor<4xf32> into tensor<8xf32>
      }
    }

    %shift_init = tensor.empty() : tensor<8x8xf32>
    %shift = linalg.generic {fuse,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%scores, %rmax : tensor<8x8xf32>, tensor<8xf32>)
        outs(%shift_init : tensor<8x8xf32>) {
      ^bb0(%in: f32, %m: f32, %out: f32):
        %shifted = arith.subf %in, %m : f32
        linalg.yield %shifted : f32
    } -> tensor<8x8xf32>
    return %shift : tensor<8x8xf32>
  }
}

// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}) -> (tensor<4x8xf32>, tensor<4xf32>, tensor<4x8xf32>)
// CHECK: %[[RMAX_NEW:.*]] = linalg.generic {{.*}} ins(%{{.*}} : tensor<4x4xf32>) outs(%{{.*}} : tensor<4xf32>)
// CHECK-NOT: tensor.extract_slice %[[RMAX_NEW]]
// CHECK: %[[SIDECAR:.*]] = linalg.generic {{.*}} ins(%{{.*}}, %[[RMAX_NEW]] : tensor<4x4xf32>, tensor<4xf32>)
