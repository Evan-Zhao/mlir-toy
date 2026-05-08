// RUN: mlir-opt %s \
// RUN:   --load-dialect-plugin=%neptune_linalg_ext_plugin \
// RUN:   --transform-preload-library=transform-library-paths=%S/Inputs/clone_fuse_elemwise.transform.mlir \
// RUN:   --transform-interpreter 2>&1 | FileCheck %s

// The forall tiles along i (rows). Both scores and rmax share the same i-tile
// offset. The consumer scores - rmax broadcasts rmax across the j dimension.
module {
  func.func @toy(%arg0: tensor<8x8xf32>) -> tensor<8xf32> {
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

      // Both panels are extracted at the same i-offset
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

    // One elemwise op consuming both forall results (broadcast rmax across j)
    %shift_init = tensor.empty() : tensor<8x8xf32>
    %shift = linalg.generic {
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

    // Reduction
    %sum_init_e = tensor.empty() : tensor<8xf32>
    %zero = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero : f32)
        outs(%sum_init_e : tensor<8xf32>) -> tensor<8xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%shift : tensor<8x8xf32>) outs(%sum_init : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %next = arith.addf %in, %out : f32
        linalg.yield %next : f32
    } -> tensor<8xf32>
    return %sum : tensor<8xf32>
  }
}

// The forall should now have 3 shared_outs.
// CHECK: scf.forall {{.*}} shared_outs({{.*}}, {{.*}}, {{.*}})
// After fusion, the inner for should have a 3rd iter_arg for the sidecar.
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}) -> (tensor<4x8xf32>, tensor<4xf32>, tensor<4x8xf32>)
// CHECK: arith.subf
// CHECK: tensor.insert_slice
// CHECK: scf.yield
