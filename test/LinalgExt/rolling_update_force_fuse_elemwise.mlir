// RUN: mlir-opt %s \
// RUN:   --load-dialect-plugin=%neptune_linalg_ext_plugin \
// RUN:   --transform-preload-library=transform-library-paths=%S/Inputs/rolling_update_force_fuse_elemwise.transform.mlir \
// RUN:   --transform-interpreter | FileCheck %s

module {
  func.func @toy(%arg0: tensor<8x8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 5.000000e-01 : f32

    %scores_init = tensor.empty() : tensor<8x8xf32>
    %scores = scf.for %arg1 = %c0 to %c2 step %c1
        iter_args(%arg2 = %scores_init) -> (tensor<8x8xf32>) {
      %j = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1)
      %tile = tensor.extract_slice %arg0[0, %j] [8, 4] [1, 1]
          : tensor<8x8xf32> to tensor<8x4xf32>
      %init = tensor.extract_slice %arg2[0, %j] [8, 4] [1, 1]
          : tensor<8x8xf32> to tensor<8x4xf32>
      %score_tile = linalg.generic {
          indexing_maps = [affine_map<(i, j) -> (i, j)>,
                           affine_map<(i, j) -> (i, j)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%tile : tensor<8x4xf32>) outs(%init : tensor<8x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %scaled = arith.addf %in, %cst : f32
          linalg.yield %scaled : f32
      } -> tensor<8x4xf32>
      %updated = tensor.insert_slice %score_tile into %arg2[0, %j] [8, 4] [1, 1]
          : tensor<8x4xf32> into tensor<8x8xf32>
      scf.yield %updated : tensor<8x8xf32>
    }

    %shift_init = tensor.empty() : tensor<8x8xf32>
    %shift = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%scores : tensor<8x8xf32>) outs(%shift_init : tensor<8x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %shifted = arith.addf %in, %cst : f32
        linalg.yield %shifted : f32
    } -> tensor<8x8xf32>

    %exp_init = tensor.empty() : tensor<8x8xf32>
    %exp = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%shift : tensor<8x8xf32>) outs(%exp_init : tensor<8x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %e = math.exp %in : f32
        linalg.yield %e : f32
    } -> tensor<8x8xf32>

    %sum_init_e = tensor.empty() : tensor<8xf32>
    %zero = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero : f32)
        outs(%sum_init_e : tensor<8xf32>) -> tensor<8xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%exp : tensor<8x8xf32>) outs(%sum_init : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %next = arith.addf %in, %out : f32
        linalg.yield %next : f32
    } -> tensor<8xf32>
    return %sum : tensor<8xf32>
  }
}

// CHECK: %[[LOOP:.*]]:3 = scf.for
// CHECK-SAME: -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK: %[[SCORE_TILE:.*]] = linalg.generic
// CHECK: %[[SHIFT_TILE:.*]] = linalg.generic
// CHECK: arith.addf
// CHECK: %[[SHIFT_INSERT:.*]] = tensor.insert_slice %[[SHIFT_TILE]]
// CHECK: %[[EXP_TILE:.*]] = linalg.generic
// CHECK: math.exp
// CHECK: %[[EXP_INSERT:.*]] = tensor.insert_slice %[[EXP_TILE]]
// CHECK: scf.yield {{.*}}, %[[SHIFT_INSERT]], %[[EXP_INSERT]]
// CHECK: %[[SHIFT:.*]] = linalg.generic
// CHECK: arith.addf
// CHECK: %[[EXP:.*]] = linalg.generic
// CHECK: math.exp
