// RUN: mlir-opt %s \
// RUN:   --load-dialect-plugin=%neptune_linalg_ext_plugin \
// RUN:   --transform-preload-library=transform-library-paths=%S/Inputs/rolling_update_fwd_frontier.transform.mlir \
// RUN:   --transform-interpreter | FileCheck %s

// CHECK: IR printer
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: math.exp
// CHECK: IR printer
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK: arith.addf

module {
  func.func @toy(%arg0: tensor<128x128xf32>) -> tensor<128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 5.000000e-01 : f32
    %neg_inf = arith.constant -3.40282347E+38 : f32
    %scores_init = tensor.empty() : tensor<128x128xf32>
    %row_max_init_e = tensor.empty() : tensor<128xf32>
    %row_max_init = linalg.fill ins(%neg_inf : f32) outs(%row_max_init_e : tensor<128xf32>) -> tensor<128xf32>
    %scores, %row_max = scf.forall (%arg1) in (2) shared_outs(%arg2 = %scores_init, %arg3 = %row_max_init) -> (tensor<128x128xf32>, tensor<128xf32>) {
      %i = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg1)
      %scores_tile = tensor.extract_slice %arg2[%i, 0] [64, 128] [1, 1] : tensor<128x128xf32> to tensor<64x128xf32>
      %row_max_tile = tensor.extract_slice %arg3[%i] [64] [1] : tensor<128xf32> to tensor<64xf32>
      %loop_scores, %loop_row_max = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %scores_tile, %arg6 = %row_max_tile) -> (tensor<64x128xf32>, tensor<64xf32>) {
        %j = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg4)
        %tile = tensor.extract_slice %arg0[%i, %j] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
        %init = tensor.extract_slice %arg5[0, %j] [64, 64] [1, 1] : tensor<64x128xf32> to tensor<64x64xf32>
        %mapped = linalg.map ins(%tile : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>)
          (%in: f32, %out: f32) {
            %scaled = arith.mulf %in, %cst : f32
            linalg.yield %scaled : f32
          }
        %inserted = tensor.insert_slice %mapped into %arg5[0, %j] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<64x128xf32>
        %next_row_max = linalg.generic {
            indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
            iterator_types = ["parallel", "reduction"]}
            ins(%mapped : tensor<64x64xf32>) outs(%arg6 : tensor<64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %m = arith.maxnumf %in, %out : f32
            linalg.yield %m : f32
        } -> tensor<64xf32>
        scf.yield %inserted, %next_row_max : tensor<64x128xf32>, tensor<64xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %loop_scores into %arg2[%i, 0] [64, 128] [1, 1] : tensor<64x128xf32> into tensor<128x128xf32>
        tensor.parallel_insert_slice %loop_row_max into %arg3[%i] [64] [1] : tensor<64xf32> into tensor<128xf32>
      }
    }
    %exp_init = tensor.empty() : tensor<128x128xf32>
    %exp = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%scores, %row_max : tensor<128x128xf32>, tensor<128xf32>) outs(%exp_init : tensor<128x128xf32>) {
      ^bb0(%in: f32, %mx: f32, %out: f32):
        %sub = arith.subf %in, %mx : f32
        %e = math.exp %sub : f32
        linalg.yield %e : f32
    } -> tensor<128x128xf32>
    %row_sum_init_e = tensor.empty() : tensor<128xf32>
    %zero = arith.constant 0.0 : f32
    %row_sum_init = linalg.fill ins(%zero : f32) outs(%row_sum_init_e : tensor<128xf32>) -> tensor<128xf32>
    %row_sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%exp : tensor<128x128xf32>) outs(%row_sum_init : tensor<128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
    } -> tensor<128xf32>
    return %row_sum : tensor<128xf32>
  }
}
