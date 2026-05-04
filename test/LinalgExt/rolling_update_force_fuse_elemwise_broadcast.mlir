// RUN: mlir-opt %s \
// RUN:   --load-dialect-plugin=%neptune_linalg_ext_plugin \
// RUN:   --transform-preload-library=transform-library-paths=%S/Inputs/rolling_update_force_fuse_elemwise_broadcast.transform.mlir \
// RUN:   --transform-interpreter | FileCheck %s
//
// Exercise the same broadcast access pattern from the attention softmax
// pipeline: the streaming loop produces two results (a 2D full tensor and a 1D
// projection), and the out-of-loop elementwise chain consumes both with a
// broadcast on the second operand.
//
// This mirrors the real attention pattern:
//   full = scores[b, h, :, :]    (2D, tile dim=1)
//   row  = rowmax[b, h, :]        (1D, tile dim=0)
//   elemwise(full, row)[i, j] = full[i, j] + row[j]

module {
  func.func @toy(%arg0: tensor<8x8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 5.000000e-01 : f32

    // ----------------------------------------------------------------
    // Streaming loop with TWO tile-carried results:
    //   %full#0 : tensor<8x8xf32>  — full matrix built tile-by-tile
    //   %row#1  : tensor<8xf32>   — per-row max    built tile-by-tile
    // ----------------------------------------------------------------
    %scores_init = tensor.empty() : tensor<8x8xf32>
    %row_init_e = tensor.empty() : tensor<8xf32>
    %zero = arith.constant 0.0 : f32
    %row_init = linalg.fill ins(%zero : f32)
        outs(%row_init_e : tensor<8xf32>) -> tensor<8xf32>

    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index

    %full:2 = scf.for %arg1 = %c0 to %c8 step %c4
        iter_args(%arg2 = %scores_init, %arg3 = %row_init)
        -> (tensor<8x8xf32>, tensor<8xf32>) {

      // Tile from input (rows %arg1..%arg1+4)
      %tile = tensor.extract_slice %arg0[%arg1, 0] [4, 8] [1, 1]
          : tensor<8x8xf32> to tensor<4x8xf32>
      // Tile from full matrix (same row range)
      %full_tile = tensor.extract_slice %arg2[%arg1, 0] [4, 8] [1, 1]
          : tensor<8x8xf32> to tensor<4x8xf32>
      // Tile from row vector (same row range)
      %row_tile = tensor.extract_slice %arg3[%arg1] [4] [1]
          : tensor<8xf32> to tensor<4xf32>

      // Compute full tile update
      %updated_full = linalg.generic {
          indexing_maps = [affine_map<(i, j) -> (i, j)>,
                           affine_map<(i, j) -> (i, j)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%tile : tensor<4x8xf32>) outs(%full_tile : tensor<4x8xf32>) {
        ^bb0(%in: f32, %out: f32):
          %x = arith.addf %in, %out : f32
          linalg.yield %x : f32
      } -> tensor<4x8xf32>

      // Compute row max (reduction along j)
      %updated_row = linalg.generic {
          indexing_maps = [affine_map<(i, j) -> (i, j)>,
                           affine_map<(i, j) -> (i)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%tile : tensor<4x8xf32>) outs(%row_tile : tensor<4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %x = arith.maxnumf %in, %out : f32
          linalg.yield %x : f32
      } -> tensor<4xf32>

      %new_full = tensor.insert_slice %updated_full into %arg2[%arg1, 0] [4, 8] [1, 1]
          : tensor<4x8xf32> into tensor<8x8xf32>
      %new_row = tensor.insert_slice %updated_row into %arg3[%arg1] [4] [1]
          : tensor<4xf32> into tensor<8xf32>

      scf.yield %new_full, %new_row : tensor<8x8xf32>, tensor<8xf32>
    }

    // ----------------------------------------------------------------
    // Elementwise op consuming BOTH loop results with a broadcast:
    //   sub[i, j] = full[i, j] - row[i]
    //   row uses indexing map  (i, j) -> (i)   (broadcast over j)
    //
    // This is the same pattern as the attention softmax:
    //   exp[i, j] = exp(scores[i, j] - rowmax[i])
    // ----------------------------------------------------------------
    %sub_init = tensor.empty() : tensor<8x8xf32>
    %sub = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%full#0, %full#1 : tensor<8x8xf32>, tensor<8xf32>)
        outs(%sub_init : tensor<8x8xf32>) {
      ^bb0(%a: f32, %b: f32, %out: f32):
        %x = arith.subf %a, %b : f32
        linalg.yield %x : f32
    } -> tensor<8x8xf32>

    // ----------------------------------------------------------------
    // Reduction consuming the elementwise result (sum across j)
    // ----------------------------------------------------------------
    %sum_init_e = tensor.empty() : tensor<8xf32>
    %zero_2 = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero_2 : f32)
        outs(%sum_init_e : tensor<8xf32>) -> tensor<8xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%sub : tensor<8x8xf32>) outs(%sum_init : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %next = arith.addf %in, %out : f32
        linalg.yield %next : f32
    } -> tensor<8xf32>
    return %sum : tensor<8xf32>
  }
}

// CHECK: %[[LOOP:.*]]:3 = scf.for
// CHECK-SAME: -> (tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>)
// CHECK:      linalg.generic
// CHECK:      linalg.generic
// CHECK:      %[[SUB_TILE:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map1, #map]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK:      tensor.insert_slice %[[SUB_TILE]]
// CHECK:      scf.yield
// CHECK:      %[[SUB_FULL:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map1, #map]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
