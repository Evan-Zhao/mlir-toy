// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter | FileCheck %s
//
// Exercise the same broadcast access pattern from the attention softmax
// pipeline: the forall loop produces two results (a 2D full tensor and a 1D
// projection), and the out-of-loop elementwise chain consumes both with a
// broadcast on the second operand.
//
// This mirrors the real attention pattern:
//   full = scores[b, h, :, :]    (2D, tile dim=1)
//   row  = rowmax[b, h, :]        (1D, tile dim=0)
//   elemwise(full, row)[i, j] = full[i, j] - row[i]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %forall_loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %inner_loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %reduce, %elemwise =
      transform.match.loop_ru.rolling_update_next_reduction %forall_loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %sidecar, %new_forall, %new_inner =
      transform.loop_ru.clone_fuse_elemwise %elemwise into %forall_loop, %inner_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }

  func.func @toy(%arg0: tensor<8x8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 5.000000e-01 : f32

    %scores_init = tensor.empty() : tensor<8x8xf32>
    %row_init_e = tensor.empty() : tensor<8xf32>
    %zero = arith.constant 0.0 : f32
    %row_init = linalg.fill ins(%zero : f32)
        outs(%row_init_e : tensor<8xf32>) -> tensor<8xf32>

    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index

    // Wrap the row-tiling loop inside a forall for the two-loop test.
    %full, %row = scf.forall (%arg1) in (2)
        shared_outs(%arg2 = %scores_init, %arg3 = %row_init)
        -> (tensor<8x8xf32>, tensor<8xf32>) {
      %off = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1)
      %full_panel = tensor.extract_slice %arg2[%off, 0] [4, 8] [1, 1]
          : tensor<8x8xf32> to tensor<4x8xf32>
      %row_panel = tensor.extract_slice %arg3[%off] [4] [1]
          : tensor<8xf32> to tensor<4xf32>

      %panel_upd:2 = scf.for %iv = %c0 to %c2 step %c1
          iter_args(%acc = %full_panel, %row_acc = %row_panel)
          -> (tensor<4x8xf32>, tensor<4xf32>) {
        %j = affine.apply affine_map<(d0) -> (d0 * 4)>(%iv)
        %tile = tensor.extract_slice %arg0[0, %j] [4, 4] [1, 1]
            : tensor<8x8xf32> to tensor<4x4xf32>
        %full_tile = tensor.extract_slice %acc[0, %j] [4, 4] [1, 1]
            : tensor<4x8xf32> to tensor<4x4xf32>
        %updated_full = linalg.generic {
            indexing_maps = [affine_map<(i, j) -> (i, j)>,
                             affine_map<(i, j) -> (i, j)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%tile : tensor<4x4xf32>) outs(%full_tile : tensor<4x4xf32>) {
          ^bb0(%in: f32, %out: f32):
            %x = arith.addf %in, %out : f32
            linalg.yield %x : f32
        } -> tensor<4x4xf32>
        %updated_row = linalg.generic {
            indexing_maps = [affine_map<(i, j) -> (i, j)>,
                             affine_map<(i, j) -> (i)>],
            iterator_types = ["parallel", "reduction"]}
            ins(%tile : tensor<4x4xf32>) outs(%row_acc : tensor<4xf32>) {
          ^bb0(%in: f32, %out: f32):
            %x = arith.maxnumf %in, %out : f32
            linalg.yield %x : f32
        } -> tensor<4xf32>
        %new_full = tensor.insert_slice %updated_full into %acc[0, %j] [4, 4] [1, 1]
            : tensor<4x4xf32> into tensor<4x8xf32>
        scf.yield %new_full, %updated_row : tensor<4x8xf32>, tensor<4xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %panel_upd#0 into %arg2[%off, 0] [4, 8] [1, 1]
            : tensor<4x8xf32> into tensor<8x8xf32>
        tensor.parallel_insert_slice %panel_upd#1 into %arg3[%off] [4] [1]
            : tensor<4xf32> into tensor<8xf32>
      }
    }

    // Elementwise op consuming BOTH loop results with a broadcast:
    //   sub[i, j] = full[i, j] - row[i]
    %sub_init = tensor.empty() : tensor<8x8xf32>
    %sub = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%full, %row : tensor<8x8xf32>, tensor<8xf32>)
        outs(%sub_init : tensor<8x8xf32>) {
      ^bb0(%a: f32, %b: f32, %out: f32):
        %x = arith.subf %a, %b : f32
        linalg.yield %x : f32
    } -> tensor<8x8xf32>

    // Reduction consuming the elementwise result (sum across j)
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

// The forall should now have 3 shared_outs.
// CHECK: scf.forall {{.*}} shared_outs({{.*}}, {{.*}}, {{.*}})
// After fusion, the inner for should have a 3rd iter_arg for the sidecar.
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}) -> (tensor<4x8xf32>, tensor<4xf32>, tensor<4x8xf32>)
// CHECK:      linalg.generic
// CHECK:      linalg.generic
// CHECK:      arith.subf
// CHECK:      tensor.insert_slice
// CHECK:      scf.yield
