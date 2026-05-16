// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s

// Minimal clone-fuse case: the forall/for loop nest produces a single tensor,
// and one out-of-loop unary elementwise consumer is cloned and fused back into
// the nest as an extra sidecar result.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %forall_loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %inner_loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %elemwise = transform.structured.match ops{["linalg.generic"]} attributes {fuse} in %func
        : (!transform.any_op) -> !transform.any_op
    %sidecar =
      transform.loop_ru.clone_fuse_elemwise %elemwise into %forall_loop, %inner_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @toy(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cf2 = arith.constant 2.000000e+00 : f32
    %cst = arith.constant 5.000000e-01 : f32

    %scores_init = tensor.empty() : tensor<8x8xf32>
    %rm_init_e = tensor.empty() : tensor<8xf32>
    %neg_inf = arith.constant -3.40282347E+38 : f32

    %scores = scf.forall (%arg1) in (2) shared_outs(%arg2 = %scores_init) -> tensor<8x8xf32> {
      %off = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1)

      %scores_panel = tensor.extract_slice %arg2[%off, 0] [4, 8] [1, 1]
          : tensor<8x8xf32> to tensor<4x8xf32>

      %tile_upd = scf.for %iv = %c0 to %c2 step %c1 iter_args(%acc = %scores_panel) -> tensor<4x8xf32> {
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
        scf.yield %updated : tensor<4x8xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %tile_upd into %arg2[%off, 0] [4, 8] [1, 1]
            : tensor<4x8xf32> into tensor<8x8xf32>
      }
    }

    %shift_init = tensor.empty() : tensor<8x8xf32>
    // One unary elementwise consumer of the forall result.
    %shift = linalg.generic {
        fuse,   // used for pattern matching
        indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%scores : tensor<8x8xf32>) outs(%shift_init : tensor<8x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %shifted = arith.subf %in, %cf2 : f32
        linalg.yield %shifted : f32
    } -> tensor<8x8xf32>
    return %shift : tensor<8x8xf32>
  }
}

// The forall should now have 2 shared_outs/results: the original scores tensor
// plus one relayed sidecar tensor for the fused unary elementwise op.
// CHECK: %{{.*}}:2 = scf.forall
// CHECK-SAME: shared_outs({{[^)]*}}, {{[^)]*}}) -> (tensor<8x8xf32>, tensor<8x8xf32>)
// After fusion, the inner for should have a 2nd iter_arg/result for the sidecar.
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}) -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK: arith.subf
// CHECK: scf.yield
