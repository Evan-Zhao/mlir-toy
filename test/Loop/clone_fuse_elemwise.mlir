// RUN: neptune-opt %s --transform-interpreter --split-input-file 2>&1 | FileCheck %s

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
      transform.fusion.clone_fuse_elemwise %elemwise into %forall_loop, %inner_loop
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

// -----

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
    %elemwise = transform.structured.match ops{["linalg.generic"]} attributes {fuse} in %func
        : (!transform.any_op) -> !transform.any_op
    %sidecar =
      transform.fusion.clone_fuse_elemwise %elemwise into %forall_loop, %inner_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @toy(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
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
        %new_row = tensor.insert_slice %updated_row into %row_acc[0] [4] [1]
            : tensor<4xf32> into tensor<4xf32>
        scf.yield %new_full, %new_row : tensor<4x8xf32>, tensor<4xf32>
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
    %sub = linalg.generic {fuse,
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
    return %sub : tensor<8x8xf32>
  }
}

// The forall should now have 3 shared_outs.
// CHECK: %{{.*}}:3 = scf.forall
// CHECK-SAME: shared_outs({{[^)]*}}, {{[^)]*}}, {{[^)]*}}) -> (tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>)
// After fusion, the inner for should have a 3rd iter_arg/result for the sidecar.
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}) -> (tensor<4x8xf32>, tensor<4xf32>, tensor<4x8xf32>)
// CHECK:      linalg.generic
// CHECK:      arith.subf
// CHECK:      scf.yield

// -----

// Two unary elemwise consumers where B consumes A. This exercises sidecar
// chaining from one fused sidecar into the next.

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
      transform.fusion.clone_fuse_elemwise %elemwise into %forall_loop, %inner_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @toy(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 5.000000e-01 : f32

    %scores_init = tensor.empty() : tensor<8x8xf32>

    %scores = scf.forall (%arg1) in (2)
        shared_outs(%arg2 = %scores_init)
        -> tensor<8x8xf32> {
      %off = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1)
      %scores_panel = tensor.extract_slice %arg2[%off, 0] [4, 8] [1, 1]
          : tensor<8x8xf32> to tensor<4x8xf32>

      %panel_upd = scf.for %iv = %c0 to %c2 step %c1
          iter_args(%acc = %scores_panel)
          -> tensor<4x8xf32> {
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
        tensor.parallel_insert_slice %panel_upd into %arg2[%off, 0] [4, 8] [1, 1]
            : tensor<4x8xf32> into tensor<8x8xf32>
      }
    }

    // The first consumer reads the loop result directly.
    %sub_init = tensor.empty() : tensor<8x8xf32>
    %sub = linalg.generic {fuse0,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%scores : tensor<8x8xf32>)
        outs(%sub_init : tensor<8x8xf32>) {
      ^bb0(%a: f32, %out: f32):
        %x = arith.subf %a, %cst : f32
        linalg.yield %x : f32
    } -> tensor<8x8xf32>

    // The second consumer reads the result of the first, forming a chain.
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

// One original loop result plus two fused sidecars should produce three loop
// results at both the forall and inner for levels.
// CHECK: %{{.*}}:3 = scf.forall
// CHECK-SAME: shared_outs({{[^)]*}}, {{[^)]*}}, {{[^)]*}}) -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}) -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: scf.yield

// -----

// Two unary elemwise consumers A and B both read directly from the same loop
// result.
// This exercises the case where the fused sidecars are independent siblings.

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
      transform.fusion.clone_fuse_elemwise %elemwise into %forall_loop, %inner_loop
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @toy(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 5.000000e-01 : f32

    %scores_init = tensor.empty() : tensor<8x8xf32>

    %scores = scf.forall (%arg1) in (2)
        shared_outs(%arg2 = %scores_init)
        -> tensor<8x8xf32> {
      %off = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1)
      %scores_panel = tensor.extract_slice %arg2[%off, 0] [4, 8] [1, 1]
          : tensor<8x8xf32> to tensor<4x8xf32>

      %panel_upd = scf.for %iv = %c0 to %c2 step %c1
          iter_args(%acc = %scores_panel)
          -> tensor<4x8xf32> {
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
        tensor.parallel_insert_slice %panel_upd into %arg2[%off, 0] [4, 8] [1, 1]
            : tensor<4x8xf32> into tensor<8x8xf32>
      }
    }

    // Two independent consumers read the same loop result directly.
    %sub_init = tensor.empty() : tensor<8x8xf32>
    %sub = linalg.generic {fuse0,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%scores : tensor<8x8xf32>)
        outs(%sub_init : tensor<8x8xf32>) {
      ^bb0(%a: f32, %out: f32):
        %x = arith.subf %a, %cst : f32
        linalg.yield %x : f32
    } -> tensor<8x8xf32>

    %mul_init = tensor.empty() : tensor<8x8xf32>
    %mul = linalg.generic {fuse1,
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%scores : tensor<8x8xf32>)
        outs(%mul_init : tensor<8x8xf32>) {
      ^bb0(%a: f32, %out: f32):
        %x = arith.mulf %a, %cst : f32
        linalg.yield %x : f32
    } -> tensor<8x8xf32>
    return %mul : tensor<8x8xf32>
  }
}

// One original loop result plus two sibling sidecars should produce three loop
// results at both the forall and inner for levels.
// CHECK: %{{.*}}:3 = scf.forall
// CHECK-SAME: shared_outs({{[^)]*}}, {{[^)]*}}, {{[^)]*}}) -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}) -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: scf.yield
