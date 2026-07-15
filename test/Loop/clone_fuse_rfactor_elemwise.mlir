// RUN: neptune-opt %s --transform-interpreter | FileCheck %s

// SplitK sidecar fusion case:
//   - the forall publishes score tiles and split-local row-max partials,
//   - a write-back reduction merges the row-max partials after the forall,
//   - the sidecar chain reads both the score tensor and the write-back max.
//
// clone_fuse_rfactor_elemwise should clone the sidecar chain into the forall
// and substitute the write-back max with the split-local rfactor max tile.

#map = affine_map<(d0) -> (d0 * 4)>
#id2 = affine_map<(i, j) -> (i, j)>
#row = affine_map<(i, j) -> (i)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %forall_loop = transform.structured.match ops{["scf.forall"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %writeback = transform.structured.match ops{["linalg.generic"]} attributes {writeback} in %func
        : (!transform.any_op) -> !transform.any_op
    %rfactor = transform.structured.match ops{["linalg.generic"]} attributes {rfactor} in %func
        : (!transform.any_op) -> !transform.any_op
    %sub = transform.structured.match ops{["linalg.generic"]} attributes {sidecar0} in %func
        : (!transform.any_op) -> !transform.any_op
    %exp = transform.structured.match ops{["linalg.generic"]} attributes {sidecar1} in %func
        : (!transform.any_op) -> !transform.any_op
    %elemwise = transform.merge_handles %sub, %exp : !transform.any_op
    %sidecars = transform.fusion.clone_fuse_rfactor_elemwise
        %elemwise into %forall_loop substituting (%writeback -> %rfactor)
        : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
          -> !transform.any_op
    transform.yield
  }

  func.func @splitk_sidecar(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %neg_inf = arith.constant -3.40282347E+38 : f32

    %scores_e = tensor.empty() : tensor<8x8xf32>
    %m_rf_e = tensor.empty() : tensor<8x2xf32>
    %m_rf_init = linalg.fill ins(%neg_inf : f32)
        outs(%m_rf_e : tensor<8x2xf32>) -> tensor<8x2xf32>

    %scores, %m_rf = scf.forall (%arg1, %arg2) in (2, 2)
        shared_outs(%arg3 = %scores_e, %arg4 = %m_rf_init)
        -> (tensor<8x8xf32>, tensor<8x2xf32>) {
      %i_off = affine.apply #map(%arg1)
      %j_off = affine.apply #map(%arg2)

      %src = tensor.extract_slice %arg0[%i_off, %j_off] [4, 4] [1, 1]
          : tensor<8x8xf32> to tensor<4x4xf32>
      %score_init = tensor.extract_slice %arg3[%i_off, %j_off] [4, 4] [1, 1]
          : tensor<8x8xf32> to tensor<4x4xf32>
      %score_tile = linalg.generic {
          indexing_maps = [#id2, #id2],
          iterator_types = ["parallel", "parallel"]}
          ins(%src : tensor<4x4xf32>)
          outs(%score_init : tensor<4x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %x = arith.addf %in, %zero : f32
          linalg.yield %x : f32
      } -> tensor<4x4xf32>

      %m_part_init = tensor.extract_slice %arg4[%i_off, %arg2] [4, 1] [1, 1]
          : tensor<8x2xf32> to tensor<4xf32>
      %m_part = linalg.generic {
          rfactor,
          indexing_maps = [#id2, #row],
          iterator_types = ["parallel", "reduction"]}
          ins(%score_tile : tensor<4x4xf32>)
          outs(%m_part_init : tensor<4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %x = arith.maximumf %out, %in : f32
          linalg.yield %x : f32
      } -> tensor<4xf32>

      scf.forall.in_parallel {
        tensor.parallel_insert_slice %score_tile into %arg3[%i_off, %j_off] [4, 4] [1, 1]
            : tensor<4x4xf32> into tensor<8x8xf32>
        tensor.parallel_insert_slice %m_part into %arg4[%i_off, %arg2] [4, 1] [1, 1]
            : tensor<4xf32> into tensor<8x2xf32>
      }
    }

    %m_e = tensor.empty() : tensor<8xf32>
    %m_init = linalg.fill ins(%neg_inf : f32)
        outs(%m_e : tensor<8xf32>) -> tensor<8xf32>
    %m = linalg.generic {
        writeback,
        indexing_maps = [#id2, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%m_rf : tensor<8x2xf32>)
        outs(%m_init : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %x = arith.maximumf %out, %in : f32
        linalg.yield %x : f32
    } -> tensor<8xf32>

    %sub_e = tensor.empty() : tensor<8x8xf32>
    %sub = linalg.generic {
        sidecar0,
        indexing_maps = [#id2, #row, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%scores, %m : tensor<8x8xf32>, tensor<8xf32>)
        outs(%sub_e : tensor<8x8xf32>) {
      ^bb0(%score: f32, %max: f32, %out: f32):
        %x = arith.subf %score, %max : f32
        linalg.yield %x : f32
    } -> tensor<8x8xf32>

    %exp_e = tensor.empty() : tensor<8x8xf32>
    %exp = linalg.generic {
        sidecar1,
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%sub : tensor<8x8xf32>)
        outs(%exp_e : tensor<8x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %x = math.exp2 %in : f32
        linalg.yield %x : f32
    } -> tensor<8x8xf32>

    return %exp : tensor<8x8xf32>
  }
}

// CHECK-LABEL: func.func @splitk_sidecar
// CHECK: %[[LOOP:.*]]:4 = scf.forall
// CHECK-SAME: shared_outs({{[^)]*}}, {{[^)]*}}, {{[^)]*}}, {{[^)]*}}) -> (tensor<8x8xf32>, tensor<8x2xf32>, tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK: %[[SCORE_TILE:.*]] = linalg.generic
// CHECK: %[[M_PART:.*]] = linalg.generic
// CHECK-SAME: rfactor
// CHECK: %[[SUB_INIT:.*]] = tensor.extract_slice {{.*}} : tensor<8x8xf32> to tensor<4x4xf32>
// CHECK: %[[SUB_TILE:.*]] = linalg.generic
// CHECK-SAME: ins(%[[SCORE_TILE]], %[[M_PART]]
// CHECK-SAME: outs(%[[SUB_INIT]]
// CHECK: arith.subf
// CHECK: %[[EXP_INIT:.*]] = tensor.extract_slice {{.*}} : tensor<8x8xf32> to tensor<4x4xf32>
// CHECK: %[[EXP_TILE:.*]] = linalg.generic
// CHECK-SAME: ins(%[[SUB_TILE]]
// CHECK-SAME: outs(%[[EXP_INIT]]
// CHECK: math.exp2
// CHECK: scf.forall.in_parallel
// CHECK: tensor.parallel_insert_slice %[[SUB_TILE]]
// CHECK: tensor.parallel_insert_slice %[[EXP_TILE]]
// CHECK: %[[M:.*]] = linalg.generic
// CHECK-SAME: writeback
// CHECK: %[[ORIG_SUB:.*]] = linalg.generic
// CHECK: ins(%[[LOOP]]#0, %[[M]]
// CHECK-SAME: sidecar0
// CHECK: %[[ORIG_EXP:.*]] = linalg.generic
// CHECK: ins(%[[ORIG_SUB]]
// CHECK-SAME: sidecar1
