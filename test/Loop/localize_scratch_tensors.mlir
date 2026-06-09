// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: func.func @localize_inner_scratch
// CHECK: %[[ROW_INIT:.*]] = linalg.fill
// CHECK: %[[FOR:.*]] = scf.for %[[J:.*]] = %c0 to %c2 step %c1 iter_args(%[[ROW:.*]] = %{{.*}}) -> (tensor<8xf32>) {
// CHECK: %[[OFF:.*]] = affine.apply #map(%[[J]])
// CHECK: %[[MAT_INIT:.*]] = tensor.empty() : tensor<8x4xf32>
// CHECK: %[[MAT_SEED:.*]] = linalg.fill
// CHECK: %[[MAT:.*]] = linalg.matmul
// CHECK: %[[SCRATCH:.*]] = tensor.empty() : tensor<8x4xf32>
// CHECK: %[[MAPPED:.*]] = linalg.generic
// CHECK-NOT: tensor.insert_slice %[[MAPPED]] into %{{.*}} : tensor<8x4xf32> into tensor<8x8xf32>
// CHECK: %[[ROW_NEXT:.*]] = linalg.generic
// CHECK: scf.yield %[[ROW_NEXT]] : tensor<8xf32>

#map = affine_map<(d0) -> (d0 * 4)>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.scf.localize_scratch_tensors %func : !transform.any_op
    transform.yield
  }

  func.func @localize_inner_scratch(%arg0: tensor<8x4xf32>, %arg1: tensor<8x4xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.0 : f32
    %neg_inf = arith.constant -3.40282347E+38 : f32

    %rows = tensor.empty() : tensor<8xf32>
    %rows_init = linalg.fill ins(%neg_inf : f32) outs(%rows : tensor<8xf32>) -> tensor<8xf32>
    %result = scf.forall (%i) in (1) shared_outs(%out = %rows_init) -> (tensor<8xf32>) {
      %scratch = tensor.empty() : tensor<8x8xf32>
      %row_init = tensor.extract_slice %out[0] [8] [1] : tensor<8xf32> to tensor<8xf32>
      %inner:2 = scf.for %j = %c0 to %c2 step %c1
          iter_args(%panel = %scratch, %row = %row_init) -> (tensor<8x8xf32>, tensor<8xf32>) {
        %offset = affine.apply #map(%j)
        %lhs = tensor.extract_slice %arg0[0, 0] [8, 4] [1, 1] : tensor<8x4xf32> to tensor<8x4xf32>
        %rhs = tensor.extract_slice %arg1[%offset, 0] [4, 4] [1, 1] : tensor<8x4xf32> to tensor<4x4xf32>
        %matmul_init = tensor.empty() : tensor<8x4xf32>
        %matmul_seed = linalg.fill ins(%zero : f32) outs(%matmul_init : tensor<8x4xf32>) -> tensor<8x4xf32>
        %matmul = linalg.matmul
            ins(%lhs, %rhs : tensor<8x4xf32>, tensor<4x4xf32>)
            outs(%matmul_seed : tensor<8x4xf32>) -> tensor<8x4xf32>
        %panel_slice = tensor.extract_slice %panel[0, %offset] [8, 4] [1, 1] : tensor<8x8xf32> to tensor<8x4xf32>
        %scaled = linalg.generic
            {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
             iterator_types = ["parallel", "parallel"]}
            ins(%matmul : tensor<8x4xf32>)
            outs(%panel_slice : tensor<8x4xf32>) {
          ^bb0(%in: f32, %acc: f32):
            %v = arith.addf %in, %zero : f32
            linalg.yield %v : f32
        } -> tensor<8x4xf32>
        %panel_next = tensor.insert_slice %scaled into %panel[0, %offset] [8, 4] [1, 1]
            : tensor<8x4xf32> into tensor<8x8xf32>
        %row_next = linalg.generic
            {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
             iterator_types = ["parallel", "reduction"]}
            ins(%scaled : tensor<8x4xf32>)
            outs(%row : tensor<8xf32>) {
          ^bb0(%in: f32, %acc: f32):
            %m = arith.maximumf %in, %acc : f32
            linalg.yield %m : f32
        } -> tensor<8xf32>
        scf.yield %panel_next, %row_next : tensor<8x8xf32>, tensor<8xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %inner#1 into %out[0] [8] [1] : tensor<8xf32> into tensor<8xf32>
      }
    }
    return %result : tensor<8xf32>
  }

  // CHECK-LABEL: func.func @localize_multi_use_fill_slices
  // CHECK-NOT: tensor.empty() : tensor<8x8xf32>
  // CHECK-COUNT-2: linalg.fill
  // CHECK-NOT: tensor.extract_slice
  func.func @localize_multi_use_fill_slices() -> (tensor<8x4xf32>, tensor<8x4xf32>) {
    %zero = arith.constant 0.0 : f32
    %full_empty = tensor.empty() : tensor<8x8xf32>
    %full = linalg.fill ins(%zero : f32) outs(%full_empty : tensor<8x8xf32>) -> tensor<8x8xf32>

    %left = tensor.extract_slice %full[0, 0] [8, 4] [1, 1]
        : tensor<8x8xf32> to tensor<8x4xf32>
    %right = tensor.extract_slice %full[0, 4] [8, 4] [1, 1]
        : tensor<8x8xf32> to tensor<8x4xf32>

    return %left, %right : tensor<8x4xf32>, tensor<8x4xf32>
  }
}
