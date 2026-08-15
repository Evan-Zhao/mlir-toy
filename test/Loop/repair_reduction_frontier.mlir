// RUN: neptune-opt %s --transform-interpreter --split-input-file 2>&1 | FileCheck %s


// CHECK-LABEL: func.func @repair_rowsum_like
// CHECK: scf.forall
// CHECK-SAME: shared_outs({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK: %{{.*}} = linalg.generic {{.*}}iterator_types = ["parallel"]{{.*}}outs(%{{.*}} : tensor<2xf32>)
// CHECK: ^bb0(%[[OLDMAX:.+]]: f32, %[[NEWMAX:.+]]: f32, %[[ACC:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK: %[[DELTA:.+]] = arith.subf %[[OLDMAX]], %[[NEWMAX]] : f32
// CHECK: %[[EXP:.+]] = math.exp %[[DELTA]] : f32
// CHECK: %[[SCALED:.+]] = arith.mulf %[[ACC]], %[[EXP]] : f32
// CHECK: linalg.yield %[[SCALED]] : f32

// Start from an already clone-fused payload so this test only applies frontier repair.
!any = !transform.any_op
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} attributes {case_rowsum} in %module : (!any) -> !any
    %forall_loop = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %inner_loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %producer = transform.structured.match ops{["linalg.generic"]} attributes {producer_rowmax} in %func : (!any) -> !any
    %reduce = transform.structured.match ops{["linalg.generic"]} attributes {rowsum_reduce} in %func : (!any) -> !any
    %original_shift = transform.structured.match ops{["linalg.generic"]} attributes {original_shift} in %func : (!any) -> !any
    %original_exp = transform.structured.match ops{["linalg.generic"]} attributes {original_exp} in %func : (!any) -> !any
    %sidecar_shift = transform.structured.match ops{["linalg.generic"]} attributes {sidecar_shift} in %func : (!any) -> !any
    %sidecar_exp = transform.structured.match ops{["linalg.generic"]} attributes {sidecar_exp} in %func : (!any) -> !any
    %elemwise = transform.merge_handles %original_shift, %original_exp : !any
    %sidecars = transform.merge_handles %sidecar_shift, %sidecar_exp : !any
    %_ = transform.fusion.repair_reduction_frontier
        %reduce reduce_producer %producer
        substituting elemwise %elemwise -> %sidecars
        into %forall_loop, %inner_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.yield
  }
  func.func @repair_rowsum_like(%arg0: tensor<4x4xf32>) -> tensor<4xf32> attributes {case_rowsum} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = tensor.empty() : tensor<4xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32>
    %3 = tensor.empty() : tensor<4x4xf32>
    %4 = tensor.empty() : tensor<4x4xf32>
    %5:4 = scf.forall (%arg1) in (2) shared_outs(%arg2 = %0, %arg3 = %2, %arg4 = %3, %arg5 = %4) -> (tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
      %11 = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg2[%11, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
      %extracted_slice_2 = tensor.extract_slice %arg3[%11] [2] [1] : tensor<4xf32> to tensor<2xf32>
      %extracted_slice_3 = tensor.extract_slice %arg4[%11, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
      %extracted_slice_4 = tensor.extract_slice %arg5[%11, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
      %12:4 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %extracted_slice, %arg8 = %extracted_slice_2, %arg9 = %extracted_slice_3, %arg10 = %extracted_slice_4) -> (tensor<2x4xf32>, tensor<2xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
        %13 = affine.apply #map(%arg6)
        %extracted_slice_5 = tensor.extract_slice %arg0[%11, %13] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
        %extracted_slice_6 = tensor.extract_slice %arg7[0, %13] [2, 2] [1, 1] : tensor<2x4xf32> to tensor<2x2xf32>
        %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_5 : tensor<2x2xf32>) outs(%extracted_slice_6 : tensor<2x2xf32>) {
        ^bb0(%in: f32, %out: f32):
          %18 = arith.mulf %in, %cst_1 : f32
          linalg.yield %18 : f32
        } -> tensor<2x2xf32>
        %15 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%14 : tensor<2x2xf32>) outs(%arg8 : tensor<2xf32>) attrs =  {producer_rowmax} {
        ^bb0(%in: f32, %out: f32):
          %18 = arith.maximumf %in, %out : f32
          linalg.yield %18 : f32
        } -> tensor<2xf32>
        %extracted_slice_7 = tensor.extract_slice %arg9[0, %13] [2, 2] [1, 1] : tensor<2x4xf32> to tensor<2x2xf32>
        %16 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<2x2xf32>, tensor<2xf32>) outs(%extracted_slice_7 : tensor<2x2xf32>) attrs =  {sidecar_shift} {
        ^bb0(%in: f32, %in_13: f32, %out: f32):
          %18 = arith.subf %in, %in_13 : f32
          linalg.yield %18 : f32
        } -> tensor<2x2xf32>
        %inserted_slice_8 = tensor.insert_slice %14 into %arg7[0, %13] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x4xf32>
        %inserted_slice_9 = tensor.insert_slice %15 into %arg8[0] [2] [1] : tensor<2xf32> into tensor<2xf32>
        %inserted_slice_10 = tensor.insert_slice %16 into %arg9[0, %13] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x4xf32>
        %extracted_slice_11 = tensor.extract_slice %arg10[0, %13] [2, 2] [1, 1] : tensor<2x4xf32> to tensor<2x2xf32>
        %17 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<2x2xf32>) outs(%extracted_slice_11 : tensor<2x2xf32>) attrs =  {sidecar_exp} {
        ^bb0(%in: f32, %out: f32):
          %18 = math.exp %in : f32
          linalg.yield %18 : f32
        } -> tensor<2x2xf32>
        %inserted_slice_12 = tensor.insert_slice %17 into %arg10[0, %13] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x4xf32>
        scf.yield %inserted_slice_8, %inserted_slice_9, %inserted_slice_10, %inserted_slice_12 : tensor<2x4xf32>, tensor<2xf32>, tensor<2x4xf32>, tensor<2x4xf32>
      }
      %inserted_slice = tensor.insert_slice %12#2 into %arg4[%11, 0] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<4x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %12#0 into %arg2[%11, 0] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<4x4xf32>
        tensor.parallel_insert_slice %12#1 into %arg3[%11] [2] [1] : tensor<2xf32> into tensor<4xf32>
        tensor.parallel_insert_slice %12#2 into %arg4[%11, 0] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<4x4xf32>
        tensor.parallel_insert_slice %12#3 into %arg5[%11, 0] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<4x4xf32>
      }
    }
    %6 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%5#0, %5#1 : tensor<4x4xf32>, tensor<4xf32>) outs(%3 : tensor<4x4xf32>) attrs =  {original_shift} {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %11 = arith.subf %in, %in_2 : f32
      linalg.yield %11 : f32
    } -> tensor<4x4xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<4x4xf32>) outs(%4 : tensor<4x4xf32>) attrs =  {original_exp} {
    ^bb0(%in: f32, %out: f32):
      %11 = math.exp %in : f32
      linalg.yield %11 : f32
    } -> tensor<4x4xf32>
    %8 = tensor.empty() : tensor<4xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<4xf32>) -> tensor<4xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<4x4xf32>) outs(%9 : tensor<4xf32>) attrs =  {rowsum_reduce} {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.addf %in, %out : f32
      linalg.yield %11 : f32
    } -> tensor<4xf32>
    return %10 : tensor<4xf32>
  }
}

// -----

// CHECK-LABEL: func.func @repair_out_like
// CHECK: scf.forall
// CHECK-SAME: shared_outs({{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK: %{{.*}} = linalg.generic {{.*}}iterator_types = ["parallel", "parallel"]{{.*}}outs(%{{.*}} : tensor<2x2xf32>)
// CHECK: ^bb0(%[[OLDSUM:.+]]: f32, %[[NEWSUM:.+]]: f32, %[[ACC:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK: %[[MUL0:.+]] = arith.mulf %[[ACC]], %[[OLDSUM]] : f32
// CHECK: %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[INV:.+]] = arith.divf %[[ONE]], %[[NEWSUM]] : f32
// CHECK: %[[MUL1:.+]] = arith.mulf %[[MUL0]], %[[INV]] : f32
// CHECK: linalg.yield %[[MUL1]] : f32

// Start from an already clone-fused payload so this test only applies frontier repair.
!any = !transform.any_op
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} attributes {case_out} in %module : (!any) -> !any
    %forall_loop = transform.structured.match ops{["scf.forall"]} in %func : (!any) -> !any
    %inner_loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %producer = transform.structured.match ops{["linalg.generic"]} attributes {producer_rowsum} in %func : (!any) -> !any
    %reduce = transform.structured.match ops{["linalg.generic"]} attributes {out_reduce} in %func : (!any) -> !any
    %original_normalize = transform.structured.match ops{["linalg.generic"]} attributes {original_normalize} in %func : (!any) -> !any
    %sidecar_normalize = transform.structured.match ops{["linalg.generic"]} attributes {sidecar_normalize} in %func : (!any) -> !any
    %_ = transform.fusion.repair_reduction_frontier
        %reduce reduce_producer %producer
        substituting elemwise %original_normalize -> %sidecar_normalize
        into %forall_loop, %inner_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.yield
  }
  func.func @repair_out_like(%arg0: tensor<4x4xf32>, %arg1: tensor<4x2xf32>) -> tensor<4x2xf32> attributes {case_out} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = tensor.empty() : tensor<4xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32>
    %3 = tensor.empty() : tensor<4x4xf32>
    %4:3 = scf.forall (%arg2) in (2) shared_outs(%arg3 = %0, %arg4 = %2, %arg5 = %3) -> (tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>) {
      %9 = affine.apply #map(%arg2)
      %extracted_slice = tensor.extract_slice %arg3[%9, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
      %extracted_slice_0 = tensor.extract_slice %arg4[%9] [2] [1] : tensor<4xf32> to tensor<2xf32>
      %extracted_slice_1 = tensor.extract_slice %arg5[%9, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
      %10:3 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %extracted_slice, %arg8 = %extracted_slice_0, %arg9 = %extracted_slice_1) -> (tensor<2x4xf32>, tensor<2xf32>, tensor<2x4xf32>) {
        %11 = affine.apply #map(%arg6)
        %extracted_slice_3 = tensor.extract_slice %arg0[%9, %11] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
        %12 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_3 : tensor<2x2xf32>) outs(%arg8 : tensor<2xf32>) attrs =  {producer_rowsum} {
        ^bb0(%in: f32, %out: f32):
          %14 = arith.addf %in, %out : f32
          linalg.yield %14 : f32
        } -> tensor<2xf32>
        %inserted_slice_4 = tensor.insert_slice %extracted_slice_3 into %arg7[0, %11] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x4xf32>
        %inserted_slice_5 = tensor.insert_slice %12 into %arg8[0] [2] [1] : tensor<2xf32> into tensor<2xf32>
        %extracted_slice_6 = tensor.extract_slice %arg9[0, %11] [2, 2] [1, 1] : tensor<2x4xf32> to tensor<2x2xf32>
        %13 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_3, %12 : tensor<2x2xf32>, tensor<2xf32>) outs(%extracted_slice_6 : tensor<2x2xf32>) attrs =  {sidecar_normalize} {
        ^bb0(%in: f32, %in_8: f32, %out: f32):
          %14 = arith.divf %in, %in_8 : f32
          linalg.yield %14 : f32
        } -> tensor<2x2xf32>
        %inserted_slice_7 = tensor.insert_slice %13 into %arg9[0, %11] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x4xf32>
        scf.yield %inserted_slice_4, %inserted_slice_5, %inserted_slice_7 : tensor<2x4xf32>, tensor<2xf32>, tensor<2x4xf32>
      }
      %inserted_slice = tensor.insert_slice %10#0 into %arg3[%9, 0] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<4x4xf32>
      %inserted_slice_2 = tensor.insert_slice %10#1 into %arg4[%9] [2] [1] : tensor<2xf32> into tensor<4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10#0 into %arg3[%9, 0] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<4x4xf32>
        tensor.parallel_insert_slice %10#1 into %arg4[%9] [2] [1] : tensor<2xf32> into tensor<4xf32>
        tensor.parallel_insert_slice %10#2 into %arg5[%9, 0] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<4x4xf32>
      }
    }
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%4#0, %4#1 : tensor<4x4xf32>, tensor<4xf32>) outs(%3 : tensor<4x4xf32>) attrs =  {original_normalize} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %9 = arith.divf %in, %in_0 : f32
      linalg.yield %9 : f32
    } -> tensor<4x4xf32>
    %6 = tensor.empty() : tensor<4x2xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<4x2xf32>) -> tensor<4x2xf32>
    %8 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5, %arg1 : tensor<4x4xf32>, tensor<4x2xf32>) outs(%7 : tensor<4x2xf32>) attrs =  {out_reduce} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %9 = arith.mulf %in, %in_0 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<4x2xf32>
    return %8 : tensor<4x2xf32>
  }
}
