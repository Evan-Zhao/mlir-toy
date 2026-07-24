// RUN: neptune-opt --transform-interpreter %s 2>&1 | FileCheck %s
//
// Transform-dialect schedule for Torch-MLIR ALiBi causal attention.
// This intentionally starts from the masked-attention schedule because ALiBi
// is a score-side bias before the causal mask and online softmax.

!any = !transform.any_op

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_4d_matmul_transb(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.ta.einsum %candidate
        {equation = "b h i d, b h j d -> b h i j"} : (!any) -> !any
    transform.yield %matched : !any
  }

  transform.named_sequence @__transform_main(%module: !any) {
    %func0 = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    %func = transform.apply_registered_pass "stablehlo-to-ta" to %func0 : (!any) -> !any
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.ta.exp_to_exp2
      transform.apply_patterns.ta.sink_div_after_matmul
    } : !any
    transform.apply_cse to %func : !any
    %bmm0 = transform.collect_matching @match_4d_matmul_transb in %func : (!any) -> !any
    transform.ta.to_linalg %func : !any

    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 64, 64, 0] : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %bmax, %_2 = transform.fusion.find_next_reduction %forall_loop : (!any) -> (!any, !any)
    %prefix = transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %bmax { inline_elementwise } : (!any, !any) -> !any
    transform.linalg.erase_unused_operands_and_results %prefix : !any

    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.scf.fuse_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    %bmm1, %elemwise = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    %_3 = transform.fusion.repair_reduction_frontier
        %bmm1 reduce_producer %fused_bmax
        substituting elemwise %elemwise -> %elemwise_sidecars
        into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    %bsum, %elemwise_1 = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_4 = transform.fusion.repair_reduction_frontier
        %bsum reduce_producer %fused_bmax
        substituting elemwise %elemwise_1 -> %elemwise_sidecars_1
        into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    %ret = transform.structured.match ops{["func.return"]} in %func : (!any) -> !any
    transform.fusion.greedy_consumers_into_producer %forall_loop[0] until %ret : (!any, !any) -> !any

    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !any
    transform.apply_cse to %func : !any

    // 0xFF800000: -inf in f32
    %live_loop, %mixed_loop = transform.loop.specialize_dead_tile in %j0_loop
        {dead_value = 0xFF800000 : f32} : (!any) -> (!any, !any)

    // --- HTile lowering begins ---
    transform.htile.linalg_to_semantic %func : !any
    %launches, %kernels = transform.htile.outline_kernels %forall_loop
        {kernel_names = ["attention_kernel"]} : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.verify %func : !any
    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf16>, %arg2: tensor<1x4x1024x64xf16>, %arg3: tensor<4xf32>) -> tensor<1x4x1024x64xf16> {
    %c = stablehlo.constant dense<0> : tensor<1024x1024xi64>
    %c_0 = stablehlo.constant dense<0> : tensor<1024xi64>
    %c_1 = stablehlo.constant dense<1> : tensor<1024xi64>
    %c_2 = stablehlo.constant dense<1024> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %c_4 = stablehlo.constant dense<true> : tensor<1024x1024xi1>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_5 = stablehlo.constant dense<false> : tensor<1024x1024xi1>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_7 = arith.constant dense<1.250000e-01> : tensor<1xf64>
    %0 = stablehlo.convert %arg0 : (tensor<1x4x1024x64xf16>) -> tensor<1x4x1024x64xf32>
    %1 = stablehlo.convert %arg1 : (tensor<1x4x1024x64xf16>) -> tensor<1x4x1024x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1, 3, 2] : (tensor<1x4x1024x64xf32>) -> tensor<1x4x64x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<1x4x64x1024xf32>) -> tensor<1x4x64x1024xf32>
    %4 = stablehlo.dot_general %0, %3, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x1024x64xf32>, tensor<1x4x64x1024xf32>) -> tensor<1x4x1024x1024xf32>
    %5 = stablehlo.convert %cst_7 : (tensor<1xf64>) -> tensor<1xf32>
    %6 = stablehlo.reshape %5 : (tensor<1xf32>) -> tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<1x4x1024x1024xf32>
    %8 = stablehlo.multiply %4, %7 : tensor<1x4x1024x1024xf32>
    %9 = stablehlo.convert %c_3 : (tensor<i64>) -> tensor<f64>
    %10 = stablehlo.convert %c_2 : (tensor<i64>) -> tensor<f64>
    %11 = stablehlo.divide %10, %9 : tensor<f64>
    %12 = stablehlo.ceil %11 : tensor<f64>
    %13 = stablehlo.convert %12 : (tensor<f64>) -> tensor<i64>
    %14 = stablehlo.reshape %13 : (tensor<i64>) -> tensor<1xi64>
    %15 = stablehlo.dynamic_iota %14, dim = 0 : (tensor<1xi64>) -> tensor<1024xi64>
    %16 = stablehlo.multiply %15, %c_1 : tensor<1024xi64>
    %17 = stablehlo.add %16, %c_0 : tensor<1024xi64>
    %18 = stablehlo.convert %17 : (tensor<1024xi64>) -> tensor<1024xf32>
    %19 = stablehlo.reshape %18 : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %20 = stablehlo.reshape %18 : (tensor<1024xf32>) -> tensor<1024x1xf32>
    %21 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<1024x1024xf32>
    %22 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<1024x1xf32>) -> tensor<1024x1024xf32>
    %23 = stablehlo.subtract %21, %22 : tensor<1024x1024xf32>
    %24 = stablehlo.reshape %arg3 : (tensor<4xf32>) -> tensor<4x1xf32>
    %25 = stablehlo.reshape %24 : (tensor<4x1xf32>) -> tensor<4x1x1xf32>
    %26 = stablehlo.broadcast_in_dim %23, dims = [1, 2] : (tensor<1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %27 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<4x1x1xf32>) -> tensor<4x1024x1024xf32>
    %28 = stablehlo.multiply %26, %27 : tensor<4x1024x1024xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [1, 2, 3] : (tensor<4x1024x1024xf32>) -> tensor<1x4x1024x1024xf32>
    %30 = stablehlo.add %8, %29 : tensor<1x4x1024x1024xf32>
    %31 = stablehlo.iota dim = 1 : tensor<1024x1024xi64>
    %32 = stablehlo.iota dim = 0 : tensor<1024x1024xi64>
    %33 = stablehlo.add %32, %c : tensor<1024x1024xi64>
    %34 = stablehlo.compare LE, %31, %33, SIGNED : (tensor<1024x1024xi64>, tensor<1024x1024xi64>) -> tensor<1024x1024xi1>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1] : (tensor<1024x1024xi1>) -> tensor<1024x1024xi1>
    %36 = stablehlo.select %35, %c_4, %c_5 : tensor<1024x1024xi1>, tensor<1024x1024xi1>
    %37 = stablehlo.reshape %36 : (tensor<1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2, 3] : (tensor<1x1x1024x1024xi1>) -> tensor<1x4x1024x1024xi1>
    %39 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2, 3] : (tensor<1x4x1024x1024xf32>) -> tensor<1x4x1024x1024xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x1024x1024xf32>
    %41 = stablehlo.select %38, %39, %40 : tensor<1x4x1024x1024xi1>, tensor<1x4x1024x1024xf32>
    %42 = stablehlo.reduce(%41 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x1024x1024xf32>, tensor<f32>) -> tensor<1x4x1024xf32>
    %43 = stablehlo.reshape %42 : (tensor<1x4x1024xf32>) -> tensor<1x4x1024x1xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2, 3] : (tensor<1x4x1024x1xf32>) -> tensor<1x4x1024x1024xf32>
    %45 = stablehlo.subtract %41, %44 : tensor<1x4x1024x1024xf32>
    %46 = stablehlo.exponential %45 : tensor<1x4x1024x1024xf32>
    %47 = stablehlo.reduce(%46 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x4x1024x1024xf32>, tensor<f32>) -> tensor<1x4x1024xf32>
    %48 = stablehlo.reshape %47 : (tensor<1x4x1024xf32>) -> tensor<1x4x1024x1xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2, 3] : (tensor<1x4x1024x1xf32>) -> tensor<1x4x1024x1024xf32>
    %50 = stablehlo.divide %46, %49 : tensor<1x4x1024x1024xf32>
    %51 = stablehlo.convert %arg2 : (tensor<1x4x1024x64xf16>) -> tensor<1x4x1024x64xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2, 3] : (tensor<1x4x1024x64xf32>) -> tensor<1x4x1024x64xf32>
    %53 = stablehlo.dot_general %50, %52, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x1024x1024xf32>, tensor<1x4x1024x64xf32>) -> tensor<1x4x1024x64xf32>
    %54 = stablehlo.convert %53 : (tensor<1x4x1024x64xf32>) -> tensor<1x4x1024x64xf16>
    return %54 : tensor<1x4x1024x64xf16>
  }
}

// CHECK-LABEL: func.func @attention(
// CHECK-SAME: %arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf16>, %arg2: tensor<1x4x1024x64xf16>, %arg3: tensor<4xf32>) -> tensor<1x4x1024x64xf16>
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf32>
// CHECK-NOT: tensor.empty() : tensor<4x1024x64xf16>
// CHECK: htile.launch_func @attention_kernel
// CHECK-SAME: {program_bounds = array<i64: 4, 16>}
// CHECK-NOT: scf.forall
// CHECK: return

// CHECK-LABEL: htile.kernel @attention_kernel
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<4xf32>
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: memref<1x4x1024x64xf16>
// CHECK-SAME: attributes {program_bounds = array<i64: 4, 16>}
// CHECK: htile.program_id 0
// CHECK: htile.program_id 1
// CHECK-NOT: scf.forall
// CHECK: htile.full %{{.*}} : f32 -> tensor<64xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<64x64xf32>
// CHECK: %[[RAW_BOUND:.+]] = arith.maxsi %{{.*}}, %c0 : index
// CHECK: %[[CAPPED_BOUND:.+]] = arith.minsi %[[RAW_BOUND]], %c16 : index
// CHECK: %[[LIVE:.+]]:3 = scf.for %[[J_TILE:.*]] = %c0 to %[[CAPPED_BOUND]] step %c1 iter_args(
// CHECK: htile.load %arg0
// CHECK: htile.load %arg1
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} {transpose_b} : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.load %arg2
// CHECK: htile.broadcast %{{.*}} dimensions = [0, 1] : tensor<f32> -> tensor<64x64xf32>
// CHECK: htile.arange %c0 to %c64 : tensor<64xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [1] : tensor<64xindex> -> tensor<64x64xindex>
// CHECK: arith.muli %{{.*}}, %{{.*}} {{.*}} : tensor<64x64xindex>
// CHECK: arith.addi %{{.*}}, %{{.*}} : tensor<64x64xindex>
// CHECK: htile.arange %c0 to %c64 : tensor<64xindex>
// CHECK: htile.broadcast %{{.*}} dimensions = [0] : tensor<64xindex> -> tensor<64x64xindex>
// CHECK: arith.subf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<64x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: %[[LIVE_SCORES:.+]] = arith.addf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK-NOT: arith.select
// CHECK: htile.reduce %[[LIVE_SCORES]] axis 1 kind "max" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: math.exp2
// CHECK: htile.load %arg3
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: %[[DEAD_BOUND_RAW:.+]] = affine.apply
// CHECK: %[[DEAD_BOUND_MAX:.+]] = arith.maxsi %[[DEAD_BOUND_RAW]], %c0 : index
// CHECK: %[[DEAD_BOUND:.+]] = arith.minsi %[[DEAD_BOUND_MAX]], %c16 : index
// CHECK: %[[MIXED:.+]]:3 = scf.for %{{.*}} = %[[CAPPED_BOUND]] to %[[DEAD_BOUND]] step %c1 iter_args(%{{.*}} = %[[LIVE]]#0, %{{.*}} = %[[LIVE]]#1, %{{.*}} = %[[LIVE]]#2)
// CHECK: htile.load %arg0
// CHECK: htile.load %arg1
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} {transpose_b} : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.load %arg2
// CHECK: htile.broadcast %{{.*}} dimensions = [0, 1] : tensor<f32> -> tensor<64x64xf32>
// CHECK: arith.subf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: htile.full %{{.*}} : f32 -> tensor<64x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.addf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.cmpi sle, %{{.*}}, %{{.*}} : tensor<64x64xi64>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "max" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: math.exp2
// CHECK: htile.load %arg3
// CHECK: htile.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: htile.reduce %{{.*}} axis 1 kind "sum" : tensor<64x64xf32> -> tensor<64xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<64x64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK: htile.store %{{.*}}, %arg4
// CHECK-NOT: tensor.parallel_insert_slice
// CHECK-NOT: ta.
// CHECK: htile.return
