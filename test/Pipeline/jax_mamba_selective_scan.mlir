// RUN: neptune-opt %s --canonicalize --cse | FileCheck %s --check-prefix=CLEAN
// RUN: neptune-opt %s --canonicalize --cse --transform-interpreter --canonicalize --cse | FileCheck %s
//
// StableHLO payload obtained by inlining the default output of:
//
//   python examples/jax_mamba_selective_scan.py
//
// Unlike the attention payloads in this directory, selective scan arrives as
// a sequential recurrence. The intended persistent-kernel schedule is:
//
//   scf.forall (%b, %c_block) in (8, 12) {
//     %h0 = zero : tensor<1x128x16xf32>
//     %h = scf.for %t = 0 to 2048 iter_args(%h_t = %h0) {
//       %u_t, %delta_t = load [b, t, c_block:c_block+128]
//       %b_t, %c_t = load [b, t, 0:16]
//       %dt = softplus(%delta_t + %delta_bias)
//       %h_next = exp(%dt[..., none] * %a) * %h_t
//               + %dt[..., none] * %b_t[:, none, :] * %u_t[..., none]
//       %y = reduce_sum(%h_next * %c_t[:, none, :], axis = 2)
//          + %d_skip * %u_t
//       store %y, %output[b, t, c_block:c_block+128]
//       scf.yield %h_next
//     }
//   }
//
// In schedule terms, this requires: normalize the StableHLO while to an
// scf.for; destination-bufferize the dynamic output
// update; distribute BxC tiles around the whole recurrence; fuse the per-token
// producer/reduction graph; hoist A, D, and delta_bias tile loads; lower the
// state-dimension reduction; and outline the forall as one persistent kernel.
// `transform.stablehlo.legalize_control_flow`, adapted from IREE, performs the
// loop normalization before StableHLO's native conversion lowers the body to
// Linalg. Running that conversion without loop normalization would leave the
// StableHLO loop shell intact.

!any = !transform.any_op

module @jit_selective_scan attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.stablehlo.legalize_control_flow %funcs : !any
    // Primitive Linalg leaves the output projection as the only generic op,
    // making it a stable anchor for the BxC schedule.
    %func = transform.apply_registered_pass "stablehlo-legalize-to-linalg"
        with options = {"enable-primitive-ops" = true} to %funcs : (!any) -> !any
    transform.apply_patterns to %func {
      transform.apply_patterns.stablehlo.simplify_in_bounds_clamps
      transform.apply_patterns.canonicalization
    } : !any
    %outer_loop = transform.structured.match ops{["scf.for"]} in %func : (!any) -> !any
    %project = transform.structured.match ops{["linalg.generic"]} in %func : (!any) -> !any
    %outer_yield = transform.structured.match ops{["scf.yield"]} in %outer_loop : (!any) -> !any
    %tiled_project, %inner_loop = transform.structured.tile_using_forall
        %project tile_sizes [1, 128, 0] : (!any) -> (!any, !any)

    // Powerful and aggressive fusion to fuse ops before and after the scf.forall loop into the loop.
    transform.fusion.greedy_consumers_into_producer %inner_loop until %outer_yield: (!any, !any) -> !any
    transform.fusion.greedy_input_producers_into_consumer %inner_loop : (!any) -> !any

    // Exchange the time recurrence with the BxC worker loop. This requires a lot of preparation steps
    // (below) because it only works with perfectly nested scf.for and scf.forall.
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization 
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } : !any
    transform.apply_cse to %func : !any
    transform.apply_licm to %outer_loop : !any
    %forall_loop, %for_loop = transform.scf.interchange_for_and_forall
        %outer_loop with %inner_loop : (!any, !any) -> (!any, !any)

    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func {
      transform.apply_patterns.scf.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
      transform.apply_patterns.canonicalization
    } : !any
    transform.apply_cse to %func : !any

    // --- HTile lowering begins ---
    transform.htile.linalg_to_semantic %func : !any
    %launches, %kernels = transform.htile.outline_kernels %forall_loop
        {kernel_names = ["mamba_selective_scan_kernel"]} : (!any) -> (!any, !any)
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.verify %func : !any
    transform.yield
  }

  func.func public @main(%arg0: tensor<8x2048x1536xbf16>, %arg1: tensor<8x2048x1536xbf16>, %arg2: tensor<1536x16xf32>, %arg3: tensor<8x2048x16xbf16>, %arg4: tensor<8x2048x16xbf16>, %arg5: tensor<1536xf32>, %arg6: tensor<1536xf32>) -> (tensor<8x2048x1536xbf16> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<2048> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1536x16xf32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<8x2048x1536xbf16>
    %2:11 = stablehlo.while(%iterArg = %arg0, %iterArg_3 = %arg1, %iterArg_4 = %arg3, %iterArg_5 = %arg4, %iterArg_6 = %arg6, %iterArg_7 = %arg2, %iterArg_8 = %arg5, %iterArg_9 = %c_1, %iterArg_10 = %c_1, %iterArg_11 = %0, %iterArg_12 = %1) : tensor<8x2048x1536xbf16>, tensor<8x2048x1536xbf16>, tensor<8x2048x16xbf16>, tensor<8x2048x16xbf16>, tensor<1536xf32>, tensor<1536x16xf32>, tensor<1536xf32>, tensor<i32>, tensor<i32>, tensor<8x1536x16xf32>, tensor<8x2048x1536xbf16>
    cond {
      %3 = stablehlo.compare LT, %iterArg_9, %c_0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %iterArg_10, %c : tensor<i32>
      %4 = stablehlo.dynamic_slice %iterArg, %c_1, %iterArg_10, %c_1, sizes = [8, 1, 1536] : (tensor<8x2048x1536xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x1x1536xbf16>
      %5 = stablehlo.reshape %4 : (tensor<8x1x1536xbf16>) -> tensor<8x1536xbf16>
      %6 = stablehlo.convert %5 : (tensor<8x1536xbf16>) -> tensor<8x1536xf32>
      %7 = stablehlo.dynamic_slice %iterArg_3, %c_1, %iterArg_10, %c_1, sizes = [8, 1, 1536] : (tensor<8x2048x1536xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x1x1536xbf16>
      %8 = stablehlo.reshape %7 : (tensor<8x1x1536xbf16>) -> tensor<8x1536xbf16>
      %9 = stablehlo.convert %8 : (tensor<8x1536xbf16>) -> tensor<8x1536xf32>
      %10 = stablehlo.dynamic_slice %iterArg_4, %c_1, %iterArg_10, %c_1, sizes = [8, 1, 16] : (tensor<8x2048x16xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x1x16xbf16>
      %11 = stablehlo.reshape %10 : (tensor<8x1x16xbf16>) -> tensor<8x16xbf16>
      %12 = stablehlo.convert %11 : (tensor<8x16xbf16>) -> tensor<8x16xf32>
      %13 = stablehlo.dynamic_slice %iterArg_5, %c_1, %iterArg_10, %c_1, sizes = [8, 1, 16] : (tensor<8x2048x16xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x1x16xbf16>
      %14 = stablehlo.reshape %13 : (tensor<8x1x16xbf16>) -> tensor<8x16xbf16>
      %15 = stablehlo.convert %14 : (tensor<8x16xbf16>) -> tensor<8x16xf32>
      %16 = stablehlo.broadcast_in_dim %iterArg_6, dims = [1] : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x1536xf32>) -> tensor<8x1536xf32>
      %18 = stablehlo.add %9, %17 : tensor<8x1536xf32>
      %19 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1536xf32>
      %20 = stablehlo.maximum %18, %19 : tensor<8x1536xf32>
      %21 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1536xf32>
      %22 = stablehlo.subtract %18, %21 : tensor<8x1536xf32>
      %23 = stablehlo.compare NE, %22, %22, FLOAT : (tensor<8x1536xf32>, tensor<8x1536xf32>) -> tensor<8x1536xi1>
      %24 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1536xf32>
      %25 = stablehlo.add %18, %24 : tensor<8x1536xf32>
      %26 = stablehlo.abs %22 : tensor<8x1536xf32>
      %27 = stablehlo.negate %26 : tensor<8x1536xf32>
      %28 = stablehlo.exponential %27 : tensor<8x1536xf32>
      %29 = stablehlo.log_plus_one %28 : tensor<8x1536xf32>
      %30 = stablehlo.add %20, %29 : tensor<8x1536xf32>
      %31 = stablehlo.select %23, %25, %30 : tensor<8x1536xi1>, tensor<8x1536xf32>
      %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<8x1536xf32>) -> tensor<8x1536x1xf32>
      %33 = stablehlo.broadcast_in_dim %iterArg_7, dims = [1, 2] : (tensor<1536x16xf32>) -> tensor<1x1536x16xf32>
      %34 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<8x1536x1xf32>) -> tensor<8x1536x16xf32>
      %35 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2] : (tensor<1x1536x16xf32>) -> tensor<8x1536x16xf32>
      %36 = stablehlo.multiply %34, %35 : tensor<8x1536x16xf32>
      %37 = stablehlo.exponential %36 : tensor<8x1536x16xf32>
      %38 = stablehlo.multiply %37, %iterArg_11 : tensor<8x1536x16xf32>
      %39 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<8x1536xf32>) -> tensor<8x1536x1xf32>
      %40 = stablehlo.broadcast_in_dim %12, dims = [0, 2] : (tensor<8x16xf32>) -> tensor<8x1x16xf32>
      %41 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<8x1536x1xf32>) -> tensor<8x1536x16xf32>
      %42 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2] : (tensor<8x1x16xf32>) -> tensor<8x1536x16xf32>
      %43 = stablehlo.multiply %41, %42 : tensor<8x1536x16xf32>
      %44 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<8x1536xf32>) -> tensor<8x1536x1xf32>
      %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2] : (tensor<8x1536x1xf32>) -> tensor<8x1536x16xf32>
      %46 = stablehlo.multiply %43, %45 : tensor<8x1536x16xf32>
      %47 = stablehlo.add %38, %46 : tensor<8x1536x16xf32>
      %48 = stablehlo.dot_general %47, %15, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x1536x16xf32>, tensor<8x16xf32>) -> tensor<8x1536xf32>
      %49 = stablehlo.broadcast_in_dim %iterArg_8, dims = [1] : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %50 = stablehlo.broadcast_in_dim %49, dims = [0, 1] : (tensor<1x1536xf32>) -> tensor<8x1536xf32>
      %51 = stablehlo.multiply %50, %6 : tensor<8x1536xf32>
      %52 = stablehlo.add %48, %51 : tensor<8x1536xf32>
      %53 = stablehlo.convert %52 : (tensor<8x1536xf32>) -> tensor<8x1536xbf16>
      %54 = stablehlo.broadcast_in_dim %53, dims = [0, 2] : (tensor<8x1536xbf16>) -> tensor<8x1x1536xbf16>
      %55 = stablehlo.dynamic_update_slice %iterArg_12, %54, %c_1, %iterArg_10, %c_1 : (tensor<8x2048x1536xbf16>, tensor<8x1x1536xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x2048x1536xbf16>
      %56 = stablehlo.add %iterArg_9, %c : tensor<i32>
      stablehlo.return %iterArg, %iterArg_3, %iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %56, %3, %47, %55 : tensor<8x2048x1536xbf16>, tensor<8x2048x1536xbf16>, tensor<8x2048x16xbf16>, tensor<8x2048x16xbf16>, tensor<1536xf32>, tensor<1536x16xf32>, tensor<1536xf32>, tensor<i32>, tensor<i32>, tensor<8x1536x16xf32>, tensor<8x2048x1536xbf16>
    }
    return %2#10 : tensor<8x2048x1536xbf16>
  }
}


// The payload is already inlined, but the recurrence is still StableHLO control flow.
// CLEAN-LABEL: func.func public @main(
// CLEAN-NOT: func.call
// CLEAN: stablehlo.while
// CLEAN: stablehlo.dynamic_slice
// CLEAN: stablehlo.log_plus_one
// CLEAN: stablehlo.dot_general
// CLEAN: stablehlo.dynamic_update_slice
// CLEAN-NOT: func.func private

// HTile converts every Linalg op, bufferizes the worker boundary, and outlines
// the exchanged forall as one persistent kernel with an inner sequential scan.
// CHECK-LABEL: func.func public @main(
// CHECK-NOT: stablehlo.
// CHECK-NOT: linalg.
// CHECK: htile.launch_func @mamba_selective_scan_kernel(
// CHECK-SAME: {program_bounds = array<i64: 8, 12>}
// CHECK: return

// CHECK-LABEL: htile.kernel @mamba_selective_scan_kernel(
// CHECK-SAME: attributes {program_bounds = array<i64: 8, 12>}
// CHECK: %[[B:.+]] = htile.program_id 0
// CHECK: %[[CB:.+]] = htile.program_id 1
// CHECK: %[[C_OFFSET:.+]] = affine.apply {{.*}}[%[[CB]]]
// CHECK: %[[STATE_INIT:.+]] = htile.load %arg7[%[[B]], %[[C_OFFSET]], %c0]
// CHECK-SAME: -> tensor<128x16xf32>
// CHECK: %[[OUTPUT_SLAB:.+]] = htile.load %arg8[%[[B]], %c0, %[[C_OFFSET]]]
// CHECK-SAME: -> tensor<2048x128xbf16>
// CHECK: %[[SCAN:.+]]:2 = scf.for %[[T:[^ ]+]] =
// CHECK-SAME: iter_args(%[[STATE:.+]] = %[[STATE_INIT]], %[[SLAB:.+]] = %[[OUTPUT_SLAB]])
// CHECK-SAME: -> (tensor<128x16xf32>, tensor<2048x128xbf16>)
// CHECK: %[[STATE_3D:.+]] = htile.unsqueeze %[[STATE]]
// CHECK-SAME: tensor<128x16xf32> -> tensor<1x128x16xf32>
// CHECK: %[[SLAB_3D:.+]] = htile.unsqueeze %[[SLAB]]
// CHECK-SAME: tensor<2048x128xbf16> -> tensor<1x2048x128xbf16>
// CHECK: htile.load %arg1[%[[B]], %[[T]], %[[C_OFFSET]]]
// CHECK: htile.load %arg6[%[[C_OFFSET]]]
// CHECK: math.absf
// CHECK: math.exp
// CHECK: math.log1p
// CHECK: htile.load %arg2[%[[C_OFFSET]], %c0]
// CHECK: math.exp
// CHECK: htile.load %arg3[%[[B]], %[[T]], %c0]
// CHECK: htile.load %arg0[%[[B]], %[[T]], %[[C_OFFSET]]]
// CHECK: htile.load %arg4[%[[B]], %[[T]], %c0]
// CHECK: %[[STATE_NEXT:.+]] = htile.squeeze {{.*}} : tensor<1x128x16xf32> -> tensor<128x16xf32>
// CHECK: htile.dot %[[STATE_NEXT]], {{.*}} : tensor<128x16xf32>, tensor<16xf32> -> tensor<128xf32>
// CHECK: htile.load %arg5[%[[C_OFFSET]]]
// CHECK: %[[INSERTED:.+]] = tensor.insert_slice {{.*}} into %[[SLAB_3D]][0, %[[T]], 0]
// CHECK-SAME: [1, 1, 128]
// CHECK: %[[SLAB_NEXT:.+]] = htile.squeeze %[[INSERTED]]
// CHECK-SAME: -> tensor<2048x128xbf16>
// CHECK: scf.yield %[[STATE_NEXT]], %[[SLAB_NEXT]]
// CHECK: htile.store %[[SCAN]]#1, %arg10[%[[B]], %c0, %[[C_OFFSET]]]
// CHECK: htile.return
// CHECK-NOT: stablehlo.
// CHECK-NOT: linalg.
