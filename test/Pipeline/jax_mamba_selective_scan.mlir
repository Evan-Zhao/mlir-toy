// RUN: neptune-opt %s --inline --canonicalize --cse | FileCheck %s --check-prefix=CLEAN
// RUN: neptune-opt %s --inline --canonicalize --cse --transform-interpreter --canonicalize --cse | FileCheck %s
//
// StableHLO payload from the default output of:
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
// In schedule terms, this requires: inline the exporter helpers; normalize the
// StableHLO while to an scf.for; destination-bufferize the dynamic output
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
    // Skip the StableHLO-to-TA conversion because we have no rewrite to perform in TA.
    %func = transform.apply_registered_pass "stablehlo-legalize-to-linalg" to %funcs : (!any) -> !any
    transform.verify %func : !any
    transform.yield
  }

  func.func public @main(%arg0: tensor<8x2048x1536xbf16>, %arg1: tensor<8x2048x1536xbf16>, %arg2: tensor<1536x16xf32>, %arg3: tensor<8x2048x16xbf16>, %arg4: tensor<8x2048x16xbf16>, %arg5: tensor<1536xf32>, %arg6: tensor<1536xf32>) -> (tensor<8x2048x1536xbf16> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x1536x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<8x2048x1536xbf16>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %2:11 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %arg3, %iterArg_4 = %arg4, %iterArg_5 = %arg6, %iterArg_6 = %arg2, %iterArg_7 = %arg5, %iterArg_8 = %c_1, %iterArg_9 = %c, %iterArg_10 = %0, %iterArg_11 = %1) : tensor<8x2048x1536xbf16>, tensor<8x2048x1536xbf16>, tensor<8x2048x16xbf16>, tensor<8x2048x16xbf16>, tensor<1536xf32>, tensor<1536x16xf32>, tensor<1536xf32>, tensor<i32>, tensor<i32>, tensor<8x1536x16xf32>, tensor<8x2048x1536xbf16>
    cond {
      %c_12 = stablehlo.constant dense<2048> : tensor<i32>
      %3 = stablehlo.compare LT, %iterArg_8, %c_12, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3:3 = func.call @closed_call(%iterArg, %iterArg_2, %iterArg_3, %iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_9, %iterArg_10, %iterArg_11) : (tensor<8x2048x1536xbf16>, tensor<8x2048x1536xbf16>, tensor<8x2048x16xbf16>, tensor<8x2048x16xbf16>, tensor<1536xf32>, tensor<1536x16xf32>, tensor<1536xf32>, tensor<i32>, tensor<8x1536x16xf32>, tensor<8x2048x1536xbf16>) -> (tensor<i32>, tensor<8x1536x16xf32>, tensor<8x2048x1536xbf16>)
      %c_12 = stablehlo.constant dense<1> : tensor<i32>
      %4 = stablehlo.add %iterArg_8, %c_12 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_2, %iterArg_3, %iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %4, %3#0, %3#1, %3#2 : tensor<8x2048x1536xbf16>, tensor<8x2048x1536xbf16>, tensor<8x2048x16xbf16>, tensor<8x2048x16xbf16>, tensor<1536xf32>, tensor<1536x16xf32>, tensor<1536xf32>, tensor<i32>, tensor<i32>, tensor<8x1536x16xf32>, tensor<8x2048x1536xbf16>
    }
    return %2#10 : tensor<8x2048x1536xbf16>
  }
  func.func private @closed_call(%arg0: tensor<8x2048x1536xbf16>, %arg1: tensor<8x2048x1536xbf16>, %arg2: tensor<8x2048x16xbf16>, %arg3: tensor<8x2048x16xbf16>, %arg4: tensor<1536xf32>, %arg5: tensor<1536x16xf32>, %arg6: tensor<1536xf32>, %arg7: tensor<i32>, %arg8: tensor<8x1536x16xf32>, %arg9: tensor<8x2048x1536xbf16>) -> (tensor<i32>, tensor<8x1536x16xf32>, tensor<8x2048x1536xbf16>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg7, %c : tensor<i32>
    %1 = call @dynamic_index_in_dim(%arg0, %arg7) : (tensor<8x2048x1536xbf16>, tensor<i32>) -> tensor<8x1536xbf16>
    %2 = stablehlo.convert %1 : (tensor<8x1536xbf16>) -> tensor<8x1536xf32>
    %3 = call @dynamic_index_in_dim(%arg1, %arg7) : (tensor<8x2048x1536xbf16>, tensor<i32>) -> tensor<8x1536xbf16>
    %4 = stablehlo.convert %3 : (tensor<8x1536xbf16>) -> tensor<8x1536xf32>
    %5 = call @dynamic_index_in_dim_0(%arg2, %arg7) : (tensor<8x2048x16xbf16>, tensor<i32>) -> tensor<8x16xbf16>
    %6 = stablehlo.convert %5 : (tensor<8x16xbf16>) -> tensor<8x16xf32>
    %7 = call @dynamic_index_in_dim_0(%arg3, %arg7) : (tensor<8x2048x16xbf16>, tensor<i32>) -> tensor<8x16xbf16>
    %8 = stablehlo.convert %7 : (tensor<8x16xbf16>) -> tensor<8x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<1x1536xf32>) -> tensor<8x1536xf32>
    %11 = stablehlo.add %4, %10 : tensor<8x1536xf32>
    %12 = call @softplus(%11) : (tensor<8x1536xf32>) -> tensor<8x1536xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<8x1536xf32>) -> tensor<8x1536x1xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [1, 2] : (tensor<1536x16xf32>) -> tensor<1x1536x16xf32>
    %15 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<8x1536x1xf32>) -> tensor<8x1536x16xf32>
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2] : (tensor<1x1536x16xf32>) -> tensor<8x1536x16xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<8x1536x16xf32>
    %18 = stablehlo.exponential %17 : tensor<8x1536x16xf32>
    %19 = stablehlo.multiply %18, %arg8 : tensor<8x1536x16xf32>
    %20 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<8x1536xf32>) -> tensor<8x1536x1xf32>
    %21 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<8x16xf32>) -> tensor<8x1x16xf32>
    %22 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<8x1536x1xf32>) -> tensor<8x1536x16xf32>
    %23 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<8x1x16xf32>) -> tensor<8x1536x16xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<8x1536x16xf32>
    %25 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<8x1536xf32>) -> tensor<8x1536x1xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<8x1536x1xf32>) -> tensor<8x1536x16xf32>
    %27 = stablehlo.multiply %24, %26 : tensor<8x1536x16xf32>
    %28 = stablehlo.add %19, %27 : tensor<8x1536x16xf32>
    %29 = stablehlo.dot_general %28, %8, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x1536x16xf32>, tensor<8x16xf32>) -> tensor<8x1536xf32>
    %30 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1] : (tensor<1x1536xf32>) -> tensor<8x1536xf32>
    %32 = stablehlo.multiply %31, %2 : tensor<8x1536xf32>
    %33 = stablehlo.add %29, %32 : tensor<8x1536xf32>
    %34 = stablehlo.convert %33 : (tensor<8x1536xf32>) -> tensor<8x1536xbf16>
    %35 = call @dynamic_update_index_in_dim(%arg9, %34, %arg7) : (tensor<8x2048x1536xbf16>, tensor<8x1536xbf16>, tensor<i32>) -> tensor<8x2048x1536xbf16>
    return %0, %28, %35 : tensor<i32>, tensor<8x1536x16xf32>, tensor<8x2048x1536xbf16>
  }
  func.func private @dynamic_index_in_dim(%arg0: tensor<8x2048x1536xbf16>, %arg1: tensor<i32>) -> tensor<8x1536xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %c, %arg1, %c_0, sizes = [8, 1, 1536] : (tensor<8x2048x1536xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x1x1536xbf16>
    %1 = stablehlo.reshape %0 : (tensor<8x1x1536xbf16>) -> tensor<8x1536xbf16>
    return %1 : tensor<8x1536xbf16>
  }
  func.func private @dynamic_index_in_dim_0(%arg0: tensor<8x2048x16xbf16>, %arg1: tensor<i32>) -> tensor<8x16xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %c, %arg1, %c_0, sizes = [8, 1, 16] : (tensor<8x2048x16xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x1x16xbf16>
    %1 = stablehlo.reshape %0 : (tensor<8x1x16xbf16>) -> tensor<8x16xbf16>
    return %1 : tensor<8x16xbf16>
  }
  func.func private @softplus(%arg0: tensor<8x1536xf32>) -> tensor<8x1536xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x1536xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<8x1536xf32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x1536xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x1536xf32>
    %4 = stablehlo.compare NE, %3, %3, FLOAT : (tensor<8x1536xf32>, tensor<8x1536xf32>) -> tensor<8x1536xi1>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x1536xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<8x1536xf32>
    %7 = stablehlo.abs %3 : tensor<8x1536xf32>
    %8 = stablehlo.negate %7 : tensor<8x1536xf32>
    %9 = stablehlo.exponential %8 : tensor<8x1536xf32>
    %10 = stablehlo.log_plus_one %9 : tensor<8x1536xf32>
    %11 = stablehlo.add %1, %10 : tensor<8x1536xf32>
    %12 = stablehlo.select %4, %6, %11 : tensor<8x1536xi1>, tensor<8x1536xf32>
    return %12 : tensor<8x1536xf32>
  }
  func.func private @dynamic_update_index_in_dim(%arg0: tensor<8x2048x1536xbf16>, %arg1: tensor<8x1536xbf16>, %arg2: tensor<i32>) -> tensor<8x2048x1536xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<8x1536xbf16>) -> tensor<8x1x1536xbf16>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %c, %arg2, %c_0 : (tensor<8x2048x1536xbf16>, tensor<8x1x1536xbf16>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x2048x1536xbf16>
    return %1 : tensor<8x2048x1536xbf16>
  }
}


// Helper calls inline cleanly, but the recurrence is still StableHLO control flow.
// CLEAN-LABEL: func.func public @main(
// CLEAN-NOT: func.call
// CLEAN: stablehlo.while
// CLEAN: stablehlo.dynamic_slice
// CLEAN: stablehlo.log_plus_one
// CLEAN: stablehlo.dot_general
// CLEAN: stablehlo.dynamic_update_slice
// CLEAN-NOT: func.func private

// JAX's counted while is recognized as a for loop. Canonicalization drops the
// invariant loop operands and leaves the state, output, and JAX's second synchronized index.
// CHECK-LABEL: func.func public @main(
// CHECK-NOT: stablehlo.while
// CHECK: scf.for
// CHECK-SAME: iter_args(
// CHECK: tensor.extract_slice
// CHECK: math.absf
// CHECK: math.exp
// CHECK: math.log1p
// CHECK: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: tensor.insert_slice
// CHECK: scf.yield
// CHECK-NOT: stablehlo.
