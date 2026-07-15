// RUN: neptune-opt --pass-pipeline='builtin.module(htile-dot-transpose-to-load-order,cse,canonicalize)' %s | FileCheck %s

#shared = #htile.encoding<placement = shared>
#local = #htile.encoding<placement = local>

module {
  // CHECK-LABEL: func.func @transpose_b_from_load(
  func.func @transpose_b_from_load(
      %lhs_mem: memref<4x8xf16>,
      %rhs_mem: memref<2x8xf16>,
      %out_mem: memref<4x2xf32>) {
    %c0 = arith.constant 0 : index

    // CHECK-DAG: %[[LHS:.+]] = htile.load %arg0[%{{.*}}, %{{.*}}] : memref<4x8xf16> -> tensor<4x8xf16, #htile.encoding<placement = shared>>
    %lhs = htile.load %lhs_mem[%c0, %c0]
        : memref<4x8xf16> -> tensor<4x8xf16, #shared>

    // CHECK-DAG: %[[RHS:.+]] = htile.load %arg1[%{{.*}}, %{{.*}}] {dimension_order = array<i64: 1, 0>} : memref<2x8xf16> -> tensor<8x2xf16, #htile.encoding<placement = shared>>
    %rhs = htile.load %rhs_mem[%c0, %c0]
        : memref<2x8xf16> -> tensor<2x8xf16, #shared>

    // CHECK: htile.dot %[[LHS]], %[[RHS]] {warp_policy = "full_row"} : tensor<4x8xf16, #htile.encoding<placement = shared>>, tensor<8x2xf16, #htile.encoding<placement = shared>> -> tensor<4x2xf32, #htile.encoding<placement = local>>
    // CHECK-NOT: transpose_b
    // CHECK-NOT: htile.permute
    %acc = htile.dot %lhs, %rhs {transpose_b, warp_policy = "full_row"}
        : tensor<4x8xf16, #shared>, tensor<2x8xf16, #shared>
        -> tensor<4x2xf32, #local>

    htile.store %acc, %out_mem[%c0, %c0]
        : tensor<4x2xf32, #local>, memref<4x2xf32>
    return
  }
}
