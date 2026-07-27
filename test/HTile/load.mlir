// RUN: neptune-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @unmasked_load
  func.func @unmasked_load(%source: tensor<16x8xf32>) -> tensor<4x8xf32> {
    %c0 = arith.constant 0 : index
    // CHECK: htile.load %{{.*}}[%{{.*}}, %{{.*}}] : tensor<16x8xf32> -> tensor<4x8xf32>
    %result = htile.load %source[%c0, %c0]
        : tensor<16x8xf32> -> tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func @masked_load
  func.func @masked_load(%source: tensor<16x8xf32>,
                         %mask: tensor<4x8xi1>, %other: f32)
      -> tensor<4x8xf32> {
    %c0 = arith.constant 0 : index
    // CHECK: htile.load %{{.*}}[%{{.*}}, %{{.*}}] mask(%{{.*}} : tensor<4x8xi1>) other(%{{.*}} : f32) : tensor<16x8xf32> -> tensor<4x8xf32>
    %result = htile.load %source[%c0, %c0]
        mask(%mask : tensor<4x8xi1>) other(%other : f32)
        : tensor<16x8xf32> -> tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }
}
