// RUN: mlir-opt --load-dialect-plugin=%neptune_htile_plugin %s | FileCheck %s

module {
  %c1 = arith.constant 1 : index
  htile.launch_func @decode_partial(%c1) {grid = array<i64: 4, 16, 1>} : index

  htile.kernel @decode_partial(%arg0: memref<4x16xf32>) {
    htile.return
  }
}

// CHECK: htile.launch_func @decode_partial(%{{.*}}) {grid = array<i64: 4, 16, 1>} : index
// CHECK: htile.kernel @decode_partial(%{{.*}}: memref<4x16xf32>) {
// CHECK: htile.return
