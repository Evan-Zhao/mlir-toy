// RUN: neptune-opt --verify-diagnostics --split-input-file %s | FileCheck %s

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

// -----

module {
  // CHECK-LABEL: func.func @masked_parallel_insert_slice
  func.func @masked_parallel_insert_slice(
      %source: tensor<4x1x8xf32>, %mask: tensor<4x1x8xi1>,
      %init: tensor<16x2x8xf32>) -> tensor<16x2x8xf32> {
    %c0 = arith.constant 0 : index
    %c15 = arith.constant 15 : index
    %result = scf.forall (%head) in (2) shared_outs(%out = %init)
        -> tensor<16x2x8xf32> {
      scf.forall.in_parallel {
        // CHECK: htile.masked_parallel_insert_slice %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] [4, 1, 8] [1, 1, 1] mask(%{{.*}} : tensor<4x1x8xi1>)
        htile.masked_parallel_insert_slice
            %source into %out[%c15, %head, %c0] [4, 1, 8] [1, 1, 1]
            mask(%mask : tensor<4x1x8xi1>)
            : tensor<4x1x8xf32> into tensor<16x2x8xf32>
      }
    }
    return %result : tensor<16x2x8xf32>
  }

  // CHECK-LABEL: func.func @rank_reduced_masked_parallel_insert_slice
  func.func @rank_reduced_masked_parallel_insert_slice(
      %source: tensor<4x8xf32>, %mask: tensor<4x8xi1>,
      %init: tensor<16x1x8xf32>) -> tensor<16x1x8xf32> {
    %result = scf.forall (%i) in (1) shared_outs(%out = %init)
        -> tensor<16x1x8xf32> {
      scf.forall.in_parallel {
        // CHECK: htile.masked_parallel_insert_slice %{{.*}} into %{{.*}}[15, 0, 0] [4, 1, 8] [1, 1, 1] mask(%{{.*}} : tensor<4x8xi1>)
        htile.masked_parallel_insert_slice
            %source into %out[15, 0, 0] [4, 1, 8] [1, 1, 1]
            mask(%mask : tensor<4x8xi1>)
            : tensor<4x8xf32> into tensor<16x1x8xf32>
      }
    }
    return %result : tensor<16x1x8xf32>
  }
}

// -----

module {
  func.func @wrong_insert_mask_shape(
      %source: tensor<4x1x8xf32>, %mask: tensor<2x1x8xi1>,
      %init: tensor<16x2x8xf32>) -> tensor<16x2x8xf32> {
    %result = scf.forall (%head) in (2) shared_outs(%out = %init)
        -> tensor<16x2x8xf32> {
      scf.forall.in_parallel {
        // expected-error@+1 {{requires mask shape to match source shape}}
        htile.masked_parallel_insert_slice
            %source into %out[0, %head, 0] [4, 1, 8] [1, 1, 1]
            mask(%mask : tensor<2x1x8xi1>)
            : tensor<4x1x8xf32> into tensor<16x2x8xf32>
      }
    }
    return %result : tensor<16x2x8xf32>
  }
}

// -----

module {
  func.func @wrong_insert_mask_element_type(
      %source: tensor<4x1x8xf32>, %mask: tensor<4x1x8xi8>,
      %init: tensor<16x2x8xf32>) -> tensor<16x2x8xf32> {
    %result = scf.forall (%head) in (2) shared_outs(%out = %init)
        -> tensor<16x2x8xf32> {
      scf.forall.in_parallel {
        // expected-error@+1 {{requires mask to have i1 element type}}
        htile.masked_parallel_insert_slice
            %source into %out[0, %head, 0] [4, 1, 8] [1, 1, 1]
            mask(%mask : tensor<4x1x8xi8>)
            : tensor<4x1x8xf32> into tensor<16x2x8xf32>
      }
    }
    return %result : tensor<16x2x8xf32>
  }
}

// -----

module {
  func.func @wrong_insert_source_shape(
      %source: tensor<4x2x8xf32>, %mask: tensor<4x2x8xi1>,
      %init: tensor<16x2x8xf32>) -> tensor<16x2x8xf32> {
    %result = scf.forall (%head) in (2) shared_outs(%out = %init)
        -> tensor<16x2x8xf32> {
      scf.forall.in_parallel {
        // expected-error@+1 {{requires source shape to match slice sizes after dropping static unit dimensions}}
        htile.masked_parallel_insert_slice
            %source into %out[0, %head, 0] [4, 1, 8] [1, 1, 1]
            mask(%mask : tensor<4x2x8xi1>)
            : tensor<4x2x8xf32> into tensor<16x2x8xf32>
      }
    }
    return %result : tensor<16x2x8xf32>
  }
}

// -----

module {
  func.func @missing_in_parallel_parent(
      %source: tensor<4xf32>, %mask: tensor<4xi1>, %dest: tensor<16xf32>) {
    // expected-error@+1 {{must be directly nested in an in-parallel operation}}
    htile.masked_parallel_insert_slice %source into %dest[0] [4] [1]
        mask(%mask : tensor<4xi1>)
        : tensor<4xf32> into tensor<16xf32>
    return
  }
}

// -----

module {
  func.func @missing_mask(%source: tensor<16x8xf32>, %other: f32) {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{requires mask and other to be supplied together}}
    %0 = "htile.load"(%source, %c0, %c0, %other) <{
      operandSegmentSizes = array<i32: 1, 2, 0, 1>
    }> : (tensor<16x8xf32>, index, index, f32) -> tensor<4x8xf32>
    return
  }
}

// -----

module {
  func.func @wrong_mask_shape(%source: tensor<16x8xf32>,
                              %mask: tensor<2x8xi1>, %other: f32) {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{requires mask shape to match result shape}}
    %0 = htile.load %source[%c0, %c0]
        mask(%mask : tensor<2x8xi1>) other(%other : f32)
        : tensor<16x8xf32> -> tensor<4x8xf32>
    return
  }
}

// -----

module {
  func.func @wrong_mask_element_type(%source: tensor<16x8xf32>,
                                     %mask: tensor<4x8xi8>, %other: f32) {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{requires mask to have i1 element type}}
    %0 = htile.load %source[%c0, %c0]
        mask(%mask : tensor<4x8xi8>) other(%other : f32)
        : tensor<16x8xf32> -> tensor<4x8xf32>
    return
  }
}

// -----

module {
  func.func @wrong_other_type(%source: tensor<16x8xf32>,
                              %mask: tensor<4x8xi1>, %other: i32) {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{requires other type to match the result element type}}
    %0 = htile.load %source[%c0, %c0]
        mask(%mask : tensor<4x8xi1>) other(%other : i32)
        : tensor<16x8xf32> -> tensor<4x8xf32>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func.func @unmasked_store
  func.func @unmasked_store(%value: tensor<4x8xf32>, %dest: memref<16x8xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: htile.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : tensor<4x8xf32>, memref<16x8xf32>
    htile.store %value, %dest[%c0, %c0]
        : tensor<4x8xf32>, memref<16x8xf32>
    return
  }

  // CHECK-LABEL: func.func @masked_store
  func.func @masked_store(%value: tensor<4x8xf32>, %dest: memref<16x8xf32>,
                          %mask: tensor<4x8xi1>) {
    %c0 = arith.constant 0 : index
    // CHECK: htile.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] mask(%{{.*}} : tensor<4x8xi1>) : tensor<4x8xf32>, memref<16x8xf32>
    htile.store %value, %dest[%c0, %c0]
        mask(%mask : tensor<4x8xi1>)
        : tensor<4x8xf32>, memref<16x8xf32>
    return
  }
}

// -----

module {
  func.func @wrong_store_mask_shape(
      %value: tensor<4x8xf32>, %dest: memref<16x8xf32>,
      %mask: tensor<2x8xi1>) {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{requires mask shape to match value shape}}
    htile.store %value, %dest[%c0, %c0]
        mask(%mask : tensor<2x8xi1>)
        : tensor<4x8xf32>, memref<16x8xf32>
    return
  }
}

// -----

module {
  func.func @wrong_store_mask_element_type(
      %value: tensor<4x8xf32>, %dest: memref<16x8xf32>,
      %mask: tensor<4x8xi8>) {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{requires mask to have i1 element type}}
    htile.store %value, %dest[%c0, %c0]
        mask(%mask : tensor<4x8xi8>)
        : tensor<4x8xf32>, memref<16x8xf32>
    return
  }
}
