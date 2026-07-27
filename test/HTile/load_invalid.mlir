// RUN: neptune-opt --verify-diagnostics --split-input-file %s

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
