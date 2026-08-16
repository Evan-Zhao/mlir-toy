// RUN: neptune-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @bounded
// CHECK: scf.for %[[IV:.+]] =
// CHECK: %[[INDEX:.+]] = arith.index_cast %[[IV]] : i32 to index
// CHECK-NEXT: func.call @consume(%[[INDEX]])
// CHECK-NOT: arith.maxsi
// CHECK-NOT: arith.minsi

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.stablehlo.simplify_in_bounds_clamps
    } : !transform.any_op
    transform.yield
  }

  func.func private @consume(index)

  func.func @bounded() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c0 = arith.constant 0 : index
    %c2047 = arith.constant 2047 : index
    scf.for %iv = %c0_i32 to %c2048_i32 step %c1_i32 : i32 {
      %index = arith.index_cast %iv : i32 to index
      %lower = arith.maxsi %index, %c0 : index
      %start = arith.minsi %lower, %c2047 : index
      func.call @consume(%start) : (index) -> ()
    }
    return
  }
}
// -----

// CHECK-LABEL: func.func @not_bounded
// CHECK: %[[LOWER:.+]] = arith.maxsi
// CHECK: arith.minsi %[[LOWER]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.stablehlo.simplify_in_bounds_clamps
    } : !transform.any_op
    transform.yield
  }

  func.func private @consume(index)

  func.func @not_bounded() {
    %cm1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2049_i32 = arith.constant 2049 : i32
    %c0 = arith.constant 0 : index
    %c2047 = arith.constant 2047 : index
    scf.for %iv = %cm1_i32 to %c2049_i32 step %c1_i32 : i32 {
      %index = arith.index_cast %iv : i32 to index
      %lower = arith.maxsi %index, %c0 : index
      %start = arith.minsi %lower, %c2047 : index
      func.call @consume(%start) : (index) -> ()
    }
    return
  }
}
// -----

// CHECK-LABEL: func.func @bounded_index_cast_round_trip
// CHECK: scf.for %[[IV:.+]] =
// CHECK-NEXT: func.call @consume(%[[IV]])
// CHECK-NOT: arith.index_cast

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.stablehlo.simplify_in_bounds_clamps
    } : !transform.any_op
    transform.yield
  }

  func.func private @consume(index)

  func.func @bounded_index_cast_round_trip() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2048 = arith.constant 2048 : index
    scf.for %iv = %c0 to %c2048 step %c1 {
      %narrow = arith.index_cast %iv : index to i32
      %widen = arith.index_cast %narrow : i32 to index
      func.call @consume(%widen) : (index) -> ()
    }
    return
  }
}
// -----

// CHECK-LABEL: func.func @unbounded_index_cast_round_trip
// CHECK: %[[NARROW:.+]] = arith.index_cast %{{.+}} : index to i8
// CHECK: %[[WIDEN:.+]] = arith.index_cast %[[NARROW]] : i8 to index
// CHECK: func.call @consume(%[[WIDEN]])

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.stablehlo.simplify_in_bounds_clamps
    } : !transform.any_op
    transform.yield
  }

  func.func private @consume(index)

  func.func @unbounded_index_cast_round_trip() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    scf.for %iv = %c0 to %c256 step %c1 {
      %narrow = arith.index_cast %iv : index to i8
      %widen = arith.index_cast %narrow : i8 to index
      func.call @consume(%widen) : (index) -> ()
    }
    return
  }
}
