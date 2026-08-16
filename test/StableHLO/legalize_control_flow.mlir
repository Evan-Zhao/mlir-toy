// RUN: neptune-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fori_loop(
// CHECK-DAG: %[[LB_I32:.+]] = tensor.extract %arg0[] : tensor<i32>
// CHECK-DAG: %[[LB:.+]] = arith.index_cast %[[LB_I32]] : i32 to index
// CHECK-DAG: %[[UB_I32:.+]] = tensor.extract %arg1[] : tensor<i32>
// CHECK-DAG: %[[UB:.+]] = arith.index_cast %[[UB_I32]] : i32 to index
// CHECK-DAG: %[[STEP_I32:.+]] = tensor.extract %arg2[] : tensor<i32>
// CHECK-DAG: %[[STEP:.+]] = arith.index_cast %[[STEP_I32]] : i32 to index
// CHECK: %[[RESULT:.+]] = scf.for %[[IV:.+]] = %[[LB]] to %[[UB]] step %[[STEP]]
// CHECK-SAME: iter_args(%[[ARG:.+]] = %arg3)
// CHECK: %[[IV_I32:.+]] = arith.index_cast %[[IV]] : index to i32
// CHECK: %[[TENSOR_IV:.+]] = tensor.from_elements %[[IV_I32]] : tensor<i32>
// CHECK-NOT: stablehlo.add %[[TENSOR_IV]], %arg2
// CHECK: scf.yield %[[ARG]] : tensor<4xf32>
// CHECK: return %[[RESULT]] : tensor<4xf32>
!any = !transform.any_op
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.stablehlo.legalize_control_flow %funcs : !any
    transform.verify %funcs : !any
    transform.yield
  }

  func.func @fori_loop(%lb: tensor<i32>, %ub: tensor<i32>, %step: tensor<i32>,
                     %value: tensor<4xf32>) -> tensor<4xf32> {
  %0:2 = stablehlo.while(%i = %lb, %arg = %value) : tensor<i32>, tensor<4xf32> cond {
    %pred = stablehlo.compare LT, %i, %ub : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %pred : tensor<i1>
  } do {
    %next = stablehlo.add %i, %step : tensor<i32>
    stablehlo.return %next, %arg : tensor<i32>, tensor<4xf32>
  }
  return %0#1 : tensor<4xf32>
}
}
// -----

// CHECK-LABEL: func.func @fori_loop_synchronized_index(
// CHECK: %[[RESULT:.+]] = scf.for %[[IV:[^ ]+]]
// CHECK-SAME: iter_args(%{{.+}} = %arg0)
// CHECK: %[[IV_I32:.+]] = arith.index_cast %[[IV]] : index to i32
// CHECK: %[[TENSOR_IV:.+]] = tensor.from_elements %[[IV_I32]] : tensor<i32>
// CHECK-NOT: stablehlo.add
// CHECK: scf.yield %[[TENSOR_IV]] : tensor<i32>
// CHECK: return %[[RESULT]] : tensor<i32>

!any = !transform.any_op
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.stablehlo.legalize_control_flow %funcs : !any
    transform.verify %funcs : !any
    transform.yield
  }

  func.func @fori_loop_synchronized_index(%lb: tensor<i32>, %ub: tensor<i32>,
                                         %step: tensor<i32>) -> tensor<i32> {
  %0:4 = stablehlo.while(%i = %lb, %j = %lb, %k = %lb, %last = %lb)
      : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32> cond {
    %pred = stablehlo.compare LT, %i, %ub : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %pred : tensor<i1>
  } do {
    %next_i = stablehlo.add %i, %step : tensor<i32>
    %next_j = stablehlo.add %j, %step : tensor<i32>
    %next_k = stablehlo.add %k, %step : tensor<i32>
    stablehlo.return %next_i, %next_j, %next_k, %j
        : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
  }
  return %0#3 : tensor<i32>
}
}
// -----

// CHECK-LABEL: func.func @fori_loop_used_index(
// CHECK: %[[RESULT:.+]]:2 = scf.for %[[IV:[^ ]+]]
// CHECK-SAME: iter_args(%{{.+}} = %arg0, %[[ARG:.+]] = %arg3)
// CHECK: %[[IV_I32:.+]] = arith.index_cast %[[IV]] : index to i32
// CHECK: %[[TENSOR_IV:.+]] = tensor.from_elements %[[IV_I32]] : tensor<i32>
// CHECK: %[[NEXT:.+]] = stablehlo.add %[[TENSOR_IV]], %arg2 : tensor<i32>
// CHECK: scf.yield %[[NEXT]], %[[ARG]] : tensor<i32>, tensor<4xf32>
// CHECK: return %[[RESULT]]#0, %[[RESULT]]#1

!any = !transform.any_op
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.stablehlo.legalize_control_flow %funcs : !any
    transform.verify %funcs : !any
    transform.yield
  }

  func.func @fori_loop_used_index(%lb: tensor<i32>, %ub: tensor<i32>, %step: tensor<i32>,
                                %value: tensor<4xf32>)
    -> (tensor<i32>, tensor<4xf32>) {
  %0:2 = stablehlo.while(%i = %lb, %arg = %value) : tensor<i32>, tensor<4xf32> cond {
    %pred = stablehlo.compare LT, %i, %ub : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %pred : tensor<i1>
  } do {
    %next = stablehlo.add %i, %step : tensor<i32>
    stablehlo.return %next, %arg : tensor<i32>, tensor<4xf32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<4xf32>
}
}
// -----

// CHECK-LABEL: func.func @general_while(
// CHECK: %[[RESULT:.+]] = scf.while
// CHECK: %[[PRED:.+]] = stablehlo.compare LT
// CHECK: %[[SCALAR:.+]] = tensor.extract %[[PRED]][] : tensor<i1>
// CHECK: scf.condition(%[[SCALAR]])
// CHECK: } do {
// CHECK: stablehlo.add
// CHECK: scf.yield
// CHECK: return %[[RESULT]]

!any = !transform.any_op
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.stablehlo.legalize_control_flow %funcs : !any
    transform.verify %funcs : !any
    transform.yield
  }

  func.func @general_while(%value: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.while(%arg = %value) : tensor<i32> cond {
    %pred = stablehlo.compare LT, %arg, %arg : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %pred : tensor<i1>
  } do {
    %next = stablehlo.add %arg, %arg : tensor<i32>
    stablehlo.return %next : tensor<i32>
  }
  return %0 : tensor<i32>
}
}
// -----

// CHECK-LABEL: func.func @if(
// CHECK: %[[PRED:.+]] = tensor.extract %arg0[] : tensor<i1>
// CHECK: %[[RESULT:.+]] = scf.if %[[PRED]] -> (tensor<4xf32>) {
// CHECK: scf.yield %arg1
// CHECK: } else {
// CHECK: scf.yield %arg2
// CHECK: return %[[RESULT]]

!any = !transform.any_op
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.stablehlo.legalize_control_flow %funcs : !any
    transform.verify %funcs : !any
    transform.yield
  }

  func.func @if(%predicate: tensor<i1>, %lhs: tensor<4xf32>,
              %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.if"(%predicate) ({
    "stablehlo.return"(%lhs) : (tensor<4xf32>) -> ()
  }, {
    "stablehlo.return"(%rhs) : (tensor<4xf32>) -> ()
  }) : (tensor<i1>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
}
// -----

// CHECK-LABEL: func.func @case(
// CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK: %[[CMP:.+]] = stablehlo.compare EQ, %arg0, %[[ZERO]]
// CHECK: %[[PRED:.+]] = tensor.extract %[[CMP]][] : tensor<i1>
// CHECK: %[[RESULT:.+]] = scf.if %[[PRED]] -> (tensor<4xf32>) {
// CHECK: scf.yield %arg1
// CHECK: } else {
// CHECK: scf.yield %arg2
// CHECK: return %[[RESULT]]

!any = !transform.any_op
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !any) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    transform.stablehlo.legalize_control_flow %funcs : !any
    transform.verify %funcs : !any
    transform.yield
  }

  func.func @case(%index: tensor<i32>, %first: tensor<4xf32>,
                %default: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.case"(%index) ({
    "stablehlo.return"(%first) : (tensor<4xf32>) -> ()
  }, {
    "stablehlo.return"(%default) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
}
