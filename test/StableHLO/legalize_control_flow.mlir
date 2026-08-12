// RUN: neptune-opt %s --transform-interpreter | FileCheck %s

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

// CHECK-LABEL: func.func @fori_loop(
// CHECK-DAG: %[[LB:.+]] = tensor.extract %arg0[] : tensor<i32>
// CHECK-DAG: %[[UB:.+]] = tensor.extract %arg1[] : tensor<i32>
// CHECK-DAG: %[[STEP:.+]] = tensor.extract %arg2[] : tensor<i32>
// CHECK: %[[RESULT:.+]]:2 = scf.for %[[IV:.+]] = %[[LB]] to %[[UB]] step %[[STEP]]
// CHECK-SAME: iter_args(%{{.+}} = %arg0, %[[ARG:.+]] = %arg3)
// CHECK: %[[TENSOR_IV:.+]] = tensor.from_elements %[[IV]] : tensor<i32>
// CHECK: %[[NEXT:.+]] = stablehlo.add %[[TENSOR_IV]], %arg2 : tensor<i32>
// CHECK: scf.yield %[[NEXT]], %[[ARG]] : tensor<i32>, tensor<4xf32>
// CHECK: return %[[RESULT]]#1 : tensor<4xf32>

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

// CHECK-LABEL: func.func @general_while(
// CHECK: %[[RESULT:.+]] = scf.while
// CHECK: %[[PRED:.+]] = stablehlo.compare LT
// CHECK: %[[SCALAR:.+]] = tensor.extract %[[PRED]][] : tensor<i1>
// CHECK: scf.condition(%[[SCALAR]])
// CHECK: } do {
// CHECK: stablehlo.add
// CHECK: scf.yield
// CHECK: return %[[RESULT]]

func.func @if(%predicate: tensor<i1>, %lhs: tensor<4xf32>,
              %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.if"(%predicate) ({
    "stablehlo.return"(%lhs) : (tensor<4xf32>) -> ()
  }, {
    "stablehlo.return"(%rhs) : (tensor<4xf32>) -> ()
  }) : (tensor<i1>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @if(
// CHECK: %[[PRED:.+]] = tensor.extract %arg0[] : tensor<i1>
// CHECK: %[[RESULT:.+]] = scf.if %[[PRED]] -> (tensor<4xf32>) {
// CHECK: scf.yield %arg1
// CHECK: } else {
// CHECK: scf.yield %arg2
// CHECK: return %[[RESULT]]

func.func @case(%index: tensor<i32>, %first: tensor<4xf32>,
                %default: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.case"(%index) ({
    "stablehlo.return"(%first) : (tensor<4xf32>) -> ()
  }, {
    "stablehlo.return"(%default) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @case(
// CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK: %[[CMP:.+]] = stablehlo.compare EQ, %arg0, %[[ZERO]]
// CHECK: %[[PRED:.+]] = tensor.extract %[[CMP]][] : tensor<i1>
// CHECK: %[[RESULT:.+]] = scf.if %[[PRED]] -> (tensor<4xf32>) {
// CHECK: scf.yield %arg1
// CHECK: } else {
// CHECK: scf.yield %arg2
// CHECK: return %[[RESULT]]
}
