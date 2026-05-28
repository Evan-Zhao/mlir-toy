// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %attn_func = transform.structured.match ops{["func.func"]} attributes {case_attn} in %module
        : (!transform.any_op) -> !transform.any_op
    %attn_target = transform.structured.match ops{["linalg.generic"]} attributes {target} in %attn_func
        : (!transform.any_op) -> !transform.any_op
    %attn_reduction, %attn_division = transform.linalg.exchange_div_and_matmul %attn_target
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %gqa_func = transform.structured.match ops{["func.func"]} attributes {case_gqa} in %module
        : (!transform.any_op) -> !transform.any_op
    %gqa_target = transform.structured.match ops{["linalg.generic"]} attributes {target} in %gqa_func
        : (!transform.any_op) -> !transform.any_op
    %gqa_reduction, %gqa_division = transform.linalg.exchange_div_and_matmul %gqa_target
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  // CHECK-DAG: #[[ATTN_DEN:.+]] = affine_map<(d0, d1) -> (d0)>
  // CHECK-DAG: #[[GQA_DEN:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
  // CHECK-LABEL: func.func @attention(
  // CHECK: %[[RED:.+]] = linalg.generic
  // CHECK-SAME: ins(%arg0, %arg2
  // CHECK: arith.mulf
  // CHECK-NOT: arith.divf
  // CHECK: } -> tensor<2x4xf32>
  // CHECK: %[[DIV:.+]] = linalg.generic
  // CHECK-SAME: #[[ATTN_DEN]]
  // CHECK-SAME: ins(%[[RED]], %arg1
  // CHECK: arith.divf
  // CHECK: return %[[DIV]]
  func.func @attention(%arg0: tensor<2x3xf32>, %arg1: tensor<2xf32>,
                       %arg2: tensor<3x4xf32>) -> tensor<2x4xf32> attributes {case_attn} {
    %zero = arith.constant 0.0 : f32
    %norm_empty = tensor.empty() : tensor<2x3xf32>
    %normalized = linalg.generic {
        indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i)>,
          affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2xf32>)
        outs(%norm_empty : tensor<2x3xf32>) {
      ^bb0(%num: f32, %den: f32, %out: f32):
        %v = arith.divf %num, %den : f32
        linalg.yield %v : f32
    } -> tensor<2x3xf32>

    %out_empty = tensor.empty() : tensor<2x4xf32>
    %out_init = linalg.fill ins(%zero : f32)
        outs(%out_empty : tensor<2x4xf32>) -> tensor<2x4xf32>
    %out = linalg.generic {target,
        indexing_maps = [
          affine_map<(i, d, j) -> (i, j)>,
          affine_map<(i, d, j) -> (j, d)>,
          affine_map<(i, d, j) -> (i, d)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%normalized, %arg2 : tensor<2x3xf32>, tensor<3x4xf32>)
        outs(%out_init : tensor<2x4xf32>) {
      ^bb0(%prob: f32, %value: f32, %acc: f32):
        %prod = arith.mulf %prob, %value : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<2x4xf32>

    return %out : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @gqa(
  // CHECK: %[[GQA_RED:.+]] = linalg.generic
  // CHECK-SAME: ins(%arg0, %arg2
  // CHECK: } -> tensor<1x2x2x3x4xf32>
  // CHECK: %[[GQA_DIV:.+]] = linalg.generic
  // CHECK-SAME: #[[GQA_DEN]]
  // CHECK-SAME: ins(%[[GQA_RED]], %arg1
  // CHECK: arith.divf
  // CHECK: return %[[GQA_DIV]]
  func.func @gqa(%arg0: tensor<1x2x2x3x3xf32>, %arg1: tensor<1x2x2x3xf32>,
                 %arg2: tensor<1x2x2x3x4xf32>) -> tensor<1x2x2x3x4xf32>
      attributes {case_gqa} {
    %zero = arith.constant 0.0 : f32
    %norm_empty = tensor.empty() : tensor<1x2x2x3x3xf32>
    %normalized = linalg.generic {
        indexing_maps = [
          affine_map<(b, g, h, i, j) -> (b, g, h, i, j)>,
          affine_map<(b, g, h, i, j) -> (b, g, h, i)>,
          affine_map<(b, g, h, i, j) -> (b, g, h, i, j)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<1x2x2x3x3xf32>, tensor<1x2x2x3xf32>)
        outs(%norm_empty : tensor<1x2x2x3x3xf32>) {
      ^bb0(%num: f32, %den: f32, %out: f32):
        %v = arith.divf %num, %den : f32
        linalg.yield %v : f32
    } -> tensor<1x2x2x3x3xf32>

    %out_empty = tensor.empty() : tensor<1x2x2x3x4xf32>
    %out_init = linalg.fill ins(%zero : f32)
        outs(%out_empty : tensor<1x2x2x3x4xf32>) -> tensor<1x2x2x3x4xf32>
    %out = linalg.generic {target,
        indexing_maps = [
          affine_map<(b, g, h, i, d, j) -> (b, g, h, i, j)>,
          affine_map<(b, g, h, i, d, j) -> (b, g, h, j, d)>,
          affine_map<(b, g, h, i, d, j) -> (b, g, h, i, d)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel",
                          "reduction"]}
        ins(%normalized, %arg2 : tensor<1x2x2x3x3xf32>, tensor<1x2x2x3x4xf32>)
        outs(%out_init : tensor<1x2x2x3x4xf32>) {
      ^bb0(%prob: f32, %value: f32, %acc: f32):
        %prod = arith.mulf %prob, %value : f32
        %sum = arith.addf %acc, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<1x2x2x3x4xf32>

    return %out : tensor<1x2x2x3x4xf32>
  }
}
