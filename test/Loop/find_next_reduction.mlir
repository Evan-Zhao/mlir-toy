// RUN: neptune-opt %s --transform-interpreter --split-input-file 2>&1 | FileCheck %s

// @linear_chain: loop -> scale(mulf) -> sum(addf). One elemwise, one reduction.
// Reduction result: addf with reduction iterator.
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.addf
// Elemwise result: scale op (mulf, all-parallel).
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "parallel"]
// CHECK: arith.mulf

// Payload functions are structured to also serve as inputs for
// rolling_update_elemwise_chain tests.

// loop -> scale -> sum

// loop -> max (no elemwise; reduction at BFS distance 1)

// loop -> scale(mulf) -> exp(math.exp) -> product(mulf reduction)

// loop fans out to:
//   near_max  (maxnumf reduction, BFS distance 1 from loop)
//   far_sum   (addf reduction, BFS distance 2 via an intermediate scale)
// BFS must find near_max first.

  // distance 1: directly consumes %scores

  // distance 2: %scores -> scale -> far_sum

  // Combine both to keep %far_sum live.

// loop -> red1(max) -> elem1(exp_shift, consumes loop+red1)
//      -> red2(sum) -> elem2(norm, consumes elem1+red2) -> red3(weighted sum)
// red1 is at BFS distance 1; elem1 is also at distance 1 but is not a reduction.

  // red1: row max

  // elem1: exp(scores - row_max), broadcasts row_max along j

  // red2: row sum of exp

  // elem2: normalize by row sum

  // red3: weighted sum

// loop -> broadcasted scale with a constant-zero indexing result -> sum


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "linear_chain"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %reduction, %elemwise = transform.fusion.find_next_reduction %loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %reduction : !transform.any_op
    transform.print %elemwise : !transform.any_op
    transform.yield
  }

  func.func @linear_chain(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 2.0 : f32

  %empty = tensor.empty() : tensor<4x4xf32>
  %scores = scf.for %iv = %c0 to %c2 step %c1
      iter_args(%acc = %empty) -> (tensor<4x4xf32>) {
    %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%iv)
    %tile = tensor.extract_slice %arg0[0, %j] [4, 2] [1, 1]
        : tensor<4x4xf32> to tensor<4x2xf32>
    %inserted = tensor.insert_slice %tile into %acc[0, %j] [4, 2] [1, 1]
        : tensor<4x2xf32> into tensor<4x4xf32>
    scf.yield %inserted : tensor<4x4xf32>
  }

  %scale_empty = tensor.empty() : tensor<4x4xf32>
  %scaled = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%scores : tensor<4x4xf32>) outs(%scale_empty : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %cst : f32
      linalg.yield %v : f32
  } -> tensor<4x4xf32>

  %zero = arith.constant 0.0 : f32
  %sum_empty = tensor.empty() : tensor<4xf32>
  %sum_init = linalg.fill ins(%zero : f32) outs(%sum_empty : tensor<4xf32>) -> tensor<4xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%scaled : tensor<4x4xf32>) outs(%sum_init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.addf %in, %out : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %sum : tensor<4xf32>
}
}
// -----

// @direct_reduce: loop -> max(maxnumf). Reduction at BFS distance 1, no elemwise.
// Reduction result: maxnumf with reduction iterator.
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.maxnumf
// Elemwise result: empty — no checks needed.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "direct_reduce"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %reduction, %elemwise = transform.fusion.find_next_reduction %loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %reduction : !transform.any_op
    transform.print %elemwise : !transform.any_op
    transform.yield
  }

  func.func @direct_reduce(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %empty = tensor.empty() : tensor<4x4xf32>
  %scores = scf.for %iv = %c0 to %c2 step %c1
      iter_args(%acc = %empty) -> (tensor<4x4xf32>) {
    %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%iv)
    %tile = tensor.extract_slice %arg0[0, %j] [4, 2] [1, 1]
        : tensor<4x4xf32> to tensor<4x2xf32>
    %inserted = tensor.insert_slice %tile into %acc[0, %j] [4, 2] [1, 1]
        : tensor<4x2xf32> into tensor<4x4xf32>
    scf.yield %inserted : tensor<4x4xf32>
  }

  %neg_inf = arith.constant -3.40282347E+38 : f32
  %max_empty = tensor.empty() : tensor<4xf32>
  %max_init = linalg.fill ins(%neg_inf : f32) outs(%max_empty : tensor<4xf32>) -> tensor<4xf32>
  %max = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%scores : tensor<4x4xf32>) outs(%max_init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maxnumf %in, %out : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %max : tensor<4xf32>
}
}
// -----

// @two_elemwise: loop -> scale(mulf) -> exp -> product(mulf). Two elemwise hops.
// Reduction result: product op (mulf, with reduction iterator).
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.mulf
// Elemwise result: scale (mulf) then exp, in def-use order.
// CHECK: IR printer
// CHECK: arith.mulf
// CHECK: math.exp

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "two_elemwise"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %reduction, %elemwise = transform.fusion.find_next_reduction %loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %reduction : !transform.any_op
    transform.print %elemwise : !transform.any_op
    transform.yield
  }

  func.func @two_elemwise(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.5 : f32

  %empty = tensor.empty() : tensor<4x4xf32>
  %scores = scf.for %iv = %c0 to %c2 step %c1
      iter_args(%acc = %empty) -> (tensor<4x4xf32>) {
    %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%iv)
    %tile = tensor.extract_slice %arg0[0, %j] [4, 2] [1, 1]
        : tensor<4x4xf32> to tensor<4x2xf32>
    %inserted = tensor.insert_slice %tile into %acc[0, %j] [4, 2] [1, 1]
        : tensor<4x2xf32> into tensor<4x4xf32>
    scf.yield %inserted : tensor<4x4xf32>
  }

  %scale_empty = tensor.empty() : tensor<4x4xf32>
  %scaled = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%scores : tensor<4x4xf32>) outs(%scale_empty : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %cst : f32
      linalg.yield %v : f32
  } -> tensor<4x4xf32>

  %exp_empty = tensor.empty() : tensor<4x4xf32>
  %exped = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%scaled : tensor<4x4xf32>) outs(%exp_empty : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.exp %in : f32
      linalg.yield %v : f32
  } -> tensor<4x4xf32>

  %one = arith.constant 1.0 : f32
  %prod_empty = tensor.empty() : tensor<4xf32>
  %prod_init = linalg.fill ins(%one : f32) outs(%prod_empty : tensor<4xf32>) -> tensor<4xf32>
  %prod = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%exped : tensor<4x4xf32>) outs(%prod_init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %out : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %prod : tensor<4xf32>
}
}
// -----

// @two_reductions: loop fans out to near_max (distance 1) and far_sum (distance 2).
// BFS must return near_max; finding far_sum would produce arith.addf instead.
// Reduction result: near_max (maxnumf). Elemwise result: empty.
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.maxnumf
// CHECK-NOT: arith.addf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "two_reductions"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %reduction, %elemwise = transform.fusion.find_next_reduction %loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %reduction : !transform.any_op
    transform.print %elemwise : !transform.any_op
    transform.yield
  }

  func.func @two_reductions(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 2.0 : f32

  %empty = tensor.empty() : tensor<4x4xf32>
  %scores = scf.for %iv = %c0 to %c2 step %c1
      iter_args(%acc = %empty) -> (tensor<4x4xf32>) {
    %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%iv)
    %tile = tensor.extract_slice %arg0[0, %j] [4, 2] [1, 1]
        : tensor<4x4xf32> to tensor<4x2xf32>
    %inserted = tensor.insert_slice %tile into %acc[0, %j] [4, 2] [1, 1]
        : tensor<4x2xf32> into tensor<4x4xf32>
    scf.yield %inserted : tensor<4x4xf32>
  }
  %neg_inf = arith.constant -3.40282347E+38 : f32
  %max_empty = tensor.empty() : tensor<4xf32>
  %max_init = linalg.fill ins(%neg_inf : f32) outs(%max_empty : tensor<4xf32>) -> tensor<4xf32>
  %near_max = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%scores : tensor<4x4xf32>) outs(%max_init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maxnumf %in, %out : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  %scale_empty = tensor.empty() : tensor<4x4xf32>
  %scaled = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%scores : tensor<4x4xf32>) outs(%scale_empty : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %cst : f32
      linalg.yield %v : f32
  } -> tensor<4x4xf32>

  %zero = arith.constant 0.0 : f32
  %sum_empty = tensor.empty() : tensor<4xf32>
  %sum_init = linalg.fill ins(%zero : f32) outs(%sum_empty : tensor<4xf32>) -> tensor<4xf32>
  %far_sum = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%scaled : tensor<4x4xf32>) outs(%sum_init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.addf %in, %out : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  %out_empty = tensor.empty() : tensor<4xf32>
  %combined = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%near_max, %far_sum : tensor<4xf32>, tensor<4xf32>)
      outs(%out_empty : tensor<4xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.addf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %combined : tensor<4xf32>
}
}
// -----

// @attention_like: loop -> red1(max) + elem1(exp_shift, uses loop+red1)
//                       -> red2(sum) -> elem2(norm, uses elem1+red2) -> red3(sum).
// red1 is at BFS distance 1; BFS stops there despite the deeper reduction chain.
// Elemwise result is empty because red1 directly consumes the loop output.
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.maxnumf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "attention_like"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %reduction, %elemwise = transform.fusion.find_next_reduction %loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %reduction : !transform.any_op
    transform.print %elemwise : !transform.any_op
    transform.yield
  }

  func.func @attention_like(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %empty = tensor.empty() : tensor<4x4xf32>
  %scores = scf.for %iv = %c0 to %c2 step %c1
      iter_args(%acc = %empty) -> (tensor<4x4xf32>) {
    %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%iv)
    %tile = tensor.extract_slice %arg0[0, %j] [4, 2] [1, 1]
        : tensor<4x4xf32> to tensor<4x2xf32>
    %inserted = tensor.insert_slice %tile into %acc[0, %j] [4, 2] [1, 1]
        : tensor<4x2xf32> into tensor<4x4xf32>
    scf.yield %inserted : tensor<4x4xf32>
  }
  %neg_inf = arith.constant -3.40282347E+38 : f32
  %max_empty = tensor.empty() : tensor<4xf32>
  %max_init = linalg.fill ins(%neg_inf : f32) outs(%max_empty : tensor<4xf32>) -> tensor<4xf32>
  %row_max = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%scores : tensor<4x4xf32>) outs(%max_init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maxnumf %in, %out : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  %shifted_empty = tensor.empty() : tensor<4x4xf32>
  %shifted = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%scores, %row_max : tensor<4x4xf32>, tensor<4xf32>)
      outs(%shifted_empty : tensor<4x4xf32>) {
    ^bb0(%in: f32, %m: f32, %out: f32):
      %sub = arith.subf %in, %m : f32
      %v = math.exp %sub : f32
      linalg.yield %v : f32
  } -> tensor<4x4xf32>
  %zero = arith.constant 0.0 : f32
  %sum_empty = tensor.empty() : tensor<4xf32>
  %sum_init = linalg.fill ins(%zero : f32) outs(%sum_empty : tensor<4xf32>) -> tensor<4xf32>
  %sum_exp = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%shifted : tensor<4x4xf32>) outs(%sum_init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.addf %in, %out : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  %norm_empty = tensor.empty() : tensor<4x4xf32>
  %normalized = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%shifted, %sum_exp : tensor<4x4xf32>, tensor<4xf32>)
      outs(%norm_empty : tensor<4x4xf32>) {
    ^bb0(%in: f32, %s: f32, %out: f32):
      %v = arith.divf %in, %s : f32
      linalg.yield %v : f32
  } -> tensor<4x4xf32>
  %wsum_empty = tensor.empty() : tensor<4xf32>
  %wsum_init = linalg.fill ins(%zero : f32) outs(%wsum_empty : tensor<4xf32>) -> tensor<4xf32>
  %weighted_sum = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%normalized : tensor<4x4xf32>) outs(%wsum_init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.addf %in, %out : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %weighted_sum : tensor<4xf32>
}
}
// -----

// @zero_index_broadcast: loop -> broadcasted scale -> sum. Constant-zero indexing-map results
// are singleton broadcasts and remain valid elementwise operations on the path to the reduction.
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.addf
// CHECK: IR printer
// CHECK: affine_map<(d0, d1) -> (d0, 0)>
// CHECK: iterator_types = ["parallel", "parallel"]
// CHECK: arith.mulf
// CHECK: module attributes {transform.with_named_sequence} {

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "zero_index_broadcast"} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %reduction, %elemwise = transform.fusion.find_next_reduction %loop
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %reduction : !transform.any_op
    transform.print %elemwise : !transform.any_op
    transform.yield
  }

  func.func @zero_index_broadcast(%arg0: tensor<4x4xf32>,
                                %scale: tensor<4x1xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %empty = tensor.empty() : tensor<4x4xf32>
  %scores = scf.for %iv = %c0 to %c2 step %c1
      iter_args(%acc = %empty) -> (tensor<4x4xf32>) {
    %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%iv)
    %tile = tensor.extract_slice %arg0[0, %j] [4, 2] [1, 1]
        : tensor<4x4xf32> to tensor<4x2xf32>
    %inserted = tensor.insert_slice %tile into %acc[0, %j] [4, 2] [1, 1]
        : tensor<4x2xf32> into tensor<4x4xf32>
    scf.yield %inserted : tensor<4x4xf32>
  }

  %scaled_empty = tensor.empty() : tensor<4x4xf32>
  %scaled = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, 0)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%scores, %scale : tensor<4x4xf32>, tensor<4x1xf32>)
      outs(%scaled_empty : tensor<4x4xf32>) {
    ^bb0(%score: f32, %factor: f32, %out: f32):
      %value = arith.mulf %score, %factor : f32
      linalg.yield %value : f32
  } -> tensor<4x4xf32>

  %zero = arith.constant 0.0 : f32
  %sum_empty = tensor.empty() : tensor<4xf32>
  %sum_init = linalg.fill ins(%zero : f32) outs(%sum_empty : tensor<4xf32>) -> tensor<4xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%scaled : tensor<4x4xf32>) outs(%sum_init : tensor<4xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %next = arith.addf %value, %acc : f32
      linalg.yield %next : f32
  } -> tensor<4xf32>
  return %sum : tensor<4xf32>
}
}
// -----

// CHECK: IR printer
// CHECK: arith.maxnumf
// CHECK: IR printer
// CHECK: arith.addf

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op

    %max, %max_elemwise = transform.fusion.find_next_reduction %loop[0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %max : !transform.any_op

    %sum, %sum_elemwise = transform.fusion.find_next_reduction %loop[1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %sum : !transform.any_op

    transform.yield
  }

  func.func @two_result_loop(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %empty0 = tensor.empty() : tensor<4x4xf32>
    %empty1 = tensor.empty() : tensor<4x4xf32>
    %scores:2 = scf.for %iv = %c0 to %c2 step %c1
        iter_args(%acc0 = %empty0, %acc1 = %empty1)
        -> (tensor<4x4xf32>, tensor<4x4xf32>) {
      %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%iv)
      %tile0 = tensor.extract_slice %arg0[0, %j] [4, 2] [1, 1]
          : tensor<4x4xf32> to tensor<4x2xf32>
      %inserted0 = tensor.insert_slice %tile0 into %acc0[0, %j] [4, 2] [1, 1]
          : tensor<4x2xf32> into tensor<4x4xf32>
      %tile1 = tensor.extract_slice %arg1[0, %j] [4, 2] [1, 1]
          : tensor<4x4xf32> to tensor<4x2xf32>
      %inserted1 = tensor.insert_slice %tile1 into %acc1[0, %j] [4, 2] [1, 1]
          : tensor<4x2xf32> into tensor<4x4xf32>
      scf.yield %inserted0, %inserted1 : tensor<4x4xf32>, tensor<4x4xf32>
    }

    %neg_inf = arith.constant -3.40282347E+38 : f32
    %max_empty = tensor.empty() : tensor<4xf32>
    %max_init = linalg.fill ins(%neg_inf : f32)
        outs(%max_empty : tensor<4xf32>) -> tensor<4xf32>
    %max = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%scores#0 : tensor<4x4xf32>) outs(%max_init : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %v = arith.maxnumf %in, %out : f32
        linalg.yield %v : f32
    } -> tensor<4xf32>

    %zero = arith.constant 0.0 : f32
    %sum_empty = tensor.empty() : tensor<4xf32>
    %sum_init = linalg.fill ins(%zero : f32)
        outs(%sum_empty : tensor<4xf32>) -> tensor<4xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%scores#1 : tensor<4x4xf32>) outs(%sum_init : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %v = arith.addf %in, %out : f32
        linalg.yield %v : f32
    } -> tensor<4xf32>

    return %max, %sum : tensor<4xf32>, tensor<4xf32>
  }
}
