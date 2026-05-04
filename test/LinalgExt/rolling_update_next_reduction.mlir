// RUN: mlir-opt %s \
// RUN:   --load-dialect-plugin=%neptune_linalg_ext_plugin \
// RUN:   --transform-preload-library=transform-library-paths=%S/Inputs/rolling_update_next_reduction.transform.mlir \
// RUN:   --transform-interpreter | FileCheck %s

// Payload functions are structured to also serve as inputs for
// rolling_update_elemwise_chain tests.

// @linear_chain: loop -> scale(mulf) -> sum(addf). One elemwise, one reduction.
// Reduction result: addf with reduction iterator.
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.addf
// Elemwise result: scale op (mulf, all-parallel).
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "parallel"]
// CHECK: arith.mulf

// @direct_reduce: loop -> max(maxnumf). Reduction at BFS distance 1, no elemwise.
// Reduction result: maxnumf with reduction iterator.
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.maxnumf
// Elemwise result: empty — no checks needed.

// @two_elemwise: loop -> scale(mulf) -> exp -> product(mulf). Two elemwise hops.
// Reduction result: product op (mulf, with reduction iterator).
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.mulf
// Elemwise result: scale (mulf) then exp, in def-use order.
// CHECK: IR printer
// CHECK: arith.mulf
// CHECK: math.exp

// @two_reductions: loop fans out to near_max (distance 1) and far_sum (distance 2).
// BFS must return near_max; finding far_sum would produce arith.addf instead.
// Reduction result: near_max (maxnumf). Elemwise result: empty.
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.maxnumf
// CHECK-NOT: arith.addf

// @attention_like: loop -> red1(max) + elem1(exp_shift, uses loop+red1)
//                       -> red2(sum) -> elem2(norm, uses elem1+red2) -> red3(sum).
// red1 is at BFS distance 1; BFS stops there despite the deeper reduction chain.
// Elemwise result is empty because red1 directly consumes the loop output.
// CHECK: IR printer
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.maxnumf
// CHECK: module {

module {

// loop -> scale -> sum
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

// loop -> max (no elemwise; reduction at BFS distance 1)
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

// loop -> scale(mulf) -> exp(math.exp) -> product(mulf reduction)
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

// loop fans out to:
//   near_max  (maxnumf reduction, BFS distance 1 from loop)
//   far_sum   (addf reduction, BFS distance 2 via an intermediate scale)
// BFS must find near_max first.
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

  // distance 1: directly consumes %scores
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

  // distance 2: %scores -> scale -> far_sum
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

  // Combine both to keep %far_sum live.
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

// loop -> red1(max) -> elem1(exp_shift, consumes loop+red1)
//      -> red2(sum) -> elem2(norm, consumes elem1+red2) -> red3(weighted sum)
// red1 is at BFS distance 1; elem1 is also at distance 1 but is not a reduction.
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

  // red1: row max
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

  // elem1: exp(scores - row_max), broadcasts row_max along j
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

  // red2: row sum of exp
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

  // elem2: normalize by row sum
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

  // red3: weighted sum
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

} // module
