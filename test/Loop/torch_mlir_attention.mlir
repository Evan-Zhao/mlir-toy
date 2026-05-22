// RUN: mlir-opt --load-dialect-plugin=%neptune_loop_plugin %s --transform-interpreter 2>&1 | FileCheck %s --check-prefix=MATCH
//
// Transform-dialect schedule that transforms the Torch-MLIR attention payload
// below into a FlashAttention-like fused program. Mirrors
// `_schedule_attention_flash` from Neptune, with some extra dimension
// reshaping capabilities.

!any = !transform.any_op
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>

module attributes {transform.with_named_sequence} {
  // Checks if an op is a linalg operation with exactly 4D of an iteration space,
  // where the first 3 dims are parallel and the last dim is reduction.
  transform.named_sequence @match_3d_1d_reduction(%candidate: !any {transform.readonly}) -> !any {
    %matched = transform.match.structured %candidate : (!any) -> !any {
    ^bb0(%op: !any):
      transform.match.structured.dim %op[0, 1, 2] {parallel} : !any
      transform.match.structured.dim %op[3] {reduction} : !any
      transform.match.structured.yield %op : !any
    }
    transform.yield %matched : !any
  }

  transform.named_sequence @return_matched(%arg: !any {transform.readonly}) -> !any {
    transform.yield %arg : !any
  }

  transform.named_sequence @__transform_main(%module: !any) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!any) -> !any
    // Prepass 1. The input program from TorchMLIR has some "keepdim" reductions with an indexing map
    // like `(d0, d1, d2, d3) -> (d0, d1, d2, 0)`, and many MLIR transformations will reject these maps.
    // This step converts these keepdim reductions to non-keepdim ones, and remove the `0` dim
    // in these indexing maps.
    transform.linalg.fold_zero_indexed_unit_dims %func : !any

    // Prepass 2. TorchMLIR attention describes 4D matmuls with linalg.batch_matmul,
    // which requires exactly one "batch" dimension, by fusing the batch and head dims together.
    // We separate these dims by "generalizing" the batch matmul (and its producers,
    // like linalg.transpose) to a linalg.generic, then fuse the reshaping operation into the generic op.
    // Do that for the transpose op first (there should be exactly one for the QK^T matmul).
    %transposes = transform.structured.match ops{["linalg.transpose"]} in %func : (!any) -> !any
    %transposes_lg = transform.structured.generalize %transposes : (!any) -> !any
    transform.linalg.fold_expanding_reshape %transposes_lg : !any
    // Then do batch matmul ops.
    %bmms = transform.structured.match ops{["linalg.batch_matmul"]} in %func : (!any) -> !any
    %bmms_lg = transform.structured.generalize %bmms : (!any) -> !any
    transform.linalg.fold_expanding_reshape %bmms_lg : !any
    // This pattern cancels out back-to-back expand+collapse pairs.
    transform.apply_patterns to %func { transform.apply_patterns.tensor.reassociative_reshape_folding } : !any

    // Take the first batch matmul `bmm0`.
    // Inline elementwise ops before bmm0 (in this case, should be F16->F32 casts) into it.
    %bmm0, %_0 = transform.split_handle %bmms_lg : (!any) -> (!any, !any)
    transform.linalg.greedy_inline_elementwise %bmm0 : !any
    transform.linalg.erase_unused_operands_and_results %bmm0 : !any
    // Tile all parallel dimensions of bmm0 (b, h, i, j) into a scf.forall loop.
    // We'll fuse everything else into this loop nest.
    %_1, %forall_loop = transform.structured.tile_using_forall
        %bmm0 tile_sizes [1, 1, 128, 64, 0] : (!any) -> (!any, !any)

    // Fusion 1. Match an element-wise op that is a consumer of mm0, and fuse it into mm0.
    //   TVM: sch.reverse_compute_at(bscale, j0)
    %bscale = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %fused_bscale = transform.fusion.into_producer %bscale into %forall_loop : (!any, !any) -> !any
    // Fusion can create redundant loop-carried values, and canonicalization removes them.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any

    // Fusion 2. Fuse row-max into forall, splitting a serial `for` loop from forall in the process.
    // Because of how MLIR scf.for works, this fusion implicitly also r-factors the reduction.
    //   TVM: sch.reverse_compute_at(bmax, j0); sch.rfactor(...)
    %consumers = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %_2, %bmax = transform.foreach_match restrict_root in %consumers
        @match_3d_1d_reduction -> @return_matched : (!any) -> (!any, !any)
    // The "row-max" in the input program has two outputs: the max value and the argmax.
    // The subsequent fusion only supports single-output ops, so we remove the unused argmax
    // output before fusion.
    transform.linalg.erase_unused_operands_and_results %bmax : !any
    %fused_bmax, %j0_loop = transform.scf.fuse_reduction_into_forall
        %bmax into %forall_loop : (!any, !any) -> (!any, !any)

    // Fusion 3 (rolling update). First find the nearest reduction reachable from the loop's
    // output value, together with the ordered elementwise chain between them.
    %bsum, %elemwise = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    // Clone and fuse that elementwise chain under %forall_loop and %j0_loop,
    // publishing the "sidecar" tensors as extra loop results.
    %elemwise_sidecars = transform.fusion.clone_fuse_elemwise
        %elemwise into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    // Repair the first reduction frontier by turning it into loop-carried state
    // driven by the relayed sidecar value.
    %fused_bsum = transform.fusion.repair_reduction_frontier
        (%fused_bmax, %bsum) and (%elemwise, %elemwise_sidecars) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    // Fusion 4. Apply rolling update again, this time with the second matmul being the reduction.
    %bmm1, %elemwise_1 = transform.fusion.find_next_reduction
        %forall_loop : (!any) -> (!any, !any)
    // Also inline (F16->F32 casts) into the second matmul. `operand_number = 1` says only
    // inline producers of the RHS of the matmul.
    transform.linalg.greedy_inline_elementwise %bmm1 { operand_number = 1 } : !any
    transform.linalg.erase_unused_operands_and_results %bmm1 : !any
    %elemwise_sidecars_1 = transform.fusion.clone_fuse_elemwise
        %elemwise_1 into %forall_loop, %j0_loop : (!any, !any, !any) -> !any
    %_3 = transform.fusion.repair_reduction_frontier
        (%fused_bsum, %bmm1) and (%elemwise_1, %elemwise_sidecars_1) into %forall_loop, %j0_loop
        : (!any, !any, !any, !any, !any, !any) -> !any

    // Fusion 5. Fuse the trailing FP32->FP16 cast into the forall loop (but outside the for loop).
    %trunc = transform.get_consumers_of_result %forall_loop[0] : (!any) -> !any
    %fused_trunc = transform.fusion.into_producer %trunc into %forall_loop : (!any, !any) -> !any

    // Post-pass: pushes lingering init tensor (see destination-passing style)
    // before and outside the loops into the loop body.
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.scf.localize_scratch_tensors %func : !any
    transform.apply_patterns to %func { transform.apply_patterns.canonicalization } : !any
    transform.apply_cse to %func : !any

    transform.yield
  }

  func.func @attention(%arg0: tensor<1x4x128x64xf16>, %arg1: tensor<1x4x128x64xf16>,
      %arg2: tensor<1x4x128x64xf16>) -> tensor<1x4x128x64xf16> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<1x4x128x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x4x128x64xf16>) outs(%0 : tensor<1x4x128x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x4x128x64xf16>) outs(%0 : tensor<1x4x128x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x64xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x4x128x64xf16>) outs(%0 : tensor<1x4x128x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x64xf32>
    %4 = tensor.empty() : tensor<1x4x64x128xf32>
    %transposed = linalg.transpose ins(%2 : tensor<1x4x128x64xf32>) outs(%4 : tensor<1x4x64x128xf32>) permutation = [0, 1, 3, 2]
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x4x128x64xf32> into tensor<4x128x64xf32>
    %collapsed_2 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x4x64x128xf32> into tensor<4x64x128xf32>
    %5 = tensor.empty() : tensor<4x128x128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %7 = linalg.batch_matmul ins(%collapsed, %collapsed_2 : tensor<4x128x64xf32>, tensor<4x64x128xf32>) outs(%6 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %expanded = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 4, 128, 128] : tensor<4x128x128xf32> into tensor<1x4x128x128xf32>
    %8 = tensor.empty() : tensor<1x4x128x128xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x4x128x128xf32>) outs(%8 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.mulf %in, %cst_1 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x128xf32>
    %10 = tensor.empty() : tensor<1x4x128xi64>
    %11 = linalg.fill ins(%c0_i64 : i64) outs(%10 : tensor<1x4x128xi64>) -> tensor<1x4x128xi64>
    %12 = tensor.empty() : tensor<1x4x128xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<1x4x128xf32>) -> tensor<1x4x128xf32>
    %14:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9 : tensor<1x4x128x128xf32>) outs(%13, %11 : tensor<1x4x128xf32>, tensor<1x4x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_7: i64):
      %29 = linalg.index 3 : index
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.maximumf %in, %out : f32
      %32 = arith.cmpf ogt, %in, %out : f32
      %33 = arith.select %32, %30, %out_7 : i64
      linalg.yield %31, %33 : f32, i64
    } -> (tensor<1x4x128xf32>, tensor<1x4x128xi64>)
    %expanded_3 = tensor.expand_shape %14#0 [[0], [1], [2, 3]] output_shape [1, 4, 128, 1] : tensor<1x4x128xf32> into tensor<1x4x128x1xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %expanded_3 : tensor<1x4x128x128xf32>, tensor<1x4x128x1xf32>) outs(%8 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.subf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x128xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<1x4x128x128xf32>) outs(%8 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.exp %in : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x128xf32>
    %17 = tensor.empty() : tensor<1x4x128x1xf32>
    %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<1x4x128x1xf32>) -> tensor<1x4x128x1xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%16 : tensor<1x4x128x128xf32>) outs(%18 : tensor<1x4x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.addf %in, %out : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x1xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %19 : tensor<1x4x128x128xf32>, tensor<1x4x128x1xf32>) outs(%8 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.divf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x128xf32>
    %21 = tensor.empty() : tensor<1x4x128x128xf16>
    %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20 : tensor<1x4x128x128xf32>) outs(%21 : tensor<1x4x128x128xf16>) {
    ^bb0(%in: f32, %out: f16):
      %29 = arith.truncf %in : f32 to f16
      linalg.yield %29 : f16
    } -> tensor<1x4x128x128xf16>
    %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22 : tensor<1x4x128x128xf16>) outs(%8 : tensor<1x4x128x128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %29 = arith.extf %in : f16 to f32
      linalg.yield %29 : f32
    } -> tensor<1x4x128x128xf32>
    %collapsed_4 = tensor.collapse_shape %23 [[0, 1], [2], [3]] : tensor<1x4x128x128xf32> into tensor<4x128x128xf32>
    %collapsed_5 = tensor.collapse_shape %3 [[0, 1], [2], [3]] : tensor<1x4x128x64xf32> into tensor<4x128x64xf32>
    %24 = tensor.empty() : tensor<4x128x64xf32>
    %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %26 = linalg.batch_matmul ins(%collapsed_4, %collapsed_5 : tensor<4x128x128xf32>, tensor<4x128x64xf32>) outs(%25 : tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %expanded_6 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 4, 128, 64] : tensor<4x128x64xf32> into tensor<1x4x128x64xf32>
    %27 = tensor.empty() : tensor<1x4x128x64xf16>
    %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<1x4x128x64xf32>) outs(%27 : tensor<1x4x128x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %29 = arith.truncf %in : f32 to f16
      linalg.yield %29 : f16
    } -> tensor<1x4x128x64xf16>
    return %28 : tensor<1x4x128x64xf16>
  }
}

// MATCH-LABEL: func.func @attention(
// MATCH-NOT: linalg.batch_matmul
// MATCH-NOT: linalg.transpose
// MATCH-NOT: tensor.collapse_shape
// MATCH-NOT: tensor.expand_shape
// MATCH: %[[OUT:.+]] = scf.forall (%{{.*}}) in (4) shared_outs(%{{.*}} = %{{.*}}) -> (tensor<1x4x128x64xf32>)
// MATCH: %{{.*}}:3 = scf.for %{{.*}} = %c0 to %c2 step %c1 iter_args(
// MATCH: tensor.empty() : tensor<1x1x128x64xf32>
// MATCH: linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
// MATCH: math.exp
// MATCH: arith.divf %cst, %{{.*}} : f32
// MATCH: linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%{{.*}} : tensor<1x4x128x64xf32>) outs(%{{.*}} : tensor<1x4x128x64xf16>)
