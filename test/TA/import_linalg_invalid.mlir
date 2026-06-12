// RUN: mlir-opt --load-dialect-plugin=%neptune_ta_plugin --load-pass-plugin=%neptune_ta_plugin --pass-pipeline='builtin.module(func.func(linalg-to-ta))' %s --verify-diagnostics

#id1 = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#project_col = affine_map<(d0, d1) -> (d1)>

func.func @incompatible_expanded_pack() -> tensor<1x4xi64> {
  %src_empty = tensor.empty() : tensor<4xi64>
  // expected-error @+1 {{incompatible logical axis packs: existing pack has 1 axes, but newly required pack has 2 axes}}
  %src = linalg.generic {indexing_maps = [#id1], iterator_types = ["parallel"]}
      outs(%src_empty : tensor<4xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 0 : index
    %ii = arith.index_cast %i : index to i64
    linalg.yield %ii : i64
  } -> tensor<4xi64>
  %expanded = tensor.expand_shape %src [[0, 1]] output_shape [1, 4]
      : tensor<4xi64> into tensor<1x4xi64>
  %out_empty = tensor.empty() : tensor<1x4xi64>
  %out = linalg.generic {
      indexing_maps = [#project_col, #id2, #id2],
      iterator_types = ["parallel", "parallel"]}
      ins(%src, %expanded : tensor<4xi64>, tensor<1x4xi64>)
      outs(%out_empty : tensor<1x4xi64>) {
  ^bb0(%direct: i64, %expanded_arg: i64, %out_arg: i64):
    linalg.yield %expanded_arg : i64
  } -> tensor<1x4xi64>
  return %out : tensor<1x4xi64>
}
