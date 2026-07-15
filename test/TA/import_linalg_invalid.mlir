// RUN: neptune-opt --pass-pipeline='builtin.module(func.func(linalg-to-ta))' %s --verify-diagnostics

#id1 = affine_map<(d0) -> (d0)>

func.func @unsupported_non_unit_expanded_pack() -> tensor<2x2xi64> {
  %src_empty = tensor.empty() : tensor<4xi64>
  // expected-error @+1 {{cannot erase tensor axis during relabel}}
  %src = linalg.generic {indexing_maps = [#id1], iterator_types = ["parallel"]}
      outs(%src_empty : tensor<4xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 0 : index
    %ii = arith.index_cast %i : index to i64
    linalg.yield %ii : i64
  } -> tensor<4xi64>
  %expanded = tensor.expand_shape %src [[0, 1]] output_shape [2, 2]
      : tensor<4xi64> into tensor<2x2xi64>
  return %expanded : tensor<2x2xi64>
}
