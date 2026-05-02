// Attention at "Level 0: Algorithmic IR".
//
// Plain mathematical description of scaled dot-product attention with no
// scheduling decisions:
//   - no tiling of any axis,
//   - no loop-carried recurrence (no online softmax),
//   - no parallel/sequential decomposition,
//   - no memory placement, layout, or hardware information.
//
// Math:
//   S[b, h, i, j] = (1 / sqrt(D)) * sum_k Q[b, h, i, k] * K[b, h, j, k]
//   P[b, h, i, j] = softmax_j( S[b, h, i, :] )[j]
//   O[b, h, i, d] = sum_j P[b, h, i, j] * V[b, h, j, d]
//
// Shapes:
//   Q, K, V, O : tensor<B x H x N x D>   with B=1, H=32, N=4096, D=128.
//   S, P       : tensor<B x H x N x N>   (explicit tensor values at this level).

module {
  func.func @attention(
      %q: tensor<1x32x4096x128xf16>,
      %k: tensor<1x32x4096x128xf16>,
      %v: tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16> {

    %f0    = arith.constant 0.0                : f32
    %scale = arith.constant 0.0883883476483184 : f32   // 1 / sqrt(128)

    // S = scale * (Q @ K^T)   :   [B, H, N, N]
    %s_e    = tensor.empty() : tensor<1x32x4096x4096xf32>
    %s_zero = linalg.fill ins(%f0 : f32)
        outs(%s_e : tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %qk = linalg.generic {
        indexing_maps = [
          affine_map<(b, h, i, j, k) -> (b, h, i, k)>,   // Q
          affine_map<(b, h, i, j, k) -> (b, h, j, k)>,   // K  (contract on k)
          affine_map<(b, h, i, j, k) -> (b, h, i, j)>],  // S
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%q, %k : tensor<1x32x4096x128xf16>, tensor<1x32x4096x128xf16>)
        outs(%s_zero : tensor<1x32x4096x4096xf32>) {
      ^bb0(%qe: f16, %ke: f16, %acc: f32):
        %qf  = arith.extf %qe  : f16 to f32
        %kf  = arith.extf %ke  : f16 to f32
        %prd = arith.mulf %qf, %kf : f32
        %sum = arith.addf %acc, %prd : f32
        linalg.yield %sum : f32
    } -> tensor<1x32x4096x4096xf32>

    %s = linalg.map
        ins(%qk : tensor<1x32x4096x4096xf32>)
        outs(%s_e : tensor<1x32x4096x4096xf32>)
        (%x: f32, %_: f32) {
          %y = arith.mulf %x, %scale : f32
          linalg.yield %y : f32
    }

    // P = softmax(S, dim = j)
    %p_e = tensor.empty() : tensor<1x32x4096x4096xf32>
    %p   = linalg.softmax dimension(3)
        ins(%s  : tensor<1x32x4096x4096xf32>)
        outs(%p_e : tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %p_f16 = arith.truncf %p
        : tensor<1x32x4096x4096xf32> to tensor<1x32x4096x4096xf16>

    // O = P @ V   :   [B, H, N, D]   (accumulate in f32, truncate to f16).
    %o_e    = tensor.empty() : tensor<1x32x4096x128xf32>
    %o_zero = linalg.fill ins(%f0 : f32)
        outs(%o_e : tensor<1x32x4096x128xf32>) -> tensor<1x32x4096x128xf32>
    %o_f32 = linalg.generic {
        indexing_maps = [
          affine_map<(b, h, i, d, j) -> (b, h, i, j)>,   // P
          affine_map<(b, h, i, d, j) -> (b, h, j, d)>,   // V
          affine_map<(b, h, i, d, j) -> (b, h, i, d)>],  // O
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%p_f16, %v : tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>)
        outs(%o_zero : tensor<1x32x4096x128xf32>) {
      ^bb0(%pe: f16, %ve: f16, %acc: f32):
        %pf  = arith.extf %pe  : f16 to f32
        %vf  = arith.extf %ve  : f16 to f32
        %prd = arith.mulf %pf, %vf : f32
        %sum = arith.addf %acc, %prd : f32
        linalg.yield %sum : f32
    } -> tensor<1x32x4096x128xf32>

    %o = arith.truncf %o_f32
        : tensor<1x32x4096x128xf32> to tensor<1x32x4096x128xf16>
    return %o : tensor<1x32x4096x128xf16>
  }
}
