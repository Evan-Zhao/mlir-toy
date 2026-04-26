// FlashAttention at "Level 1: Scheduled Tile IR".
//
// Iterator-defining ops:
//   - scf.forall          (outer parallel decomposition; produces disjoint
//                          output tiles via tensor.parallel_insert_slice)
//   - affine.for          (inner sequential K-block loop; carries online
//                          softmax recurrence through iter_args)
//
// Tile bodies use linalg structured ops (matmul, reduce, broadcast, fill)
// and elementwise arith/math on tensor values. No GPU hierarchy, no
// memory-placement encodings, no warp/fragment information.

module {
  func.func @flash_attention(
      %q: tensor<1x32x4096x128xf16>,
      %k: tensor<1x32x4096x128xf16>,
      %v: tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16> {
    %f0       = arith.constant 0.0          : f32
    %neg_inf  = arith.constant -3.40282347E+38 : f32
    // Scale by log2(e) / sqrt(d) because the softmax uses exp2.
    %scale    = arith.constant 1.2751743082459868E-1 : f32
    %c128     = arith.constant 128 : index
    %c64      = arith.constant 64  : index

    %out_init = tensor.empty() : tensor<1x32x4096x128xf16>
    %result = scf.forall (%b, %h, %i_block) in (1, 32, 32)
        shared_outs(%out_iter = %out_init) -> tensor<1x32x4096x128xf16> {

      %i_base = arith.muli %i_block, %c128 : index

      %q_tile = tensor.extract_slice
          %q[%b, %h, %i_base, 0] [1, 1, 128, 128] [1, 1, 1, 1]
          : tensor<1x32x4096x128xf16> to tensor<128x128xf16>

      // Online-softmax state: accumulator O, row max m, row normalizer l.
      %acc_e = tensor.empty() : tensor<128x128xf32>
      %acc0  = linalg.fill ins(%f0 : f32) outs(%acc_e : tensor<128x128xf32>)
          -> tensor<128x128xf32>
      %m_e = tensor.empty() : tensor<128xf32>
      %m0  = linalg.fill ins(%neg_inf : f32) outs(%m_e : tensor<128xf32>)
          -> tensor<128xf32>
      %l_e = tensor.empty() : tensor<128xf32>
      %l0  = linalg.fill ins(%f0 : f32) outs(%l_e : tensor<128xf32>)
          -> tensor<128xf32>

      %l_final, %acc_final, %m_final = affine.for %j = 0 to 64
          iter_args(%l_iter = %l0, %acc_iter = %acc0, %m_iter = %m0)
          -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>) {

        %j_base = affine.apply affine_map<(d) -> (d * 64)>(%j)

        %k_tile = tensor.extract_slice
            %k[%b, %h, %j_base, 0] [1, 1, 64, 128] [1, 1, 1, 1]
            : tensor<1x32x4096x128xf16> to tensor<64x128xf16>

        // S = Q @ K^T : [128,128] x [64,128]^T -> [128,64]
        %s_e = tensor.empty() : tensor<128x64xf32>
        %s0  = linalg.fill ins(%f0 : f32) outs(%s_e : tensor<128x64xf32>)
            -> tensor<128x64xf32>
        %qk = linalg.matmul
            indexing_maps = [
              affine_map<(m, n, k) -> (m, k)>,
              affine_map<(m, n, k) -> (n, k)>,
              affine_map<(m, n, k) -> (m, n)>]
            ins(%q_tile, %k_tile : tensor<128x128xf16>, tensor<64x128xf16>)
            outs(%s0 : tensor<128x64xf32>) -> tensor<128x64xf32>

        // Scale by log2(e)/sqrt(d).
        %score = linalg.map ins(%qk : tensor<128x64xf32>) outs(%s_e : tensor<128x64xf32>)
            (%x: f32, %_: f32) {
              %y = arith.mulf %x, %scale : f32
              linalg.yield %y : f32
        }

        // Row-wise max over K dim.
        %rm_e = tensor.empty() : tensor<128xf32>
        %rm_seed = linalg.fill ins(%neg_inf : f32) outs(%rm_e : tensor<128xf32>)
            -> tensor<128xf32>
        %row_max = linalg.reduce { arith.maximumf }
            ins(%score : tensor<128x64xf32>)
            outs(%rm_seed : tensor<128xf32>)
            dimensions = [1]

        %m_next = arith.maximumf %m_iter, %row_max : tensor<128xf32>

        // P = exp2(score - m_next) (broadcast m_next over K dim).
        %mbc_e = tensor.empty() : tensor<128x64xf32>
        %m_next_bc = linalg.broadcast
            ins(%m_next : tensor<128xf32>)
            outs(%mbc_e : tensor<128x64xf32>)
            dimensions = [1]
        %shifted = arith.subf %score, %m_next_bc : tensor<128x64xf32>
        %p       = math.exp2 %shifted : tensor<128x64xf32>

        // Row-wise sum of P.
        %rs_e = tensor.empty() : tensor<128xf32>
        %rs_seed = linalg.fill ins(%f0 : f32) outs(%rs_e : tensor<128xf32>)
            -> tensor<128xf32>
        %row_sum = linalg.reduce { arith.addf }
            ins(%p : tensor<128x64xf32>)
            outs(%rs_seed : tensor<128xf32>)
            dimensions = [1]

        // l_next = exp2(m_iter - m_next) * l_iter + row_sum
        %l_d   = arith.subf %m_iter, %m_next : tensor<128xf32>
        %l_s   = math.exp2 %l_d : tensor<128xf32>
        %l_sc  = arith.mulf %l_iter, %l_s : tensor<128xf32>
        %l_next = arith.addf %l_sc, %row_sum : tensor<128xf32>

        // PV = P @ V : [128,64] x [64,128] -> [128,128]
        %p_f16 = arith.truncf %p : tensor<128x64xf32> to tensor<128x64xf16>
        %v_tile = tensor.extract_slice
            %v[%b, %h, %j_base, 0] [1, 1, 64, 128] [1, 1, 1, 1]
            : tensor<1x32x4096x128xf16> to tensor<64x128xf16>
        %pv_e = tensor.empty() : tensor<128x128xf32>
        %pv0  = linalg.fill ins(%f0 : f32) outs(%pv_e : tensor<128x128xf32>)
            -> tensor<128x128xf32>
        %pv = linalg.matmul
            ins(%p_f16, %v_tile : tensor<128x64xf16>, tensor<64x128xf16>)
            outs(%pv0 : tensor<128x128xf32>) -> tensor<128x128xf32>

        // acc_next = exp2(m_iter - m_next) * acc_iter + PV.
        %mp_e = tensor.empty() : tensor<128x128xf32>
        %m_prev_bc = linalg.broadcast
            ins(%m_iter : tensor<128xf32>)
            outs(%mp_e : tensor<128x128xf32>)
            dimensions = [1]
        %mn_e = tensor.empty() : tensor<128x128xf32>
        %m_next_bc128 = linalg.broadcast
            ins(%m_next : tensor<128xf32>)
            outs(%mn_e : tensor<128x128xf32>)
            dimensions = [1]
        %acc_d  = arith.subf %m_prev_bc, %m_next_bc128 : tensor<128x128xf32>
        %acc_s  = math.exp2 %acc_d : tensor<128x128xf32>
        %acc_sc = arith.mulf %acc_iter, %acc_s : tensor<128x128xf32>
        %acc_next = arith.addf %acc_sc, %pv : tensor<128x128xf32>

        affine.yield %l_next, %acc_next, %m_next
            : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>
      } {pipeline_stages = 2 : i32}

      // O / l (broadcast l over N dim), then truncate.
      %lbc_e = tensor.empty() : tensor<128x128xf32>
      %l_bc = linalg.broadcast
          ins(%l_final : tensor<128xf32>)
          outs(%lbc_e : tensor<128x128xf32>)
          dimensions = [1]
      %norm     = arith.divf  %acc_final, %l_bc : tensor<128x128xf32>
      %norm_f16 = arith.truncf %norm
          : tensor<128x128xf32> to tensor<128x128xf16>

      scf.forall.in_parallel {
        tensor.parallel_insert_slice %norm_f16 into
            %out_iter[%b, %h, %i_base, 0] [1, 1, 128, 128] [1, 1, 1, 1]
            : tensor<128x128xf16> into tensor<1x32x4096x128xf16>
      }
    }

    return %result : tensor<1x32x4096x128xf16>
  }
}
