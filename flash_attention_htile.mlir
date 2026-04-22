// Hand translation of Neptune's attention_after_mem2reg into value-based HTile IR.
//
// This models Neptune's VR tile-IR shape: global memory accesses happen at the
// htile.load/htile.store boundaries, while local/shared tile intermediates are
// SSA tensor values carrying HTile placement encodings.

#shared = #htile.encoding<placement = shared>
#local = #htile.encoding<placement = local>

module {
  func.func @flash_attention_htile(
      %q: memref<1x32x4096x128xf16>,
      %k: memref<1x32x4096x128xf16>,
      %v: memref<1x32x4096x128xf16>,
      %out: memref<1x32x4096x128xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index

    %f0 = arith.constant 0.0 : f32
    %neg_f32_max = arith.constant -3.40282347E+38 : f32
    %scale = arith.constant 1.2751743082459868E-1 : f32

    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c32, %grid_y = %c32, %grid_z = %c1)
               threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
      %i_base = arith.muli %bx, %c128 : index

      %q_shared = htile.load %q[%bz, %by, %i_base, %c0]
          : memref<1x32x4096x128xf16> -> tensor<128x128xf16, #shared>

      %max_prev0 = htile.full %neg_f32_max
          : f32 -> tensor<128xf32, #local>
      %acc0 = htile.full %f0
          : f32 -> tensor<128x128xf32, #local>
      %exp_sum0 = htile.full %f0
          : f32 -> tensor<128xf32, #local>

      %exp_sum_final, %acc_final, %max_final = scf.for %j = %c0 to %c64 step %c1
          iter_args(%exp_sum_i = %exp_sum0,
                    %acc_i = %acc0,
                    %max_prev_i = %max_prev0)
          -> (tensor<128xf32, #local>,
              tensor<128x128xf32, #local>,
              tensor<128xf32, #local>) {
        %j_base = arith.muli %j, %c64 : index

        %k_shared = htile.load %k[%bz, %by, %j_base, %c0]
            : memref<1x32x4096x128xf16> -> tensor<64x128xf16, #shared>
        %k_shared_t = htile.permute %k_shared permutation [1, 0]
            : tensor<64x128xf16, #shared> -> tensor<128x64xf16, #shared>

        %qk = htile.dot %q_shared, %k_shared_t
            : tensor<128x128xf16, #shared>, tensor<128x64xf16, #shared>
            -> tensor<128x64xf32, #local>

        %scale_tile = htile.full %scale
            : f32 -> tensor<128x64xf32, #local>
        %score = arith.mulf %qk, %scale_tile
            : tensor<128x64xf32, #local>

        %row_max = htile.reduce %score axis 1 kind "max"
            : tensor<128x64xf32, #local> -> tensor<128xf32, #local>
        %max_next = arith.maximumf %max_prev_i, %row_max
            : tensor<128xf32, #local>

        // Neptune's tile arithmetic broadcasts the row vector after unsqueeze.
        // MLIR arith ops require equal operand types, so materialize the
        // broadcast explicitly.
        %max_next_64_empty = tensor.empty()
            : tensor<128x64xf32, #local>
        %max_next_64 = linalg.broadcast
            ins(%max_next : tensor<128xf32, #local>)
            outs(%max_next_64_empty : tensor<128x64xf32, #local>)
            dimensions = [1]

        %shifted_score = arith.subf %score, %max_next_64
            : tensor<128x64xf32, #local>
        %softmax_exp = math.exp2 %shifted_score
            : tensor<128x64xf32, #local>

        %row_exp_sum = htile.reduce %softmax_exp axis 1 kind "sum"
            : tensor<128x64xf32, #local> -> tensor<128xf32, #local>

        %exp_sum_delta = arith.subf %max_prev_i, %max_next
            : tensor<128xf32, #local>
        %exp_sum_scale = math.exp2 %exp_sum_delta
            : tensor<128xf32, #local>
        %exp_sum_scaled = arith.mulf %exp_sum_i, %exp_sum_scale
            : tensor<128xf32, #local>
        %exp_sum_next = arith.addf %exp_sum_scaled, %row_exp_sum
            : tensor<128xf32, #local>

        %softmax_exp_f16 = arith.truncf %softmax_exp
            : tensor<128x64xf32, #local> to tensor<128x64xf16, #local>

        %v_shared = htile.load %v[%bz, %by, %j_base, %c0]
            : memref<1x32x4096x128xf16> -> tensor<64x128xf16, #shared>
        %acc_rf = htile.dot %softmax_exp_f16, %v_shared
            : tensor<128x64xf16, #local>, tensor<64x128xf16, #shared>
            -> tensor<128x128xf32, #local>

        %max_prev_128_empty = tensor.empty()
            : tensor<128x128xf32, #local>
        %max_prev_128 = linalg.broadcast
            ins(%max_prev_i : tensor<128xf32, #local>)
            outs(%max_prev_128_empty : tensor<128x128xf32, #local>)
            dimensions = [1]
        %max_next_128_empty = tensor.empty()
            : tensor<128x128xf32, #local>
        %max_next_128 = linalg.broadcast
            ins(%max_next : tensor<128xf32, #local>)
            outs(%max_next_128_empty : tensor<128x128xf32, #local>)
            dimensions = [1]

        %acc_delta = arith.subf %max_prev_128, %max_next_128
            : tensor<128x128xf32, #local>
        %acc_scale = math.exp2 %acc_delta
            : tensor<128x128xf32, #local>
        %acc_scaled = arith.mulf %acc_i, %acc_scale
            : tensor<128x128xf32, #local>
        %acc_next = arith.addf %acc_scaled, %acc_rf
            : tensor<128x128xf32, #local>

        scf.yield %exp_sum_next, %acc_next, %max_next
            : tensor<128xf32, #local>,
              tensor<128x128xf32, #local>,
              tensor<128xf32, #local>
      }

      %exp_sum_128_empty = tensor.empty()
          : tensor<128x128xf32, #local>
      %exp_sum_128 = linalg.broadcast
          ins(%exp_sum_final : tensor<128xf32, #local>)
          outs(%exp_sum_128_empty : tensor<128x128xf32, #local>)
          dimensions = [1]

      %norm = arith.divf %acc_final, %exp_sum_128
          : tensor<128x128xf32, #local>
      %norm_f16 = arith.truncf %norm
          : tensor<128x128xf32, #local> to tensor<128x128xf16, #local>

      htile.store %norm_f16, %out[%bz, %by, %i_base, %c0]
          : tensor<128x128xf16, #local>, memref<1x32x4096x128xf16>

      gpu.terminator
    }

    return
  }
}
