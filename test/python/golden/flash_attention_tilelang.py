import tilelang.language as T

@T.prim_func
def flash_attention_htile(buf_0: T.Buffer((1, 32, 4096, 128), 'float16'), buf_1: T.Buffer((1, 32, 4096, 128), 'float16'), buf_2: T.Buffer((1, 32, 4096, 128), 'float16'), buf_3: T.Buffer((1, 32, 4096, 128), 'float16')):
    c_4 = 0
    c_5 = 1
    c_6 = 32
    c_7 = 64
    c_8 = 128
    c_9 = 0.0
    c_10 = -3.4028234663852886e+38
    c_11 = 0.127517431974411
    with T.Kernel(c_6, c_6, c_5, threads=c_8) as (bx, by, bz):
        shared_12 = T.alloc_shared([128, 128], 'float16')
        frag_13 = T.alloc_fragment([128], 'float32')
        frag_14 = T.alloc_fragment([128, 128], 'float32')
        frag_15 = T.alloc_fragment([128], 'float32')
        shared_16 = T.alloc_shared([64, 128], 'float16')
        dot_17 = T.alloc_fragment([128, 64], 'float32')
        frag_18 = T.alloc_fragment([128, 64], 'float32')
        frag_19 = T.alloc_fragment([128, 64], 'float32')
        red_20 = T.alloc_fragment([128], 'float32')
        frag_21 = T.alloc_fragment([128], 'float32')
        frag_22 = T.alloc_fragment([128, 64], 'float32')
        exp_23 = T.alloc_fragment([128, 64], 'float32')
        red_24 = T.alloc_fragment([128], 'float32')
        frag_25 = T.alloc_fragment([128], 'float32')
        exp_26 = T.alloc_fragment([128], 'float32')
        frag_27 = T.alloc_fragment([128], 'float32')
        frag_28 = T.alloc_fragment([128], 'float32')
        cast_29 = T.alloc_fragment([128, 64], 'float16')
        shared_30 = T.alloc_shared([64, 128], 'float16')
        dot_31 = T.alloc_fragment([128, 128], 'float32')
        frag_32 = T.alloc_fragment([128, 128], 'float32')
        exp_33 = T.alloc_fragment([128, 128], 'float32')
        frag_34 = T.alloc_fragment([128, 128], 'float32')
        frag_35 = T.alloc_fragment([128, 128], 'float32')
        frag_36 = T.alloc_fragment([128, 128], 'float32')
        cast_37 = T.alloc_fragment([128, 128], 'float16')
        v_38 = bx * c_8
        T.copy(buf_0[c_4, by, v_38:v_38 + 128, c_4:c_4 + 128], shared_12)
        T.fill(frag_13, c_10)
        T.fill(frag_14, c_9)
        T.fill(frag_15, c_9)
        for j_39 in T.Pipelined(c_4, c_7, num_stages=2):
            v_40 = j_39 * c_7
            T.copy(buf_1[c_4, by, v_40:v_40 + 64, c_4:c_4 + 128], shared_16)
            T.gemm(shared_12, shared_16, dot_17, clear_accum=True, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.fill(frag_18, c_11)
            for i0_41, i1_42 in T.Parallel(128, 64):
                frag_19[i0_41, i1_42] = dot_17[i0_41, i1_42] * frag_18[i0_41, i1_42]
            T.reduce_max(frag_19, red_20, dim=1)
            for i0_43 in T.Parallel(128):
                frag_21[i0_43] = T.max(frag_13[i0_43], red_20[i0_43])
            for i0_44, i1_45 in T.Parallel(128, 64):
                frag_22[i0_44, i1_45] = frag_19[i0_44, i1_45] - frag_21[i0_44]
            for i0_46, i1_47 in T.Parallel(128, 64):
                exp_23[i0_46, i1_47] = T.exp2(frag_22[i0_46, i1_47])
            T.reduce_sum(exp_23, red_24, dim=1)
            for i0_48 in T.Parallel(128):
                frag_25[i0_48] = frag_13[i0_48] - frag_21[i0_48]
            for i0_49 in T.Parallel(128):
                exp_26[i0_49] = T.exp2(frag_25[i0_49])
            for i0_50 in T.Parallel(128):
                frag_27[i0_50] = frag_15[i0_50] * exp_26[i0_50]
            for i0_51 in T.Parallel(128):
                frag_28[i0_51] = frag_27[i0_51] + red_24[i0_51]
            for i0_52, i1_53 in T.Parallel(128, 64):
                cast_29[i0_52, i1_53] = T.cast(exp_23[i0_52, i1_53], 'float16')
            T.copy(buf_2[c_4, by, v_40:v_40 + 64, c_4:c_4 + 128], shared_30)
            T.gemm(cast_29, shared_30, dot_31, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            for i0_54, i1_55 in T.Parallel(128, 128):
                frag_32[i0_54, i1_55] = frag_13[i0_54] - frag_21[i0_54]
            for i0_56, i1_57 in T.Parallel(128, 128):
                exp_33[i0_56, i1_57] = T.exp2(frag_32[i0_56, i1_57])
            for i0_58, i1_59 in T.Parallel(128, 128):
                frag_34[i0_58, i1_59] = frag_14[i0_58, i1_59] * exp_33[i0_58, i1_59]
            for i0_60, i1_61 in T.Parallel(128, 128):
                frag_35[i0_60, i1_61] = frag_34[i0_60, i1_61] + dot_31[i0_60, i1_61]
            T.copy(frag_28, frag_15)
            T.copy(frag_35, frag_14)
            T.copy(frag_21, frag_13)
        for i0_62, i1_63 in T.Parallel(128, 128):
            frag_36[i0_62, i1_63] = frag_14[i0_62, i1_63] / frag_15[i0_62]
        for i0_64, i1_65 in T.Parallel(128, 128):
            cast_37[i0_64, i1_65] = T.cast(frag_36[i0_64, i1_65], 'float16')
        T.copy(cast_37, buf_3[c_4, by, v_38:v_38 + 128, c_4:c_4 + 128])
