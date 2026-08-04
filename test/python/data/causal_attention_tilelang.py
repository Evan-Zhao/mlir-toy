import tilelang.language as T


@T.prim_func
def attention_kernel(
    buf_0: T.Tensor((1, 4, 1024, 64), "float16"),  # type: ignore
    buf_1: T.Tensor((1, 4, 1024, 64), "float16"),  # type: ignore
    buf_2: T.Tensor((1, 4, 1024, 64), "float16"),  # type: ignore
    buf_3: T.Tensor((1, 4, 1024, 64), "float16"),  # type: ignore
):
    with T.Kernel(4, 8, threads=128) as (bid_4, bid_5):
        frag_6 = T.alloc_fragment([128], "float32")
        frag_7 = T.alloc_fragment([128], "float32")
        frag_8 = T.alloc_fragment([128, 64], "float32")
        shared_9 = T.alloc_shared([128, 64], "float16")
        shared_10 = T.alloc_shared([64, 64], "float16")
        dot_11 = T.alloc_fragment([128, 64], "float32")
        frag_12 = T.alloc_fragment([128, 64], "float32")
        frag_13 = T.alloc_fragment([128, 64], "float32")
        red_14 = T.alloc_fragment([128], "float32")
        frag_15 = T.alloc_fragment([128], "float32")
        frag_16 = T.alloc_fragment([128, 64], "float32")
        exp_17 = T.alloc_fragment([128, 64], "float32")
        frag_18 = T.alloc_fragment([128], "float32")
        exp_19 = T.alloc_fragment([128], "float32")
        frag_20 = T.alloc_fragment([128], "float32")
        red_21 = T.alloc_fragment([128], "float32")
        frag_22 = T.alloc_fragment([128], "float32")
        cast_23 = T.alloc_fragment([128, 64], "float16")
        frag_24 = T.alloc_fragment([128, 64], "float32")
        exp_25 = T.alloc_fragment([128, 64], "float32")
        frag_26 = T.alloc_fragment([128, 64], "float32")
        shared_27 = T.alloc_shared([64, 64], "float16")
        dot_28 = T.alloc_fragment([128, 64], "float32")
        shared_29 = T.alloc_shared([128, 64], "float16")
        shared_30 = T.alloc_shared([64, 64], "float16")
        dot_31 = T.alloc_fragment([128, 64], "float32")
        frag_32 = T.alloc_fragment([128, 64], "float32")
        frag_33 = T.alloc_fragment([128, 64], "float32")
        frag_34 = T.alloc_fragment([128], "int64")
        frag_35 = T.alloc_fragment([128, 64], "int64")
        frag_36 = T.alloc_fragment([128, 64], "int64")
        frag_37 = T.alloc_fragment([128, 64], "int64")
        frag_38 = T.alloc_fragment([128, 64], "int64")
        frag_39 = T.alloc_fragment([64], "int64")
        frag_40 = T.alloc_fragment([128, 64], "int64")
        frag_41 = T.alloc_fragment([128, 64], "int64")
        frag_42 = T.alloc_fragment([128, 64], "int64")
        frag_43 = T.alloc_fragment([128, 64], "int64")
        frag_44 = T.alloc_fragment([128, 64], "int64")
        frag_45 = T.alloc_fragment([128, 64], "int64")
        frag_46 = T.alloc_fragment([128, 64], "bool")
        frag_47 = T.alloc_fragment([128, 64], "float32")
        frag_48 = T.alloc_fragment([128, 64], "float32")
        red_49 = T.alloc_fragment([128], "float32")
        frag_50 = T.alloc_fragment([128], "float32")
        frag_51 = T.alloc_fragment([128, 64], "float32")
        exp_52 = T.alloc_fragment([128, 64], "float32")
        frag_53 = T.alloc_fragment([128], "float32")
        exp_54 = T.alloc_fragment([128], "float32")
        frag_55 = T.alloc_fragment([128], "float32")
        red_56 = T.alloc_fragment([128], "float32")
        frag_57 = T.alloc_fragment([128], "float32")
        cast_58 = T.alloc_fragment([128, 64], "float16")
        frag_59 = T.alloc_fragment([128, 64], "float32")
        exp_60 = T.alloc_fragment([128, 64], "float32")
        frag_61 = T.alloc_fragment([128, 64], "float32")
        shared_62 = T.alloc_shared([64, 64], "float16")
        dot_63 = T.alloc_fragment([128, 64], "float32")
        frag_64 = T.alloc_fragment([128, 64], "float32")
        cast_65 = T.alloc_fragment([128, 64], "float16")
        c_66 = 2
        c_67 = 64
        c_68 = 128
        c_69 = 1
        c_70 = 0.1803368777036667
        c_71 = 16
        c_72 = 0
        c_73 = 0.0
        c_74 = -1e309
        v_75 = bid_5 * c_68
        T.fill(frag_6, c_74)
        T.fill(frag_7, c_73)
        T.fill(frag_8, c_73)
        v_76 = bid_5 * c_66
        v_77 = T.max(v_76, c_72)
        v_78 = T.min(v_77, c_71)
        for j_79 in T.serial(c_72, v_78, c_69):
            v_80 = j_79 * c_67
            T.copy(buf_0[c_72, bid_4, v_75 : v_75 + 128, c_72 : c_72 + 64], shared_9)
            T.copy(buf_1[c_72, bid_4, v_80 : v_80 + 64, c_72 : c_72 + 64], shared_10)
            T.gemm(
                shared_9,
                shared_10,
                dot_11,
                clear_accum=True,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.fill(frag_12, c_70)
            for i0_81, i1_82 in T.Parallel(128, 64):
                frag_13[i0_81, i1_82] = dot_11[i0_81, i1_82] * frag_12[i0_81, i1_82]
            T.reduce_max(frag_13, red_14, dim=1)
            for i0_83 in T.Parallel(128):
                frag_15[i0_83] = T.max(frag_6[i0_83], red_14[i0_83])
            for i0_84, i1_85 in T.Parallel(128, 64):
                frag_16[i0_84, i1_85] = frag_13[i0_84, i1_85] - frag_15[i0_84]
            for i0_86, i1_87 in T.Parallel(128, 64):
                exp_17[i0_86, i1_87] = T.exp2(frag_16[i0_86, i1_87])
            for i0_88 in T.Parallel(128):
                frag_18[i0_88] = frag_6[i0_88] - frag_15[i0_88]
            for i0_89 in T.Parallel(128):
                exp_19[i0_89] = T.exp2(frag_18[i0_89])
            for i0_90 in T.Parallel(128):
                frag_20[i0_90] = frag_7[i0_90] * exp_19[i0_90]
            T.reduce_sum(exp_17, red_21, dim=1)
            for i0_91 in T.Parallel(128):
                frag_22[i0_91] = frag_20[i0_91] + red_21[i0_91]
            for i0_92, i1_93 in T.Parallel(128, 64):
                cast_23[i0_92, i1_93] = T.cast(exp_17[i0_92, i1_93], "float16")
            for i0_94, i1_95 in T.Parallel(128, 64):
                frag_24[i0_94, i1_95] = frag_6[i0_94] - frag_15[i0_94]
            for i0_96, i1_97 in T.Parallel(128, 64):
                exp_25[i0_96, i1_97] = T.exp2(frag_24[i0_96, i1_97])
            for i0_98, i1_99 in T.Parallel(128, 64):
                frag_26[i0_98, i1_99] = frag_8[i0_98, i1_99] * exp_25[i0_98, i1_99]
            T.copy(buf_2[c_72, bid_4, v_80 : v_80 + 64, c_72 : c_72 + 64], shared_27)
            T.copy(frag_26, dot_28)
            T.gemm(cast_23, shared_27, dot_28, clear_accum=False, policy=T.GemmWarpPolicy.FullRow)
            T.copy(frag_15, frag_6)
            T.copy(frag_22, frag_7)
            T.copy(dot_28, frag_8)
        v_100 = v_76 + c_66
        v_101 = T.max(v_100, c_72)
        v_102 = T.min(v_101, c_71)
        for j_103 in T.serial(v_78, v_102, c_69):
            v_104 = j_103 * c_67
            T.copy(buf_0[c_72, bid_4, v_75 : v_75 + 128, c_72 : c_72 + 64], shared_29)
            T.copy(buf_1[c_72, bid_4, v_104 : v_104 + 64, c_72 : c_72 + 64], shared_30)
            T.gemm(
                shared_29,
                shared_30,
                dot_31,
                clear_accum=True,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.fill(frag_32, c_70)
            for i0_105, i1_106 in T.Parallel(128, 64):
                frag_33[i0_105, i1_106] = dot_31[i0_105, i1_106] * frag_32[i0_105, i1_106]
            for i0_107 in T.Parallel(128):
                frag_34[i0_107] = c_72 + i0_107
            T.fill(frag_35, bid_5)
            T.fill(frag_36, c_68)
            for i0_108, i1_109 in T.Parallel(128, 64):
                frag_37[i0_108, i1_109] = frag_35[i0_108, i1_109] * frag_36[i0_108, i1_109]
            for i0_110, i1_111 in T.Parallel(128, 64):
                frag_38[i0_110, i1_111] = frag_37[i0_110, i1_111] + frag_34[i0_110]
            for i0_112 in T.Parallel(64):
                frag_39[i0_112] = c_72 + i0_112
            T.fill(frag_40, j_103)
            T.fill(frag_41, c_67)
            for i0_113, i1_114 in T.Parallel(128, 64):
                frag_42[i0_113, i1_114] = frag_40[i0_113, i1_114] * frag_41[i0_113, i1_114]
            for i0_115, i1_116 in T.Parallel(128, 64):
                frag_43[i0_115, i1_116] = frag_42[i0_115, i1_116] + frag_39[i1_116]
            for i0_117, i1_118 in T.Parallel(128, 64):
                frag_44[i0_117, i1_118] = T.cast(frag_43[i0_117, i1_118], "int64")
            for i0_119, i1_120 in T.Parallel(128, 64):
                frag_45[i0_119, i1_120] = T.cast(frag_38[i0_119, i1_120], "int64")
            for i0_121, i1_122 in T.Parallel(128, 64):
                frag_46[i0_121, i1_122] = frag_44[i0_121, i1_122] <= frag_45[i0_121, i1_122]
            T.fill(frag_47, c_74)
            for i0_123, i1_124 in T.Parallel(128, 64):
                frag_48[i0_123, i1_124] = T.if_then_else(
                    frag_46[i0_123, i1_124], frag_33[i0_123, i1_124], frag_47[i0_123, i1_124]
                )
            T.reduce_max(frag_48, red_49, dim=1)
            for i0_125 in T.Parallel(128):
                frag_50[i0_125] = T.max(frag_6[i0_125], red_49[i0_125])
            for i0_126, i1_127 in T.Parallel(128, 64):
                frag_51[i0_126, i1_127] = frag_48[i0_126, i1_127] - frag_50[i0_126]
            for i0_128, i1_129 in T.Parallel(128, 64):
                exp_52[i0_128, i1_129] = T.exp2(frag_51[i0_128, i1_129])
            for i0_130 in T.Parallel(128):
                frag_53[i0_130] = frag_6[i0_130] - frag_50[i0_130]
            for i0_131 in T.Parallel(128):
                exp_54[i0_131] = T.exp2(frag_53[i0_131])
            for i0_132 in T.Parallel(128):
                frag_55[i0_132] = frag_7[i0_132] * exp_54[i0_132]
            T.reduce_sum(exp_52, red_56, dim=1)
            for i0_133 in T.Parallel(128):
                frag_57[i0_133] = frag_55[i0_133] + red_56[i0_133]
            for i0_134, i1_135 in T.Parallel(128, 64):
                cast_58[i0_134, i1_135] = T.cast(exp_52[i0_134, i1_135], "float16")
            for i0_136, i1_137 in T.Parallel(128, 64):
                frag_59[i0_136, i1_137] = frag_6[i0_136] - frag_50[i0_136]
            for i0_138, i1_139 in T.Parallel(128, 64):
                exp_60[i0_138, i1_139] = T.exp2(frag_59[i0_138, i1_139])
            for i0_140, i1_141 in T.Parallel(128, 64):
                frag_61[i0_140, i1_141] = frag_8[i0_140, i1_141] * exp_60[i0_140, i1_141]
            T.copy(buf_2[c_72, bid_4, v_104 : v_104 + 64, c_72 : c_72 + 64], shared_62)
            T.copy(frag_61, dot_63)
            T.gemm(cast_58, shared_62, dot_63, clear_accum=False, policy=T.GemmWarpPolicy.FullRow)
            T.copy(frag_50, frag_6)
            T.copy(frag_57, frag_7)
            T.copy(dot_63, frag_8)
        for i0_142, i1_143 in T.Parallel(128, 64):
            frag_64[i0_142, i1_143] = frag_8[i0_142, i1_143] / frag_7[i0_142]
        for i0_144, i1_145 in T.Parallel(128, 64):
            cast_65[i0_144, i1_145] = T.cast(frag_64[i0_144, i1_145], "float16")
        T.copy(cast_65, buf_3[c_72, bid_4, v_75 : v_75 + 128, c_72 : c_72 + 64])
