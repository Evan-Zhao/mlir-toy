import triton
from triton import language as tl


@triton.jit
def attention_kernel(ptr_0, ptr_1, ptr_2, ptr_3):
    c_4: tl.constexpr = 2
    c_5: tl.constexpr = 64
    c_6: tl.constexpr = 128
    c_7: tl.constexpr = 1
    c_8: tl.constexpr = 0.1803368777036667
    c_9: tl.constexpr = 16
    c_10: tl.constexpr = 0
    c_11: tl.constexpr = 0.0
    c_12: tl.constexpr = -1e309
    pid_13 = tl.program_id(0)
    pid_14 = tl.program_id(1)
    v_15 = pid_14 * c_6
    tile_16 = tl.full((128,), c_12, tl.float32)
    tile_17 = tl.full((128,), c_11, tl.float32)
    tile_18 = tl.full((128, 64), c_11, tl.float32)
    v_19 = pid_14 * c_4
    v_20 = max(v_19, c_10)
    v_21 = min(v_20, c_9)
    acc_22 = tile_16
    acc_23 = tile_17
    acc_24 = tile_18
    for j_25 in range(c_10, v_21, c_7):
        v_26 = j_25 * c_5
        ptr_27 = ptr_0 + (0 + c_10 * 262144 + pid_13 * 65536)
        bp_28 = tl.make_block_ptr(
            base=ptr_27,
            shape=[1024, 64],
            strides=[64, 1],
            offsets=[v_15, c_10],
            block_shape=[128, 64],
            order=[1, 0],
        )
        tile_29 = tl.load(bp_28)
        ptr_30 = ptr_1 + (0 + c_10 * 262144 + pid_13 * 65536)
        bp_31 = tl.make_block_ptr(
            base=ptr_30,
            shape=[64, 1024],
            strides=[1, 64],
            offsets=[c_10, v_26],
            block_shape=[64, 64],
            order=[0, 1],
        )
        tile_32 = tl.load(bp_31)
        tile_33 = tl.dot(tile_29, tile_32, tile_18)
        tile_34 = tl.full((128, 64), c_8, tl.float32)
        v_35 = tile_33 * tile_34
        red_36 = tl.max(v_35, 1, keep_dims=False)
        v_37 = tl.maximum(acc_22, red_36)
        bcast_38 = v_37[:, None]
        v_39 = v_35 - bcast_38
        v_40 = tl.exp2(v_39)
        v_41 = acc_22 - v_37
        v_42 = tl.exp2(v_41)
        v_43 = acc_23 * v_42
        red_44 = tl.sum(v_40, 1, keep_dims=False)
        v_45 = v_43 + red_44
        v_46 = tl.cast(v_40, tl.float16)
        bcast_47 = acc_22[:, None]
        bcast_48 = v_37[:, None]
        v_49 = bcast_47 - bcast_48
        v_50 = tl.exp2(v_49)
        v_51 = acc_24 * v_50
        ptr_52 = ptr_2 + (0 + c_10 * 262144 + pid_13 * 65536)
        bp_53 = tl.make_block_ptr(
            base=ptr_52,
            shape=[1024, 64],
            strides=[64, 1],
            offsets=[v_26, c_10],
            block_shape=[64, 64],
            order=[1, 0],
        )
        tile_54 = tl.load(bp_53)
        tile_55 = tl.dot(v_46, tile_54, v_51)
        acc_22 = v_37
        acc_23 = v_45
        acc_24 = tile_55
    v_56 = v_19 + c_4
    v_57 = max(v_56, c_10)
    v_58 = min(v_57, c_9)
    acc_59 = acc_22
    acc_60 = acc_23
    acc_61 = acc_24
    for j_62 in range(v_21, v_58, c_7):
        v_63 = j_62 * c_5
        ptr_64 = ptr_0 + (0 + c_10 * 262144 + pid_13 * 65536)
        bp_65 = tl.make_block_ptr(
            base=ptr_64,
            shape=[1024, 64],
            strides=[64, 1],
            offsets=[v_15, c_10],
            block_shape=[128, 64],
            order=[1, 0],
        )
        tile_66 = tl.load(bp_65)
        ptr_67 = ptr_1 + (0 + c_10 * 262144 + pid_13 * 65536)
        bp_68 = tl.make_block_ptr(
            base=ptr_67,
            shape=[64, 1024],
            strides=[1, 64],
            offsets=[c_10, v_63],
            block_shape=[64, 64],
            order=[0, 1],
        )
        tile_69 = tl.load(bp_68)
        tile_70 = tl.dot(tile_66, tile_69, tile_18)
        tile_71 = tl.full((128, 64), c_8, tl.float32)
        v_72 = tile_70 * tile_71
        range_73 = tl.arange(c_10, c_6)
        bcast_74 = range_73[:, None]
        tile_75 = tl.full((128, 64), pid_14, tl.int64)
        tile_76 = tl.full((128, 64), c_6, tl.int64)
        v_77 = tile_75 * tile_76
        v_78 = v_77 + bcast_74
        range_79 = tl.arange(c_10, c_5)
        bcast_80 = range_79[None, :]
        tile_81 = tl.full((128, 64), j_62, tl.int64)
        tile_82 = tl.full((128, 64), c_5, tl.int64)
        v_83 = tile_81 * tile_82
        v_84 = v_83 + bcast_80
        cmp_85 = v_84 <= v_78
        tile_86 = tl.full((128, 64), c_12, tl.float32)
        sel_87 = tl.where(cmp_85, v_72, tile_86)
        red_88 = tl.max(sel_87, 1, keep_dims=False)
        v_89 = tl.maximum(acc_59, red_88)
        bcast_90 = v_89[:, None]
        v_91 = sel_87 - bcast_90
        v_92 = tl.exp2(v_91)
        v_93 = acc_59 - v_89
        v_94 = tl.exp2(v_93)
        v_95 = acc_60 * v_94
        red_96 = tl.sum(v_92, 1, keep_dims=False)
        v_97 = v_95 + red_96
        v_98 = tl.cast(v_92, tl.float16)
        bcast_99 = acc_59[:, None]
        bcast_100 = v_89[:, None]
        v_101 = bcast_99 - bcast_100
        v_102 = tl.exp2(v_101)
        v_103 = acc_61 * v_102
        ptr_104 = ptr_2 + (0 + c_10 * 262144 + pid_13 * 65536)
        bp_105 = tl.make_block_ptr(
            base=ptr_104,
            shape=[1024, 64],
            strides=[64, 1],
            offsets=[v_63, c_10],
            block_shape=[64, 64],
            order=[1, 0],
        )
        tile_106 = tl.load(bp_105)
        tile_107 = tl.dot(v_98, tile_106, v_103)
        acc_59 = v_89
        acc_60 = v_97
        acc_61 = tile_107
    bcast_108 = acc_60[:, None]
    v_109 = acc_61 / bcast_108
    v_110 = tl.cast(v_109, tl.float16)
    ptr_111 = ptr_3 + (0 + c_10 * 262144 + pid_13 * 65536)
    bp_112 = tl.make_block_ptr(
        base=ptr_111,
        shape=[1024, 64],
        strides=[64, 1],
        offsets=[v_15, c_10],
        block_shape=[128, 64],
        order=[1, 0],
    )
    tl.store(bp_112, v_110)
