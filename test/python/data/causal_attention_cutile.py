import cuda.tile as ct

@ct.kernel
def attention_kernel(arr_0, arr_1, arr_2, arr_3):
    c_4 = 2
    c_5 = 64
    c_6 = 128
    c_7 = 1
    c_8 = 0.1803368777036667
    c_9 = 16
    c_10 = 0
    c_11 = 0.0
    c_12 = -1e309
    bid_13 = ct.bid(0)
    bid_14 = ct.bid(1)
    v_15 = bid_14 * c_6
    tile_16 = ct.full((128,), c_12, dtype=ct.float32)
    tile_17 = ct.full((128,), c_11, dtype=ct.float32)
    tile_18 = ct.full((128, 64), c_11, dtype=ct.float32)
    v_19 = bid_14 * c_4
    v_20 = ct.maximum(v_19, c_10)
    v_21 = ct.minimum(v_20, c_9)
    acc_22 = tile_16
    acc_23 = tile_17
    acc_24 = tile_18
    for j_25 in range(c_10, v_21, c_7):
        v_26 = j_25 * c_5
        load_27 = ct.load(arr_0, (c_10, bid_13, v_15 // 128, c_10 // 64), (1, 1, 128, 64), order=(0, 1, 2, 3))
        tile_28 = ct.reshape(load_27, (128, 64))
        load_29 = ct.load(arr_1, (c_10, bid_13, c_10 // 64, v_26 // 64), (1, 1, 64, 64), order=(0, 1, 3, 2))
        tile_30 = ct.reshape(load_29, (64, 64))
        tile_31 = ct.mma(tile_28, tile_30, tile_18)
        tile_32 = ct.full((128, 64), c_8, dtype=ct.float32)
        v_33 = tile_31 * tile_32
        red_34 = ct.max(v_33, 1, keepdims=False)
        v_35 = ct.maximum(acc_22, red_34)
        bcast_36 = ct.broadcast_to(ct.expand_dims(v_35, axis=1), (128, 64))
        v_37 = v_33 - bcast_36
        v_38 = ct.exp2(v_37)
        v_39 = acc_22 - v_35
        v_40 = ct.exp2(v_39)
        v_41 = acc_23 * v_40
        red_42 = ct.sum(v_38, 1, keepdims=False)
        v_43 = v_41 + red_42
        v_44 = ct.astype(v_38, ct.float16)
        bcast_45 = ct.broadcast_to(ct.expand_dims(acc_22, axis=1), (128, 64))
        bcast_46 = ct.broadcast_to(ct.expand_dims(v_35, axis=1), (128, 64))
        v_47 = bcast_45 - bcast_46
        v_48 = ct.exp2(v_47)
        v_49 = acc_24 * v_48
        load_50 = ct.load(arr_2, (c_10, bid_13, v_26 // 64, c_10 // 64), (1, 1, 64, 64), order=(0, 1, 2, 3))
        tile_51 = ct.reshape(load_50, (64, 64))
        tile_52 = ct.mma(v_44, tile_51, v_49)
        acc_22 = v_35
        acc_23 = v_43
        acc_24 = tile_52
    v_53 = v_19 + c_4
    v_54 = ct.maximum(v_53, c_10)
    v_55 = ct.minimum(v_54, c_9)
    acc_56 = acc_22
    acc_57 = acc_23
    acc_58 = acc_24
    for j_59 in range(v_21, v_55, c_7):
        v_60 = j_59 * c_5
        load_61 = ct.load(arr_0, (c_10, bid_13, v_15 // 128, c_10 // 64), (1, 1, 128, 64), order=(0, 1, 2, 3))
        tile_62 = ct.reshape(load_61, (128, 64))
        load_63 = ct.load(arr_1, (c_10, bid_13, c_10 // 64, v_60 // 64), (1, 1, 64, 64), order=(0, 1, 3, 2))
        tile_64 = ct.reshape(load_63, (64, 64))
        tile_65 = ct.mma(tile_62, tile_64, tile_18)
        tile_66 = ct.full((128, 64), c_8, dtype=ct.float32)
        v_67 = tile_65 * tile_66
        range_68 = ct.arange(c_6 - c_10, dtype=ct.int64) + c_10
        bcast_69 = ct.broadcast_to(ct.expand_dims(range_68, axis=1), (128, 64))
        tile_70 = ct.full((128, 64), bid_14, dtype=ct.int64)
        tile_71 = ct.full((128, 64), c_6, dtype=ct.int64)
        v_72 = tile_70 * tile_71
        v_73 = v_72 + bcast_69
        range_74 = ct.arange(c_5 - c_10, dtype=ct.int64) + c_10
        bcast_75 = ct.broadcast_to(ct.expand_dims(range_74, axis=0), (128, 64))
        tile_76 = ct.full((128, 64), j_59, dtype=ct.int64)
        tile_77 = ct.full((128, 64), c_5, dtype=ct.int64)
        v_78 = tile_76 * tile_77
        v_79 = v_78 + bcast_75
        cmp_80 = v_79 <= v_73
        tile_81 = ct.full((128, 64), c_12, dtype=ct.float32)
        sel_82 = ct.where(cmp_80, v_67, tile_81)
        red_83 = ct.max(sel_82, 1, keepdims=False)
        v_84 = ct.maximum(acc_56, red_83)
        bcast_85 = ct.broadcast_to(ct.expand_dims(v_84, axis=1), (128, 64))
        v_86 = sel_82 - bcast_85
        v_87 = ct.exp2(v_86)
        v_88 = acc_56 - v_84
        v_89 = ct.exp2(v_88)
        v_90 = acc_57 * v_89
        red_91 = ct.sum(v_87, 1, keepdims=False)
        v_92 = v_90 + red_91
        v_93 = ct.astype(v_87, ct.float16)
        bcast_94 = ct.broadcast_to(ct.expand_dims(acc_56, axis=1), (128, 64))
        bcast_95 = ct.broadcast_to(ct.expand_dims(v_84, axis=1), (128, 64))
        v_96 = bcast_94 - bcast_95
        v_97 = ct.exp2(v_96)
        v_98 = acc_58 * v_97
        load_99 = ct.load(arr_2, (c_10, bid_13, v_60 // 64, c_10 // 64), (1, 1, 64, 64), order=(0, 1, 2, 3))
        tile_100 = ct.reshape(load_99, (64, 64))
        tile_101 = ct.mma(v_93, tile_100, v_98)
        acc_56 = v_84
        acc_57 = v_92
        acc_58 = tile_101
    bcast_102 = ct.broadcast_to(ct.expand_dims(acc_57, axis=1), (128, 64))
    v_103 = acc_58 / bcast_102
    v_104 = ct.astype(v_103, ct.float16)
    ct.store(arr_3, (c_10, bid_13, v_15 // 128, c_10 // 64), ct.reshape(v_104, (1, 1, 128, 64)))
