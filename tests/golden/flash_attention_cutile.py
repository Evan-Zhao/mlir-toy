import cuda.tile as ct

@ct.kernel
def flash_attention_htile(arr_0, arr_1, arr_2, arr_3):
    c_4 = 0
    c_5 = 1
    c_6 = 32
    c_7 = 64
    c_8 = 128
    c_9 = 0.0
    c_10 = -3.4028234663852886e+38
    c_11 = 0.127517431974411
    bid_m = ct.bid(0)
    bid_h = ct.bid(1)
    bid_b = ct.bid(2)
    v_12 = bid_m * c_8
    load_13 = ct.load(arr_0, (c_4, bid_h, v_12 // 128, c_4 // 128), (1, 1, 128, 128), order=(0, 1, 2, 3))
    tile_14 = ct.reshape(load_13, (128, 128))
    tile_15 = ct.full((128,), c_10, dtype=ct.float32)
    tile_16 = ct.full((128, 128), c_9, dtype=ct.float32)
    tile_17 = ct.full((128,), c_9, dtype=ct.float32)
    acc_18 = tile_17
    acc_19 = tile_16
    acc_20 = tile_15
    for j_21 in range(c_4, c_7, c_5):
        v_22 = j_21 * c_7
        load_23 = ct.load(arr_1, (c_4, bid_h, c_4 // 128, v_22 // 64), (1, 1, 128, 64), order=(0, 1, 3, 2))
        tile_24 = ct.reshape(load_23, (128, 64))
        tile_25 = ct.mma(tile_14, tile_24, ct.full((128, 64), 0, dtype=ct.float32))
        tile_26 = ct.full((128, 64), c_11, dtype=ct.float32)
        v_27 = tile_25 * tile_26
        red_28 = ct.max(v_27, 1, keepdims=False)
        v_29 = ct.maximum(acc_20, red_28)
        bcast_30 = ct.broadcast_to(ct.expand_dims(v_29, axis=1), (128, 64))
        v_31 = v_27 - bcast_30
        v_32 = ct.exp2(v_31)
        red_33 = ct.sum(v_32, 1, keepdims=False)
        v_34 = acc_20 - v_29
        v_35 = ct.exp2(v_34)
        v_36 = acc_18 * v_35
        v_37 = v_36 + red_33
        v_38 = ct.astype(v_32, ct.float16)
        load_39 = ct.load(arr_2, (c_4, bid_h, v_22 // 64, c_4 // 128), (1, 1, 64, 128), order=(0, 1, 2, 3))
        tile_40 = ct.reshape(load_39, (64, 128))
        tile_41 = ct.mma(v_38, tile_40, ct.full((128, 128), 0, dtype=ct.float32))
        bcast_42 = ct.broadcast_to(ct.expand_dims(acc_20, axis=1), (128, 128))
        bcast_43 = ct.broadcast_to(ct.expand_dims(v_29, axis=1), (128, 128))
        v_44 = bcast_42 - bcast_43
        v_45 = ct.exp2(v_44)
        v_46 = acc_19 * v_45
        v_47 = v_46 + tile_41
        acc_18 = v_37
        acc_19 = v_47
        acc_20 = v_29
    bcast_48 = ct.broadcast_to(ct.expand_dims(acc_18, axis=1), (128, 128))
    v_49 = acc_19 / bcast_48
    v_50 = ct.astype(v_49, ct.float16)
    ct.store(arr_3, (c_4, bid_h, v_12 // 128, c_4 // 128), ct.reshape(v_50, (1, 1, 128, 128)))
