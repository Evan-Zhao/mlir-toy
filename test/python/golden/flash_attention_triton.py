import triton
from triton import language as tl

@triton.jit
def flash_attention_htile(ptr_0, ptr_1, ptr_2, ptr_3):
    c_4 = 0
    c_5 = 1
    c_6 = 32
    c_7 = 64
    c_8 = 128
    c_9 = 0.0
    c_10 = -3.4028234663852886e+38
    c_11 = 0.127517431974411
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    v_12 = pid_m * c_8
    ptr_13 = ptr_0 + (0 + c_4 * 16777216 + pid_h * 524288)
    bp_14 = tl.make_block_ptr(base=ptr_13, shape=[4096, 128], strides=[128, 1], offsets=[v_12, c_4], block_shape=[128, 128], order=[1, 0])
    tile_15 = tl.load(bp_14, boundary_check=[0, 1])
    tile_16 = tl.full([128], c_10, tl.float32)
    tile_17 = tl.full([128, 128], c_9, tl.float32)
    tile_18 = tl.full([128], c_9, tl.float32)
    acc_19 = tile_18
    acc_20 = tile_17
    acc_21 = tile_16
    for j_22 in range(c_4, c_7, c_5):
        v_23 = j_22 * c_7
        ptr_24 = ptr_1 + (0 + c_4 * 16777216 + pid_h * 524288)
        bp_25 = tl.make_block_ptr(base=ptr_24, shape=[128, 4096], strides=[1, 128], offsets=[c_4, v_23], block_shape=[128, 64], order=[0, 1])
        tile_26 = tl.load(bp_25, boundary_check=[0, 1])
        tile_27 = tl.dot(tile_15, tile_26)
        tile_28 = tl.full([128, 64], c_11, tl.float32)
        v_29 = tile_27 * tile_28
        red_30 = tl.max(v_29, 1, keep_dims=False)
        v_31 = tl.maximum(acc_21, red_30)
        bcast_32 = v_31[:, None]
        v_33 = v_29 - bcast_32
        v_34 = tl.exp2(v_33)
        red_35 = tl.sum(v_34, 1, keep_dims=False)
        v_36 = acc_21 - v_31
        v_37 = tl.exp2(v_36)
        v_38 = acc_19 * v_37
        v_39 = v_38 + red_35
        v_40 = tl.cast(v_34, tl.float16)
        ptr_41 = ptr_2 + (0 + c_4 * 16777216 + pid_h * 524288)
        bp_42 = tl.make_block_ptr(base=ptr_41, shape=[4096, 128], strides=[128, 1], offsets=[v_23, c_4], block_shape=[64, 128], order=[1, 0])
        tile_43 = tl.load(bp_42, boundary_check=[0, 1])
        tile_44 = tl.dot(v_40, tile_43)
        bcast_45 = acc_21[:, None]
        bcast_46 = v_31[:, None]
        v_47 = bcast_45 - bcast_46
        v_48 = tl.exp2(v_47)
        v_49 = acc_20 * v_48
        v_50 = v_49 + tile_44
        acc_19 = v_39
        acc_20 = v_50
        acc_21 = v_31
    bcast_51 = acc_19[:, None]
    v_52 = acc_20 / bcast_51
    v_53 = tl.cast(v_52, tl.float16)
    ptr_54 = ptr_3 + (0 + c_4 * 16777216 + pid_h * 524288)
    bp_55 = tl.make_block_ptr(base=ptr_54, shape=[4096, 128], strides=[128, 1], offsets=[v_12, c_4], block_shape=[128, 128], order=[1, 0])
    tl.store(bp_55, v_53, boundary_check=[0, 1])
