import triton
from triton import language as tl

@triton.jit
def kernel(ptr_0, ptr_1, ptr_2, ptr_3):
    c_4 = 0
    c_5 = 1
    c_6 = 32
    c_7 = 64
    c_8 = 128
    c_9 = 0.0
    c_10 = -3.40282347e+38
    c_11 = 0.127517432
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    v_12 = pid_m * c_8
    ptr_13 = ptr_0 + (0 + pid_b * 16777216 + pid_h * 524288)
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
        ptr_24 = ptr_1 + (0 + pid_b * 16777216 + pid_h * 524288)
        bp_25 = tl.make_block_ptr(base=ptr_24, shape=[4096, 128], strides=[128, 1], offsets=[v_23, c_4], block_shape=[64, 128], order=[1, 0])
        tile_26 = tl.load(bp_25, boundary_check=[0, 1])
        tile_27 = tl.permute(tile_26, [1, 0])
        tile_28 = tl.dot(tile_15, tile_27)
        tile_29 = tl.full([128, 64], c_11, tl.float32)
        v_30 = tile_28 * tile_29
        red_31 = tl.max(v_30, 1, keep_dims=False)
        v_32 = tl.maximum(acc_21, red_31)
        bcast_33 = v_32[:, None]
        v_34 = v_30 - bcast_33
        v_35 = tl.exp2(v_34)
        red_36 = tl.sum(v_35, 1, keep_dims=False)
        v_37 = acc_21 - v_32
        v_38 = tl.exp2(v_37)
        v_39 = acc_19 * v_38
        v_40 = v_39 + red_36
        v_41 = tl.cast(v_35, tl.float16)
        ptr_42 = ptr_2 + (0 + pid_b * 16777216 + pid_h * 524288)
        bp_43 = tl.make_block_ptr(base=ptr_42, shape=[4096, 128], strides=[128, 1], offsets=[v_23, c_4], block_shape=[64, 128], order=[1, 0])
        tile_44 = tl.load(bp_43, boundary_check=[0, 1])
        tile_45 = tl.dot(v_41, tile_44)
        bcast_46 = acc_21[:, None]
        bcast_47 = v_32[:, None]
        v_48 = bcast_46 - bcast_47
        v_49 = tl.exp2(v_48)
        v_50 = acc_20 * v_49
        v_51 = v_50 + tile_45
        acc_19 = v_40
        acc_20 = v_51
        acc_21 = v_32
    bcast_52 = acc_19[:, None]
    v_53 = acc_20 / bcast_52
    v_54 = tl.cast(v_53, tl.float16)
    ptr_55 = ptr_3 + (0 + pid_b * 16777216 + pid_h * 524288)
    bp_56 = tl.make_block_ptr(base=ptr_55, shape=[4096, 128], strides=[128, 1], offsets=[v_12, c_4], block_shape=[128, 128], order=[1, 0])
    tl.store(bp_56, v_54, boundary_check=[0, 1])
