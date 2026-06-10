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
    ptr_13 = ptr_0 + (0 + pid_b * 16777216 + pid_h * 524288)
    bp_14 = tl.make_block_ptr(base=ptr_13, shape=[4096, 128], strides=[128, 1], offsets=[v_12, c_4], block_shape=[128, 128], order=[1, 0])
    tile_15 = tl.load(bp_14, boundary_check=[0, 1])
    acc_16 = c_9
    acc_17 = c_9
    acc_18 = c_10
    for j_19 in range(c_4, c_7, c_5):
        v_20 = j_19 * c_7
        ptr_21 = ptr_1 + (0 + pid_b * 16777216 + pid_h * 524288)
        bp_22 = tl.make_block_ptr(base=ptr_21, shape=[128, 4096], strides=[1, 128], offsets=[c_4, v_20], block_shape=[64, 128], order=[0, 1])
        tile_23 = tl.load(bp_22, boundary_check=[0, 1])
        tile_24 = tl.dot(tile_15, tile_23)
        v_25 = tile_24 * c_11
        red_26 = tl.max(v_25, 1, keep_dims=False)
        v_27 = tl.maximum(acc_18, red_26)
        bcast_28 = v_27[:, None]
        v_29 = v_25 - bcast_28
        v_30 = tl.exp2(v_29)
        red_31 = tl.sum(v_30, 1, keep_dims=False)
        v_32 = acc_18 - v_27
        v_33 = tl.exp2(v_32)
        v_34 = acc_16 * v_33
        v_35 = v_34 + red_31
        v_36 = tl.cast(v_30, tl.float16)
        ptr_37 = ptr_2 + (0 + pid_b * 16777216 + pid_h * 524288)
        bp_38 = tl.make_block_ptr(base=ptr_37, shape=[4096, 128], strides=[128, 1], offsets=[v_20, c_4], block_shape=[64, 128], order=[1, 0])
        tile_39 = tl.load(bp_38, boundary_check=[0, 1])
        tile_40 = tl.dot(v_36, tile_39)
        bcast_41 = acc_18[:, None]
        bcast_42 = v_27[:, None]
        v_43 = bcast_41 - bcast_42
        v_44 = tl.exp2(v_43)
        v_45 = acc_17 * v_44
        v_46 = v_45 + tile_40
        acc_16 = v_35
        acc_17 = v_46
        acc_18 = v_27
    bcast_47 = acc_16[:, None]
    v_48 = acc_17 / bcast_47
    v_49 = tl.cast(v_48, tl.float16)
    ptr_50 = ptr_3 + (0 + pid_b * 16777216 + pid_h * 524288)
    bp_51 = tl.make_block_ptr(base=ptr_50, shape=[4096, 128], strides=[128, 1], offsets=[v_12, c_4], block_shape=[128, 128], order=[1, 0])
    tl.store(bp_51, v_49, boundary_check=[0, 1])
