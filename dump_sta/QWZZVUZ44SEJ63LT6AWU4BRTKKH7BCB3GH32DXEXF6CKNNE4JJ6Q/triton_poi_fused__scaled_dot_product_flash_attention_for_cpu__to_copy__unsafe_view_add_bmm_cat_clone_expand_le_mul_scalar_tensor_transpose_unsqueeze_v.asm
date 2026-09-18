	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8 # -- Begin function triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8
	.p2align	2
	.type	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8,@function
triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8: # @triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294097424.py"
	.loc	1 2 0                           # k135114294097424.py:2:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -2032
	.cfi_def_cfa_offset 2032
	sd	ra, 2024(sp)                    # 8-byte Folded Spill
	sd	s0, 2016(sp)                    # 8-byte Folded Spill
	sd	s2, 2008(sp)                    # 8-byte Folded Spill
	sd	s3, 2000(sp)                    # 8-byte Folded Spill
	sd	s4, 1992(sp)                    # 8-byte Folded Spill
	sd	s5, 1984(sp)                    # 8-byte Folded Spill
	sd	s6, 1976(sp)                    # 8-byte Folded Spill
	sd	s7, 1968(sp)                    # 8-byte Folded Spill
	sd	s8, 1960(sp)                    # 8-byte Folded Spill
	sd	s9, 1952(sp)                    # 8-byte Folded Spill
	sd	s10, 1944(sp)                   # 8-byte Folded Spill
	sd	s11, 1936(sp)                   # 8-byte Folded Spill
	fsd	fs0, 1928(sp)                   # 8-byte Folded Spill
	fsd	fs1, 1920(sp)                   # 8-byte Folded Spill
	fsd	fs2, 1912(sp)                   # 8-byte Folded Spill
	fsd	fs3, 1904(sp)                   # 8-byte Folded Spill
	fsd	fs4, 1896(sp)                   # 8-byte Folded Spill
	fsd	fs5, 1888(sp)                   # 8-byte Folded Spill
	fsd	fs6, 1880(sp)                   # 8-byte Folded Spill
	fsd	fs7, 1872(sp)                   # 8-byte Folded Spill
	fsd	fs8, 1864(sp)                   # 8-byte Folded Spill
	fsd	fs9, 1856(sp)                   # 8-byte Folded Spill
	fsd	fs10, 1848(sp)                  # 8-byte Folded Spill
	fsd	fs11, 1840(sp)                  # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s2, -24
	.cfi_offset s3, -32
	.cfi_offset s4, -40
	.cfi_offset s5, -48
	.cfi_offset s6, -56
	.cfi_offset s7, -64
	.cfi_offset s8, -72
	.cfi_offset s9, -80
	.cfi_offset s10, -88
	.cfi_offset s11, -96
	.cfi_offset fs0, -104
	.cfi_offset fs1, -112
	.cfi_offset fs2, -120
	.cfi_offset fs3, -128
	.cfi_offset fs4, -136
	.cfi_offset fs5, -144
	.cfi_offset fs6, -152
	.cfi_offset fs7, -160
	.cfi_offset fs8, -168
	.cfi_offset fs9, -176
	.cfi_offset fs10, -184
	.cfi_offset fs11, -192
	addi	s0, sp, 2032
	.cfi_def_cfa s0, 0
	lui	a3, 3
	addi	a3, a3, 1168
	sub	sp, sp, a3
	csrr	a3, vlenb
	li	a5, 200
	mul	a3, a3, a5
	sub	sp, sp, a3
	andi	sp, sp, -128
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114294097424.py:4:33
	slliw	t2, a4, 7
	li	a3, 32
	lui	a4, 599186
	li	a6, -64
	.loc	1 5 23                          # k135114294097424.py:5:23
	vsetvli	zero, a3, e32, m8, ta, ma
	vid.v	v8
	addi	a7, a4, 1171
	vor.vx	v0, v8, t2
	.loc	1 8 19                          # k135114294097424.py:8:19
	vmulh.vx	v16, v0, a7
	vadd.vv	v16, v16, v0
	vsra.vi	v16, v16, 9
	vsrl.vi	v24, v16, 31
	vadd.vv	v8, v16, v24
	.loc	1 5 23                          # k135114294097424.py:5:23
	vmv.v.x	v16, t2
	.loc	1 10 21                         # k135114294097424.py:10:21
	vsra.vi	v16, v16, 31
	csrr	a4, vlenb
	li	a5, 184
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 4
	addi	a5, a5, -1088
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	vsrl.vi	v16, v16, 26
	csrr	a4, vlenb
	li	a5, 152
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 4
	addi	a5, a5, -1088
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	vadd.vv	v24, v0, v16
	vmv.v.v	v16, v0
	csrr	a4, vlenb
	li	a5, 168
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 4
	addi	a5, a5, -1088
	add	a4, a4, a5
	vs8r.v	v0, (a4)                        # vscale x 64-byte Folded Spill
	.loc	1 9 19                          # k135114294097424.py:9:19
	vand.vx	v0, v24, a6
	vsub.vv	v0, v16, v0
	csrr	a4, vlenb
	li	a5, 72
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 4
	addi	a5, a5, -1088
	add	a4, a4, a5
	vs8r.v	v8, (a4)                        # vscale x 64-byte Folded Spill
	.loc	1 13 38                         # k135114294097424.py:13:38
	vsll.vi	v8, v8, 6
	.loc	1 13 35 is_stmt 0               # k135114294097424.py:13:35
	vadd.vv	v16, v8, v0
	.loc	1 10 21 is_stmt 1               # k135114294097424.py:10:21
	vsra.vi	v8, v24, 6
	.loc	1 10 27 is_stmt 0               # k135114294097424.py:10:27
	vmulh.vx	v24, v8, a7
	vadd.vv	v24, v24, v8
	vsra.vi	v24, v24, 3
	vsrl.vi	v0, v24, 31
	vadd.vv	v24, v24, v0
	li	a5, 14
	vnmsub.vx	v24, a5, v8
	li	a4, 896
	.loc	1 13 43 is_stmt 1               # k135114294097424.py:13:43
	vmacc.vx	v16, a4, v24
	csrr	t0, vlenb
	li	t1, 96
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 4
	addi	t1, t1, -1088
	add	t0, t0, t1
	vs8r.v	v16, (t0)                       # vscale x 64-byte Folded Spill
	li	t0, 96
	li	t1, 64
	lui	t3, 3
	li	t4, -32
	addi	t3, t3, 256
	vid.v	v24
	.loc	1 5 23                          # k135114294097424.py:5:23
	vadd.vx	v8, v24, t0
	vadd.vx	v16, v24, t1
	vadd.vx	v0, v24, a3
	vor.vx	v24, v8, t2
	vor.vx	v16, v16, t2
	vor.vx	v8, v0, t2
	csrr	t0, vlenb
	li	t5, 192
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t5, 184
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 12 36                         # k135114294097424.py:12:36
	vsrl.vi	v8, v8, 27
	vadd.vv	v0, v24, v8
	csrr	t0, vlenb
	li	t5, 184
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs8r.v	v0, (t0)                        # vscale x 64-byte Folded Spill
	vadd.vv	v0, v16, v8
	csrr	t0, vlenb
	li	t5, 176
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs8r.v	v0, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t5, 168
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v0, v0, v8
	csrr	t0, vlenb
	slli	t0, t0, 7
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs8r.v	v0, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t5, 192
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v0, v8
	csrr	t0, vlenb
	li	t5, 120
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t5, 168
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 6 21                          # k135114294097424.py:6:21
	vmslt.vx	v6, v8, t3
	csrr	t0, vlenb
	li	t5, 192
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vmslt.vx	v7, v8, t3
	csrr	t0, vlenb
	li	t5, 160
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs1r.v	v7, (t0)                        # vscale x 8-byte Folded Spill
	vmv1r.v	v8, v6
	csrr	t0, vlenb
	li	t5, 112
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs1r.v	v6, (t0)                        # vscale x 8-byte Folded Spill
	csrr	t0, vlenb
	li	t5, 160
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vl1r.v	v9, (t0)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v8, v9, 4
	csrr	t0, vlenb
	li	t5, 136
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs1r.v	v8, (t0)                        # vscale x 8-byte Folded Spill
	csrr	t0, vlenb
	li	t5, 184
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 12 36                         # k135114294097424.py:12:36
	vsetvli	zero, a3, e32, m8, ta, ma
	vand.vx	v8, v8, t4
	csrr	t0, vlenb
	li	t5, 184
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t5, 176
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, t4
	csrr	t0, vlenb
	li	t5, 144
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	slli	t0, t0, 7
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, t4
	csrr	t0, vlenb
	li	t5, 80
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t5, 120
	mul	t0, t0, t5
	add	t0, sp, t0
	lui	t5, 4
	addi	t5, t5, -1088
	add	t0, t0, t5
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, t4
	.loc	1 6 21                          # k135114294097424.py:6:21
	vmslt.vx	v7, v16, t3
	vmslt.vx	v6, v24, t3
	csrr	t0, vlenb
	li	t3, 120
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vs1r.v	v7, (t0)                        # vscale x 8-byte Folded Spill
	csrr	t0, vlenb
	slli	t0, t0, 7
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vs1r.v	v6, (t0)                        # vscale x 8-byte Folded Spill
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v7, v6, 4
	csrr	t0, vlenb
	li	t3, 104
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vs1r.v	v7, (t0)                        # vscale x 8-byte Folded Spill
	csrr	t0, vlenb
	li	t3, 176
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vs8r.v	v24, (t0)                       # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t3, 184
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 12 36                         # k135114294097424.py:12:36
	vsetvli	zero, a3, e32, m8, ta, ma
	vsub.vv	v24, v24, v0
	csrr	t0, vlenb
	li	t3, 88
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vs8r.v	v24, (t0)                       # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t3, 184
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vs8r.v	v16, (t0)                       # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t3, 144
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	vsub.vv	v16, v16, v24
	csrr	t0, vlenb
	li	t3, 168
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	csrr	t0, vlenb
	li	t3, 80
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	vsub.vv	v24, v24, v0
	csrr	t0, vlenb
	li	t3, 80
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vs8r.v	v24, (t0)                       # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t3, 192
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	vsub.vv	v0, v24, v8
	csrr	t0, vlenb
	li	t3, 136
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vl1r.v	v8, (t0)                        # vscale x 8-byte Folded Reload
	csrr	t0, vlenb
	li	t3, 104
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vl1r.v	v9, (t0)                        # vscale x 8-byte Folded Reload
	.loc	1 6 21                          # k135114294097424.py:6:21
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v8, v9, 8
	csrr	t0, vlenb
	slli	t0, t0, 5
	add	t0, sp, t0
	lui	t3, 4
	addi	t3, t3, -1088
	add	t0, t0, t3
	vs1r.v	v8, (t0)                        # vscale x 8-byte Folded Spill
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	t0, v8
	andi	t3, t0, 1
	csrr	t4, vlenb
	li	t5, 96
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vl8r.v	v24, (t4)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v8, v24, v24
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v8, a2
	csrr	t4, vlenb
	li	t5, 144
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vs8r.v	v8, (t4)                        # vscale x 64-byte Folded Spill
	beqz	t3, .LBB0_2
# %bb.1:                                # %cond.load
	vmv.x.s	t3, v8
	lhu	t3, 0(t3)
	slli	t3, t3, 16
	fmv.w.x	fs8, zero
	fmv.w.x	fs7, t3
	fsw	fs8, 952(sp)                    # 4-byte Folded Spill
	fsw	fs8, 944(sp)                    # 4-byte Folded Spill
	fmv.s	ft0, fs8
	fsw	fs8, 960(sp)                    # 4-byte Folded Spill
	fmv.s	fa1, fs8
	fsw	fs8, 968(sp)                    # 4-byte Folded Spill
	fmv.s	ft1, fs8
	fsw	fs8, 976(sp)                    # 4-byte Folded Spill
	fmv.s	ft2, fs8
	fsw	fs8, 984(sp)                    # 4-byte Folded Spill
	fmv.s	ft3, fs8
	fsw	fs8, 992(sp)                    # 4-byte Folded Spill
	fmv.s	ft4, fs8
	fsw	fs8, 1000(sp)                   # 4-byte Folded Spill
	fmv.s	ft5, fs8
	fsw	fs8, 1008(sp)                   # 4-byte Folded Spill
	fmv.s	ft6, fs8
	fsw	fs8, 1016(sp)                   # 4-byte Folded Spill
	fmv.s	ft7, fs8
	fsw	fs8, 1024(sp)                   # 4-byte Folded Spill
	fmv.s	fa6, fs8
	fsw	fs8, 1032(sp)                   # 4-byte Folded Spill
	fmv.s	fa7, fs8
	fsw	fs8, 1040(sp)                   # 4-byte Folded Spill
	fmv.s	fa0, fs8
	fsw	fs8, 1048(sp)                   # 4-byte Folded Spill
	fmv.s	ft8, fs8
	fsw	fs8, 1056(sp)                   # 4-byte Folded Spill
	fmv.s	ft10, fs8
	fsw	fs8, 1064(sp)                   # 4-byte Folded Spill
	fmv.s	ft9, fs8
	fsw	fs8, 1072(sp)                   # 4-byte Folded Spill
	fmv.s	ft11, fs8
	fsw	fs8, 1080(sp)                   # 4-byte Folded Spill
	fsw	fs8, 416(sp)                    # 4-byte Folded Spill
	fsw	fs8, 424(sp)                    # 4-byte Folded Spill
	fsw	fs8, 432(sp)                    # 4-byte Folded Spill
	fsw	fs8, 440(sp)                    # 4-byte Folded Spill
	fsw	fs8, 448(sp)                    # 4-byte Folded Spill
	fsw	fs8, 456(sp)                    # 4-byte Folded Spill
	fsw	fs8, 464(sp)                    # 4-byte Folded Spill
	fsw	fs8, 472(sp)                    # 4-byte Folded Spill
	fsw	fs8, 484(sp)                    # 4-byte Folded Spill
	fsw	fs8, 1096(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1104(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1112(sp)                   # 4-byte Folded Spill
	fmv.s	fs0, fs8
	fmv.s	fs1, fs8
	fmv.s	fs9, fs8
	fmv.s	fs10, fs8
	fmv.s	fs11, fs8
	fmv.s	fs2, fs8
	fmv.s	fs4, fs8
	fmv.s	fs3, fs8
	fmv.s	fs6, fs8
	fmv.s	fs5, fs8
	fmv.s	fa4, fs8
	fsw	fs8, 1088(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1120(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1128(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1136(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1144(sp)                   # 4-byte Folded Spill
	fmv.s	fa3, fs8
	fmv.s	fa5, fs8
	fsw	fs8, 504(sp)                    # 4-byte Folded Spill
	fsw	fs8, 508(sp)                    # 4-byte Folded Spill
	fsw	fs8, 512(sp)                    # 4-byte Folded Spill
	fsw	fs8, 516(sp)                    # 4-byte Folded Spill
	fsw	fs8, 520(sp)                    # 4-byte Folded Spill
	fsw	fs8, 524(sp)                    # 4-byte Folded Spill
	fsw	fs8, 528(sp)                    # 4-byte Folded Spill
	fsw	fs8, 532(sp)                    # 4-byte Folded Spill
	fsw	fs8, 536(sp)                    # 4-byte Folded Spill
	fsw	fs8, 540(sp)                    # 4-byte Folded Spill
	fsw	fs8, 544(sp)                    # 4-byte Folded Spill
	fsw	fs8, 548(sp)                    # 4-byte Folded Spill
	fsw	fs8, 552(sp)                    # 4-byte Folded Spill
	fsw	fs8, 560(sp)                    # 4-byte Folded Spill
	fsw	fs8, 568(sp)                    # 4-byte Folded Spill
	fsw	fs8, 576(sp)                    # 4-byte Folded Spill
	fsw	fs8, 584(sp)                    # 4-byte Folded Spill
	fsw	fs8, 592(sp)                    # 4-byte Folded Spill
	fsw	fs8, 600(sp)                    # 4-byte Folded Spill
	fsw	fs8, 608(sp)                    # 4-byte Folded Spill
	fsw	fs8, 616(sp)                    # 4-byte Folded Spill
	fsw	fs8, 624(sp)                    # 4-byte Folded Spill
	fsw	fs8, 632(sp)                    # 4-byte Folded Spill
	fsw	fs8, 640(sp)                    # 4-byte Folded Spill
	fsw	fs8, 648(sp)                    # 4-byte Folded Spill
	fsw	fs8, 656(sp)                    # 4-byte Folded Spill
	fsw	fs8, 664(sp)                    # 4-byte Folded Spill
	fsw	fs8, 672(sp)                    # 4-byte Folded Spill
	fsw	fs8, 680(sp)                    # 4-byte Folded Spill
	fsw	fs8, 688(sp)                    # 4-byte Folded Spill
	fsw	fs8, 696(sp)                    # 4-byte Folded Spill
	fsw	fs8, 704(sp)                    # 4-byte Folded Spill
	fsw	fs8, 712(sp)                    # 4-byte Folded Spill
	fsw	fs8, 720(sp)                    # 4-byte Folded Spill
	fsw	fs8, 728(sp)                    # 4-byte Folded Spill
	fsw	fs8, 736(sp)                    # 4-byte Folded Spill
	fsw	fs8, 744(sp)                    # 4-byte Folded Spill
	fsw	fs8, 752(sp)                    # 4-byte Folded Spill
	fsw	fs8, 760(sp)                    # 4-byte Folded Spill
	fsw	fs8, 768(sp)                    # 4-byte Folded Spill
	fsw	fs8, 776(sp)                    # 4-byte Folded Spill
	fsw	fs8, 784(sp)                    # 4-byte Folded Spill
	fsw	fs8, 792(sp)                    # 4-byte Folded Spill
	fsw	fs8, 800(sp)                    # 4-byte Folded Spill
	fsw	fs8, 808(sp)                    # 4-byte Folded Spill
	fsw	fs8, 816(sp)                    # 4-byte Folded Spill
	fsw	fs8, 824(sp)                    # 4-byte Folded Spill
	fsw	fs8, 832(sp)                    # 4-byte Folded Spill
	fsw	fs8, 840(sp)                    # 4-byte Folded Spill
	fsw	fs8, 848(sp)                    # 4-byte Folded Spill
	fsw	fs8, 856(sp)                    # 4-byte Folded Spill
	fsw	fs8, 864(sp)                    # 4-byte Folded Spill
	fsw	fs8, 872(sp)                    # 4-byte Folded Spill
	fsw	fs8, 880(sp)                    # 4-byte Folded Spill
	fsw	fs8, 888(sp)                    # 4-byte Folded Spill
	fsw	fs8, 896(sp)                    # 4-byte Folded Spill
	fsw	fs8, 904(sp)                    # 4-byte Folded Spill
	fsw	fs8, 912(sp)                    # 4-byte Folded Spill
	fsw	fs8, 920(sp)                    # 4-byte Folded Spill
	fsw	fs8, 928(sp)                    # 4-byte Folded Spill
	fsw	fs8, 936(sp)                    # 4-byte Folded Spill
	fmv.s	fa2, fs8
	j	.LBB0_3
.LBB0_2:
	.loc	1 0 52 is_stmt 0                # k135114294097424.py:0:52
	fmv.w.x	fs7, zero
	fmv.s	fs8, fs7
	fsw	fs7, 952(sp)                    # 4-byte Folded Spill
	fsw	fs7, 944(sp)                    # 4-byte Folded Spill
	fmv.s	ft0, fs7
	fsw	fs7, 960(sp)                    # 4-byte Folded Spill
	fmv.s	fa1, fs7
	fsw	fs7, 968(sp)                    # 4-byte Folded Spill
	fmv.s	ft1, fs7
	fsw	fs7, 976(sp)                    # 4-byte Folded Spill
	fmv.s	ft2, fs7
	fsw	fs7, 984(sp)                    # 4-byte Folded Spill
	fmv.s	ft3, fs7
	fsw	fs7, 992(sp)                    # 4-byte Folded Spill
	fmv.s	ft4, fs7
	fsw	fs7, 1000(sp)                   # 4-byte Folded Spill
	fmv.s	ft5, fs7
	fsw	fs7, 1008(sp)                   # 4-byte Folded Spill
	fmv.s	ft6, fs7
	fsw	fs7, 1016(sp)                   # 4-byte Folded Spill
	fmv.s	ft7, fs7
	fsw	fs7, 1024(sp)                   # 4-byte Folded Spill
	fmv.s	fa6, fs7
	fsw	fs7, 1032(sp)                   # 4-byte Folded Spill
	fmv.s	fa7, fs7
	fsw	fs7, 1040(sp)                   # 4-byte Folded Spill
	fmv.s	fa0, fs7
	fsw	fs7, 1048(sp)                   # 4-byte Folded Spill
	fmv.s	ft8, fs7
	fsw	fs7, 1056(sp)                   # 4-byte Folded Spill
	fmv.s	ft10, fs7
	fsw	fs7, 1064(sp)                   # 4-byte Folded Spill
	fmv.s	ft9, fs7
	fsw	fs7, 1072(sp)                   # 4-byte Folded Spill
	fmv.s	ft11, fs7
	fsw	fs7, 1080(sp)                   # 4-byte Folded Spill
	fsw	fs7, 416(sp)                    # 4-byte Folded Spill
	fsw	fs7, 424(sp)                    # 4-byte Folded Spill
	fsw	fs7, 432(sp)                    # 4-byte Folded Spill
	fsw	fs7, 440(sp)                    # 4-byte Folded Spill
	fsw	fs7, 448(sp)                    # 4-byte Folded Spill
	fsw	fs7, 456(sp)                    # 4-byte Folded Spill
	fsw	fs7, 464(sp)                    # 4-byte Folded Spill
	fsw	fs7, 472(sp)                    # 4-byte Folded Spill
	fsw	fs7, 484(sp)                    # 4-byte Folded Spill
	fsw	fs7, 1096(sp)                   # 4-byte Folded Spill
	fsw	fs7, 1104(sp)                   # 4-byte Folded Spill
	fsw	fs7, 1112(sp)                   # 4-byte Folded Spill
	fmv.s	fs0, fs7
	fmv.s	fs1, fs7
	fmv.s	fs9, fs7
	fmv.s	fs10, fs7
	fmv.s	fs11, fs7
	fmv.s	fs2, fs7
	fmv.s	fs4, fs7
	fmv.s	fs3, fs7
	fmv.s	fs6, fs7
	fmv.s	fs5, fs7
	fmv.s	fa4, fs7
	fsw	fs7, 1088(sp)                   # 4-byte Folded Spill
	fsw	fs7, 1120(sp)                   # 4-byte Folded Spill
	fsw	fs7, 1128(sp)                   # 4-byte Folded Spill
	fsw	fs7, 1136(sp)                   # 4-byte Folded Spill
	fsw	fs7, 1144(sp)                   # 4-byte Folded Spill
	fmv.s	fa3, fs7
	fmv.s	fa5, fs7
	fsw	fs7, 504(sp)                    # 4-byte Folded Spill
	fsw	fs7, 508(sp)                    # 4-byte Folded Spill
	fsw	fs7, 512(sp)                    # 4-byte Folded Spill
	fsw	fs7, 516(sp)                    # 4-byte Folded Spill
	fsw	fs7, 520(sp)                    # 4-byte Folded Spill
	fsw	fs7, 524(sp)                    # 4-byte Folded Spill
	fsw	fs7, 528(sp)                    # 4-byte Folded Spill
	fsw	fs7, 532(sp)                    # 4-byte Folded Spill
	fsw	fs7, 536(sp)                    # 4-byte Folded Spill
	fsw	fs7, 540(sp)                    # 4-byte Folded Spill
	fsw	fs7, 544(sp)                    # 4-byte Folded Spill
	fsw	fs7, 548(sp)                    # 4-byte Folded Spill
	fsw	fs7, 552(sp)                    # 4-byte Folded Spill
	fsw	fs7, 560(sp)                    # 4-byte Folded Spill
	fsw	fs7, 568(sp)                    # 4-byte Folded Spill
	fsw	fs7, 576(sp)                    # 4-byte Folded Spill
	fsw	fs7, 584(sp)                    # 4-byte Folded Spill
	fsw	fs7, 592(sp)                    # 4-byte Folded Spill
	fsw	fs7, 600(sp)                    # 4-byte Folded Spill
	fsw	fs7, 608(sp)                    # 4-byte Folded Spill
	fsw	fs7, 616(sp)                    # 4-byte Folded Spill
	fsw	fs7, 624(sp)                    # 4-byte Folded Spill
	fsw	fs7, 632(sp)                    # 4-byte Folded Spill
	fsw	fs7, 640(sp)                    # 4-byte Folded Spill
	fsw	fs7, 648(sp)                    # 4-byte Folded Spill
	fsw	fs7, 656(sp)                    # 4-byte Folded Spill
	fsw	fs7, 664(sp)                    # 4-byte Folded Spill
	fsw	fs7, 672(sp)                    # 4-byte Folded Spill
	fsw	fs7, 680(sp)                    # 4-byte Folded Spill
	fsw	fs7, 688(sp)                    # 4-byte Folded Spill
	fsw	fs7, 696(sp)                    # 4-byte Folded Spill
	fsw	fs7, 704(sp)                    # 4-byte Folded Spill
	fsw	fs7, 712(sp)                    # 4-byte Folded Spill
	fsw	fs7, 720(sp)                    # 4-byte Folded Spill
	fsw	fs7, 728(sp)                    # 4-byte Folded Spill
	fsw	fs7, 736(sp)                    # 4-byte Folded Spill
	fsw	fs7, 744(sp)                    # 4-byte Folded Spill
	fsw	fs7, 752(sp)                    # 4-byte Folded Spill
	fsw	fs7, 760(sp)                    # 4-byte Folded Spill
	fsw	fs7, 768(sp)                    # 4-byte Folded Spill
	fsw	fs7, 776(sp)                    # 4-byte Folded Spill
	fsw	fs7, 784(sp)                    # 4-byte Folded Spill
	fsw	fs7, 792(sp)                    # 4-byte Folded Spill
	fsw	fs7, 800(sp)                    # 4-byte Folded Spill
	fsw	fs7, 808(sp)                    # 4-byte Folded Spill
	fsw	fs7, 816(sp)                    # 4-byte Folded Spill
	fsw	fs7, 824(sp)                    # 4-byte Folded Spill
	fsw	fs7, 832(sp)                    # 4-byte Folded Spill
	fsw	fs7, 840(sp)                    # 4-byte Folded Spill
	fsw	fs7, 848(sp)                    # 4-byte Folded Spill
	fsw	fs7, 856(sp)                    # 4-byte Folded Spill
	fsw	fs7, 864(sp)                    # 4-byte Folded Spill
	fsw	fs7, 872(sp)                    # 4-byte Folded Spill
	fsw	fs7, 880(sp)                    # 4-byte Folded Spill
	fsw	fs7, 888(sp)                    # 4-byte Folded Spill
	fsw	fs7, 896(sp)                    # 4-byte Folded Spill
	fsw	fs7, 904(sp)                    # 4-byte Folded Spill
	fsw	fs7, 912(sp)                    # 4-byte Folded Spill
	fsw	fs7, 920(sp)                    # 4-byte Folded Spill
	fsw	fs7, 928(sp)                    # 4-byte Folded Spill
	fsw	fs7, 936(sp)                    # 4-byte Folded Spill
	fmv.s	fa2, fs7
.LBB0_3:                                # %else
	slli	t2, t2, 1
	csrr	t3, vlenb
	li	t4, 88
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 4
	addi	t4, t4, -1088
	add	t3, t3, t4
	vl8r.v	v8, (t3)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	t3, vlenb
	li	t4, 40
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 4
	addi	t4, t4, -1088
	add	t3, t3, t4
	vs8r.v	v8, (t3)                        # vscale x 64-byte Folded Spill
	vmv4r.v	v8, v16
	csrr	t3, vlenb
	li	t4, 56
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 4
	addi	t4, t4, -1088
	add	t3, t3, t4
	vs8r.v	v8, (t3)                        # vscale x 64-byte Folded Spill
	vslidedown.vi	v16, v16, 16
	csrr	t3, vlenb
	li	t4, 80
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 4
	addi	t4, t4, -1088
	add	t3, t3, t4
	vl8r.v	v8, (t3)                        # vscale x 64-byte Folded Reload
	vslidedown.vi	v24, v8, 16
	.loc	1 13 52 is_stmt 1               # k135114294097424.py:13:52
	andi	t3, t0, 2
	vmv4r.v	v8, v0
	csrr	t4, vlenb
	slli	t4, t4, 6
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vs8r.v	v8, (t4)                        # vscale x 64-byte Folded Spill
	.loc	1 0 0 is_stmt 0                 # k135114294097424.py:0
	vslidedown.vi	v0, v0, 16
	.loc	1 13 52                         # k135114294097424.py:13:52
	beqz	t3, .LBB0_5
# %bb.4:                                # %cond.load1
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	t3, vlenb
	li	t4, 144
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 4
	addi	t4, t4, -1088
	add	t3, t3, t4
	vl8r.v	v8, (t3)                        # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	t3, v8
	lhu	t3, 0(t3)
	slli	t3, t3, 16
	fmv.w.x	fs8, t3
.LBB0_5:                                # %else2
	.loc	1 0 52                          # k135114294097424.py:0:52
	add	t3, a0, t2
	li	a0, 4
	csrr	t2, vlenb
	slli	t2, t2, 7
	add	t2, sp, t2
	lui	t4, 4
	addi	t4, t4, -1088
	add	t2, t2, t4
	vl1r.v	v8, (t2)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v6, v8, 2
	csrr	t2, vlenb
	li	t4, 120
	mul	t2, t2, t4
	add	t2, sp, t2
	lui	t4, 4
	addi	t4, t4, -1088
	add	t2, t2, t4
	vl1r.v	v8, (t2)                        # vscale x 8-byte Folded Reload
	vslidedown.vi	v4, v8, 2
	csrr	t2, vlenb
	li	t4, 112
	mul	t2, t2, t4
	add	t2, sp, t2
	lui	t4, 4
	addi	t4, t4, -1088
	add	t2, t2, t4
	vl1r.v	v8, (t2)                        # vscale x 8-byte Folded Reload
	vslidedown.vi	v5, v8, 2
	.loc	1 13 52                         # k135114294097424.py:13:52
	andi	t2, t0, 4
	csrr	t4, vlenb
	li	t5, 160
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vl1r.v	v8, (t4)                        # vscale x 8-byte Folded Reload
	.loc	1 0 0                           # k135114294097424.py:0
	vslidedown.vi	v7, v8, 2
	fsw	fa2, 260(sp)                    # 4-byte Folded Spill
	.loc	1 13 52                         # k135114294097424.py:13:52
	beqz	t2, .LBB0_7
# %bb.6:                                # %cond.load4
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	t2, vlenb
	li	t4, 144
	mul	t2, t2, t4
	add	t2, sp, t2
	lui	t4, 4
	addi	t4, t4, -1088
	add	t2, t2, t4
	vl8r.v	v8, (t2)                        # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetvli	zero, zero, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	t2, v8
	lhu	t2, 0(t2)
	slli	t2, t2, 16
	fmv.w.x	fa2, t2
	fsw	fa2, 952(sp)                    # 4-byte Folded Spill
.LBB0_7:                                # %else5
	.loc	1 0 52                          # k135114294097424.py:0:52
	addi	t2, t3, 128
	vsetivli	zero, 16, e32, m4, ta, ma
	vwmulsu.vx	v8, v24, a0
	csrr	t4, vlenb
	li	t5, 24
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vs8r.v	v8, (t4)                        # vscale x 64-byte Folded Spill
	csrr	t4, vlenb
	li	t5, 80
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vl8r.v	v8, (t4)                        # vscale x 64-byte Folded Reload
	vwmulsu.vx	v24, v8, a0
	csrr	t4, vlenb
	li	t5, 48
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vs8r.v	v24, (t4)                       # vscale x 64-byte Folded Spill
	vwmulsu.vx	v8, v16, a0
	csrr	t4, vlenb
	slli	t4, t4, 4
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vs8r.v	v8, (t4)                        # vscale x 64-byte Folded Spill
	csrr	t4, vlenb
	li	t5, 56
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vl8r.v	v8, (t4)                        # vscale x 64-byte Folded Reload
	vwmulsu.vx	v16, v8, a0
	csrr	t4, vlenb
	slli	t4, t4, 3
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vs8r.v	v16, (t4)                       # vscale x 64-byte Folded Spill
	csrr	t4, vlenb
	li	t5, 40
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vl8r.v	v8, (t4)                        # vscale x 64-byte Folded Reload
	vwmulsu.vx	v16, v8, a0
	csrr	t4, vlenb
	li	t5, 56
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vs8r.v	v16, (t4)                       # vscale x 64-byte Folded Spill
	csrr	t4, vlenb
	li	t5, 88
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vl8r.v	v8, (t4)                        # vscale x 64-byte Folded Reload
	vwmulsu.vx	v16, v8, a0
	lui	t4, 4
	addi	t4, t4, -1088
	add	t4, sp, t4
	vs8r.v	v16, (t4)                       # vscale x 64-byte Folded Spill
	vwmulsu.vx	v8, v0, a0
	csrr	t4, vlenb
	li	t5, 88
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vs8r.v	v8, (t4)                        # vscale x 64-byte Folded Spill
	csrr	t4, vlenb
	slli	t4, t4, 6
	add	t4, sp, t4
	lui	t5, 4
	addi	t5, t5, -1088
	add	t4, t4, t5
	vl8r.v	v8, (t4)                        # vscale x 64-byte Folded Reload
	vwmulsu.vx	v16, v8, a0
	csrr	a0, vlenb
	li	t4, 80
	mul	a0, a0, t4
	add	a0, sp, a0
	lui	t4, 4
	addi	t4, t4, -1088
	add	a0, a0, t4
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 13 52                         # k135114294097424.py:13:52
	andi	a0, t0, 8
	.loc	1 0 0                           # k135114294097424.py:0
	vmv.v.i	v8, 0
	.loc	1 13 52                         # k135114294097424.py:13:52
	beqz	a0, .LBB0_9
# %bb.8:                                # %cond.load7
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a0, vlenb
	li	t4, 144
	mul	a0, a0, t4
	add	a0, sp, a0
	lui	t4, 4
	addi	t4, t4, -1088
	add	a0, a0, t4
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vmv1r.v	v12, v5
	vmv1r.v	v13, v4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v4, v16, 3
	vmv1r.v	v5, v12
	vmv.x.s	a0, v4
	vmv1r.v	v4, v13
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa2, a0
	fsw	fa2, 944(sp)                    # 4-byte Folded Spill
.LBB0_9:                                # %else8
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a0, vlenb
	li	t4, 104
	mul	a0, a0, t4
	add	a0, sp, a0
	lui	t4, 4
	addi	t4, t4, -1088
	add	a0, a0, t4
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, t1, e16, m8, ta, mu
	vmv.v.i	v16, 0
	vmv.v.i	v24, 0
	vle16.v	v24, (t2), v0.t
	csrr	a0, vlenb
	li	t1, 40
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	t1, 136
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vle16.v	v16, (t3), v0.t
	csrr	a0, vlenb
	li	t1, 136
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vmv4r.v	v16, v8
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, -1088
	add	a0, sp, a0
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, mu
	vluxei64.v	v8, (a1), v24, v0.t
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vmv4r.v	v8, v16
	vmv1r.v	v0, v6
	csrr	a0, vlenb
	li	t1, 56
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v8, (a1), v24, v0.t
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vmv4r.v	v8, v16
	csrr	a0, vlenb
	li	t1, 120
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v8, (a1), v24, v0.t
	csrr	a0, vlenb
	li	t1, 120
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vmv4r.v	v8, v16
	vmv1r.v	v0, v4
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v8, (a1), v24, v0.t
	csrr	a0, vlenb
	li	t1, 56
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vmv4r.v	v8, v16
	csrr	a0, vlenb
	li	t1, 112
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	t1, 48
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v8, (a1), v24, v0.t
	csrr	a0, vlenb
	li	t1, 112
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vmv4r.v	v8, v16
	vmv1r.v	v0, v5
	csrr	a0, vlenb
	li	t1, 24
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v8, (a1), v24, v0.t
	csrr	a0, vlenb
	li	t1, 48
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vmv4r.v	v8, v16
	csrr	a0, vlenb
	li	t1, 160
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	t1, 80
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v8, (a1), v24, v0.t
	csrr	a0, vlenb
	li	t1, 104
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vmv1r.v	v0, v7
	csrr	a0, vlenb
	li	t1, 88
	mul	a0, a0, t1
	add	a0, sp, a0
	lui	t1, 4
	addi	t1, t1, -1088
	add	a0, a0, t1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vluxei64.v	v16, (a1), v8, v0.t
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 13 52                         # k135114294097424.py:13:52
	andi	a1, t0, 16
	lui	a0, 3
	addi	a0, a0, 1528
	add	a0, sp, a0
	beqz	a1, .LBB0_11
# %bb.10:                               # %cond.load10
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 4
	addi	a1, a1, -1280
	add	a1, sp, a1
	csrr	t1, vlenb
	li	t4, 144
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v24, (t1)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a1, 1320(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft0, a1
	andi	a1, t0, 32
	fsw	ft0, 264(sp)                    # 4-byte Folded Spill
	bnez	a1, .LBB0_12
	j	.LBB0_13
.LBB0_11:
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	li	t1, 144
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	andi	a1, t0, 32
	fsw	ft0, 264(sp)                    # 4-byte Folded Spill
	beqz	a1, .LBB0_13
.LBB0_12:                               # %cond.load13
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 4
	addi	a1, a1, -1408
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a1, 1200(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft0, a1
	fsw	ft0, 960(sp)                    # 4-byte Folded Spill
.LBB0_13:                               # %else14
	andi	a1, t0, 64
	fmv.s	ft0, fs1
	bnez	a1, .LBB0_44
# %bb.14:                               # %else17
	andi	a1, t0, 128
	fsw	fa1, 272(sp)                    # 4-byte Folded Spill
	bnez	a1, .LBB0_45
.LBB0_15:                               # %else20
	andi	a1, t0, 256
	fmv.s	fa1, fs0
	bnez	a1, .LBB0_46
.LBB0_16:                               # %else23
	andi	a1, t0, 512
	fsw	ft1, 280(sp)                    # 4-byte Folded Spill
	bnez	a1, .LBB0_47
.LBB0_17:                               # %else26
	andi	a1, t0, 1024
	fmv.s	ft1, fs9
	bnez	a1, .LBB0_48
.LBB0_18:                               # %else29
	slli	a1, t0, 52
	fsw	ft2, 288(sp)                    # 4-byte Folded Spill
	bltz	a1, .LBB0_49
.LBB0_19:                               # %else32
	slli	a1, t0, 51
	fmv.s	ft2, fs10
	bgez	a1, .LBB0_21
.LBB0_20:                               # %cond.load34
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 1792
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a1, 360(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft3, a1
.LBB0_21:                               # %else35
	slli	a1, t0, 50
	csrr	t1, vlenb
	li	t4, 96
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v8, (t1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	fsw	ft3, 296(sp)                    # 4-byte Folded Spill
	bltz	a1, .LBB0_50
# %bb.22:                               # %else38
	slli	a1, t0, 49
	fmv.s	ft3, fs11
	bltz	a1, .LBB0_51
.LBB0_23:                               # %else41
	slli	a1, t0, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	fsw	ft4, 304(sp)                    # 4-byte Folded Spill
	bltz	a1, .LBB0_52
.LBB0_24:                               # %else44
	slli	a0, t0, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v16, a2
	fmv.s	ft4, fs2
	bltz	a0, .LBB0_53
.LBB0_25:                               # %else47
	slli	a0, t0, 46
	fsw	ft5, 312(sp)                    # 4-byte Folded Spill
	bltz	a0, .LBB0_54
.LBB0_26:                               # %else50
	slli	a0, t0, 45
	fmv.s	ft5, fs3
	bltz	a0, .LBB0_55
.LBB0_27:                               # %else53
	slli	a0, t0, 44
	fsw	ft6, 320(sp)                    # 4-byte Folded Spill
	bltz	a0, .LBB0_56
.LBB0_28:                               # %else56
	slli	a1, t0, 43
	lui	a0, 3
	addi	a0, a0, -584
	add	a0, sp, a0
	fmv.s	ft6, fs4
	bgez	a1, .LBB0_30
.LBB0_29:                               # %cond.load58
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 1280
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1896(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft7, a1
.LBB0_30:                               # %else59
	slli	a1, t0, 42
	csrr	t1, vlenb
	li	t4, 192
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v8, (t1)                        # vscale x 64-byte Folded Reload
	csrr	t1, vlenb
	li	t4, 152
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v16, v8, v16
	fsw	ft7, 328(sp)                    # 4-byte Folded Spill
	bgez	a1, .LBB0_32
# %bb.31:                               # %cond.load61
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 1152
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1776(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft7, a1
	fsw	ft7, 1024(sp)                   # 4-byte Folded Spill
.LBB0_32:                               # %else62
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	li	t1, 192
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vmulh.vx	v8, v8, a7
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 41
	vsra.vi	v0, v16, 6
	csrr	t1, vlenb
	li	t4, 80
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	fmv.s	ft7, fs5
	bgez	a1, .LBB0_34
# %bb.33:                               # %cond.load64
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a1, 13
	slli	a1, a1, 10
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1656(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa6, a1
.LBB0_34:                               # %else65
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	li	t1, 192
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 40
	vmulh.vx	v16, v0, a7
	fsw	fa6, 336(sp)                    # 4-byte Folded Spill
	bgez	a1, .LBB0_36
# %bb.35:                               # %cond.load67
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 896
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1536(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa6, a1
	fsw	fa6, 1032(sp)                   # 4-byte Folded Spill
.LBB0_36:                               # %else68
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v8, v8, 9
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 39
	vadd.vv	v16, v16, v0
	csrr	t1, vlenb
	li	t4, 88
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vs8r.v	v0, (t1)                        # vscale x 64-byte Folded Spill
	fmv.s	fa6, fs6
	bgez	a1, .LBB0_38
# %bb.37:                               # %cond.load70
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 768
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1416(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa7, a1
.LBB0_38:                               # %else71
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vsrl.vi	v0, v8, 31
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 38
	vsra.vi	v16, v16, 3
	fsw	fa7, 344(sp)                    # 4-byte Folded Spill
	bgez	a1, .LBB0_40
# %bb.39:                               # %cond.load73
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 640
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1296(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa7, a1
	fsw	fa7, 1040(sp)                   # 4-byte Folded Spill
.LBB0_40:                               # %else74
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v0
	csrr	a1, vlenb
	li	t1, 96
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	t1, 80
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a6
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 37
	vsrl.vi	v0, v16, 31
	csrr	t1, vlenb
	li	t4, 144
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vs8r.v	v24, (t1)                       # vscale x 64-byte Folded Spill
	fmv.s	fa7, fa4
	bgez	a1, .LBB0_42
# %bb.41:                               # %cond.load76
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a1, 25
	slli	a1, a1, 9
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1176(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa0, a1
.LBB0_42:                               # %else77
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	li	t1, 192
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vsub.vv	v8, v24, v8
	csrr	a1, vlenb
	li	t1, 80
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vadd.vv	v24, v16, v0
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 36
	csrr	t1, vlenb
	li	t4, 96
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v0, (t1)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v0, 6
	fsw	fa0, 352(sp)                    # 4-byte Folded Spill
	bgez	a1, .LBB0_57
# %bb.43:                               # %cond.load79
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 384
	add	a1, sp, a1
	csrr	t1, vlenb
	li	t4, 144
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vmv8r.v	v0, v16
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a1)
	ld	a1, 1056(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa0, a1
	fsw	fa0, 1048(sp)                   # 4-byte Folded Spill
	j	.LBB0_58
.LBB0_44:                               # %cond.load16
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a1, 29
	slli	a1, a1, 9
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a1, 1080(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa1, a1
	andi	a1, t0, 128
	fsw	fa1, 272(sp)                    # 4-byte Folded Spill
	beqz	a1, .LBB0_15
.LBB0_45:                               # %cond.load19
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 4
	addi	a1, a1, -1664
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a1, 960(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa1, a1
	fsw	fa1, 968(sp)                    # 4-byte Folded Spill
	andi	a1, t0, 256
	fmv.s	fa1, fs0
	beqz	a1, .LBB0_16
.LBB0_46:                               # %cond.load22
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 4
	addi	a1, a1, -1792
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a1, 840(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft1, a1
	andi	a1, t0, 512
	fsw	ft1, 280(sp)                    # 4-byte Folded Spill
	beqz	a1, .LBB0_17
.LBB0_47:                               # %cond.load25
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 4
	addi	a1, a1, -1920
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a1, 720(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft1, a1
	fsw	ft1, 976(sp)                    # 4-byte Folded Spill
	andi	a1, t0, 1024
	fmv.s	ft1, fs9
	beqz	a1, .LBB0_18
.LBB0_48:                               # %cond.load28
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a1, 7
	slli	a1, a1, 11
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a1, 600(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft2, a1
	slli	a1, t0, 52
	fsw	ft2, 288(sp)                    # 4-byte Folded Spill
	bgez	a1, .LBB0_19
.LBB0_49:                               # %cond.load31
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 1920
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a1, 480(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft2, a1
	fsw	ft2, 984(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 51
	fmv.s	ft2, fs10
	bltz	a1, .LBB0_20
	j	.LBB0_21
.LBB0_50:                               # %cond.load37
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 1664
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 240(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft3, a1
	fsw	ft3, 992(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 49
	fmv.s	ft3, fs11
	bgez	a1, .LBB0_23
.LBB0_51:                               # %cond.load40
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a1, 27
	slli	a1, a1, 9
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 120(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft4, a1
	slli	a1, t0, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	fsw	ft4, 304(sp)                    # 4-byte Folded Spill
	bgez	a1, .LBB0_24
.LBB0_52:                               # %cond.load43
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 1408
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a0, 0(a0)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft4, a0
	fsw	ft4, 1000(sp)                   # 4-byte Folded Spill
	slli	a0, t0, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v16, a2
	fmv.s	ft4, fs2
	bgez	a0, .LBB0_25
.LBB0_53:                               # %cond.load46
	vmv.x.s	a0, v24
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft5, a0
	slli	a0, t0, 46
	fsw	ft5, 312(sp)                    # 4-byte Folded Spill
	bgez	a0, .LBB0_26
.LBB0_54:                               # %cond.load49
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a0, v8
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft5, a0
	fsw	ft5, 1008(sp)                   # 4-byte Folded Spill
	slli	a0, t0, 45
	fmv.s	ft5, fs3
	bgez	a0, .LBB0_27
.LBB0_55:                               # %cond.load52
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a0, v8
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft6, a0
	slli	a0, t0, 44
	fsw	ft6, 320(sp)                    # 4-byte Folded Spill
	bgez	a0, .LBB0_28
.LBB0_56:                               # %cond.load55
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a0, v8
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft6, a0
	fsw	ft6, 1016(sp)                   # 4-byte Folded Spill
	slli	a1, t0, 43
	lui	a0, 3
	addi	a0, a0, -584
	add	a0, sp, a0
	fmv.s	ft6, fs4
	bltz	a1, .LBB0_29
	j	.LBB0_30
.LBB0_57:
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	li	t1, 144
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
.LBB0_58:                               # %else80
	csrr	a1, vlenb
	li	t1, 88
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vnmsub.vx	v24, a5, v16
	vmv.v.v	v16, v24
	.loc	1 13 52 is_stmt 1               # k135114294097424.py:13:52
	slli	a1, t0, 35
	csrr	t1, vlenb
	li	t4, 80
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v24, (t1)                       # vscale x 64-byte Folded Reload
	vadd.vv	v8, v8, v24
	flw	fs0, 1096(sp)                   # 4-byte Folded Reload
	bgez	a1, .LBB0_60
# %bb.59:                               # %cond.load82
	.loc	1 0 52 is_stmt 0                # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 256
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a1)
	ld	a1, 936(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft8, a1
.LBB0_60:                               # %else83
	.loc	1 0 52                          # k135114294097424.py:0:52
	fmv.s	fa0, fa5
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 34
	vsetvli	zero, a3, e32, m8, ta, ma
	vmadd.vx	v16, a4, v8
	flw	fa5, 1112(sp)                   # 4-byte Folded Reload
	fsw	ft8, 360(sp)                    # 4-byte Folded Spill
	bgez	a1, .LBB0_62
# %bb.61:                               # %cond.load85
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, 128
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a1)
	ld	a1, 816(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft8, a1
	fsw	ft8, 1056(sp)                   # 4-byte Folded Spill
.LBB0_62:                               # %else86
	.loc	1 0 52                          # k135114294097424.py:0:52
	flw	fa4, 1104(sp)                   # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 33
	flw	fs6, 416(sp)                    # 4-byte Folded Reload
	fmv.s	ft8, fa3
	bgez	a1, .LBB0_64
# %bb.63:                               # %cond.load88
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a1)
	ld	a1, 696(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft10, a1
.LBB0_64:                               # %else89
	.loc	1 0 52                          # k135114294097424.py:0:52
	flw	fs5, 424(sp)                    # 4-byte Folded Reload
	flw	fa3, 1128(sp)                   # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v16
	flw	fs4, 432(sp)                    # 4-byte Folded Reload
	fsw	ft10, 368(sp)                   # 4-byte Folded Spill
	bgez	a1, .LBB0_66
# %bb.65:                               # %cond.load91
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -128
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v0, (a1)
	ld	a1, 576(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft10, a1
	fsw	ft10, 1064(sp)                  # 4-byte Folded Spill
.LBB0_66:                               # %else92
	.loc	1 0 52                          # k135114294097424.py:0:52
	flw	fs3, 440(sp)                    # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v8, a2
	flw	fs2, 448(sp)                    # 4-byte Folded Reload
	flw	ft10, 1144(sp)                  # 4-byte Folded Reload
	bltz	a1, .LBB0_102
# %bb.67:                               # %else95
	.loc	1 0 52                          # k135114294097424.py:0:52
	flw	fs11, 456(sp)                   # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 30
	flw	fs10, 464(sp)                   # 4-byte Folded Reload
	fsw	ft9, 376(sp)                    # 4-byte Folded Spill
	bltz	a1, .LBB0_103
.LBB0_68:                               # %else98
	.loc	1 0 52                          # k135114294097424.py:0:52
	flw	fs9, 472(sp)                    # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 29
	flw	fs1, 484(sp)                    # 4-byte Folded Reload
	flw	ft9, 1120(sp)                   # 4-byte Folded Reload
	bltz	a1, .LBB0_104
.LBB0_69:                               # %else101
	slli	a1, t0, 28
	fsw	ft11, 384(sp)                   # 4-byte Folded Spill
	bltz	a1, .LBB0_105
.LBB0_70:                               # %else104
	slli	a1, t0, 27
	flw	ft11, 1136(sp)                  # 4-byte Folded Reload
	bltz	a1, .LBB0_106
.LBB0_71:                               # %else107
	slli	a1, t0, 26
	bltz	a1, .LBB0_107
.LBB0_72:                               # %else110
	slli	a1, t0, 25
	bltz	a1, .LBB0_108
.LBB0_73:                               # %else113
	slli	a1, t0, 24
	bltz	a1, .LBB0_109
.LBB0_74:                               # %else116
	slli	a1, t0, 23
	lui	a0, 2
	addi	a0, a0, 1376
	add	a0, sp, a0
	bltz	a1, .LBB0_110
.LBB0_75:                               # %else119
	slli	a1, t0, 22
	bltz	a1, .LBB0_111
.LBB0_76:                               # %else122
	slli	a1, t0, 21
	bltz	a1, .LBB0_112
.LBB0_77:                               # %else125
	slli	a1, t0, 20
	bltz	a1, .LBB0_113
.LBB0_78:                               # %else128
	slli	a1, t0, 19
	bltz	a1, .LBB0_114
.LBB0_79:                               # %else131
	slli	a1, t0, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v16, 16
	bltz	a1, .LBB0_115
.LBB0_80:                               # %else134
	slli	a1, t0, 17
	bltz	a1, .LBB0_116
.LBB0_81:                               # %else137
	slli	a1, t0, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bltz	a1, .LBB0_117
.LBB0_82:                               # %else140
	slli	a1, t0, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v16, a2
	bltz	a1, .LBB0_118
.LBB0_83:                               # %else143
	slli	a1, t0, 14
	bltz	a1, .LBB0_119
.LBB0_84:                               # %else146
	slli	a1, t0, 13
	bltz	a1, .LBB0_120
.LBB0_85:                               # %else149
	slli	a1, t0, 12
	bltz	a1, .LBB0_121
.LBB0_86:                               # %else152
	slli	a1, t0, 11
	bgez	a1, .LBB0_88
.LBB0_87:                               # %cond.load154
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -1792
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 960(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft3, a1
.LBB0_88:                               # %else155
	slli	a1, t0, 10
	csrr	t1, vlenb
	li	t4, 184
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v8, (t1)                        # vscale x 64-byte Folded Reload
	csrr	t1, vlenb
	li	t4, 152
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v16, v8, v16
	bgez	a1, .LBB0_90
# %bb.89:                               # %cond.load157
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -1920
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 840(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft4, a1
.LBB0_90:                               # %else158
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	li	t1, 184
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vmulh.vx	v8, v8, a7
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 9
	vsra.vi	v0, v16, 6
	csrr	t1, vlenb
	li	t4, 24
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	bgez	a1, .LBB0_92
# %bb.91:                               # %cond.load160
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a1, 5
	slli	a1, a1, 11
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 720(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft6, a1
.LBB0_92:                               # %else161
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	li	t1, 184
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 8
	vmulh.vx	v16, v0, a7
	bgez	a1, .LBB0_94
# %bb.93:                               # %cond.load163
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 2
	addi	a1, a1, 1920
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 600(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft5, a1
.LBB0_94:                               # %else164
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v8, v8, 9
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 7
	vadd.vv	v16, v16, v0
	csrr	t1, vlenb
	li	t4, 80
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vs8r.v	v0, (t1)                        # vscale x 64-byte Folded Spill
	bgez	a1, .LBB0_96
# %bb.95:                               # %cond.load166
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 2
	addi	a1, a1, 1792
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 480(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa6, a1
.LBB0_96:                               # %else167
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vsrl.vi	v0, v8, 31
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 6
	vsra.vi	v16, v16, 3
	bgez	a1, .LBB0_98
# %bb.97:                               # %cond.load169
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 2
	addi	a1, a1, 1664
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 360(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft7, a1
.LBB0_98:                               # %else170
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v0
	csrr	a1, vlenb
	li	t1, 88
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	t1, 24
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a6
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 5
	vsrl.vi	v0, v16, 31
	csrr	t1, vlenb
	li	t4, 144
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vs8r.v	v24, (t1)                       # vscale x 64-byte Folded Spill
	bgez	a1, .LBB0_100
# %bb.99:                               # %cond.load172
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a1, 19
	slli	a1, a1, 9
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 240(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa7, a1
.LBB0_100:                              # %else173
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	li	t1, 184
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vsub.vv	v8, v24, v8
	csrr	a1, vlenb
	li	t1, 24
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vadd.vv	v16, v16, v0
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a1, t0, 4
	csrr	t1, vlenb
	li	t4, 88
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v24, (t1)                       # vscale x 64-byte Folded Reload
	vsll.vi	v8, v24, 6
	vmv.v.v	v24, v16
	bgez	a1, .LBB0_122
# %bb.101:                              # %cond.load175
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 2
	addi	a1, a1, 1408
	add	a1, sp, a1
	csrr	t1, vlenb
	li	t4, 144
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vmv8r.v	v0, v16
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a1)
	ld	a1, 120(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa2, a1
	fsw	fa2, 1088(sp)                   # 4-byte Folded Spill
	j	.LBB0_123
.LBB0_102:                              # %cond.load94
	vmv.x.s	a1, v24
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft9, a1
	flw	fs11, 456(sp)                   # 4-byte Folded Reload
	slli	a1, t0, 30
	flw	fs10, 464(sp)                   # 4-byte Folded Reload
	fsw	ft9, 376(sp)                    # 4-byte Folded Spill
	bgez	a1, .LBB0_68
.LBB0_103:                              # %cond.load97
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a1, v8
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft9, a1
	fsw	ft9, 1072(sp)                   # 4-byte Folded Spill
	flw	fs9, 472(sp)                    # 4-byte Folded Reload
	slli	a1, t0, 29
	flw	fs1, 484(sp)                    # 4-byte Folded Reload
	flw	ft9, 1120(sp)                   # 4-byte Folded Reload
	bgez	a1, .LBB0_69
.LBB0_104:                              # %cond.load100
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a1, v8
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft11, a1
	slli	a1, t0, 28
	fsw	ft11, 384(sp)                   # 4-byte Folded Spill
	bgez	a1, .LBB0_70
.LBB0_105:                              # %cond.load103
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a1, v8
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft11, a1
	fsw	ft11, 1080(sp)                  # 4-byte Folded Spill
	slli	a1, t0, 27
	flw	ft11, 1136(sp)                  # 4-byte Folded Reload
	bgez	a1, .LBB0_71
.LBB0_106:                              # %cond.load106
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -256
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 360(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fs6, a1
	slli	a1, t0, 26
	bgez	a1, .LBB0_72
.LBB0_107:                              # %cond.load109
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -384
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 240(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fs5, a1
	slli	a1, t0, 25
	bgez	a1, .LBB0_73
.LBB0_108:                              # %cond.load112
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a1, 23
	slli	a1, a1, 9
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 120(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fs4, a1
	slli	a1, t0, 24
	bgez	a1, .LBB0_74
.LBB0_109:                              # %cond.load115
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -640
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a0, 0(a0)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fs3, a0
	slli	a1, t0, 23
	lui	a0, 2
	addi	a0, a0, 1376
	add	a0, sp, a0
	bgez	a1, .LBB0_75
.LBB0_110:                              # %cond.load118
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -768
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 2016(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fs2, a1
	slli	a1, t0, 22
	bgez	a1, .LBB0_76
.LBB0_111:                              # %cond.load121
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -896
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1896(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fs11, a1
	slli	a1, t0, 21
	bgez	a1, .LBB0_77
.LBB0_112:                              # %cond.load124
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a1, 11
	slli	a1, a1, 10
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1776(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fs10, a1
	slli	a1, t0, 20
	bgez	a1, .LBB0_78
.LBB0_113:                              # %cond.load127
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -1152
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1656(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fs9, a1
	slli	a1, t0, 19
	bgez	a1, .LBB0_79
.LBB0_114:                              # %cond.load130
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -1280
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1536(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fs1, a1
	slli	a1, t0, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v16, 16
	bgez	a1, .LBB0_80
.LBB0_115:                              # %cond.load133
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -1408
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1416(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fs0, a1
	slli	a1, t0, 17
	bgez	a1, .LBB0_81
.LBB0_116:                              # %cond.load136
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a1, 21
	slli	a1, a1, 9
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1296(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa4, a1
	slli	a1, t0, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a1, .LBB0_82
.LBB0_117:                              # %cond.load139
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a1, 3
	addi	a1, a1, -1664
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a1)
	ld	a1, 1176(a0)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	slli	a1, t0, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v16, a2
	bgez	a1, .LBB0_83
.LBB0_118:                              # %cond.load142
	vmv.x.s	a1, v24
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa1, a1
	slli	a1, t0, 14
	bgez	a1, .LBB0_84
.LBB0_119:                              # %cond.load145
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a1, v8
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft0, a1
	slli	a1, t0, 13
	bgez	a1, .LBB0_85
.LBB0_120:                              # %cond.load148
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a1, v8
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft1, a1
	slli	a1, t0, 12
	bgez	a1, .LBB0_86
.LBB0_121:                              # %cond.load151
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a1, v8
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft2, a1
	slli	a1, t0, 11
	bltz	a1, .LBB0_87
	j	.LBB0_88
.LBB0_122:
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	li	t1, 144
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
.LBB0_123:                              # %else176
	csrr	a1, vlenb
	li	t1, 80
	mul	a1, a1, t1
	add	a1, sp, a1
	lui	t1, 4
	addi	t1, t1, -1088
	add	a1, a1, t1
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vnmsub.vx	v24, a5, v16
	vmv.v.v	v16, v24
	.loc	1 13 52 is_stmt 1               # k135114294097424.py:13:52
	slli	a1, t0, 3
	csrr	t1, vlenb
	li	t4, 24
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl8r.v	v24, (t1)                       # vscale x 64-byte Folded Reload
	vadd.vv	v8, v8, v24
	bgez	a1, .LBB0_125
# %bb.124:                              # %cond.load178
	.loc	1 0 52 is_stmt 0                # k135114294097424.py:0:52
	lui	a1, 2
	addi	a1, a1, 1280
	add	a1, sp, a1
	vmv8r.v	v24, v0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a1)
	ld	a0, 0(a0)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft9, a0
	j	.LBB0_126
.LBB0_125:
	.loc	1 0 52                          # k135114294097424.py:0:52
	vmv8r.v	v24, v0
.LBB0_126:                              # %else179
	vsetvli	zero, a3, e32, m8, ta, ma
	vmadd.vx	v16, a4, v8
	.loc	1 13 52 is_stmt 1               # k135114294097424.py:13:52
	slli	a0, t0, 2
	lui	a1, 2
	addi	a1, a1, -736
	add	a1, sp, a1
	bgez	a0, .LBB0_128
# %bb.127:                              # %cond.load181
	.loc	1 0 52 is_stmt 0                # k135114294097424.py:0:52
	lui	a0, 2
	addi	a0, a0, 1152
	add	a0, sp, a0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a0)
	ld	a0, 1992(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa3, a0
.LBB0_128:                              # %else182
	slli	a0, t0, 1
	csrr	t1, vlenb
	slli	t1, t1, 5
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vl1r.v	v8, (t1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v7, v8, 1
	bltz	a0, .LBB0_158
# %bb.129:                              # %else185
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v16
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	a0, v7
	bltz	t0, .LBB0_159
.LBB0_130:                              # %else188
	andi	t0, a0, 1
	vadd.vx	v24, v8, a2
	bnez	t0, .LBB0_160
.LBB0_131:                              # %else191
	andi	t0, a0, 2
	bnez	t0, .LBB0_161
.LBB0_132:                              # %else194
	andi	t0, a0, 4
	bnez	t0, .LBB0_162
.LBB0_133:                              # %else197
	andi	t0, a0, 8
	bnez	t0, .LBB0_163
.LBB0_134:                              # %else200
	andi	t0, a0, 16
	bnez	t0, .LBB0_164
.LBB0_135:                              # %else203
	andi	t0, a0, 32
	bnez	t0, .LBB0_165
.LBB0_136:                              # %else206
	andi	t0, a0, 64
	bnez	t0, .LBB0_166
.LBB0_137:                              # %else209
	andi	t0, a0, 128
	bnez	t0, .LBB0_167
.LBB0_138:                              # %else212
	andi	t0, a0, 256
	bnez	t0, .LBB0_168
.LBB0_139:                              # %else215
	andi	t0, a0, 512
	bnez	t0, .LBB0_169
.LBB0_140:                              # %else218
	andi	t0, a0, 1024
	bnez	t0, .LBB0_170
.LBB0_141:                              # %else221
	slli	t0, a0, 52
	bltz	t0, .LBB0_171
.LBB0_142:                              # %else224
	slli	t0, a0, 51
	bltz	t0, .LBB0_172
.LBB0_143:                              # %else227
	slli	t0, a0, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v16, 16
	bltz	t0, .LBB0_173
.LBB0_144:                              # %else230
	slli	t0, a0, 49
	bltz	t0, .LBB0_174
.LBB0_145:                              # %else233
	slli	t0, a0, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bltz	t0, .LBB0_175
.LBB0_146:                              # %else236
	slli	t0, a0, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v16, a2
	bltz	t0, .LBB0_176
.LBB0_147:                              # %else239
	slli	t0, a0, 46
	bltz	t0, .LBB0_177
.LBB0_148:                              # %else242
	slli	t0, a0, 45
	bltz	t0, .LBB0_178
.LBB0_149:                              # %else245
	slli	t0, a0, 44
	bltz	t0, .LBB0_179
.LBB0_150:                              # %else248
	slli	t0, a0, 43
	bgez	t0, .LBB0_152
.LBB0_151:                              # %cond.load250
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	t0, 29
	slli	t0, t0, 8
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	a1, 0(a1)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa2, a1
	fsw	fa2, 600(sp)                    # 4-byte Folded Spill
.LBB0_152:                              # %else251
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	li	t0, 176
	mul	a1, a1, t0
	add	a1, sp, a1
	lui	t0, 4
	addi	t0, t0, -1088
	add	a1, a1, t0
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	t0, 152
	mul	a1, a1, t0
	add	a1, sp, a1
	lui	t0, 4
	addi	t0, t0, -1088
	add	a1, a1, t0
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v16, v8, v16
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	t0, a0, 42
	lui	a1, 1
	addi	a1, a1, 1224
	add	a1, sp, a1
	bgez	t0, .LBB0_154
# %bb.153:                              # %cond.load253
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	addi	t0, t0, -896
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 2016(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 608(sp)                    # 4-byte Folded Spill
.LBB0_154:                              # %else254
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	t0, vlenb
	li	t1, 176
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 4
	addi	t1, t1, -1088
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vmulh.vx	v8, v8, a7
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	t0, a0, 41
	vsra.vi	v0, v16, 6
	csrr	t1, vlenb
	li	t4, 152
	mul	t1, t1, t4
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vs8r.v	v24, (t1)                       # vscale x 64-byte Folded Spill
	csrr	t1, vlenb
	slli	t1, t1, 5
	add	t1, sp, t1
	lui	t4, 4
	addi	t4, t4, -1088
	add	t1, t1, t4
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	bgez	t0, .LBB0_156
# %bb.155:                              # %cond.load256
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	t0, 7
	slli	t0, t0, 10
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1896(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 616(sp)                    # 4-byte Folded Spill
.LBB0_156:                              # %else257
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vmulh.vx	v16, v0, a7
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a7, a0, 40
	csrr	t0, vlenb
	li	t1, 176
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 4
	addi	t1, t1, -1088
	add	t0, t0, t1
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	vadd.vv	v8, v8, v24
	bgez	a7, .LBB0_180
# %bb.157:                              # %cond.load259
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a7, 2
	addi	a7, a7, -1152
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t1, 152
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 4
	addi	t1, t1, -1088
	add	t0, t0, t1
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 1776(a1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa2, a7
	fsw	fa2, 624(sp)                    # 4-byte Folded Spill
	j	.LBB0_181
.LBB0_158:                              # %cond.load184
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a0, 9
	slli	a0, a0, 10
	add	a0, sp, a0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a0)
	ld	a0, 1872(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft11, a0
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v16
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	a0, v7
	bgez	t0, .LBB0_130
.LBB0_159:                              # %cond.load187
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	addi	t0, t0, 896
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (t0)
	ld	t0, 1752(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	ft10, t0
	andi	t0, a0, 1
	vadd.vx	v24, v8, a2
	beqz	t0, .LBB0_131
.LBB0_160:                              # %cond.load190
	vmv.x.s	t0, v24
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	ft8, t0
	andi	t0, a0, 2
	beqz	t0, .LBB0_132
.LBB0_161:                              # %cond.load193
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa0, t0
	andi	t0, a0, 4
	beqz	t0, .LBB0_133
.LBB0_162:                              # %cond.load196
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 504(sp)                    # 4-byte Folded Spill
	andi	t0, a0, 8
	beqz	t0, .LBB0_134
.LBB0_163:                              # %cond.load199
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 508(sp)                    # 4-byte Folded Spill
	andi	t0, a0, 16
	beqz	t0, .LBB0_135
.LBB0_164:                              # %cond.load202
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	addi	t0, t0, 768
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1536(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 512(sp)                    # 4-byte Folded Spill
	andi	t0, a0, 32
	beqz	t0, .LBB0_136
.LBB0_165:                              # %cond.load205
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	addi	t0, t0, 640
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1416(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 516(sp)                    # 4-byte Folded Spill
	andi	t0, a0, 64
	beqz	t0, .LBB0_137
.LBB0_166:                              # %cond.load208
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	t0, 17
	slli	t0, t0, 9
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1296(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 520(sp)                    # 4-byte Folded Spill
	andi	t0, a0, 128
	beqz	t0, .LBB0_138
.LBB0_167:                              # %cond.load211
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	addi	t0, t0, 384
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1176(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 524(sp)                    # 4-byte Folded Spill
	andi	t0, a0, 256
	beqz	t0, .LBB0_139
.LBB0_168:                              # %cond.load214
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	addi	t0, t0, 256
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1056(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 528(sp)                    # 4-byte Folded Spill
	andi	t0, a0, 512
	beqz	t0, .LBB0_140
.LBB0_169:                              # %cond.load217
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	addi	t0, t0, 128
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 936(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 532(sp)                    # 4-byte Folded Spill
	andi	t0, a0, 1024
	beqz	t0, .LBB0_141
.LBB0_170:                              # %cond.load220
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 816(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 536(sp)                    # 4-byte Folded Spill
	slli	t0, a0, 52
	bgez	t0, .LBB0_142
.LBB0_171:                              # %cond.load223
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	addi	t0, t0, -128
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 696(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 540(sp)                    # 4-byte Folded Spill
	slli	t0, a0, 51
	bgez	t0, .LBB0_143
.LBB0_172:                              # %cond.load226
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	t0, 31
	slli	t0, t0, 8
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 576(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 544(sp)                    # 4-byte Folded Spill
	slli	t0, a0, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v16, 16
	bgez	t0, .LBB0_144
.LBB0_173:                              # %cond.load229
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	addi	t0, t0, -384
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 456(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 548(sp)                    # 4-byte Folded Spill
	slli	t0, a0, 49
	bgez	t0, .LBB0_145
.LBB0_174:                              # %cond.load232
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	t0, 15
	slli	t0, t0, 9
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 336(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 552(sp)                    # 4-byte Folded Spill
	slli	t0, a0, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	t0, .LBB0_146
.LBB0_175:                              # %cond.load235
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	t0, 2
	addi	t0, t0, -640
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (t0)
	ld	t0, 216(a1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 560(sp)                    # 4-byte Folded Spill
	slli	t0, a0, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v16, a2
	bgez	t0, .LBB0_147
.LBB0_176:                              # %cond.load238
	vmv.x.s	t0, v24
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 568(sp)                    # 4-byte Folded Spill
	slli	t0, a0, 46
	bgez	t0, .LBB0_148
.LBB0_177:                              # %cond.load241
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 576(sp)                    # 4-byte Folded Spill
	slli	t0, a0, 45
	bgez	t0, .LBB0_149
.LBB0_178:                              # %cond.load244
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 584(sp)                    # 4-byte Folded Spill
	slli	t0, a0, 44
	bgez	t0, .LBB0_150
.LBB0_179:                              # %cond.load247
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 592(sp)                    # 4-byte Folded Spill
	slli	t0, a0, 43
	bltz	t0, .LBB0_151
	j	.LBB0_152
.LBB0_180:
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a7, vlenb
	li	t0, 152
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1088
	add	a7, a7, t0
	vl8r.v	v24, (a7)                       # vscale x 64-byte Folded Reload
.LBB0_181:                              # %else260
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v8, v8, 9
	.loc	1 13 52 is_stmt 1               # k135114294097424.py:13:52
	slli	a7, a0, 39
	vadd.vv	v16, v16, v0
	csrr	t0, vlenb
	li	t1, 144
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 4
	addi	t1, t1, -1088
	add	t0, t0, t1
	vs8r.v	v0, (t0)                        # vscale x 64-byte Folded Spill
	bgez	a7, .LBB0_183
# %bb.182:                              # %cond.load262
	.loc	1 0 52 is_stmt 0                # k135114294097424.py:0:52
	li	a7, 27
	slli	a7, a7, 8
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 1656(a1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa2, a7
	fsw	fa2, 632(sp)                    # 4-byte Folded Spill
.LBB0_183:                              # %else263
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vsrl.vi	v0, v8, 31
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a7, a0, 38
	vsra.vi	v16, v16, 3
	bgez	a7, .LBB0_185
# %bb.184:                              # %cond.load265
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a7, 2
	addi	a7, a7, -1408
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 1536(a1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa2, a7
	fsw	fa2, 640(sp)                    # 4-byte Folded Spill
.LBB0_185:                              # %else266
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v0
	csrr	a7, vlenb
	li	t0, 80
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1088
	add	a7, a7, t0
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	csrr	a7, vlenb
	slli	a7, a7, 5
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1088
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a6
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a6, a0, 37
	vsrl.vi	v0, v16, 31
	bgez	a6, .LBB0_187
# %bb.186:                              # %cond.load268
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a6, 13
	slli	a6, a6, 9
	add	a6, sp, a6
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1416(a1)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa2, a6
	fsw	fa2, 648(sp)                    # 4-byte Folded Spill
.LBB0_187:                              # %else269
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a6, vlenb
	li	a7, 176
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 4
	addi	a7, a7, -1088
	add	a6, a6, a7
	vl8r.v	v24, (a6)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vsub.vv	v8, v24, v8
	csrr	a6, vlenb
	slli	a6, a6, 5
	add	a6, sp, a6
	lui	a7, 4
	addi	a7, a7, -1088
	add	a6, a6, a7
	vs8r.v	v8, (a6)                        # vscale x 64-byte Folded Spill
	vadd.vv	v16, v16, v0
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a6, a0, 36
	csrr	a7, vlenb
	li	t0, 80
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1088
	add	a7, a7, t0
	vl8r.v	v24, (a7)                       # vscale x 64-byte Folded Reload
	vsll.vi	v8, v24, 6
	vmv.v.v	v24, v16
	bgez	a6, .LBB0_189
# %bb.188:                              # %cond.load271
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a6, 2
	addi	a6, a6, -1664
	add	a6, sp, a6
	csrr	a7, vlenb
	li	t0, 152
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1088
	add	a7, a7, t0
	vl8r.v	v16, (a7)                       # vscale x 64-byte Folded Reload
	vmv8r.v	v0, v16
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a6)
	ld	a6, 1296(a1)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa2, a6
	fsw	fa2, 656(sp)                    # 4-byte Folded Spill
	j	.LBB0_190
.LBB0_189:
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a6, vlenb
	li	a7, 152
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 4
	addi	a7, a7, -1088
	add	a6, a6, a7
	vl8r.v	v0, (a6)                        # vscale x 64-byte Folded Reload
.LBB0_190:                              # %else272
	csrr	a6, vlenb
	li	a7, 144
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 4
	addi	a7, a7, -1088
	add	a6, a6, a7
	vl8r.v	v16, (a6)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vnmsub.vx	v24, a5, v16
	vmv.v.v	v16, v24
	.loc	1 13 52 is_stmt 1               # k135114294097424.py:13:52
	slli	a5, a0, 35
	csrr	a6, vlenb
	slli	a6, a6, 5
	add	a6, sp, a6
	lui	a7, 4
	addi	a7, a7, -1088
	add	a6, a6, a7
	vl8r.v	v24, (a6)                       # vscale x 64-byte Folded Reload
	vadd.vv	v8, v8, v24
	bltz	a5, .LBB0_232
# %bb.191:                              # %else275
	slli	a5, a0, 34
	vsetvli	zero, a3, e32, m8, ta, ma
	vmadd.vx	v16, a4, v8
	bltz	a5, .LBB0_233
.LBB0_192:                              # %else278
	slli	a4, a0, 33
	bltz	a4, .LBB0_234
.LBB0_193:                              # %else281
	slli	a4, a0, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v16
	bltz	a4, .LBB0_235
.LBB0_194:                              # %else284
	slli	a4, a0, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v8, a2
	bltz	a4, .LBB0_236
.LBB0_195:                              # %else287
	slli	a4, a0, 30
	bltz	a4, .LBB0_237
.LBB0_196:                              # %else290
	slli	a4, a0, 29
	bltz	a4, .LBB0_238
.LBB0_197:                              # %else293
	slli	a4, a0, 28
	bltz	a4, .LBB0_239
.LBB0_198:                              # %else296
	slli	a4, a0, 27
	bltz	a4, .LBB0_240
.LBB0_199:                              # %else299
	slli	a4, a0, 26
	bltz	a4, .LBB0_241
.LBB0_200:                              # %else302
	slli	a4, a0, 25
	bltz	a4, .LBB0_242
.LBB0_201:                              # %else305
	slli	a4, a0, 24
	bltz	a4, .LBB0_243
.LBB0_202:                              # %else308
	slli	a4, a0, 23
	bltz	a4, .LBB0_244
.LBB0_203:                              # %else311
	slli	a4, a0, 22
	bltz	a4, .LBB0_245
.LBB0_204:                              # %else314
	slli	a4, a0, 21
	addi	a1, sp, 2047
	addi	a1, a1, 1137
	bltz	a4, .LBB0_246
.LBB0_205:                              # %else317
	slli	a4, a0, 20
	bltz	a4, .LBB0_247
.LBB0_206:                              # %else320
	slli	a4, a0, 19
	bltz	a4, .LBB0_248
.LBB0_207:                              # %else323
	slli	a4, a0, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v16, 16
	bltz	a4, .LBB0_249
.LBB0_208:                              # %else326
	slli	a4, a0, 17
	bltz	a4, .LBB0_250
.LBB0_209:                              # %else329
	slli	a4, a0, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bltz	a4, .LBB0_251
.LBB0_210:                              # %else332
	slli	a4, a0, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v16, a2
	bgez	a4, .LBB0_212
.LBB0_211:                              # %cond.load334
	vmv.x.s	a2, v0
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 824(sp)                    # 4-byte Folded Spill
.LBB0_212:                              # %else335
	.loc	1 0 52 is_stmt 0                # k135114294097424.py:0:52
	csrr	a2, vlenb
	li	a4, 40
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 4
	addi	a4, a4, -1088
	add	a2, a2, a4
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a2, a0, 14
	bltz	a2, .LBB0_252
# %bb.213:                              # %else338
	slli	a2, a0, 13
	bltz	a2, .LBB0_253
.LBB0_214:                              # %else341
	slli	a2, a0, 12
	bltz	a2, .LBB0_254
.LBB0_215:                              # %else344
	slli	a2, a0, 11
	bltz	a2, .LBB0_255
.LBB0_216:                              # %else347
	slli	a2, a0, 10
	bltz	a2, .LBB0_256
.LBB0_217:                              # %else350
	slli	a2, a0, 9
	bltz	a2, .LBB0_257
.LBB0_218:                              # %else353
	slli	a2, a0, 8
	bltz	a2, .LBB0_258
.LBB0_219:                              # %else356
	slli	a2, a0, 7
	bltz	a2, .LBB0_259
.LBB0_220:                              # %else359
	slli	a2, a0, 6
	bltz	a2, .LBB0_260
.LBB0_221:                              # %else362
	slli	a2, a0, 5
	bltz	a2, .LBB0_261
.LBB0_222:                              # %else365
	slli	a2, a0, 4
	bgez	a2, .LBB0_224
.LBB0_223:                              # %cond.load367
	.loc	1 0 52                          # k135114294097424.py:0:52
	addi	a2, sp, 2047
	addi	a2, a2, 1409
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 360(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 912(sp)                    # 4-byte Folded Spill
.LBB0_224:                              # %else368
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e16, m8, ta, ma
	vslidedown.vx	v8, v16, a3
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a2, a0, 3
	csrr	a4, vlenb
	li	a5, 136
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 4
	addi	a5, a5, -1088
	add	a4, a4, a5
	vl8r.v	v24, (a4)                       # vscale x 64-byte Folded Reload
	vslidedown.vx	v24, v24, a3
	bgez	a2, .LBB0_226
# %bb.225:                              # %cond.load370
	.loc	1 0 52                          # k135114294097424.py:0:52
	addi	a2, sp, 2047
	addi	a2, a2, 1281
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 240(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 920(sp)                    # 4-byte Folded Spill
.LBB0_226:                              # %else371
	slli	a2, a0, 2
	csrr	a4, vlenb
	slli	a4, a4, 5
	add	a4, sp, a4
	lui	a5, 4
	addi	a5, a5, -1088
	add	a4, a4, a5
	vs8r.v	v0, (a4)                        # vscale x 64-byte Folded Spill
	bgez	a2, .LBB0_228
# %bb.227:                              # %cond.load373
	.loc	1 0 52                          # k135114294097424.py:0:52
	addi	a2, sp, 2047
	addi	a2, a2, 1153
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 120(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 928(sp)                    # 4-byte Folded Spill
.LBB0_228:                              # %else374
	.loc	1 0 52                          # k135114294097424.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vzext.vf2	v0, v8
	csrr	a2, vlenb
	li	a4, 24
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 4
	addi	a4, a4, -1088
	add	a2, a2, a4
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vzext.vf2	v8, v16
	csrr	a2, vlenb
	li	a4, 152
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 4
	addi	a4, a4, -1088
	add	a2, a2, a4
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vzext.vf2	v8, v24
	csrr	a2, vlenb
	li	a4, 40
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 4
	addi	a4, a4, -1088
	add	a2, a2, a4
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 13 52                         # k135114294097424.py:13:52
	slli	a2, a0, 1
	csrr	a4, vlenb
	li	a5, 136
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 4
	addi	a5, a5, -1088
	add	a4, a4, a5
	vl8r.v	v0, (a4)                        # vscale x 64-byte Folded Reload
	vzext.vf2	v8, v0
	csrr	a4, vlenb
	li	a5, 144
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 4
	addi	a5, a5, -1088
	add	a4, a4, a5
	vs8r.v	v8, (a4)                        # vscale x 64-byte Folded Spill
	bgez	a2, .LBB0_230
# %bb.229:                              # %cond.load376
	.loc	1 0 52                          # k135114294097424.py:0:52
	addi	a2, sp, 2047
	addi	a2, a2, 1025
	csrr	a4, vlenb
	slli	a4, a4, 5
	add	a4, sp, a4
	lui	a5, 4
	addi	a5, a5, -1088
	add	a4, a4, a5
	vl8r.v	v8, (a4)                        # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a2)
	ld	a1, 0(a1)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa2, a1
	fsw	fa2, 936(sp)                    # 4-byte Folded Spill
.LBB0_230:                              # %else377
	.loc	1 0 52                          # k135114294097424.py:0:52
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vslideup.vi	v0, v24, 16
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vslideup.vi	v0, v8, 16
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vslideup.vi	v0, v16, 16
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslideup.vi	v0, v8, 16
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsll.vi	v24, v8, 16
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsll.vi	v24, v8, 16
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	sd	t3, 496(sp)                     # 8-byte Folded Spill
	sd	t2, 488(sp)                     # 8-byte Folded Spill
	fsw	ft8, 484(sp)                    # 4-byte Folded Spill
	fsw	ft10, 1144(sp)                  # 4-byte Folded Spill
	fsw	ft11, 1136(sp)                  # 4-byte Folded Spill
	fsw	fa3, 1128(sp)                   # 4-byte Folded Spill
	fsw	ft9, 1120(sp)                   # 4-byte Folded Spill
	fsw	fa7, 472(sp)                    # 4-byte Folded Spill
	fsw	ft7, 464(sp)                    # 4-byte Folded Spill
	fsw	fa6, 456(sp)                    # 4-byte Folded Spill
	fsw	ft5, 448(sp)                    # 4-byte Folded Spill
	fsw	ft6, 440(sp)                    # 4-byte Folded Spill
	fsw	ft4, 432(sp)                    # 4-byte Folded Spill
	fsw	ft3, 424(sp)                    # 4-byte Folded Spill
	fsw	ft2, 416(sp)                    # 4-byte Folded Spill
	fsw	ft1, 408(sp)                    # 4-byte Folded Spill
	fsw	ft0, 400(sp)                    # 4-byte Folded Spill
	fsw	fa1, 392(sp)                    # 4-byte Folded Spill
	fsw	fa5, 1112(sp)                   # 4-byte Folded Spill
	fsw	fa4, 1104(sp)                   # 4-byte Folded Spill
	fsw	fs0, 1096(sp)                   # 4-byte Folded Spill
	.loc	1 13 52                         # k135114294097424.py:13:52
	bgez	a0, .LBB0_262
# %bb.231:                              # %cond.load379
	.loc	1 0 52                          # k135114294097424.py:0:52
	addi	a0, sp, 2047
	addi	a0, a0, 897
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1032(a0)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fs0, a0
	j	.LBB0_263
.LBB0_232:                              # %cond.load274
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a5, 25
	slli	a5, a5, 8
	add	a5, sp, a5
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a5)
	ld	a5, 1176(a1)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa2, a5
	fsw	fa2, 664(sp)                    # 4-byte Folded Spill
	slli	a5, a0, 34
	vsetvli	zero, a3, e32, m8, ta, ma
	vmadd.vx	v16, a4, v8
	bgez	a5, .LBB0_192
.LBB0_233:                              # %cond.load277
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a4, 2
	addi	a4, a4, -1920
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a4)
	ld	a4, 1056(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 672(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 33
	bgez	a4, .LBB0_193
.LBB0_234:                              # %cond.load280
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a4, 3
	slli	a4, a4, 11
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a4)
	ld	a4, 936(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 680(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v16
	bgez	a4, .LBB0_194
.LBB0_235:                              # %cond.load283
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a4, 1
	addi	a4, a4, 1920
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v0, (a4)
	ld	a4, 816(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 688(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v8, a2
	bgez	a4, .LBB0_195
.LBB0_236:                              # %cond.load286
	vmv.x.s	a4, v24
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 696(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 30
	bgez	a4, .LBB0_196
.LBB0_237:                              # %cond.load289
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a4, v8
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 704(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 29
	bgez	a4, .LBB0_197
.LBB0_238:                              # %cond.load292
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a4, v8
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 712(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 28
	bgez	a4, .LBB0_198
.LBB0_239:                              # %cond.load295
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a4, v8
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 720(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 27
	bgez	a4, .LBB0_199
.LBB0_240:                              # %cond.load298
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a4, 23
	slli	a4, a4, 8
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 600(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 728(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 26
	bgez	a4, .LBB0_200
.LBB0_241:                              # %cond.load301
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a4, 1
	addi	a4, a4, 1664
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 480(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 736(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 25
	bgez	a4, .LBB0_201
.LBB0_242:                              # %cond.load304
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a4, 11
	slli	a4, a4, 9
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 360(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 744(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 24
	bgez	a4, .LBB0_202
.LBB0_243:                              # %cond.load307
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a4, 1
	addi	a4, a4, 1408
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 240(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 752(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 23
	bgez	a4, .LBB0_203
.LBB0_244:                              # %cond.load310
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a4, 21
	slli	a4, a4, 8
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 120(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 760(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 22
	bgez	a4, .LBB0_204
.LBB0_245:                              # %cond.load313
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a4, 1
	addi	a4, a4, 1152
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a1, 0(a1)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa2, a1
	fsw	fa2, 768(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 21
	addi	a1, sp, 2047
	addi	a1, a1, 1137
	bgez	a4, .LBB0_205
.LBB0_246:                              # %cond.load316
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a4, 5
	slli	a4, a4, 10
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 2016(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 776(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 20
	bgez	a4, .LBB0_206
.LBB0_247:                              # %cond.load319
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a4, 1
	addi	a4, a4, 896
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 1896(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 784(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 19
	bgez	a4, .LBB0_207
.LBB0_248:                              # %cond.load322
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a4, 19
	slli	a4, a4, 8
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 1776(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 792(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v16, 16
	bgez	a4, .LBB0_208
.LBB0_249:                              # %cond.load325
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a4, 1
	addi	a4, a4, 640
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 1656(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 800(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 17
	bgez	a4, .LBB0_209
.LBB0_250:                              # %cond.load328
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a4, 9
	slli	a4, a4, 9
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 1536(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 808(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a4, .LBB0_210
.LBB0_251:                              # %cond.load331
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a4, 1
	addi	a4, a4, 384
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294097424.py:13:52
	vse64.v	v24, (a4)
	ld	a4, 1416(a1)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa2, a4
	fsw	fa2, 816(sp)                    # 4-byte Folded Spill
	slli	a4, a0, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v16, a2
	bltz	a4, .LBB0_211
	j	.LBB0_212
.LBB0_252:                              # %cond.load337
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	a2, v8
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 832(sp)                    # 4-byte Folded Spill
	slli	a2, a0, 13
	bgez	a2, .LBB0_214
.LBB0_253:                              # %cond.load340
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	a2, v8
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 840(sp)                    # 4-byte Folded Spill
	slli	a2, a0, 12
	bgez	a2, .LBB0_215
.LBB0_254:                              # %cond.load343
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	a2, v8
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 848(sp)                    # 4-byte Folded Spill
	slli	a2, a0, 11
	bgez	a2, .LBB0_216
.LBB0_255:                              # %cond.load346
	.loc	1 0 52                          # k135114294097424.py:0:52
	li	a2, 17
	slli	a2, a2, 8
	add	a2, sp, a2
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 1200(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 856(sp)                    # 4-byte Folded Spill
	slli	a2, a0, 10
	bgez	a2, .LBB0_217
.LBB0_256:                              # %cond.load349
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a2, 1
	addi	a2, a2, 128
	add	a2, sp, a2
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 1080(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 864(sp)                    # 4-byte Folded Spill
	slli	a2, a0, 9
	bgez	a2, .LBB0_218
.LBB0_257:                              # %cond.load352
	.loc	1 0 52                          # k135114294097424.py:0:52
	lui	a2, 1
	add	a2, sp, a2
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 960(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 872(sp)                    # 4-byte Folded Spill
	slli	a2, a0, 8
	bgez	a2, .LBB0_219
.LBB0_258:                              # %cond.load355
	.loc	1 0 52                          # k135114294097424.py:0:52
	addi	a2, sp, 2047
	addi	a2, a2, 1921
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 840(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 880(sp)                    # 4-byte Folded Spill
	slli	a2, a0, 7
	bgez	a2, .LBB0_220
.LBB0_259:                              # %cond.load358
	.loc	1 0 52                          # k135114294097424.py:0:52
	addi	a2, sp, 2047
	addi	a2, a2, 1793
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 720(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 888(sp)                    # 4-byte Folded Spill
	slli	a2, a0, 6
	bgez	a2, .LBB0_221
.LBB0_260:                              # %cond.load361
	.loc	1 0 52                          # k135114294097424.py:0:52
	addi	a2, sp, 2047
	addi	a2, a2, 1665
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 600(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 896(sp)                    # 4-byte Folded Spill
	slli	a2, a0, 5
	bgez	a2, .LBB0_222
.LBB0_261:                              # %cond.load364
	.loc	1 0 52                          # k135114294097424.py:0:52
	addi	a2, sp, 2047
	addi	a2, a2, 1537
	.loc	1 13 52                         # k135114294097424.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 480(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa2, a2
	fsw	fa2, 904(sp)                    # 4-byte Folded Spill
	slli	a2, a0, 4
	bltz	a2, .LBB0_223
	j	.LBB0_224
.LBB0_262:
	.loc	1 0 52                          # k135114294097424.py:0:52
	flw	fs0, 260(sp)                    # 4-byte Folded Reload
.LBB0_263:                              # %else380
	lui	a0, 3
	li	a1, 32
	addi	a0, a0, 256
	csrr	a2, vlenb
	li	a3, 176
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1088
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 6 21 is_stmt 1                # k135114294097424.py:6:21
	vsetvli	zero, a1, e32, m8, ta, ma
	vmslt.vx	v8, v16, a0
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v10, v16, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v9, v16, a0
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v11, v16, a0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v10, v8, 4
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs1r.v	v10, (a0)                       # vscale x 8-byte Folded Spill
	vslideup.vi	v11, v9, 4
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs1r.v	v11, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 13 62                         # k135114294097424.py:13:62
	call	__truncsfbf2
	fsw	fa0, 260(sp)                    # 4-byte Folded Spill
	flw	fa0, 504(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 504(sp)                    # 4-byte Folded Spill
	flw	fa0, 508(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 508(sp)                    # 4-byte Folded Spill
	flw	fa0, 512(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 512(sp)                    # 4-byte Folded Spill
	flw	fa0, 516(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 516(sp)                    # 4-byte Folded Spill
	flw	fa0, 520(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 520(sp)                    # 4-byte Folded Spill
	flw	fa0, 524(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 524(sp)                    # 4-byte Folded Spill
	flw	fa0, 528(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 528(sp)                    # 4-byte Folded Spill
	flw	fa0, 532(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 532(sp)                    # 4-byte Folded Spill
	flw	fa0, 536(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 536(sp)                    # 4-byte Folded Spill
	flw	fa0, 540(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 540(sp)                    # 4-byte Folded Spill
	flw	fa0, 544(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 544(sp)                    # 4-byte Folded Spill
	flw	fa0, 548(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 548(sp)                    # 4-byte Folded Spill
	flw	fa0, 552(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 552(sp)                    # 4-byte Folded Spill
	flw	fa0, 560(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 560(sp)                    # 4-byte Folded Spill
	flw	fa0, 568(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 568(sp)                    # 4-byte Folded Spill
	flw	fa0, 576(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 576(sp)                    # 4-byte Folded Spill
	flw	fa0, 584(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 584(sp)                    # 4-byte Folded Spill
	flw	fa0, 592(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 592(sp)                    # 4-byte Folded Spill
	flw	fa0, 600(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 600(sp)                    # 4-byte Folded Spill
	flw	fa0, 608(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 608(sp)                    # 4-byte Folded Spill
	flw	fa0, 616(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 616(sp)                    # 4-byte Folded Spill
	flw	fa0, 624(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 624(sp)                    # 4-byte Folded Spill
	flw	fa0, 632(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 632(sp)                    # 4-byte Folded Spill
	flw	fa0, 640(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 640(sp)                    # 4-byte Folded Spill
	flw	fa0, 648(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 648(sp)                    # 4-byte Folded Spill
	flw	fa0, 656(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 656(sp)                    # 4-byte Folded Spill
	flw	fa0, 664(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 664(sp)                    # 4-byte Folded Spill
	flw	fa0, 672(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 672(sp)                    # 4-byte Folded Spill
	flw	fa0, 680(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 680(sp)                    # 4-byte Folded Spill
	flw	fa0, 688(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 688(sp)                    # 4-byte Folded Spill
	flw	fa0, 696(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 696(sp)                    # 4-byte Folded Spill
	flw	fa0, 704(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 704(sp)                    # 4-byte Folded Spill
	flw	fa0, 712(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 712(sp)                    # 4-byte Folded Spill
	flw	fa0, 720(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 720(sp)                    # 4-byte Folded Spill
	flw	fa0, 728(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 728(sp)                    # 4-byte Folded Spill
	flw	fa0, 736(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 736(sp)                    # 4-byte Folded Spill
	flw	fa0, 744(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 744(sp)                    # 4-byte Folded Spill
	flw	fa0, 752(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 752(sp)                    # 4-byte Folded Spill
	flw	fa0, 760(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 760(sp)                    # 4-byte Folded Spill
	flw	fa0, 768(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 768(sp)                    # 4-byte Folded Spill
	flw	fa0, 776(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 776(sp)                    # 4-byte Folded Spill
	flw	fa0, 784(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 784(sp)                    # 4-byte Folded Spill
	flw	fa0, 792(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 792(sp)                    # 4-byte Folded Spill
	flw	fa0, 800(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 800(sp)                    # 4-byte Folded Spill
	flw	fa0, 808(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 808(sp)                    # 4-byte Folded Spill
	flw	fa0, 816(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 816(sp)                    # 4-byte Folded Spill
	flw	fa0, 824(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 824(sp)                    # 4-byte Folded Spill
	flw	fa0, 832(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 832(sp)                    # 4-byte Folded Spill
	flw	fa0, 840(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 840(sp)                    # 4-byte Folded Spill
	flw	fa0, 848(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 848(sp)                    # 4-byte Folded Spill
	flw	fa0, 856(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 856(sp)                    # 4-byte Folded Spill
	flw	fa0, 864(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 864(sp)                    # 4-byte Folded Spill
	flw	fa0, 872(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 872(sp)                    # 4-byte Folded Spill
	flw	fa0, 880(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 880(sp)                    # 4-byte Folded Spill
	flw	fa0, 888(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 888(sp)                    # 4-byte Folded Spill
	flw	fa0, 896(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 896(sp)                    # 4-byte Folded Spill
	flw	fa0, 904(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 904(sp)                    # 4-byte Folded Spill
	flw	fa0, 912(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 912(sp)                    # 4-byte Folded Spill
	flw	fa0, 920(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 920(sp)                    # 4-byte Folded Spill
	flw	fa0, 928(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 928(sp)                    # 4-byte Folded Spill
	flw	fa0, 936(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 936(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs0
	call	__truncsfbf2
	fsw	fa0, 248(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs7
	call	__truncsfbf2
	fsw	fa0, 240(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs8
	call	__truncsfbf2
	fsw	fa0, 232(sp)                    # 4-byte Folded Spill
	flw	fa0, 952(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 952(sp)                    # 4-byte Folded Spill
	flw	fa0, 944(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 944(sp)                    # 4-byte Folded Spill
	flw	fa0, 264(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 264(sp)                    # 4-byte Folded Spill
	flw	fa0, 960(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 960(sp)                    # 4-byte Folded Spill
	flw	fa0, 272(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 272(sp)                    # 4-byte Folded Spill
	flw	fa0, 968(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 968(sp)                    # 4-byte Folded Spill
	flw	fa0, 280(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 280(sp)                    # 4-byte Folded Spill
	flw	fa0, 976(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 976(sp)                    # 4-byte Folded Spill
	flw	fa0, 288(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 288(sp)                    # 4-byte Folded Spill
	flw	fa0, 984(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 984(sp)                    # 4-byte Folded Spill
	flw	fa0, 296(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 296(sp)                    # 4-byte Folded Spill
	flw	fa0, 992(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 992(sp)                    # 4-byte Folded Spill
	flw	fa0, 304(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 304(sp)                    # 4-byte Folded Spill
	flw	fa0, 1000(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1000(sp)                   # 4-byte Folded Spill
	flw	fa0, 312(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 312(sp)                    # 4-byte Folded Spill
	flw	fa0, 1008(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1008(sp)                   # 4-byte Folded Spill
	flw	fa0, 320(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 320(sp)                    # 4-byte Folded Spill
	flw	fa0, 1016(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1016(sp)                   # 4-byte Folded Spill
	flw	fa0, 328(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 328(sp)                    # 4-byte Folded Spill
	flw	fa0, 1024(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 224(sp)                    # 4-byte Folded Spill
	flw	fa0, 336(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 336(sp)                    # 4-byte Folded Spill
	flw	fa0, 1032(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 216(sp)                    # 4-byte Folded Spill
	flw	fa0, 344(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 344(sp)                    # 4-byte Folded Spill
	flw	fa0, 1040(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 208(sp)                    # 4-byte Folded Spill
	flw	fa0, 352(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 352(sp)                    # 4-byte Folded Spill
	flw	fa0, 1048(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 184(sp)                    # 4-byte Folded Spill
	flw	fa0, 360(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 160(sp)                    # 4-byte Folded Spill
	flw	fa0, 1056(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 168(sp)                    # 4-byte Folded Spill
	flw	fa0, 368(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 176(sp)                    # 4-byte Folded Spill
	flw	fa0, 1064(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 192(sp)                    # 4-byte Folded Spill
	flw	fa0, 376(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 200(sp)                    # 4-byte Folded Spill
	flw	fa0, 1072(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 360(sp)                    # 4-byte Folded Spill
	flw	fa0, 384(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 368(sp)                    # 4-byte Folded Spill
	flw	fa0, 1080(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 376(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs6
	call	__truncsfbf2
	fsw	fa0, 384(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs5
	call	__truncsfbf2
	fsw	fa0, 152(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs4
	call	__truncsfbf2
	fsw	fa0, 144(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs3
	call	__truncsfbf2
	fsw	fa0, 136(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs2
	call	__truncsfbf2
	fsw	fa0, 132(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs11
	call	__truncsfbf2
	fsw	fa0, 128(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs10
	call	__truncsfbf2
	fsw	fa0, 124(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs9
	call	__truncsfbf2
	fsw	fa0, 120(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs1
	call	__truncsfbf2
	fsw	fa0, 116(sp)                    # 4-byte Folded Spill
	flw	fa0, 1096(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 112(sp)                    # 4-byte Folded Spill
	flw	fa0, 1104(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 108(sp)                    # 4-byte Folded Spill
	flw	fa0, 1112(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 104(sp)                    # 4-byte Folded Spill
	flw	fa0, 392(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1024(sp)                   # 4-byte Folded Spill
	flw	fa0, 400(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1032(sp)                   # 4-byte Folded Spill
	flw	fa0, 408(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1040(sp)                   # 4-byte Folded Spill
	flw	fa0, 416(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs11, fa0
	flw	fa0, 424(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs10, fa0
	flw	fa0, 432(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs9, fa0
	flw	fa0, 440(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs8, fa0
	flw	fa0, 448(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs7, fa0
	flw	fa0, 456(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs6, fa0
	flw	fa0, 464(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs5, fa0
	flw	fa0, 472(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs4, fa0
	flw	fa0, 1088(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs3, fa0
	flw	fa0, 1120(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs2, fa0
	flw	fa0, 1128(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs1, fa0
	flw	fa0, 1136(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs0, fa0
	flw	fa0, 1144(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	sd	a0, 1144(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs0
	sd	a0, 1136(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs1
	sd	a0, 1128(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs2
	sd	a0, 1120(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs3
	sd	a0, 1112(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs4
	sd	a0, 1104(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs5
	sd	a0, 1096(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs6
	sd	a0, 1088(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs7
	sd	a0, 1080(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs8
	sd	a0, 1072(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs9
	sd	a0, 1064(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs10
	sd	a0, 1056(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs11
	sd	a0, 1048(sp)                    # 8-byte Folded Spill
	flw	fa5, 1040(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1040(sp)                    # 8-byte Folded Spill
	flw	fa5, 1032(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1032(sp)                    # 8-byte Folded Spill
	flw	fa5, 1024(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1024(sp)                    # 8-byte Folded Spill
	flw	fa5, 104(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	flw	fa5, 108(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	flw	fa5, 112(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	flw	fa5, 116(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 448(sp)                     # 8-byte Folded Spill
	flw	fa5, 120(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 440(sp)                     # 8-byte Folded Spill
	flw	fa5, 124(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 432(sp)                     # 8-byte Folded Spill
	flw	fa5, 128(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 424(sp)                     # 8-byte Folded Spill
	flw	fa5, 132(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 416(sp)                     # 8-byte Folded Spill
	flw	fa5, 136(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 408(sp)                     # 8-byte Folded Spill
	flw	fa5, 144(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 400(sp)                     # 8-byte Folded Spill
	flw	fa5, 152(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 392(sp)                     # 8-byte Folded Spill
	flw	fa5, 384(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 384(sp)                     # 8-byte Folded Spill
	flw	fa5, 376(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 376(sp)                     # 8-byte Folded Spill
	flw	fa5, 368(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 368(sp)                     # 8-byte Folded Spill
	flw	fa5, 360(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 360(sp)                     # 8-byte Folded Spill
	flw	fa5, 200(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	flw	fa5, 192(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 192(sp)                     # 8-byte Folded Spill
	flw	fa5, 176(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 176(sp)                     # 8-byte Folded Spill
	flw	fa5, 168(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	flw	fa5, 160(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 160(sp)                     # 8-byte Folded Spill
	flw	fa5, 184(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	flw	fa5, 352(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 352(sp)                     # 8-byte Folded Spill
	flw	fa5, 208(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	flw	fa5, 344(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 344(sp)                     # 8-byte Folded Spill
	flw	fa5, 216(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	flw	fa5, 336(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 336(sp)                     # 8-byte Folded Spill
	flw	fa5, 224(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 224(sp)                     # 8-byte Folded Spill
	flw	fa5, 328(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 328(sp)                     # 8-byte Folded Spill
	flw	fa5, 1016(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1016(sp)                    # 8-byte Folded Spill
	flw	fa5, 320(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 320(sp)                     # 8-byte Folded Spill
	flw	fa5, 1008(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1008(sp)                    # 8-byte Folded Spill
	flw	fa5, 312(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 312(sp)                     # 8-byte Folded Spill
	flw	fa5, 1000(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1000(sp)                    # 8-byte Folded Spill
	flw	fa5, 304(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 304(sp)                     # 8-byte Folded Spill
	flw	fa5, 992(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 992(sp)                     # 8-byte Folded Spill
	flw	fa5, 296(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 296(sp)                     # 8-byte Folded Spill
	flw	fa5, 984(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 984(sp)                     # 8-byte Folded Spill
	flw	fa5, 288(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 288(sp)                     # 8-byte Folded Spill
	flw	fa5, 976(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 976(sp)                     # 8-byte Folded Spill
	flw	fa5, 280(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 280(sp)                     # 8-byte Folded Spill
	flw	fa5, 968(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 968(sp)                     # 8-byte Folded Spill
	flw	fa5, 272(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 272(sp)                     # 8-byte Folded Spill
	flw	fa5, 960(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 960(sp)                     # 8-byte Folded Spill
	flw	fa5, 264(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 264(sp)                     # 8-byte Folded Spill
	flw	fa5, 944(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 944(sp)                     # 8-byte Folded Spill
	flw	fa5, 952(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 952(sp)                     # 8-byte Folded Spill
	flw	fa5, 232(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 232(sp)                     # 8-byte Folded Spill
	flw	fa5, 240(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 240(sp)                     # 8-byte Folded Spill
	flw	fa5, 248(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 248(sp)                     # 8-byte Folded Spill
	flw	fa5, 936(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 936(sp)                     # 8-byte Folded Spill
	flw	fa5, 928(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 928(sp)                     # 8-byte Folded Spill
	flw	fa5, 920(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 920(sp)                     # 8-byte Folded Spill
	flw	fa5, 912(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 912(sp)                     # 8-byte Folded Spill
	flw	fa5, 904(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 904(sp)                     # 8-byte Folded Spill
	flw	fa5, 896(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 896(sp)                     # 8-byte Folded Spill
	flw	fa5, 888(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 888(sp)                     # 8-byte Folded Spill
	flw	fa5, 880(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 880(sp)                     # 8-byte Folded Spill
	flw	fa5, 872(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 872(sp)                     # 8-byte Folded Spill
	flw	fa5, 864(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 864(sp)                     # 8-byte Folded Spill
	flw	fa5, 856(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 856(sp)                     # 8-byte Folded Spill
	flw	fa5, 848(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 848(sp)                     # 8-byte Folded Spill
	flw	fa5, 840(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 840(sp)                     # 8-byte Folded Spill
	flw	fa5, 832(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 832(sp)                     # 8-byte Folded Spill
	flw	fa5, 824(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 824(sp)                     # 8-byte Folded Spill
	flw	fa5, 816(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 816(sp)                     # 8-byte Folded Spill
	flw	fa5, 808(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 808(sp)                     # 8-byte Folded Spill
	flw	fa5, 800(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 800(sp)                     # 8-byte Folded Spill
	flw	fa5, 792(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 792(sp)                     # 8-byte Folded Spill
	flw	fa5, 784(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 784(sp)                     # 8-byte Folded Spill
	flw	fa5, 776(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 776(sp)                     # 8-byte Folded Spill
	flw	fa5, 768(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 768(sp)                     # 8-byte Folded Spill
	flw	fa5, 760(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 760(sp)                     # 8-byte Folded Spill
	flw	fa5, 752(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 752(sp)                     # 8-byte Folded Spill
	flw	fa5, 744(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 744(sp)                     # 8-byte Folded Spill
	flw	fa5, 736(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 736(sp)                     # 8-byte Folded Spill
	flw	fa5, 728(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 728(sp)                     # 8-byte Folded Spill
	flw	fa5, 720(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 720(sp)                     # 8-byte Folded Spill
	flw	fa5, 712(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 712(sp)                     # 8-byte Folded Spill
	flw	fa5, 704(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 704(sp)                     # 8-byte Folded Spill
	flw	fa5, 696(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 696(sp)                     # 8-byte Folded Spill
	flw	fa5, 688(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 688(sp)                     # 8-byte Folded Spill
	flw	fa5, 680(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 680(sp)                     # 8-byte Folded Spill
	flw	fa5, 672(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 672(sp)                     # 8-byte Folded Spill
	flw	fa5, 664(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 664(sp)                     # 8-byte Folded Spill
	flw	fa5, 656(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 656(sp)                     # 8-byte Folded Spill
	flw	fa5, 648(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 648(sp)                     # 8-byte Folded Spill
	flw	fa5, 640(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 640(sp)                     # 8-byte Folded Spill
	flw	fa5, 632(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 632(sp)                     # 8-byte Folded Spill
	flw	fa5, 624(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 624(sp)                     # 8-byte Folded Spill
	flw	fa5, 616(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 616(sp)                     # 8-byte Folded Spill
	flw	fa5, 608(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 608(sp)                     # 8-byte Folded Spill
	flw	fa5, 600(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 600(sp)                     # 8-byte Folded Spill
	flw	fa5, 592(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 592(sp)                     # 8-byte Folded Spill
	flw	fa5, 584(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	flw	fa5, 576(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	flw	fa5, 568(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	flw	fa5, 560(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 560(sp)                     # 8-byte Folded Spill
	flw	fa5, 552(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 552(sp)                     # 8-byte Folded Spill
	flw	fa5, 548(sp)                    # 4-byte Folded Reload
	fmv.x.w	s9, fa5
	flw	fa5, 544(sp)                    # 4-byte Folded Reload
	fmv.x.w	s10, fa5
	flw	fa5, 540(sp)                    # 4-byte Folded Reload
	fmv.x.w	s11, fa5
	flw	fa5, 536(sp)                    # 4-byte Folded Reload
	fmv.x.w	s3, fa5
	flw	fa5, 532(sp)                    # 4-byte Folded Reload
	fmv.x.w	s2, fa5
	flw	fa5, 528(sp)                    # 4-byte Folded Reload
	fmv.x.w	s4, fa5
	flw	fa5, 524(sp)                    # 4-byte Folded Reload
	fmv.x.w	s5, fa5
	flw	fa5, 520(sp)                    # 4-byte Folded Reload
	fmv.x.w	s6, fa5
	flw	fa5, 516(sp)                    # 4-byte Folded Reload
	fmv.x.w	s7, fa5
	flw	fa5, 512(sp)                    # 4-byte Folded Reload
	fmv.x.w	s8, fa5
	flw	fa5, 508(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 584(sp)                     # 8-byte Folded Spill
	flw	fa5, 504(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 576(sp)                     # 8-byte Folded Spill
	flw	fa5, 260(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 568(sp)                     # 8-byte Folded Spill
	flw	fa0, 484(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	lui	a0, 1
	add	a0, sp, a0
	sh	s8, -1272(a0)
	lui	a0, 1
	add	a0, sp, a0
	sh	s7, -1270(a0)
	lui	a0, 1
	add	a0, sp, a0
	sh	s6, -1268(a0)
	lui	a0, 1
	add	a0, sp, a0
	sh	s5, -1266(a0)
	lui	a0, 1
	add	a0, sp, a0
	sh	s4, -1264(a0)
	lui	a0, 1
	add	a0, sp, a0
	sh	s2, -1262(a0)
	lui	a0, 1
	add	a0, sp, a0
	sh	s3, -1260(a0)
	lui	a0, 1
	add	a0, sp, a0
	sh	s11, -1258(a0)
	lui	a0, 1
	add	a0, sp, a0
	sh	s10, -1256(a0)
	lui	a0, 1
	add	a0, sp, a0
	sh	s9, -1254(a0)
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1252(a1)
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1250(a1)
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1248(a1)
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1246(a1)
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1244(a1)
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1242(a1)
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1240(a1)
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1238(a1)
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1236(a1)
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1234(a1)
	ld	a0, 632(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1232(a1)
	ld	a0, 640(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1230(a1)
	ld	a0, 648(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1228(a1)
	ld	a0, 656(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1226(a1)
	ld	a0, 664(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1224(a1)
	ld	a0, 672(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1222(a1)
	ld	a0, 680(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1220(a1)
	ld	a0, 688(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1218(a1)
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1278(a1)
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1276(a1)
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1274(a1)
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	li	s2, 32
	.loc	1 14 19                         # k135114294097424.py:14:19
	vsetvli	zero, s2, e32, m8, ta, ma
	vfcvt.f.x.v	v8, v8
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 16 18                         # k135114294097424.py:16:18
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 14 19                         # k135114294097424.py:14:19
	vfcvt.f.x.v	v8, v8
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 16 18                         # k135114294097424.py:16:18
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 14 19                         # k135114294097424.py:14:19
	vfcvt.f.x.v	v8, v8
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 16 18                         # k135114294097424.py:16:18
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 14 19                         # k135114294097424.py:14:19
	vfcvt.f.x.v	v8, v8
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 16 18                         # k135114294097424.py:16:18
	vfmul.vv	v0, v16, v8
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 13 62                         # k135114294097424.py:13:62
	fmv.x.w	a0, fa0
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1280(a1)
	ld	a0, 696(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1216(a1)
	ld	a0, 704(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1214(a1)
	ld	a0, 712(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1212(a1)
	ld	a0, 720(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1210(a1)
	ld	a0, 728(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1208(a1)
	ld	a0, 736(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1206(a1)
	ld	a0, 744(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1204(a1)
	ld	a0, 752(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1202(a1)
	ld	a0, 760(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1200(a1)
	ld	a0, 768(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1198(a1)
	ld	a0, 776(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1196(a1)
	ld	a0, 784(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1194(a1)
	ld	a0, 792(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1192(a1)
	ld	a0, 800(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1190(a1)
	ld	a0, 808(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1188(a1)
	ld	a0, 816(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1186(a1)
	ld	a0, 824(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1184(a1)
	ld	a0, 832(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1182(a1)
	ld	a0, 840(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1180(a1)
	ld	a0, 848(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1178(a1)
	ld	a0, 856(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1176(a1)
	ld	a0, 864(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1174(a1)
	ld	a0, 872(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1172(a1)
	ld	a0, 880(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1170(a1)
	ld	a0, 888(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1168(a1)
	ld	a0, 896(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1166(a1)
	ld	a0, 904(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1164(a1)
	ld	a0, 912(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1162(a1)
	ld	a0, 920(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1160(a1)
	ld	a0, 928(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1158(a1)
	ld	a0, 936(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1156(a1)
	ld	a0, 248(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1154(a1)
	ld	a0, 240(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1408(a1)
	ld	a0, 232(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1406(a1)
	ld	a0, 952(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1404(a1)
	ld	a0, 944(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1402(a1)
	ld	a0, 264(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1400(a1)
	ld	a0, 960(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1398(a1)
	ld	a0, 272(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1396(a1)
	ld	a0, 968(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1394(a1)
	ld	a0, 280(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1392(a1)
	ld	a0, 976(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1390(a1)
	ld	a0, 288(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1388(a1)
	ld	a0, 984(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1386(a1)
	ld	a0, 296(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1384(a1)
	ld	a0, 992(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1382(a1)
	ld	a0, 304(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1380(a1)
	ld	a0, 1000(sp)                    # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1378(a1)
	ld	a0, 312(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1376(a1)
	ld	a0, 1008(sp)                    # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1374(a1)
	ld	a0, 320(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1372(a1)
	ld	a0, 1016(sp)                    # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1370(a1)
	ld	a0, 328(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1368(a1)
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1366(a1)
	ld	a0, 336(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1364(a1)
	ld	a0, 216(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1362(a1)
	ld	a0, 344(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1360(a1)
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1358(a1)
	ld	a0, 352(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1356(a1)
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1354(a1)
	ld	a0, 160(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1352(a1)
	ld	a0, 168(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1350(a1)
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1348(a1)
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1346(a1)
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1344(a1)
	ld	a0, 360(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1342(a1)
	ld	a0, 368(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1340(a1)
	ld	a0, 376(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1338(a1)
	ld	a0, 384(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1336(a1)
	ld	a0, 392(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1334(a1)
	ld	a0, 400(sp)                     # 8-byte Folded Reload
	lui	a1, 1
	add	a1, sp, a1
	sh	a0, -1332(a1)
	addi	a0, sp, 2047
	addi	a0, a0, 769
	ld	a1, 408(sp)                     # 8-byte Folded Reload
	lui	a2, 1
	add	a2, sp, a2
	sh	a1, -1330(a2)
	addi	a1, sp, 2047
	addi	a1, a1, 705
	ld	a2, 416(sp)                     # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1328(a3)
	ld	a2, 424(sp)                     # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1326(a3)
	ld	a2, 432(sp)                     # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1324(a3)
	ld	a2, 440(sp)                     # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1322(a3)
	ld	a2, 448(sp)                     # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1320(a3)
	ld	a2, 456(sp)                     # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1318(a3)
	ld	a2, 464(sp)                     # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1316(a3)
	ld	a2, 472(sp)                     # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1314(a3)
	ld	a2, 1024(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1312(a3)
	ld	a2, 1032(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1310(a3)
	ld	a2, 1040(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1308(a3)
	ld	a2, 1048(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1306(a3)
	ld	a2, 1056(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1304(a3)
	ld	a2, 1064(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1302(a3)
	ld	a2, 1072(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1300(a3)
	ld	a2, 1080(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1298(a3)
	ld	a2, 1088(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1296(a3)
	ld	a2, 1096(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1294(a3)
	ld	a2, 1104(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1292(a3)
	ld	a2, 1112(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1290(a3)
	ld	a2, 1120(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1288(a3)
	ld	a2, 1128(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1286(a3)
	ld	a2, 1136(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1284(a3)
	ld	a2, 1144(sp)                    # 8-byte Folded Reload
	lui	a3, 1
	add	a3, sp, a3
	sh	a2, -1282(a3)
	vle16.v	v16, (a1)
	addi	a1, sp, 2047
	addi	a1, a1, 833
	addi	a2, sp, 2047
	addi	a2, a2, 641
	vle16.v	v20, (a2)
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 112
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1088
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v16, (a1)
	vle16.v	v24, (a0)
	vzext.vf2	v8, v20
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 1664
	.loc	1 17 23                         # k135114294097424.py:17:23
	vse32.v	v0, (a0)
	flw	fa0, 1788(sp)
	fsw	fa0, 1144(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1796(a0)
	flw	fa0, 1784(sp)
	fsw	fa0, 1136(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1800(a0)
	flw	fa0, 1780(sp)
	fsw	fa0, 1128(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1804(a0)
	flw	fa0, 1776(sp)
	fsw	fa0, 1120(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1808(a0)
	flw	fa0, 1772(sp)
	fsw	fa0, 1112(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1812(a0)
	flw	fa0, 1768(sp)
	fsw	fa0, 1104(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1816(a0)
	flw	fa0, 1764(sp)
	fsw	fa0, 1096(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1820(a0)
	flw	fa0, 1760(sp)
	fsw	fa0, 1088(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1824(a0)
	flw	fa0, 1756(sp)
	fsw	fa0, 1080(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1828(a0)
	flw	fa0, 1752(sp)
	fsw	fa0, 1072(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1832(a0)
	flw	fa0, 1748(sp)
	fsw	fa0, 1064(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1836(a0)
	flw	fa0, 1744(sp)
	fsw	fa0, 1056(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1840(a0)
	flw	fa0, 1740(sp)
	fsw	fa0, 1048(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1844(a0)
	flw	fa0, 1736(sp)
	fsw	fa0, 1040(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1848(a0)
	flw	fa0, 1732(sp)
	fsw	fa0, 1032(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1852(a0)
	flw	fa0, 1728(sp)
	fsw	fa0, 1024(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1856(a0)
	flw	fa0, 1724(sp)
	fsw	fa0, 1016(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1860(a0)
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 1008(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1920(a0)
	flw	fa0, 1720(sp)
	fsw	fa0, 1000(sp)                   # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1864(a0)
	flw	fa0, 1716(sp)
	fsw	fa0, 992(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1868(a0)
	flw	fa0, 1712(sp)
	fsw	fa0, 984(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1872(a0)
	flw	fa0, 1708(sp)
	fsw	fa0, 976(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1876(a0)
	flw	fa0, 1704(sp)
	fsw	fa0, 968(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1880(a0)
	flw	fa0, 1700(sp)
	fsw	fa0, 960(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1884(a0)
	flw	fa0, 1696(sp)
	fsw	fa0, 952(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1888(a0)
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 944(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1908(a0)
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 936(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1912(a0)
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 928(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1916(a0)
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 920(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1892(a0)
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 912(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1896(a0)
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 904(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1900(a0)
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 896(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1904(a0)
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s2, e32, m8, ta, ma
	li	s2, 32
	vse32.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1924(a0)
	fsw	fa0, 888(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1540(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1928(a0)
	fsw	fa0, 880(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1544(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1932(a0)
	fsw	fa0, 872(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1548(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1936(a0)
	fsw	fa0, 864(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1552(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1940(a0)
	fsw	fa0, 856(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1556(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1944(a0)
	fsw	fa0, 848(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1560(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1948(a0)
	fsw	fa0, 840(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1564(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1952(a0)
	fsw	fa0, 832(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1568(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1956(a0)
	fsw	fa0, 824(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1572(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1960(a0)
	fsw	fa0, 816(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1576(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1964(a0)
	fsw	fa0, 808(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1580(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1968(a0)
	fsw	fa0, 800(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1584(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1972(a0)
	fsw	fa0, 792(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1588(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1976(a0)
	fsw	fa0, 784(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1592(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1980(a0)
	fsw	fa0, 776(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1596(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1984(a0)
	fsw	fa0, 768(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1600(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1988(a0)
	fsw	fa0, 760(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1604(a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 752(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1664(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1992(a0)
	fsw	fa0, 744(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1608(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -1996(a0)
	fsw	fa0, 736(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1612(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -2000(a0)
	fsw	fa0, 728(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1616(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -2004(a0)
	fsw	fa0, 720(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1620(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -2008(a0)
	fsw	fa0, 712(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1624(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -2012(a0)
	fsw	fa0, 704(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1628(a0)
	lui	a0, 1
	add	a0, sp, a0
	flw	fa0, -2016(a0)
	fsw	fa0, 696(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1632(a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 688(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1652(a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 680(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1656(a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 672(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1660(a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 664(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1636(a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 656(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1640(a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 648(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1644(a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 640(sp)                    # 4-byte Folded Spill
	call	cosf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1648(a0)
	addi	a0, sp, 1152
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s2, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 1276(sp)
	fsw	fa0, 632(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1532(sp)
	flw	fa0, 1272(sp)
	fsw	fa0, 624(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1528(sp)
	flw	fa0, 1268(sp)
	fsw	fa0, 616(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1524(sp)
	flw	fa0, 1264(sp)
	fsw	fa0, 608(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1520(sp)
	flw	fa0, 1260(sp)
	fsw	fa0, 600(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1516(sp)
	flw	fa0, 1256(sp)
	fsw	fa0, 592(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1512(sp)
	flw	fa0, 1252(sp)
	fsw	fa0, 584(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1508(sp)
	flw	fa0, 1248(sp)
	fsw	fa0, 576(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1504(sp)
	flw	fa0, 1244(sp)
	fsw	fa0, 568(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1500(sp)
	flw	fa0, 1240(sp)
	fsw	fa0, 560(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1496(sp)
	flw	fa0, 1236(sp)
	fsw	fa0, 552(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1492(sp)
	flw	fa0, 1232(sp)
	fsw	fa0, 548(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1488(sp)
	flw	fa0, 1228(sp)
	fsw	fa0, 544(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1484(sp)
	flw	fa0, 1224(sp)
	fsw	fa0, 540(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1480(sp)
	flw	fa0, 1220(sp)
	fsw	fa0, 536(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1476(sp)
	flw	fa0, 1216(sp)
	fsw	fa0, 532(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1472(sp)
	flw	fa0, 1212(sp)
	fsw	fa0, 528(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1468(sp)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 524(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1408(sp)
	flw	fa0, 1208(sp)
	fsw	fa0, 520(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1464(sp)
	flw	fa0, 1204(sp)
	fsw	fa0, 516(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1460(sp)
	flw	fa0, 1200(sp)
	fsw	fa0, 512(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1456(sp)
	flw	fa0, 1196(sp)
	fsw	fa0, 508(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1452(sp)
	flw	fa0, 1192(sp)
	fsw	fa0, 504(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1448(sp)
	flw	fa0, 1188(sp)
	fsw	fa0, 484(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1444(sp)
	flw	fa0, 1184(sp)
	fsw	fa0, 472(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1440(sp)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 464(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1420(sp)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 456(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1416(sp)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 448(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1412(sp)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 440(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1436(sp)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 432(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1432(sp)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 424(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1428(sp)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 416(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1424(sp)
	addi	a0, sp, 1280
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s2, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 1404(sp)
	fsw	fa0, 408(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1916(sp)
	flw	fa0, 1400(sp)
	fsw	fa0, 400(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1912(sp)
	flw	fa0, 1396(sp)
	fsw	fa0, 392(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1908(sp)
	flw	fa0, 1392(sp)
	fsw	fa0, 384(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1904(sp)
	flw	fa0, 1388(sp)
	fsw	fa0, 376(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1900(sp)
	flw	fa0, 1384(sp)
	fsw	fa0, 368(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1896(sp)
	flw	fa0, 1380(sp)
	fsw	fa0, 360(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1892(sp)
	flw	fa0, 1376(sp)
	fsw	fa0, 352(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1888(sp)
	flw	fa0, 1372(sp)
	fsw	fa0, 344(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1884(sp)
	flw	fa0, 1368(sp)
	fsw	fa0, 336(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1880(sp)
	flw	fa0, 1364(sp)
	fsw	fa0, 328(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1876(sp)
	flw	fa0, 1360(sp)
	fsw	fa0, 320(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1872(sp)
	flw	fa0, 1356(sp)
	fsw	fa0, 312(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1868(sp)
	flw	fa0, 1352(sp)
	fsw	fa0, 304(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1864(sp)
	flw	fa0, 1348(sp)
	fsw	fa0, 296(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1860(sp)
	flw	fa0, 1344(sp)
	fsw	fa0, 288(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1856(sp)
	flw	fa0, 1340(sp)
	fsw	fa0, 280(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1852(sp)
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 272(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1792(sp)
	flw	fa0, 1336(sp)
	fsw	fa0, 264(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1848(sp)
	flw	fa0, 1332(sp)
	fsw	fa0, 260(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1844(sp)
	flw	fs8, 1328(sp)
	fmv.s	fa0, fs8
	call	cosf
	fsw	fa0, 1840(sp)
	flw	fs9, 1324(sp)
	fmv.s	fa0, fs9
	call	cosf
	fsw	fa0, 1836(sp)
	flw	fs10, 1320(sp)
	fmv.s	fa0, fs10
	call	cosf
	fsw	fa0, 1832(sp)
	flw	fs11, 1316(sp)
	fmv.s	fa0, fs11
	call	cosf
	fsw	fa0, 1828(sp)
	flw	fs0, 1312(sp)
	fmv.s	fa0, fs0
	call	cosf
	fsw	fa0, 1824(sp)
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fs1, v8
	fmv.s	fa0, fs1
	call	cosf
	fsw	fa0, 1804(sp)
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fs2, v8
	fmv.s	fa0, fs2
	call	cosf
	fsw	fa0, 1800(sp)
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fs3, v8
	fmv.s	fa0, fs3
	call	cosf
	fsw	fa0, 1796(sp)
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fs4, v8
	fmv.s	fa0, fs4
	call	cosf
	fsw	fa0, 1820(sp)
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fs5, v8
	fmv.s	fa0, fs5
	call	cosf
	fsw	fa0, 1816(sp)
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fs6, v8
	fmv.s	fa0, fs6
	call	cosf
	fsw	fa0, 1812(sp)
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fs7, v8
	fmv.s	fa0, fs7
	call	cosf
	fsw	fa0, 1808(sp)
	addi	a0, sp, 2047
	addi	a0, a0, 129
	vsetvli	zero, s2, e32, m8, ta, ma
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 2047
	addi	a0, a0, 385
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 1408
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 1792
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	flw	fa0, 408(sp)                    # 4-byte Folded Reload
	.loc	1 20 23                         # k135114294097424.py:20:23
	call	sinf
	fsw	fa0, 2044(sp)
	flw	fa0, 400(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2040(sp)
	flw	fa0, 392(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2036(sp)
	flw	fa0, 384(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2032(sp)
	flw	fa0, 376(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2028(sp)
	flw	fa0, 368(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2024(sp)
	flw	fa0, 360(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2020(sp)
	flw	fa0, 352(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2016(sp)
	flw	fa0, 344(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2012(sp)
	flw	fa0, 336(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2008(sp)
	flw	fa0, 328(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2004(sp)
	flw	fa0, 320(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2000(sp)
	flw	fa0, 312(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1996(sp)
	flw	fa0, 304(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1992(sp)
	flw	fa0, 296(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1988(sp)
	flw	fa0, 288(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1984(sp)
	flw	fa0, 280(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1980(sp)
	flw	fa0, 272(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1920(sp)
	flw	fa0, 264(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1976(sp)
	flw	fa0, 260(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1972(sp)
	fmv.s	fa0, fs8
	call	sinf
	fsw	fa0, 1968(sp)
	fmv.s	fa0, fs9
	call	sinf
	fsw	fa0, 1964(sp)
	fmv.s	fa0, fs10
	call	sinf
	fsw	fa0, 1960(sp)
	fmv.s	fa0, fs11
	call	sinf
	fsw	fa0, 1956(sp)
	fmv.s	fa0, fs0
	call	sinf
	fsw	fa0, 1952(sp)
	fmv.s	fa0, fs1
	call	sinf
	fsw	fa0, 1932(sp)
	fmv.s	fa0, fs2
	call	sinf
	fsw	fa0, 1928(sp)
	fmv.s	fa0, fs3
	call	sinf
	fsw	fa0, 1924(sp)
	fmv.s	fa0, fs4
	call	sinf
	fsw	fa0, 1948(sp)
	fmv.s	fa0, fs5
	call	sinf
	fsw	fa0, 1944(sp)
	fmv.s	fa0, fs6
	call	sinf
	fsw	fa0, 1940(sp)
	fmv.s	fa0, fs7
	call	sinf
	fsw	fa0, 1936(sp)
	flw	fa0, 632(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1660(sp)
	flw	fa0, 624(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1656(sp)
	flw	fa0, 616(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1652(sp)
	flw	fa0, 608(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1648(sp)
	flw	fa0, 600(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1644(sp)
	flw	fa0, 592(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1640(sp)
	flw	fa0, 584(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1636(sp)
	flw	fa0, 576(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1632(sp)
	flw	fa0, 568(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1628(sp)
	flw	fa0, 560(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1624(sp)
	flw	fa0, 552(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1620(sp)
	flw	fa0, 548(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1616(sp)
	flw	fa0, 544(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1612(sp)
	flw	fa0, 540(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1608(sp)
	flw	fa0, 536(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1604(sp)
	flw	fa0, 532(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1600(sp)
	flw	fa0, 528(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1596(sp)
	flw	fa0, 524(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1536(sp)
	flw	fa0, 520(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1592(sp)
	flw	fa0, 516(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1588(sp)
	flw	fa0, 512(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1584(sp)
	flw	fa0, 508(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1580(sp)
	flw	fa0, 504(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1576(sp)
	flw	fa0, 484(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1572(sp)
	flw	fa0, 472(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1568(sp)
	flw	fa0, 464(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1548(sp)
	flw	fa0, 456(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1544(sp)
	flw	fa0, 448(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1540(sp)
	flw	fa0, 440(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1564(sp)
	flw	fa0, 432(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1560(sp)
	flw	fa0, 424(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1556(sp)
	flw	fa0, 416(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1552(sp)
	flw	fa0, 888(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1412(a0)
	flw	fa0, 880(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1416(a0)
	flw	fa0, 872(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1420(a0)
	flw	fa0, 864(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1424(a0)
	flw	fa0, 856(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1428(a0)
	flw	fa0, 848(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1432(a0)
	flw	fa0, 840(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1436(a0)
	flw	fa0, 832(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1440(a0)
	flw	fa0, 824(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1444(a0)
	flw	fa0, 816(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1448(a0)
	flw	fa0, 808(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1452(a0)
	flw	fa0, 800(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1456(a0)
	flw	fa0, 792(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1460(a0)
	flw	fa0, 784(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1464(a0)
	flw	fa0, 776(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1468(a0)
	flw	fa0, 768(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1472(a0)
	flw	fa0, 760(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1476(a0)
	flw	fa0, 752(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1536(a0)
	flw	fa0, 744(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1480(a0)
	flw	fa0, 736(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1484(a0)
	flw	fa0, 728(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1488(a0)
	flw	fa0, 720(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1492(a0)
	flw	fa0, 712(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1496(a0)
	flw	fa0, 704(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1500(a0)
	flw	fa0, 696(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1504(a0)
	flw	fa0, 688(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1524(a0)
	flw	fa0, 680(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1528(a0)
	flw	fa0, 672(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1532(a0)
	flw	fa0, 664(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1508(a0)
	flw	fa0, 656(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1512(a0)
	flw	fa0, 648(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1516(a0)
	flw	fa0, 640(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1520(a0)
	flw	fa0, 1144(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1668(a0)
	flw	fa0, 1136(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1672(a0)
	flw	fa0, 1128(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1676(a0)
	flw	fa0, 1120(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1680(a0)
	flw	fa0, 1112(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1684(a0)
	flw	fa0, 1104(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1688(a0)
	flw	fa0, 1096(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1692(a0)
	flw	fa0, 1088(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1696(a0)
	flw	fa0, 1080(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1700(a0)
	flw	fa0, 1072(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1704(a0)
	flw	fa0, 1064(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1708(a0)
	flw	fa0, 1056(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1712(a0)
	flw	fa0, 1048(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1716(a0)
	flw	fa0, 1040(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1720(a0)
	flw	fa0, 1032(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1724(a0)
	flw	fa0, 1024(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1728(a0)
	flw	fa0, 1016(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1732(a0)
	flw	fa0, 1008(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1792(a0)
	flw	fa0, 1000(sp)                   # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1736(a0)
	flw	fa0, 992(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1740(a0)
	flw	fa0, 984(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1744(a0)
	flw	fa0, 976(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1748(a0)
	flw	fa0, 968(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1752(a0)
	flw	fa0, 960(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1756(a0)
	flw	fa0, 952(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1760(a0)
	flw	fa0, 944(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1780(a0)
	flw	fa0, 936(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1784(a0)
	flw	fa0, 928(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1788(a0)
	flw	fa0, 920(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1764(a0)
	flw	fa0, 912(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1768(a0)
	flw	fa0, 904(sp)                    # 4-byte Folded Reload
	call	sinf
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1772(a0)
	flw	fa0, 896(sp)                    # 4-byte Folded Reload
	call	sinf
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	lui	a0, 1
	add	a0, sp, a0
	fsw	fa0, -1776(a0)
	addi	a0, sp, 1920
	vsetvli	zero, s2, e32, m8, ta, ma
	vle32.v	v8, (a0)
	addi	a0, sp, 1536
	vle32.v	v16, (a0)
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 19                         # k135114294097424.py:22:19
	vfmul.vv	v8, v8, v24
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 23 19                         # k135114294097424.py:23:19
	vfmadd.vv	v16, v24, v8
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 19                         # k135114294097424.py:22:19
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 2047
	addi	a0, a0, 513
	.loc	1 20 23                         # k135114294097424.py:20:23
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 2047
	addi	a0, a0, 257
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 23 19                         # k135114294097424.py:23:19
	vfmadd.vv	v16, v8, v24
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 19                         # k135114294097424.py:22:19
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 23 19                         # k135114294097424.py:23:19
	vfmadd.vv	v16, v24, v8
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 19                         # k135114294097424.py:22:19
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 23 19                         # k135114294097424.py:23:19
	vfmadd.vv	v16, v24, v8
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	li	a0, 64
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 24 40                         # k135114294097424.py:24:40
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	li	a2, 32
	vslideup.vx	v8, v16, a2
	ld	a1, 496(sp)                     # 8-byte Folded Reload
	vse16.v	v8, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 152
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 4
	addi	a3, a3, -1088
	add	a1, a1, a3
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	li	a1, 32
	vnsrl.wi	v8, v16, 16
	csrr	a2, vlenb
	li	a3, 160
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1088
	add	a2, a2, a3
	vl8r.v	v24, (a2)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v8, v16, a1
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1088
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	vse16.v	v8, (a0), v0.t
	.loc	1 24 4 epilogue_begin is_stmt 0 # k135114294097424.py:24:4
	addi	sp, s0, -2032
	.cfi_def_cfa sp, 2032
	ld	ra, 2024(sp)                    # 8-byte Folded Reload
	ld	s0, 2016(sp)                    # 8-byte Folded Reload
	ld	s2, 2008(sp)                    # 8-byte Folded Reload
	ld	s3, 2000(sp)                    # 8-byte Folded Reload
	ld	s4, 1992(sp)                    # 8-byte Folded Reload
	ld	s5, 1984(sp)                    # 8-byte Folded Reload
	ld	s6, 1976(sp)                    # 8-byte Folded Reload
	ld	s7, 1968(sp)                    # 8-byte Folded Reload
	ld	s8, 1960(sp)                    # 8-byte Folded Reload
	ld	s9, 1952(sp)                    # 8-byte Folded Reload
	ld	s10, 1944(sp)                   # 8-byte Folded Reload
	ld	s11, 1936(sp)                   # 8-byte Folded Reload
	fld	fs0, 1928(sp)                   # 8-byte Folded Reload
	fld	fs1, 1920(sp)                   # 8-byte Folded Reload
	fld	fs2, 1912(sp)                   # 8-byte Folded Reload
	fld	fs3, 1904(sp)                   # 8-byte Folded Reload
	fld	fs4, 1896(sp)                   # 8-byte Folded Reload
	fld	fs5, 1888(sp)                   # 8-byte Folded Reload
	fld	fs6, 1880(sp)                   # 8-byte Folded Reload
	fld	fs7, 1872(sp)                   # 8-byte Folded Reload
	fld	fs8, 1864(sp)                   # 8-byte Folded Reload
	fld	fs9, 1856(sp)                   # 8-byte Folded Reload
	fld	fs10, 1848(sp)                  # 8-byte Folded Reload
	fld	fs11, 1840(sp)                  # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
	.cfi_restore s6
	.cfi_restore s7
	.cfi_restore s8
	.cfi_restore s9
	.cfi_restore s10
	.cfi_restore s11
	.cfi_restore fs0
	.cfi_restore fs1
	.cfi_restore fs2
	.cfi_restore fs3
	.cfi_restore fs4
	.cfi_restore fs5
	.cfi_restore fs6
	.cfi_restore fs7
	.cfi_restore fs8
	.cfi_restore fs9
	.cfi_restore fs10
	.cfi_restore fs11
	addi	sp, sp, 2032
	.cfi_def_cfa_offset 0
	ret
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8, .Lfunc_end0-triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.half	4                               # DWARF version number
	.word	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.word	.Linfo_string0                  # DW_AT_producer
	.half	2                               # DW_AT_language
	.word	.Linfo_string1                  # DW_AT_name
	.word	.Lline_table_start0             # DW_AT_stmt_list
	.word	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        # string offset=0 ; triton
.Linfo_string1:
	.asciz	"k135114294097424.py"           # string offset=7 ; k135114294097424.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
