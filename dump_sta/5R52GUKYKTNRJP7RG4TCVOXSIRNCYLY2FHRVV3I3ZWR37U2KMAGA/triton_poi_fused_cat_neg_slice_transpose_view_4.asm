	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused_cat_neg_slice_transpose_view_4 # -- Begin function triton_poi_fused_cat_neg_slice_transpose_view_4
	.p2align	2
	.type	triton_poi_fused_cat_neg_slice_transpose_view_4,@function
triton_poi_fused_cat_neg_slice_transpose_view_4: # @triton_poi_fused_cat_neg_slice_transpose_view_4
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135112023917904.py"
	.loc	1 2 0                           # k135112023917904.py:2:0
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
	fsd	fs0, 1984(sp)                   # 8-byte Folded Spill
	fsd	fs1, 1976(sp)                   # 8-byte Folded Spill
	fsd	fs2, 1968(sp)                   # 8-byte Folded Spill
	fsd	fs3, 1960(sp)                   # 8-byte Folded Spill
	fsd	fs4, 1952(sp)                   # 8-byte Folded Spill
	fsd	fs5, 1944(sp)                   # 8-byte Folded Spill
	fsd	fs6, 1936(sp)                   # 8-byte Folded Spill
	fsd	fs7, 1928(sp)                   # 8-byte Folded Spill
	fsd	fs8, 1920(sp)                   # 8-byte Folded Spill
	fsd	fs9, 1912(sp)                   # 8-byte Folded Spill
	fsd	fs10, 1904(sp)                  # 8-byte Folded Spill
	fsd	fs11, 1896(sp)                  # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s2, -24
	.cfi_offset s3, -32
	.cfi_offset s4, -40
	.cfi_offset fs0, -48
	.cfi_offset fs1, -56
	.cfi_offset fs2, -64
	.cfi_offset fs3, -72
	.cfi_offset fs4, -80
	.cfi_offset fs5, -88
	.cfi_offset fs6, -96
	.cfi_offset fs7, -104
	.cfi_offset fs8, -112
	.cfi_offset fs9, -120
	.cfi_offset fs10, -128
	.cfi_offset fs11, -136
	addi	s0, sp, 2032
	.cfi_def_cfa s0, 0
	.cfi_remember_state
	lui	a2, 3
	addi	a2, a2, -1504
	sub	sp, sp, a2
	csrr	a2, vlenb
	li	a4, 48
	mul	a2, a2, a4
	sub	sp, sp, a2
	andi	sp, sp, -128
	mv	s2, a1
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135112023917904.py:4:33
	slli	a1, a3, 6
	li	a2, 32
	li	a4, -32
	li	a3, -64
	.loc	1 5 23                          # k135112023917904.py:5:23
	vsetvli	zero, a2, e32, m8, ta, ma
	vmv.v.x	v8, a1
	vid.v	v16
	vor.vx	v0, v16, a1
	.loc	1 8 19                          # k135112023917904.py:8:19
	vsra.vi	v8, v8, 31
	vsrl.vi	v8, v8, 27
	csrr	a5, vlenb
	slli	a5, a5, 5
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 384
	add	a5, a5, a6
	vs8r.v	v8, (a5)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v0, v8
	.loc	1 7 19                          # k135112023917904.py:7:19
	vand.vx	v24, v8, a4
	.loc	1 9 38                          # k135112023917904.py:9:38
	vadd.vv	v8, v8, v8
	.loc	1 7 19                          # k135112023917904.py:7:19
	vsub.vv	v24, v0, v24
	.loc	1 9 38                          # k135112023917904.py:9:38
	vand.vx	v8, v8, a3
	.loc	1 9 35 is_stmt 0                # k135112023917904.py:9:35
	vadd.vv	v8, v8, v24
	li	a5, 64
	.loc	1 5 23 is_stmt 1                # k135112023917904.py:5:23
	vadd.vx	v16, v16, a2
	vor.vx	v24, v16, a1
	csrr	a1, vlenb
	li	a6, 24
	mul	a1, a1, a6
	add	a1, sp, a1
	lui	a6, 3
	addi	a6, a6, 384
	add	a1, a1, a6
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135112023917904.py:6:21
	vmslt.vx	v16, v0, a5
	csrr	a1, vlenb
	li	a6, 40
	mul	a1, a1, a6
	add	a1, sp, a1
	lui	a6, 3
	addi	a6, a6, 384
	add	a1, a1, a6
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmslt.vx	v17, v24, a5
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v16, v17, 4
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vmv.x.s	a1, v16
	andi	a5, a1, 1
	vsext.vf2	v16, v8
	csrr	a6, vlenb
	slli	a6, a6, 4
	add	a6, sp, a6
	lui	a7, 3
	addi	a7, a7, 384
	add	a6, a6, a7
	vs8r.v	v16, (a6)                       # vscale x 64-byte Folded Spill
	vadd.vv	v24, v16, v16
	vadd.vx	v24, v24, a0
	beqz	a5, .LBB0_2
# %bb.1:                                # %cond.load
	vmv.x.s	a5, v24
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fs1, zero
	fmv.w.x	fa0, a5
	fmv.s	fs2, fs1
	fmv.s	fs0, fs1
	fmv.s	fs11, fs1
	fmv.s	fs10, fs1
	fmv.s	fs9, fs1
	fmv.s	fs8, fs1
	fmv.s	fs7, fs1
	fmv.s	fs6, fs1
	fmv.s	fs5, fs1
	fmv.s	fs4, fs1
	fmv.s	fs3, fs1
	fsw	fs1, 52(sp)                     # 4-byte Folded Spill
	fsw	fs1, 56(sp)                     # 4-byte Folded Spill
	fsw	fs1, 60(sp)                     # 4-byte Folded Spill
	fsw	fs1, 64(sp)                     # 4-byte Folded Spill
	fsw	fs1, 68(sp)                     # 4-byte Folded Spill
	fsw	fs1, 72(sp)                     # 4-byte Folded Spill
	fsw	fs1, 76(sp)                     # 4-byte Folded Spill
	fsw	fs1, 80(sp)                     # 4-byte Folded Spill
	fsw	fs1, 84(sp)                     # 4-byte Folded Spill
	fsw	fs1, 88(sp)                     # 4-byte Folded Spill
	fsw	fs1, 92(sp)                     # 4-byte Folded Spill
	fsw	fs1, 96(sp)                     # 4-byte Folded Spill
	fsw	fs1, 100(sp)                    # 4-byte Folded Spill
	fsw	fs1, 104(sp)                    # 4-byte Folded Spill
	fsw	fs1, 108(sp)                    # 4-byte Folded Spill
	fsw	fs1, 112(sp)                    # 4-byte Folded Spill
	fsw	fs1, 116(sp)                    # 4-byte Folded Spill
	fsw	fs1, 120(sp)                    # 4-byte Folded Spill
	fsw	fs1, 124(sp)                    # 4-byte Folded Spill
	fsw	fs1, 128(sp)                    # 4-byte Folded Spill
	fsw	fs1, 132(sp)                    # 4-byte Folded Spill
	fsw	fs1, 136(sp)                    # 4-byte Folded Spill
	fsw	fs1, 140(sp)                    # 4-byte Folded Spill
	fsw	fs1, 144(sp)                    # 4-byte Folded Spill
	fsw	fs1, 148(sp)                    # 4-byte Folded Spill
	fsw	fs1, 152(sp)                    # 4-byte Folded Spill
	fsw	fs1, 156(sp)                    # 4-byte Folded Spill
	fsw	fs1, 160(sp)                    # 4-byte Folded Spill
	fsw	fs1, 164(sp)                    # 4-byte Folded Spill
	fsw	fs1, 168(sp)                    # 4-byte Folded Spill
	fsw	fs1, 172(sp)                    # 4-byte Folded Spill
	fsw	fs1, 176(sp)                    # 4-byte Folded Spill
	fsw	fs1, 180(sp)                    # 4-byte Folded Spill
	fsw	fs1, 184(sp)                    # 4-byte Folded Spill
	fsw	fs1, 188(sp)                    # 4-byte Folded Spill
	fsw	fs1, 192(sp)                    # 4-byte Folded Spill
	fsw	fs1, 196(sp)                    # 4-byte Folded Spill
	fsw	fs1, 200(sp)                    # 4-byte Folded Spill
	fsw	fs1, 204(sp)                    # 4-byte Folded Spill
	fsw	fs1, 208(sp)                    # 4-byte Folded Spill
	fsw	fs1, 212(sp)                    # 4-byte Folded Spill
	fsw	fs1, 216(sp)                    # 4-byte Folded Spill
	fsw	fs1, 220(sp)                    # 4-byte Folded Spill
	fsw	fs1, 224(sp)                    # 4-byte Folded Spill
	fsw	fs1, 228(sp)                    # 4-byte Folded Spill
	fsw	fs1, 232(sp)                    # 4-byte Folded Spill
	fsw	fs1, 236(sp)                    # 4-byte Folded Spill
	fsw	fs1, 240(sp)                    # 4-byte Folded Spill
	fsw	fs1, 244(sp)                    # 4-byte Folded Spill
	fsw	fs1, 248(sp)                    # 4-byte Folded Spill
	fsw	fs1, 252(sp)                    # 4-byte Folded Spill
	andi	a5, a1, 2
	bnez	a5, .LBB0_3
	j	.LBB0_4
.LBB0_2:
	.loc	1 0 43 is_stmt 0                # k135112023917904.py:0:43
	fmv.w.x	fa0, zero
	fmv.s	fs1, fa0
	fmv.s	fs2, fa0
	fmv.s	fs0, fa0
	fmv.s	fs11, fa0
	fmv.s	fs10, fa0
	fmv.s	fs9, fa0
	fmv.s	fs8, fa0
	fmv.s	fs7, fa0
	fmv.s	fs6, fa0
	fmv.s	fs5, fa0
	fmv.s	fs4, fa0
	fmv.s	fs3, fa0
	fsw	fa0, 52(sp)                     # 4-byte Folded Spill
	fsw	fa0, 56(sp)                     # 4-byte Folded Spill
	fsw	fa0, 60(sp)                     # 4-byte Folded Spill
	fsw	fa0, 64(sp)                     # 4-byte Folded Spill
	fsw	fa0, 68(sp)                     # 4-byte Folded Spill
	fsw	fa0, 72(sp)                     # 4-byte Folded Spill
	fsw	fa0, 76(sp)                     # 4-byte Folded Spill
	fsw	fa0, 80(sp)                     # 4-byte Folded Spill
	fsw	fa0, 84(sp)                     # 4-byte Folded Spill
	fsw	fa0, 88(sp)                     # 4-byte Folded Spill
	fsw	fa0, 92(sp)                     # 4-byte Folded Spill
	fsw	fa0, 96(sp)                     # 4-byte Folded Spill
	fsw	fa0, 100(sp)                    # 4-byte Folded Spill
	fsw	fa0, 104(sp)                    # 4-byte Folded Spill
	fsw	fa0, 108(sp)                    # 4-byte Folded Spill
	fsw	fa0, 112(sp)                    # 4-byte Folded Spill
	fsw	fa0, 116(sp)                    # 4-byte Folded Spill
	fsw	fa0, 120(sp)                    # 4-byte Folded Spill
	fsw	fa0, 124(sp)                    # 4-byte Folded Spill
	fsw	fa0, 128(sp)                    # 4-byte Folded Spill
	fsw	fa0, 132(sp)                    # 4-byte Folded Spill
	fsw	fa0, 136(sp)                    # 4-byte Folded Spill
	fsw	fa0, 140(sp)                    # 4-byte Folded Spill
	fsw	fa0, 144(sp)                    # 4-byte Folded Spill
	fsw	fa0, 148(sp)                    # 4-byte Folded Spill
	fsw	fa0, 152(sp)                    # 4-byte Folded Spill
	fsw	fa0, 156(sp)                    # 4-byte Folded Spill
	fsw	fa0, 160(sp)                    # 4-byte Folded Spill
	fsw	fa0, 164(sp)                    # 4-byte Folded Spill
	fsw	fa0, 168(sp)                    # 4-byte Folded Spill
	fsw	fa0, 172(sp)                    # 4-byte Folded Spill
	fsw	fa0, 176(sp)                    # 4-byte Folded Spill
	fsw	fa0, 180(sp)                    # 4-byte Folded Spill
	fsw	fa0, 184(sp)                    # 4-byte Folded Spill
	fsw	fa0, 188(sp)                    # 4-byte Folded Spill
	fsw	fa0, 192(sp)                    # 4-byte Folded Spill
	fsw	fa0, 196(sp)                    # 4-byte Folded Spill
	fsw	fa0, 200(sp)                    # 4-byte Folded Spill
	fsw	fa0, 204(sp)                    # 4-byte Folded Spill
	fsw	fa0, 208(sp)                    # 4-byte Folded Spill
	fsw	fa0, 212(sp)                    # 4-byte Folded Spill
	fsw	fa0, 216(sp)                    # 4-byte Folded Spill
	fsw	fa0, 220(sp)                    # 4-byte Folded Spill
	fsw	fa0, 224(sp)                    # 4-byte Folded Spill
	fsw	fa0, 228(sp)                    # 4-byte Folded Spill
	fsw	fa0, 232(sp)                    # 4-byte Folded Spill
	fsw	fa0, 236(sp)                    # 4-byte Folded Spill
	fsw	fa0, 240(sp)                    # 4-byte Folded Spill
	fsw	fa0, 244(sp)                    # 4-byte Folded Spill
	fsw	fa0, 248(sp)                    # 4-byte Folded Spill
	fsw	fa0, 252(sp)                    # 4-byte Folded Spill
	.loc	1 9 43                          # k135112023917904.py:9:43
	andi	a5, a1, 2
	beqz	a5, .LBB0_4
.LBB0_3:                                # %cond.load1
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v24, 1
	vmv.x.s	a5, v16
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fs1, a5
.LBB0_4:                                # %else2
	andi	a5, a1, 4
	bnez	a5, .LBB0_17
# %bb.5:                                # %else5
	andi	a5, a1, 8
	bnez	a5, .LBB0_18
.LBB0_6:                                # %else8
	andi	a6, a1, 16
	lui	a5, 3
	addi	a5, a5, -1496
	add	a5, sp, a5
	bnez	a6, .LBB0_19
.LBB0_7:                                # %else11
	andi	a6, a1, 32
	bnez	a6, .LBB0_20
.LBB0_8:                                # %else14
	andi	a6, a1, 64
	bnez	a6, .LBB0_21
.LBB0_9:                                # %else17
	andi	a6, a1, 128
	bnez	a6, .LBB0_22
.LBB0_10:                               # %else20
	andi	a6, a1, 256
	bnez	a6, .LBB0_23
.LBB0_11:                               # %else23
	andi	a6, a1, 512
	bnez	a6, .LBB0_24
.LBB0_12:                               # %else26
	andi	a6, a1, 1024
	bnez	a6, .LBB0_25
.LBB0_13:                               # %else29
	slli	a6, a1, 52
	bltz	a6, .LBB0_26
.LBB0_14:                               # %else32
	slli	a6, a1, 51
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	bltz	a6, .LBB0_27
.LBB0_15:                               # %else35
	slli	a6, a1, 50
	bgez	a6, .LBB0_28
.LBB0_16:                               # %cond.load37
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a6, 11
	slli	a6, a6, 10
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 576(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 52(sp)                     # 4-byte Folded Spill
	slli	a6, a1, 49
	vsext.vf2	v16, v8
	bltz	a6, .LBB0_29
	j	.LBB0_30
.LBB0_17:                               # %cond.load4
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v24, 2
	vmv.x.s	a5, v16
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fs2, a5
	andi	a5, a1, 8
	beqz	a5, .LBB0_6
.LBB0_18:                               # %cond.load7
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v24, 3
	vmv.x.s	a5, v16
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fs0, a5
	andi	a6, a1, 16
	lui	a5, 3
	addi	a5, a5, -1496
	add	a5, sp, a5
	beqz	a6, .LBB0_7
.LBB0_19:                               # %cond.load10
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, 128
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1656(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs11, a6
	andi	a6, a1, 32
	beqz	a6, .LBB0_8
.LBB0_20:                               # %cond.load13
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1536(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs10, a6
	andi	a6, a1, 64
	beqz	a6, .LBB0_9
.LBB0_21:                               # %cond.load16
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -128
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1416(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs9, a6
	andi	a6, a1, 128
	beqz	a6, .LBB0_10
.LBB0_22:                               # %cond.load19
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -256
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1296(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs8, a6
	andi	a6, a1, 256
	beqz	a6, .LBB0_11
.LBB0_23:                               # %cond.load22
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -384
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1176(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs7, a6
	andi	a6, a1, 512
	beqz	a6, .LBB0_12
.LBB0_24:                               # %cond.load25
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a6, 23
	slli	a6, a6, 9
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1056(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs6, a6
	andi	a6, a1, 1024
	beqz	a6, .LBB0_13
.LBB0_25:                               # %cond.load28
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -640
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 936(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs5, a6
	slli	a6, a1, 52
	bgez	a6, .LBB0_14
.LBB0_26:                               # %cond.load31
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -768
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 816(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs4, a6
	slli	a6, a1, 51
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	bgez	a6, .LBB0_15
.LBB0_27:                               # %cond.load34
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -896
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 696(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs3, a6
	slli	a6, a1, 50
	bltz	a6, .LBB0_16
.LBB0_28:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	a6, a1, 49
	vsext.vf2	v16, v8
	bgez	a6, .LBB0_30
.LBB0_29:                               # %cond.load40
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -1152
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vse64.v	v24, (a6)
	ld	a6, 456(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 56(sp)                     # 4-byte Folded Spill
.LBB0_30:                               # %else41
	slli	a6, a1, 48
	vadd.vv	v8, v16, v16
	bltz	a6, .LBB0_49
# %bb.31:                               # %else44
	slli	a6, a1, 47
	vadd.vx	v24, v8, a0
	bltz	a6, .LBB0_50
.LBB0_32:                               # %else47
	slli	a6, a1, 46
	bltz	a6, .LBB0_51
.LBB0_33:                               # %else50
	slli	a6, a1, 45
	bltz	a6, .LBB0_52
.LBB0_34:                               # %else53
	slli	a6, a1, 44
	bltz	a6, .LBB0_53
.LBB0_35:                               # %else56
	slli	a6, a1, 43
	bltz	a6, .LBB0_54
.LBB0_36:                               # %else59
	slli	a6, a1, 42
	bltz	a6, .LBB0_55
.LBB0_37:                               # %else62
	slli	a6, a1, 41
	lui	a5, 2
	addi	a5, a5, 464
	add	a5, sp, a5
	bltz	a6, .LBB0_56
.LBB0_38:                               # %else65
	slli	a6, a1, 40
	bltz	a6, .LBB0_57
.LBB0_39:                               # %else68
	slli	a6, a1, 39
	bgez	a6, .LBB0_41
.LBB0_40:                               # %cond.load70
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -1920
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1776(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 96(sp)                     # 4-byte Folded Spill
.LBB0_41:                               # %else71
	slli	a6, a1, 38
	csrr	a7, vlenb
	li	t0, 40
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 3
	addi	t0, t0, 384
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	csrr	a7, vlenb
	slli	a7, a7, 5
	add	a7, sp, a7
	lui	t0, 3
	addi	t0, t0, 384
	add	a7, a7, t0
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e32, m8, ta, ma
	vadd.vv	v0, v8, v0
	bgez	a6, .LBB0_43
# %bb.42:                               # %cond.load73
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a6, 5
	slli	a6, a6, 11
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1656(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 100(sp)                    # 4-byte Folded Spill
.LBB0_43:                               # %else74
	.loc	1 0 43                          # k135112023917904.py:0:43
	vsetvli	zero, a2, e32, m8, ta, ma
	vand.vx	v8, v0, a4
	.loc	1 9 43                          # k135112023917904.py:9:43
	slli	a4, a1, 37
	vadd.vv	v0, v0, v0
	lui	a6, 3
	addi	a6, a6, 384
	add	a6, sp, a6
	vs8r.v	v16, (a6)                       # vscale x 64-byte Folded Spill
	bgez	a4, .LBB0_45
# %bb.44:                               # %cond.load76
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a4, 2
	addi	a4, a4, 1920
	add	a4, sp, a4
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 1536(a5)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 104(sp)                    # 4-byte Folded Spill
.LBB0_45:                               # %else77
	.loc	1 0 43                          # k135112023917904.py:0:43
	vsetvli	zero, a2, e32, m8, ta, ma
	vand.vx	v0, v0, a3
	.loc	1 9 43                          # k135112023917904.py:9:43
	slli	a3, a1, 36
	csrr	a4, vlenb
	li	a6, 40
	mul	a4, a4, a6
	add	a4, sp, a4
	lui	a6, 3
	addi	a6, a6, 384
	add	a4, a4, a6
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	vsub.vv	v8, v16, v8
	bltz	a3, .LBB0_58
# %bb.46:                               # %else80
	slli	a3, a1, 35
	vsetvli	zero, a2, e32, m8, ta, ma
	vadd.vv	v8, v0, v8
	bltz	a3, .LBB0_59
.LBB0_47:                               # %else83
	slli	a2, a1, 34
	bgez	a2, .LBB0_60
.LBB0_48:                               # %cond.load85
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a2, 19
	slli	a2, a2, 9
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 1176(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 116(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 33
	vsext.vf2	v16, v8
	bltz	a2, .LBB0_61
	j	.LBB0_62
.LBB0_49:                               # %cond.load43
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -1280
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vse64.v	v24, (a6)
	ld	a6, 336(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 60(sp)                     # 4-byte Folded Spill
	slli	a6, a1, 47
	vadd.vx	v24, v8, a0
	bgez	a6, .LBB0_32
.LBB0_50:                               # %cond.load46
	vmv.x.s	a6, v24
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 64(sp)                     # 4-byte Folded Spill
	slli	a6, a1, 46
	bgez	a6, .LBB0_33
.LBB0_51:                               # %cond.load49
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 68(sp)                     # 4-byte Folded Spill
	slli	a6, a1, 45
	bgez	a6, .LBB0_34
.LBB0_52:                               # %cond.load52
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 72(sp)                     # 4-byte Folded Spill
	slli	a6, a1, 44
	bgez	a6, .LBB0_35
.LBB0_53:                               # %cond.load55
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 76(sp)                     # 4-byte Folded Spill
	slli	a6, a1, 43
	bgez	a6, .LBB0_36
.LBB0_54:                               # %cond.load58
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -1408
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 120(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 80(sp)                     # 4-byte Folded Spill
	slli	a6, a1, 42
	bgez	a6, .LBB0_37
.LBB0_55:                               # %cond.load61
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a6, 21
	slli	a6, a6, 9
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a5, 0(a5)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 84(sp)                     # 4-byte Folded Spill
	slli	a6, a1, 41
	lui	a5, 2
	addi	a5, a5, 464
	add	a5, sp, a5
	bgez	a6, .LBB0_38
.LBB0_56:                               # %cond.load64
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -1664
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 2016(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 88(sp)                     # 4-byte Folded Spill
	slli	a6, a1, 40
	bgez	a6, .LBB0_39
.LBB0_57:                               # %cond.load67
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a6, 3
	addi	a6, a6, -1792
	add	a6, sp, a6
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1896(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 92(sp)                     # 4-byte Folded Spill
	slli	a6, a1, 39
	bltz	a6, .LBB0_40
	j	.LBB0_41
.LBB0_58:                               # %cond.load79
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a3, 2
	addi	a3, a3, 1792
	add	a3, sp, a3
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a3)
	ld	a3, 1416(a5)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 108(sp)                    # 4-byte Folded Spill
	slli	a3, a1, 35
	vsetvli	zero, a2, e32, m8, ta, ma
	vadd.vv	v8, v0, v8
	bgez	a3, .LBB0_47
.LBB0_59:                               # %cond.load82
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a2, 2
	addi	a2, a2, 1664
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 1296(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 112(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 34
	bltz	a2, .LBB0_48
.LBB0_60:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	a2, a1, 33
	vsext.vf2	v16, v8
	bgez	a2, .LBB0_62
.LBB0_61:                               # %cond.load88
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a2, 2
	addi	a2, a2, 1408
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vse64.v	v24, (a2)
	ld	a2, 1056(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 120(sp)                    # 4-byte Folded Spill
.LBB0_62:                               # %else89
	slli	a2, a1, 32
	vadd.vv	v0, v16, v16
	bltz	a2, .LBB0_79
# %bb.63:                               # %else92
	slli	a2, a1, 31
	vadd.vx	v24, v0, a0
	bgez	a2, .LBB0_65
.LBB0_64:                               # %cond.load94
	vmv.x.s	a2, v24
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 128(sp)                    # 4-byte Folded Spill
.LBB0_65:                               # %else95
	slli	a2, a1, 30
	csrr	a3, vlenb
	slli	a3, a3, 3
	add	a3, sp, a3
	lui	a4, 3
	addi	a4, a4, 384
	add	a3, a3, a4
	vs8r.v	v16, (a3)                       # vscale x 64-byte Folded Spill
	bltz	a2, .LBB0_80
# %bb.66:                               # %else98
	slli	a2, a1, 29
	bltz	a2, .LBB0_81
.LBB0_67:                               # %else101
	slli	a2, a1, 28
	bltz	a2, .LBB0_82
.LBB0_68:                               # %else104
	slli	a2, a1, 27
	bltz	a2, .LBB0_83
.LBB0_69:                               # %else107
	slli	a2, a1, 26
	bltz	a2, .LBB0_84
.LBB0_70:                               # %else110
	slli	a2, a1, 25
	bltz	a2, .LBB0_85
.LBB0_71:                               # %else113
	slli	a2, a1, 24
	bltz	a2, .LBB0_86
.LBB0_72:                               # %else116
	slli	a2, a1, 23
	bltz	a2, .LBB0_87
.LBB0_73:                               # %else119
	slli	a2, a1, 22
	bltz	a2, .LBB0_88
.LBB0_74:                               # %else122
	slli	a2, a1, 21
	bltz	a2, .LBB0_89
.LBB0_75:                               # %else125
	slli	a3, a1, 20
	lui	a2, 2
	addi	a2, a2, -1672
	add	a2, sp, a2
	bltz	a3, .LBB0_90
.LBB0_76:                               # %else128
	slli	a3, a1, 19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	bltz	a3, .LBB0_91
.LBB0_77:                               # %else131
	slli	a3, a1, 18
	bgez	a3, .LBB0_92
.LBB0_78:                               # %cond.load133
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a3, 2
	add	a3, sp, a3
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a3)
	ld	a3, 1776(a2)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 180(sp)                    # 4-byte Folded Spill
	slli	a3, a1, 17
	vsext.vf2	v16, v8
	bltz	a3, .LBB0_93
	j	.LBB0_94
.LBB0_79:                               # %cond.load91
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a2, 2
	addi	a2, a2, 1280
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vse64.v	v24, (a2)
	ld	a2, 936(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 124(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 31
	vadd.vx	v24, v0, a0
	bltz	a2, .LBB0_64
	j	.LBB0_65
.LBB0_80:                               # %cond.load97
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v24, 1
	vmv.x.s	a2, v16
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 132(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 29
	bgez	a2, .LBB0_67
.LBB0_81:                               # %cond.load100
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v24, 2
	vmv.x.s	a2, v16
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 136(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 28
	bgez	a2, .LBB0_68
.LBB0_82:                               # %cond.load103
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v24, 3
	vmv.x.s	a2, v16
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 140(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 27
	bgez	a2, .LBB0_69
.LBB0_83:                               # %cond.load106
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a2, 2
	addi	a2, a2, 1152
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 720(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 144(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 26
	bgez	a2, .LBB0_70
.LBB0_84:                               # %cond.load109
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a2, 9
	slli	a2, a2, 10
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 600(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 148(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 25
	bgez	a2, .LBB0_71
.LBB0_85:                               # %cond.load112
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a2, 2
	addi	a2, a2, 896
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 480(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 152(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 24
	bgez	a2, .LBB0_72
.LBB0_86:                               # %cond.load115
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a2, 2
	addi	a2, a2, 768
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 360(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 156(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 23
	bgez	a2, .LBB0_73
.LBB0_87:                               # %cond.load118
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a2, 2
	addi	a2, a2, 640
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 240(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 160(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 22
	bgez	a2, .LBB0_74
.LBB0_88:                               # %cond.load121
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a2, 17
	slli	a2, a2, 9
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 120(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 164(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 21
	bgez	a2, .LBB0_75
.LBB0_89:                               # %cond.load124
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a2, 2
	addi	a2, a2, 384
	add	a2, sp, a2
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 0(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 168(sp)                    # 4-byte Folded Spill
	slli	a3, a1, 20
	lui	a2, 2
	addi	a2, a2, -1672
	add	a2, sp, a2
	bgez	a3, .LBB0_76
.LBB0_90:                               # %cond.load127
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a3, 2
	addi	a3, a3, 256
	add	a3, sp, a3
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a3)
	ld	a3, 2016(a2)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 172(sp)                    # 4-byte Folded Spill
	slli	a3, a1, 19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	bgez	a3, .LBB0_77
.LBB0_91:                               # %cond.load130
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a3, 2
	addi	a3, a3, 128
	add	a3, sp, a3
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a3)
	ld	a3, 1896(a2)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 176(sp)                    # 4-byte Folded Spill
	slli	a3, a1, 18
	bltz	a3, .LBB0_78
.LBB0_92:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	a3, a1, 17
	vsext.vf2	v16, v8
	bgez	a3, .LBB0_94
.LBB0_93:                               # %cond.load136
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a3, 2
	addi	a3, a3, -128
	add	a3, sp, a3
	.loc	1 9 43                          # k135112023917904.py:9:43
	vse64.v	v24, (a3)
	ld	a3, 1656(a2)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 184(sp)                    # 4-byte Folded Spill
.LBB0_94:                               # %else137
	slli	a3, a1, 16
	vadd.vv	v8, v16, v16
	bltz	a3, .LBB0_185
# %bb.95:                               # %else140
	slli	a3, a1, 15
	vadd.vx	v8, v8, a0
	bgez	a3, .LBB0_97
.LBB0_96:                               # %cond.load142
	vmv.x.s	a0, v8
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 192(sp)                    # 4-byte Folded Spill
.LBB0_97:                               # %else143
	slli	a0, a1, 14
	csrr	a3, vlenb
	slli	a3, a3, 5
	add	a3, sp, a3
	lui	a4, 3
	addi	a4, a4, 384
	add	a3, a3, a4
	vs8r.v	v16, (a3)                       # vscale x 64-byte Folded Spill
	bltz	a0, .LBB0_186
# %bb.98:                               # %else146
	slli	a0, a1, 13
	bltz	a0, .LBB0_187
.LBB0_99:                               # %else149
	slli	a0, a1, 12
	bltz	a0, .LBB0_188
.LBB0_100:                              # %else152
	slli	a0, a1, 11
	bltz	a0, .LBB0_189
.LBB0_101:                              # %else155
	slli	a0, a1, 10
	bltz	a0, .LBB0_190
.LBB0_102:                              # %else158
	slli	a0, a1, 9
	bltz	a0, .LBB0_191
.LBB0_103:                              # %else161
	slli	a0, a1, 8
	bltz	a0, .LBB0_192
.LBB0_104:                              # %else164
	slli	a0, a1, 7
	bltz	a0, .LBB0_193
.LBB0_105:                              # %else167
	slli	a0, a1, 6
	bltz	a0, .LBB0_194
.LBB0_106:                              # %else170
	slli	a0, a1, 5
	bltz	a0, .LBB0_195
.LBB0_107:                              # %else173
	slli	a0, a1, 4
	bltz	a0, .LBB0_196
.LBB0_108:                              # %else176
	slli	a0, a1, 3
	bltz	a0, .LBB0_197
.LBB0_109:                              # %else179
	slli	a0, a1, 2
	bltz	a0, .LBB0_198
.LBB0_110:                              # %else182
	slli	a0, a1, 1
	bltz	a0, .LBB0_199
.LBB0_111:                              # %else185
	bgez	a1, .LBB0_113
.LBB0_112:                              # %cond.load187
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a0, 25
	slli	a0, a0, 8
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 252(sp)                    # 4-byte Folded Spill
.LBB0_113:                              # %else188
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a0, 32
	li	a1, 64
	csrr	a2, vlenb
	li	a3, 40
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 3
	addi	a3, a3, 384
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 6 21 is_stmt 1                # k135112023917904.py:6:21
	vsetvli	zero, a0, e32, m8, ta, ma
	vmslt.vx	v8, v16, a1
	csrr	a0, vlenb
	li	a2, 24
	mul	a0, a0, a2
	add	a0, sp, a0
	lui	a2, 3
	addi	a2, a2, 384
	add	a0, a0, a2
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v16, v24, a1
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v16, v8, 4
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 384
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	vmv.x.s	s3, v16
	andi	a0, s3, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bnez	a0, .LBB0_200
# %bb.114:                              # %else192
	andi	a0, s3, 2
	bnez	a0, .LBB0_201
.LBB0_115:                              # %else195
	andi	a0, s3, 4
	bnez	a0, .LBB0_202
.LBB0_116:                              # %else198
	andi	a0, s3, 8
	bnez	a0, .LBB0_203
.LBB0_117:                              # %else201
	andi	a0, s3, 16
	lui	a1, 1
	addi	a1, a1, 312
	add	s4, sp, a1
	bnez	a0, .LBB0_204
.LBB0_118:                              # %else204
	andi	a0, s3, 32
	bnez	a0, .LBB0_205
.LBB0_119:                              # %else207
	andi	a0, s3, 64
	bnez	a0, .LBB0_206
.LBB0_120:                              # %else210
	andi	a0, s3, 128
	bnez	a0, .LBB0_207
.LBB0_121:                              # %else213
	andi	a0, s3, 256
	bnez	a0, .LBB0_208
.LBB0_122:                              # %else216
	andi	a0, s3, 512
	bnez	a0, .LBB0_209
.LBB0_123:                              # %else219
	andi	a0, s3, 1024
	bnez	a0, .LBB0_210
.LBB0_124:                              # %else222
	slli	a0, s3, 52
	bltz	a0, .LBB0_211
.LBB0_125:                              # %else225
	slli	a0, s3, 51
	bltz	a0, .LBB0_212
.LBB0_126:                              # %else228
	slli	a0, s3, 50
	bltz	a0, .LBB0_213
.LBB0_127:                              # %else231
	slli	a0, s3, 49
	bgez	a0, .LBB0_129
.LBB0_128:                              # %cond.store232
	.loc	1 0 44 is_stmt 0                # k135112023917904.py:0:44
	flw	fa0, 56(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_129:                              # %else234
	slli	a0, s3, 48
	lui	a1, 3
	addi	a1, a1, 384
	add	a1, sp, a1
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_131
# %bb.130:                              # %cond.store235
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 60(sp)                     # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 384
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 384
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 576(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_131:                              # %else237
	slli	a0, s3, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bltz	a0, .LBB0_214
# %bb.132:                              # %else240
	slli	a0, s3, 46
	bltz	a0, .LBB0_215
.LBB0_133:                              # %else243
	slli	a0, s3, 45
	bltz	a0, .LBB0_216
.LBB0_134:                              # %else246
	slli	a0, s3, 44
	bltz	a0, .LBB0_217
.LBB0_135:                              # %else249
	slli	a0, s3, 43
	bltz	a0, .LBB0_218
.LBB0_136:                              # %else252
	slli	a0, s3, 42
	bltz	a0, .LBB0_219
.LBB0_137:                              # %else255
	slli	a0, s3, 41
	bltz	a0, .LBB0_220
.LBB0_138:                              # %else258
	slli	a0, s3, 40
	bltz	a0, .LBB0_221
.LBB0_139:                              # %else261
	slli	a0, s3, 39
	addi	s4, sp, 2047
	addi	s4, s4, 225
	bltz	a0, .LBB0_222
.LBB0_140:                              # %else264
	slli	a0, s3, 38
	bltz	a0, .LBB0_223
.LBB0_141:                              # %else267
	slli	a0, s3, 37
	bltz	a0, .LBB0_224
.LBB0_142:                              # %else270
	slli	a0, s3, 36
	bltz	a0, .LBB0_225
.LBB0_143:                              # %else273
	slli	a0, s3, 35
	bltz	a0, .LBB0_226
.LBB0_144:                              # %else276
	slli	a0, s3, 34
	bltz	a0, .LBB0_227
.LBB0_145:                              # %else279
	slli	a0, s3, 33
	bgez	a0, .LBB0_147
.LBB0_146:                              # %cond.store280
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 120(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1409
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_147:                              # %else282
	slli	a0, s3, 32
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_149
# %bb.148:                              # %cond.store283
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 124(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 384
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1281
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 384
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_149:                              # %else285
	slli	a0, s3, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bltz	a0, .LBB0_228
# %bb.150:                              # %else288
	slli	a0, s3, 30
	bltz	a0, .LBB0_229
.LBB0_151:                              # %else291
	slli	a0, s3, 29
	bltz	a0, .LBB0_230
.LBB0_152:                              # %else294
	slli	a0, s3, 28
	bltz	a0, .LBB0_231
.LBB0_153:                              # %else297
	slli	a0, s3, 27
	bltz	a0, .LBB0_232
.LBB0_154:                              # %else300
	slli	a0, s3, 26
	bltz	a0, .LBB0_233
.LBB0_155:                              # %else303
	slli	a0, s3, 25
	bltz	a0, .LBB0_234
.LBB0_156:                              # %else306
	slli	a0, s3, 24
	bgez	a0, .LBB0_157
	j	.LBB0_235
.LBB0_157:                              # %else309
	slli	a0, s3, 23
	bgez	a0, .LBB0_158
	j	.LBB0_236
.LBB0_158:                              # %else312
	slli	a0, s3, 22
	bgez	a0, .LBB0_159
	j	.LBB0_237
.LBB0_159:                              # %else315
	slli	a0, s3, 21
	bgez	a0, .LBB0_160
	j	.LBB0_238
.LBB0_160:                              # %else318
	slli	a0, s3, 20
	bgez	a0, .LBB0_161
	j	.LBB0_239
.LBB0_161:                              # %else321
	slli	a0, s3, 19
	bgez	a0, .LBB0_162
	j	.LBB0_240
.LBB0_162:                              # %else324
	slli	a0, s3, 18
	bgez	a0, .LBB0_163
	j	.LBB0_241
.LBB0_163:                              # %else327
	slli	a0, s3, 17
	bgez	a0, .LBB0_165
.LBB0_164:                              # %cond.store328
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 184(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1920
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2032(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_165:                              # %else330
	slli	a0, s3, 16
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_167
# %bb.166:                              # %cond.store331
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 188(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 384
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1792
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 384
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1912(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_167:                              # %else333
	slli	a0, s3, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_168
	j	.LBB0_242
.LBB0_168:                              # %else336
	slli	a0, s3, 14
	bgez	a0, .LBB0_169
	j	.LBB0_243
.LBB0_169:                              # %else339
	slli	a0, s3, 13
	bgez	a0, .LBB0_170
	j	.LBB0_244
.LBB0_170:                              # %else342
	slli	a0, s3, 12
	bgez	a0, .LBB0_171
	j	.LBB0_245
.LBB0_171:                              # %else345
	slli	a0, s3, 11
	bgez	a0, .LBB0_172
	j	.LBB0_246
.LBB0_172:                              # %else348
	slli	a0, s3, 10
	bgez	a0, .LBB0_173
	j	.LBB0_247
.LBB0_173:                              # %else351
	slli	a0, s3, 9
	bgez	a0, .LBB0_174
	j	.LBB0_248
.LBB0_174:                              # %else354
	slli	a0, s3, 8
	bgez	a0, .LBB0_175
	j	.LBB0_249
.LBB0_175:                              # %else357
	slli	a0, s3, 7
	bgez	a0, .LBB0_176
	j	.LBB0_250
.LBB0_176:                              # %else360
	slli	a0, s3, 6
	bgez	a0, .LBB0_177
	j	.LBB0_251
.LBB0_177:                              # %else363
	slli	a0, s3, 5
	bgez	a0, .LBB0_178
	j	.LBB0_252
.LBB0_178:                              # %else366
	slli	a0, s3, 4
	bgez	a0, .LBB0_179
	j	.LBB0_253
.LBB0_179:                              # %else369
	slli	a0, s3, 3
	bgez	a0, .LBB0_180
	j	.LBB0_254
.LBB0_180:                              # %else372
	slli	a0, s3, 2
	bgez	a0, .LBB0_181
	j	.LBB0_255
.LBB0_181:                              # %else375
	slli	a0, s3, 1
	bgez	a0, .LBB0_182
	j	.LBB0_256
.LBB0_182:                              # %else378
	bgez	s3, .LBB0_184
.LBB0_183:                              # %cond.store379
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 252(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 256
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 376(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_184:                              # %else381
	.loc	1 10 4 epilogue_begin           # k135112023917904.py:10:4
	addi	sp, s0, -2032
	.cfi_def_cfa sp, 2032
	ld	ra, 2024(sp)                    # 8-byte Folded Reload
	ld	s0, 2016(sp)                    # 8-byte Folded Reload
	ld	s2, 2008(sp)                    # 8-byte Folded Reload
	ld	s3, 2000(sp)                    # 8-byte Folded Reload
	ld	s4, 1992(sp)                    # 8-byte Folded Reload
	fld	fs0, 1984(sp)                   # 8-byte Folded Reload
	fld	fs1, 1976(sp)                   # 8-byte Folded Reload
	fld	fs2, 1968(sp)                   # 8-byte Folded Reload
	fld	fs3, 1960(sp)                   # 8-byte Folded Reload
	fld	fs4, 1952(sp)                   # 8-byte Folded Reload
	fld	fs5, 1944(sp)                   # 8-byte Folded Reload
	fld	fs6, 1936(sp)                   # 8-byte Folded Reload
	fld	fs7, 1928(sp)                   # 8-byte Folded Reload
	fld	fs8, 1920(sp)                   # 8-byte Folded Reload
	fld	fs9, 1912(sp)                   # 8-byte Folded Reload
	fld	fs10, 1904(sp)                  # 8-byte Folded Reload
	fld	fs11, 1896(sp)                  # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
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
.LBB0_185:                              # %cond.load139
	.cfi_restore_state
	.loc	1 0 4                           # k135112023917904.py:0:4
	li	a3, 31
	slli	a3, a3, 8
	add	a3, sp, a3
	.loc	1 9 43 is_stmt 1                # k135112023917904.py:9:43
	vse64.v	v24, (a3)
	ld	a3, 1536(a2)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 188(sp)                    # 4-byte Folded Spill
	slli	a3, a1, 15
	vadd.vx	v8, v8, a0
	bltz	a3, .LBB0_96
	j	.LBB0_97
.LBB0_186:                              # %cond.load145
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 196(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 13
	bgez	a0, .LBB0_99
.LBB0_187:                              # %cond.load148
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 2
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 200(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 12
	bgez	a0, .LBB0_100
.LBB0_188:                              # %cond.load151
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 3
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 204(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 11
	bgez	a0, .LBB0_101
.LBB0_189:                              # %cond.load154
	.loc	1 0 43 is_stmt 0                # k135112023917904.py:0:43
	lui	a0, 2
	addi	a0, a0, -384
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1320(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 208(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 10
	bgez	a0, .LBB0_102
.LBB0_190:                              # %cond.load157
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a0, 15
	slli	a0, a0, 9
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1200(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 212(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 9
	bgez	a0, .LBB0_103
.LBB0_191:                              # %cond.load160
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a0, 2
	addi	a0, a0, -640
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 216(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 8
	bgez	a0, .LBB0_104
.LBB0_192:                              # %cond.load163
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a0, 29
	slli	a0, a0, 8
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 220(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 7
	bgez	a0, .LBB0_105
.LBB0_193:                              # %cond.load166
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a0, 2
	addi	a0, a0, -896
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 224(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 6
	bgez	a0, .LBB0_106
.LBB0_194:                              # %cond.load169
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a0, 7
	slli	a0, a0, 10
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 228(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 5
	bgez	a0, .LBB0_107
.LBB0_195:                              # %cond.load172
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a0, 2
	addi	a0, a0, -1152
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 232(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 4
	bgez	a0, .LBB0_108
.LBB0_196:                              # %cond.load175
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a0, 27
	slli	a0, a0, 8
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 236(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 3
	bgez	a0, .LBB0_109
.LBB0_197:                              # %cond.load178
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a0, 2
	addi	a0, a0, -1408
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 240(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 2
	bgez	a0, .LBB0_110
.LBB0_198:                              # %cond.load181
	.loc	1 0 43                          # k135112023917904.py:0:43
	li	a0, 13
	slli	a0, a0, 9
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 244(sp)                    # 4-byte Folded Spill
	slli	a0, a1, 1
	bgez	a0, .LBB0_111
.LBB0_199:                              # %cond.load184
	.loc	1 0 43                          # k135112023917904.py:0:43
	lui	a0, 2
	addi	a0, a0, -1664
	add	a0, sp, a0
	.loc	1 9 43                          # k135112023917904.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(a2)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 248(sp)                    # 4-byte Folded Spill
	bltz	a1, .LBB0_112
	j	.LBB0_113
.LBB0_200:                              # %cond.store
	.loc	1 10 44 is_stmt 1               # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 2
	beqz	a0, .LBB0_115
.LBB0_201:                              # %cond.store193
	fmv.s	fa0, fs1
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 4
	beqz	a0, .LBB0_116
.LBB0_202:                              # %cond.store196
	fmv.s	fa0, fs2
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 8
	beqz	a0, .LBB0_117
.LBB0_203:                              # %cond.store199
	fmv.s	fa0, fs0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 16
	lui	a1, 1
	addi	a1, a1, 312
	add	s4, sp, a1
	beqz	a0, .LBB0_118
.LBB0_204:                              # %cond.store202
	fmv.s	fa0, fs11
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 32
	beqz	a0, .LBB0_119
.LBB0_205:                              # %cond.store205
	fmv.s	fa0, fs10
	call	__truncsfbf2
	li	a0, 3
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 64
	beqz	a0, .LBB0_120
.LBB0_206:                              # %cond.store208
	fmv.s	fa0, fs9
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 128
	beqz	a0, .LBB0_121
.LBB0_207:                              # %cond.store211
	fmv.s	fa0, fs8
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 256
	beqz	a0, .LBB0_122
.LBB0_208:                              # %cond.store214
	fmv.s	fa0, fs7
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 512
	beqz	a0, .LBB0_123
.LBB0_209:                              # %cond.store217
	fmv.s	fa0, fs6
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 1024
	beqz	a0, .LBB0_124
.LBB0_210:                              # %cond.store220
	fmv.s	fa0, fs5
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 52
	bgez	a0, .LBB0_125
.LBB0_211:                              # %cond.store223
	fmv.s	fa0, fs4
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 51
	bgez	a0, .LBB0_126
.LBB0_212:                              # %cond.store226
	fmv.s	fa0, fs3
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 50
	bgez	a0, .LBB0_127
.LBB0_213:                              # %cond.store229
	.loc	1 0 44 is_stmt 0                # k135112023917904.py:0:44
	flw	fa0, 52(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 49
	bltz	a0, .LBB0_128
	j	.LBB0_129
.LBB0_214:                              # %cond.store238
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 64(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 46
	bgez	a0, .LBB0_133
.LBB0_215:                              # %cond.store241
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 68(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 45
	bgez	a0, .LBB0_134
.LBB0_216:                              # %cond.store244
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 72(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 44
	bgez	a0, .LBB0_135
.LBB0_217:                              # %cond.store247
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 76(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 43
	bgez	a0, .LBB0_136
.LBB0_218:                              # %cond.store250
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 80(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 42
	bgez	a0, .LBB0_137
.LBB0_219:                              # %cond.store253
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 84(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 41
	bgez	a0, .LBB0_138
.LBB0_220:                              # %cond.store256
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 88(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 40
	bgez	a0, .LBB0_139
.LBB0_221:                              # %cond.store259
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 92(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 39
	addi	s4, sp, 2047
	addi	s4, s4, 225
	bgez	a0, .LBB0_140
.LBB0_222:                              # %cond.store262
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 96(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 38
	bgez	a0, .LBB0_141
.LBB0_223:                              # %cond.store265
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 100(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 37
	bgez	a0, .LBB0_142
.LBB0_224:                              # %cond.store268
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 104(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1921
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 36
	bgez	a0, .LBB0_143
.LBB0_225:                              # %cond.store271
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 108(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1793
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 35
	bgez	a0, .LBB0_144
.LBB0_226:                              # %cond.store274
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 112(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1665
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 34
	bgez	a0, .LBB0_145
.LBB0_227:                              # %cond.store277
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 116(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1537
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 33
	bltz	a0, .LBB0_146
	j	.LBB0_147
.LBB0_228:                              # %cond.store286
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 128(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 30
	bgez	a0, .LBB0_151
.LBB0_229:                              # %cond.store289
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 132(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 29
	bgez	a0, .LBB0_152
.LBB0_230:                              # %cond.store292
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 136(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 28
	bgez	a0, .LBB0_153
.LBB0_231:                              # %cond.store295
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 140(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 27
	bgez	a0, .LBB0_154
.LBB0_232:                              # %cond.store298
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 144(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1153
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 26
	bgez	a0, .LBB0_155
.LBB0_233:                              # %cond.store301
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 148(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1025
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 25
	bgez	a0, .LBB0_156
.LBB0_234:                              # %cond.store304
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 152(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 897
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 24
	bltz	a0, .LBB0_235
	j	.LBB0_157
.LBB0_235:                              # %cond.store307
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 156(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 769
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 23
	bltz	a0, .LBB0_236
	j	.LBB0_158
.LBB0_236:                              # %cond.store310
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 160(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 641
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 22
	bltz	a0, .LBB0_237
	j	.LBB0_159
.LBB0_237:                              # %cond.store313
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 164(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 513
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 21
	bltz	a0, .LBB0_238
	j	.LBB0_160
.LBB0_238:                              # %cond.store316
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 168(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 385
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 20
	bltz	a0, .LBB0_239
	j	.LBB0_161
.LBB0_239:                              # %cond.store319
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 172(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 257
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 19
	bltz	a0, .LBB0_240
	j	.LBB0_162
.LBB0_240:                              # %cond.store322
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 176(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 129
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 18
	bltz	a0, .LBB0_241
	j	.LBB0_163
.LBB0_241:                              # %cond.store325
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 180(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1944(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 17
	bgez	a0, .LBB0_257
	j	.LBB0_164
.LBB0_257:                              # %cond.store325
	j	.LBB0_165
.LBB0_242:                              # %cond.store334
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 192(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 14
	bltz	a0, .LBB0_243
	j	.LBB0_169
.LBB0_243:                              # %cond.store337
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 196(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 13
	bltz	a0, .LBB0_244
	j	.LBB0_170
.LBB0_244:                              # %cond.store340
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 200(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 12
	bltz	a0, .LBB0_245
	j	.LBB0_171
.LBB0_245:                              # %cond.store343
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 204(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 11
	bltz	a0, .LBB0_246
	j	.LBB0_172
.LBB0_246:                              # %cond.store346
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 208(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1664
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1696(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 10
	bltz	a0, .LBB0_247
	j	.LBB0_173
.LBB0_247:                              # %cond.store349
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 212(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1536
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1576(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 9
	bltz	a0, .LBB0_248
	j	.LBB0_174
.LBB0_248:                              # %cond.store352
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 216(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1408
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1456(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 8
	bltz	a0, .LBB0_249
	j	.LBB0_175
.LBB0_249:                              # %cond.store355
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 220(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1280
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1336(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 7
	bltz	a0, .LBB0_250
	j	.LBB0_176
.LBB0_250:                              # %cond.store358
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 224(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1152
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1216(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 6
	bltz	a0, .LBB0_251
	j	.LBB0_177
.LBB0_251:                              # %cond.store361
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 228(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1024
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1096(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 5
	bltz	a0, .LBB0_252
	j	.LBB0_178
.LBB0_252:                              # %cond.store364
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 232(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 896
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 976(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 4
	bltz	a0, .LBB0_253
	j	.LBB0_179
.LBB0_253:                              # %cond.store367
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 236(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 768
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 856(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 3
	bltz	a0, .LBB0_254
	j	.LBB0_180
.LBB0_254:                              # %cond.store370
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 240(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 640
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 736(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 2
	bltz	a0, .LBB0_255
	j	.LBB0_181
.LBB0_255:                              # %cond.store373
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 244(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 512
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 616(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 1
	bltz	a0, .LBB0_256
	j	.LBB0_182
.LBB0_256:                              # %cond.store376
	.loc	1 0 44                          # k135112023917904.py:0:44
	flw	fa0, 248(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135112023917904.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 384
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 384
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 496(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	bgez	s3, .LBB0_258
	j	.LBB0_183
.LBB0_258:                              # %cond.store376
	j	.LBB0_184
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused_cat_neg_slice_transpose_view_4, .Lfunc_end0-triton_poi_fused_cat_neg_slice_transpose_view_4
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
	.asciz	"k135112023917904.py"           # string offset=7 ; k135112023917904.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
