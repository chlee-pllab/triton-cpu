	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused_neg_slice_transpose_view_3 # -- Begin function triton_poi_fused_neg_slice_transpose_view_3
	.p2align	2
	.type	triton_poi_fused_neg_slice_transpose_view_3,@function
triton_poi_fused_neg_slice_transpose_view_3: # @triton_poi_fused_neg_slice_transpose_view_3
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114449653664.py"
	.loc	1 2 0                           # k135114449653664.py:2:0
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
	.cfi_remember_state
	lui	a2, 3
	addi	a2, a2, -880
	sub	sp, sp, a2
	csrr	a2, vlenb
	li	a4, 40
	mul	a2, a2, a4
	sub	sp, sp, a2
	andi	sp, sp, -128
	sd	a1, 632(sp)                     # 8-byte Folded Spill
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114449653664.py:4:33
	slli	a1, a3, 6
	li	a2, 32
	li	a4, -32
	li	a3, -64
	.loc	1 5 23                          # k135114449653664.py:5:23
	vsetvli	zero, a2, e32, m8, ta, ma
	vmv.v.x	v16, a1
	vid.v	v8
	vor.vx	v24, v8, a1
	.loc	1 8 19                          # k135114449653664.py:8:19
	vsra.vi	v16, v16, 31
	vsrl.vi	v16, v16, 27
	csrr	a5, vlenb
	slli	a5, a5, 4
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 960
	add	a5, a5, a6
	vs8r.v	v16, (a5)                       # vscale x 64-byte Folded Spill
	vadd.vv	v16, v24, v16
	.loc	1 7 19                          # k135114449653664.py:7:19
	vand.vx	v0, v16, a4
	.loc	1 9 43                          # k135114449653664.py:9:43
	vadd.vv	v16, v16, v16
	.loc	1 7 19                          # k135114449653664.py:7:19
	vsub.vv	v0, v24, v0
	.loc	1 9 43                          # k135114449653664.py:9:43
	vand.vx	v16, v16, a3
	.loc	1 9 35 is_stmt 0                # k135114449653664.py:9:35
	vadd.vv	v16, v16, v0
	li	a5, 64
	.loc	1 5 23 is_stmt 1                # k135114449653664.py:5:23
	vadd.vx	v8, v8, a2
	vor.vx	v0, v8, a1
	csrr	a1, vlenb
	li	a6, 24
	mul	a1, a1, a6
	add	a1, sp, a1
	lui	a6, 3
	addi	a6, a6, 960
	add	a1, a1, a6
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114449653664.py:6:21
	vmslt.vx	v8, v24, a5
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a6, 3
	addi	a6, a6, 960
	add	a1, a1, a6
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	vmslt.vx	v9, v0, a5
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v8, v9, 4
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetvli	zero, zero, e64, m4, ta, ma
	vmv.x.s	a1, v8
	andi	a5, a1, 1
	lui	a6, 3
	addi	a6, a6, 960
	add	a6, sp, a6
	vs8r.v	v16, (a6)                       # vscale x 64-byte Folded Spill
	.loc	1 9 40 is_stmt 0                # k135114449653664.py:9:40
	vsetvli	zero, a2, e32, m8, ta, ma
	vadd.vx	v8, v16, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v16, a0
	beqz	a5, .LBB0_2
# %bb.1:                                # %cond.load
	vmv.x.s	a5, v0
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa0, zero
	fmv.w.x	fs0, a5
	fmv.s	fa1, fa0
	fsw	fa0, 472(sp)                    # 4-byte Folded Spill
	fmv.s	ft0, fa0
	fsw	fa0, 480(sp)                    # 4-byte Folded Spill
	fmv.s	ft1, fa0
	fsw	fa0, 488(sp)                    # 4-byte Folded Spill
	fmv.s	ft2, fa0
	fsw	fa0, 496(sp)                    # 4-byte Folded Spill
	fmv.s	ft3, fa0
	fsw	fa0, 504(sp)                    # 4-byte Folded Spill
	fmv.s	ft4, fa0
	fsw	fa0, 512(sp)                    # 4-byte Folded Spill
	fmv.s	ft5, fa0
	fsw	fa0, 520(sp)                    # 4-byte Folded Spill
	fmv.s	ft6, fa0
	fsw	fa0, 528(sp)                    # 4-byte Folded Spill
	fmv.s	ft7, fa0
	fsw	fa0, 536(sp)                    # 4-byte Folded Spill
	fmv.s	fa6, fa0
	fsw	fa0, 544(sp)                    # 4-byte Folded Spill
	fmv.s	fa7, fa0
	fsw	fa0, 552(sp)                    # 4-byte Folded Spill
	fmv.s	ft8, fa0
	fsw	fa0, 560(sp)                    # 4-byte Folded Spill
	fmv.s	ft9, fa0
	fsw	fa0, 568(sp)                    # 4-byte Folded Spill
	fmv.s	ft10, fa0
	fsw	fa0, 576(sp)                    # 4-byte Folded Spill
	fmv.s	ft11, fa0
	fsw	fa0, 584(sp)                    # 4-byte Folded Spill
	fmv.s	fa3, fa0
	fsw	fa0, 592(sp)                    # 4-byte Folded Spill
	fsw	fa0, 376(sp)                    # 4-byte Folded Spill
	fsw	fa0, 448(sp)                    # 4-byte Folded Spill
	fsw	fa0, 440(sp)                    # 4-byte Folded Spill
	fsw	fa0, 432(sp)                    # 4-byte Folded Spill
	fsw	fa0, 424(sp)                    # 4-byte Folded Spill
	fsw	fa0, 416(sp)                    # 4-byte Folded Spill
	fsw	fa0, 408(sp)                    # 4-byte Folded Spill
	fsw	fa0, 456(sp)                    # 4-byte Folded Spill
	fsw	fa0, 400(sp)                    # 4-byte Folded Spill
	fsw	fa0, 392(sp)                    # 4-byte Folded Spill
	fsw	fa0, 384(sp)                    # 4-byte Folded Spill
	fsw	fa0, 600(sp)                    # 4-byte Folded Spill
	fsw	fa0, 608(sp)                    # 4-byte Folded Spill
	fsw	fa0, 616(sp)                    # 4-byte Folded Spill
	fsw	fa0, 624(sp)                    # 4-byte Folded Spill
	fmv.s	fs8, fa0
	fmv.s	fs1, fa0
	fmv.s	fs7, fa0
	fmv.s	fs6, fa0
	fmv.s	fs5, fa0
	fmv.s	fs4, fa0
	fmv.s	fs3, fa0
	fmv.s	fs2, fa0
	fmv.s	fs11, fa0
	fmv.s	fs10, fa0
	fmv.s	fs9, fa0
	fmv.s	fa5, fa0
	fmv.s	fa4, fa0
	fmv.s	fa2, fa0
	fsw	fa0, 464(sp)                    # 4-byte Folded Spill
	andi	a5, a1, 2
	bnez	a5, .LBB0_3
	j	.LBB0_4
.LBB0_2:
	.loc	1 0 48                          # k135114449653664.py:0:48
	fmv.w.x	fs0, zero
	fmv.s	fa0, fs0
	fmv.s	fa1, fs0
	fsw	fs0, 472(sp)                    # 4-byte Folded Spill
	fmv.s	ft0, fs0
	fsw	fs0, 480(sp)                    # 4-byte Folded Spill
	fmv.s	ft1, fs0
	fsw	fs0, 488(sp)                    # 4-byte Folded Spill
	fmv.s	ft2, fs0
	fsw	fs0, 496(sp)                    # 4-byte Folded Spill
	fmv.s	ft3, fs0
	fsw	fs0, 504(sp)                    # 4-byte Folded Spill
	fmv.s	ft4, fs0
	fsw	fs0, 512(sp)                    # 4-byte Folded Spill
	fmv.s	ft5, fs0
	fsw	fs0, 520(sp)                    # 4-byte Folded Spill
	fmv.s	ft6, fs0
	fsw	fs0, 528(sp)                    # 4-byte Folded Spill
	fmv.s	ft7, fs0
	fsw	fs0, 536(sp)                    # 4-byte Folded Spill
	fmv.s	fa6, fs0
	fsw	fs0, 544(sp)                    # 4-byte Folded Spill
	fmv.s	fa7, fs0
	fsw	fs0, 552(sp)                    # 4-byte Folded Spill
	fmv.s	ft8, fs0
	fsw	fs0, 560(sp)                    # 4-byte Folded Spill
	fmv.s	ft9, fs0
	fsw	fs0, 568(sp)                    # 4-byte Folded Spill
	fmv.s	ft10, fs0
	fsw	fs0, 576(sp)                    # 4-byte Folded Spill
	fmv.s	ft11, fs0
	fsw	fs0, 584(sp)                    # 4-byte Folded Spill
	fmv.s	fa3, fs0
	fsw	fs0, 592(sp)                    # 4-byte Folded Spill
	fsw	fs0, 376(sp)                    # 4-byte Folded Spill
	fsw	fs0, 448(sp)                    # 4-byte Folded Spill
	fsw	fs0, 440(sp)                    # 4-byte Folded Spill
	fsw	fs0, 432(sp)                    # 4-byte Folded Spill
	fsw	fs0, 424(sp)                    # 4-byte Folded Spill
	fsw	fs0, 416(sp)                    # 4-byte Folded Spill
	fsw	fs0, 408(sp)                    # 4-byte Folded Spill
	fsw	fs0, 456(sp)                    # 4-byte Folded Spill
	fsw	fs0, 400(sp)                    # 4-byte Folded Spill
	fsw	fs0, 392(sp)                    # 4-byte Folded Spill
	fsw	fs0, 384(sp)                    # 4-byte Folded Spill
	fsw	fs0, 600(sp)                    # 4-byte Folded Spill
	fsw	fs0, 608(sp)                    # 4-byte Folded Spill
	fsw	fs0, 616(sp)                    # 4-byte Folded Spill
	fsw	fs0, 624(sp)                    # 4-byte Folded Spill
	fmv.s	fs8, fs0
	fmv.s	fs1, fs0
	fmv.s	fs7, fs0
	fmv.s	fs6, fs0
	fmv.s	fs5, fs0
	fmv.s	fs4, fs0
	fmv.s	fs3, fs0
	fmv.s	fs2, fs0
	fmv.s	fs11, fs0
	fmv.s	fs10, fs0
	fmv.s	fs9, fs0
	fmv.s	fa5, fs0
	fmv.s	fa4, fs0
	fmv.s	fa2, fs0
	fsw	fs0, 464(sp)                    # 4-byte Folded Spill
	.loc	1 9 48                          # k135114449653664.py:9:48
	andi	a5, a1, 2
	beqz	a5, .LBB0_4
.LBB0_3:                                # %cond.load1
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v0, 1
	vmv.x.s	a5, v16
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa0, a5
.LBB0_4:                                # %else2
	andi	a5, a1, 4
	bnez	a5, .LBB0_43
# %bb.5:                                # %else5
	andi	a5, a1, 8
	fsw	fa1, 248(sp)                    # 4-byte Folded Spill
	bnez	a5, .LBB0_44
.LBB0_6:                                # %else8
	andi	a6, a1, 16
	lui	a5, 3
	addi	a5, a5, -1216
	add	a5, sp, a5
	fmv.s	fa1, fs8
	bnez	a6, .LBB0_45
.LBB0_7:                                # %else11
	andi	a6, a1, 32
	fsw	ft0, 252(sp)                    # 4-byte Folded Spill
	bnez	a6, .LBB0_46
.LBB0_8:                                # %else14
	andi	a6, a1, 64
	fmv.s	ft0, fs1
	bnez	a6, .LBB0_47
.LBB0_9:                                # %else17
	andi	a6, a1, 128
	fsw	ft1, 256(sp)                    # 4-byte Folded Spill
	bnez	a6, .LBB0_48
.LBB0_10:                               # %else20
	andi	a6, a1, 256
	fmv.s	ft1, fs7
	bnez	a6, .LBB0_49
.LBB0_11:                               # %else23
	andi	a6, a1, 512
	fsw	ft2, 260(sp)                    # 4-byte Folded Spill
	bnez	a6, .LBB0_50
.LBB0_12:                               # %else26
	andi	a6, a1, 1024
	fmv.s	ft2, fs6
	bnez	a6, .LBB0_51
.LBB0_13:                               # %else29
	slli	a6, a1, 52
	fsw	ft3, 264(sp)                    # 4-byte Folded Spill
	bltz	a6, .LBB0_52
.LBB0_14:                               # %else32
	slli	a6, a1, 51
	fmv.s	ft3, fs5
	bltz	a6, .LBB0_53
.LBB0_15:                               # %else35
	slli	a6, a1, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	fsw	ft4, 268(sp)                    # 4-byte Folded Spill
	bltz	a6, .LBB0_54
.LBB0_16:                               # %else38
	slli	a6, a1, 49
	fmv.s	ft4, fs4
	bltz	a6, .LBB0_55
.LBB0_17:                               # %else41
	slli	a6, a1, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	fsw	ft5, 272(sp)                    # 4-byte Folded Spill
	bltz	a6, .LBB0_56
.LBB0_18:                               # %else44
	slli	a6, a1, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v16, a0
	fmv.s	ft5, fs3
	bltz	a6, .LBB0_57
.LBB0_19:                               # %else47
	slli	a6, a1, 46
	fsw	ft6, 280(sp)                    # 4-byte Folded Spill
	bltz	a6, .LBB0_58
.LBB0_20:                               # %else50
	slli	a6, a1, 45
	fmv.s	ft6, fs2
	bltz	a6, .LBB0_59
.LBB0_21:                               # %else53
	slli	a6, a1, 44
	fsw	ft7, 288(sp)                    # 4-byte Folded Spill
	bltz	a6, .LBB0_60
.LBB0_22:                               # %else56
	slli	a6, a1, 43
	fmv.s	ft7, fs11
	bltz	a6, .LBB0_61
.LBB0_23:                               # %else59
	slli	a6, a1, 42
	fsw	fa6, 296(sp)                    # 4-byte Folded Spill
	bltz	a6, .LBB0_62
.LBB0_24:                               # %else62
	slli	a6, a1, 41
	fmv.s	fa6, fs10
	bltz	a6, .LBB0_63
.LBB0_25:                               # %else65
	slli	a6, a1, 40
	fsw	fa7, 304(sp)                    # 4-byte Folded Spill
	bltz	a6, .LBB0_64
.LBB0_26:                               # %else68
	slli	a6, a1, 39
	fmv.s	fa7, fs9
	bgez	a6, .LBB0_28
.LBB0_27:                               # %cond.load70
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, -1280
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a5, 0(a5)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	ft8, a5
.LBB0_28:                               # %else71
	.loc	1 0 48                          # k135114449653664.py:0:48
	csrr	a5, vlenb
	slli	a5, a5, 5
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 960
	add	a5, a5, a6
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	csrr	a5, vlenb
	slli	a5, a5, 4
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 960
	add	a5, a5, a6
	vl8r.v	v16, (a5)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 9 48                          # k135114449653664.py:9:48
	slli	a6, a1, 38
	lui	a5, 2
	addi	a5, a5, 744
	add	a5, sp, a5
	fsw	ft8, 312(sp)                    # 4-byte Folded Spill
	bgez	a6, .LBB0_30
# %bb.29:                               # %cond.load73
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, -1408
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 2016(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft8, a6
	fsw	ft8, 560(sp)                    # 4-byte Folded Spill
.LBB0_30:                               # %else74
	.loc	1 0 48                          # k135114449653664.py:0:48
	vsetvli	zero, a2, e32, m8, ta, ma
	vand.vx	v24, v8, a4
	.loc	1 9 48                          # k135114449653664.py:9:48
	slli	a4, a1, 37
	vadd.vv	v8, v8, v8
	fmv.s	ft8, fa5
	bgez	a4, .LBB0_32
# %bb.31:                               # %cond.load76
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a4, 21
	slli	a4, a4, 9
	add	a4, sp, a4
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a4)
	ld	a4, 1896(a5)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	ft9, a4
.LBB0_32:                               # %else77
	.loc	1 0 48                          # k135114449653664.py:0:48
	vsetvli	zero, a2, e32, m8, ta, ma
	vand.vx	v8, v8, a3
	.loc	1 9 48                          # k135114449653664.py:9:48
	slli	a3, a1, 36
	csrr	a4, vlenb
	slli	a4, a4, 5
	add	a4, sp, a4
	lui	a6, 3
	addi	a6, a6, 960
	add	a4, a4, a6
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	vsub.vv	v16, v16, v24
	fsw	ft9, 320(sp)                    # 4-byte Folded Spill
	bgez	a3, .LBB0_34
# %bb.33:                               # %cond.load79
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a3, 3
	addi	a3, a3, -1664
	add	a3, sp, a3
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a3)
	ld	a3, 1776(a5)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	ft9, a3
	fsw	ft9, 568(sp)                    # 4-byte Folded Spill
.LBB0_34:                               # %else80
	slli	a3, a1, 35
	vsetvli	zero, a2, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	csrr	a4, vlenb
	slli	a4, a4, 3
	add	a4, sp, a4
	lui	a6, 3
	addi	a6, a6, 960
	add	a4, a4, a6
	vs8r.v	v8, (a4)                        # vscale x 64-byte Folded Spill
	fmv.s	ft9, fa4
	bgez	a3, .LBB0_36
# %bb.35:                               # %cond.load82
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a3, 3
	addi	a3, a3, -1792
	add	a3, sp, a3
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a3)
	ld	a3, 1656(a5)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	ft10, a3
.LBB0_36:                               # %else83
	slli	a3, a1, 34
	csrr	a4, vlenb
	slli	a4, a4, 3
	add	a4, sp, a4
	lui	a6, 3
	addi	a6, a6, 960
	add	a4, a4, a6
	vl8r.v	v8, (a4)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e32, m8, ta, ma
	vadd.vx	v8, v8, a2
	fsw	ft10, 328(sp)                   # 4-byte Folded Spill
	bltz	a3, .LBB0_65
# %bb.37:                               # %else86
	slli	a2, a1, 33
	fmv.s	ft10, fa2
	bltz	a2, .LBB0_66
.LBB0_38:                               # %else89
	slli	a2, a1, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	fsw	ft11, 336(sp)                   # 4-byte Folded Spill
	bltz	a2, .LBB0_67
.LBB0_39:                               # %else92
	slli	a2, a1, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v24, a0
	bltz	a2, .LBB0_68
.LBB0_40:                               # %else95
	slli	a2, a1, 30
	fsw	fa3, 344(sp)                    # 4-byte Folded Spill
	bltz	a2, .LBB0_69
.LBB0_41:                               # %else98
	slli	a2, a1, 29
	flw	fa2, 624(sp)                    # 4-byte Folded Reload
	flw	fa3, 616(sp)                    # 4-byte Folded Reload
	flw	fa4, 608(sp)                    # 4-byte Folded Reload
	flw	fa5, 600(sp)                    # 4-byte Folded Reload
	bgez	a2, .LBB0_70
.LBB0_42:                               # %cond.load100
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v24, 2
	vmv.x.s	a2, v16
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs8, a2
	j	.LBB0_71
.LBB0_43:                               # %cond.load4
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v0, 2
	vmv.x.s	a5, v16
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa1, a5
	andi	a5, a1, 8
	fsw	fa1, 248(sp)                    # 4-byte Folded Spill
	beqz	a5, .LBB0_6
.LBB0_44:                               # %cond.load7
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v0, 3
	vmv.x.s	a5, v16
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa1, a5
	fsw	fa1, 472(sp)                    # 4-byte Folded Spill
	andi	a6, a1, 16
	lui	a5, 3
	addi	a5, a5, -1216
	add	a5, sp, a5
	fmv.s	fa1, fs8
	beqz	a6, .LBB0_7
.LBB0_45:                               # %cond.load10
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, 768
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 2016(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	andi	a6, a1, 32
	fsw	ft0, 252(sp)                    # 4-byte Folded Spill
	beqz	a6, .LBB0_8
.LBB0_46:                               # %cond.load13
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, 640
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1896(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 480(sp)                    # 4-byte Folded Spill
	andi	a6, a1, 64
	fmv.s	ft0, fs1
	beqz	a6, .LBB0_9
.LBB0_47:                               # %cond.load16
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a6, 25
	slli	a6, a6, 9
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1776(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft1, a6
	andi	a6, a1, 128
	fsw	ft1, 256(sp)                    # 4-byte Folded Spill
	beqz	a6, .LBB0_10
.LBB0_48:                               # %cond.load19
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, 384
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1656(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft1, a6
	fsw	ft1, 488(sp)                    # 4-byte Folded Spill
	andi	a6, a1, 256
	fmv.s	ft1, fs7
	beqz	a6, .LBB0_11
.LBB0_49:                               # %cond.load22
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, 256
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1536(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft2, a6
	andi	a6, a1, 512
	fsw	ft2, 260(sp)                    # 4-byte Folded Spill
	beqz	a6, .LBB0_12
.LBB0_50:                               # %cond.load25
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, 128
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1416(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft2, a6
	fsw	ft2, 496(sp)                    # 4-byte Folded Spill
	andi	a6, a1, 1024
	fmv.s	ft2, fs6
	beqz	a6, .LBB0_13
.LBB0_51:                               # %cond.load28
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1296(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft3, a6
	slli	a6, a1, 52
	fsw	ft3, 264(sp)                    # 4-byte Folded Spill
	bgez	a6, .LBB0_14
.LBB0_52:                               # %cond.load31
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, -128
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1176(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft3, a6
	fsw	ft3, 504(sp)                    # 4-byte Folded Spill
	slli	a6, a1, 51
	fmv.s	ft3, fs5
	bgez	a6, .LBB0_15
.LBB0_53:                               # %cond.load34
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, -256
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1056(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft4, a6
	slli	a6, a1, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	fsw	ft4, 268(sp)                    # 4-byte Folded Spill
	bgez	a6, .LBB0_16
.LBB0_54:                               # %cond.load37
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, -384
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 936(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft4, a6
	fsw	ft4, 512(sp)                    # 4-byte Folded Spill
	slli	a6, a1, 49
	fmv.s	ft4, fs4
	bgez	a6, .LBB0_17
.LBB0_55:                               # %cond.load40
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a6, 23
	slli	a6, a6, 9
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 816(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft5, a6
	slli	a6, a1, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	fsw	ft5, 272(sp)                    # 4-byte Folded Spill
	bgez	a6, .LBB0_18
.LBB0_56:                               # %cond.load43
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, -640
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vse64.v	v0, (a6)
	ld	a6, 696(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft5, a6
	fsw	ft5, 520(sp)                    # 4-byte Folded Spill
	slli	a6, a1, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v16, a0
	fmv.s	ft5, fs3
	bgez	a6, .LBB0_19
.LBB0_57:                               # %cond.load46
	vmv.x.s	a6, v0
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft6, a6
	slli	a6, a1, 46
	fsw	ft6, 280(sp)                    # 4-byte Folded Spill
	bgez	a6, .LBB0_20
.LBB0_58:                               # %cond.load49
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft6, a6
	fsw	ft6, 528(sp)                    # 4-byte Folded Spill
	slli	a6, a1, 45
	fmv.s	ft6, fs2
	bgez	a6, .LBB0_21
.LBB0_59:                               # %cond.load52
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft7, a6
	slli	a6, a1, 44
	fsw	ft7, 288(sp)                    # 4-byte Folded Spill
	bgez	a6, .LBB0_22
.LBB0_60:                               # %cond.load55
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft7, a6
	fsw	ft7, 536(sp)                    # 4-byte Folded Spill
	slli	a6, a1, 43
	fmv.s	ft7, fs11
	bgez	a6, .LBB0_23
.LBB0_61:                               # %cond.load58
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, -768
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 480(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa6, a6
	slli	a6, a1, 42
	fsw	fa6, 296(sp)                    # 4-byte Folded Spill
	bgez	a6, .LBB0_24
.LBB0_62:                               # %cond.load61
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, -896
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 360(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa6, a6
	fsw	fa6, 544(sp)                    # 4-byte Folded Spill
	slli	a6, a1, 41
	fmv.s	fa6, fs10
	bgez	a6, .LBB0_25
.LBB0_63:                               # %cond.load64
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a6, 11
	slli	a6, a6, 10
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 240(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa7, a6
	slli	a6, a1, 40
	fsw	fa7, 304(sp)                    # 4-byte Folded Spill
	bgez	a6, .LBB0_26
.LBB0_64:                               # %cond.load67
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a6, 3
	addi	a6, a6, -1152
	add	a6, sp, a6
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 120(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa7, a6
	fsw	fa7, 552(sp)                    # 4-byte Folded Spill
	slli	a6, a1, 39
	fmv.s	fa7, fs9
	bltz	a6, .LBB0_27
	j	.LBB0_28
.LBB0_65:                               # %cond.load85
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a2, 3
	addi	a2, a2, -1920
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 1536(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	ft10, a2
	fsw	ft10, 576(sp)                   # 4-byte Folded Spill
	slli	a2, a1, 33
	fmv.s	ft10, fa2
	bgez	a2, .LBB0_38
.LBB0_66:                               # %cond.load88
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a2, 5
	slli	a2, a2, 11
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a2)
	ld	a2, 1416(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	ft11, a2
	slli	a2, a1, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	fsw	ft11, 336(sp)                   # 4-byte Folded Spill
	bgez	a2, .LBB0_39
.LBB0_67:                               # %cond.load91
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a2, 2
	addi	a2, a2, 1920
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vse64.v	v0, (a2)
	ld	a2, 1296(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	ft11, a2
	fsw	ft11, 584(sp)                   # 4-byte Folded Spill
	slli	a2, a1, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v24, a0
	bgez	a2, .LBB0_40
.LBB0_68:                               # %cond.load94
	vmv.x.s	a2, v24
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa3, a2
	slli	a2, a1, 30
	fsw	fa3, 344(sp)                    # 4-byte Folded Spill
	bgez	a2, .LBB0_41
.LBB0_69:                               # %cond.load97
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v24, 1
	vmv.x.s	a2, v16
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 592(sp)                    # 4-byte Folded Spill
	slli	a2, a1, 29
	flw	fa2, 624(sp)                    # 4-byte Folded Reload
	flw	fa3, 616(sp)                    # 4-byte Folded Reload
	flw	fa4, 608(sp)                    # 4-byte Folded Reload
	flw	fa5, 600(sp)                    # 4-byte Folded Reload
	bltz	a2, .LBB0_42
.LBB0_70:
	.loc	1 0 48                          # k135114449653664.py:0:48
	flw	fs8, 376(sp)                    # 4-byte Folded Reload
.LBB0_71:                               # %else101
	flw	fs7, 448(sp)                    # 4-byte Folded Reload
	flw	fs6, 440(sp)                    # 4-byte Folded Reload
	flw	fs5, 432(sp)                    # 4-byte Folded Reload
	flw	fs4, 424(sp)                    # 4-byte Folded Reload
	flw	fs3, 416(sp)                    # 4-byte Folded Reload
	flw	fs2, 408(sp)                    # 4-byte Folded Reload
	flw	fs1, 456(sp)                    # 4-byte Folded Reload
	.loc	1 9 48 is_stmt 1                # k135114449653664.py:9:48
	slli	a2, a1, 28
	flw	fs11, 400(sp)                   # 4-byte Folded Reload
	flw	fs10, 392(sp)                   # 4-byte Folded Reload
	flw	fs9, 384(sp)                    # 4-byte Folded Reload
	bgez	a2, .LBB0_72
	j	.LBB0_178
.LBB0_72:                               # %else104
	slli	a2, a1, 27
	bgez	a2, .LBB0_73
	j	.LBB0_179
.LBB0_73:                               # %else107
	slli	a2, a1, 26
	bgez	a2, .LBB0_74
	j	.LBB0_180
.LBB0_74:                               # %else110
	slli	a2, a1, 25
	bgez	a2, .LBB0_75
	j	.LBB0_181
.LBB0_75:                               # %else113
	slli	a2, a1, 24
	bgez	a2, .LBB0_76
	j	.LBB0_182
.LBB0_76:                               # %else116
	slli	a2, a1, 23
	bgez	a2, .LBB0_77
	j	.LBB0_183
.LBB0_77:                               # %else119
	slli	a2, a1, 22
	bgez	a2, .LBB0_78
	j	.LBB0_184
.LBB0_78:                               # %else122
	slli	a2, a1, 21
	bgez	a2, .LBB0_79
	j	.LBB0_185
.LBB0_79:                               # %else125
	slli	a2, a1, 20
	bgez	a2, .LBB0_80
	j	.LBB0_186
.LBB0_80:                               # %else128
	slli	a2, a1, 19
	bgez	a2, .LBB0_81
	j	.LBB0_187
.LBB0_81:                               # %else131
	slli	a2, a1, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	bgez	a2, .LBB0_82
	j	.LBB0_188
.LBB0_82:                               # %else134
	slli	a2, a1, 17
	lui	a3, 2
	addi	a3, a3, -1352
	add	s5, sp, a3
	bgez	a2, .LBB0_83
	j	.LBB0_189
.LBB0_83:                               # %else137
	slli	a2, a1, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v0, v8, v8
	bgez	a2, .LBB0_84
	j	.LBB0_190
.LBB0_84:                               # %else140
	slli	a2, a1, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v0, a0
	bgez	a2, .LBB0_85
	j	.LBB0_191
.LBB0_85:                               # %else143
	slli	a0, a1, 14
	bgez	a0, .LBB0_86
	j	.LBB0_192
.LBB0_86:                               # %else146
	slli	a0, a1, 13
	bgez	a0, .LBB0_87
	j	.LBB0_193
.LBB0_87:                               # %else149
	slli	a0, a1, 12
	bgez	a0, .LBB0_88
	j	.LBB0_194
.LBB0_88:                               # %else152
	slli	a0, a1, 11
	bgez	a0, .LBB0_89
	j	.LBB0_195
.LBB0_89:                               # %else155
	slli	a0, a1, 10
	bgez	a0, .LBB0_90
	j	.LBB0_196
.LBB0_90:                               # %else158
	slli	a0, a1, 9
	bgez	a0, .LBB0_91
	j	.LBB0_197
.LBB0_91:                               # %else161
	slli	a0, a1, 8
	bgez	a0, .LBB0_92
	j	.LBB0_198
.LBB0_92:                               # %else164
	slli	a0, a1, 7
	bgez	a0, .LBB0_93
	j	.LBB0_199
.LBB0_93:                               # %else167
	slli	a0, a1, 6
	bgez	a0, .LBB0_94
	j	.LBB0_200
.LBB0_94:                               # %else170
	slli	a0, a1, 5
	bgez	a0, .LBB0_95
	j	.LBB0_201
.LBB0_95:                               # %else173
	slli	a0, a1, 4
	bgez	a0, .LBB0_96
	j	.LBB0_202
.LBB0_96:                               # %else176
	slli	a0, a1, 3
	bgez	a0, .LBB0_97
	j	.LBB0_203
.LBB0_97:                               # %else179
	slli	a0, a1, 2
	bgez	a0, .LBB0_98
	j	.LBB0_204
.LBB0_98:                               # %else182
	slli	a0, a1, 1
	bgez	a0, .LBB0_100
.LBB0_99:                               # %cond.load184
	.loc	1 0 48 is_stmt 0                # k135114449653664.py:0:48
	li	a0, 7
	slli	a0, a0, 10
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 440(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft10, a0
.LBB0_100:                              # %else185
	.loc	1 0 48                          # k135114449653664.py:0:48
	fsw	ft10, 456(sp)                   # 4-byte Folded Spill
	fsw	ft9, 448(sp)                    # 4-byte Folded Spill
	fsw	ft8, 440(sp)                    # 4-byte Folded Spill
	fsw	fa7, 432(sp)                    # 4-byte Folded Spill
	fsw	fa6, 424(sp)                    # 4-byte Folded Spill
	fsw	ft7, 416(sp)                    # 4-byte Folded Spill
	fsw	ft6, 408(sp)                    # 4-byte Folded Spill
	fsw	ft5, 400(sp)                    # 4-byte Folded Spill
	fsw	ft4, 392(sp)                    # 4-byte Folded Spill
	fsw	ft3, 384(sp)                    # 4-byte Folded Spill
	fsw	ft2, 376(sp)                    # 4-byte Folded Spill
	fsw	ft1, 368(sp)                    # 4-byte Folded Spill
	fsw	ft0, 360(sp)                    # 4-byte Folded Spill
	fsw	fa1, 352(sp)                    # 4-byte Folded Spill
	fsw	fa2, 624(sp)                    # 4-byte Folded Spill
	fsw	fa3, 616(sp)                    # 4-byte Folded Spill
	fsw	fa4, 608(sp)                    # 4-byte Folded Spill
	fsw	fa5, 600(sp)                    # 4-byte Folded Spill
	.loc	1 9 48                          # k135114449653664.py:9:48
	bgez	a1, .LBB0_102
# %bb.101:                              # %cond.load187
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a0, 2
	addi	a0, a0, -1152
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 320(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 464(sp)                    # 4-byte Folded Spill
.LBB0_102:                              # %else188
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a1, 32
	li	a0, 64
	csrr	a2, vlenb
	slli	a2, a2, 5
	add	a2, sp, a2
	lui	a3, 3
	addi	a3, a3, 960
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 6 21 is_stmt 1                # k135114449653664.py:6:21
	vsetvli	zero, a1, e32, m8, ta, ma
	vmslt.vx	v8, v16, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v9, v16, a0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vs1r.v	v9, (a0)                        # vscale x 8-byte Folded Spill
	.loc	1 9 58                          # k135114449653664.py:9:58
	call	__truncsfbf2
	fsw	fa0, 244(sp)                    # 4-byte Folded Spill
	flw	fa0, 248(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 248(sp)                    # 4-byte Folded Spill
	flw	fa0, 472(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 472(sp)                    # 4-byte Folded Spill
	flw	fa0, 252(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 252(sp)                    # 4-byte Folded Spill
	flw	fa0, 480(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 480(sp)                    # 4-byte Folded Spill
	flw	fa0, 256(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 256(sp)                    # 4-byte Folded Spill
	flw	fa0, 488(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 488(sp)                    # 4-byte Folded Spill
	flw	fa0, 260(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 260(sp)                    # 4-byte Folded Spill
	flw	fa0, 496(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 496(sp)                    # 4-byte Folded Spill
	flw	fa0, 264(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 264(sp)                    # 4-byte Folded Spill
	flw	fa0, 504(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 504(sp)                    # 4-byte Folded Spill
	flw	fa0, 268(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 268(sp)                    # 4-byte Folded Spill
	flw	fa0, 512(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 512(sp)                    # 4-byte Folded Spill
	flw	fa0, 272(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 272(sp)                    # 4-byte Folded Spill
	flw	fa0, 520(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 520(sp)                    # 4-byte Folded Spill
	flw	fa0, 280(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 280(sp)                    # 4-byte Folded Spill
	flw	fa0, 528(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 528(sp)                    # 4-byte Folded Spill
	flw	fa0, 288(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 288(sp)                    # 4-byte Folded Spill
	flw	fa0, 536(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 536(sp)                    # 4-byte Folded Spill
	flw	fa0, 296(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 296(sp)                    # 4-byte Folded Spill
	flw	fa0, 544(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 544(sp)                    # 4-byte Folded Spill
	flw	fa0, 304(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 304(sp)                    # 4-byte Folded Spill
	flw	fa0, 552(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 552(sp)                    # 4-byte Folded Spill
	flw	fa0, 312(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 312(sp)                    # 4-byte Folded Spill
	flw	fa0, 560(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 560(sp)                    # 4-byte Folded Spill
	flw	fa0, 320(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 320(sp)                    # 4-byte Folded Spill
	flw	fa0, 568(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 568(sp)                    # 4-byte Folded Spill
	flw	fa0, 328(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 328(sp)                    # 4-byte Folded Spill
	flw	fa0, 576(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 576(sp)                    # 4-byte Folded Spill
	flw	fa0, 336(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 336(sp)                    # 4-byte Folded Spill
	flw	fa0, 584(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 584(sp)                    # 4-byte Folded Spill
	flw	fa0, 344(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 344(sp)                    # 4-byte Folded Spill
	flw	fa0, 592(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 592(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs8
	call	__truncsfbf2
	fsw	fa0, 232(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs7
	call	__truncsfbf2
	fsw	fa0, 224(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs6
	call	__truncsfbf2
	fsw	fa0, 216(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs5
	call	__truncsfbf2
	fsw	fa0, 208(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs4
	call	__truncsfbf2
	fsw	fa0, 200(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs3
	call	__truncsfbf2
	fsw	fa0, 192(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs2
	call	__truncsfbf2
	fsw	fa0, 184(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs1
	call	__truncsfbf2
	fsw	fa0, 176(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs11
	call	__truncsfbf2
	fsw	fa0, 168(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs10
	call	__truncsfbf2
	fsw	fa0, 160(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs9
	call	__truncsfbf2
	fsw	fa0, 152(sp)                    # 4-byte Folded Spill
	flw	fa0, 600(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 144(sp)                    # 4-byte Folded Spill
	flw	fa0, 608(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 136(sp)                    # 4-byte Folded Spill
	flw	fa0, 616(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 128(sp)                    # 4-byte Folded Spill
	flw	fa0, 624(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 124(sp)                    # 4-byte Folded Spill
	flw	fa0, 352(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 352(sp)                    # 4-byte Folded Spill
	flw	fa0, 360(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 360(sp)                    # 4-byte Folded Spill
	flw	fa0, 368(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 368(sp)                    # 4-byte Folded Spill
	flw	fa0, 376(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs1, fa0
	flw	fa0, 384(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs11, fa0
	flw	fa0, 392(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs10, fa0
	flw	fa0, 400(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs9, fa0
	flw	fa0, 408(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs8, fa0
	flw	fa0, 416(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs7, fa0
	flw	fa0, 424(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs6, fa0
	flw	fa0, 432(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs5, fa0
	flw	fa0, 440(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs4, fa0
	flw	fa0, 448(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs3, fa0
	flw	fa0, 456(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs2, fa0
	flw	fa0, 464(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	sd	a0, 624(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs2
	sd	a0, 616(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs3
	sd	a0, 608(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs4
	sd	a0, 600(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs5
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs6
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs7
	sd	a0, 448(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs8
	sd	a0, 440(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs9
	sd	a0, 432(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs10
	sd	a0, 424(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs11
	sd	a0, 416(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs1
	sd	a0, 408(sp)                     # 8-byte Folded Spill
	flw	fa5, 368(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 400(sp)                     # 8-byte Folded Spill
	flw	fa5, 360(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 392(sp)                     # 8-byte Folded Spill
	flw	fa5, 352(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 384(sp)                     # 8-byte Folded Spill
	flw	fa5, 124(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 376(sp)                     # 8-byte Folded Spill
	flw	fa5, 128(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 368(sp)                     # 8-byte Folded Spill
	flw	fa5, 136(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 360(sp)                     # 8-byte Folded Spill
	flw	fa5, 144(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 352(sp)                     # 8-byte Folded Spill
	flw	fa5, 152(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	flw	fa5, 160(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 160(sp)                     # 8-byte Folded Spill
	flw	fa5, 168(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	flw	fa5, 176(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 176(sp)                     # 8-byte Folded Spill
	flw	fa5, 184(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	flw	fa5, 192(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 192(sp)                     # 8-byte Folded Spill
	flw	fa5, 200(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	flw	fa5, 208(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	flw	fa5, 216(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	flw	fa5, 224(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 224(sp)                     # 8-byte Folded Spill
	flw	fa5, 232(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 232(sp)                     # 8-byte Folded Spill
	flw	fa5, 592(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 592(sp)                     # 8-byte Folded Spill
	flw	fa5, 344(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 344(sp)                     # 8-byte Folded Spill
	flw	fa5, 584(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 584(sp)                     # 8-byte Folded Spill
	flw	fa5, 336(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 336(sp)                     # 8-byte Folded Spill
	flw	fa5, 576(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 576(sp)                     # 8-byte Folded Spill
	flw	fa5, 328(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 328(sp)                     # 8-byte Folded Spill
	flw	fa5, 568(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 568(sp)                     # 8-byte Folded Spill
	flw	fa5, 320(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 320(sp)                     # 8-byte Folded Spill
	flw	fa5, 560(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 560(sp)                     # 8-byte Folded Spill
	flw	fa5, 312(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 312(sp)                     # 8-byte Folded Spill
	flw	fa5, 552(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 552(sp)                     # 8-byte Folded Spill
	flw	fa5, 304(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 304(sp)                     # 8-byte Folded Spill
	flw	fa5, 544(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 544(sp)                     # 8-byte Folded Spill
	flw	fa5, 296(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 296(sp)                     # 8-byte Folded Spill
	flw	fa5, 536(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 536(sp)                     # 8-byte Folded Spill
	flw	fa5, 288(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 288(sp)                     # 8-byte Folded Spill
	flw	fa5, 528(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	flw	fa5, 280(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 280(sp)                     # 8-byte Folded Spill
	flw	fa5, 520(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	flw	fa5, 272(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 272(sp)                     # 8-byte Folded Spill
	flw	fa5, 512(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	flw	fa5, 268(sp)                    # 4-byte Folded Reload
	fmv.x.w	s4, fa5
	flw	fa5, 504(sp)                    # 4-byte Folded Reload
	fmv.x.w	s6, fa5
	flw	fa5, 264(sp)                    # 4-byte Folded Reload
	fmv.x.w	s7, fa5
	flw	fa5, 496(sp)                    # 4-byte Folded Reload
	fmv.x.w	s8, fa5
	flw	fa5, 260(sp)                    # 4-byte Folded Reload
	fmv.x.w	s9, fa5
	flw	fa5, 488(sp)                    # 4-byte Folded Reload
	fmv.x.w	s10, fa5
	flw	fa5, 256(sp)                    # 4-byte Folded Reload
	fmv.x.w	s3, fa5
	flw	fa5, 480(sp)                    # 4-byte Folded Reload
	fmv.x.w	s11, fa5
	flw	fa5, 252(sp)                    # 4-byte Folded Reload
	fmv.x.w	s2, fa5
	flw	fa5, 472(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 528(sp)                     # 8-byte Folded Spill
	flw	fa5, 248(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 520(sp)                     # 8-byte Folded Spill
	flw	fa5, 244(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 512(sp)                     # 8-byte Folded Spill
	fmv.s	fa0, fs0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 696
	add	a0, sp, a0
	sh	s2, 2000(a0)
	sh	s11, 2002(a0)
	sh	s3, 2004(a0)
	mv	s3, a0
	sh	s10, 2006(a0)
	sh	s9, 2008(a0)
	sh	s8, 2010(a0)
	sh	s7, 2012(a0)
	sh	s6, 2014(a0)
	sh	s4, 2016(a0)
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	sh	a0, 2018(s3)
	ld	a0, 272(sp)                     # 8-byte Folded Reload
	sh	a0, 2020(s3)
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	sh	a0, 2022(s3)
	ld	a0, 280(sp)                     # 8-byte Folded Reload
	sh	a0, 2024(s3)
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	sh	a0, 2026(s3)
	ld	a0, 288(sp)                     # 8-byte Folded Reload
	sh	a0, 2028(s3)
	ld	a0, 536(sp)                     # 8-byte Folded Reload
	sh	a0, 2030(s3)
	ld	a0, 296(sp)                     # 8-byte Folded Reload
	sh	a0, 2032(s3)
	ld	a0, 544(sp)                     # 8-byte Folded Reload
	sh	a0, 2034(s3)
	ld	a0, 304(sp)                     # 8-byte Folded Reload
	sh	a0, 2036(s3)
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	sh	a0, 2038(s3)
	ld	a0, 312(sp)                     # 8-byte Folded Reload
	sh	a0, 2040(s3)
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	sh	a0, 2042(s3)
	ld	a0, 320(sp)                     # 8-byte Folded Reload
	sh	a0, 2044(s3)
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	sh	a0, 2046(s3)
	ld	a0, 328(sp)                     # 8-byte Folded Reload
	sh	a0, 0(s5)
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	sh	a0, 2(s5)
	ld	a0, 336(sp)                     # 8-byte Folded Reload
	sh	a0, 4(s5)
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	sh	a0, 6(s5)
	ld	a0, 512(sp)                     # 8-byte Folded Reload
	sh	a0, 1994(s3)
	ld	a0, 520(sp)                     # 8-byte Folded Reload
	sh	a0, 1996(s3)
	ld	a0, 528(sp)                     # 8-byte Folded Reload
	sh	a0, 1998(s3)
	fmv.x.w	a0, fa0
	sh	a0, 1992(s3)
	ld	a0, 344(sp)                     # 8-byte Folded Reload
	sh	a0, 72(s5)
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	sh	a0, 74(s5)
	ld	a0, 232(sp)                     # 8-byte Folded Reload
	sh	a0, 76(s5)
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	sh	a0, 78(s5)
	ld	a0, 216(sp)                     # 8-byte Folded Reload
	sh	a0, 80(s5)
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	sh	a0, 82(s5)
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	sh	a0, 84(s5)
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	sh	a0, 86(s5)
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	sh	a0, 88(s5)
	lui	a0, 2
	addi	a0, a0, -1408
	add	a0, sp, a0
	ld	a1, 176(sp)                     # 8-byte Folded Reload
	sh	a1, 90(s5)
	li	a1, 27
	slli	a1, a1, 8
	add	a1, sp, a1
	fmv.w.x	fa5, zero
	ld	a2, 168(sp)                     # 8-byte Folded Reload
	sh	a2, 92(s5)
	ld	a2, 160(sp)                     # 8-byte Folded Reload
	sh	a2, 94(s5)
	ld	a2, 152(sp)                     # 8-byte Folded Reload
	sh	a2, 96(s5)
	ld	a2, 352(sp)                     # 8-byte Folded Reload
	sh	a2, 98(s5)
	ld	a2, 360(sp)                     # 8-byte Folded Reload
	sh	a2, 100(s5)
	ld	a2, 368(sp)                     # 8-byte Folded Reload
	sh	a2, 102(s5)
	ld	a2, 376(sp)                     # 8-byte Folded Reload
	sh	a2, 104(s5)
	ld	a2, 384(sp)                     # 8-byte Folded Reload
	sh	a2, 106(s5)
	ld	a2, 392(sp)                     # 8-byte Folded Reload
	sh	a2, 108(s5)
	ld	a2, 400(sp)                     # 8-byte Folded Reload
	sh	a2, 110(s5)
	ld	a2, 408(sp)                     # 8-byte Folded Reload
	sh	a2, 112(s5)
	ld	a2, 416(sp)                     # 8-byte Folded Reload
	sh	a2, 114(s5)
	ld	a2, 424(sp)                     # 8-byte Folded Reload
	sh	a2, 116(s5)
	ld	a2, 432(sp)                     # 8-byte Folded Reload
	sh	a2, 118(s5)
	ld	a2, 440(sp)                     # 8-byte Folded Reload
	sh	a2, 120(s5)
	ld	a2, 448(sp)                     # 8-byte Folded Reload
	sh	a2, 122(s5)
	ld	a2, 456(sp)                     # 8-byte Folded Reload
	sh	a2, 124(s5)
	ld	a2, 464(sp)                     # 8-byte Folded Reload
	sh	a2, 126(s5)
	ld	a2, 600(sp)                     # 8-byte Folded Reload
	sh	a2, 128(s5)
	ld	a2, 608(sp)                     # 8-byte Folded Reload
	sh	a2, 130(s5)
	ld	a2, 616(sp)                     # 8-byte Folded Reload
	sh	a2, 132(s5)
	ld	a2, 624(sp)                     # 8-byte Folded Reload
	sh	a2, 134(s5)
	li	a2, 32
	vsetvli	zero, a2, e32, m8, ta, ma
	vle16.v	v16, (a0)
	vle16.v	v24, (a1)
	lui	a0, 2
	addi	a0, a0, -1344
	add	a0, sp, a0
	lui	a1, 2
	addi	a1, a1, -1216
	add	a1, sp, a1
	vzext.vf2	v8, v16
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	vsll.vi	v8, v8, 16
	.loc	1 10 12                         # k135114449653664.py:10:12
	vfrsub.vf	v8, v8, fa5
	vfrsub.vf	v16, v16, fa5
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vmv2r.v	v8, v16
	csrr	a2, vlenb
	li	a3, 24
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 3
	addi	a3, a3, 960
	add	a2, a2, a3
	vs4r.v	v8, (a2)                        # vscale x 32-byte Folded Spill
	vse16.v	v16, (a0)
	vmv2r.v	v8, v24
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a2, 3
	addi	a2, a2, 960
	add	a0, a0, a2
	vs4r.v	v8, (a0)                        # vscale x 32-byte Folded Spill
	vse16.v	v24, (a1)
	lh	a0, 64(s5)
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	lh	a0, 66(s5)
	sd	a0, 480(sp)                     # 8-byte Folded Spill
	lh	a0, 68(s5)
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	lh	a0, 70(s5)
	sd	a0, 496(sp)                     # 8-byte Folded Spill
	lh	a0, 56(s5)
	sd	a0, 440(sp)                     # 8-byte Folded Spill
	lh	a0, 58(s5)
	sd	a0, 448(sp)                     # 8-byte Folded Spill
	lh	a0, 60(s5)
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	lh	a0, 62(s5)
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	lh	s8, 48(s5)
	lh	s7, 50(s5)
	lh	s6, 52(s5)
	lh	s4, 54(s5)
	lh	s2, 40(s5)
	lh	s11, 42(s5)
	lh	s10, 44(s5)
	lh	s9, 46(s5)
	lh	a0, 192(s5)
	sd	a0, 600(sp)                     # 8-byte Folded Spill
	lh	a0, 194(s5)
	sd	a0, 608(sp)                     # 8-byte Folded Spill
	lh	a0, 196(s5)
	sd	a0, 616(sp)                     # 8-byte Folded Spill
	lh	a0, 198(s5)
	sd	a0, 624(sp)                     # 8-byte Folded Spill
	lh	a0, 184(s5)
	sd	a0, 568(sp)                     # 8-byte Folded Spill
	lh	a0, 186(s5)
	sd	a0, 576(sp)                     # 8-byte Folded Spill
	lh	a0, 188(s5)
	sd	a0, 584(sp)                     # 8-byte Folded Spill
	lh	a0, 190(s5)
	sd	a0, 592(sp)                     # 8-byte Folded Spill
	lh	a0, 176(s5)
	sd	a0, 536(sp)                     # 8-byte Folded Spill
	lh	a0, 178(s5)
	sd	a0, 544(sp)                     # 8-byte Folded Spill
	lh	a0, 180(s5)
	sd	a0, 552(sp)                     # 8-byte Folded Spill
	lh	a0, 182(s5)
	sd	a0, 560(sp)                     # 8-byte Folded Spill
	lh	a0, 168(s5)
	sd	a0, 504(sp)                     # 8-byte Folded Spill
	lh	a0, 170(s5)
	sd	a0, 512(sp)                     # 8-byte Folded Spill
	lh	a0, 172(s5)
	sd	a0, 520(sp)                     # 8-byte Folded Spill
	lh	a0, 174(s5)
	sd	a0, 528(sp)                     # 8-byte Folded Spill
	lui	a0, 3
	addi	a0, a0, 960
	add	a0, sp, a0
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v16
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	add	a0, a0, a1
	ld	s5, 960(a0)                     # 8-byte Folded Reload
	andi	a0, s5, 1
	vsetvli	zero, zero, e64, m8, ta, ma
	ld	a1, 632(sp)                     # 8-byte Folded Reload
	vadd.vx	v8, v8, a1
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bnez	a0, .LBB0_205
# %bb.103:                              # %else192
	andi	a0, s5, 2
	bnez	a0, .LBB0_206
.LBB0_104:                              # %else195
	andi	a0, s5, 4
	bnez	a0, .LBB0_207
.LBB0_105:                              # %else198
	andi	a0, s5, 8
	beqz	a0, .LBB0_106
	j	.LBB0_208
.LBB0_106:                              # %else201
	andi	a0, s5, 16
	beqz	a0, .LBB0_107
	j	.LBB0_209
.LBB0_107:                              # %else204
	andi	a0, s5, 32
	beqz	a0, .LBB0_108
	j	.LBB0_210
.LBB0_108:                              # %else207
	andi	a0, s5, 64
	beqz	a0, .LBB0_109
	j	.LBB0_211
.LBB0_109:                              # %else210
	andi	a0, s5, 128
	beqz	a0, .LBB0_110
	j	.LBB0_212
.LBB0_110:                              # %else213
	andi	a0, s5, 256
	beqz	a0, .LBB0_111
	j	.LBB0_213
.LBB0_111:                              # %else216
	andi	a0, s5, 512
	beqz	a0, .LBB0_112
	j	.LBB0_214
.LBB0_112:                              # %else219
	andi	a0, s5, 1024
	beqz	a0, .LBB0_113
	j	.LBB0_215
.LBB0_113:                              # %else222
	slli	a0, s5, 52
	bgez	a0, .LBB0_114
	j	.LBB0_216
.LBB0_114:                              # %else225
	slli	a0, s5, 51
	bgez	a0, .LBB0_116
.LBB0_115:                              # %cond.store226
	.loc	1 0 44 is_stmt 0                # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_116:                              # %else228
	slli	a0, s5, 50
	lui	a1, 3
	addi	a1, a1, 960
	add	a1, sp, a1
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	lui	a1, 3
	addi	a1, a1, 960
	add	a1, sp, a1
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_118
# %bb.117:                              # %cond.store229
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_118:                              # %else231
	slli	a0, s5, 49
	bgez	a0, .LBB0_120
# %bb.119:                              # %cond.store232
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_120:                              # %else234
	slli	a0, s5, 48
	lui	a1, 3
	addi	a1, a1, 960
	add	a1, sp, a1
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_122
# %bb.121:                              # %cond.store235
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_122:                              # %else237
	slli	a0, s5, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	ld	a1, 632(sp)                     # 8-byte Folded Reload
	vadd.vx	v8, v16, a1
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_123
	j	.LBB0_217
.LBB0_123:                              # %else240
	slli	a0, s5, 46
	bgez	a0, .LBB0_124
	j	.LBB0_218
.LBB0_124:                              # %else243
	slli	a0, s5, 45
	bgez	a0, .LBB0_125
	j	.LBB0_219
.LBB0_125:                              # %else246
	slli	a0, s5, 44
	bgez	a0, .LBB0_126
	j	.LBB0_220
.LBB0_126:                              # %else249
	slli	a0, s5, 43
	bgez	a0, .LBB0_127
	j	.LBB0_221
.LBB0_127:                              # %else252
	slli	a0, s5, 42
	bgez	a0, .LBB0_128
	j	.LBB0_222
.LBB0_128:                              # %else255
	slli	a0, s5, 41
	bgez	a0, .LBB0_129
	j	.LBB0_223
.LBB0_129:                              # %else258
	slli	a0, s5, 40
	bgez	a0, .LBB0_130
	j	.LBB0_224
.LBB0_130:                              # %else261
	slli	a0, s5, 39
	addi	s3, sp, 2047
	addi	s3, s3, 609
	bgez	a0, .LBB0_131
	j	.LBB0_225
.LBB0_131:                              # %else264
	slli	a0, s5, 38
	bgez	a0, .LBB0_132
	j	.LBB0_226
.LBB0_132:                              # %else267
	slli	a0, s5, 37
	bgez	a0, .LBB0_133
	j	.LBB0_227
.LBB0_133:                              # %else270
	slli	a0, s5, 36
	bgez	a0, .LBB0_134
	j	.LBB0_228
.LBB0_134:                              # %else273
	slli	a0, s5, 35
	bgez	a0, .LBB0_135
	j	.LBB0_229
.LBB0_135:                              # %else276
	slli	a0, s5, 34
	bgez	a0, .LBB0_136
	j	.LBB0_230
.LBB0_136:                              # %else279
	slli	a0, s5, 33
	bgez	a0, .LBB0_138
.LBB0_137:                              # %cond.store280
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1793
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_138:                              # %else282
	slli	a0, s5, 32
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_140
# %bb.139:                              # %cond.store283
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 496(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 1665
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_140:                              # %else285
	slli	a0, s5, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	ld	a1, 632(sp)                     # 8-byte Folded Reload
	vadd.vx	v8, v16, a1
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_141
	j	.LBB0_231
.LBB0_141:                              # %else288
	slli	a0, s5, 30
	bgez	a0, .LBB0_142
	j	.LBB0_232
.LBB0_142:                              # %else291
	slli	a0, s5, 29
	bgez	a0, .LBB0_143
	j	.LBB0_233
.LBB0_143:                              # %else294
	slli	a0, s5, 28
	bgez	a0, .LBB0_144
	j	.LBB0_234
.LBB0_144:                              # %else297
	slli	a0, s5, 27
	bgez	a0, .LBB0_145
	j	.LBB0_235
.LBB0_145:                              # %else300
	slli	a0, s5, 26
	bgez	a0, .LBB0_146
	j	.LBB0_236
.LBB0_146:                              # %else303
	slli	a0, s5, 25
	bgez	a0, .LBB0_147
	j	.LBB0_237
.LBB0_147:                              # %else306
	slli	a0, s5, 24
	bgez	a0, .LBB0_148
	j	.LBB0_238
.LBB0_148:                              # %else309
	slli	a0, s5, 23
	bgez	a0, .LBB0_149
	j	.LBB0_239
.LBB0_149:                              # %else312
	slli	a0, s5, 22
	bgez	a0, .LBB0_150
	j	.LBB0_240
.LBB0_150:                              # %else315
	slli	a0, s5, 21
	bgez	a0, .LBB0_151
	j	.LBB0_241
.LBB0_151:                              # %else318
	slli	a0, s5, 20
	bgez	a0, .LBB0_152
	j	.LBB0_242
.LBB0_152:                              # %else321
	slli	a0, s5, 19
	bgez	a0, .LBB0_154
.LBB0_153:                              # %cond.store322
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 513
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_154:                              # %else324
	slli	a0, s5, 18
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_156
# %bb.155:                              # %cond.store325
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 385
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1560(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_156:                              # %else327
	slli	a0, s5, 17
	bgez	a0, .LBB0_158
# %bb.157:                              # %cond.store328
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 257
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1680(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_158:                              # %else330
	slli	a0, s5, 16
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_160
# %bb.159:                              # %cond.store331
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 129
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1800(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_160:                              # %else333
	slli	a0, s5, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	ld	a1, 632(sp)                     # 8-byte Folded Reload
	vadd.vx	v8, v16, a1
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_161
	j	.LBB0_243
.LBB0_161:                              # %else336
	slli	a0, s5, 14
	bgez	a0, .LBB0_162
	j	.LBB0_244
.LBB0_162:                              # %else339
	slli	a0, s5, 13
	bgez	a0, .LBB0_163
	j	.LBB0_245
.LBB0_163:                              # %else342
	slli	a0, s5, 12
	bgez	a0, .LBB0_164
	j	.LBB0_246
.LBB0_164:                              # %else345
	slli	a0, s5, 11
	bgez	a0, .LBB0_165
	j	.LBB0_247
.LBB0_165:                              # %else348
	slli	a0, s5, 10
	bgez	a0, .LBB0_166
	j	.LBB0_248
.LBB0_166:                              # %else351
	slli	a0, s5, 9
	bgez	a0, .LBB0_167
	j	.LBB0_249
.LBB0_167:                              # %else354
	slli	a0, s5, 8
	bgez	a0, .LBB0_168
	j	.LBB0_250
.LBB0_168:                              # %else357
	slli	a0, s5, 7
	bgez	a0, .LBB0_169
	j	.LBB0_251
.LBB0_169:                              # %else360
	slli	a0, s5, 6
	bgez	a0, .LBB0_170
	j	.LBB0_252
.LBB0_170:                              # %else363
	slli	a0, s5, 5
	bgez	a0, .LBB0_171
	j	.LBB0_253
.LBB0_171:                              # %else366
	slli	a0, s5, 4
	bgez	a0, .LBB0_172
	j	.LBB0_254
.LBB0_172:                              # %else369
	slli	a0, s5, 3
	bgez	a0, .LBB0_173
	j	.LBB0_255
.LBB0_173:                              # %else372
	slli	a0, s5, 2
	bgez	a0, .LBB0_174
	j	.LBB0_256
.LBB0_174:                              # %else375
	slli	a0, s5, 1
	bgez	a0, .LBB0_175
	j	.LBB0_257
.LBB0_175:                              # %else378
	bgez	s5, .LBB0_177
.LBB0_176:                              # %cond.store379
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 640
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 760(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_177:                              # %else381
	.loc	1 11 4 epilogue_begin           # k135114449653664.py:11:4
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
.LBB0_178:                              # %cond.load103
	.cfi_restore_state
	.loc	1 9 48 is_stmt 1                # k135114449653664.py:9:48
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v24, 3
	vmv.x.s	a2, v16
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs7, a2
	slli	a2, a1, 27
	bltz	a2, .LBB0_179
	j	.LBB0_73
.LBB0_179:                              # %cond.load106
	.loc	1 0 48 is_stmt 0                # k135114449653664.py:0:48
	lui	a2, 2
	addi	a2, a2, 1792
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 1080(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs6, a2
	slli	a2, a1, 26
	bltz	a2, .LBB0_180
	j	.LBB0_74
.LBB0_180:                              # %cond.load109
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a2, 2
	addi	a2, a2, 1664
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 960(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs5, a2
	slli	a2, a1, 25
	bltz	a2, .LBB0_181
	j	.LBB0_75
.LBB0_181:                              # %cond.load112
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a2, 19
	slli	a2, a2, 9
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 840(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs4, a2
	slli	a2, a1, 24
	bltz	a2, .LBB0_182
	j	.LBB0_76
.LBB0_182:                              # %cond.load115
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a2, 2
	addi	a2, a2, 1408
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 720(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs3, a2
	slli	a2, a1, 23
	bltz	a2, .LBB0_183
	j	.LBB0_77
.LBB0_183:                              # %cond.load118
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a2, 2
	addi	a2, a2, 1280
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 600(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs2, a2
	slli	a2, a1, 22
	bltz	a2, .LBB0_184
	j	.LBB0_78
.LBB0_184:                              # %cond.load121
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a2, 2
	addi	a2, a2, 1152
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 480(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs1, a2
	slli	a2, a1, 21
	bltz	a2, .LBB0_185
	j	.LBB0_79
.LBB0_185:                              # %cond.load124
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a2, 9
	slli	a2, a2, 10
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 360(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs11, a2
	slli	a2, a1, 20
	bltz	a2, .LBB0_186
	j	.LBB0_80
.LBB0_186:                              # %cond.load127
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a2, 2
	addi	a2, a2, 896
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 240(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs10, a2
	slli	a2, a1, 19
	bltz	a2, .LBB0_187
	j	.LBB0_81
.LBB0_187:                              # %cond.load130
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a2, 2
	addi	a2, a2, 768
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 120(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs9, a2
	slli	a2, a1, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	bltz	a2, .LBB0_188
	j	.LBB0_82
.LBB0_188:                              # %cond.load133
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a2, 2
	addi	a2, a2, 640
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 0(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	slli	a2, a1, 17
	lui	a3, 2
	addi	a3, a3, -1352
	add	s5, sp, a3
	bltz	a2, .LBB0_189
	j	.LBB0_83
.LBB0_189:                              # %cond.load136
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a2, 17
	slli	a2, a2, 9
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 1976(s5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa4, a2
	slli	a2, a1, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v0, v8, v8
	bltz	a2, .LBB0_190
	j	.LBB0_84
.LBB0_190:                              # %cond.load139
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a2, 2
	addi	a2, a2, 384
	add	a2, sp, a2
	.loc	1 9 48                          # k135114449653664.py:9:48
	vse64.v	v24, (a2)
	ld	a2, 1856(s5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa3, a2
	slli	a2, a1, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v0, a0
	bltz	a2, .LBB0_191
	j	.LBB0_85
.LBB0_191:                              # %cond.load142
	vmv.x.s	a0, v8
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa2, a0
	slli	a0, a1, 14
	bltz	a0, .LBB0_192
	j	.LBB0_86
.LBB0_192:                              # %cond.load145
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa1, a0
	slli	a0, a1, 13
	bltz	a0, .LBB0_193
	j	.LBB0_87
.LBB0_193:                              # %cond.load148
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 2
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft0, a0
	slli	a0, a1, 12
	bltz	a0, .LBB0_194
	j	.LBB0_88
.LBB0_194:                              # %cond.load151
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 3
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft1, a0
	slli	a0, a1, 11
	bltz	a0, .LBB0_195
	j	.LBB0_89
.LBB0_195:                              # %cond.load154
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a0, 2
	addi	a0, a0, 256
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1640(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft2, a0
	slli	a0, a1, 10
	bltz	a0, .LBB0_196
	j	.LBB0_90
.LBB0_196:                              # %cond.load157
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a0, 2
	addi	a0, a0, 128
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1520(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft3, a0
	slli	a0, a1, 9
	bltz	a0, .LBB0_197
	j	.LBB0_91
.LBB0_197:                              # %cond.load160
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a0, 2
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1400(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft4, a0
	slli	a0, a1, 8
	bltz	a0, .LBB0_198
	j	.LBB0_92
.LBB0_198:                              # %cond.load163
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a0, 2
	addi	a0, a0, -128
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1280(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft5, a0
	slli	a0, a1, 7
	bltz	a0, .LBB0_199
	j	.LBB0_93
.LBB0_199:                              # %cond.load166
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a0, 31
	slli	a0, a0, 8
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1160(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft6, a0
	slli	a0, a1, 6
	bltz	a0, .LBB0_200
	j	.LBB0_94
.LBB0_200:                              # %cond.load169
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a0, 2
	addi	a0, a0, -384
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1040(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft7, a0
	slli	a0, a1, 5
	bltz	a0, .LBB0_201
	j	.LBB0_95
.LBB0_201:                              # %cond.load172
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a0, 15
	slli	a0, a0, 9
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 920(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa6, a0
	slli	a0, a1, 4
	bltz	a0, .LBB0_202
	j	.LBB0_96
.LBB0_202:                              # %cond.load175
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a0, 2
	addi	a0, a0, -640
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 800(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa7, a0
	slli	a0, a1, 3
	bltz	a0, .LBB0_203
	j	.LBB0_97
.LBB0_203:                              # %cond.load178
	.loc	1 0 48                          # k135114449653664.py:0:48
	li	a0, 29
	slli	a0, a0, 8
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 680(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft8, a0
	slli	a0, a1, 2
	bltz	a0, .LBB0_204
	j	.LBB0_98
.LBB0_204:                              # %cond.load181
	.loc	1 0 48                          # k135114449653664.py:0:48
	lui	a0, 2
	addi	a0, a0, -896
	add	a0, sp, a0
	.loc	1 9 48                          # k135114449653664.py:9:48
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 560(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	ft9, a0
	slli	a0, a1, 1
	bgez	a0, .LBB0_258
	j	.LBB0_99
.LBB0_258:                              # %cond.load181
	j	.LBB0_100
.LBB0_205:                              # %cond.store
	.loc	1 0 48                          # k135114449653664.py:0:48
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44 is_stmt 1               # k135114449653664.py:11:44
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 2
	beqz	a0, .LBB0_104
.LBB0_206:                              # %cond.store193
	.loc	1 0 44 is_stmt 0                # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 4
	beqz	a0, .LBB0_105
.LBB0_207:                              # %cond.store196
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 8
	beqz	a0, .LBB0_106
.LBB0_208:                              # %cond.store199
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 16
	bnez	a0, .LBB0_209
	j	.LBB0_107
.LBB0_209:                              # %cond.store202
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 13
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 32
	bnez	a0, .LBB0_210
	j	.LBB0_108
.LBB0_210:                              # %cond.store205
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 64
	bnez	a0, .LBB0_211
	j	.LBB0_109
.LBB0_211:                              # %cond.store208
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 128
	bnez	a0, .LBB0_212
	j	.LBB0_110
.LBB0_212:                              # %cond.store211
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 256
	bnez	a0, .LBB0_213
	j	.LBB0_111
.LBB0_213:                              # %cond.store214
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 3
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 512
	bnez	a0, .LBB0_214
	j	.LBB0_112
.LBB0_214:                              # %cond.store217
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 1024
	bnez	a0, .LBB0_215
	j	.LBB0_113
.LBB0_215:                              # %cond.store220
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 52
	bltz	a0, .LBB0_216
	j	.LBB0_114
.LBB0_216:                              # %cond.store223
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	.loc	1 11 44                         # k135114449653664.py:11:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 51
	bgez	a0, .LBB0_259
	j	.LBB0_115
.LBB0_259:                              # %cond.store223
	j	.LBB0_116
.LBB0_217:                              # %cond.store238
	slli	s2, s2, 16
	fmv.w.x	fa0, s2
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 46
	bltz	a0, .LBB0_218
	j	.LBB0_124
.LBB0_218:                              # %cond.store241
	slli	s11, s11, 16
	fmv.w.x	fa0, s11
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 45
	bltz	a0, .LBB0_219
	j	.LBB0_125
.LBB0_219:                              # %cond.store244
	slli	s10, s10, 16
	fmv.w.x	fa0, s10
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 44
	bltz	a0, .LBB0_220
	j	.LBB0_126
.LBB0_220:                              # %cond.store247
	slli	s9, s9, 16
	fmv.w.x	fa0, s9
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 43
	bltz	a0, .LBB0_221
	j	.LBB0_127
.LBB0_221:                              # %cond.store250
	slli	s8, s8, 16
	fmv.w.x	fa0, s8
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 42
	bltz	a0, .LBB0_222
	j	.LBB0_128
.LBB0_222:                              # %cond.store253
	slli	s7, s7, 16
	fmv.w.x	fa0, s7
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 41
	bltz	a0, .LBB0_223
	j	.LBB0_129
.LBB0_223:                              # %cond.store256
	slli	s6, s6, 16
	fmv.w.x	fa0, s6
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 40
	bltz	a0, .LBB0_224
	j	.LBB0_130
.LBB0_224:                              # %cond.store259
	slli	s4, s4, 16
	fmv.w.x	fa0, s4
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 39
	addi	s3, sp, 2047
	addi	s3, s3, 609
	bltz	a0, .LBB0_225
	j	.LBB0_131
.LBB0_225:                              # %cond.store262
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 440(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 38
	bltz	a0, .LBB0_226
	j	.LBB0_132
.LBB0_226:                              # %cond.store265
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 448(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 37
	bltz	a0, .LBB0_227
	j	.LBB0_133
.LBB0_227:                              # %cond.store268
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 456(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 36
	bltz	a0, .LBB0_228
	j	.LBB0_134
.LBB0_228:                              # %cond.store271
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 464(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 35
	bltz	a0, .LBB0_229
	j	.LBB0_135
.LBB0_229:                              # %cond.store274
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 472(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 34
	bltz	a0, .LBB0_230
	j	.LBB0_136
.LBB0_230:                              # %cond.store277
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 480(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1921
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 33
	bgez	a0, .LBB0_260
	j	.LBB0_137
.LBB0_260:                              # %cond.store277
	j	.LBB0_138
.LBB0_231:                              # %cond.store286
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 30
	bltz	a0, .LBB0_232
	j	.LBB0_142
.LBB0_232:                              # %cond.store289
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 29
	bltz	a0, .LBB0_233
	j	.LBB0_143
.LBB0_233:                              # %cond.store292
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 28
	bltz	a0, .LBB0_234
	j	.LBB0_144
.LBB0_234:                              # %cond.store295
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 27
	bltz	a0, .LBB0_235
	j	.LBB0_145
.LBB0_235:                              # %cond.store298
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1537
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 26
	bltz	a0, .LBB0_236
	j	.LBB0_146
.LBB0_236:                              # %cond.store301
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1409
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 25
	bltz	a0, .LBB0_237
	j	.LBB0_147
.LBB0_237:                              # %cond.store304
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1281
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 24
	bltz	a0, .LBB0_238
	j	.LBB0_148
.LBB0_238:                              # %cond.store307
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1153
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 23
	bltz	a0, .LBB0_239
	j	.LBB0_149
.LBB0_239:                              # %cond.store310
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1025
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 22
	bltz	a0, .LBB0_240
	j	.LBB0_150
.LBB0_240:                              # %cond.store313
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 897
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 21
	bltz	a0, .LBB0_241
	j	.LBB0_151
.LBB0_241:                              # %cond.store316
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 769
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 20
	bltz	a0, .LBB0_242
	j	.LBB0_152
.LBB0_242:                              # %cond.store319
	.loc	1 0 44                          # k135114449653664.py:0:44
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 960
	add	a0, a0, a1
	vl4r.v	v8, (a0)                        # vscale x 32-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 641
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 19
	bgez	a0, .LBB0_261
	j	.LBB0_153
.LBB0_261:                              # %cond.store319
	j	.LBB0_154
.LBB0_243:                              # %cond.store334
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 504(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 14
	bltz	a0, .LBB0_244
	j	.LBB0_162
.LBB0_244:                              # %cond.store337
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 512(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 13
	bltz	a0, .LBB0_245
	j	.LBB0_163
.LBB0_245:                              # %cond.store340
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 520(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 12
	bltz	a0, .LBB0_246
	j	.LBB0_164
.LBB0_246:                              # %cond.store343
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 528(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 11
	bltz	a0, .LBB0_247
	j	.LBB0_165
.LBB0_247:                              # %cond.store346
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 536(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -2016(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 10
	bltz	a0, .LBB0_248
	j	.LBB0_166
.LBB0_248:                              # %cond.store349
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 544(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1920
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1960(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 9
	bltz	a0, .LBB0_249
	j	.LBB0_167
.LBB0_249:                              # %cond.store352
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1792
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1840(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 8
	bltz	a0, .LBB0_250
	j	.LBB0_168
.LBB0_250:                              # %cond.store355
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1664
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1720(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 7
	bltz	a0, .LBB0_251
	j	.LBB0_169
.LBB0_251:                              # %cond.store358
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1536
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1600(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 6
	bltz	a0, .LBB0_252
	j	.LBB0_170
.LBB0_252:                              # %cond.store361
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1408
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1480(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 5
	bltz	a0, .LBB0_253
	j	.LBB0_171
.LBB0_253:                              # %cond.store364
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1280
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1360(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 4
	bltz	a0, .LBB0_254
	j	.LBB0_172
.LBB0_254:                              # %cond.store367
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1152
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1240(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 3
	bltz	a0, .LBB0_255
	j	.LBB0_173
.LBB0_255:                              # %cond.store370
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1024
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1120(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 2
	bltz	a0, .LBB0_256
	j	.LBB0_174
.LBB0_256:                              # %cond.store373
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 896
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1000(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 1
	bltz	a0, .LBB0_257
	j	.LBB0_175
.LBB0_257:                              # %cond.store376
	.loc	1 0 44                          # k135114449653664.py:0:44
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 11 44                         # k135114449653664.py:11:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 768
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 960
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 880(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	bgez	s5, .LBB0_262
	j	.LBB0_176
.LBB0_262:                              # %cond.store376
	j	.LBB0_177
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused_neg_slice_transpose_view_3, .Lfunc_end0-triton_poi_fused_neg_slice_transpose_view_3
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
	.asciz	"k135114449653664.py"           # string offset=7 ; k135114449653664.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
