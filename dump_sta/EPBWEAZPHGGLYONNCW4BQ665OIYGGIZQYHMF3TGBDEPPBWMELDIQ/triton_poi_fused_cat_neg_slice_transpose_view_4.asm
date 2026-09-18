	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused_cat_neg_slice_transpose_view_4 # -- Begin function triton_poi_fused_cat_neg_slice_transpose_view_4
	.p2align	2
	.type	triton_poi_fused_cat_neg_slice_transpose_view_4,@function
triton_poi_fused_cat_neg_slice_transpose_view_4: # @triton_poi_fused_cat_neg_slice_transpose_view_4
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294384224.py"
	.loc	1 2 0                           # k135114294384224.py:2:0
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
	fsd	fs0, 1968(sp)                   # 8-byte Folded Spill
	fsd	fs1, 1960(sp)                   # 8-byte Folded Spill
	fsd	fs2, 1952(sp)                   # 8-byte Folded Spill
	fsd	fs3, 1944(sp)                   # 8-byte Folded Spill
	fsd	fs4, 1936(sp)                   # 8-byte Folded Spill
	fsd	fs5, 1928(sp)                   # 8-byte Folded Spill
	fsd	fs6, 1920(sp)                   # 8-byte Folded Spill
	fsd	fs7, 1912(sp)                   # 8-byte Folded Spill
	fsd	fs8, 1904(sp)                   # 8-byte Folded Spill
	fsd	fs9, 1896(sp)                   # 8-byte Folded Spill
	fsd	fs10, 1888(sp)                  # 8-byte Folded Spill
	fsd	fs11, 1880(sp)                  # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s2, -24
	.cfi_offset s3, -32
	.cfi_offset s4, -40
	.cfi_offset s5, -48
	.cfi_offset s6, -56
	.cfi_offset fs0, -64
	.cfi_offset fs1, -72
	.cfi_offset fs2, -80
	.cfi_offset fs3, -88
	.cfi_offset fs4, -96
	.cfi_offset fs5, -104
	.cfi_offset fs6, -112
	.cfi_offset fs7, -120
	.cfi_offset fs8, -128
	.cfi_offset fs9, -136
	.cfi_offset fs10, -144
	.cfi_offset fs11, -152
	addi	s0, sp, 2032
	.cfi_def_cfa s0, 0
	.cfi_remember_state
	lui	a2, 6
	addi	a2, a2, -1248
	sub	sp, sp, a2
	csrr	a2, vlenb
	li	a4, 112
	mul	a2, a2, a4
	sub	sp, sp, a2
	andi	sp, sp, -128
	mv	s2, a1
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114294384224.py:4:33
	slli	a5, a3, 7
	li	a1, 32
	lui	a2, 599186
	.loc	1 5 23                          # k135114294384224.py:5:23
	vsetvli	zero, a1, e32, m8, ta, ma
	vid.v	v8
	addi	a4, a2, 1171
	vor.vx	v0, v8, a5
	csrr	a2, vlenb
	li	a3, 56
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 6
	addi	a3, a3, 624
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 9 19                          # k135114294384224.py:9:19
	vmulh.vx	v16, v0, a4
	vadd.vv	v16, v16, v0
	vsra.vi	v16, v16, 8
	vsrl.vi	v24, v16, 31
	vadd.vv	v16, v16, v24
	.loc	1 5 23                          # k135114294384224.py:5:23
	vmv.v.x	v24, a5
	.loc	1 8 21                          # k135114294384224.py:8:21
	vsra.vi	v24, v24, 31
	vsrl.vi	v8, v24, 27
	csrr	a2, vlenb
	li	a3, 48
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 6
	addi	a3, a3, 624
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vadd.vv	v24, v0, v8
	vsra.vi	v8, v24, 5
	csrr	a2, vlenb
	li	a3, 40
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 6
	addi	a3, a3, 624
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 8 27 is_stmt 0                # k135114294384224.py:8:27
	vmulh.vx	v0, v8, a4
	vadd.vv	v0, v0, v8
	vsra.vi	v0, v0, 3
	vsrl.vi	v8, v0, 31
	vadd.vv	v8, v0, v8
	li	a3, -32
	li	a2, 14
	.loc	1 7 19 is_stmt 1                # k135114294384224.py:7:19
	vand.vx	v24, v24, a3
	csrr	a6, vlenb
	li	a7, 56
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 624
	add	a6, a6, a7
	vl8r.v	v0, (a6)                        # vscale x 64-byte Folded Reload
	vsub.vv	v24, v0, v24
	.loc	1 11 38                         # k135114294384224.py:11:38
	vsll.vi	v16, v16, 6
	csrr	a6, vlenb
	slli	a6, a6, 3
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 624
	add	a6, a6, a7
	vs8r.v	v24, (a6)                       # vscale x 64-byte Folded Spill
	.loc	1 11 35 is_stmt 0               # k135114294384224.py:11:35
	vadd.vv	v16, v16, v24
	csrr	a6, vlenb
	li	a7, 40
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 624
	add	a6, a6, a7
	vl8r.v	v24, (a6)                       # vscale x 64-byte Folded Reload
	.loc	1 8 27 is_stmt 1                # k135114294384224.py:8:27
	vnmsub.vx	v8, a2, v24
	.loc	1 11 47                         # k135114294384224.py:11:47
	vsll.vi	v8, v8, 7
	.loc	1 11 43 is_stmt 0               # k135114294384224.py:11:43
	vadd.vv	v0, v16, v8
	li	a6, 96
	li	a7, 64
	li	t0, 896
	vid.v	v24
	.loc	1 5 23 is_stmt 1                # k135114294384224.py:5:23
	vadd.vx	v8, v24, a6
	vor.vx	v16, v8, a5
	vadd.vx	v8, v24, a7
	vor.vx	v24, v8, a5
	csrr	a6, vlenb
	li	a7, 104
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 624
	add	a6, a6, a7
	vs8r.v	v16, (a6)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114294384224.py:6:21
	vmslt.vx	v8, v16, t0
	csrr	a6, vlenb
	li	a7, 96
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 624
	add	a6, a6, a7
	vs8r.v	v24, (a6)                       # vscale x 64-byte Folded Spill
	vmslt.vx	v9, v24, t0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	csrr	a6, vlenb
	li	a7, 80
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 624
	add	a6, a6, a7
	vs1r.v	v9, (a6)                        # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294384224.py:5:23
	vsetvli	zero, a1, e32, m8, ta, ma
	vid.v	v8
	vadd.vx	v8, v8, a1
	vor.vx	v24, v8, a5
	csrr	a5, vlenb
	li	a6, 56
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 6
	addi	a6, a6, 624
	add	a5, a5, a6
	vl8r.v	v16, (a5)                       # vscale x 64-byte Folded Reload
	.loc	1 6 21                          # k135114294384224.py:6:21
	vmslt.vx	v9, v16, t0
	csrr	a5, vlenb
	li	a6, 88
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 6
	addi	a6, a6, 624
	add	a5, a5, a6
	vs8r.v	v24, (a5)                       # vscale x 64-byte Folded Spill
	vmslt.vx	v8, v24, t0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	csrr	a5, vlenb
	li	a6, 80
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 6
	addi	a6, a6, 624
	add	a5, a5, a6
	vl1r.v	v8, (a5)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v9, v8, 8
	csrr	a5, vlenb
	slli	a5, a5, 5
	add	a5, sp, a5
	lui	a6, 6
	addi	a6, a6, 624
	add	a5, a5, a6
	vs1r.v	v9, (a5)                        # vscale x 8-byte Folded Spill
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	a5, v9
	andi	a6, a5, 1
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v8, v0, v0
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v8, a0
	beqz	a6, .LBB0_2
# %bb.1:                                # %cond.load
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs5, zero
	fmv.w.x	fa0, a6
	fmv.s	fs6, fs5
	fmv.s	fs4, fs5
	fmv.s	fs3, fs5
	fmv.s	fs2, fs5
	fmv.s	fs1, fs5
	fmv.s	fs0, fs5
	fmv.s	fs11, fs5
	fmv.s	fs10, fs5
	fmv.s	fs9, fs5
	fmv.s	fs8, fs5
	fmv.s	fs7, fs5
	fsw	fs5, 52(sp)                     # 4-byte Folded Spill
	fsw	fs5, 56(sp)                     # 4-byte Folded Spill
	fsw	fs5, 60(sp)                     # 4-byte Folded Spill
	fsw	fs5, 64(sp)                     # 4-byte Folded Spill
	fsw	fs5, 68(sp)                     # 4-byte Folded Spill
	fsw	fs5, 72(sp)                     # 4-byte Folded Spill
	fsw	fs5, 76(sp)                     # 4-byte Folded Spill
	fsw	fs5, 80(sp)                     # 4-byte Folded Spill
	fsw	fs5, 84(sp)                     # 4-byte Folded Spill
	fsw	fs5, 88(sp)                     # 4-byte Folded Spill
	fsw	fs5, 92(sp)                     # 4-byte Folded Spill
	fsw	fs5, 96(sp)                     # 4-byte Folded Spill
	fsw	fs5, 100(sp)                    # 4-byte Folded Spill
	fsw	fs5, 104(sp)                    # 4-byte Folded Spill
	fsw	fs5, 108(sp)                    # 4-byte Folded Spill
	fsw	fs5, 112(sp)                    # 4-byte Folded Spill
	fsw	fs5, 116(sp)                    # 4-byte Folded Spill
	fsw	fs5, 120(sp)                    # 4-byte Folded Spill
	fsw	fs5, 124(sp)                    # 4-byte Folded Spill
	fsw	fs5, 128(sp)                    # 4-byte Folded Spill
	fsw	fs5, 132(sp)                    # 4-byte Folded Spill
	fsw	fs5, 136(sp)                    # 4-byte Folded Spill
	fsw	fs5, 140(sp)                    # 4-byte Folded Spill
	fsw	fs5, 144(sp)                    # 4-byte Folded Spill
	fsw	fs5, 148(sp)                    # 4-byte Folded Spill
	fsw	fs5, 152(sp)                    # 4-byte Folded Spill
	fsw	fs5, 156(sp)                    # 4-byte Folded Spill
	fsw	fs5, 160(sp)                    # 4-byte Folded Spill
	fsw	fs5, 164(sp)                    # 4-byte Folded Spill
	fsw	fs5, 168(sp)                    # 4-byte Folded Spill
	fsw	fs5, 172(sp)                    # 4-byte Folded Spill
	fsw	fs5, 176(sp)                    # 4-byte Folded Spill
	fsw	fs5, 180(sp)                    # 4-byte Folded Spill
	fsw	fs5, 184(sp)                    # 4-byte Folded Spill
	fsw	fs5, 188(sp)                    # 4-byte Folded Spill
	fsw	fs5, 192(sp)                    # 4-byte Folded Spill
	fsw	fs5, 196(sp)                    # 4-byte Folded Spill
	fsw	fs5, 200(sp)                    # 4-byte Folded Spill
	fsw	fs5, 204(sp)                    # 4-byte Folded Spill
	fsw	fs5, 208(sp)                    # 4-byte Folded Spill
	fsw	fs5, 212(sp)                    # 4-byte Folded Spill
	fsw	fs5, 216(sp)                    # 4-byte Folded Spill
	fsw	fs5, 220(sp)                    # 4-byte Folded Spill
	fsw	fs5, 224(sp)                    # 4-byte Folded Spill
	fsw	fs5, 228(sp)                    # 4-byte Folded Spill
	fsw	fs5, 232(sp)                    # 4-byte Folded Spill
	fsw	fs5, 236(sp)                    # 4-byte Folded Spill
	fsw	fs5, 240(sp)                    # 4-byte Folded Spill
	fsw	fs5, 244(sp)                    # 4-byte Folded Spill
	fsw	fs5, 248(sp)                    # 4-byte Folded Spill
	fsw	fs5, 252(sp)                    # 4-byte Folded Spill
	fsw	fs5, 256(sp)                    # 4-byte Folded Spill
	fsw	fs5, 260(sp)                    # 4-byte Folded Spill
	fsw	fs5, 264(sp)                    # 4-byte Folded Spill
	fsw	fs5, 268(sp)                    # 4-byte Folded Spill
	fsw	fs5, 272(sp)                    # 4-byte Folded Spill
	fsw	fs5, 276(sp)                    # 4-byte Folded Spill
	fsw	fs5, 280(sp)                    # 4-byte Folded Spill
	fsw	fs5, 284(sp)                    # 4-byte Folded Spill
	fsw	fs5, 288(sp)                    # 4-byte Folded Spill
	fsw	fs5, 292(sp)                    # 4-byte Folded Spill
	fsw	fs5, 296(sp)                    # 4-byte Folded Spill
	fsw	fs5, 300(sp)                    # 4-byte Folded Spill
	fsw	fs5, 304(sp)                    # 4-byte Folded Spill
	fsw	fs5, 308(sp)                    # 4-byte Folded Spill
	fsw	fs5, 312(sp)                    # 4-byte Folded Spill
	fsw	fs5, 316(sp)                    # 4-byte Folded Spill
	fsw	fs5, 320(sp)                    # 4-byte Folded Spill
	fsw	fs5, 324(sp)                    # 4-byte Folded Spill
	fsw	fs5, 328(sp)                    # 4-byte Folded Spill
	fsw	fs5, 332(sp)                    # 4-byte Folded Spill
	fsw	fs5, 336(sp)                    # 4-byte Folded Spill
	fsw	fs5, 340(sp)                    # 4-byte Folded Spill
	fsw	fs5, 344(sp)                    # 4-byte Folded Spill
	fsw	fs5, 348(sp)                    # 4-byte Folded Spill
	fsw	fs5, 352(sp)                    # 4-byte Folded Spill
	fsw	fs5, 356(sp)                    # 4-byte Folded Spill
	fsw	fs5, 360(sp)                    # 4-byte Folded Spill
	fsw	fs5, 364(sp)                    # 4-byte Folded Spill
	fsw	fs5, 368(sp)                    # 4-byte Folded Spill
	fsw	fs5, 372(sp)                    # 4-byte Folded Spill
	fsw	fs5, 376(sp)                    # 4-byte Folded Spill
	fsw	fs5, 380(sp)                    # 4-byte Folded Spill
	fsw	fs5, 384(sp)                    # 4-byte Folded Spill
	fsw	fs5, 388(sp)                    # 4-byte Folded Spill
	fsw	fs5, 392(sp)                    # 4-byte Folded Spill
	fsw	fs5, 396(sp)                    # 4-byte Folded Spill
	fsw	fs5, 400(sp)                    # 4-byte Folded Spill
	fsw	fs5, 404(sp)                    # 4-byte Folded Spill
	fsw	fs5, 408(sp)                    # 4-byte Folded Spill
	fsw	fs5, 412(sp)                    # 4-byte Folded Spill
	fsw	fs5, 416(sp)                    # 4-byte Folded Spill
	fsw	fs5, 420(sp)                    # 4-byte Folded Spill
	fsw	fs5, 424(sp)                    # 4-byte Folded Spill
	fsw	fs5, 428(sp)                    # 4-byte Folded Spill
	fsw	fs5, 432(sp)                    # 4-byte Folded Spill
	fsw	fs5, 436(sp)                    # 4-byte Folded Spill
	fsw	fs5, 440(sp)                    # 4-byte Folded Spill
	fsw	fs5, 444(sp)                    # 4-byte Folded Spill
	fsw	fs5, 448(sp)                    # 4-byte Folded Spill
	fsw	fs5, 452(sp)                    # 4-byte Folded Spill
	fsw	fs5, 456(sp)                    # 4-byte Folded Spill
	fsw	fs5, 460(sp)                    # 4-byte Folded Spill
	fsw	fs5, 464(sp)                    # 4-byte Folded Spill
	fsw	fs5, 468(sp)                    # 4-byte Folded Spill
	fsw	fs5, 472(sp)                    # 4-byte Folded Spill
	fsw	fs5, 476(sp)                    # 4-byte Folded Spill
	fsw	fs5, 480(sp)                    # 4-byte Folded Spill
	fsw	fs5, 484(sp)                    # 4-byte Folded Spill
	fsw	fs5, 488(sp)                    # 4-byte Folded Spill
	fsw	fs5, 492(sp)                    # 4-byte Folded Spill
	fsw	fs5, 496(sp)                    # 4-byte Folded Spill
	fsw	fs5, 500(sp)                    # 4-byte Folded Spill
	fsw	fs5, 504(sp)                    # 4-byte Folded Spill
	fsw	fs5, 508(sp)                    # 4-byte Folded Spill
	andi	a6, a5, 2
	bnez	a6, .LBB0_3
	j	.LBB0_4
.LBB0_2:
	.loc	1 0 52 is_stmt 0                # k135114294384224.py:0:52
	fmv.w.x	fa0, zero
	fmv.s	fs5, fa0
	fmv.s	fs6, fa0
	fmv.s	fs4, fa0
	fmv.s	fs3, fa0
	fmv.s	fs2, fa0
	fmv.s	fs1, fa0
	fmv.s	fs0, fa0
	fmv.s	fs11, fa0
	fmv.s	fs10, fa0
	fmv.s	fs9, fa0
	fmv.s	fs8, fa0
	fmv.s	fs7, fa0
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
	fsw	fa0, 256(sp)                    # 4-byte Folded Spill
	fsw	fa0, 260(sp)                    # 4-byte Folded Spill
	fsw	fa0, 264(sp)                    # 4-byte Folded Spill
	fsw	fa0, 268(sp)                    # 4-byte Folded Spill
	fsw	fa0, 272(sp)                    # 4-byte Folded Spill
	fsw	fa0, 276(sp)                    # 4-byte Folded Spill
	fsw	fa0, 280(sp)                    # 4-byte Folded Spill
	fsw	fa0, 284(sp)                    # 4-byte Folded Spill
	fsw	fa0, 288(sp)                    # 4-byte Folded Spill
	fsw	fa0, 292(sp)                    # 4-byte Folded Spill
	fsw	fa0, 296(sp)                    # 4-byte Folded Spill
	fsw	fa0, 300(sp)                    # 4-byte Folded Spill
	fsw	fa0, 304(sp)                    # 4-byte Folded Spill
	fsw	fa0, 308(sp)                    # 4-byte Folded Spill
	fsw	fa0, 312(sp)                    # 4-byte Folded Spill
	fsw	fa0, 316(sp)                    # 4-byte Folded Spill
	fsw	fa0, 320(sp)                    # 4-byte Folded Spill
	fsw	fa0, 324(sp)                    # 4-byte Folded Spill
	fsw	fa0, 328(sp)                    # 4-byte Folded Spill
	fsw	fa0, 332(sp)                    # 4-byte Folded Spill
	fsw	fa0, 336(sp)                    # 4-byte Folded Spill
	fsw	fa0, 340(sp)                    # 4-byte Folded Spill
	fsw	fa0, 344(sp)                    # 4-byte Folded Spill
	fsw	fa0, 348(sp)                    # 4-byte Folded Spill
	fsw	fa0, 352(sp)                    # 4-byte Folded Spill
	fsw	fa0, 356(sp)                    # 4-byte Folded Spill
	fsw	fa0, 360(sp)                    # 4-byte Folded Spill
	fsw	fa0, 364(sp)                    # 4-byte Folded Spill
	fsw	fa0, 368(sp)                    # 4-byte Folded Spill
	fsw	fa0, 372(sp)                    # 4-byte Folded Spill
	fsw	fa0, 376(sp)                    # 4-byte Folded Spill
	fsw	fa0, 380(sp)                    # 4-byte Folded Spill
	fsw	fa0, 384(sp)                    # 4-byte Folded Spill
	fsw	fa0, 388(sp)                    # 4-byte Folded Spill
	fsw	fa0, 392(sp)                    # 4-byte Folded Spill
	fsw	fa0, 396(sp)                    # 4-byte Folded Spill
	fsw	fa0, 400(sp)                    # 4-byte Folded Spill
	fsw	fa0, 404(sp)                    # 4-byte Folded Spill
	fsw	fa0, 408(sp)                    # 4-byte Folded Spill
	fsw	fa0, 412(sp)                    # 4-byte Folded Spill
	fsw	fa0, 416(sp)                    # 4-byte Folded Spill
	fsw	fa0, 420(sp)                    # 4-byte Folded Spill
	fsw	fa0, 424(sp)                    # 4-byte Folded Spill
	fsw	fa0, 428(sp)                    # 4-byte Folded Spill
	fsw	fa0, 432(sp)                    # 4-byte Folded Spill
	fsw	fa0, 436(sp)                    # 4-byte Folded Spill
	fsw	fa0, 440(sp)                    # 4-byte Folded Spill
	fsw	fa0, 444(sp)                    # 4-byte Folded Spill
	fsw	fa0, 448(sp)                    # 4-byte Folded Spill
	fsw	fa0, 452(sp)                    # 4-byte Folded Spill
	fsw	fa0, 456(sp)                    # 4-byte Folded Spill
	fsw	fa0, 460(sp)                    # 4-byte Folded Spill
	fsw	fa0, 464(sp)                    # 4-byte Folded Spill
	fsw	fa0, 468(sp)                    # 4-byte Folded Spill
	fsw	fa0, 472(sp)                    # 4-byte Folded Spill
	fsw	fa0, 476(sp)                    # 4-byte Folded Spill
	fsw	fa0, 480(sp)                    # 4-byte Folded Spill
	fsw	fa0, 484(sp)                    # 4-byte Folded Spill
	fsw	fa0, 488(sp)                    # 4-byte Folded Spill
	fsw	fa0, 492(sp)                    # 4-byte Folded Spill
	fsw	fa0, 496(sp)                    # 4-byte Folded Spill
	fsw	fa0, 500(sp)                    # 4-byte Folded Spill
	fsw	fa0, 504(sp)                    # 4-byte Folded Spill
	fsw	fa0, 508(sp)                    # 4-byte Folded Spill
	.loc	1 11 52                         # k135114294384224.py:11:52
	andi	a6, a5, 2
	beqz	a6, .LBB0_4
.LBB0_3:                                # %cond.load1
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	vmv.x.s	a6, v16
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs5, a6
.LBB0_4:                                # %else2
	andi	a6, a5, 4
	bnez	a6, .LBB0_39
# %bb.5:                                # %else5
	andi	a6, a5, 8
	bnez	a6, .LBB0_40
.LBB0_6:                                # %else8
	andi	a7, a5, 16
	lui	a6, 6
	addi	a6, a6, -664
	add	a6, sp, a6
	bnez	a7, .LBB0_41
.LBB0_7:                                # %else11
	andi	a7, a5, 32
	bnez	a7, .LBB0_42
.LBB0_8:                                # %else14
	andi	a7, a5, 64
	bnez	a7, .LBB0_43
.LBB0_9:                                # %else17
	andi	a7, a5, 128
	bnez	a7, .LBB0_44
.LBB0_10:                               # %else20
	andi	a7, a5, 256
	bnez	a7, .LBB0_45
.LBB0_11:                               # %else23
	andi	a7, a5, 512
	bnez	a7, .LBB0_46
.LBB0_12:                               # %else26
	andi	a7, a5, 1024
	bnez	a7, .LBB0_47
.LBB0_13:                               # %else29
	slli	a7, a5, 52
	bltz	a7, .LBB0_48
.LBB0_14:                               # %else32
	slli	a7, a5, 51
	bltz	a7, .LBB0_49
.LBB0_15:                               # %else35
	slli	a7, a5, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v0, 16
	bltz	a7, .LBB0_50
.LBB0_16:                               # %else38
	slli	a7, a5, 49
	lui	a6, 5
	addi	a6, a6, 1320
	add	a6, sp, a6
	bltz	a7, .LBB0_51
.LBB0_17:                               # %else41
	slli	a7, a5, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bltz	a7, .LBB0_52
.LBB0_18:                               # %else44
	slli	a7, a5, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v24, a0
	bltz	a7, .LBB0_53
.LBB0_19:                               # %else47
	slli	a7, a5, 46
	bltz	a7, .LBB0_54
.LBB0_20:                               # %else50
	slli	a7, a5, 45
	bltz	a7, .LBB0_55
.LBB0_21:                               # %else53
	slli	a7, a5, 44
	bgez	a7, .LBB0_23
.LBB0_22:                               # %cond.load55
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 76(sp)                     # 4-byte Folded Spill
.LBB0_23:                               # %else56
	slli	a7, a5, 43
	csrr	t0, vlenb
	li	t1, 88
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	csrr	t0, vlenb
	li	t1, 48
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a7, .LBB0_25
# %bb.24:                               # %cond.load58
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -1152
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 1656(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 80(sp)                     # 4-byte Folded Spill
.LBB0_25:                               # %else59
	slli	a7, a5, 42
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v8, v8, 5
	csrr	t0, vlenb
	li	t1, 72
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t1, 80
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vs8r.v	v24, (t0)                       # vscale x 64-byte Folded Spill
	bgez	a7, .LBB0_27
# %bb.26:                               # %cond.load61
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -1280
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 1536(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 84(sp)                     # 4-byte Folded Spill
.LBB0_27:                               # %else62
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a7, vlenb
	li	t0, 72
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vmulh.vx	v16, v8, a4
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 41
	csrr	t0, vlenb
	li	t1, 88
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vmulh.vx	v24, v8, a4
	bgez	a7, .LBB0_29
# %bb.28:                               # %cond.load64
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -1408
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t1, 80
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1416(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 88(sp)                     # 4-byte Folded Spill
.LBB0_29:                               # %else65
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a7, vlenb
	li	t0, 72
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v16, v8
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 40
	csrr	t0, vlenb
	li	t1, 88
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v0, v24, v8
	bltz	a7, .LBB0_56
# %bb.30:                               # %else68
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v24, v16, 3
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 39
	vsra.vi	v16, v0, 8
	bltz	a7, .LBB0_57
.LBB0_31:                               # %else71
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vsrl.vi	v0, v24, 31
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 38
	vsrl.vi	v8, v16, 31
	bgez	a7, .LBB0_33
.LBB0_32:                               # %cond.load73
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -1792
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t1, 24
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vs8r.v	v0, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t1, 80
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	csrr	a7, vlenb
	li	t0, 24
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	ld	a7, 1056(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 100(sp)                    # 4-byte Folded Spill
.LBB0_33:                               # %else74
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v24, v0
	vadd.vv	v16, v16, v8
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 37
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a3
	bgez	a7, .LBB0_35
# %bb.34:                               # %cond.load76
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -1920
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t1, 80
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 936(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 104(sp)                    # 4-byte Folded Spill
.LBB0_35:                               # %else77
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a7, vlenb
	li	t0, 88
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v8, v0, v8
	csrr	a7, vlenb
	slli	a7, a7, 4
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	csrr	a7, vlenb
	li	t0, 72
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	vnmsub.vx	v24, a2, v8
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 36
	vsll.vi	v8, v16, 6
	bgez	a7, .LBB0_37
# %bb.36:                               # %cond.load79
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a7, 11
	slli	a7, a7, 11
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t1, 80
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 816(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 108(sp)                    # 4-byte Folded Spill
.LBB0_37:                               # %else80
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a7, vlenb
	slli	a7, a7, 4
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v16, (a7)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v8, v16
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 35
	vsll.vi	v8, v24, 7
	bgez	a7, .LBB0_58
# %bb.38:                               # %cond.load82
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 1920
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t1, 80
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 696(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 112(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 34
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v0, v16, v8
	bltz	a7, .LBB0_59
	j	.LBB0_60
.LBB0_39:                               # %cond.load4
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 2
	vmv.x.s	a6, v16
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs6, a6
	andi	a6, a5, 8
	beqz	a6, .LBB0_6
.LBB0_40:                               # %cond.load7
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 3
	vmv.x.s	a6, v16
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs4, a6
	andi	a7, a5, 16
	lui	a6, 6
	addi	a6, a6, -664
	add	a6, sp, a6
	beqz	a7, .LBB0_7
.LBB0_41:                               # %cond.load10
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, 384
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1080(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs3, a7
	andi	a7, a5, 32
	beqz	a7, .LBB0_8
.LBB0_42:                               # %cond.load13
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, 256
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 960(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs2, a7
	andi	a7, a5, 64
	beqz	a7, .LBB0_9
.LBB0_43:                               # %cond.load16
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, 128
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 840(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs1, a7
	andi	a7, a5, 128
	beqz	a7, .LBB0_10
.LBB0_44:                               # %cond.load19
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 720(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs0, a7
	andi	a7, a5, 256
	beqz	a7, .LBB0_11
.LBB0_45:                               # %cond.load22
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -128
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 600(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs11, a7
	andi	a7, a5, 512
	beqz	a7, .LBB0_12
.LBB0_46:                               # %cond.load25
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -256
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 480(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs10, a7
	andi	a7, a5, 1024
	beqz	a7, .LBB0_13
.LBB0_47:                               # %cond.load28
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -384
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 360(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs9, a7
	slli	a7, a5, 52
	bgez	a7, .LBB0_14
.LBB0_48:                               # %cond.load31
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -512
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 240(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs8, a7
	slli	a7, a5, 51
	bgez	a7, .LBB0_15
.LBB0_49:                               # %cond.load34
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -640
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 120(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs7, a7
	slli	a7, a5, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v0, 16
	bgez	a7, .LBB0_16
.LBB0_50:                               # %cond.load37
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -768
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a6, 0(a6)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 52(sp)                     # 4-byte Folded Spill
	slli	a7, a5, 49
	lui	a6, 5
	addi	a6, a6, 1320
	add	a6, sp, a6
	bgez	a7, .LBB0_17
.LBB0_51:                               # %cond.load40
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -896
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1992(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 56(sp)                     # 4-byte Folded Spill
	slli	a7, a5, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bgez	a7, .LBB0_18
.LBB0_52:                               # %cond.load43
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a7, 23
	slli	a7, a7, 10
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vse64.v	v8, (a7)
	ld	a7, 1872(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 60(sp)                     # 4-byte Folded Spill
	slli	a7, a5, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v24, a0
	bgez	a7, .LBB0_19
.LBB0_53:                               # %cond.load46
	vmv.x.s	a7, v24
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 64(sp)                     # 4-byte Folded Spill
	slli	a7, a5, 46
	bgez	a7, .LBB0_20
.LBB0_54:                               # %cond.load49
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 68(sp)                     # 4-byte Folded Spill
	slli	a7, a5, 45
	bgez	a7, .LBB0_21
.LBB0_55:                               # %cond.load52
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 72(sp)                     # 4-byte Folded Spill
	slli	a7, a5, 44
	bltz	a7, .LBB0_22
	j	.LBB0_23
.LBB0_56:                               # %cond.load67
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -1536
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t1, 80
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1296(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 92(sp)                     # 4-byte Folded Spill
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v24, v16, 3
	slli	a7, a5, 39
	vsra.vi	v16, v0, 8
	bgez	a7, .LBB0_31
.LBB0_57:                               # %cond.load70
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 6
	addi	a7, a7, -1664
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t1, 80
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1176(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 96(sp)                     # 4-byte Folded Spill
	vsetvli	zero, a1, e32, m8, ta, ma
	vsrl.vi	v0, v24, 31
	slli	a7, a5, 38
	vsrl.vi	v8, v16, 31
	bltz	a7, .LBB0_32
	j	.LBB0_33
.LBB0_58:
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a7, vlenb
	li	t0, 80
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v24, (a7)                       # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 34
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v0, v16, v8
	bgez	a7, .LBB0_60
.LBB0_59:                               # %cond.load85
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 1792
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 576(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 116(sp)                    # 4-byte Folded Spill
.LBB0_60:                               # %else86
	slli	a7, a5, 33
	bltz	a7, .LBB0_99
# %bb.61:                               # %else89
	slli	a7, a5, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v0, v0
	bltz	a7, .LBB0_100
.LBB0_62:                               # %else92
	slli	a7, a5, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, a0
	bltz	a7, .LBB0_101
.LBB0_63:                               # %else95
	slli	a7, a5, 30
	bltz	a7, .LBB0_102
.LBB0_64:                               # %else98
	slli	a7, a5, 29
	bltz	a7, .LBB0_103
.LBB0_65:                               # %else101
	slli	a7, a5, 28
	bltz	a7, .LBB0_104
.LBB0_66:                               # %else104
	slli	a7, a5, 27
	bltz	a7, .LBB0_105
.LBB0_67:                               # %else107
	slli	a7, a5, 26
	bltz	a7, .LBB0_106
.LBB0_68:                               # %else110
	slli	a7, a5, 25
	lui	a6, 5
	addi	a6, a6, -816
	add	a6, sp, a6
	bltz	a7, .LBB0_107
.LBB0_69:                               # %else113
	slli	a7, a5, 24
	bltz	a7, .LBB0_108
.LBB0_70:                               # %else116
	slli	a7, a5, 23
	bltz	a7, .LBB0_109
.LBB0_71:                               # %else119
	slli	a7, a5, 22
	bltz	a7, .LBB0_110
.LBB0_72:                               # %else122
	slli	a7, a5, 21
	bltz	a7, .LBB0_111
.LBB0_73:                               # %else125
	slli	a7, a5, 20
	bltz	a7, .LBB0_112
.LBB0_74:                               # %else128
	slli	a7, a5, 19
	bltz	a7, .LBB0_113
.LBB0_75:                               # %else131
	slli	a7, a5, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v0, 16
	bltz	a7, .LBB0_114
.LBB0_76:                               # %else134
	slli	a7, a5, 17
	bltz	a7, .LBB0_115
.LBB0_77:                               # %else137
	slli	a7, a5, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bltz	a7, .LBB0_116
.LBB0_78:                               # %else140
	slli	a7, a5, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v24, a0
	bltz	a7, .LBB0_117
.LBB0_79:                               # %else143
	slli	a7, a5, 14
	bltz	a7, .LBB0_118
.LBB0_80:                               # %else146
	slli	a7, a5, 13
	bltz	a7, .LBB0_119
.LBB0_81:                               # %else149
	slli	a7, a5, 12
	bgez	a7, .LBB0_83
.LBB0_82:                               # %cond.load151
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 204(sp)                    # 4-byte Folded Spill
.LBB0_83:                               # %else152
	slli	a7, a5, 11
	csrr	t0, vlenb
	li	t1, 96
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	csrr	t0, vlenb
	li	t1, 48
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a7, .LBB0_85
# %bb.84:                               # %cond.load154
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, -128
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 720(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 208(sp)                    # 4-byte Folded Spill
.LBB0_85:                               # %else155
	slli	a7, a5, 10
	csrr	t0, vlenb
	li	t1, 24
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v8, v8, 5
	csrr	t0, vlenb
	li	t1, 80
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vs8r.v	v24, (t0)                       # vscale x 64-byte Folded Spill
	bgez	a7, .LBB0_87
# %bb.86:                               # %cond.load157
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, -256
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 600(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 212(sp)                    # 4-byte Folded Spill
.LBB0_87:                               # %else158
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a7, vlenb
	li	t0, 80
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vmulh.vx	v16, v8, a4
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 9
	csrr	t0, vlenb
	li	t1, 96
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vmulh.vx	v24, v8, a4
	bgez	a7, .LBB0_89
# %bb.88:                               # %cond.load160
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, -384
	add	a7, sp, a7
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 480(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 216(sp)                    # 4-byte Folded Spill
.LBB0_89:                               # %else161
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a7, vlenb
	li	t0, 80
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v16, v8
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 8
	csrr	t0, vlenb
	li	t1, 96
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v0, v24, v8
	bltz	a7, .LBB0_120
# %bb.90:                               # %else164
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v24, v16, 3
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 7
	vsra.vi	v16, v0, 8
	bltz	a7, .LBB0_121
.LBB0_91:                               # %else167
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vsrl.vi	v0, v24, 31
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 6
	vsrl.vi	v8, v16, 31
	bgez	a7, .LBB0_93
.LBB0_92:                               # %cond.load169
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, -768
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	t0, sp, t0
	vs8r.v	v0, (t0)                        # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	lui	a7, 6
	addi	a7, a7, 624
	add	a7, sp, a7
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	ld	a7, 120(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 228(sp)                    # 4-byte Folded Spill
.LBB0_93:                               # %else170
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v24, v0
	vadd.vv	v16, v16, v8
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a5, 5
	csrr	t0, vlenb
	li	t1, 24
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a3
	bgez	a7, .LBB0_95
# %bb.94:                               # %cond.load172
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, -896
	add	a7, sp, a7
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a6, 0(a6)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 232(sp)                    # 4-byte Folded Spill
.LBB0_95:                               # %else173
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a6, vlenb
	li	a7, 96
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 624
	add	a6, a6, a7
	vl8r.v	v0, (a6)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v8, v0, v8
	csrr	a6, vlenb
	li	a7, 24
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 624
	add	a6, a6, a7
	vs8r.v	v8, (a6)                        # vscale x 64-byte Folded Spill
	csrr	a6, vlenb
	li	a7, 80
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 624
	add	a6, a6, a7
	vl8r.v	v8, (a6)                        # vscale x 64-byte Folded Reload
	vnmsub.vx	v24, a2, v8
	vsll.vi	v16, v16, 6
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a6, a5, 4
	lui	a7, 4
	addi	a7, a7, 1144
	add	a7, sp, a7
	bgez	a6, .LBB0_97
# %bb.96:                               # %cond.load175
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a6, 19
	slli	a6, a6, 10
	add	a6, sp, a6
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a6)
	ld	a6, 2016(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 236(sp)                    # 4-byte Folded Spill
.LBB0_97:                               # %else176
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a6, vlenb
	li	t0, 24
	mul	a6, a6, t0
	add	a6, sp, a6
	lui	t0, 6
	addi	t0, t0, 624
	add	a6, a6, t0
	vl8r.v	v8, (a6)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v16, v8
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a6, a5, 3
	vsll.vi	v8, v24, 7
	bgez	a6, .LBB0_122
# %bb.98:                               # %cond.load178
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a6, 5
	addi	a6, a6, -1152
	add	a6, sp, a6
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1896(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 240(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 2
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v0, v16, v8
	bltz	a6, .LBB0_123
	j	.LBB0_124
.LBB0_99:                               # %cond.load88
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 1664
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 456(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 120(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v0, v0
	bgez	a7, .LBB0_62
.LBB0_100:                              # %cond.load91
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 1536
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vse64.v	v24, (a7)
	ld	a7, 336(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 124(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, a0
	bgez	a7, .LBB0_63
.LBB0_101:                              # %cond.load94
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 128(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 30
	bgez	a7, .LBB0_64
.LBB0_102:                              # %cond.load97
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	vmv.x.s	a7, v16
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 132(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 29
	bgez	a7, .LBB0_65
.LBB0_103:                              # %cond.load100
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 2
	vmv.x.s	a7, v16
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 136(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 28
	bgez	a7, .LBB0_66
.LBB0_104:                              # %cond.load103
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 3
	vmv.x.s	a7, v16
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 140(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 27
	bgez	a7, .LBB0_67
.LBB0_105:                              # %cond.load106
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 1408
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 120(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 144(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 26
	bgez	a7, .LBB0_68
.LBB0_106:                              # %cond.load109
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 1280
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a6, 0(a6)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 148(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 25
	lui	a6, 5
	addi	a6, a6, -816
	add	a6, sp, a6
	bgez	a7, .LBB0_69
.LBB0_107:                              # %cond.load112
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 1152
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 2016(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 152(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 24
	bgez	a7, .LBB0_70
.LBB0_108:                              # %cond.load115
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a7, 21
	slli	a7, a7, 10
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1896(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 156(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 23
	bgez	a7, .LBB0_71
.LBB0_109:                              # %cond.load118
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 896
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1776(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 160(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 22
	bgez	a7, .LBB0_72
.LBB0_110:                              # %cond.load121
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 768
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1656(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 164(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 21
	bgez	a7, .LBB0_73
.LBB0_111:                              # %cond.load124
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 640
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1536(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 168(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 20
	bgez	a7, .LBB0_74
.LBB0_112:                              # %cond.load127
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 512
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1416(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 172(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 19
	bgez	a7, .LBB0_75
.LBB0_113:                              # %cond.load130
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 384
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1296(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 176(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v0, 16
	bgez	a7, .LBB0_76
.LBB0_114:                              # %cond.load133
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 256
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1176(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 180(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 17
	bgez	a7, .LBB0_77
.LBB0_115:                              # %cond.load136
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, 128
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1056(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 184(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bgez	a7, .LBB0_78
.LBB0_116:                              # %cond.load139
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vse64.v	v8, (a7)
	ld	a7, 936(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 188(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v24, a0
	bgez	a7, .LBB0_79
.LBB0_117:                              # %cond.load142
	vmv.x.s	a7, v24
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 192(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 14
	bgez	a7, .LBB0_80
.LBB0_118:                              # %cond.load145
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 196(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 13
	bgez	a7, .LBB0_81
.LBB0_119:                              # %cond.load148
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 200(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 12
	bltz	a7, .LBB0_82
	j	.LBB0_83
.LBB0_120:                              # %cond.load163
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, -512
	add	a7, sp, a7
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 360(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 220(sp)                    # 4-byte Folded Spill
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v24, v16, 3
	slli	a7, a5, 7
	vsra.vi	v16, v0, 8
	bgez	a7, .LBB0_91
.LBB0_121:                              # %cond.load166
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 5
	addi	a7, a7, -640
	add	a7, sp, a7
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 240(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 224(sp)                    # 4-byte Folded Spill
	vsetvli	zero, a1, e32, m8, ta, ma
	vsrl.vi	v0, v24, 31
	slli	a7, a5, 6
	vsrl.vi	v8, v16, 31
	bltz	a7, .LBB0_92
	j	.LBB0_93
.LBB0_122:
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a6, vlenb
	slli	a6, a6, 6
	add	a6, sp, a6
	lui	t0, 6
	addi	t0, t0, 624
	add	a6, a6, t0
	vl8r.v	v24, (a6)                       # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a6, a5, 2
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v0, v16, v8
	bgez	a6, .LBB0_124
.LBB0_123:                              # %cond.load181
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a6, 5
	addi	a6, a6, -1280
	add	a6, sp, a6
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1776(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 244(sp)                    # 4-byte Folded Spill
.LBB0_124:                              # %else182
	slli	a6, a5, 1
	csrr	t0, vlenb
	slli	t0, t0, 5
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl1r.v	v8, (t0)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	bltz	a6, .LBB0_165
# %bb.125:                              # %else185
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v0, v0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	a6, v8
	bltz	a5, .LBB0_166
.LBB0_126:                              # %else188
	andi	a5, a6, 1
	vadd.vx	v8, v16, a0
	bnez	a5, .LBB0_167
.LBB0_127:                              # %else191
	andi	a5, a6, 2
	bnez	a5, .LBB0_168
.LBB0_128:                              # %else194
	andi	a5, a6, 4
	bnez	a5, .LBB0_169
.LBB0_129:                              # %else197
	andi	a5, a6, 8
	bnez	a5, .LBB0_170
.LBB0_130:                              # %else200
	andi	a5, a6, 16
	bnez	a5, .LBB0_171
.LBB0_131:                              # %else203
	andi	a5, a6, 32
	bnez	a5, .LBB0_172
.LBB0_132:                              # %else206
	andi	a5, a6, 64
	bnez	a5, .LBB0_173
.LBB0_133:                              # %else209
	andi	a5, a6, 128
	bnez	a5, .LBB0_174
.LBB0_134:                              # %else212
	andi	a5, a6, 256
	bnez	a5, .LBB0_175
.LBB0_135:                              # %else215
	andi	a5, a6, 512
	bnez	a5, .LBB0_176
.LBB0_136:                              # %else218
	andi	a5, a6, 1024
	bnez	a5, .LBB0_177
.LBB0_137:                              # %else221
	slli	a5, a6, 52
	bltz	a5, .LBB0_178
.LBB0_138:                              # %else224
	slli	a5, a6, 51
	bltz	a5, .LBB0_179
.LBB0_139:                              # %else227
	slli	a5, a6, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v0, 16
	bltz	a5, .LBB0_180
.LBB0_140:                              # %else230
	slli	a5, a6, 49
	bltz	a5, .LBB0_181
.LBB0_141:                              # %else233
	slli	a5, a6, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bltz	a5, .LBB0_182
.LBB0_142:                              # %else236
	slli	a5, a6, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v24, a0
	bltz	a5, .LBB0_183
.LBB0_143:                              # %else239
	slli	a5, a6, 46
	bltz	a5, .LBB0_184
.LBB0_144:                              # %else242
	slli	a5, a6, 45
	bltz	a5, .LBB0_185
.LBB0_145:                              # %else245
	slli	a5, a6, 44
	bgez	a5, .LBB0_147
.LBB0_146:                              # %cond.load247
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a5, v8
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 332(sp)                    # 4-byte Folded Spill
.LBB0_147:                              # %else248
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a5, vlenb
	li	a7, 104
	mul	a5, a5, a7
	add	a5, sp, a5
	lui	a7, 6
	addi	a7, a7, 624
	add	a5, a5, a7
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	csrr	a5, vlenb
	li	a7, 48
	mul	a5, a5, a7
	add	a5, sp, a5
	lui	a7, 6
	addi	a7, a7, 624
	add	a5, a5, a7
	vl8r.v	v16, (a5)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a6, 43
	lui	a5, 4
	addi	a5, a5, -1088
	add	a5, sp, a5
	bgez	a7, .LBB0_149
# %bb.148:                              # %cond.load250
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 4
	addi	a7, a7, 896
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 2016(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 336(sp)                    # 4-byte Folded Spill
.LBB0_149:                              # %else251
	slli	a7, a6, 42
	csrr	t0, vlenb
	slli	t0, t0, 5
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v8, v8, 5
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vs8r.v	v24, (t0)                       # vscale x 64-byte Folded Spill
	bgez	a7, .LBB0_151
# %bb.150:                              # %cond.load253
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a7, 4
	addi	a7, a7, 768
	add	a7, sp, a7
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 1896(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 340(sp)                    # 4-byte Folded Spill
.LBB0_151:                              # %else254
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vmulh.vx	v16, v8, a4
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a7, a6, 41
	csrr	t0, vlenb
	li	t1, 104
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 624
	add	t0, t0, t1
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	vmulh.vx	v24, v24, a4
	bgez	a7, .LBB0_153
# %bb.152:                              # %cond.load256
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a4, 4
	addi	a4, a4, 640
	add	a4, sp, a4
	csrr	a7, vlenb
	slli	a7, a7, 6
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a4)
	ld	a4, 1776(a5)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 344(sp)                    # 4-byte Folded Spill
.LBB0_153:                              # %else257
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v16, v8
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a4, a6, 40
	csrr	a7, vlenb
	li	t0, 104
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	vadd.vv	v0, v24, v0
	bgez	a4, .LBB0_155
# %bb.154:                              # %cond.load259
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a4, 4
	addi	a4, a4, 512
	add	a4, sp, a4
	csrr	a7, vlenb
	slli	a7, a7, 6
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v24, (a7)                       # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 1656(a5)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 348(sp)                    # 4-byte Folded Spill
.LBB0_155:                              # %else260
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v24, v16, 3
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a4, a6, 39
	vsra.vi	v16, v0, 8
	csrr	a7, vlenb
	li	t0, 48
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	bgez	a4, .LBB0_157
# %bb.156:                              # %cond.load262
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a4, 4
	addi	a4, a4, 384
	add	a4, sp, a4
	csrr	a7, vlenb
	slli	a7, a7, 6
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a4)
	ld	a4, 1536(a5)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 352(sp)                    # 4-byte Folded Spill
.LBB0_157:                              # %else263
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vsrl.vi	v0, v24, 31
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a4, a6, 38
	vsrl.vi	v8, v16, 31
	bgez	a4, .LBB0_159
# %bb.158:                              # %cond.load265
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a4, 4
	addi	a4, a4, 256
	add	a4, sp, a4
	lui	a7, 6
	addi	a7, a7, 624
	add	a7, sp, a7
	vs8r.v	v0, (a7)                        # vscale x 64-byte Folded Spill
	csrr	a7, vlenb
	slli	a7, a7, 6
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a4)
	lui	a4, 6
	addi	a4, a4, 624
	add	a4, sp, a4
	vl8r.v	v0, (a4)                        # vscale x 64-byte Folded Reload
	ld	a4, 1416(a5)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 356(sp)                    # 4-byte Folded Spill
.LBB0_159:                              # %else266
	.loc	1 0 52                          # k135114294384224.py:0:52
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v24, v0
	vadd.vv	v16, v16, v8
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a4, a6, 37
	csrr	a7, vlenb
	slli	a7, a7, 5
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 624
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a3
	bgez	a4, .LBB0_161
# %bb.160:                              # %cond.load268
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a3, 4
	addi	a3, a3, 128
	add	a3, sp, a3
	csrr	a4, vlenb
	slli	a4, a4, 6
	add	a4, sp, a4
	lui	a7, 6
	addi	a7, a7, 624
	add	a4, a4, a7
	vl8r.v	v0, (a4)                        # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a3)
	ld	a3, 1296(a5)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 360(sp)                    # 4-byte Folded Spill
.LBB0_161:                              # %else269
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a3, vlenb
	li	a4, 104
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 6
	addi	a4, a4, 624
	add	a3, a3, a4
	vl8r.v	v0, (a3)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v8, v0, v8
	csrr	a3, vlenb
	slli	a3, a3, 5
	add	a3, sp, a3
	lui	a4, 6
	addi	a4, a4, 624
	add	a3, a3, a4
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	csrr	a3, vlenb
	li	a4, 48
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 6
	addi	a4, a4, 624
	add	a3, a3, a4
	vl8r.v	v8, (a3)                        # vscale x 64-byte Folded Reload
	vnmsub.vx	v24, a2, v8
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a2, a6, 36
	vsll.vi	v8, v16, 6
	bgez	a2, .LBB0_163
# %bb.162:                              # %cond.load271
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a2, 4
	add	a2, sp, a2
	csrr	a3, vlenb
	slli	a3, a3, 6
	add	a3, sp, a3
	lui	a4, 6
	addi	a4, a4, 624
	add	a3, a3, a4
	vl8r.v	v16, (a3)                       # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 1176(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 364(sp)                    # 4-byte Folded Spill
.LBB0_163:                              # %else272
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a2, vlenb
	slli	a2, a2, 5
	add	a2, sp, a2
	lui	a3, 6
	addi	a3, a3, 624
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v8, v16
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a2, a6, 35
	vsll.vi	v8, v24, 7
	bgez	a2, .LBB0_186
# %bb.164:                              # %cond.load274
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a2, 4
	addi	a2, a2, -128
	add	a2, sp, a2
	csrr	a3, vlenb
	slli	a3, a3, 6
	add	a3, sp, a3
	lui	a4, 6
	addi	a4, a4, 624
	add	a3, a3, a4
	vl8r.v	v24, (a3)                       # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 1056(a5)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 368(sp)                    # 4-byte Folded Spill
	slli	a2, a6, 34
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v0, v16, v8
	bltz	a2, .LBB0_187
	j	.LBB0_188
.LBB0_165:                              # %cond.load184
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a6, 5
	addi	a6, a6, -1408
	add	a6, sp, a6
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1656(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 248(sp)                    # 4-byte Folded Spill
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v0, v0
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	a6, v8
	bgez	a5, .LBB0_126
.LBB0_166:                              # %cond.load187
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 5
	addi	a5, a5, -1536
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vse64.v	v24, (a5)
	ld	a5, 1536(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 252(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 1
	vadd.vx	v8, v16, a0
	beqz	a5, .LBB0_127
.LBB0_167:                              # %cond.load190
	vmv.x.s	a5, v8
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 256(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 2
	beqz	a5, .LBB0_128
.LBB0_168:                              # %cond.load193
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	vmv.x.s	a5, v16
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 260(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 4
	beqz	a5, .LBB0_129
.LBB0_169:                              # %cond.load196
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 2
	vmv.x.s	a5, v16
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 264(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 8
	beqz	a5, .LBB0_130
.LBB0_170:                              # %cond.load199
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 3
	vmv.x.s	a5, v16
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 268(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 16
	beqz	a5, .LBB0_131
.LBB0_171:                              # %cond.load202
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 5
	addi	a5, a5, -1664
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 1320(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 272(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 32
	beqz	a5, .LBB0_132
.LBB0_172:                              # %cond.load205
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 5
	addi	a5, a5, -1792
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 1200(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 276(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 64
	beqz	a5, .LBB0_133
.LBB0_173:                              # %cond.load208
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 5
	addi	a5, a5, -1920
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 1080(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 280(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 128
	beqz	a5, .LBB0_134
.LBB0_174:                              # %cond.load211
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a5, 9
	slli	a5, a5, 11
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 960(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 284(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 256
	beqz	a5, .LBB0_135
.LBB0_175:                              # %cond.load214
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 4
	addi	a5, a5, 1920
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 840(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 288(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 512
	beqz	a5, .LBB0_136
.LBB0_176:                              # %cond.load217
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 4
	addi	a5, a5, 1792
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 720(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 292(sp)                    # 4-byte Folded Spill
	andi	a5, a6, 1024
	beqz	a5, .LBB0_137
.LBB0_177:                              # %cond.load220
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 4
	addi	a5, a5, 1664
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 600(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 296(sp)                    # 4-byte Folded Spill
	slli	a5, a6, 52
	bgez	a5, .LBB0_138
.LBB0_178:                              # %cond.load223
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 4
	addi	a5, a5, 1536
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 480(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 300(sp)                    # 4-byte Folded Spill
	slli	a5, a6, 51
	bgez	a5, .LBB0_139
.LBB0_179:                              # %cond.load226
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 4
	addi	a5, a5, 1408
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 360(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 304(sp)                    # 4-byte Folded Spill
	slli	a5, a6, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v0, 16
	bgez	a5, .LBB0_140
.LBB0_180:                              # %cond.load229
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 4
	addi	a5, a5, 1280
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 240(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 308(sp)                    # 4-byte Folded Spill
	slli	a5, a6, 49
	bgez	a5, .LBB0_141
.LBB0_181:                              # %cond.load232
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a5, 4
	addi	a5, a5, 1152
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a5)
	ld	a5, 120(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 312(sp)                    # 4-byte Folded Spill
	slli	a5, a6, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bgez	a5, .LBB0_142
.LBB0_182:                              # %cond.load235
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a5, 17
	slli	a5, a5, 10
	add	a5, sp, a5
	.loc	1 11 52                         # k135114294384224.py:11:52
	vse64.v	v8, (a5)
	ld	a5, 0(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 316(sp)                    # 4-byte Folded Spill
	slli	a5, a6, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v24, a0
	bgez	a5, .LBB0_143
.LBB0_183:                              # %cond.load238
	vmv.x.s	a5, v24
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 320(sp)                    # 4-byte Folded Spill
	slli	a5, a6, 46
	bgez	a5, .LBB0_144
.LBB0_184:                              # %cond.load241
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a5, v8
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 324(sp)                    # 4-byte Folded Spill
	slli	a5, a6, 45
	bgez	a5, .LBB0_145
.LBB0_185:                              # %cond.load244
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a5, v8
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 328(sp)                    # 4-byte Folded Spill
	slli	a5, a6, 44
	bltz	a5, .LBB0_146
	j	.LBB0_147
.LBB0_186:
	.loc	1 0 52                          # k135114294384224.py:0:52
	csrr	a2, vlenb
	slli	a2, a2, 6
	add	a2, sp, a2
	lui	a3, 6
	addi	a3, a3, 624
	add	a2, a2, a3
	vl8r.v	v24, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 11 52                         # k135114294384224.py:11:52
	slli	a2, a6, 34
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v0, v16, v8
	bgez	a2, .LBB0_188
.LBB0_187:                              # %cond.load277
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a1, 4
	addi	a1, a1, -256
	add	a1, sp, a1
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 936(a5)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 372(sp)                    # 4-byte Folded Spill
.LBB0_188:                              # %else278
	slli	a1, a6, 33
	bgez	a1, .LBB0_189
	j	.LBB0_384
.LBB0_189:                              # %else281
	slli	a1, a6, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v0, v0
	bgez	a1, .LBB0_190
	j	.LBB0_385
.LBB0_190:                              # %else284
	slli	a1, a6, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, a0
	bgez	a1, .LBB0_191
	j	.LBB0_386
.LBB0_191:                              # %else287
	slli	a1, a6, 30
	bgez	a1, .LBB0_192
	j	.LBB0_387
.LBB0_192:                              # %else290
	slli	a1, a6, 29
	bgez	a1, .LBB0_193
	j	.LBB0_388
.LBB0_193:                              # %else293
	slli	a1, a6, 28
	bgez	a1, .LBB0_194
	j	.LBB0_389
.LBB0_194:                              # %else296
	slli	a1, a6, 27
	bgez	a1, .LBB0_195
	j	.LBB0_390
.LBB0_195:                              # %else299
	slli	a1, a6, 26
	bgez	a1, .LBB0_196
	j	.LBB0_391
.LBB0_196:                              # %else302
	slli	a1, a6, 25
	bgez	a1, .LBB0_197
	j	.LBB0_392
.LBB0_197:                              # %else305
	slli	a1, a6, 24
	bgez	a1, .LBB0_198
	j	.LBB0_393
.LBB0_198:                              # %else308
	slli	a1, a6, 23
	bgez	a1, .LBB0_199
	j	.LBB0_394
.LBB0_199:                              # %else311
	slli	a2, a6, 22
	lui	a1, 3
	addi	a1, a1, 872
	add	a1, sp, a1
	bgez	a2, .LBB0_200
	j	.LBB0_395
.LBB0_200:                              # %else314
	slli	a2, a6, 21
	bgez	a2, .LBB0_201
	j	.LBB0_396
.LBB0_201:                              # %else317
	slli	a2, a6, 20
	bgez	a2, .LBB0_202
	j	.LBB0_397
.LBB0_202:                              # %else320
	slli	a2, a6, 19
	bgez	a2, .LBB0_203
	j	.LBB0_398
.LBB0_203:                              # %else323
	slli	a2, a6, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v0, 16
	bgez	a2, .LBB0_204
	j	.LBB0_399
.LBB0_204:                              # %else326
	slli	a2, a6, 17
	bgez	a2, .LBB0_205
	j	.LBB0_400
.LBB0_205:                              # %else329
	slli	a2, a6, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bgez	a2, .LBB0_206
	j	.LBB0_401
.LBB0_206:                              # %else332
	slli	a2, a6, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, a0
	bgez	a2, .LBB0_207
	j	.LBB0_402
.LBB0_207:                              # %else335
	slli	a0, a6, 14
	bgez	a0, .LBB0_208
	j	.LBB0_403
.LBB0_208:                              # %else338
	slli	a0, a6, 13
	bgez	a0, .LBB0_209
	j	.LBB0_404
.LBB0_209:                              # %else341
	slli	a0, a6, 12
	bgez	a0, .LBB0_210
	j	.LBB0_405
.LBB0_210:                              # %else344
	slli	a0, a6, 11
	bgez	a0, .LBB0_211
	j	.LBB0_406
.LBB0_211:                              # %else347
	slli	a0, a6, 10
	bgez	a0, .LBB0_212
	j	.LBB0_407
.LBB0_212:                              # %else350
	slli	a0, a6, 9
	bgez	a0, .LBB0_213
	j	.LBB0_408
.LBB0_213:                              # %else353
	slli	a0, a6, 8
	bgez	a0, .LBB0_214
	j	.LBB0_409
.LBB0_214:                              # %else356
	slli	a0, a6, 7
	bgez	a0, .LBB0_215
	j	.LBB0_410
.LBB0_215:                              # %else359
	slli	a0, a6, 6
	bgez	a0, .LBB0_216
	j	.LBB0_411
.LBB0_216:                              # %else362
	slli	a0, a6, 5
	bgez	a0, .LBB0_217
	j	.LBB0_412
.LBB0_217:                              # %else365
	slli	a0, a6, 4
	bgez	a0, .LBB0_218
	j	.LBB0_413
.LBB0_218:                              # %else368
	slli	a0, a6, 3
	bgez	a0, .LBB0_219
	j	.LBB0_414
.LBB0_219:                              # %else371
	slli	a0, a6, 2
	bgez	a0, .LBB0_220
	j	.LBB0_415
.LBB0_220:                              # %else374
	slli	a0, a6, 1
	lui	a1, 3
	addi	a1, a1, -1240
	add	s5, sp, a1
	bgez	a0, .LBB0_221
	j	.LBB0_416
.LBB0_221:                              # %else377
	bgez	a6, .LBB0_223
.LBB0_222:                              # %cond.load379
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a0, 25
	slli	a0, a0, 9
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1872(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 508(sp)                    # 4-byte Folded Spill
.LBB0_223:                              # %else380
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	s3, 32
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 12 33 is_stmt 1               # k135114294384224.py:12:33
	vsetvli	zero, s3, e32, m8, ta, ma
	vsll.vi	v8, v8, 6
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 12 30 is_stmt 0               # k135114294384224.py:12:30
	vadd.vv	v0, v8, v16
	li	a0, 896
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 6 21 is_stmt 1                # k135114294384224.py:6:21
	vmslt.vx	v8, v16, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v9, v16, a0
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v10, v16, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v24, v16, a0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	vslideup.vi	v24, v10, 4
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v24, v9, 8
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v8, v0, v0
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs1r.v	v24, (a0)                       # vscale x 8-byte Folded Spill
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	s4, v24
	andi	a0, s4, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_224
	j	.LBB0_417
.LBB0_224:                              # %else384
	andi	a0, s4, 2
	beqz	a0, .LBB0_225
	j	.LBB0_418
.LBB0_225:                              # %else387
	andi	a0, s4, 4
	beqz	a0, .LBB0_226
	j	.LBB0_419
.LBB0_226:                              # %else390
	andi	a0, s4, 8
	beqz	a0, .LBB0_227
	j	.LBB0_420
.LBB0_227:                              # %else393
	andi	a0, s4, 16
	beqz	a0, .LBB0_228
	j	.LBB0_421
.LBB0_228:                              # %else396
	andi	a0, s4, 32
	beqz	a0, .LBB0_229
	j	.LBB0_422
.LBB0_229:                              # %else399
	andi	a0, s4, 64
	beqz	a0, .LBB0_230
	j	.LBB0_423
.LBB0_230:                              # %else402
	andi	a0, s4, 128
	beqz	a0, .LBB0_231
	j	.LBB0_424
.LBB0_231:                              # %else405
	andi	a0, s4, 256
	beqz	a0, .LBB0_232
	j	.LBB0_425
.LBB0_232:                              # %else408
	andi	a0, s4, 512
	beqz	a0, .LBB0_233
	j	.LBB0_426
.LBB0_233:                              # %else411
	andi	a0, s4, 1024
	beqz	a0, .LBB0_234
	j	.LBB0_427
.LBB0_234:                              # %else414
	slli	a0, s4, 52
	bgez	a0, .LBB0_235
	j	.LBB0_428
.LBB0_235:                              # %else417
	slli	a0, s4, 51
	bgez	a0, .LBB0_237
.LBB0_236:                              # %cond.store418
	fmv.s	fa0, fs7
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_237:                              # %else420
	slli	a0, s4, 50
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_239
# %bb.238:                              # %cond.store421
	.loc	1 0 44 is_stmt 0                # k135114294384224.py:0:44
	flw	fa0, 52(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_239:                              # %else423
	slli	a0, s4, 49
	bgez	a0, .LBB0_241
# %bb.240:                              # %cond.store424
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 56(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 456(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_241:                              # %else426
	slli	a0, s4, 48
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_243
# %bb.242:                              # %cond.store427
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 60(sp)                     # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 11
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 336(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_243:                              # %else429
	slli	a0, s4, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s2
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_244
	j	.LBB0_429
.LBB0_244:                              # %else432
	slli	a0, s4, 46
	bgez	a0, .LBB0_245
	j	.LBB0_430
.LBB0_245:                              # %else435
	slli	a0, s4, 45
	bgez	a0, .LBB0_246
	j	.LBB0_431
.LBB0_246:                              # %else438
	slli	a0, s4, 44
	bgez	a0, .LBB0_247
	j	.LBB0_432
.LBB0_247:                              # %else441
	slli	a0, s4, 43
	bgez	a0, .LBB0_248
	j	.LBB0_433
.LBB0_248:                              # %else444
	slli	a0, s4, 42
	bgez	a0, .LBB0_249
	j	.LBB0_434
.LBB0_249:                              # %else447
	slli	a0, s4, 41
	lui	a1, 2
	addi	a1, a1, 720
	add	s5, sp, a1
	bgez	a0, .LBB0_250
	j	.LBB0_435
.LBB0_250:                              # %else450
	slli	a0, s4, 40
	bgez	a0, .LBB0_251
	j	.LBB0_436
.LBB0_251:                              # %else453
	slli	a0, s4, 39
	bgez	a0, .LBB0_252
	j	.LBB0_437
.LBB0_252:                              # %else456
	slli	a0, s4, 38
	bgez	a0, .LBB0_253
	j	.LBB0_438
.LBB0_253:                              # %else459
	slli	a0, s4, 37
	bgez	a0, .LBB0_254
	j	.LBB0_439
.LBB0_254:                              # %else462
	slli	a0, s4, 36
	bgez	a0, .LBB0_256
.LBB0_255:                              # %cond.store463
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 108(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_256:                              # %else465
	slli	a0, s4, 35
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_258
# %bb.257:                              # %cond.store466
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 112(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_258:                              # %else468
	slli	a0, s4, 34
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_260
# %bb.259:                              # %cond.store469
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 116(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_260:                              # %else471
	slli	a0, s4, 33
	bgez	a0, .LBB0_262
# %bb.261:                              # %cond.store472
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 120(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_262:                              # %else474
	slli	a0, s4, 32
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_264
# %bb.263:                              # %cond.store475
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 124(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 19
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_264:                              # %else477
	slli	a0, s4, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s2
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_265
	j	.LBB0_440
.LBB0_265:                              # %else480
	slli	a0, s4, 30
	bgez	a0, .LBB0_266
	j	.LBB0_441
.LBB0_266:                              # %else483
	slli	a0, s4, 29
	bgez	a0, .LBB0_267
	j	.LBB0_442
.LBB0_267:                              # %else486
	slli	a0, s4, 28
	bgez	a0, .LBB0_268
	j	.LBB0_443
.LBB0_268:                              # %else489
	slli	a0, s4, 27
	bgez	a0, .LBB0_269
	j	.LBB0_444
.LBB0_269:                              # %else492
	slli	a0, s4, 26
	bgez	a0, .LBB0_270
	j	.LBB0_445
.LBB0_270:                              # %else495
	slli	a0, s4, 25
	bgez	a0, .LBB0_271
	j	.LBB0_446
.LBB0_271:                              # %else498
	slli	a0, s4, 24
	bgez	a0, .LBB0_272
	j	.LBB0_447
.LBB0_272:                              # %else501
	slli	a0, s4, 23
	bgez	a0, .LBB0_273
	j	.LBB0_448
.LBB0_273:                              # %else504
	slli	a0, s4, 22
	bgez	a0, .LBB0_274
	j	.LBB0_449
.LBB0_274:                              # %else507
	slli	a0, s4, 21
	bgez	a0, .LBB0_275
	j	.LBB0_450
.LBB0_275:                              # %else510
	slli	a0, s4, 20
	lui	a1, 2
	addi	a1, a1, -1416
	add	s6, sp, a1
	bgez	a0, .LBB0_276
	j	.LBB0_451
.LBB0_276:                              # %else513
	slli	a0, s4, 19
	bgez	a0, .LBB0_278
.LBB0_277:                              # %cond.store514
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 176(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_278:                              # %else516
	slli	a0, s4, 18
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_280
# %bb.279:                              # %cond.store517
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 180(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_280:                              # %else519
	slli	a0, s4, 17
	bgez	a0, .LBB0_282
# %bb.281:                              # %cond.store520
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 184(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_282:                              # %else522
	slli	a0, s4, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_284
# %bb.283:                              # %cond.store523
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 188(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_284:                              # %else525
	slli	a0, s4, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s2
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_285
	j	.LBB0_452
.LBB0_285:                              # %else528
	slli	a0, s4, 14
	bgez	a0, .LBB0_286
	j	.LBB0_453
.LBB0_286:                              # %else531
	slli	a0, s4, 13
	bgez	a0, .LBB0_287
	j	.LBB0_454
.LBB0_287:                              # %else534
	slli	a0, s4, 12
	bgez	a0, .LBB0_288
	j	.LBB0_455
.LBB0_288:                              # %else537
	slli	a0, s4, 11
	bgez	a0, .LBB0_289
	j	.LBB0_456
.LBB0_289:                              # %else540
	slli	a0, s4, 10
	bgez	a0, .LBB0_290
	j	.LBB0_457
.LBB0_290:                              # %else543
	slli	a0, s4, 9
	bgez	a0, .LBB0_291
	j	.LBB0_458
.LBB0_291:                              # %else546
	slli	a0, s4, 8
	bgez	a0, .LBB0_292
	j	.LBB0_459
.LBB0_292:                              # %else549
	slli	a0, s4, 7
	bgez	a0, .LBB0_293
	j	.LBB0_460
.LBB0_293:                              # %else552
	slli	a0, s4, 6
	bgez	a0, .LBB0_294
	j	.LBB0_461
.LBB0_294:                              # %else555
	slli	a0, s4, 5
	bgez	a0, .LBB0_295
	j	.LBB0_462
.LBB0_295:                              # %else558
	slli	a0, s4, 4
	bgez	a0, .LBB0_297
.LBB0_296:                              # %cond.store559
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 236(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 7
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_297:                              # %else561
	slli	a0, s4, 3
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_299
# %bb.298:                              # %cond.store562
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 240(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_299:                              # %else564
	slli	a0, s4, 2
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_301
# %bb.300:                              # %cond.store565
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 244(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 27
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_301:                              # %else567
	slli	a0, s4, 1
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl1r.v	v8, (a1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	bgez	a0, .LBB0_303
# %bb.302:                              # %cond.store568
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 248(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs1r.v	v24, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl1r.v	v24, (a0)                       # vscale x 8-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_303:                              # %else570
	.loc	1 0 44                          # k135114294384224.py:0:44
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	.loc	1 12 44                         # k135114294384224.py:12:44
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	s5, v24
	bgez	s4, .LBB0_305
# %bb.304:                              # %cond.store571
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 252(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 13
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_305:                              # %else573
	andi	a0, s5, 1
	vadd.vx	v8, v16, s2
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_306
	j	.LBB0_463
.LBB0_306:                              # %else576
	andi	a0, s5, 2
	beqz	a0, .LBB0_307
	j	.LBB0_464
.LBB0_307:                              # %else579
	andi	a0, s5, 4
	beqz	a0, .LBB0_308
	j	.LBB0_465
.LBB0_308:                              # %else582
	andi	a0, s5, 8
	beqz	a0, .LBB0_309
	j	.LBB0_466
.LBB0_309:                              # %else585
	andi	a0, s5, 16
	lui	a1, 1
	addi	a1, a1, 568
	add	s4, sp, a1
	beqz	a0, .LBB0_310
	j	.LBB0_467
.LBB0_310:                              # %else588
	andi	a0, s5, 32
	beqz	a0, .LBB0_311
	j	.LBB0_468
.LBB0_311:                              # %else591
	andi	a0, s5, 64
	beqz	a0, .LBB0_312
	j	.LBB0_469
.LBB0_312:                              # %else594
	andi	a0, s5, 128
	beqz	a0, .LBB0_313
	j	.LBB0_470
.LBB0_313:                              # %else597
	andi	a0, s5, 256
	beqz	a0, .LBB0_314
	j	.LBB0_471
.LBB0_314:                              # %else600
	andi	a0, s5, 512
	beqz	a0, .LBB0_315
	j	.LBB0_472
.LBB0_315:                              # %else603
	andi	a0, s5, 1024
	beqz	a0, .LBB0_316
	j	.LBB0_473
.LBB0_316:                              # %else606
	slli	a0, s5, 52
	bgez	a0, .LBB0_317
	j	.LBB0_474
.LBB0_317:                              # %else609
	slli	a0, s5, 51
	bgez	a0, .LBB0_319
.LBB0_318:                              # %cond.store610
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 304(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_319:                              # %else612
	slli	a0, s5, 50
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_321
# %bb.320:                              # %cond.store613
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 308(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_321:                              # %else615
	slli	a0, s5, 49
	bgez	a0, .LBB0_323
# %bb.322:                              # %cond.store616
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 312(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_323:                              # %else618
	slli	a0, s5, 48
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_325
# %bb.324:                              # %cond.store619
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 316(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 5
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_325:                              # %else621
	slli	a0, s5, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s2
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_326
	j	.LBB0_475
.LBB0_326:                              # %else624
	slli	a0, s5, 46
	bgez	a0, .LBB0_327
	j	.LBB0_476
.LBB0_327:                              # %else627
	slli	a0, s5, 45
	bgez	a0, .LBB0_328
	j	.LBB0_477
.LBB0_328:                              # %else630
	slli	a0, s5, 44
	bgez	a0, .LBB0_329
	j	.LBB0_478
.LBB0_329:                              # %else633
	slli	a0, s5, 43
	bgez	a0, .LBB0_330
	j	.LBB0_479
.LBB0_330:                              # %else636
	slli	a0, s5, 42
	bgez	a0, .LBB0_331
	j	.LBB0_480
.LBB0_331:                              # %else639
	slli	a0, s5, 41
	bgez	a0, .LBB0_332
	j	.LBB0_481
.LBB0_332:                              # %else642
	slli	a0, s5, 40
	bgez	a0, .LBB0_333
	j	.LBB0_482
.LBB0_333:                              # %else645
	slli	a0, s5, 39
	addi	s4, sp, 2047
	addi	s4, s4, 481
	bgez	a0, .LBB0_334
	j	.LBB0_483
.LBB0_334:                              # %else648
	slli	a0, s5, 38
	bgez	a0, .LBB0_335
	j	.LBB0_484
.LBB0_335:                              # %else651
	slli	a0, s5, 37
	bgez	a0, .LBB0_336
	j	.LBB0_485
.LBB0_336:                              # %else654
	slli	a0, s5, 36
	bgez	a0, .LBB0_338
.LBB0_337:                              # %cond.store655
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 364(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 1
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_338:                              # %else657
	slli	a0, s5, 35
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_340
# %bb.339:                              # %cond.store658
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 368(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 1921
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_340:                              # %else660
	slli	a0, s5, 34
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_342
# %bb.341:                              # %cond.store661
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 372(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1793
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_342:                              # %else663
	slli	a0, s5, 33
	bgez	a0, .LBB0_344
# %bb.343:                              # %cond.store664
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 376(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1665
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_344:                              # %else666
	slli	a0, s5, 32
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_346
# %bb.345:                              # %cond.store667
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 380(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 1537
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_346:                              # %else669
	slli	a0, s5, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s2
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_347
	j	.LBB0_486
.LBB0_347:                              # %else672
	slli	a0, s5, 30
	bgez	a0, .LBB0_348
	j	.LBB0_487
.LBB0_348:                              # %else675
	slli	a0, s5, 29
	bgez	a0, .LBB0_349
	j	.LBB0_488
.LBB0_349:                              # %else678
	slli	a0, s5, 28
	bgez	a0, .LBB0_350
	j	.LBB0_489
.LBB0_350:                              # %else681
	slli	a0, s5, 27
	bgez	a0, .LBB0_351
	j	.LBB0_490
.LBB0_351:                              # %else684
	slli	a0, s5, 26
	bgez	a0, .LBB0_352
	j	.LBB0_491
.LBB0_352:                              # %else687
	slli	a0, s5, 25
	bgez	a0, .LBB0_353
	j	.LBB0_492
.LBB0_353:                              # %else690
	slli	a0, s5, 24
	bgez	a0, .LBB0_354
	j	.LBB0_493
.LBB0_354:                              # %else693
	slli	a0, s5, 23
	bgez	a0, .LBB0_355
	j	.LBB0_494
.LBB0_355:                              # %else696
	slli	a0, s5, 22
	bgez	a0, .LBB0_356
	j	.LBB0_495
.LBB0_356:                              # %else699
	slli	a0, s5, 21
	bgez	a0, .LBB0_357
	j	.LBB0_496
.LBB0_357:                              # %else702
	slli	a0, s5, 20
	bgez	a0, .LBB0_358
	j	.LBB0_497
.LBB0_358:                              # %else705
	slli	a0, s5, 19
	bgez	a0, .LBB0_360
.LBB0_359:                              # %cond.store706
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 432(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 385
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_360:                              # %else708
	slli	a0, s5, 18
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_362
# %bb.361:                              # %cond.store709
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 436(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 257
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1688(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_362:                              # %else711
	slli	a0, s5, 17
	bgez	a0, .LBB0_364
# %bb.363:                              # %cond.store712
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 440(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 129
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1808(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_364:                              # %else714
	slli	a0, s5, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_366
# %bb.365:                              # %cond.store715
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 444(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 624
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1928(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_366:                              # %else717
	slli	a0, s5, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s2
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_367
	j	.LBB0_498
.LBB0_367:                              # %else720
	slli	a0, s5, 14
	bgez	a0, .LBB0_368
	j	.LBB0_499
.LBB0_368:                              # %else723
	slli	a0, s5, 13
	bgez	a0, .LBB0_369
	j	.LBB0_500
.LBB0_369:                              # %else726
	slli	a0, s5, 12
	bgez	a0, .LBB0_370
	j	.LBB0_501
.LBB0_370:                              # %else729
	slli	a0, s5, 11
	bgez	a0, .LBB0_371
	j	.LBB0_502
.LBB0_371:                              # %else732
	slli	a0, s5, 10
	bgez	a0, .LBB0_372
	j	.LBB0_503
.LBB0_372:                              # %else735
	slli	a0, s5, 9
	bgez	a0, .LBB0_373
	j	.LBB0_504
.LBB0_373:                              # %else738
	slli	a0, s5, 8
	bgez	a0, .LBB0_374
	j	.LBB0_505
.LBB0_374:                              # %else741
	slli	a0, s5, 7
	bgez	a0, .LBB0_375
	j	.LBB0_506
.LBB0_375:                              # %else744
	slli	a0, s5, 6
	bgez	a0, .LBB0_376
	j	.LBB0_507
.LBB0_376:                              # %else747
	slli	a0, s5, 5
	bgez	a0, .LBB0_377
	j	.LBB0_508
.LBB0_377:                              # %else750
	slli	a0, s5, 4
	bgez	a0, .LBB0_378
	j	.LBB0_509
.LBB0_378:                              # %else753
	slli	a0, s5, 3
	bgez	a0, .LBB0_379
	j	.LBB0_510
.LBB0_379:                              # %else756
	slli	a0, s5, 2
	bgez	a0, .LBB0_380
	j	.LBB0_511
.LBB0_380:                              # %else759
	slli	a0, s5, 1
	bgez	a0, .LBB0_381
	j	.LBB0_512
.LBB0_381:                              # %else762
	bgez	s5, .LBB0_383
.LBB0_382:                              # %cond.store763
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 508(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 512
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 632(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_383:                              # %else765
	.loc	1 12 4 epilogue_begin           # k135114294384224.py:12:4
	addi	sp, s0, -2032
	.cfi_def_cfa sp, 2032
	ld	ra, 2024(sp)                    # 8-byte Folded Reload
	ld	s0, 2016(sp)                    # 8-byte Folded Reload
	ld	s2, 2008(sp)                    # 8-byte Folded Reload
	ld	s3, 2000(sp)                    # 8-byte Folded Reload
	ld	s4, 1992(sp)                    # 8-byte Folded Reload
	ld	s5, 1984(sp)                    # 8-byte Folded Reload
	ld	s6, 1976(sp)                    # 8-byte Folded Reload
	fld	fs0, 1968(sp)                   # 8-byte Folded Reload
	fld	fs1, 1960(sp)                   # 8-byte Folded Reload
	fld	fs2, 1952(sp)                   # 8-byte Folded Reload
	fld	fs3, 1944(sp)                   # 8-byte Folded Reload
	fld	fs4, 1936(sp)                   # 8-byte Folded Reload
	fld	fs5, 1928(sp)                   # 8-byte Folded Reload
	fld	fs6, 1920(sp)                   # 8-byte Folded Reload
	fld	fs7, 1912(sp)                   # 8-byte Folded Reload
	fld	fs8, 1904(sp)                   # 8-byte Folded Reload
	fld	fs9, 1896(sp)                   # 8-byte Folded Reload
	fld	fs10, 1888(sp)                  # 8-byte Folded Reload
	fld	fs11, 1880(sp)                  # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
	.cfi_restore s6
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
.LBB0_384:                              # %cond.load280
	.cfi_restore_state
	.loc	1 0 4                           # k135114294384224.py:0:4
	lui	a1, 4
	addi	a1, a1, -384
	add	a1, sp, a1
	.loc	1 11 52 is_stmt 1               # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 816(a5)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 376(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v0, v0
	bltz	a1, .LBB0_385
	j	.LBB0_190
.LBB0_385:                              # %cond.load283
	.loc	1 0 52 is_stmt 0                # k135114294384224.py:0:52
	li	a1, 31
	slli	a1, a1, 9
	add	a1, sp, a1
	.loc	1 11 52                         # k135114294384224.py:11:52
	vse64.v	v24, (a1)
	ld	a1, 696(a5)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 380(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, a0
	bltz	a1, .LBB0_386
	j	.LBB0_191
.LBB0_386:                              # %cond.load286
	vmv.x.s	a1, v8
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 384(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 30
	bltz	a1, .LBB0_387
	j	.LBB0_192
.LBB0_387:                              # %cond.load289
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	vmv.x.s	a1, v16
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 388(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 29
	bltz	a1, .LBB0_388
	j	.LBB0_193
.LBB0_388:                              # %cond.load292
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 2
	vmv.x.s	a1, v16
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 392(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 28
	bltz	a1, .LBB0_389
	j	.LBB0_194
.LBB0_389:                              # %cond.load295
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 3
	vmv.x.s	a1, v16
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 396(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 27
	bltz	a1, .LBB0_390
	j	.LBB0_195
.LBB0_390:                              # %cond.load298
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a1, 4
	addi	a1, a1, -640
	add	a1, sp, a1
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 480(a5)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 400(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 26
	bltz	a1, .LBB0_391
	j	.LBB0_196
.LBB0_391:                              # %cond.load301
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a1, 4
	addi	a1, a1, -768
	add	a1, sp, a1
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 360(a5)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 404(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 25
	bltz	a1, .LBB0_392
	j	.LBB0_197
.LBB0_392:                              # %cond.load304
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a1, 4
	addi	a1, a1, -896
	add	a1, sp, a1
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 240(a5)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 408(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 24
	bltz	a1, .LBB0_393
	j	.LBB0_198
.LBB0_393:                              # %cond.load307
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a1, 15
	slli	a1, a1, 10
	add	a1, sp, a1
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 120(a5)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 412(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 23
	bltz	a1, .LBB0_394
	j	.LBB0_199
.LBB0_394:                              # %cond.load310
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a1, 4
	addi	a1, a1, -1152
	add	a1, sp, a1
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 0(a5)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 416(sp)                    # 4-byte Folded Spill
	slli	a2, a6, 22
	lui	a1, 3
	addi	a1, a1, 872
	add	a1, sp, a1
	bltz	a2, .LBB0_395
	j	.LBB0_200
.LBB0_395:                              # %cond.load313
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a2, 4
	addi	a2, a2, -1280
	add	a2, sp, a2
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a2)
	ld	a2, 2016(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 420(sp)                    # 4-byte Folded Spill
	slli	a2, a6, 21
	bltz	a2, .LBB0_396
	j	.LBB0_201
.LBB0_396:                              # %cond.load316
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a2, 4
	addi	a2, a2, -1408
	add	a2, sp, a2
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a2)
	ld	a2, 1896(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 424(sp)                    # 4-byte Folded Spill
	slli	a2, a6, 20
	bltz	a2, .LBB0_397
	j	.LBB0_202
.LBB0_397:                              # %cond.load319
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a2, 29
	slli	a2, a2, 9
	add	a2, sp, a2
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a2)
	ld	a2, 1776(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 428(sp)                    # 4-byte Folded Spill
	slli	a2, a6, 19
	bltz	a2, .LBB0_398
	j	.LBB0_203
.LBB0_398:                              # %cond.load322
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a2, 4
	addi	a2, a2, -1664
	add	a2, sp, a2
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a2)
	ld	a2, 1656(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 432(sp)                    # 4-byte Folded Spill
	slli	a2, a6, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v0, 16
	bltz	a2, .LBB0_399
	j	.LBB0_204
.LBB0_399:                              # %cond.load325
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a2, 4
	addi	a2, a2, -1792
	add	a2, sp, a2
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a2)
	ld	a2, 1536(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 436(sp)                    # 4-byte Folded Spill
	slli	a2, a6, 17
	bltz	a2, .LBB0_400
	j	.LBB0_205
.LBB0_400:                              # %cond.load328
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a2, 4
	addi	a2, a2, -1920
	add	a2, sp, a2
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a2)
	ld	a2, 1416(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 440(sp)                    # 4-byte Folded Spill
	slli	a2, a6, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bltz	a2, .LBB0_401
	j	.LBB0_206
.LBB0_401:                              # %cond.load331
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a2, 7
	slli	a2, a2, 11
	add	a2, sp, a2
	.loc	1 11 52                         # k135114294384224.py:11:52
	vse64.v	v8, (a2)
	ld	a2, 1296(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 444(sp)                    # 4-byte Folded Spill
	slli	a2, a6, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, a0
	bltz	a2, .LBB0_402
	j	.LBB0_207
.LBB0_402:                              # %cond.load334
	vmv.x.s	a0, v8
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 448(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 14
	bltz	a0, .LBB0_403
	j	.LBB0_208
.LBB0_403:                              # %cond.load337
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 452(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 13
	bltz	a0, .LBB0_404
	j	.LBB0_209
.LBB0_404:                              # %cond.load340
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 2
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 456(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 12
	bltz	a0, .LBB0_405
	j	.LBB0_210
.LBB0_405:                              # %cond.load343
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 3
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 460(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 11
	bltz	a0, .LBB0_406
	j	.LBB0_211
.LBB0_406:                              # %cond.load346
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a0, 3
	addi	a0, a0, 1920
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 464(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 10
	bltz	a0, .LBB0_407
	j	.LBB0_212
.LBB0_407:                              # %cond.load349
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a0, 3
	addi	a0, a0, 1792
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 468(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 9
	bltz	a0, .LBB0_408
	j	.LBB0_213
.LBB0_408:                              # %cond.load352
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a0, 3
	addi	a0, a0, 1664
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 472(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 8
	bltz	a0, .LBB0_409
	j	.LBB0_214
.LBB0_409:                              # %cond.load355
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a0, 27
	slli	a0, a0, 9
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 476(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 7
	bltz	a0, .LBB0_410
	j	.LBB0_215
.LBB0_410:                              # %cond.load358
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a0, 3
	addi	a0, a0, 1408
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 480(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 6
	bltz	a0, .LBB0_411
	j	.LBB0_216
.LBB0_411:                              # %cond.load361
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a0, 3
	addi	a0, a0, 1280
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 484(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 5
	bltz	a0, .LBB0_412
	j	.LBB0_217
.LBB0_412:                              # %cond.load364
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a0, 3
	addi	a0, a0, 1152
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 488(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 4
	bltz	a0, .LBB0_413
	j	.LBB0_218
.LBB0_413:                              # %cond.load367
	.loc	1 0 52                          # k135114294384224.py:0:52
	li	a0, 13
	slli	a0, a0, 10
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 492(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 3
	bltz	a0, .LBB0_414
	j	.LBB0_219
.LBB0_414:                              # %cond.load370
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a0, 3
	addi	a0, a0, 896
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 496(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 2
	bltz	a0, .LBB0_415
	j	.LBB0_220
.LBB0_415:                              # %cond.load373
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a0, 3
	addi	a0, a0, 768
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 500(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 1
	lui	a1, 3
	addi	a1, a1, -1240
	add	s5, sp, a1
	bltz	a0, .LBB0_416
	j	.LBB0_221
.LBB0_416:                              # %cond.load376
	.loc	1 0 52                          # k135114294384224.py:0:52
	lui	a0, 3
	addi	a0, a0, 640
	add	a0, sp, a0
	.loc	1 11 52                         # k135114294384224.py:11:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1992(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 504(sp)                    # 4-byte Folded Spill
	bgez	a6, .LBB0_513
	j	.LBB0_222
.LBB0_513:                              # %cond.load376
	j	.LBB0_223
.LBB0_417:                              # %cond.store
	.loc	1 12 44 is_stmt 1               # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 2
	bnez	a0, .LBB0_418
	j	.LBB0_225
.LBB0_418:                              # %cond.store385
	fmv.s	fa0, fs5
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 4
	bnez	a0, .LBB0_419
	j	.LBB0_226
.LBB0_419:                              # %cond.store388
	fmv.s	fa0, fs6
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 8
	bnez	a0, .LBB0_420
	j	.LBB0_227
.LBB0_420:                              # %cond.store391
	fmv.s	fa0, fs4
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 16
	bnez	a0, .LBB0_421
	j	.LBB0_228
.LBB0_421:                              # %cond.store394
	fmv.s	fa0, fs3
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 32
	bnez	a0, .LBB0_422
	j	.LBB0_229
.LBB0_422:                              # %cond.store397
	fmv.s	fa0, fs2
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 64
	bnez	a0, .LBB0_423
	j	.LBB0_230
.LBB0_423:                              # %cond.store400
	fmv.s	fa0, fs1
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 128
	bnez	a0, .LBB0_424
	j	.LBB0_231
.LBB0_424:                              # %cond.store403
	fmv.s	fa0, fs0
	call	__truncsfbf2
	lui	a0, 3
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 256
	bnez	a0, .LBB0_425
	j	.LBB0_232
.LBB0_425:                              # %cond.store406
	fmv.s	fa0, fs11
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 512
	bnez	a0, .LBB0_426
	j	.LBB0_233
.LBB0_426:                              # %cond.store409
	fmv.s	fa0, fs10
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 1024
	bnez	a0, .LBB0_427
	j	.LBB0_234
.LBB0_427:                              # %cond.store412
	fmv.s	fa0, fs9
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 52
	bltz	a0, .LBB0_428
	j	.LBB0_235
.LBB0_428:                              # %cond.store415
	fmv.s	fa0, fs8
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 51
	bgez	a0, .LBB0_514
	j	.LBB0_236
.LBB0_514:                              # %cond.store415
	j	.LBB0_237
.LBB0_429:                              # %cond.store430
	.loc	1 0 44 is_stmt 0                # k135114294384224.py:0:44
	flw	fa0, 64(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 46
	bltz	a0, .LBB0_430
	j	.LBB0_245
.LBB0_430:                              # %cond.store433
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 68(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 45
	bltz	a0, .LBB0_431
	j	.LBB0_246
.LBB0_431:                              # %cond.store436
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 72(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 44
	bltz	a0, .LBB0_432
	j	.LBB0_247
.LBB0_432:                              # %cond.store439
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 76(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 43
	bltz	a0, .LBB0_433
	j	.LBB0_248
.LBB0_433:                              # %cond.store442
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 80(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 42
	bltz	a0, .LBB0_434
	j	.LBB0_249
.LBB0_434:                              # %cond.store445
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 84(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 41
	lui	a1, 2
	addi	a1, a1, 720
	add	s5, sp, a1
	bltz	a0, .LBB0_435
	j	.LBB0_250
.LBB0_435:                              # %cond.store448
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 88(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 40
	bltz	a0, .LBB0_436
	j	.LBB0_251
.LBB0_436:                              # %cond.store451
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 92(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 39
	bltz	a0, .LBB0_437
	j	.LBB0_252
.LBB0_437:                              # %cond.store454
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 96(sp)                     # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 38
	bltz	a0, .LBB0_438
	j	.LBB0_253
.LBB0_438:                              # %cond.store457
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 100(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 37
	bltz	a0, .LBB0_439
	j	.LBB0_254
.LBB0_439:                              # %cond.store460
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 104(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 36
	bgez	a0, .LBB0_515
	j	.LBB0_255
.LBB0_515:                              # %cond.store460
	j	.LBB0_256
.LBB0_440:                              # %cond.store478
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 128(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 30
	bltz	a0, .LBB0_441
	j	.LBB0_266
.LBB0_441:                              # %cond.store481
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 132(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 29
	bltz	a0, .LBB0_442
	j	.LBB0_267
.LBB0_442:                              # %cond.store484
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 136(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 28
	bltz	a0, .LBB0_443
	j	.LBB0_268
.LBB0_443:                              # %cond.store487
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 140(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 27
	bltz	a0, .LBB0_444
	j	.LBB0_269
.LBB0_444:                              # %cond.store490
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 144(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 26
	bltz	a0, .LBB0_445
	j	.LBB0_270
.LBB0_445:                              # %cond.store493
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 148(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 25
	bltz	a0, .LBB0_446
	j	.LBB0_271
.LBB0_446:                              # %cond.store496
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 152(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 24
	bltz	a0, .LBB0_447
	j	.LBB0_272
.LBB0_447:                              # %cond.store499
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 156(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 23
	bltz	a0, .LBB0_448
	j	.LBB0_273
.LBB0_448:                              # %cond.store502
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 160(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 22
	bltz	a0, .LBB0_449
	j	.LBB0_274
.LBB0_449:                              # %cond.store505
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 164(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 21
	bltz	a0, .LBB0_450
	j	.LBB0_275
.LBB0_450:                              # %cond.store508
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 168(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 20
	lui	a1, 2
	addi	a1, a1, -1416
	add	s6, sp, a1
	bltz	a0, .LBB0_451
	j	.LBB0_276
.LBB0_451:                              # %cond.store511
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 172(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 19
	bgez	a0, .LBB0_516
	j	.LBB0_277
.LBB0_516:                              # %cond.store511
	j	.LBB0_278
.LBB0_452:                              # %cond.store526
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 192(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 14
	bltz	a0, .LBB0_453
	j	.LBB0_286
.LBB0_453:                              # %cond.store529
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 196(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 13
	bltz	a0, .LBB0_454
	j	.LBB0_287
.LBB0_454:                              # %cond.store532
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 200(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 12
	bltz	a0, .LBB0_455
	j	.LBB0_288
.LBB0_455:                              # %cond.store535
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 204(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 11
	bltz	a0, .LBB0_456
	j	.LBB0_289
.LBB0_456:                              # %cond.store538
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 208(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1320(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 10
	bltz	a0, .LBB0_457
	j	.LBB0_290
.LBB0_457:                              # %cond.store541
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 212(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 31
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1200(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 9
	bltz	a0, .LBB0_458
	j	.LBB0_291
.LBB0_458:                              # %cond.store544
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 216(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 8
	bltz	a0, .LBB0_459
	j	.LBB0_292
.LBB0_459:                              # %cond.store547
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 220(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 15
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 7
	bltz	a0, .LBB0_460
	j	.LBB0_293
.LBB0_460:                              # %cond.store550
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 224(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 6
	bltz	a0, .LBB0_461
	j	.LBB0_294
.LBB0_461:                              # %cond.store553
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 228(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 29
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 5
	bltz	a0, .LBB0_462
	j	.LBB0_295
.LBB0_462:                              # %cond.store556
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 232(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 4
	bgez	a0, .LBB0_517
	j	.LBB0_296
.LBB0_517:                              # %cond.store556
	j	.LBB0_297
.LBB0_463:                              # %cond.store574
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 256(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 2
	bnez	a0, .LBB0_464
	j	.LBB0_307
.LBB0_464:                              # %cond.store577
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 260(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 4
	bnez	a0, .LBB0_465
	j	.LBB0_308
.LBB0_465:                              # %cond.store580
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 264(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 8
	bnez	a0, .LBB0_466
	j	.LBB0_309
.LBB0_466:                              # %cond.store583
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 268(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 16
	lui	a1, 1
	addi	a1, a1, 568
	add	s4, sp, a1
	bnez	a0, .LBB0_467
	j	.LBB0_310
.LBB0_467:                              # %cond.store586
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 272(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 32
	bnez	a0, .LBB0_468
	j	.LBB0_311
.LBB0_468:                              # %cond.store589
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 276(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 64
	bnez	a0, .LBB0_469
	j	.LBB0_312
.LBB0_469:                              # %cond.store592
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 280(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 128
	bnez	a0, .LBB0_470
	j	.LBB0_313
.LBB0_470:                              # %cond.store595
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 284(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 3
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 256
	bnez	a0, .LBB0_471
	j	.LBB0_314
.LBB0_471:                              # %cond.store598
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 288(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 512
	bnez	a0, .LBB0_472
	j	.LBB0_315
.LBB0_472:                              # %cond.store601
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 292(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 1024
	bnez	a0, .LBB0_473
	j	.LBB0_316
.LBB0_473:                              # %cond.store604
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 296(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 52
	bltz	a0, .LBB0_474
	j	.LBB0_317
.LBB0_474:                              # %cond.store607
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 300(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 51
	bgez	a0, .LBB0_518
	j	.LBB0_318
.LBB0_518:                              # %cond.store607
	j	.LBB0_319
.LBB0_475:                              # %cond.store622
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 320(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 46
	bltz	a0, .LBB0_476
	j	.LBB0_327
.LBB0_476:                              # %cond.store625
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 324(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 45
	bltz	a0, .LBB0_477
	j	.LBB0_328
.LBB0_477:                              # %cond.store628
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 328(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 44
	bltz	a0, .LBB0_478
	j	.LBB0_329
.LBB0_478:                              # %cond.store631
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 332(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 43
	bltz	a0, .LBB0_479
	j	.LBB0_330
.LBB0_479:                              # %cond.store634
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 336(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 42
	bltz	a0, .LBB0_480
	j	.LBB0_331
.LBB0_480:                              # %cond.store637
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 340(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 41
	bltz	a0, .LBB0_481
	j	.LBB0_332
.LBB0_481:                              # %cond.store640
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 344(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 40
	bltz	a0, .LBB0_482
	j	.LBB0_333
.LBB0_482:                              # %cond.store643
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 348(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 39
	addi	s4, sp, 2047
	addi	s4, s4, 481
	bltz	a0, .LBB0_483
	j	.LBB0_334
.LBB0_483:                              # %cond.store646
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 352(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 38
	bltz	a0, .LBB0_484
	j	.LBB0_335
.LBB0_484:                              # %cond.store649
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 356(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 37
	bltz	a0, .LBB0_485
	j	.LBB0_336
.LBB0_485:                              # %cond.store652
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 360(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 36
	bgez	a0, .LBB0_519
	j	.LBB0_337
.LBB0_519:                              # %cond.store652
	j	.LBB0_338
.LBB0_486:                              # %cond.store670
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 384(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 30
	bltz	a0, .LBB0_487
	j	.LBB0_348
.LBB0_487:                              # %cond.store673
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 388(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 29
	bltz	a0, .LBB0_488
	j	.LBB0_349
.LBB0_488:                              # %cond.store676
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 392(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 28
	bltz	a0, .LBB0_489
	j	.LBB0_350
.LBB0_489:                              # %cond.store679
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 396(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 27
	bltz	a0, .LBB0_490
	j	.LBB0_351
.LBB0_490:                              # %cond.store682
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 400(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1409
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 26
	bltz	a0, .LBB0_491
	j	.LBB0_352
.LBB0_491:                              # %cond.store685
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 404(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1281
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 25
	bltz	a0, .LBB0_492
	j	.LBB0_353
.LBB0_492:                              # %cond.store688
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 408(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1153
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 24
	bltz	a0, .LBB0_493
	j	.LBB0_354
.LBB0_493:                              # %cond.store691
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 412(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1025
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 23
	bltz	a0, .LBB0_494
	j	.LBB0_355
.LBB0_494:                              # %cond.store694
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 416(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 897
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 22
	bltz	a0, .LBB0_495
	j	.LBB0_356
.LBB0_495:                              # %cond.store697
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 420(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 769
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 21
	bltz	a0, .LBB0_496
	j	.LBB0_357
.LBB0_496:                              # %cond.store700
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 424(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 641
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 20
	bltz	a0, .LBB0_497
	j	.LBB0_358
.LBB0_497:                              # %cond.store703
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 428(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 513
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 19
	bgez	a0, .LBB0_520
	j	.LBB0_359
.LBB0_520:                              # %cond.store703
	j	.LBB0_360
.LBB0_498:                              # %cond.store718
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 448(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 14
	bltz	a0, .LBB0_499
	j	.LBB0_368
.LBB0_499:                              # %cond.store721
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 452(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 13
	bltz	a0, .LBB0_500
	j	.LBB0_369
.LBB0_500:                              # %cond.store724
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 456(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 12
	bltz	a0, .LBB0_501
	j	.LBB0_370
.LBB0_501:                              # %cond.store727
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 460(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 11
	bltz	a0, .LBB0_502
	j	.LBB0_371
.LBB0_502:                              # %cond.store730
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 464(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 1920
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1952(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 10
	bltz	a0, .LBB0_503
	j	.LBB0_372
.LBB0_503:                              # %cond.store733
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 468(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 1792
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1832(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 9
	bltz	a0, .LBB0_504
	j	.LBB0_373
.LBB0_504:                              # %cond.store736
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 472(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 1664
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1712(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 8
	bltz	a0, .LBB0_505
	j	.LBB0_374
.LBB0_505:                              # %cond.store739
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 476(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 1536
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1592(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 7
	bltz	a0, .LBB0_506
	j	.LBB0_375
.LBB0_506:                              # %cond.store742
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 480(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 1408
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1472(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 6
	bltz	a0, .LBB0_507
	j	.LBB0_376
.LBB0_507:                              # %cond.store745
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 484(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 1280
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1352(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 5
	bltz	a0, .LBB0_508
	j	.LBB0_377
.LBB0_508:                              # %cond.store748
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 488(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 1152
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1232(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 4
	bltz	a0, .LBB0_509
	j	.LBB0_378
.LBB0_509:                              # %cond.store751
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 492(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 1024
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1112(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 3
	bltz	a0, .LBB0_510
	j	.LBB0_379
.LBB0_510:                              # %cond.store754
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 496(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 896
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 992(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 2
	bltz	a0, .LBB0_511
	j	.LBB0_380
.LBB0_511:                              # %cond.store757
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 500(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 768
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 872(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 1
	bltz	a0, .LBB0_512
	j	.LBB0_381
.LBB0_512:                              # %cond.store760
	.loc	1 0 44                          # k135114294384224.py:0:44
	flw	fa0, 504(sp)                    # 4-byte Folded Reload
	.loc	1 12 44                         # k135114294384224.py:12:44
	call	__truncsfbf2
	addi	a0, sp, 640
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 624
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 752(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	bgez	s5, .LBB0_521
	j	.LBB0_382
.LBB0_521:                              # %cond.store760
	j	.LBB0_383
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
	.asciz	"k135114294384224.py"           # string offset=7 ; k135114294384224.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
