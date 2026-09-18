	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused_cat_neg_slice_transpose_view_2 # -- Begin function triton_poi_fused_cat_neg_slice_transpose_view_2
	.p2align	2
	.type	triton_poi_fused_cat_neg_slice_transpose_view_2,@function
triton_poi_fused_cat_neg_slice_transpose_view_2: # @triton_poi_fused_cat_neg_slice_transpose_view_2
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114449651120.py"
	.loc	1 2 0                           # k135114449651120.py:2:0
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
	fsd	fs0, 1960(sp)                   # 8-byte Folded Spill
	fsd	fs1, 1952(sp)                   # 8-byte Folded Spill
	fsd	fs2, 1944(sp)                   # 8-byte Folded Spill
	fsd	fs3, 1936(sp)                   # 8-byte Folded Spill
	fsd	fs4, 1928(sp)                   # 8-byte Folded Spill
	fsd	fs5, 1920(sp)                   # 8-byte Folded Spill
	fsd	fs6, 1912(sp)                   # 8-byte Folded Spill
	fsd	fs7, 1904(sp)                   # 8-byte Folded Spill
	fsd	fs8, 1896(sp)                   # 8-byte Folded Spill
	fsd	fs9, 1888(sp)                   # 8-byte Folded Spill
	fsd	fs10, 1880(sp)                  # 8-byte Folded Spill
	fsd	fs11, 1872(sp)                  # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s2, -24
	.cfi_offset s3, -32
	.cfi_offset s4, -40
	.cfi_offset s5, -48
	.cfi_offset s6, -56
	.cfi_offset s7, -64
	.cfi_offset fs0, -72
	.cfi_offset fs1, -80
	.cfi_offset fs2, -88
	.cfi_offset fs3, -96
	.cfi_offset fs4, -104
	.cfi_offset fs5, -112
	.cfi_offset fs6, -120
	.cfi_offset fs7, -128
	.cfi_offset fs8, -136
	.cfi_offset fs9, -144
	.cfi_offset fs10, -152
	.cfi_offset fs11, -160
	addi	s0, sp, 2032
	.cfi_def_cfa s0, 0
	.cfi_remember_state
	lui	a2, 12
	addi	a2, a2, -752
	sub	sp, sp, a2
	csrr	a2, vlenb
	li	a4, 200
	mul	a2, a2, a4
	sub	sp, sp, a2
	andi	sp, sp, -128
	mv	s2, a1
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114449651120.py:4:33
	slli	a5, a3, 8
	li	a1, 32
	li	a3, -32
	li	a2, -64
	.loc	1 5 23                          # k135114449651120.py:5:23
	vsetvli	zero, a1, e32, m8, ta, ma
	vmv.v.x	v8, a5
	vid.v	v0
	vor.vx	v24, v0, a5
	.loc	1 8 19                          # k135114449651120.py:8:19
	vsra.vi	v8, v8, 31
	vsrl.vi	v8, v8, 27
	csrr	a4, vlenb
	li	a6, 192
	mul	a4, a4, a6
	add	a4, sp, a4
	lui	a6, 12
	addi	a6, a6, 1120
	add	a4, a4, a6
	vs8r.v	v8, (a4)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v24, v8
	.loc	1 7 19                          # k135114449651120.py:7:19
	vand.vx	v16, v8, a3
	.loc	1 9 38                          # k135114449651120.py:9:38
	vadd.vv	v8, v8, v8
	.loc	1 7 19                          # k135114449651120.py:7:19
	vsub.vv	v16, v24, v16
	.loc	1 9 38                          # k135114449651120.py:9:38
	vand.vx	v8, v8, a2
	.loc	1 9 35 is_stmt 0                # k135114449651120.py:9:35
	vadd.vv	v8, v8, v16
	csrr	a4, vlenb
	li	a6, 176
	mul	a4, a4, a6
	add	a4, sp, a4
	lui	a6, 12
	addi	a6, a6, 1120
	add	a4, a4, a6
	vs8r.v	v8, (a4)                        # vscale x 64-byte Folded Spill
	li	a6, 96
	li	a7, 64
	li	a4, 448
	.loc	1 5 23 is_stmt 1                # k135114449651120.py:5:23
	vadd.vx	v8, v0, a6
	vor.vx	v16, v8, a5
	vadd.vx	v8, v0, a7
	vor.vx	v0, v8, a5
	csrr	a6, vlenb
	li	a7, 152
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 12
	addi	a7, a7, 1120
	add	a6, a6, a7
	vs8r.v	v16, (a6)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114449651120.py:6:21
	vmslt.vx	v8, v16, a4
	csrr	a6, vlenb
	li	a7, 144
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 12
	addi	a7, a7, 1120
	add	a6, a6, a7
	vs8r.v	v0, (a6)                        # vscale x 64-byte Folded Spill
	vmslt.vx	v16, v0, a4
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v16, v8, 4
	.loc	1 5 23                          # k135114449651120.py:5:23
	vsetvli	zero, a1, e32, m8, ta, ma
	vid.v	v8
	vadd.vx	v8, v8, a1
	vor.vx	v0, v8, a5
	csrr	a6, vlenb
	li	a7, 104
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 12
	addi	a7, a7, 1120
	add	a6, a6, a7
	vs8r.v	v24, (a6)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114449651120.py:6:21
	vmslt.vx	v9, v24, a4
	csrr	a6, vlenb
	li	a7, 136
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 12
	addi	a7, a7, 1120
	add	a6, a6, a7
	vs8r.v	v0, (a6)                        # vscale x 64-byte Folded Spill
	vmslt.vx	v8, v0, a4
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v9, v16, 8
	csrr	a6, vlenb
	li	a7, 176
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 12
	addi	a7, a7, 1120
	add	a6, a6, a7
	vl8r.v	v16, (a6)                       # vscale x 64-byte Folded Reload
	csrr	a6, vlenb
	li	a7, 184
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 12
	addi	a7, a7, 1120
	add	a6, a6, a7
	vs1r.v	v9, (a6)                        # vscale x 8-byte Folded Spill
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	a6, v9
	andi	a7, a6, 1
	vsext.vf2	v8, v16
	csrr	t0, vlenb
	li	t1, 96
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 12
	addi	t1, t1, 1120
	add	t0, t0, t1
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	vadd.vx	v24, v8, a0
	beqz	a7, .LBB0_2
# %bb.1:                                # %cond.load
	vmv.x.s	a7, v24
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa0, a7
	fmv.w.x	fs1, zero
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
	fsw	fs1, 256(sp)                    # 4-byte Folded Spill
	fsw	fs1, 260(sp)                    # 4-byte Folded Spill
	fsw	fs1, 264(sp)                    # 4-byte Folded Spill
	fsw	fs1, 268(sp)                    # 4-byte Folded Spill
	fsw	fs1, 272(sp)                    # 4-byte Folded Spill
	fsw	fs1, 276(sp)                    # 4-byte Folded Spill
	fsw	fs1, 280(sp)                    # 4-byte Folded Spill
	fsw	fs1, 284(sp)                    # 4-byte Folded Spill
	fsw	fs1, 288(sp)                    # 4-byte Folded Spill
	fsw	fs1, 292(sp)                    # 4-byte Folded Spill
	fsw	fs1, 296(sp)                    # 4-byte Folded Spill
	fsw	fs1, 300(sp)                    # 4-byte Folded Spill
	fsw	fs1, 304(sp)                    # 4-byte Folded Spill
	fsw	fs1, 308(sp)                    # 4-byte Folded Spill
	fsw	fs1, 312(sp)                    # 4-byte Folded Spill
	fsw	fs1, 316(sp)                    # 4-byte Folded Spill
	fsw	fs1, 320(sp)                    # 4-byte Folded Spill
	fsw	fs1, 324(sp)                    # 4-byte Folded Spill
	fsw	fs1, 328(sp)                    # 4-byte Folded Spill
	fsw	fs1, 332(sp)                    # 4-byte Folded Spill
	fsw	fs1, 336(sp)                    # 4-byte Folded Spill
	fsw	fs1, 340(sp)                    # 4-byte Folded Spill
	fsw	fs1, 344(sp)                    # 4-byte Folded Spill
	fsw	fs1, 348(sp)                    # 4-byte Folded Spill
	fsw	fs1, 352(sp)                    # 4-byte Folded Spill
	fsw	fs1, 356(sp)                    # 4-byte Folded Spill
	fsw	fs1, 360(sp)                    # 4-byte Folded Spill
	fsw	fs1, 364(sp)                    # 4-byte Folded Spill
	fsw	fs1, 368(sp)                    # 4-byte Folded Spill
	fsw	fs1, 372(sp)                    # 4-byte Folded Spill
	fsw	fs1, 376(sp)                    # 4-byte Folded Spill
	fsw	fs1, 380(sp)                    # 4-byte Folded Spill
	fsw	fs1, 384(sp)                    # 4-byte Folded Spill
	fsw	fs1, 388(sp)                    # 4-byte Folded Spill
	fsw	fs1, 392(sp)                    # 4-byte Folded Spill
	fsw	fs1, 396(sp)                    # 4-byte Folded Spill
	fsw	fs1, 400(sp)                    # 4-byte Folded Spill
	fsw	fs1, 404(sp)                    # 4-byte Folded Spill
	fsw	fs1, 408(sp)                    # 4-byte Folded Spill
	fsw	fs1, 412(sp)                    # 4-byte Folded Spill
	fsw	fs1, 416(sp)                    # 4-byte Folded Spill
	fsw	fs1, 420(sp)                    # 4-byte Folded Spill
	fsw	fs1, 424(sp)                    # 4-byte Folded Spill
	fsw	fs1, 428(sp)                    # 4-byte Folded Spill
	fsw	fs1, 432(sp)                    # 4-byte Folded Spill
	fsw	fs1, 436(sp)                    # 4-byte Folded Spill
	fsw	fs1, 440(sp)                    # 4-byte Folded Spill
	fsw	fs1, 444(sp)                    # 4-byte Folded Spill
	fsw	fs1, 448(sp)                    # 4-byte Folded Spill
	fsw	fs1, 452(sp)                    # 4-byte Folded Spill
	fsw	fs1, 456(sp)                    # 4-byte Folded Spill
	fsw	fs1, 460(sp)                    # 4-byte Folded Spill
	fsw	fs1, 464(sp)                    # 4-byte Folded Spill
	fsw	fs1, 468(sp)                    # 4-byte Folded Spill
	fsw	fs1, 472(sp)                    # 4-byte Folded Spill
	fsw	fs1, 476(sp)                    # 4-byte Folded Spill
	fsw	fs1, 480(sp)                    # 4-byte Folded Spill
	fsw	fs1, 484(sp)                    # 4-byte Folded Spill
	fsw	fs1, 488(sp)                    # 4-byte Folded Spill
	fsw	fs1, 492(sp)                    # 4-byte Folded Spill
	fsw	fs1, 496(sp)                    # 4-byte Folded Spill
	fsw	fs1, 500(sp)                    # 4-byte Folded Spill
	fsw	fs1, 504(sp)                    # 4-byte Folded Spill
	fsw	fs1, 508(sp)                    # 4-byte Folded Spill
	fsw	fs1, 512(sp)                    # 4-byte Folded Spill
	fsw	fs1, 516(sp)                    # 4-byte Folded Spill
	fsw	fs1, 520(sp)                    # 4-byte Folded Spill
	fsw	fs1, 524(sp)                    # 4-byte Folded Spill
	fsw	fs1, 528(sp)                    # 4-byte Folded Spill
	fsw	fs1, 532(sp)                    # 4-byte Folded Spill
	fsw	fs1, 536(sp)                    # 4-byte Folded Spill
	fsw	fs1, 540(sp)                    # 4-byte Folded Spill
	fsw	fs1, 544(sp)                    # 4-byte Folded Spill
	fsw	fs1, 548(sp)                    # 4-byte Folded Spill
	fsw	fs1, 552(sp)                    # 4-byte Folded Spill
	fsw	fs1, 556(sp)                    # 4-byte Folded Spill
	fsw	fs1, 560(sp)                    # 4-byte Folded Spill
	fsw	fs1, 564(sp)                    # 4-byte Folded Spill
	fsw	fs1, 568(sp)                    # 4-byte Folded Spill
	fsw	fs1, 572(sp)                    # 4-byte Folded Spill
	fsw	fs1, 576(sp)                    # 4-byte Folded Spill
	fsw	fs1, 580(sp)                    # 4-byte Folded Spill
	fsw	fs1, 584(sp)                    # 4-byte Folded Spill
	fsw	fs1, 588(sp)                    # 4-byte Folded Spill
	fsw	fs1, 592(sp)                    # 4-byte Folded Spill
	fsw	fs1, 596(sp)                    # 4-byte Folded Spill
	fsw	fs1, 600(sp)                    # 4-byte Folded Spill
	fsw	fs1, 604(sp)                    # 4-byte Folded Spill
	fsw	fs1, 608(sp)                    # 4-byte Folded Spill
	fsw	fs1, 612(sp)                    # 4-byte Folded Spill
	fsw	fs1, 616(sp)                    # 4-byte Folded Spill
	fsw	fs1, 620(sp)                    # 4-byte Folded Spill
	fsw	fs1, 624(sp)                    # 4-byte Folded Spill
	fsw	fs1, 628(sp)                    # 4-byte Folded Spill
	fsw	fs1, 632(sp)                    # 4-byte Folded Spill
	fsw	fs1, 636(sp)                    # 4-byte Folded Spill
	fsw	fs1, 640(sp)                    # 4-byte Folded Spill
	fsw	fs1, 644(sp)                    # 4-byte Folded Spill
	fsw	fs1, 648(sp)                    # 4-byte Folded Spill
	fsw	fs1, 652(sp)                    # 4-byte Folded Spill
	fsw	fs1, 656(sp)                    # 4-byte Folded Spill
	fsw	fs1, 660(sp)                    # 4-byte Folded Spill
	fsw	fs1, 664(sp)                    # 4-byte Folded Spill
	fsw	fs1, 668(sp)                    # 4-byte Folded Spill
	fsw	fs1, 672(sp)                    # 4-byte Folded Spill
	fsw	fs1, 676(sp)                    # 4-byte Folded Spill
	fsw	fs1, 680(sp)                    # 4-byte Folded Spill
	fsw	fs1, 684(sp)                    # 4-byte Folded Spill
	fsw	fs1, 688(sp)                    # 4-byte Folded Spill
	fsw	fs1, 692(sp)                    # 4-byte Folded Spill
	fsw	fs1, 696(sp)                    # 4-byte Folded Spill
	fsw	fs1, 700(sp)                    # 4-byte Folded Spill
	fsw	fs1, 704(sp)                    # 4-byte Folded Spill
	fsw	fs1, 708(sp)                    # 4-byte Folded Spill
	fsw	fs1, 712(sp)                    # 4-byte Folded Spill
	fsw	fs1, 716(sp)                    # 4-byte Folded Spill
	fsw	fs1, 720(sp)                    # 4-byte Folded Spill
	fsw	fs1, 724(sp)                    # 4-byte Folded Spill
	fsw	fs1, 728(sp)                    # 4-byte Folded Spill
	fsw	fs1, 732(sp)                    # 4-byte Folded Spill
	fsw	fs1, 736(sp)                    # 4-byte Folded Spill
	fsw	fs1, 740(sp)                    # 4-byte Folded Spill
	fsw	fs1, 744(sp)                    # 4-byte Folded Spill
	fsw	fs1, 748(sp)                    # 4-byte Folded Spill
	fsw	fs1, 752(sp)                    # 4-byte Folded Spill
	fsw	fs1, 756(sp)                    # 4-byte Folded Spill
	fsw	fs1, 760(sp)                    # 4-byte Folded Spill
	fsw	fs1, 764(sp)                    # 4-byte Folded Spill
	fsw	fs1, 768(sp)                    # 4-byte Folded Spill
	fsw	fs1, 772(sp)                    # 4-byte Folded Spill
	fsw	fs1, 776(sp)                    # 4-byte Folded Spill
	fsw	fs1, 780(sp)                    # 4-byte Folded Spill
	fsw	fs1, 784(sp)                    # 4-byte Folded Spill
	fsw	fs1, 788(sp)                    # 4-byte Folded Spill
	fsw	fs1, 792(sp)                    # 4-byte Folded Spill
	fsw	fs1, 796(sp)                    # 4-byte Folded Spill
	fsw	fs1, 800(sp)                    # 4-byte Folded Spill
	fsw	fs1, 804(sp)                    # 4-byte Folded Spill
	fsw	fs1, 808(sp)                    # 4-byte Folded Spill
	fsw	fs1, 812(sp)                    # 4-byte Folded Spill
	fsw	fs1, 816(sp)                    # 4-byte Folded Spill
	fsw	fs1, 820(sp)                    # 4-byte Folded Spill
	fsw	fs1, 824(sp)                    # 4-byte Folded Spill
	fsw	fs1, 828(sp)                    # 4-byte Folded Spill
	fsw	fs1, 832(sp)                    # 4-byte Folded Spill
	fsw	fs1, 836(sp)                    # 4-byte Folded Spill
	fsw	fs1, 840(sp)                    # 4-byte Folded Spill
	fsw	fs1, 844(sp)                    # 4-byte Folded Spill
	fsw	fs1, 848(sp)                    # 4-byte Folded Spill
	fsw	fs1, 852(sp)                    # 4-byte Folded Spill
	fsw	fs1, 856(sp)                    # 4-byte Folded Spill
	fsw	fs1, 860(sp)                    # 4-byte Folded Spill
	fsw	fs1, 864(sp)                    # 4-byte Folded Spill
	fsw	fs1, 868(sp)                    # 4-byte Folded Spill
	fsw	fs1, 872(sp)                    # 4-byte Folded Spill
	fsw	fs1, 876(sp)                    # 4-byte Folded Spill
	fsw	fs1, 880(sp)                    # 4-byte Folded Spill
	fsw	fs1, 884(sp)                    # 4-byte Folded Spill
	fsw	fs1, 888(sp)                    # 4-byte Folded Spill
	fsw	fs1, 892(sp)                    # 4-byte Folded Spill
	fsw	fs1, 896(sp)                    # 4-byte Folded Spill
	fsw	fs1, 900(sp)                    # 4-byte Folded Spill
	fsw	fs1, 904(sp)                    # 4-byte Folded Spill
	fsw	fs1, 908(sp)                    # 4-byte Folded Spill
	fsw	fs1, 912(sp)                    # 4-byte Folded Spill
	fsw	fs1, 916(sp)                    # 4-byte Folded Spill
	fsw	fs1, 920(sp)                    # 4-byte Folded Spill
	fsw	fs1, 924(sp)                    # 4-byte Folded Spill
	fsw	fs1, 928(sp)                    # 4-byte Folded Spill
	fsw	fs1, 932(sp)                    # 4-byte Folded Spill
	fsw	fs1, 936(sp)                    # 4-byte Folded Spill
	fsw	fs1, 940(sp)                    # 4-byte Folded Spill
	fsw	fs1, 944(sp)                    # 4-byte Folded Spill
	fsw	fs1, 948(sp)                    # 4-byte Folded Spill
	fsw	fs1, 952(sp)                    # 4-byte Folded Spill
	fsw	fs1, 956(sp)                    # 4-byte Folded Spill
	fsw	fs1, 960(sp)                    # 4-byte Folded Spill
	fsw	fs1, 964(sp)                    # 4-byte Folded Spill
	fsw	fs1, 968(sp)                    # 4-byte Folded Spill
	fsw	fs1, 972(sp)                    # 4-byte Folded Spill
	fsw	fs1, 976(sp)                    # 4-byte Folded Spill
	fsw	fs1, 980(sp)                    # 4-byte Folded Spill
	fsw	fs1, 984(sp)                    # 4-byte Folded Spill
	fsw	fs1, 988(sp)                    # 4-byte Folded Spill
	fsw	fs1, 992(sp)                    # 4-byte Folded Spill
	fsw	fs1, 996(sp)                    # 4-byte Folded Spill
	fsw	fs1, 1000(sp)                   # 4-byte Folded Spill
	fsw	fs1, 1004(sp)                   # 4-byte Folded Spill
	fsw	fs1, 1008(sp)                   # 4-byte Folded Spill
	fsw	fs1, 1012(sp)                   # 4-byte Folded Spill
	fsw	fs1, 1016(sp)                   # 4-byte Folded Spill
	fsw	fs1, 1020(sp)                   # 4-byte Folded Spill
	andi	a7, a6, 2
	bnez	a7, .LBB0_3
	j	.LBB0_4
.LBB0_2:
	.loc	1 0 43 is_stmt 0                # k135114449651120.py:0:43
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
	fsw	fa0, 512(sp)                    # 4-byte Folded Spill
	fsw	fa0, 516(sp)                    # 4-byte Folded Spill
	fsw	fa0, 520(sp)                    # 4-byte Folded Spill
	fsw	fa0, 524(sp)                    # 4-byte Folded Spill
	fsw	fa0, 528(sp)                    # 4-byte Folded Spill
	fsw	fa0, 532(sp)                    # 4-byte Folded Spill
	fsw	fa0, 536(sp)                    # 4-byte Folded Spill
	fsw	fa0, 540(sp)                    # 4-byte Folded Spill
	fsw	fa0, 544(sp)                    # 4-byte Folded Spill
	fsw	fa0, 548(sp)                    # 4-byte Folded Spill
	fsw	fa0, 552(sp)                    # 4-byte Folded Spill
	fsw	fa0, 556(sp)                    # 4-byte Folded Spill
	fsw	fa0, 560(sp)                    # 4-byte Folded Spill
	fsw	fa0, 564(sp)                    # 4-byte Folded Spill
	fsw	fa0, 568(sp)                    # 4-byte Folded Spill
	fsw	fa0, 572(sp)                    # 4-byte Folded Spill
	fsw	fa0, 576(sp)                    # 4-byte Folded Spill
	fsw	fa0, 580(sp)                    # 4-byte Folded Spill
	fsw	fa0, 584(sp)                    # 4-byte Folded Spill
	fsw	fa0, 588(sp)                    # 4-byte Folded Spill
	fsw	fa0, 592(sp)                    # 4-byte Folded Spill
	fsw	fa0, 596(sp)                    # 4-byte Folded Spill
	fsw	fa0, 600(sp)                    # 4-byte Folded Spill
	fsw	fa0, 604(sp)                    # 4-byte Folded Spill
	fsw	fa0, 608(sp)                    # 4-byte Folded Spill
	fsw	fa0, 612(sp)                    # 4-byte Folded Spill
	fsw	fa0, 616(sp)                    # 4-byte Folded Spill
	fsw	fa0, 620(sp)                    # 4-byte Folded Spill
	fsw	fa0, 624(sp)                    # 4-byte Folded Spill
	fsw	fa0, 628(sp)                    # 4-byte Folded Spill
	fsw	fa0, 632(sp)                    # 4-byte Folded Spill
	fsw	fa0, 636(sp)                    # 4-byte Folded Spill
	fsw	fa0, 640(sp)                    # 4-byte Folded Spill
	fsw	fa0, 644(sp)                    # 4-byte Folded Spill
	fsw	fa0, 648(sp)                    # 4-byte Folded Spill
	fsw	fa0, 652(sp)                    # 4-byte Folded Spill
	fsw	fa0, 656(sp)                    # 4-byte Folded Spill
	fsw	fa0, 660(sp)                    # 4-byte Folded Spill
	fsw	fa0, 664(sp)                    # 4-byte Folded Spill
	fsw	fa0, 668(sp)                    # 4-byte Folded Spill
	fsw	fa0, 672(sp)                    # 4-byte Folded Spill
	fsw	fa0, 676(sp)                    # 4-byte Folded Spill
	fsw	fa0, 680(sp)                    # 4-byte Folded Spill
	fsw	fa0, 684(sp)                    # 4-byte Folded Spill
	fsw	fa0, 688(sp)                    # 4-byte Folded Spill
	fsw	fa0, 692(sp)                    # 4-byte Folded Spill
	fsw	fa0, 696(sp)                    # 4-byte Folded Spill
	fsw	fa0, 700(sp)                    # 4-byte Folded Spill
	fsw	fa0, 704(sp)                    # 4-byte Folded Spill
	fsw	fa0, 708(sp)                    # 4-byte Folded Spill
	fsw	fa0, 712(sp)                    # 4-byte Folded Spill
	fsw	fa0, 716(sp)                    # 4-byte Folded Spill
	fsw	fa0, 720(sp)                    # 4-byte Folded Spill
	fsw	fa0, 724(sp)                    # 4-byte Folded Spill
	fsw	fa0, 728(sp)                    # 4-byte Folded Spill
	fsw	fa0, 732(sp)                    # 4-byte Folded Spill
	fsw	fa0, 736(sp)                    # 4-byte Folded Spill
	fsw	fa0, 740(sp)                    # 4-byte Folded Spill
	fsw	fa0, 744(sp)                    # 4-byte Folded Spill
	fsw	fa0, 748(sp)                    # 4-byte Folded Spill
	fsw	fa0, 752(sp)                    # 4-byte Folded Spill
	fsw	fa0, 756(sp)                    # 4-byte Folded Spill
	fsw	fa0, 760(sp)                    # 4-byte Folded Spill
	fsw	fa0, 764(sp)                    # 4-byte Folded Spill
	fsw	fa0, 768(sp)                    # 4-byte Folded Spill
	fsw	fa0, 772(sp)                    # 4-byte Folded Spill
	fsw	fa0, 776(sp)                    # 4-byte Folded Spill
	fsw	fa0, 780(sp)                    # 4-byte Folded Spill
	fsw	fa0, 784(sp)                    # 4-byte Folded Spill
	fsw	fa0, 788(sp)                    # 4-byte Folded Spill
	fsw	fa0, 792(sp)                    # 4-byte Folded Spill
	fsw	fa0, 796(sp)                    # 4-byte Folded Spill
	fsw	fa0, 800(sp)                    # 4-byte Folded Spill
	fsw	fa0, 804(sp)                    # 4-byte Folded Spill
	fsw	fa0, 808(sp)                    # 4-byte Folded Spill
	fsw	fa0, 812(sp)                    # 4-byte Folded Spill
	fsw	fa0, 816(sp)                    # 4-byte Folded Spill
	fsw	fa0, 820(sp)                    # 4-byte Folded Spill
	fsw	fa0, 824(sp)                    # 4-byte Folded Spill
	fsw	fa0, 828(sp)                    # 4-byte Folded Spill
	fsw	fa0, 832(sp)                    # 4-byte Folded Spill
	fsw	fa0, 836(sp)                    # 4-byte Folded Spill
	fsw	fa0, 840(sp)                    # 4-byte Folded Spill
	fsw	fa0, 844(sp)                    # 4-byte Folded Spill
	fsw	fa0, 848(sp)                    # 4-byte Folded Spill
	fsw	fa0, 852(sp)                    # 4-byte Folded Spill
	fsw	fa0, 856(sp)                    # 4-byte Folded Spill
	fsw	fa0, 860(sp)                    # 4-byte Folded Spill
	fsw	fa0, 864(sp)                    # 4-byte Folded Spill
	fsw	fa0, 868(sp)                    # 4-byte Folded Spill
	fsw	fa0, 872(sp)                    # 4-byte Folded Spill
	fsw	fa0, 876(sp)                    # 4-byte Folded Spill
	fsw	fa0, 880(sp)                    # 4-byte Folded Spill
	fsw	fa0, 884(sp)                    # 4-byte Folded Spill
	fsw	fa0, 888(sp)                    # 4-byte Folded Spill
	fsw	fa0, 892(sp)                    # 4-byte Folded Spill
	fsw	fa0, 896(sp)                    # 4-byte Folded Spill
	fsw	fa0, 900(sp)                    # 4-byte Folded Spill
	fsw	fa0, 904(sp)                    # 4-byte Folded Spill
	fsw	fa0, 908(sp)                    # 4-byte Folded Spill
	fsw	fa0, 912(sp)                    # 4-byte Folded Spill
	fsw	fa0, 916(sp)                    # 4-byte Folded Spill
	fsw	fa0, 920(sp)                    # 4-byte Folded Spill
	fsw	fa0, 924(sp)                    # 4-byte Folded Spill
	fsw	fa0, 928(sp)                    # 4-byte Folded Spill
	fsw	fa0, 932(sp)                    # 4-byte Folded Spill
	fsw	fa0, 936(sp)                    # 4-byte Folded Spill
	fsw	fa0, 940(sp)                    # 4-byte Folded Spill
	fsw	fa0, 944(sp)                    # 4-byte Folded Spill
	fsw	fa0, 948(sp)                    # 4-byte Folded Spill
	fsw	fa0, 952(sp)                    # 4-byte Folded Spill
	fsw	fa0, 956(sp)                    # 4-byte Folded Spill
	fsw	fa0, 960(sp)                    # 4-byte Folded Spill
	fsw	fa0, 964(sp)                    # 4-byte Folded Spill
	fsw	fa0, 968(sp)                    # 4-byte Folded Spill
	fsw	fa0, 972(sp)                    # 4-byte Folded Spill
	fsw	fa0, 976(sp)                    # 4-byte Folded Spill
	fsw	fa0, 980(sp)                    # 4-byte Folded Spill
	fsw	fa0, 984(sp)                    # 4-byte Folded Spill
	fsw	fa0, 988(sp)                    # 4-byte Folded Spill
	fsw	fa0, 992(sp)                    # 4-byte Folded Spill
	fsw	fa0, 996(sp)                    # 4-byte Folded Spill
	fsw	fa0, 1000(sp)                   # 4-byte Folded Spill
	fsw	fa0, 1004(sp)                   # 4-byte Folded Spill
	fsw	fa0, 1008(sp)                   # 4-byte Folded Spill
	fsw	fa0, 1012(sp)                   # 4-byte Folded Spill
	fsw	fa0, 1016(sp)                   # 4-byte Folded Spill
	fsw	fa0, 1020(sp)                   # 4-byte Folded Spill
	.loc	1 9 43                          # k135114449651120.py:9:43
	andi	a7, a6, 2
	beqz	a7, .LBB0_4
.LBB0_3:                                # %cond.load1
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs1, a7
.LBB0_4:                                # %else2
	andi	a7, a6, 4
	bnez	a7, .LBB0_17
# %bb.5:                                # %else5
	andi	a7, a6, 8
	bnez	a7, .LBB0_18
.LBB0_6:                                # %else8
	andi	t0, a6, 16
	lui	a7, 12
	addi	a7, a7, -1088
	add	a7, sp, a7
	bnez	t0, .LBB0_19
.LBB0_7:                                # %else11
	andi	t0, a6, 32
	bnez	t0, .LBB0_20
.LBB0_8:                                # %else14
	andi	t0, a6, 64
	bnez	t0, .LBB0_21
.LBB0_9:                                # %else17
	andi	t0, a6, 128
	bnez	t0, .LBB0_22
.LBB0_10:                               # %else20
	andi	t0, a6, 256
	bnez	t0, .LBB0_23
.LBB0_11:                               # %else23
	andi	t0, a6, 512
	bnez	t0, .LBB0_24
.LBB0_12:                               # %else26
	andi	t0, a6, 1024
	bnez	t0, .LBB0_25
.LBB0_13:                               # %else29
	slli	t0, a6, 52
	bltz	t0, .LBB0_26
.LBB0_14:                               # %else32
	slli	t0, a6, 51
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v0, v16, 16
	bltz	t0, .LBB0_27
.LBB0_15:                               # %else35
	slli	t0, a6, 50
	bgez	t0, .LBB0_28
.LBB0_16:                               # %cond.load37
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -256
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 936(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 52(sp)                     # 4-byte Folded Spill
	slli	t0, a6, 49
	vsext.vf2	v8, v0
	bltz	t0, .LBB0_29
	j	.LBB0_30
.LBB0_17:                               # %cond.load4
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs2, a7
	andi	a7, a6, 8
	beqz	a7, .LBB0_6
.LBB0_18:                               # %cond.load7
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs0, a7
	andi	t0, a6, 16
	lui	a7, 12
	addi	a7, a7, -1088
	add	a7, sp, a7
	beqz	t0, .LBB0_7
.LBB0_19:                               # %cond.load10
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, 896
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 2016(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fs11, t0
	andi	t0, a6, 32
	beqz	t0, .LBB0_8
.LBB0_20:                               # %cond.load13
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, 768
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1896(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fs10, t0
	andi	t0, a6, 64
	beqz	t0, .LBB0_9
.LBB0_21:                               # %cond.load16
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, 640
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1776(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fs9, t0
	andi	t0, a6, 128
	beqz	t0, .LBB0_10
.LBB0_22:                               # %cond.load19
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, 512
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1656(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fs8, t0
	andi	t0, a6, 256
	beqz	t0, .LBB0_11
.LBB0_23:                               # %cond.load22
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, 384
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1536(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fs7, t0
	andi	t0, a6, 512
	beqz	t0, .LBB0_12
.LBB0_24:                               # %cond.load25
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, 256
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1416(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fs6, t0
	andi	t0, a6, 1024
	beqz	t0, .LBB0_13
.LBB0_25:                               # %cond.load28
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, 128
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1296(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fs5, t0
	slli	t0, a6, 52
	bgez	t0, .LBB0_14
.LBB0_26:                               # %cond.load31
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1176(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fs4, t0
	slli	t0, a6, 51
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v0, v16, 16
	bgez	t0, .LBB0_15
.LBB0_27:                               # %cond.load34
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -128
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t0)
	ld	t0, 1056(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fs3, t0
	slli	t0, a6, 50
	bltz	t0, .LBB0_16
.LBB0_28:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	t0, a6, 49
	vsext.vf2	v8, v0
	bgez	t0, .LBB0_30
.LBB0_29:                               # %cond.load40
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -384
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (t0)
	ld	t0, 816(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 56(sp)                     # 4-byte Folded Spill
.LBB0_30:                               # %else41
	slli	t0, a6, 48
	lui	t1, 12
	addi	t1, t1, 1120
	add	t1, sp, t1
	vs8r.v	v8, (t1)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	bltz	t0, .LBB0_49
# %bb.31:                               # %else44
	slli	t0, a6, 47
	vadd.vx	v0, v8, a0
	bltz	t0, .LBB0_50
.LBB0_32:                               # %else47
	slli	t0, a6, 46
	bltz	t0, .LBB0_51
.LBB0_33:                               # %else50
	slli	t0, a6, 45
	bltz	t0, .LBB0_52
.LBB0_34:                               # %else53
	slli	t0, a6, 44
	bltz	t0, .LBB0_53
.LBB0_35:                               # %else56
	slli	t0, a6, 43
	bltz	t0, .LBB0_54
.LBB0_36:                               # %else59
	slli	t0, a6, 42
	bltz	t0, .LBB0_55
.LBB0_37:                               # %else62
	slli	t0, a6, 41
	bltz	t0, .LBB0_56
.LBB0_38:                               # %else65
	slli	t0, a6, 40
	bltz	t0, .LBB0_57
.LBB0_39:                               # %else68
	slli	t0, a6, 39
	bgez	t0, .LBB0_41
.LBB0_40:                               # %cond.load70
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -1152
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	a7, 0(a7)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 96(sp)                     # 4-byte Folded Spill
.LBB0_41:                               # %else71
	.loc	1 0 43                          # k135114449651120.py:0:43
	csrr	a7, vlenb
	li	t0, 136
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	csrr	a7, vlenb
	li	t0, 192
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vl8r.v	v16, (a7)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	t0, a6, 38
	lui	a7, 11
	addi	a7, a7, 872
	add	a7, sp, a7
	bgez	t0, .LBB0_43
# %bb.42:                               # %cond.load73
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -1280
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 2016(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 100(sp)                    # 4-byte Folded Spill
.LBB0_43:                               # %else74
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetvli	zero, a1, e32, m8, ta, ma
	vand.vx	v16, v8, a3
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	t0, a6, 37
	vadd.vv	v24, v8, v8
	bgez	t0, .LBB0_45
# %bb.44:                               # %cond.load76
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -1408
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1896(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 104(sp)                    # 4-byte Folded Spill
.LBB0_45:                               # %else77
	.loc	1 0 43                          # k135114449651120.py:0:43
	csrr	t0, vlenb
	li	t1, 136
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 12
	addi	t1, t1, 1120
	add	t0, t0, t1
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	t0, a6, 36
	vand.vx	v16, v24, a2
	bltz	t0, .LBB0_58
# %bb.46:                               # %else80
	slli	t0, a6, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	bltz	t0, .LBB0_59
.LBB0_47:                               # %else83
	slli	t0, a6, 34
	bgez	t0, .LBB0_60
.LBB0_48:                               # %cond.load85
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -1792
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1536(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 116(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 33
	vsext.vf2	v8, v24
	bltz	t0, .LBB0_61
	j	.LBB0_62
.LBB0_49:                               # %cond.load43
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -512
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (t0)
	ld	t0, 696(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 60(sp)                     # 4-byte Folded Spill
	slli	t0, a6, 47
	vadd.vx	v0, v8, a0
	bgez	t0, .LBB0_32
.LBB0_50:                               # %cond.load46
	vmv.x.s	t0, v0
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 64(sp)                     # 4-byte Folded Spill
	slli	t0, a6, 46
	bgez	t0, .LBB0_33
.LBB0_51:                               # %cond.load49
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 68(sp)                     # 4-byte Folded Spill
	slli	t0, a6, 45
	bgez	t0, .LBB0_34
.LBB0_52:                               # %cond.load52
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 72(sp)                     # 4-byte Folded Spill
	slli	t0, a6, 44
	bgez	t0, .LBB0_35
.LBB0_53:                               # %cond.load55
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 76(sp)                     # 4-byte Folded Spill
	slli	t0, a6, 43
	bgez	t0, .LBB0_36
.LBB0_54:                               # %cond.load58
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -640
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 480(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 80(sp)                     # 4-byte Folded Spill
	slli	t0, a6, 42
	bgez	t0, .LBB0_37
.LBB0_55:                               # %cond.load61
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -768
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 360(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 84(sp)                     # 4-byte Folded Spill
	slli	t0, a6, 41
	bgez	t0, .LBB0_38
.LBB0_56:                               # %cond.load64
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -896
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 240(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 88(sp)                     # 4-byte Folded Spill
	slli	t0, a6, 40
	bgez	t0, .LBB0_39
.LBB0_57:                               # %cond.load67
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -1024
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 120(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 92(sp)                     # 4-byte Folded Spill
	slli	t0, a6, 39
	bltz	t0, .LBB0_40
	j	.LBB0_41
.LBB0_58:                               # %cond.load79
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -1536
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1776(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 108(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	bgez	t0, .LBB0_47
.LBB0_59:                               # %cond.load82
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -1664
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1656(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 112(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 34
	bltz	t0, .LBB0_48
.LBB0_60:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	t0, a6, 33
	vsext.vf2	v8, v24
	bgez	t0, .LBB0_62
.LBB0_61:                               # %cond.load88
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 12
	addi	t0, t0, -1920
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (t0)
	ld	t0, 1416(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 120(sp)                    # 4-byte Folded Spill
.LBB0_62:                               # %else89
	slli	t0, a6, 32
	csrr	t1, vlenb
	slli	t1, t1, 3
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vs8r.v	v8, (t1)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	bltz	t0, .LBB0_78
# %bb.63:                               # %else92
	slli	t0, a6, 31
	vadd.vx	v0, v8, a0
	bltz	t0, .LBB0_79
.LBB0_64:                               # %else95
	slli	t0, a6, 30
	bltz	t0, .LBB0_80
.LBB0_65:                               # %else98
	slli	t0, a6, 29
	bltz	t0, .LBB0_81
.LBB0_66:                               # %else101
	slli	t0, a6, 28
	bltz	t0, .LBB0_82
.LBB0_67:                               # %else104
	slli	t0, a6, 27
	bltz	t0, .LBB0_83
.LBB0_68:                               # %else107
	slli	t0, a6, 26
	bltz	t0, .LBB0_84
.LBB0_69:                               # %else110
	slli	t0, a6, 25
	bltz	t0, .LBB0_85
.LBB0_70:                               # %else113
	slli	t0, a6, 24
	bltz	t0, .LBB0_86
.LBB0_71:                               # %else116
	slli	t0, a6, 23
	bltz	t0, .LBB0_87
.LBB0_72:                               # %else119
	slli	t0, a6, 22
	bltz	t0, .LBB0_88
.LBB0_73:                               # %else122
	slli	t0, a6, 21
	bltz	t0, .LBB0_89
.LBB0_74:                               # %else125
	slli	t0, a6, 20
	bltz	t0, .LBB0_90
.LBB0_75:                               # %else128
	slli	t0, a6, 19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v24, 16
	bltz	t0, .LBB0_91
.LBB0_76:                               # %else131
	slli	t0, a6, 18
	bgez	t0, .LBB0_92
.LBB0_77:                               # %cond.load133
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, 768
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	a7, 0(a7)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 180(sp)                    # 4-byte Folded Spill
	vsext.vf2	v8, v24
	slli	a7, a6, 17
	lui	t0, 11
	addi	t0, t0, -1240
	add	t0, sp, t0
	bltz	a7, .LBB0_93
	j	.LBB0_94
.LBB0_78:                               # %cond.load91
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	t0, 23
	slli	t0, t0, 11
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (t0)
	ld	t0, 1296(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 124(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 31
	vadd.vx	v0, v8, a0
	bgez	t0, .LBB0_64
.LBB0_79:                               # %cond.load94
	vmv.x.s	t0, v0
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 128(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 30
	bgez	t0, .LBB0_65
.LBB0_80:                               # %cond.load97
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 132(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 29
	bgez	t0, .LBB0_66
.LBB0_81:                               # %cond.load100
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 136(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 28
	bgez	t0, .LBB0_67
.LBB0_82:                               # %cond.load103
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 140(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 27
	bgez	t0, .LBB0_68
.LBB0_83:                               # %cond.load106
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, 1920
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1080(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 144(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 26
	bgez	t0, .LBB0_69
.LBB0_84:                               # %cond.load109
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, 1792
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 960(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 148(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 25
	bgez	t0, .LBB0_70
.LBB0_85:                               # %cond.load112
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, 1664
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 840(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 152(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 24
	bgez	t0, .LBB0_71
.LBB0_86:                               # %cond.load115
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, 1536
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 720(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 156(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 23
	bgez	t0, .LBB0_72
.LBB0_87:                               # %cond.load118
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, 1408
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 600(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 160(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 22
	bgez	t0, .LBB0_73
.LBB0_88:                               # %cond.load121
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, 1280
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 480(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 164(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 21
	bgez	t0, .LBB0_74
.LBB0_89:                               # %cond.load124
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, 1152
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 360(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 168(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 20
	bgez	t0, .LBB0_75
.LBB0_90:                               # %cond.load127
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, 1024
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 240(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 172(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v24, 16
	bgez	t0, .LBB0_76
.LBB0_91:                               # %cond.load130
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, 896
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 120(a7)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 176(sp)                    # 4-byte Folded Spill
	slli	t0, a6, 18
	bltz	t0, .LBB0_77
.LBB0_92:
	vsetivli	zero, 16, e64, m8, ta, ma
	vsext.vf2	v8, v24
	slli	a7, a6, 17
	lui	t0, 11
	addi	t0, t0, -1240
	add	t0, sp, t0
	bgez	a7, .LBB0_94
.LBB0_93:                               # %cond.load136
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, 640
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (a7)
	ld	a7, 1992(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 184(sp)                    # 4-byte Folded Spill
.LBB0_94:                               # %else137
	slli	a7, a6, 16
	csrr	t1, vlenb
	slli	t1, t1, 4
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vs8r.v	v8, (t1)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	bltz	a7, .LBB0_113
# %bb.95:                               # %else140
	slli	a7, a6, 15
	vadd.vx	v0, v8, a0
	bltz	a7, .LBB0_114
.LBB0_96:                               # %else143
	slli	a7, a6, 14
	bltz	a7, .LBB0_115
.LBB0_97:                               # %else146
	slli	a7, a6, 13
	bltz	a7, .LBB0_116
.LBB0_98:                               # %else149
	slli	a7, a6, 12
	bltz	a7, .LBB0_117
.LBB0_99:                               # %else152
	slli	a7, a6, 11
	bltz	a7, .LBB0_118
.LBB0_100:                              # %else155
	slli	a7, a6, 10
	bltz	a7, .LBB0_119
.LBB0_101:                              # %else158
	slli	a7, a6, 9
	bltz	a7, .LBB0_120
.LBB0_102:                              # %else161
	slli	a7, a6, 8
	bltz	a7, .LBB0_121
.LBB0_103:                              # %else164
	slli	a7, a6, 7
	bgez	a7, .LBB0_105
.LBB0_104:                              # %cond.load166
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, -128
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 1176(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 224(sp)                    # 4-byte Folded Spill
.LBB0_105:                              # %else167
	slli	a7, a6, 6
	csrr	t1, vlenb
	li	t2, 144
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vl8r.v	v8, (t1)                        # vscale x 64-byte Folded Reload
	csrr	t1, vlenb
	li	t2, 192
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a7, .LBB0_107
# %bb.106:                              # %cond.load169
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, -256
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 1056(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 228(sp)                    # 4-byte Folded Spill
.LBB0_107:                              # %else170
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetvli	zero, a1, e32, m8, ta, ma
	vand.vx	v16, v8, a3
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a7, a6, 5
	vadd.vv	v24, v8, v8
	bgez	a7, .LBB0_109
# %bb.108:                              # %cond.load172
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, -384
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 936(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 232(sp)                    # 4-byte Folded Spill
.LBB0_109:                              # %else173
	.loc	1 0 43                          # k135114449651120.py:0:43
	csrr	a7, vlenb
	li	t1, 144
	mul	a7, a7, t1
	add	a7, sp, a7
	lui	t1, 12
	addi	t1, t1, 1120
	add	a7, a7, t1
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a7, a6, 4
	vand.vx	v16, v24, a2
	bltz	a7, .LBB0_122
# %bb.110:                              # %else176
	slli	a7, a6, 3
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	bltz	a7, .LBB0_123
.LBB0_111:                              # %else179
	slli	a7, a6, 2
	bgez	a7, .LBB0_124
.LBB0_112:                              # %cond.load181
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, -768
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 576(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 244(sp)                    # 4-byte Folded Spill
	j	.LBB0_125
.LBB0_113:                              # %cond.load139
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, 512
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (a7)
	ld	a7, 1872(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 188(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 15
	vadd.vx	v0, v8, a0
	bgez	a7, .LBB0_96
.LBB0_114:                              # %cond.load142
	vmv.x.s	a7, v0
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 192(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 14
	bgez	a7, .LBB0_97
.LBB0_115:                              # %cond.load145
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 196(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 13
	bgez	a7, .LBB0_98
.LBB0_116:                              # %cond.load148
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 200(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 12
	bgez	a7, .LBB0_99
.LBB0_117:                              # %cond.load151
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 204(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 11
	bgez	a7, .LBB0_100
.LBB0_118:                              # %cond.load154
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, 384
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 1656(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 208(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 10
	bgez	a7, .LBB0_101
.LBB0_119:                              # %cond.load157
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, 256
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 1536(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 212(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 9
	bgez	a7, .LBB0_102
.LBB0_120:                              # %cond.load160
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, 128
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 1416(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 216(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 8
	bgez	a7, .LBB0_103
.LBB0_121:                              # %cond.load163
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 1296(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 220(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 7
	bltz	a7, .LBB0_104
	j	.LBB0_105
.LBB0_122:                              # %cond.load175
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, -512
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 816(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 236(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 3
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	bgez	a7, .LBB0_111
.LBB0_123:                              # %cond.load178
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, -640
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 696(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 240(sp)                    # 4-byte Folded Spill
	slli	a7, a6, 2
	bltz	a7, .LBB0_112
.LBB0_124:
	vsetivli	zero, 16, e64, m8, ta, ma
.LBB0_125:                              # %else182
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsext.vf2	v8, v24
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a7, a6, 1
	csrr	t1, vlenb
	li	t2, 184
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vl1r.v	v16, (t1)                       # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v16, 1
	bgez	a7, .LBB0_127
# %bb.126:                              # %cond.load184
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a7, 11
	addi	a7, a7, -896
	add	a7, sp, a7
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 456(t0)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 248(sp)                    # 4-byte Folded Spill
.LBB0_127:                              # %else185
	.loc	1 0 43                          # k135114449651120.py:0:43
	csrr	a7, vlenb
	li	t1, 24
	mul	a7, a7, t1
	add	a7, sp, a7
	lui	t1, 12
	addi	t1, t1, 1120
	add	a7, a7, t1
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	.loc	1 9 43                          # k135114449651120.py:9:43
	vmv.x.s	a7, v16
	bltz	a6, .LBB0_143
# %bb.128:                              # %else188
	andi	a6, a7, 1
	vadd.vx	v0, v8, a0
	bnez	a6, .LBB0_144
.LBB0_129:                              # %else191
	andi	a6, a7, 2
	bnez	a6, .LBB0_145
.LBB0_130:                              # %else194
	andi	a6, a7, 4
	bnez	a6, .LBB0_146
.LBB0_131:                              # %else197
	andi	a6, a7, 8
	bnez	a6, .LBB0_147
.LBB0_132:                              # %else200
	andi	a6, a7, 16
	bnez	a6, .LBB0_148
.LBB0_133:                              # %else203
	andi	a6, a7, 32
	bnez	a6, .LBB0_149
.LBB0_134:                              # %else206
	andi	t0, a7, 64
	lui	a6, 10
	addi	a6, a6, 720
	add	a6, sp, a6
	bnez	t0, .LBB0_150
.LBB0_135:                              # %else209
	andi	t0, a7, 128
	bnez	t0, .LBB0_151
.LBB0_136:                              # %else212
	andi	t0, a7, 256
	bnez	t0, .LBB0_152
.LBB0_137:                              # %else215
	andi	t0, a7, 512
	bnez	t0, .LBB0_153
.LBB0_138:                              # %else218
	andi	t0, a7, 1024
	bnez	t0, .LBB0_154
.LBB0_139:                              # %else221
	slli	t0, a7, 52
	bltz	t0, .LBB0_155
.LBB0_140:                              # %else224
	slli	t0, a7, 51
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v24, 16
	bltz	t0, .LBB0_156
.LBB0_141:                              # %else227
	slli	t0, a7, 50
	bgez	t0, .LBB0_157
.LBB0_142:                              # %cond.load229
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 1792
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1176(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 308(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 49
	vsext.vf2	v8, v24
	bltz	t0, .LBB0_158
	j	.LBB0_159
.LBB0_143:                              # %cond.load187
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 11
	addi	a6, a6, -1024
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (a6)
	ld	a6, 336(t0)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 252(sp)                    # 4-byte Folded Spill
	andi	a6, a7, 1
	vadd.vx	v0, v8, a0
	beqz	a6, .LBB0_129
.LBB0_144:                              # %cond.load190
	vmv.x.s	a6, v0
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 256(sp)                    # 4-byte Folded Spill
	andi	a6, a7, 2
	beqz	a6, .LBB0_130
.LBB0_145:                              # %cond.load193
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 260(sp)                    # 4-byte Folded Spill
	andi	a6, a7, 4
	beqz	a6, .LBB0_131
.LBB0_146:                              # %cond.load196
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 264(sp)                    # 4-byte Folded Spill
	andi	a6, a7, 8
	beqz	a6, .LBB0_132
.LBB0_147:                              # %cond.load199
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 268(sp)                    # 4-byte Folded Spill
	andi	a6, a7, 16
	beqz	a6, .LBB0_133
.LBB0_148:                              # %cond.load202
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 11
	addi	a6, a6, -1152
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 120(t0)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 272(sp)                    # 4-byte Folded Spill
	andi	a6, a7, 32
	beqz	a6, .LBB0_134
.LBB0_149:                              # %cond.load205
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 11
	addi	a6, a6, -1280
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 0(t0)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 276(sp)                    # 4-byte Folded Spill
	andi	t0, a7, 64
	lui	a6, 10
	addi	a6, a6, 720
	add	a6, sp, a6
	beqz	t0, .LBB0_135
.LBB0_150:                              # %cond.load208
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, -1408
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 2016(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 280(sp)                    # 4-byte Folded Spill
	andi	t0, a7, 128
	beqz	t0, .LBB0_136
.LBB0_151:                              # %cond.load211
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, -1536
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1896(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 284(sp)                    # 4-byte Folded Spill
	andi	t0, a7, 256
	beqz	t0, .LBB0_137
.LBB0_152:                              # %cond.load214
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, -1664
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1776(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 288(sp)                    # 4-byte Folded Spill
	andi	t0, a7, 512
	beqz	t0, .LBB0_138
.LBB0_153:                              # %cond.load217
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, -1792
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1656(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 292(sp)                    # 4-byte Folded Spill
	andi	t0, a7, 1024
	beqz	t0, .LBB0_139
.LBB0_154:                              # %cond.load220
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 11
	addi	t0, t0, -1920
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1536(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 296(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 52
	bgez	t0, .LBB0_140
.LBB0_155:                              # %cond.load223
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	t0, 21
	slli	t0, t0, 11
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1416(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 300(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 51
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v24, 16
	bgez	t0, .LBB0_141
.LBB0_156:                              # %cond.load226
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 1920
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1296(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 304(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 50
	bltz	t0, .LBB0_142
.LBB0_157:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	t0, a7, 49
	vsext.vf2	v8, v24
	bgez	t0, .LBB0_159
.LBB0_158:                              # %cond.load232
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 1664
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (t0)
	ld	t0, 1056(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 312(sp)                    # 4-byte Folded Spill
.LBB0_159:                              # %else233
	slli	t0, a7, 48
	csrr	t1, vlenb
	slli	t1, t1, 5
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vs8r.v	v8, (t1)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	bltz	t0, .LBB0_178
# %bb.160:                              # %else236
	slli	t0, a7, 47
	vadd.vx	v0, v8, a0
	bltz	t0, .LBB0_179
.LBB0_161:                              # %else239
	slli	t0, a7, 46
	bltz	t0, .LBB0_180
.LBB0_162:                              # %else242
	slli	t0, a7, 45
	bltz	t0, .LBB0_181
.LBB0_163:                              # %else245
	slli	t0, a7, 44
	bltz	t0, .LBB0_182
.LBB0_164:                              # %else248
	slli	t0, a7, 43
	bltz	t0, .LBB0_183
.LBB0_165:                              # %else251
	slli	t0, a7, 42
	bltz	t0, .LBB0_184
.LBB0_166:                              # %else254
	slli	t0, a7, 41
	bltz	t0, .LBB0_185
.LBB0_167:                              # %else257
	slli	t0, a7, 40
	bltz	t0, .LBB0_186
.LBB0_168:                              # %else260
	slli	t0, a7, 39
	bgez	t0, .LBB0_170
.LBB0_169:                              # %cond.load262
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 896
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 240(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 352(sp)                    # 4-byte Folded Spill
.LBB0_170:                              # %else263
	slli	t0, a7, 38
	csrr	t1, vlenb
	li	t2, 152
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vl8r.v	v8, (t1)                        # vscale x 64-byte Folded Reload
	csrr	t1, vlenb
	li	t2, 192
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	t0, .LBB0_172
# %bb.171:                              # %cond.load265
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 768
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 120(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 356(sp)                    # 4-byte Folded Spill
.LBB0_172:                              # %else266
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetvli	zero, a1, e32, m8, ta, ma
	vand.vx	v16, v8, a3
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	t0, a7, 37
	vadd.vv	v24, v8, v8
	bgez	t0, .LBB0_174
# %bb.173:                              # %cond.load268
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 640
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	a6, 0(a6)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 360(sp)                    # 4-byte Folded Spill
.LBB0_174:                              # %else269
	.loc	1 0 43                          # k135114449651120.py:0:43
	csrr	a6, vlenb
	li	t0, 152
	mul	a6, a6, t0
	add	a6, sp, a6
	lui	t0, 12
	addi	t0, t0, 1120
	add	a6, a6, t0
	vl8r.v	v8, (a6)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	vand.vx	v16, v24, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	t0, a7, 36
	lui	a6, 10
	addi	a6, a6, -1416
	add	a6, sp, a6
	bltz	t0, .LBB0_187
# %bb.175:                              # %else272
	slli	t0, a7, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	bltz	t0, .LBB0_188
.LBB0_176:                              # %else275
	slli	t0, a7, 34
	bgez	t0, .LBB0_189
.LBB0_177:                              # %cond.load277
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 256
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1776(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 372(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 33
	vsext.vf2	v8, v24
	bltz	t0, .LBB0_190
	j	.LBB0_191
.LBB0_178:                              # %cond.load235
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 1536
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (t0)
	ld	t0, 936(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 316(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 47
	vadd.vx	v0, v8, a0
	bgez	t0, .LBB0_161
.LBB0_179:                              # %cond.load238
	vmv.x.s	t0, v0
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 320(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 46
	bgez	t0, .LBB0_162
.LBB0_180:                              # %cond.load241
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 324(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 45
	bgez	t0, .LBB0_163
.LBB0_181:                              # %cond.load244
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 328(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 44
	bgez	t0, .LBB0_164
.LBB0_182:                              # %cond.load247
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 332(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 43
	bgez	t0, .LBB0_165
.LBB0_183:                              # %cond.load250
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 1408
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 720(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 336(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 42
	bgez	t0, .LBB0_166
.LBB0_184:                              # %cond.load253
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 1280
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 600(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 340(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 41
	bgez	t0, .LBB0_167
.LBB0_185:                              # %cond.load256
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 1152
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 480(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 344(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 40
	bgez	t0, .LBB0_168
.LBB0_186:                              # %cond.load259
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 1024
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 360(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 348(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 39
	bltz	t0, .LBB0_169
	j	.LBB0_170
.LBB0_187:                              # %cond.load271
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 512
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 2016(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 364(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	bgez	t0, .LBB0_176
.LBB0_188:                              # %cond.load274
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 384
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1896(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 368(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 34
	bltz	t0, .LBB0_177
.LBB0_189:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	t0, a7, 33
	vsext.vf2	v8, v24
	bgez	t0, .LBB0_191
.LBB0_190:                              # %cond.load280
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, 128
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (t0)
	ld	t0, 1656(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 376(sp)                    # 4-byte Folded Spill
.LBB0_191:                              # %else281
	slli	t0, a7, 32
	csrr	t1, vlenb
	li	t2, 40
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vs8r.v	v8, (t1)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	bltz	t0, .LBB0_207
# %bb.192:                              # %else284
	slli	t0, a7, 31
	vadd.vx	v0, v8, a0
	bltz	t0, .LBB0_208
.LBB0_193:                              # %else287
	slli	t0, a7, 30
	bltz	t0, .LBB0_209
.LBB0_194:                              # %else290
	slli	t0, a7, 29
	bltz	t0, .LBB0_210
.LBB0_195:                              # %else293
	slli	t0, a7, 28
	bltz	t0, .LBB0_211
.LBB0_196:                              # %else296
	slli	t0, a7, 27
	bltz	t0, .LBB0_212
.LBB0_197:                              # %else299
	slli	t0, a7, 26
	bltz	t0, .LBB0_213
.LBB0_198:                              # %else302
	slli	t0, a7, 25
	bltz	t0, .LBB0_214
.LBB0_199:                              # %else305
	slli	t0, a7, 24
	bltz	t0, .LBB0_215
.LBB0_200:                              # %else308
	slli	t0, a7, 23
	bltz	t0, .LBB0_216
.LBB0_201:                              # %else311
	slli	t0, a7, 22
	bltz	t0, .LBB0_217
.LBB0_202:                              # %else314
	slli	t0, a7, 21
	bltz	t0, .LBB0_218
.LBB0_203:                              # %else317
	slli	t0, a7, 20
	bltz	t0, .LBB0_219
.LBB0_204:                              # %else320
	slli	t0, a7, 19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v24, 16
	bltz	t0, .LBB0_220
.LBB0_205:                              # %else323
	slli	t0, a7, 18
	bgez	t0, .LBB0_221
.LBB0_206:                              # %cond.load325
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -1280
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 240(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 436(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 17
	vsext.vf2	v8, v24
	bltz	t0, .LBB0_222
	j	.LBB0_223
.LBB0_207:                              # %cond.load283
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (t0)
	ld	t0, 1536(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 380(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 31
	vadd.vx	v0, v8, a0
	bgez	t0, .LBB0_193
.LBB0_208:                              # %cond.load286
	vmv.x.s	t0, v0
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 384(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 30
	bgez	t0, .LBB0_194
.LBB0_209:                              # %cond.load289
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 388(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 29
	bgez	t0, .LBB0_195
.LBB0_210:                              # %cond.load292
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 392(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 28
	bgez	t0, .LBB0_196
.LBB0_211:                              # %cond.load295
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	t0, v8
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 396(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 27
	bgez	t0, .LBB0_197
.LBB0_212:                              # %cond.load298
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -128
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1320(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 400(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 26
	bgez	t0, .LBB0_198
.LBB0_213:                              # %cond.load301
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -256
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1200(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 404(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 25
	bgez	t0, .LBB0_199
.LBB0_214:                              # %cond.load304
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -384
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1080(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 408(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 24
	bgez	t0, .LBB0_200
.LBB0_215:                              # %cond.load307
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -512
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 960(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 412(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 23
	bgez	t0, .LBB0_201
.LBB0_216:                              # %cond.load310
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -640
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 840(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 416(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 22
	bgez	t0, .LBB0_202
.LBB0_217:                              # %cond.load313
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -768
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 720(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 420(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 21
	bgez	t0, .LBB0_203
.LBB0_218:                              # %cond.load316
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -896
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 600(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 424(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 20
	bgez	t0, .LBB0_204
.LBB0_219:                              # %cond.load319
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -1024
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 480(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 428(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v24, 16
	bgez	t0, .LBB0_205
.LBB0_220:                              # %cond.load322
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -1152
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 360(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 432(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 18
	bltz	t0, .LBB0_206
.LBB0_221:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	t0, a7, 17
	vsext.vf2	v8, v24
	bgez	t0, .LBB0_223
.LBB0_222:                              # %cond.load328
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -1408
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (t0)
	ld	t0, 120(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 440(sp)                    # 4-byte Folded Spill
.LBB0_223:                              # %else329
	slli	t0, a7, 16
	csrr	t1, vlenb
	li	t2, 48
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vs8r.v	v8, (t1)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	bltz	t0, .LBB0_263
# %bb.224:                              # %else332
	slli	a6, a7, 15
	vadd.vx	v0, v8, a0
	bltz	a6, .LBB0_264
.LBB0_225:                              # %else335
	slli	a6, a7, 14
	bltz	a6, .LBB0_265
.LBB0_226:                              # %else338
	slli	a6, a7, 13
	bltz	a6, .LBB0_266
.LBB0_227:                              # %else341
	slli	a6, a7, 12
	bltz	a6, .LBB0_267
.LBB0_228:                              # %else344
	slli	t0, a7, 11
	lui	a6, 9
	addi	a6, a6, 448
	add	a6, sp, a6
	bltz	t0, .LBB0_268
.LBB0_229:                              # %else347
	slli	t0, a7, 10
	bltz	t0, .LBB0_269
.LBB0_230:                              # %else350
	slli	t1, a7, 9
	li	t0, 128
	bltz	t1, .LBB0_270
.LBB0_231:                              # %else353
	slli	t1, a7, 8
	vsetvli	zero, a1, e32, m8, ta, ma
	vid.v	v8
	vadd.vx	v8, v8, t0
	bgez	t1, .LBB0_233
.LBB0_232:                              # %cond.load355
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	t0, 19
	slli	t0, t0, 11
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1656(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 476(sp)                    # 4-byte Folded Spill
.LBB0_233:                              # %else356
	slli	t0, a7, 7
	vsetvli	zero, a1, e32, m8, ta, ma
	vor.vx	v8, v8, a5
	csrr	t1, vlenb
	li	t2, 176
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vs8r.v	v8, (t1)                        # vscale x 64-byte Folded Spill
	bgez	t0, .LBB0_235
# %bb.234:                              # %cond.load358
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 9
	addi	t0, t0, 1920
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1536(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 480(sp)                    # 4-byte Folded Spill
.LBB0_235:                              # %else359
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	t0, 224
	li	t1, 192
	li	t2, 160
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	t3, a7, 6
	csrr	t4, vlenb
	li	t5, 176
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 12
	addi	t5, t5, 1120
	add	t4, t4, t5
	vl8r.v	v8, (t4)                        # vscale x 64-byte Folded Reload
	csrr	t4, vlenb
	li	t5, 192
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 12
	addi	t5, t5, 1120
	add	t4, t4, t5
	vl8r.v	v16, (t4)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v8, v16
	csrr	t4, vlenb
	slli	t4, t4, 7
	add	t4, sp, t4
	lui	t5, 12
	addi	t5, t5, 1120
	add	t4, t4, t5
	vs8r.v	v0, (t4)                        # vscale x 64-byte Folded Spill
	bgez	t3, .LBB0_237
# %bb.236:                              # %cond.load361
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t3, 9
	addi	t3, t3, 1792
	add	t3, sp, t3
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t3)
	ld	t3, 1416(a6)
	lhu	t3, 0(t3)
	slli	t3, t3, 16
	fmv.w.x	fa5, t3
	fsw	fa5, 484(sp)                    # 4-byte Folded Spill
.LBB0_237:                              # %else362
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetvli	zero, a1, e32, m8, ta, ma
	vid.v	v8
	vadd.vx	v0, v8, t0
	vadd.vx	v16, v8, t1
	vadd.vx	v8, v8, t2
	csrr	t0, vlenb
	li	t1, 184
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 12
	addi	t1, t1, 1120
	add	t0, t0, t1
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	t0, a7, 5
	vand.vx	v8, v24, a3
	vadd.vv	v24, v24, v24
	bgez	t0, .LBB0_239
# %bb.238:                              # %cond.load364
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 9
	addi	t0, t0, 1664
	add	t0, sp, t0
	csrr	t1, vlenb
	li	t2, 168
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	csrr	t1, vlenb
	slli	t1, t1, 7
	add	t1, sp, t1
	lui	t2, 12
	addi	t2, t2, 1120
	add	t1, t1, t2
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t0)
	csrr	t0, vlenb
	li	t1, 168
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 12
	addi	t1, t1, 1120
	add	t0, t0, t1
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	ld	t0, 1296(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 488(sp)                    # 4-byte Folded Spill
.LBB0_239:                              # %else365
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetvli	zero, a1, e32, m8, ta, ma
	vor.vx	v0, v0, a5
	csrr	t0, vlenb
	li	t1, 160
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 12
	addi	t1, t1, 1120
	add	t0, t0, t1
	vs8r.v	v0, (t0)                        # vscale x 64-byte Folded Spill
	vor.vx	v16, v16, a5
	csrr	t0, vlenb
	li	t1, 168
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 12
	addi	t1, t1, 1120
	add	t0, t0, t1
	vs8r.v	v16, (t0)                       # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t1, 184
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 12
	addi	t1, t1, 1120
	add	t0, t0, t1
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	vor.vx	v16, v16, a5
	csrr	a5, vlenb
	li	t0, 184
	mul	a5, a5, t0
	add	a5, sp, a5
	lui	t0, 12
	addi	t0, t0, 1120
	add	a5, a5, t0
	vs8r.v	v16, (a5)                       # vscale x 64-byte Folded Spill
	csrr	a5, vlenb
	li	t0, 176
	mul	a5, a5, t0
	add	a5, sp, a5
	lui	t0, 12
	addi	t0, t0, 1120
	add	a5, a5, t0
	vl8r.v	v16, (a5)                       # vscale x 64-byte Folded Reload
	vsub.vv	v8, v16, v8
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a5, a7, 4
	vand.vx	v16, v24, a2
	bgez	a5, .LBB0_241
# %bb.240:                              # %cond.load367
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 9
	addi	a5, a5, 1536
	add	a5, sp, a5
	csrr	t0, vlenb
	slli	t0, t0, 7
	add	t0, sp, t0
	lui	t1, 12
	addi	t1, t1, 1120
	add	t0, t0, t1
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 1176(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 492(sp)                    # 4-byte Folded Spill
.LBB0_241:                              # %else368
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v16, v8
	csrr	a5, vlenb
	li	t0, 160
	mul	a5, a5, t0
	add	a5, sp, a5
	lui	t0, 12
	addi	t0, t0, 1120
	add	a5, a5, t0
	vl8r.v	v24, (a5)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v9, v24, a4
	csrr	a5, vlenb
	li	t0, 168
	mul	a5, a5, t0
	add	a5, sp, a5
	lui	t0, 12
	addi	t0, t0, 1120
	add	a5, a5, t0
	vl8r.v	v24, (a5)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v8, v24, a4
	csrr	a5, vlenb
	li	t0, 184
	mul	a5, a5, t0
	add	a5, sp, a5
	lui	t0, 12
	addi	t0, t0, 1120
	add	a5, a5, t0
	vl8r.v	v24, (a5)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v10, v24, a4
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a5, a7, 3
	csrr	t0, vlenb
	li	t1, 176
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 12
	addi	t1, t1, 1120
	add	t0, t0, t1
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v7, v24, a4
	bgez	a5, .LBB0_243
# %bb.242:                              # %cond.load370
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a4, 9
	addi	a4, a4, 1408
	add	a4, sp, a4
	csrr	a5, vlenb
	slli	a5, a5, 7
	add	a5, sp, a5
	lui	t0, 12
	addi	t0, t0, 1120
	add	a5, a5, t0
	vl8r.v	v24, (a5)                       # vscale x 64-byte Folded Reload
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 1056(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 496(sp)                    # 4-byte Folded Spill
.LBB0_243:                              # %else371
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v8, v9, 4
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a4, a7, 2
	vslideup.vi	v7, v10, 4
	bgez	a4, .LBB0_245
# %bb.244:                              # %cond.load373
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a4, 9
	addi	a4, a4, 1280
	add	a4, sp, a4
	csrr	a5, vlenb
	slli	a5, a5, 7
	add	a5, sp, a5
	lui	t0, 12
	addi	t0, t0, 1120
	add	a5, a5, t0
	vl8r.v	v24, (a5)                       # vscale x 64-byte Folded Reload
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 936(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 500(sp)                    # 4-byte Folded Spill
.LBB0_245:                              # %else374
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vsext.vf2	v24, v16
	csrr	a4, vlenb
	li	a5, 112
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 12
	addi	a5, a5, 1120
	add	a4, a4, a5
	vs8r.v	v24, (a4)                       # vscale x 64-byte Folded Spill
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a4, a7, 1
	vsetvli	zero, zero, e8, m1, ta, ma
	vslideup.vi	v7, v8, 8
	bgez	a4, .LBB0_247
# %bb.246:                              # %cond.load376
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a4, 9
	addi	a4, a4, 1152
	add	a4, sp, a4
	csrr	a5, vlenb
	slli	a5, a5, 7
	add	a5, sp, a5
	lui	t0, 12
	addi	t0, t0, 1120
	add	a5, a5, t0
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v8, (a4)
	ld	a4, 816(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 504(sp)                    # 4-byte Folded Spill
.LBB0_247:                              # %else377
	.loc	1 0 43                          # k135114449651120.py:0:43
	csrr	a4, vlenb
	li	a5, 112
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 12
	addi	a5, a5, 1120
	add	a4, a4, a5
	vl8r.v	v8, (a4)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	.loc	1 9 43                          # k135114449651120.py:9:43
	vmv.x.s	a4, v7
	bltz	a7, .LBB0_271
# %bb.248:                              # %else380
	andi	a5, a4, 1
	vadd.vx	v24, v8, a0
	bnez	a5, .LBB0_272
.LBB0_249:                              # %else383
	andi	a5, a4, 2
	bnez	a5, .LBB0_273
.LBB0_250:                              # %else386
	andi	a5, a4, 4
	bnez	a5, .LBB0_274
.LBB0_251:                              # %else389
	andi	a5, a4, 8
	bnez	a5, .LBB0_275
.LBB0_252:                              # %else392
	andi	a5, a4, 16
	bnez	a5, .LBB0_276
.LBB0_253:                              # %else395
	andi	a5, a4, 32
	bnez	a5, .LBB0_277
.LBB0_254:                              # %else398
	andi	a5, a4, 64
	bnez	a5, .LBB0_278
.LBB0_255:                              # %else401
	andi	a5, a4, 128
	bnez	a5, .LBB0_279
.LBB0_256:                              # %else404
	andi	a5, a4, 256
	bnez	a5, .LBB0_280
.LBB0_257:                              # %else407
	andi	a6, a4, 512
	lui	a5, 9
	addi	a5, a5, -1688
	add	a5, sp, a5
	bnez	a6, .LBB0_281
.LBB0_258:                              # %else410
	andi	a6, a4, 1024
	bnez	a6, .LBB0_282
.LBB0_259:                              # %else413
	slli	a6, a4, 52
	bltz	a6, .LBB0_283
.LBB0_260:                              # %else416
	slli	a6, a4, 51
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bltz	a6, .LBB0_284
.LBB0_261:                              # %else419
	slli	a6, a4, 50
	bgez	a6, .LBB0_285
.LBB0_262:                              # %cond.load421
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -256
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1536(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 564(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 49
	vsext.vf2	v8, v16
	bltz	a6, .LBB0_286
	j	.LBB0_287
.LBB0_263:                              # %cond.load331
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -1536
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v0, (t0)
	ld	a6, 0(a6)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 444(sp)                    # 4-byte Folded Spill
	slli	a6, a7, 15
	vadd.vx	v0, v8, a0
	bgez	a6, .LBB0_225
.LBB0_264:                              # %cond.load334
	vmv.x.s	a6, v0
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 448(sp)                    # 4-byte Folded Spill
	slli	a6, a7, 14
	bgez	a6, .LBB0_226
.LBB0_265:                              # %cond.load337
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 452(sp)                    # 4-byte Folded Spill
	slli	a6, a7, 13
	bgez	a6, .LBB0_227
.LBB0_266:                              # %cond.load340
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 456(sp)                    # 4-byte Folded Spill
	slli	a6, a7, 12
	bgez	a6, .LBB0_228
.LBB0_267:                              # %cond.load343
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 460(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 11
	lui	a6, 9
	addi	a6, a6, 448
	add	a6, sp, a6
	bgez	t0, .LBB0_229
.LBB0_268:                              # %cond.load346
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -1664
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 2016(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 464(sp)                    # 4-byte Folded Spill
	slli	t0, a7, 10
	bgez	t0, .LBB0_230
.LBB0_269:                              # %cond.load349
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t0, 10
	addi	t0, t0, -1792
	add	t0, sp, t0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1896(a6)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa5, t0
	fsw	fa5, 468(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 9
	li	t0, 128
	bgez	t1, .LBB0_231
.LBB0_270:                              # %cond.load352
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	t1, 10
	addi	t1, t1, -1920
	add	t1, sp, t1
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t1)
	ld	t1, 1776(a6)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa5, t1
	fsw	fa5, 472(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 8
	vsetvli	zero, a1, e32, m8, ta, ma
	vid.v	v8
	vadd.vx	v8, v8, t0
	bltz	t1, .LBB0_232
	j	.LBB0_233
.LBB0_271:                              # %cond.load379
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 9
	addi	a5, a5, 1024
	add	a5, sp, a5
	csrr	a7, vlenb
	slli	a7, a7, 7
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vl8r.v	v24, (a7)                       # vscale x 64-byte Folded Reload
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (a5)
	ld	a5, 696(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 508(sp)                    # 4-byte Folded Spill
	andi	a5, a4, 1
	vadd.vx	v24, v8, a0
	beqz	a5, .LBB0_249
.LBB0_272:                              # %cond.load382
	vmv.x.s	a5, v24
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 512(sp)                    # 4-byte Folded Spill
	andi	a5, a4, 2
	beqz	a5, .LBB0_250
.LBB0_273:                              # %cond.load385
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a5, v8
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 516(sp)                    # 4-byte Folded Spill
	andi	a5, a4, 4
	beqz	a5, .LBB0_251
.LBB0_274:                              # %cond.load388
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a5, v8
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 520(sp)                    # 4-byte Folded Spill
	andi	a5, a4, 8
	beqz	a5, .LBB0_252
.LBB0_275:                              # %cond.load391
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a5, v8
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 524(sp)                    # 4-byte Folded Spill
	andi	a5, a4, 16
	beqz	a5, .LBB0_253
.LBB0_276:                              # %cond.load394
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 9
	addi	a5, a5, 896
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 480(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 528(sp)                    # 4-byte Folded Spill
	andi	a5, a4, 32
	beqz	a5, .LBB0_254
.LBB0_277:                              # %cond.load397
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 9
	addi	a5, a5, 768
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 360(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 532(sp)                    # 4-byte Folded Spill
	andi	a5, a4, 64
	beqz	a5, .LBB0_255
.LBB0_278:                              # %cond.load400
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 9
	addi	a5, a5, 640
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 240(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 536(sp)                    # 4-byte Folded Spill
	andi	a5, a4, 128
	beqz	a5, .LBB0_256
.LBB0_279:                              # %cond.load403
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 9
	addi	a5, a5, 512
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 120(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 540(sp)                    # 4-byte Folded Spill
	andi	a5, a4, 256
	beqz	a5, .LBB0_257
.LBB0_280:                              # %cond.load406
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 9
	addi	a5, a5, 384
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 0(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 544(sp)                    # 4-byte Folded Spill
	andi	a6, a4, 512
	lui	a5, 9
	addi	a5, a5, -1688
	add	a5, sp, a5
	beqz	a6, .LBB0_258
.LBB0_281:                              # %cond.load409
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, 256
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 2016(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 548(sp)                    # 4-byte Folded Spill
	andi	a6, a4, 1024
	beqz	a6, .LBB0_259
.LBB0_282:                              # %cond.load412
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, 128
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1896(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 552(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 52
	bgez	a6, .LBB0_260
.LBB0_283:                              # %cond.load415
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1776(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 556(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 51
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bgez	a6, .LBB0_261
.LBB0_284:                              # %cond.load418
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -128
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1656(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 560(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 50
	bltz	a6, .LBB0_262
.LBB0_285:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	a6, a4, 49
	vsext.vf2	v8, v16
	bgez	a6, .LBB0_287
.LBB0_286:                              # %cond.load424
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -384
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (a6)
	ld	a6, 1416(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 568(sp)                    # 4-byte Folded Spill
.LBB0_287:                              # %else425
	slli	a6, a4, 48
	csrr	a7, vlenb
	li	t0, 56
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	bltz	a6, .LBB0_306
# %bb.288:                              # %else428
	slli	a6, a4, 47
	vadd.vx	v24, v8, a0
	bltz	a6, .LBB0_307
.LBB0_289:                              # %else431
	slli	a6, a4, 46
	bltz	a6, .LBB0_308
.LBB0_290:                              # %else434
	slli	a6, a4, 45
	bltz	a6, .LBB0_309
.LBB0_291:                              # %else437
	slli	a6, a4, 44
	bltz	a6, .LBB0_310
.LBB0_292:                              # %else440
	slli	a6, a4, 43
	bltz	a6, .LBB0_311
.LBB0_293:                              # %else443
	slli	a6, a4, 42
	bltz	a6, .LBB0_312
.LBB0_294:                              # %else446
	slli	a6, a4, 41
	bltz	a6, .LBB0_313
.LBB0_295:                              # %else449
	slli	a6, a4, 40
	bltz	a6, .LBB0_314
.LBB0_296:                              # %else452
	slli	a6, a4, 39
	bgez	a6, .LBB0_298
.LBB0_297:                              # %cond.load454
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -1152
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 600(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 608(sp)                    # 4-byte Folded Spill
.LBB0_298:                              # %else455
	slli	a6, a4, 38
	csrr	a7, vlenb
	li	t0, 184
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	csrr	a7, vlenb
	li	t0, 192
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vl8r.v	v16, (a7)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	csrr	a7, vlenb
	li	t0, 120
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vs1r.v	v7, (a7)                        # vscale x 8-byte Folded Spill
	bgez	a6, .LBB0_300
# %bb.299:                              # %cond.load457
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -1280
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 480(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 612(sp)                    # 4-byte Folded Spill
.LBB0_300:                              # %else458
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetvli	zero, a1, e32, m8, ta, ma
	vand.vx	v16, v8, a3
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a6, a4, 37
	vadd.vv	v0, v8, v8
	bgez	a6, .LBB0_302
# %bb.301:                              # %cond.load460
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -1408
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 360(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 616(sp)                    # 4-byte Folded Spill
.LBB0_302:                              # %else461
	.loc	1 0 43                          # k135114449651120.py:0:43
	csrr	a6, vlenb
	li	a7, 184
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 12
	addi	a7, a7, 1120
	add	a6, a6, a7
	vl8r.v	v8, (a6)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a6, a4, 36
	vand.vx	v16, v0, a2
	bltz	a6, .LBB0_315
# %bb.303:                              # %else464
	slli	a6, a4, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v16, v8
	bltz	a6, .LBB0_316
.LBB0_304:                              # %else467
	slli	a6, a4, 34
	bgez	a6, .LBB0_317
.LBB0_305:                              # %cond.load469
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -1792
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a5, 0(a5)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 628(sp)                    # 4-byte Folded Spill
	vsext.vf2	v8, v16
	slli	a6, a4, 33
	lui	a5, 8
	addi	a5, a5, 296
	add	a5, sp, a5
	bltz	a6, .LBB0_318
	j	.LBB0_319
.LBB0_306:                              # %cond.load427
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -512
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (a6)
	ld	a6, 1296(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 572(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 47
	vadd.vx	v24, v8, a0
	bgez	a6, .LBB0_289
.LBB0_307:                              # %cond.load430
	vmv.x.s	a6, v24
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 576(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 46
	bgez	a6, .LBB0_290
.LBB0_308:                              # %cond.load433
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 580(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 45
	bgez	a6, .LBB0_291
.LBB0_309:                              # %cond.load436
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 584(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 44
	bgez	a6, .LBB0_292
.LBB0_310:                              # %cond.load439
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 588(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 43
	bgez	a6, .LBB0_293
.LBB0_311:                              # %cond.load442
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -640
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1080(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 592(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 42
	bgez	a6, .LBB0_294
.LBB0_312:                              # %cond.load445
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -768
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 960(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 596(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 41
	bgez	a6, .LBB0_295
.LBB0_313:                              # %cond.load448
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -896
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 840(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 600(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 40
	bgez	a6, .LBB0_296
.LBB0_314:                              # %cond.load451
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -1024
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 720(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 604(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 39
	bltz	a6, .LBB0_297
	j	.LBB0_298
.LBB0_315:                              # %cond.load463
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -1536
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 240(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 620(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v16, v8
	bgez	a6, .LBB0_304
.LBB0_316:                              # %cond.load466
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -1664
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 120(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 624(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 34
	bltz	a6, .LBB0_305
.LBB0_317:
	vsetivli	zero, 16, e64, m8, ta, ma
	vsext.vf2	v8, v16
	slli	a6, a4, 33
	lui	a5, 8
	addi	a5, a5, 296
	add	a5, sp, a5
	bgez	a6, .LBB0_319
.LBB0_318:                              # %cond.load472
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 9
	addi	a6, a6, -1920
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (a6)
	ld	a6, 1992(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 632(sp)                    # 4-byte Folded Spill
.LBB0_319:                              # %else473
	slli	a6, a4, 32
	csrr	a7, vlenb
	slli	a7, a7, 6
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	bltz	a6, .LBB0_335
# %bb.320:                              # %else476
	slli	a6, a4, 31
	vadd.vx	v24, v8, a0
	bltz	a6, .LBB0_336
.LBB0_321:                              # %else479
	slli	a6, a4, 30
	bltz	a6, .LBB0_337
.LBB0_322:                              # %else482
	slli	a6, a4, 29
	bltz	a6, .LBB0_338
.LBB0_323:                              # %else485
	slli	a6, a4, 28
	bltz	a6, .LBB0_339
.LBB0_324:                              # %else488
	slli	a6, a4, 27
	bltz	a6, .LBB0_340
.LBB0_325:                              # %else491
	slli	a6, a4, 26
	bltz	a6, .LBB0_341
.LBB0_326:                              # %else494
	slli	a6, a4, 25
	bltz	a6, .LBB0_342
.LBB0_327:                              # %else497
	slli	a6, a4, 24
	bltz	a6, .LBB0_343
.LBB0_328:                              # %else500
	slli	a6, a4, 23
	bltz	a6, .LBB0_344
.LBB0_329:                              # %else503
	slli	a6, a4, 22
	bltz	a6, .LBB0_345
.LBB0_330:                              # %else506
	slli	a6, a4, 21
	bltz	a6, .LBB0_346
.LBB0_331:                              # %else509
	slli	a6, a4, 20
	bltz	a6, .LBB0_347
.LBB0_332:                              # %else512
	slli	a6, a4, 19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bltz	a6, .LBB0_348
.LBB0_333:                              # %else515
	slli	a6, a4, 18
	bgez	a6, .LBB0_349
.LBB0_334:                              # %cond.load517
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 768
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 576(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 692(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 17
	vsext.vf2	v8, v16
	bltz	a6, .LBB0_350
	j	.LBB0_351
.LBB0_335:                              # %cond.load475
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	a6, 17
	slli	a6, a6, 11
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (a6)
	ld	a6, 1872(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 636(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 31
	vadd.vx	v24, v8, a0
	bgez	a6, .LBB0_321
.LBB0_336:                              # %cond.load478
	vmv.x.s	a6, v24
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 640(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 30
	bgez	a6, .LBB0_322
.LBB0_337:                              # %cond.load481
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 644(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 29
	bgez	a6, .LBB0_323
.LBB0_338:                              # %cond.load484
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 648(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 28
	bgez	a6, .LBB0_324
.LBB0_339:                              # %cond.load487
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 652(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 27
	bgez	a6, .LBB0_325
.LBB0_340:                              # %cond.load490
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 1920
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1656(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 656(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 26
	bgez	a6, .LBB0_326
.LBB0_341:                              # %cond.load493
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 1792
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1536(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 660(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 25
	bgez	a6, .LBB0_327
.LBB0_342:                              # %cond.load496
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 1664
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1416(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 664(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 24
	bgez	a6, .LBB0_328
.LBB0_343:                              # %cond.load499
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 1536
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1296(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 668(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 23
	bgez	a6, .LBB0_329
.LBB0_344:                              # %cond.load502
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 1408
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1176(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 672(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 22
	bgez	a6, .LBB0_330
.LBB0_345:                              # %cond.load505
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 1280
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1056(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 676(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 21
	bgez	a6, .LBB0_331
.LBB0_346:                              # %cond.load508
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 1152
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 936(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 680(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 20
	bgez	a6, .LBB0_332
.LBB0_347:                              # %cond.load511
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 1024
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 816(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 684(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bgez	a6, .LBB0_333
.LBB0_348:                              # %cond.load514
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 896
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 696(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 688(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 18
	bltz	a6, .LBB0_334
.LBB0_349:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	a6, a4, 17
	vsext.vf2	v8, v16
	bgez	a6, .LBB0_351
.LBB0_350:                              # %cond.load520
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 640
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (a6)
	ld	a6, 456(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 696(sp)                    # 4-byte Folded Spill
.LBB0_351:                              # %else521
	slli	a6, a4, 16
	csrr	a7, vlenb
	li	t0, 72
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	bltz	a6, .LBB0_372
# %bb.352:                              # %else524
	slli	a6, a4, 15
	vadd.vx	v24, v8, a0
	bltz	a6, .LBB0_373
.LBB0_353:                              # %else527
	slli	a6, a4, 14
	bltz	a6, .LBB0_374
.LBB0_354:                              # %else530
	slli	a6, a4, 13
	bltz	a6, .LBB0_375
.LBB0_355:                              # %else533
	slli	a6, a4, 12
	bltz	a6, .LBB0_376
.LBB0_356:                              # %else536
	slli	a6, a4, 11
	bltz	a6, .LBB0_377
.LBB0_357:                              # %else539
	slli	a6, a4, 10
	bltz	a6, .LBB0_378
.LBB0_358:                              # %else542
	slli	a5, a4, 9
	lui	a6, 8
	addi	a6, a6, -1840
	add	a6, sp, a6
	bltz	a5, .LBB0_379
.LBB0_359:                              # %else545
	slli	a5, a4, 8
	bltz	a5, .LBB0_380
.LBB0_360:                              # %else548
	slli	a5, a4, 7
	bgez	a5, .LBB0_362
.LBB0_361:                              # %cond.load550
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 8
	addi	a5, a5, -128
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 1776(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 736(sp)                    # 4-byte Folded Spill
.LBB0_362:                              # %else551
	slli	a5, a4, 6
	csrr	a7, vlenb
	li	t0, 168
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	csrr	a7, vlenb
	li	t0, 192
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vl8r.v	v16, (a7)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a5, .LBB0_364
# %bb.363:                              # %cond.load553
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 8
	addi	a5, a5, -256
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 1656(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 740(sp)                    # 4-byte Folded Spill
.LBB0_364:                              # %else554
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetvli	zero, a1, e32, m8, ta, ma
	vand.vx	v16, v8, a3
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a5, a4, 5
	vadd.vv	v0, v8, v8
	bgez	a5, .LBB0_366
# %bb.365:                              # %cond.load556
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 8
	addi	a5, a5, -384
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 1536(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 744(sp)                    # 4-byte Folded Spill
.LBB0_366:                              # %else557
	.loc	1 0 43                          # k135114449651120.py:0:43
	csrr	a5, vlenb
	li	a7, 168
	mul	a5, a5, a7
	add	a5, sp, a5
	lui	a7, 12
	addi	a7, a7, 1120
	add	a5, a5, a7
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a5, a4, 4
	vand.vx	v16, v0, a2
	bgez	a5, .LBB0_368
# %bb.367:                              # %cond.load559
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 8
	addi	a5, a5, -512
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 1416(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 748(sp)                    # 4-byte Folded Spill
.LBB0_368:                              # %else560
	.loc	1 0 43                          # k135114449651120.py:0:43
	csrr	a5, vlenb
	li	a7, 120
	mul	a5, a5, a7
	add	a5, sp, a5
	lui	a7, 12
	addi	a7, a7, 1120
	add	a5, a5, a7
	vl1r.v	v7, (a5)                        # vscale x 8-byte Folded Reload
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a5, a4, 3
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v16, v8
	bgez	a5, .LBB0_370
# %bb.369:                              # %cond.load562
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 8
	addi	a5, a5, -640
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 1296(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 752(sp)                    # 4-byte Folded Spill
.LBB0_370:                              # %else563
	slli	a5, a4, 2
	bgez	a5, .LBB0_381
# %bb.371:                              # %cond.load565
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 8
	addi	a5, a5, -768
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 1176(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 756(sp)                    # 4-byte Folded Spill
	vsext.vf2	v8, v16
	slli	a5, a4, 1
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v7, v7, 1
	bltz	a5, .LBB0_382
	j	.LBB0_383
.LBB0_372:                              # %cond.load523
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 512
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (a6)
	ld	a6, 336(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 700(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 15
	vadd.vx	v24, v8, a0
	bgez	a6, .LBB0_353
.LBB0_373:                              # %cond.load526
	vmv.x.s	a6, v24
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 704(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 14
	bgez	a6, .LBB0_354
.LBB0_374:                              # %cond.load529
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 708(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 13
	bgez	a6, .LBB0_355
.LBB0_375:                              # %cond.load532
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 712(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 12
	bgez	a6, .LBB0_356
.LBB0_376:                              # %cond.load535
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 716(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 11
	bgez	a6, .LBB0_357
.LBB0_377:                              # %cond.load538
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 384
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 120(a5)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 720(sp)                    # 4-byte Folded Spill
	slli	a6, a4, 10
	bgez	a6, .LBB0_358
.LBB0_378:                              # %cond.load541
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 8
	addi	a6, a6, 256
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a5, 0(a5)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 724(sp)                    # 4-byte Folded Spill
	slli	a5, a4, 9
	lui	a6, 8
	addi	a6, a6, -1840
	add	a6, sp, a6
	bgez	a5, .LBB0_359
.LBB0_379:                              # %cond.load544
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 8
	addi	a5, a5, 128
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 2016(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 728(sp)                    # 4-byte Folded Spill
	slli	a5, a4, 8
	bgez	a5, .LBB0_360
.LBB0_380:                              # %cond.load547
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 8
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 1896(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 732(sp)                    # 4-byte Folded Spill
	slli	a5, a4, 7
	bltz	a5, .LBB0_361
	j	.LBB0_362
.LBB0_381:
	vsetivli	zero, 16, e64, m8, ta, ma
	vsext.vf2	v8, v16
	slli	a5, a4, 1
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v7, v7, 1
	bgez	a5, .LBB0_383
.LBB0_382:                              # %cond.load568
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a5, 8
	addi	a5, a5, -896
	add	a5, sp, a5
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a5)
	ld	a5, 1056(a6)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 760(sp)                    # 4-byte Folded Spill
.LBB0_383:                              # %else569
	.loc	1 0 43                          # k135114449651120.py:0:43
	csrr	a5, vlenb
	li	a7, 80
	mul	a5, a5, a7
	add	a5, sp, a5
	lui	a7, 12
	addi	a7, a7, 1120
	add	a5, a5, a7
	vs8r.v	v8, (a5)                        # vscale x 64-byte Folded Spill
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	.loc	1 9 43                          # k135114449651120.py:9:43
	vmv.x.s	a5, v7
	bltz	a4, .LBB0_399
# %bb.384:                              # %else572
	andi	a4, a5, 1
	vadd.vx	v24, v8, a0
	bnez	a4, .LBB0_400
.LBB0_385:                              # %else575
	andi	a4, a5, 2
	bnez	a4, .LBB0_401
.LBB0_386:                              # %else578
	andi	a4, a5, 4
	bnez	a4, .LBB0_402
.LBB0_387:                              # %else581
	andi	a4, a5, 8
	bnez	a4, .LBB0_403
.LBB0_388:                              # %else584
	andi	a4, a5, 16
	bnez	a4, .LBB0_404
.LBB0_389:                              # %else587
	andi	a4, a5, 32
	bnez	a4, .LBB0_405
.LBB0_390:                              # %else590
	andi	a4, a5, 64
	bnez	a4, .LBB0_406
.LBB0_391:                              # %else593
	andi	a4, a5, 128
	bnez	a4, .LBB0_407
.LBB0_392:                              # %else596
	andi	a4, a5, 256
	bnez	a4, .LBB0_408
.LBB0_393:                              # %else599
	andi	a4, a5, 512
	bnez	a4, .LBB0_409
.LBB0_394:                              # %else602
	andi	a4, a5, 1024
	bnez	a4, .LBB0_410
.LBB0_395:                              # %else605
	slli	a6, a5, 52
	lui	a4, 7
	addi	a4, a4, 120
	add	a4, sp, a4
	bltz	a6, .LBB0_411
.LBB0_396:                              # %else608
	slli	a6, a5, 51
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bltz	a6, .LBB0_412
.LBB0_397:                              # %else611
	slli	a6, a5, 50
	bgez	a6, .LBB0_413
.LBB0_398:                              # %cond.load613
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 7
	addi	a6, a6, 1792
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1776(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 820(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 49
	vsext.vf2	v8, v16
	bltz	a6, .LBB0_414
	j	.LBB0_415
.LBB0_399:                              # %cond.load571
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	a4, 31
	slli	a4, a4, 10
	add	a4, sp, a4
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (a4)
	ld	a4, 936(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 764(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 1
	vadd.vx	v24, v8, a0
	beqz	a4, .LBB0_385
.LBB0_400:                              # %cond.load574
	vmv.x.s	a4, v24
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 768(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 2
	beqz	a4, .LBB0_386
.LBB0_401:                              # %cond.load577
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a4, v8
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 772(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 4
	beqz	a4, .LBB0_387
.LBB0_402:                              # %cond.load580
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a4, v8
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 776(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 8
	beqz	a4, .LBB0_388
.LBB0_403:                              # %cond.load583
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a4, v8
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 780(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 16
	beqz	a4, .LBB0_389
.LBB0_404:                              # %cond.load586
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a4, 8
	addi	a4, a4, -1152
	add	a4, sp, a4
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 720(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 784(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 32
	beqz	a4, .LBB0_390
.LBB0_405:                              # %cond.load589
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a4, 8
	addi	a4, a4, -1280
	add	a4, sp, a4
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 600(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 788(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 64
	beqz	a4, .LBB0_391
.LBB0_406:                              # %cond.load592
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a4, 8
	addi	a4, a4, -1408
	add	a4, sp, a4
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 480(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 792(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 128
	beqz	a4, .LBB0_392
.LBB0_407:                              # %cond.load595
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a4, 8
	addi	a4, a4, -1536
	add	a4, sp, a4
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 360(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 796(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 256
	beqz	a4, .LBB0_393
.LBB0_408:                              # %cond.load598
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a4, 8
	addi	a4, a4, -1664
	add	a4, sp, a4
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 240(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 800(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 512
	beqz	a4, .LBB0_394
.LBB0_409:                              # %cond.load601
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a4, 8
	addi	a4, a4, -1792
	add	a4, sp, a4
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 120(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 804(sp)                    # 4-byte Folded Spill
	andi	a4, a5, 1024
	beqz	a4, .LBB0_395
.LBB0_410:                              # %cond.load604
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a4, 8
	addi	a4, a4, -1920
	add	a4, sp, a4
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 0(a6)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 808(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 52
	lui	a4, 7
	addi	a4, a4, 120
	add	a4, sp, a4
	bgez	a6, .LBB0_396
.LBB0_411:                              # %cond.load607
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	a6, 15
	slli	a6, a6, 11
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 2016(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 812(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 51
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bgez	a6, .LBB0_397
.LBB0_412:                              # %cond.load610
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 7
	addi	a6, a6, 1920
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	ld	a6, 1896(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 816(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 50
	bltz	a6, .LBB0_398
.LBB0_413:
	vsetivli	zero, 16, e64, m8, ta, ma
	slli	a6, a5, 49
	vsext.vf2	v8, v16
	bgez	a6, .LBB0_415
.LBB0_414:                              # %cond.load616
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 7
	addi	a6, a6, 1664
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (a6)
	ld	a6, 1656(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 824(sp)                    # 4-byte Folded Spill
.LBB0_415:                              # %else617
	slli	a6, a5, 48
	csrr	a7, vlenb
	li	t0, 88
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v8
	bltz	a6, .LBB0_434
# %bb.416:                              # %else620
	slli	a6, a5, 47
	vadd.vx	v16, v8, a0
	bltz	a6, .LBB0_435
.LBB0_417:                              # %else623
	slli	a6, a5, 46
	bltz	a6, .LBB0_436
.LBB0_418:                              # %else626
	slli	a6, a5, 45
	bltz	a6, .LBB0_437
.LBB0_419:                              # %else629
	slli	a6, a5, 44
	bltz	a6, .LBB0_438
.LBB0_420:                              # %else632
	slli	a6, a5, 43
	bltz	a6, .LBB0_439
.LBB0_421:                              # %else635
	slli	a6, a5, 42
	bltz	a6, .LBB0_440
.LBB0_422:                              # %else638
	slli	a6, a5, 41
	bltz	a6, .LBB0_441
.LBB0_423:                              # %else641
	slli	a6, a5, 40
	bltz	a6, .LBB0_442
.LBB0_424:                              # %else644
	slli	a6, a5, 39
	bgez	a6, .LBB0_426
.LBB0_425:                              # %cond.load646
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 7
	addi	a6, a6, 896
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a6)
	ld	a6, 840(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 864(sp)                    # 4-byte Folded Spill
.LBB0_426:                              # %else647
	slli	a6, a5, 38
	csrr	a7, vlenb
	li	t0, 160
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	csrr	a7, vlenb
	li	t0, 192
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 12
	addi	t0, t0, 1120
	add	a7, a7, t0
	vl8r.v	v24, (a7)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v8, v24
	bgez	a6, .LBB0_428
# %bb.427:                              # %cond.load649
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 7
	addi	a6, a6, 768
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a6)
	ld	a6, 720(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 868(sp)                    # 4-byte Folded Spill
.LBB0_428:                              # %else650
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetvli	zero, a1, e32, m8, ta, ma
	vand.vx	v8, v24, a3
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a3, a5, 37
	vadd.vv	v24, v24, v24
	bgez	a3, .LBB0_430
# %bb.429:                              # %cond.load652
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a3, 7
	addi	a3, a3, 640
	add	a3, sp, a3
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a3)
	ld	a3, 600(a4)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 872(sp)                    # 4-byte Folded Spill
.LBB0_430:                              # %else653
	.loc	1 0 43                          # k135114449651120.py:0:43
	vsetvli	zero, a1, e32, m8, ta, ma
	vand.vx	v24, v24, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	slli	a2, a5, 36
	csrr	a3, vlenb
	li	a6, 160
	mul	a3, a3, a6
	add	a3, sp, a3
	lui	a6, 12
	addi	a6, a6, 1120
	add	a3, a3, a6
	vl8r.v	v0, (a3)                        # vscale x 64-byte Folded Reload
	vsub.vv	v8, v0, v8
	bltz	a2, .LBB0_443
# %bb.431:                              # %else656
	slli	a2, a5, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v24, v8
	bltz	a2, .LBB0_444
.LBB0_432:                              # %else659
	slli	a1, a5, 34
	bgez	a1, .LBB0_445
.LBB0_433:                              # %cond.load661
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a1, 7
	addi	a1, a1, 256
	add	a1, sp, a1
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a1)
	ld	a1, 240(a4)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 884(sp)                    # 4-byte Folded Spill
	j	.LBB0_446
.LBB0_434:                              # %cond.load619
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 7
	addi	a6, a6, 1536
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v24, (a6)
	ld	a6, 1536(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 828(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 47
	vadd.vx	v16, v8, a0
	bgez	a6, .LBB0_417
.LBB0_435:                              # %cond.load622
	vmv.x.s	a6, v16
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 832(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 46
	bgez	a6, .LBB0_418
.LBB0_436:                              # %cond.load625
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v16, 1
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 836(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 45
	bgez	a6, .LBB0_419
.LBB0_437:                              # %cond.load628
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 2
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 840(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 44
	bgez	a6, .LBB0_420
.LBB0_438:                              # %cond.load631
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 3
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 844(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 43
	bgez	a6, .LBB0_421
.LBB0_439:                              # %cond.load634
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 7
	addi	a6, a6, 1408
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a6)
	ld	a6, 1320(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 848(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 42
	bgez	a6, .LBB0_422
.LBB0_440:                              # %cond.load637
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 7
	addi	a6, a6, 1280
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a6)
	ld	a6, 1200(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 852(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 41
	bgez	a6, .LBB0_423
.LBB0_441:                              # %cond.load640
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a6, 7
	addi	a6, a6, 1152
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a6)
	ld	a6, 1080(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 856(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 40
	bgez	a6, .LBB0_424
.LBB0_442:                              # %cond.load643
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	a6, 29
	slli	a6, a6, 10
	add	a6, sp, a6
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a6)
	ld	a6, 960(a4)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa5, a6
	fsw	fa5, 860(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 39
	bltz	a6, .LBB0_425
	j	.LBB0_426
.LBB0_443:                              # %cond.load655
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, 512
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 480(a4)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 876(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v24, v8
	bgez	a2, .LBB0_432
.LBB0_444:                              # %cond.load658
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a1, 7
	addi	a1, a1, 384
	add	a1, sp, a1
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a1)
	ld	a1, 360(a4)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 880(sp)                    # 4-byte Folded Spill
	slli	a1, a5, 34
	bltz	a1, .LBB0_433
.LBB0_445:
	vsetivli	zero, 16, e64, m8, ta, ma
.LBB0_446:                              # %else662
	slli	a1, a5, 33
	vsext.vf2	v24, v8
	csrr	a2, vlenb
	li	a3, 120
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 12
	addi	a3, a3, 1120
	add	a2, a2, a3
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	bgez	a1, .LBB0_448
# %bb.447:                              # %cond.load664
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a1, 7
	addi	a1, a1, 128
	add	a1, sp, a1
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v16, (a1)
	ld	a1, 120(a4)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 888(sp)                    # 4-byte Folded Spill
.LBB0_448:                              # %else665
	slli	a1, a5, 32
	csrr	a2, vlenb
	li	a3, 120
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 12
	addi	a3, a3, 1120
	add	a2, a2, a3
	vl8r.v	v24, (a2)                       # vscale x 64-byte Folded Reload
	vadd.vv	v24, v24, v24
	bltz	a1, .LBB0_464
# %bb.449:                              # %else668
	slli	a1, a5, 31
	vadd.vx	v16, v24, a0
	bltz	a1, .LBB0_465
.LBB0_450:                              # %else671
	slli	a1, a5, 30
	bltz	a1, .LBB0_466
.LBB0_451:                              # %else674
	slli	a1, a5, 29
	bltz	a1, .LBB0_467
.LBB0_452:                              # %else677
	slli	a1, a5, 28
	bltz	a1, .LBB0_468
.LBB0_453:                              # %else680
	slli	a2, a5, 27
	lui	a1, 6
	addi	a1, a1, 1984
	add	a1, sp, a1
	bltz	a2, .LBB0_469
.LBB0_454:                              # %else683
	slli	a2, a5, 26
	bltz	a2, .LBB0_470
.LBB0_455:                              # %else686
	slli	a2, a5, 25
	bltz	a2, .LBB0_471
.LBB0_456:                              # %else689
	slli	a2, a5, 24
	bltz	a2, .LBB0_472
.LBB0_457:                              # %else692
	slli	a2, a5, 23
	bltz	a2, .LBB0_473
.LBB0_458:                              # %else695
	slli	a2, a5, 22
	bltz	a2, .LBB0_474
.LBB0_459:                              # %else698
	slli	a2, a5, 21
	bltz	a2, .LBB0_475
.LBB0_460:                              # %else701
	slli	a2, a5, 20
	bltz	a2, .LBB0_476
.LBB0_461:                              # %else704
	slli	a2, a5, 19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	bltz	a2, .LBB0_477
.LBB0_462:                              # %else707
	slli	a2, a5, 18
	bgez	a2, .LBB0_478
.LBB0_463:                              # %cond.load709
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, -1280
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 936(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 948(sp)                    # 4-byte Folded Spill
	j	.LBB0_479
.LBB0_464:                              # %cond.load667
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a1, 7
	add	a1, sp, a1
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v16, (a1)
	ld	a1, 0(a4)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 892(sp)                    # 4-byte Folded Spill
	slli	a1, a5, 31
	vadd.vx	v16, v24, a0
	bgez	a1, .LBB0_450
.LBB0_465:                              # %cond.load670
	vmv.x.s	a1, v16
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 896(sp)                    # 4-byte Folded Spill
	slli	a1, a5, 30
	bgez	a1, .LBB0_451
.LBB0_466:                              # %cond.load673
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v16, 1
	vmv.x.s	a1, v24
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 900(sp)                    # 4-byte Folded Spill
	slli	a1, a5, 29
	bgez	a1, .LBB0_452
.LBB0_467:                              # %cond.load676
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v16, 2
	vmv.x.s	a1, v24
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 904(sp)                    # 4-byte Folded Spill
	slli	a1, a5, 28
	bgez	a1, .LBB0_453
.LBB0_468:                              # %cond.load679
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v16, 3
	vmv.x.s	a1, v24
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 908(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 27
	lui	a1, 6
	addi	a1, a1, 1984
	add	a1, sp, a1
	bgez	a2, .LBB0_454
.LBB0_469:                              # %cond.load682
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, -128
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 2016(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 912(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 26
	bgez	a2, .LBB0_455
.LBB0_470:                              # %cond.load685
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, -256
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 1896(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 916(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 25
	bgez	a2, .LBB0_456
.LBB0_471:                              # %cond.load688
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, -384
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 1776(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 920(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 24
	bgez	a2, .LBB0_457
.LBB0_472:                              # %cond.load691
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, -512
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 1656(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 924(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 23
	bgez	a2, .LBB0_458
.LBB0_473:                              # %cond.load694
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, -640
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 1536(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 928(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 22
	bgez	a2, .LBB0_459
.LBB0_474:                              # %cond.load697
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, -768
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 1416(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 932(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 21
	bgez	a2, .LBB0_460
.LBB0_475:                              # %cond.load700
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, -896
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 1296(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 936(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 20
	bgez	a2, .LBB0_461
.LBB0_476:                              # %cond.load703
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	a2, 27
	slli	a2, a2, 10
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 1176(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 940(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	bgez	a2, .LBB0_462
.LBB0_477:                              # %cond.load706
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, -1152
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 1056(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 944(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 18
	bltz	a2, .LBB0_463
.LBB0_478:
	vsetivli	zero, 16, e64, m8, ta, ma
.LBB0_479:                              # %else710
	slli	a2, a5, 17
	vsext.vf2	v24, v8
	csrr	a3, vlenb
	slli	a3, a3, 7
	add	a3, sp, a3
	lui	a4, 12
	addi	a4, a4, 1120
	add	a3, a3, a4
	vs8r.v	v24, (a3)                       # vscale x 64-byte Folded Spill
	bgez	a2, .LBB0_481
# %bb.480:                              # %cond.load712
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a2, 7
	addi	a2, a2, -1408
	add	a2, sp, a2
	.loc	1 9 43                          # k135114449651120.py:9:43
	vse64.v	v16, (a2)
	ld	a2, 816(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 952(sp)                    # 4-byte Folded Spill
.LBB0_481:                              # %else713
	slli	a2, a5, 16
	csrr	a3, vlenb
	slli	a3, a3, 7
	add	a3, sp, a3
	lui	a4, 12
	addi	a4, a4, 1120
	add	a3, a3, a4
	vl8r.v	v8, (a3)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v8, v8
	bgez	a2, .LBB0_482
	j	.LBB0_792
.LBB0_482:                              # %else716
	slli	a2, a5, 15
	vadd.vx	v8, v8, a0
	bgez	a2, .LBB0_483
	j	.LBB0_793
.LBB0_483:                              # %else719
	slli	a0, a5, 14
	bgez	a0, .LBB0_484
	j	.LBB0_794
.LBB0_484:                              # %else722
	slli	a0, a5, 13
	bgez	a0, .LBB0_485
	j	.LBB0_795
.LBB0_485:                              # %else725
	slli	a0, a5, 12
	bgez	a0, .LBB0_486
	j	.LBB0_796
.LBB0_486:                              # %else728
	slli	a0, a5, 11
	bgez	a0, .LBB0_487
	j	.LBB0_797
.LBB0_487:                              # %else731
	slli	a0, a5, 10
	bgez	a0, .LBB0_488
	j	.LBB0_798
.LBB0_488:                              # %else734
	slli	a0, a5, 9
	bgez	a0, .LBB0_489
	j	.LBB0_799
.LBB0_489:                              # %else737
	slli	a0, a5, 8
	bgez	a0, .LBB0_490
	j	.LBB0_800
.LBB0_490:                              # %else740
	slli	a0, a5, 7
	bgez	a0, .LBB0_491
	j	.LBB0_801
.LBB0_491:                              # %else743
	slli	a0, a5, 6
	lui	a1, 6
	addi	a1, a1, -152
	add	s5, sp, a1
	bgez	a0, .LBB0_492
	j	.LBB0_802
.LBB0_492:                              # %else746
	slli	a0, a5, 5
	bgez	a0, .LBB0_493
	j	.LBB0_803
.LBB0_493:                              # %else749
	slli	a0, a5, 4
	bgez	a0, .LBB0_494
	j	.LBB0_804
.LBB0_494:                              # %else752
	slli	a0, a5, 3
	bgez	a0, .LBB0_495
	j	.LBB0_805
.LBB0_495:                              # %else755
	slli	a0, a5, 2
	bgez	a0, .LBB0_496
	j	.LBB0_806
.LBB0_496:                              # %else758
	slli	a0, a5, 1
	bgez	a0, .LBB0_497
	j	.LBB0_807
.LBB0_497:                              # %else761
	bgez	a5, .LBB0_499
.LBB0_498:                              # %cond.load763
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	a0, 25
	slli	a0, a0, 10
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 1020(sp)                   # 4-byte Folded Spill
.LBB0_499:                              # %else764
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	s4, 32
	li	s3, 448
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 6 21 is_stmt 1                # k135114449651120.py:6:21
	vsetvli	zero, s4, e32, m8, ta, ma
	vmslt.vx	v8, v16, s3
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v9, v16, s3
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v10, v16, s3
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v24, v16, s3
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	vslideup.vi	v24, v10, 4
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v24, v9, 8
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs1r.v	v24, (a0)                       # vscale x 8-byte Folded Spill
	vmv.x.s	s6, v24
	andi	a0, s6, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_500
	j	.LBB0_808
.LBB0_500:                              # %else768
	andi	a0, s6, 2
	beqz	a0, .LBB0_501
	j	.LBB0_809
.LBB0_501:                              # %else771
	andi	a0, s6, 4
	beqz	a0, .LBB0_502
	j	.LBB0_810
.LBB0_502:                              # %else774
	andi	a0, s6, 8
	beqz	a0, .LBB0_503
	j	.LBB0_811
.LBB0_503:                              # %else777
	andi	a0, s6, 16
	beqz	a0, .LBB0_504
	j	.LBB0_812
.LBB0_504:                              # %else780
	andi	a0, s6, 32
	beqz	a0, .LBB0_505
	j	.LBB0_813
.LBB0_505:                              # %else783
	andi	a0, s6, 64
	beqz	a0, .LBB0_506
	j	.LBB0_814
.LBB0_506:                              # %else786
	andi	a0, s6, 128
	beqz	a0, .LBB0_507
	j	.LBB0_815
.LBB0_507:                              # %else789
	andi	a0, s6, 256
	beqz	a0, .LBB0_508
	j	.LBB0_816
.LBB0_508:                              # %else792
	andi	a0, s6, 512
	beqz	a0, .LBB0_509
	j	.LBB0_817
.LBB0_509:                              # %else795
	andi	a0, s6, 1024
	beqz	a0, .LBB0_510
	j	.LBB0_818
.LBB0_510:                              # %else798
	slli	a0, s6, 52
	bgez	a0, .LBB0_511
	j	.LBB0_819
.LBB0_511:                              # %else801
	slli	a0, s6, 51
	bgez	a0, .LBB0_512
	j	.LBB0_820
.LBB0_512:                              # %else804
	slli	a0, s6, 50
	bgez	a0, .LBB0_513
	j	.LBB0_821
.LBB0_513:                              # %else807
	slli	a0, s6, 49
	lui	a1, 5
	addi	a1, a1, 1832
	add	s5, sp, a1
	bgez	a0, .LBB0_515
.LBB0_514:                              # %cond.store808
	.loc	1 0 44 is_stmt 0                # k135114449651120.py:0:44
	flw	fa0, 56(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1992(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_515:                              # %else810
	slli	a0, s6, 48
	lui	a1, 12
	addi	a1, a1, 1120
	add	a1, sp, a1
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_517
# %bb.516:                              # %cond.store811
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 60(sp)                     # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -512
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1872(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_517:                              # %else813
	slli	a0, s6, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_518
	j	.LBB0_822
.LBB0_518:                              # %else816
	slli	a0, s6, 46
	bgez	a0, .LBB0_519
	j	.LBB0_823
.LBB0_519:                              # %else819
	slli	a0, s6, 45
	bgez	a0, .LBB0_520
	j	.LBB0_824
.LBB0_520:                              # %else822
	slli	a0, s6, 44
	bgez	a0, .LBB0_521
	j	.LBB0_825
.LBB0_521:                              # %else825
	slli	a0, s6, 43
	bgez	a0, .LBB0_522
	j	.LBB0_826
.LBB0_522:                              # %else828
	slli	a0, s6, 42
	bgez	a0, .LBB0_523
	j	.LBB0_827
.LBB0_523:                              # %else831
	slli	a0, s6, 41
	bgez	a0, .LBB0_524
	j	.LBB0_828
.LBB0_524:                              # %else834
	slli	a0, s6, 40
	bgez	a0, .LBB0_525
	j	.LBB0_829
.LBB0_525:                              # %else837
	slli	a0, s6, 39
	bgez	a0, .LBB0_526
	j	.LBB0_830
.LBB0_526:                              # %else840
	slli	a0, s6, 38
	bgez	a0, .LBB0_527
	j	.LBB0_831
.LBB0_527:                              # %else843
	slli	a0, s6, 37
	bgez	a0, .LBB0_528
	j	.LBB0_832
.LBB0_528:                              # %else846
	slli	a0, s6, 36
	bgez	a0, .LBB0_529
	j	.LBB0_833
.LBB0_529:                              # %else849
	slli	a0, s6, 35
	bgez	a0, .LBB0_530
	j	.LBB0_834
.LBB0_530:                              # %else852
	slli	a0, s6, 34
	bgez	a0, .LBB0_531
	j	.LBB0_835
.LBB0_531:                              # %else855
	slli	a0, s6, 33
	bgez	a0, .LBB0_533
.LBB0_532:                              # %cond.store856
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 120(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 456(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_533:                              # %else858
	slli	a0, s6, 32
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_535
# %bb.534:                              # %cond.store859
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 124(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 336(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_535:                              # %else861
	slli	a0, s6, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_536
	j	.LBB0_836
.LBB0_536:                              # %else864
	slli	a0, s6, 30
	bgez	a0, .LBB0_537
	j	.LBB0_837
.LBB0_537:                              # %else867
	slli	a0, s6, 29
	bgez	a0, .LBB0_538
	j	.LBB0_838
.LBB0_538:                              # %else870
	slli	a0, s6, 28
	bgez	a0, .LBB0_539
	j	.LBB0_839
.LBB0_539:                              # %else873
	slli	a0, s6, 27
	bgez	a0, .LBB0_540
	j	.LBB0_840
.LBB0_540:                              # %else876
	slli	a0, s6, 26
	bgez	a0, .LBB0_541
	j	.LBB0_841
.LBB0_541:                              # %else879
	slli	a0, s6, 25
	lui	a1, 5
	addi	a1, a1, -304
	add	s5, sp, a1
	bgez	a0, .LBB0_542
	j	.LBB0_842
.LBB0_542:                              # %else882
	slli	a0, s6, 24
	bgez	a0, .LBB0_543
	j	.LBB0_843
.LBB0_543:                              # %else885
	slli	a0, s6, 23
	bgez	a0, .LBB0_544
	j	.LBB0_844
.LBB0_544:                              # %else888
	slli	a0, s6, 22
	bgez	a0, .LBB0_545
	j	.LBB0_845
.LBB0_545:                              # %else891
	slli	a0, s6, 21
	bgez	a0, .LBB0_546
	j	.LBB0_846
.LBB0_546:                              # %else894
	slli	a0, s6, 20
	bgez	a0, .LBB0_547
	j	.LBB0_847
.LBB0_547:                              # %else897
	slli	a0, s6, 19
	bgez	a0, .LBB0_548
	j	.LBB0_848
.LBB0_548:                              # %else900
	slli	a0, s6, 18
	bgez	a0, .LBB0_549
	j	.LBB0_849
.LBB0_549:                              # %else903
	slli	a0, s6, 17
	bgez	a0, .LBB0_551
.LBB0_550:                              # %cond.store904
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 184(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_551:                              # %else906
	slli	a0, s6, 16
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_553
# %bb.552:                              # %cond.store907
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 188(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 512
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 936(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_553:                              # %else909
	slli	a0, s6, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_554
	j	.LBB0_850
.LBB0_554:                              # %else912
	slli	a0, s6, 14
	bgez	a0, .LBB0_555
	j	.LBB0_851
.LBB0_555:                              # %else915
	slli	a0, s6, 13
	bgez	a0, .LBB0_556
	j	.LBB0_852
.LBB0_556:                              # %else918
	slli	a0, s6, 12
	bgez	a0, .LBB0_557
	j	.LBB0_853
.LBB0_557:                              # %else921
	slli	a0, s6, 11
	bgez	a0, .LBB0_558
	j	.LBB0_854
.LBB0_558:                              # %else924
	slli	a0, s6, 10
	bgez	a0, .LBB0_559
	j	.LBB0_855
.LBB0_559:                              # %else927
	slli	a0, s6, 9
	bgez	a0, .LBB0_560
	j	.LBB0_856
.LBB0_560:                              # %else930
	slli	a0, s6, 8
	bgez	a0, .LBB0_561
	j	.LBB0_857
.LBB0_561:                              # %else933
	slli	a0, s6, 7
	bgez	a0, .LBB0_562
	j	.LBB0_858
.LBB0_562:                              # %else936
	slli	a0, s6, 6
	bgez	a0, .LBB0_563
	j	.LBB0_859
.LBB0_563:                              # %else939
	slli	a0, s6, 5
	bgez	a0, .LBB0_564
	j	.LBB0_860
.LBB0_564:                              # %else942
	slli	a0, s6, 4
	lui	a1, 4
	addi	a1, a1, 1656
	add	s7, sp, a1
	bgez	a0, .LBB0_565
	j	.LBB0_861
.LBB0_565:                              # %else945
	slli	a0, s6, 3
	bgez	a0, .LBB0_566
	j	.LBB0_862
.LBB0_566:                              # %else948
	slli	a0, s6, 2
	bgez	a0, .LBB0_568
.LBB0_567:                              # %cond.store949
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 244(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_568:                              # %else951
	slli	a0, s6, 1
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl1r.v	v8, (a1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	bgez	a0, .LBB0_570
# %bb.569:                              # %cond.store952
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 248(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs1r.v	v16, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl1r.v	v16, (a0)                       # vscale x 8-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_570:                              # %else954
	.loc	1 0 44                          # k135114449651120.py:0:44
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	.loc	1 10 44                         # k135114449651120.py:10:44
	vmv.x.s	s5, v16
	bgez	s6, .LBB0_572
# %bb.571:                              # %cond.store955
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 252(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_572:                              # %else957
	andi	a0, s5, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_573
	j	.LBB0_863
.LBB0_573:                              # %else960
	andi	a0, s5, 2
	beqz	a0, .LBB0_574
	j	.LBB0_864
.LBB0_574:                              # %else963
	andi	a0, s5, 4
	beqz	a0, .LBB0_575
	j	.LBB0_865
.LBB0_575:                              # %else966
	andi	a0, s5, 8
	beqz	a0, .LBB0_576
	j	.LBB0_866
.LBB0_576:                              # %else969
	andi	a0, s5, 16
	beqz	a0, .LBB0_577
	j	.LBB0_867
.LBB0_577:                              # %else972
	andi	a0, s5, 32
	beqz	a0, .LBB0_578
	j	.LBB0_868
.LBB0_578:                              # %else975
	andi	a0, s5, 64
	beqz	a0, .LBB0_579
	j	.LBB0_869
.LBB0_579:                              # %else978
	andi	a0, s5, 128
	beqz	a0, .LBB0_580
	j	.LBB0_870
.LBB0_580:                              # %else981
	andi	a0, s5, 256
	beqz	a0, .LBB0_581
	j	.LBB0_871
.LBB0_581:                              # %else984
	andi	a0, s5, 512
	beqz	a0, .LBB0_582
	j	.LBB0_872
.LBB0_582:                              # %else987
	andi	a0, s5, 1024
	beqz	a0, .LBB0_583
	j	.LBB0_873
.LBB0_583:                              # %else990
	slli	a0, s5, 52
	bgez	a0, .LBB0_584
	j	.LBB0_874
.LBB0_584:                              # %else993
	slli	a0, s5, 51
	bgez	a0, .LBB0_585
	j	.LBB0_875
.LBB0_585:                              # %else996
	slli	a0, s5, 50
	bgez	a0, .LBB0_586
	j	.LBB0_876
.LBB0_586:                              # %else999
	slli	a0, s5, 49
	bgez	a0, .LBB0_588
.LBB0_587:                              # %cond.store1000
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 312(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_588:                              # %else1002
	slli	a0, s5, 48
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_590
# %bb.589:                              # %cond.store1003
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 316(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 1536
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_590:                              # %else1005
	slli	a0, s5, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_591
	j	.LBB0_877
.LBB0_591:                              # %else1008
	slli	a0, s5, 46
	bgez	a0, .LBB0_592
	j	.LBB0_878
.LBB0_592:                              # %else1011
	slli	a0, s5, 45
	bgez	a0, .LBB0_593
	j	.LBB0_879
.LBB0_593:                              # %else1014
	slli	a0, s5, 44
	bgez	a0, .LBB0_594
	j	.LBB0_880
.LBB0_594:                              # %else1017
	slli	a0, s5, 43
	lui	a1, 4
	addi	a1, a1, -576
	add	s6, sp, a1
	bgez	a0, .LBB0_595
	j	.LBB0_881
.LBB0_595:                              # %else1020
	slli	a0, s5, 42
	bgez	a0, .LBB0_596
	j	.LBB0_882
.LBB0_596:                              # %else1023
	slli	a0, s5, 41
	bgez	a0, .LBB0_597
	j	.LBB0_883
.LBB0_597:                              # %else1026
	slli	a0, s5, 40
	bgez	a0, .LBB0_598
	j	.LBB0_884
.LBB0_598:                              # %else1029
	slli	a0, s5, 39
	bgez	a0, .LBB0_599
	j	.LBB0_885
.LBB0_599:                              # %else1032
	slli	a0, s5, 38
	bgez	a0, .LBB0_600
	j	.LBB0_886
.LBB0_600:                              # %else1035
	slli	a0, s5, 37
	bgez	a0, .LBB0_601
	j	.LBB0_887
.LBB0_601:                              # %else1038
	slli	a0, s5, 36
	bgez	a0, .LBB0_602
	j	.LBB0_888
.LBB0_602:                              # %else1041
	slli	a0, s5, 35
	bgez	a0, .LBB0_603
	j	.LBB0_889
.LBB0_603:                              # %else1044
	slli	a0, s5, 34
	bgez	a0, .LBB0_604
	j	.LBB0_890
.LBB0_604:                              # %else1047
	slli	a0, s5, 33
	bgez	a0, .LBB0_606
.LBB0_605:                              # %cond.store1048
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 376(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_606:                              # %else1050
	slli	a0, s5, 32
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_608
# %bb.607:                              # %cond.store1051
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 380(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 696(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_608:                              # %else1053
	slli	a0, s5, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_609
	j	.LBB0_891
.LBB0_609:                              # %else1056
	slli	a0, s5, 30
	bgez	a0, .LBB0_610
	j	.LBB0_892
.LBB0_610:                              # %else1059
	slli	a0, s5, 29
	bgez	a0, .LBB0_611
	j	.LBB0_893
.LBB0_611:                              # %else1062
	slli	a0, s5, 28
	bgez	a0, .LBB0_612
	j	.LBB0_894
.LBB0_612:                              # %else1065
	slli	a0, s5, 27
	bgez	a0, .LBB0_613
	j	.LBB0_895
.LBB0_613:                              # %else1068
	slli	a0, s5, 26
	bgez	a0, .LBB0_614
	j	.LBB0_896
.LBB0_614:                              # %else1071
	slli	a0, s5, 25
	bgez	a0, .LBB0_615
	j	.LBB0_897
.LBB0_615:                              # %else1074
	slli	a0, s5, 24
	bgez	a0, .LBB0_616
	j	.LBB0_898
.LBB0_616:                              # %else1077
	slli	a0, s5, 23
	bgez	a0, .LBB0_617
	j	.LBB0_899
.LBB0_617:                              # %else1080
	slli	a0, s5, 22
	lui	a1, 3
	addi	a1, a1, 1384
	add	s6, sp, a1
	bgez	a0, .LBB0_618
	j	.LBB0_900
.LBB0_618:                              # %else1083
	slli	a0, s5, 21
	bgez	a0, .LBB0_619
	j	.LBB0_901
.LBB0_619:                              # %else1086
	slli	a0, s5, 20
	bgez	a0, .LBB0_620
	j	.LBB0_902
.LBB0_620:                              # %else1089
	slli	a0, s5, 19
	bgez	a0, .LBB0_621
	j	.LBB0_903
.LBB0_621:                              # %else1092
	slli	a0, s5, 18
	bgez	a0, .LBB0_622
	j	.LBB0_904
.LBB0_622:                              # %else1095
	slli	a0, s5, 17
	bgez	a0, .LBB0_624
.LBB0_623:                              # %cond.store1096
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 440(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_624:                              # %else1098
	slli	a0, s5, 16
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_626
# %bb.625:                              # %cond.store1099
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 444(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 29
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1296(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_626:                              # %else1101
	slli	a0, s5, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_627
	j	.LBB0_905
.LBB0_627:                              # %else1104
	slli	a0, s5, 14
	bgez	a0, .LBB0_628
	j	.LBB0_906
.LBB0_628:                              # %else1107
	slli	a0, s5, 13
	bgez	a0, .LBB0_629
	j	.LBB0_907
.LBB0_629:                              # %else1110
	slli	a0, s5, 12
	bgez	a0, .LBB0_630
	j	.LBB0_908
.LBB0_630:                              # %else1113
	slli	a0, s5, 11
	bgez	a0, .LBB0_631
	j	.LBB0_909
.LBB0_631:                              # %else1116
	slli	a0, s5, 10
	bgez	a0, .LBB0_632
	j	.LBB0_910
.LBB0_632:                              # %else1119
	slli	a0, s5, 9
	bgez	a0, .LBB0_633
	j	.LBB0_911
.LBB0_633:                              # %else1122
	slli	a0, s5, 8
	bgez	a0, .LBB0_634
	j	.LBB0_912
.LBB0_634:                              # %else1125
	slli	a0, s5, 7
	bgez	a0, .LBB0_635
	j	.LBB0_913
.LBB0_635:                              # %else1128
	slli	a0, s5, 6
	bgez	a0, .LBB0_636
	j	.LBB0_914
.LBB0_636:                              # %else1131
	slli	a0, s5, 5
	bgez	a0, .LBB0_637
	j	.LBB0_915
.LBB0_637:                              # %else1134
	slli	a0, s5, 4
	bgez	a0, .LBB0_639
.LBB0_638:                              # %cond.store1135
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 492(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 27
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_639:                              # %else1137
	.loc	1 0 44                          # k135114449651120.py:0:44
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s4, e32, m8, ta, ma
	vmslt.vx	v18, v8, s3
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vmslt.vx	v17, v8, s3
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vmslt.vx	v19, v8, s3
	.loc	1 10 44                         # k135114449651120.py:10:44
	slli	a0, s5, 3
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vmslt.vx	v16, v8, s3
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs1r.v	v16, (a1)                       # vscale x 8-byte Folded Spill
	bgez	a0, .LBB0_641
# %bb.640:                              # %cond.store1138
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 496(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs1r.v	v17, (a0)                       # vscale x 8-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs1r.v	v18, (a0)                       # vscale x 8-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs1r.v	v19, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl1r.v	v19, (a0)                       # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl1r.v	v18, (a0)                       # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl1r.v	v17, (a0)                       # vscale x 8-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_641:                              # %else1140
	.loc	1 0 44                          # k135114449651120.py:0:44
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v17, v18, 4
	.loc	1 10 44                         # k135114449651120.py:10:44
	slli	a0, s5, 2
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl1r.v	v8, (a1)                        # vscale x 8-byte Folded Reload
	vslideup.vi	v8, v19, 4
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs1r.v	v8, (a1)                        # vscale x 8-byte Folded Spill
	bgez	a0, .LBB0_643
# %bb.642:                              # %cond.store1141
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 500(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs1r.v	v17, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl1r.v	v17, (a0)                       # vscale x 8-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_643:                              # %else1143
	.loc	1 0 44                          # k135114449651120.py:0:44
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl1r.v	v8, (a0)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v8, v17, 8
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs1r.v	v8, (a0)                        # vscale x 8-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	slli	a0, s5, 1
	lui	a1, 3
	addi	a1, a1, -728
	add	s4, sp, a1
	bgez	a0, .LBB0_645
# %bb.644:                              # %cond.store1144
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 504(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1992(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_645:                              # %else1146
	.loc	1 0 44                          # k135114449651120.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	.loc	1 10 44                         # k135114449651120.py:10:44
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	add	a0, a0, a1
	ld	s3, 1120(a0)                    # 8-byte Folded Reload
	bgez	s5, .LBB0_647
# %bb.646:                              # %cond.store1147
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 508(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 13
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1872(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_647:                              # %else1149
	andi	a0, s3, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_648
	j	.LBB0_916
.LBB0_648:                              # %else1152
	andi	a0, s3, 2
	beqz	a0, .LBB0_649
	j	.LBB0_917
.LBB0_649:                              # %else1155
	andi	a0, s3, 4
	beqz	a0, .LBB0_650
	j	.LBB0_918
.LBB0_650:                              # %else1158
	andi	a0, s3, 8
	beqz	a0, .LBB0_651
	j	.LBB0_919
.LBB0_651:                              # %else1161
	andi	a0, s3, 16
	beqz	a0, .LBB0_652
	j	.LBB0_920
.LBB0_652:                              # %else1164
	andi	a0, s3, 32
	beqz	a0, .LBB0_653
	j	.LBB0_921
.LBB0_653:                              # %else1167
	andi	a0, s3, 64
	beqz	a0, .LBB0_654
	j	.LBB0_922
.LBB0_654:                              # %else1170
	andi	a0, s3, 128
	beqz	a0, .LBB0_655
	j	.LBB0_923
.LBB0_655:                              # %else1173
	andi	a0, s3, 256
	beqz	a0, .LBB0_656
	j	.LBB0_924
.LBB0_656:                              # %else1176
	andi	a0, s3, 512
	beqz	a0, .LBB0_657
	j	.LBB0_925
.LBB0_657:                              # %else1179
	andi	a0, s3, 1024
	beqz	a0, .LBB0_658
	j	.LBB0_926
.LBB0_658:                              # %else1182
	slli	a0, s3, 52
	bgez	a0, .LBB0_659
	j	.LBB0_927
.LBB0_659:                              # %else1185
	slli	a0, s3, 51
	bgez	a0, .LBB0_660
	j	.LBB0_928
.LBB0_660:                              # %else1188
	slli	a0, s3, 50
	bgez	a0, .LBB0_661
	j	.LBB0_929
.LBB0_661:                              # %else1191
	slli	a0, s3, 49
	bgez	a0, .LBB0_663
.LBB0_662:                              # %cond.store1192
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 568(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 456(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_663:                              # %else1194
	slli	a0, s3, 48
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_665
# %bb.664:                              # %cond.store1195
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 572(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 336(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_665:                              # %else1197
	slli	a0, s3, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_666
	j	.LBB0_930
.LBB0_666:                              # %else1200
	slli	a0, s3, 46
	bgez	a0, .LBB0_667
	j	.LBB0_931
.LBB0_667:                              # %else1203
	slli	a0, s3, 45
	bgez	a0, .LBB0_668
	j	.LBB0_932
.LBB0_668:                              # %else1206
	slli	a0, s3, 44
	bgez	a0, .LBB0_669
	j	.LBB0_933
.LBB0_669:                              # %else1209
	slli	a0, s3, 43
	bgez	a0, .LBB0_670
	j	.LBB0_934
.LBB0_670:                              # %else1212
	slli	a0, s3, 42
	bgez	a0, .LBB0_671
	j	.LBB0_935
.LBB0_671:                              # %else1215
	slli	a0, s3, 41
	lui	a1, 2
	addi	a1, a1, 1232
	add	s4, sp, a1
	bgez	a0, .LBB0_672
	j	.LBB0_936
.LBB0_672:                              # %else1218
	slli	a0, s3, 40
	bgez	a0, .LBB0_673
	j	.LBB0_937
.LBB0_673:                              # %else1221
	slli	a0, s3, 39
	bgez	a0, .LBB0_674
	j	.LBB0_938
.LBB0_674:                              # %else1224
	slli	a0, s3, 38
	bgez	a0, .LBB0_675
	j	.LBB0_939
.LBB0_675:                              # %else1227
	slli	a0, s3, 37
	bgez	a0, .LBB0_676
	j	.LBB0_940
.LBB0_676:                              # %else1230
	slli	a0, s3, 36
	bgez	a0, .LBB0_677
	j	.LBB0_941
.LBB0_677:                              # %else1233
	slli	a0, s3, 35
	bgez	a0, .LBB0_678
	j	.LBB0_942
.LBB0_678:                              # %else1236
	slli	a0, s3, 34
	bgez	a0, .LBB0_679
	j	.LBB0_943
.LBB0_679:                              # %else1239
	slli	a0, s3, 33
	bgez	a0, .LBB0_681
.LBB0_680:                              # %cond.store1240
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 632(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_681:                              # %else1242
	slli	a0, s3, 32
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_683
# %bb.682:                              # %cond.store1243
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 636(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 936(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_683:                              # %else1245
	slli	a0, s3, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_684
	j	.LBB0_944
.LBB0_684:                              # %else1248
	slli	a0, s3, 30
	bgez	a0, .LBB0_685
	j	.LBB0_945
.LBB0_685:                              # %else1251
	slli	a0, s3, 29
	bgez	a0, .LBB0_686
	j	.LBB0_946
.LBB0_686:                              # %else1254
	slli	a0, s3, 28
	bgez	a0, .LBB0_687
	j	.LBB0_947
.LBB0_687:                              # %else1257
	slli	a0, s3, 27
	bgez	a0, .LBB0_688
	j	.LBB0_948
.LBB0_688:                              # %else1260
	slli	a0, s3, 26
	bgez	a0, .LBB0_689
	j	.LBB0_949
.LBB0_689:                              # %else1263
	slli	a0, s3, 25
	bgez	a0, .LBB0_690
	j	.LBB0_950
.LBB0_690:                              # %else1266
	slli	a0, s3, 24
	bgez	a0, .LBB0_691
	j	.LBB0_951
.LBB0_691:                              # %else1269
	slli	a0, s3, 23
	bgez	a0, .LBB0_692
	j	.LBB0_952
.LBB0_692:                              # %else1272
	slli	a0, s3, 22
	bgez	a0, .LBB0_693
	j	.LBB0_953
.LBB0_693:                              # %else1275
	slli	a0, s3, 21
	bgez	a0, .LBB0_694
	j	.LBB0_954
.LBB0_694:                              # %else1278
	slli	a0, s3, 20
	lui	a1, 2
	addi	a1, a1, -904
	add	s5, sp, a1
	bgez	a0, .LBB0_695
	j	.LBB0_955
.LBB0_695:                              # %else1281
	slli	a0, s3, 19
	bgez	a0, .LBB0_696
	j	.LBB0_956
.LBB0_696:                              # %else1284
	slli	a0, s3, 18
	bgez	a0, .LBB0_697
	j	.LBB0_957
.LBB0_697:                              # %else1287
	slli	a0, s3, 17
	bgez	a0, .LBB0_699
.LBB0_698:                              # %cond.store1288
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 696(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_699:                              # %else1290
	slli	a0, s3, 16
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_701
# %bb.700:                              # %cond.store1291
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 700(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_701:                              # %else1293
	slli	a0, s3, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_702
	j	.LBB0_958
.LBB0_702:                              # %else1296
	slli	a0, s3, 14
	bgez	a0, .LBB0_703
	j	.LBB0_959
.LBB0_703:                              # %else1299
	slli	a0, s3, 13
	bgez	a0, .LBB0_704
	j	.LBB0_960
.LBB0_704:                              # %else1302
	slli	a0, s3, 12
	bgez	a0, .LBB0_705
	j	.LBB0_961
.LBB0_705:                              # %else1305
	slli	a0, s3, 11
	bgez	a0, .LBB0_706
	j	.LBB0_962
.LBB0_706:                              # %else1308
	slli	a0, s3, 10
	bgez	a0, .LBB0_707
	j	.LBB0_963
.LBB0_707:                              # %else1311
	slli	a0, s3, 9
	bgez	a0, .LBB0_708
	j	.LBB0_964
.LBB0_708:                              # %else1314
	slli	a0, s3, 8
	bgez	a0, .LBB0_709
	j	.LBB0_965
.LBB0_709:                              # %else1317
	slli	a0, s3, 7
	bgez	a0, .LBB0_710
	j	.LBB0_966
.LBB0_710:                              # %else1320
	slli	a0, s3, 6
	bgez	a0, .LBB0_711
	j	.LBB0_967
.LBB0_711:                              # %else1323
	slli	a0, s3, 5
	bgez	a0, .LBB0_712
	j	.LBB0_968
.LBB0_712:                              # %else1326
	slli	a0, s3, 4
	bgez	a0, .LBB0_713
	j	.LBB0_969
.LBB0_713:                              # %else1329
	slli	a0, s3, 3
	bgez	a0, .LBB0_714
	j	.LBB0_970
.LBB0_714:                              # %else1332
	slli	a0, s3, 2
	bgez	a0, .LBB0_716
.LBB0_715:                              # %cond.store1333
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 756(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 29
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_716:                              # %else1335
	slli	a0, s3, 1
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl1r.v	v8, (a1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	bgez	a0, .LBB0_718
# %bb.717:                              # %cond.store1336
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 760(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs1r.v	v16, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl1r.v	v16, (a0)                       # vscale x 8-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_718:                              # %else1338
	.loc	1 0 44                          # k135114449651120.py:0:44
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	.loc	1 10 44                         # k135114449651120.py:10:44
	vmv.x.s	s4, v16
	bgez	s3, .LBB0_720
# %bb.719:                              # %cond.store1339
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 764(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 7
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_720:                              # %else1341
	andi	a0, s4, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_721
	j	.LBB0_971
.LBB0_721:                              # %else1344
	andi	a0, s4, 2
	beqz	a0, .LBB0_722
	j	.LBB0_972
.LBB0_722:                              # %else1347
	andi	a0, s4, 4
	beqz	a0, .LBB0_723
	j	.LBB0_973
.LBB0_723:                              # %else1350
	andi	a0, s4, 8
	beqz	a0, .LBB0_724
	j	.LBB0_974
.LBB0_724:                              # %else1353
	andi	a0, s4, 16
	lui	a1, 1
	addi	a1, a1, 1080
	add	s3, sp, a1
	beqz	a0, .LBB0_725
	j	.LBB0_975
.LBB0_725:                              # %else1356
	andi	a0, s4, 32
	beqz	a0, .LBB0_726
	j	.LBB0_976
.LBB0_726:                              # %else1359
	andi	a0, s4, 64
	beqz	a0, .LBB0_727
	j	.LBB0_977
.LBB0_727:                              # %else1362
	andi	a0, s4, 128
	beqz	a0, .LBB0_728
	j	.LBB0_978
.LBB0_728:                              # %else1365
	andi	a0, s4, 256
	beqz	a0, .LBB0_729
	j	.LBB0_979
.LBB0_729:                              # %else1368
	andi	a0, s4, 512
	beqz	a0, .LBB0_730
	j	.LBB0_980
.LBB0_730:                              # %else1371
	andi	a0, s4, 1024
	beqz	a0, .LBB0_731
	j	.LBB0_981
.LBB0_731:                              # %else1374
	slli	a0, s4, 52
	bgez	a0, .LBB0_732
	j	.LBB0_982
.LBB0_732:                              # %else1377
	slli	a0, s4, 51
	bgez	a0, .LBB0_733
	j	.LBB0_983
.LBB0_733:                              # %else1380
	slli	a0, s4, 50
	bgez	a0, .LBB0_734
	j	.LBB0_984
.LBB0_734:                              # %else1383
	slli	a0, s4, 49
	bgez	a0, .LBB0_736
.LBB0_735:                              # %cond.store1384
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 824(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_736:                              # %else1386
	slli	a0, s4, 48
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_738
# %bb.737:                              # %cond.store1387
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 828(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 576(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_738:                              # %else1389
	slli	a0, s4, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_739
	j	.LBB0_985
.LBB0_739:                              # %else1392
	slli	a0, s4, 46
	bgez	a0, .LBB0_740
	j	.LBB0_986
.LBB0_740:                              # %else1395
	slli	a0, s4, 45
	bgez	a0, .LBB0_741
	j	.LBB0_987
.LBB0_741:                              # %else1398
	slli	a0, s4, 44
	bgez	a0, .LBB0_742
	j	.LBB0_988
.LBB0_742:                              # %else1401
	slli	a0, s4, 43
	bgez	a0, .LBB0_743
	j	.LBB0_989
.LBB0_743:                              # %else1404
	slli	a0, s4, 42
	bgez	a0, .LBB0_744
	j	.LBB0_990
.LBB0_744:                              # %else1407
	slli	a0, s4, 41
	bgez	a0, .LBB0_745
	j	.LBB0_991
.LBB0_745:                              # %else1410
	slli	a0, s4, 40
	bgez	a0, .LBB0_746
	j	.LBB0_992
.LBB0_746:                              # %else1413
	slli	a0, s4, 39
	addi	s3, sp, 2047
	addi	s3, s3, 993
	bgez	a0, .LBB0_747
	j	.LBB0_993
.LBB0_747:                              # %else1416
	slli	a0, s4, 38
	bgez	a0, .LBB0_748
	j	.LBB0_994
.LBB0_748:                              # %else1419
	slli	a0, s4, 37
	bgez	a0, .LBB0_749
	j	.LBB0_995
.LBB0_749:                              # %else1422
	slli	a0, s4, 36
	bgez	a0, .LBB0_750
	j	.LBB0_996
.LBB0_750:                              # %else1425
	slli	a0, s4, 35
	bgez	a0, .LBB0_751
	j	.LBB0_997
.LBB0_751:                              # %else1428
	slli	a0, s4, 34
	bgez	a0, .LBB0_752
	j	.LBB0_998
.LBB0_752:                              # %else1431
	slli	a0, s4, 33
	bgez	a0, .LBB0_754
.LBB0_753:                              # %cond.store1432
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 888(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_754:                              # %else1434
	slli	a0, s4, 32
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_756
# %bb.755:                              # %cond.store1435
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 892(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1176(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_756:                              # %else1437
	slli	a0, s4, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_757
	j	.LBB0_999
.LBB0_757:                              # %else1440
	slli	a0, s4, 30
	bgez	a0, .LBB0_758
	j	.LBB0_1000
.LBB0_758:                              # %else1443
	slli	a0, s4, 29
	bgez	a0, .LBB0_759
	j	.LBB0_1001
.LBB0_759:                              # %else1446
	slli	a0, s4, 28
	bgez	a0, .LBB0_760
	j	.LBB0_1002
.LBB0_760:                              # %else1449
	slli	a0, s4, 27
	bgez	a0, .LBB0_761
	j	.LBB0_1003
.LBB0_761:                              # %else1452
	slli	a0, s4, 26
	bgez	a0, .LBB0_762
	j	.LBB0_1004
.LBB0_762:                              # %else1455
	slli	a0, s4, 25
	bgez	a0, .LBB0_763
	j	.LBB0_1005
.LBB0_763:                              # %else1458
	slli	a0, s4, 24
	bgez	a0, .LBB0_764
	j	.LBB0_1006
.LBB0_764:                              # %else1461
	slli	a0, s4, 23
	bgez	a0, .LBB0_765
	j	.LBB0_1007
.LBB0_765:                              # %else1464
	slli	a0, s4, 22
	bgez	a0, .LBB0_766
	j	.LBB0_1008
.LBB0_766:                              # %else1467
	slli	a0, s4, 21
	bgez	a0, .LBB0_767
	j	.LBB0_1009
.LBB0_767:                              # %else1470
	slli	a0, s4, 20
	bgez	a0, .LBB0_768
	j	.LBB0_1010
.LBB0_768:                              # %else1473
	slli	a0, s4, 19
	bgez	a0, .LBB0_769
	j	.LBB0_1011
.LBB0_769:                              # %else1476
	slli	a0, s4, 18
	bgez	a0, .LBB0_770
	j	.LBB0_1012
.LBB0_770:                              # %else1479
	slli	a0, s4, 17
	bgez	a0, .LBB0_772
.LBB0_771:                              # %cond.store1480
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 952(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 641
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1296(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_772:                              # %else1482
	slli	a0, s4, 16
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_774
# %bb.773:                              # %cond.store1483
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 956(sp)                    # 4-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 513
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 12
	addi	a1, a1, 1120
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1416(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_774:                              # %else1485
	slli	a0, s4, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_775
	j	.LBB0_1013
.LBB0_775:                              # %else1488
	slli	a0, s4, 14
	bgez	a0, .LBB0_776
	j	.LBB0_1014
.LBB0_776:                              # %else1491
	slli	a0, s4, 13
	bgez	a0, .LBB0_777
	j	.LBB0_1015
.LBB0_777:                              # %else1494
	slli	a0, s4, 12
	bgez	a0, .LBB0_778
	j	.LBB0_1016
.LBB0_778:                              # %else1497
	slli	a0, s4, 11
	bgez	a0, .LBB0_779
	j	.LBB0_1017
.LBB0_779:                              # %else1500
	slli	a0, s4, 10
	bgez	a0, .LBB0_780
	j	.LBB0_1018
.LBB0_780:                              # %else1503
	slli	a0, s4, 9
	bgez	a0, .LBB0_781
	j	.LBB0_1019
.LBB0_781:                              # %else1506
	slli	a0, s4, 8
	bgez	a0, .LBB0_782
	j	.LBB0_1020
.LBB0_782:                              # %else1509
	slli	a0, s4, 7
	bgez	a0, .LBB0_783
	j	.LBB0_1021
.LBB0_783:                              # %else1512
	slli	a0, s4, 6
	bgez	a0, .LBB0_784
	j	.LBB0_1022
.LBB0_784:                              # %else1515
	slli	a0, s4, 5
	bgez	a0, .LBB0_785
	j	.LBB0_1023
.LBB0_785:                              # %else1518
	slli	a0, s4, 4
	bgez	a0, .LBB0_786
	j	.LBB0_1024
.LBB0_786:                              # %else1521
	slli	a0, s4, 3
	bgez	a0, .LBB0_787
	j	.LBB0_1025
.LBB0_787:                              # %else1524
	slli	a0, s4, 2
	bgez	a0, .LBB0_788
	j	.LBB0_1026
.LBB0_788:                              # %else1527
	slli	a0, s4, 1
	bgez	a0, .LBB0_789
	j	.LBB0_1027
.LBB0_789:                              # %else1530
	bgez	s4, .LBB0_791
.LBB0_790:                              # %cond.store1531
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 1020(sp)                   # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1024
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1144(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_791:                              # %else1533
	.loc	1 10 4 epilogue_begin           # k135114449651120.py:10:4
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
	fld	fs0, 1960(sp)                   # 8-byte Folded Reload
	fld	fs1, 1952(sp)                   # 8-byte Folded Reload
	fld	fs2, 1944(sp)                   # 8-byte Folded Reload
	fld	fs3, 1936(sp)                   # 8-byte Folded Reload
	fld	fs4, 1928(sp)                   # 8-byte Folded Reload
	fld	fs5, 1920(sp)                   # 8-byte Folded Reload
	fld	fs6, 1912(sp)                   # 8-byte Folded Reload
	fld	fs7, 1904(sp)                   # 8-byte Folded Reload
	fld	fs8, 1896(sp)                   # 8-byte Folded Reload
	fld	fs9, 1888(sp)                   # 8-byte Folded Reload
	fld	fs10, 1880(sp)                  # 8-byte Folded Reload
	fld	fs11, 1872(sp)                  # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
	.cfi_restore s6
	.cfi_restore s7
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
.LBB0_792:                              # %cond.load715
	.cfi_restore_state
	.loc	1 0 4                           # k135114449651120.py:0:4
	lui	a2, 7
	addi	a2, a2, -1536
	add	a2, sp, a2
	.loc	1 9 43 is_stmt 1                # k135114449651120.py:9:43
	vse64.v	v16, (a2)
	ld	a2, 696(a1)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa5, a2
	fsw	fa5, 956(sp)                    # 4-byte Folded Spill
	slli	a2, a5, 15
	vadd.vx	v8, v8, a0
	bltz	a2, .LBB0_793
	j	.LBB0_483
.LBB0_793:                              # %cond.load718
	vmv.x.s	a0, v8
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 960(sp)                    # 4-byte Folded Spill
	slli	a0, a5, 14
	bltz	a0, .LBB0_794
	j	.LBB0_484
.LBB0_794:                              # %cond.load721
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 964(sp)                    # 4-byte Folded Spill
	slli	a0, a5, 13
	bltz	a0, .LBB0_795
	j	.LBB0_485
.LBB0_795:                              # %cond.load724
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 2
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 968(sp)                    # 4-byte Folded Spill
	slli	a0, a5, 12
	bltz	a0, .LBB0_796
	j	.LBB0_486
.LBB0_796:                              # %cond.load727
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 3
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 972(sp)                    # 4-byte Folded Spill
	slli	a0, a5, 11
	bltz	a0, .LBB0_797
	j	.LBB0_487
.LBB0_797:                              # %cond.load730
	.loc	1 0 43 is_stmt 0                # k135114449651120.py:0:43
	lui	a0, 7
	addi	a0, a0, -1664
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 976(sp)                    # 4-byte Folded Spill
	slli	a0, a5, 10
	bltz	a0, .LBB0_798
	j	.LBB0_488
.LBB0_798:                              # %cond.load733
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a0, 7
	addi	a0, a0, -1792
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 980(sp)                    # 4-byte Folded Spill
	slli	a0, a5, 9
	bltz	a0, .LBB0_799
	j	.LBB0_489
.LBB0_799:                              # %cond.load736
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a0, 7
	addi	a0, a0, -1920
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 984(sp)                    # 4-byte Folded Spill
	slli	a0, a5, 8
	bltz	a0, .LBB0_800
	j	.LBB0_490
.LBB0_800:                              # %cond.load739
	.loc	1 0 43                          # k135114449651120.py:0:43
	li	a0, 13
	slli	a0, a0, 11
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 988(sp)                    # 4-byte Folded Spill
	slli	a0, a5, 7
	bltz	a0, .LBB0_801
	j	.LBB0_491
.LBB0_801:                              # %cond.load742
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a0, 6
	addi	a0, a0, 1920
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(a1)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 992(sp)                    # 4-byte Folded Spill
	slli	a0, a5, 6
	lui	a1, 6
	addi	a1, a1, -152
	add	s5, sp, a1
	bltz	a0, .LBB0_802
	j	.LBB0_492
.LBB0_802:                              # %cond.load745
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a0, 6
	addi	a0, a0, 1792
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 996(sp)                    # 4-byte Folded Spill
	slli	a0, a5, 5
	bltz	a0, .LBB0_803
	j	.LBB0_493
.LBB0_803:                              # %cond.load748
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a0, 6
	addi	a0, a0, 1664
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 1000(sp)                   # 4-byte Folded Spill
	slli	a0, a5, 4
	bltz	a0, .LBB0_804
	j	.LBB0_494
.LBB0_804:                              # %cond.load751
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a0, 6
	addi	a0, a0, 1536
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 1004(sp)                   # 4-byte Folded Spill
	slli	a0, a5, 3
	bltz	a0, .LBB0_805
	j	.LBB0_495
.LBB0_805:                              # %cond.load754
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a0, 6
	addi	a0, a0, 1408
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 1008(sp)                   # 4-byte Folded Spill
	slli	a0, a5, 2
	bltz	a0, .LBB0_806
	j	.LBB0_496
.LBB0_806:                              # %cond.load757
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a0, 6
	addi	a0, a0, 1280
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 1012(sp)                   # 4-byte Folded Spill
	slli	a0, a5, 1
	bltz	a0, .LBB0_807
	j	.LBB0_497
.LBB0_807:                              # %cond.load760
	.loc	1 0 43                          # k135114449651120.py:0:43
	lui	a0, 6
	addi	a0, a0, 1152
	add	a0, sp, a0
	.loc	1 9 43                          # k135114449651120.py:9:43
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s5)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 1016(sp)                   # 4-byte Folded Spill
	bgez	a5, .LBB0_1028
	j	.LBB0_498
.LBB0_1028:                             # %cond.load760
	j	.LBB0_499
.LBB0_808:                              # %cond.store
	.loc	1 10 44 is_stmt 1               # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 2
	bnez	a0, .LBB0_809
	j	.LBB0_501
.LBB0_809:                              # %cond.store769
	fmv.s	fa0, fs1
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 4
	bnez	a0, .LBB0_810
	j	.LBB0_502
.LBB0_810:                              # %cond.store772
	fmv.s	fa0, fs2
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 8
	bnez	a0, .LBB0_811
	j	.LBB0_503
.LBB0_811:                              # %cond.store775
	fmv.s	fa0, fs0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 16
	bnez	a0, .LBB0_812
	j	.LBB0_504
.LBB0_812:                              # %cond.store778
	fmv.s	fa0, fs11
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 32
	bnez	a0, .LBB0_813
	j	.LBB0_505
.LBB0_813:                              # %cond.store781
	fmv.s	fa0, fs10
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 64
	bnez	a0, .LBB0_814
	j	.LBB0_506
.LBB0_814:                              # %cond.store784
	fmv.s	fa0, fs9
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 128
	bnez	a0, .LBB0_815
	j	.LBB0_507
.LBB0_815:                              # %cond.store787
	fmv.s	fa0, fs8
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 512
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 256
	bnez	a0, .LBB0_816
	j	.LBB0_508
.LBB0_816:                              # %cond.store790
	fmv.s	fa0, fs7
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 512
	bnez	a0, .LBB0_817
	j	.LBB0_509
.LBB0_817:                              # %cond.store793
	fmv.s	fa0, fs6
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 1024
	bnez	a0, .LBB0_818
	j	.LBB0_510
.LBB0_818:                              # %cond.store796
	fmv.s	fa0, fs5
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 52
	bltz	a0, .LBB0_819
	j	.LBB0_511
.LBB0_819:                              # %cond.store799
	fmv.s	fa0, fs4
	call	__truncsfbf2
	lui	a0, 6
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 51
	bltz	a0, .LBB0_820
	j	.LBB0_512
.LBB0_820:                              # %cond.store802
	fmv.s	fa0, fs3
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 50
	bltz	a0, .LBB0_821
	j	.LBB0_513
.LBB0_821:                              # %cond.store805
	.loc	1 0 44 is_stmt 0                # k135114449651120.py:0:44
	flw	fa0, 52(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 49
	lui	a1, 5
	addi	a1, a1, 1832
	add	s5, sp, a1
	bgez	a0, .LBB0_1029
	j	.LBB0_514
.LBB0_1029:                             # %cond.store805
	j	.LBB0_515
.LBB0_822:                              # %cond.store814
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 64(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 46
	bltz	a0, .LBB0_823
	j	.LBB0_519
.LBB0_823:                              # %cond.store817
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 68(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 45
	bltz	a0, .LBB0_824
	j	.LBB0_520
.LBB0_824:                              # %cond.store820
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 72(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 44
	bltz	a0, .LBB0_825
	j	.LBB0_521
.LBB0_825:                              # %cond.store823
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 76(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 43
	bltz	a0, .LBB0_826
	j	.LBB0_522
.LBB0_826:                              # %cond.store826
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 80(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 42
	bltz	a0, .LBB0_827
	j	.LBB0_523
.LBB0_827:                              # %cond.store829
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 84(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 41
	bltz	a0, .LBB0_828
	j	.LBB0_524
.LBB0_828:                              # %cond.store832
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 88(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 40
	bltz	a0, .LBB0_829
	j	.LBB0_525
.LBB0_829:                              # %cond.store835
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 92(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 39
	bltz	a0, .LBB0_830
	j	.LBB0_526
.LBB0_830:                              # %cond.store838
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 96(sp)                     # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 38
	bltz	a0, .LBB0_831
	j	.LBB0_527
.LBB0_831:                              # %cond.store841
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 100(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 37
	bltz	a0, .LBB0_832
	j	.LBB0_528
.LBB0_832:                              # %cond.store844
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 104(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 36
	bltz	a0, .LBB0_833
	j	.LBB0_529
.LBB0_833:                              # %cond.store847
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 108(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1536
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 35
	bltz	a0, .LBB0_834
	j	.LBB0_530
.LBB0_834:                              # %cond.store850
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 112(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 34
	bltz	a0, .LBB0_835
	j	.LBB0_531
.LBB0_835:                              # %cond.store853
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 116(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 33
	bgez	a0, .LBB0_1030
	j	.LBB0_532
.LBB0_1030:                             # %cond.store853
	j	.LBB0_533
.LBB0_836:                              # %cond.store862
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 128(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 30
	bltz	a0, .LBB0_837
	j	.LBB0_537
.LBB0_837:                              # %cond.store865
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 132(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 29
	bltz	a0, .LBB0_838
	j	.LBB0_538
.LBB0_838:                              # %cond.store868
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 136(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 28
	bltz	a0, .LBB0_839
	j	.LBB0_539
.LBB0_839:                              # %cond.store871
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 140(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 27
	bltz	a0, .LBB0_840
	j	.LBB0_540
.LBB0_840:                              # %cond.store874
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 144(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 26
	bltz	a0, .LBB0_841
	j	.LBB0_541
.LBB0_841:                              # %cond.store877
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 148(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 25
	lui	a1, 5
	addi	a1, a1, -304
	add	s5, sp, a1
	bltz	a0, .LBB0_842
	j	.LBB0_542
.LBB0_842:                              # %cond.store880
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 152(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 24
	bltz	a0, .LBB0_843
	j	.LBB0_543
.LBB0_843:                              # %cond.store883
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 156(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 1536
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 23
	bltz	a0, .LBB0_844
	j	.LBB0_544
.LBB0_844:                              # %cond.store886
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 160(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 22
	bltz	a0, .LBB0_845
	j	.LBB0_545
.LBB0_845:                              # %cond.store889
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 164(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 21
	bltz	a0, .LBB0_846
	j	.LBB0_546
.LBB0_846:                              # %cond.store892
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 168(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 20
	bltz	a0, .LBB0_847
	j	.LBB0_547
.LBB0_847:                              # %cond.store895
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 172(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 19
	bltz	a0, .LBB0_848
	j	.LBB0_548
.LBB0_848:                              # %cond.store898
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 176(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 18
	bltz	a0, .LBB0_849
	j	.LBB0_549
.LBB0_849:                              # %cond.store901
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 180(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 17
	bgez	a0, .LBB0_1031
	j	.LBB0_550
.LBB0_1031:                             # %cond.store901
	j	.LBB0_551
.LBB0_850:                              # %cond.store910
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 192(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 14
	bltz	a0, .LBB0_851
	j	.LBB0_555
.LBB0_851:                              # %cond.store913
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 196(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 13
	bltz	a0, .LBB0_852
	j	.LBB0_556
.LBB0_852:                              # %cond.store916
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 200(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 12
	bltz	a0, .LBB0_853
	j	.LBB0_557
.LBB0_853:                              # %cond.store919
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 204(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 11
	bltz	a0, .LBB0_854
	j	.LBB0_558
.LBB0_854:                              # %cond.store922
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 208(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 10
	bltz	a0, .LBB0_855
	j	.LBB0_559
.LBB0_855:                              # %cond.store925
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 212(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 9
	bltz	a0, .LBB0_856
	j	.LBB0_560
.LBB0_856:                              # %cond.store928
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 216(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 8
	bltz	a0, .LBB0_857
	j	.LBB0_561
.LBB0_857:                              # %cond.store931
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 220(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 7
	bltz	a0, .LBB0_858
	j	.LBB0_562
.LBB0_858:                              # %cond.store934
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 224(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 6
	bltz	a0, .LBB0_859
	j	.LBB0_563
.LBB0_859:                              # %cond.store937
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 228(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 5
	bltz	a0, .LBB0_860
	j	.LBB0_564
.LBB0_860:                              # %cond.store940
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 232(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 4
	lui	a1, 4
	addi	a1, a1, 1656
	add	s7, sp, a1
	bltz	a0, .LBB0_861
	j	.LBB0_565
.LBB0_861:                              # %cond.store943
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 236(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -512
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 3
	bltz	a0, .LBB0_862
	j	.LBB0_566
.LBB0_862:                              # %cond.store946
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 240(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 2
	bgez	a0, .LBB0_1032
	j	.LBB0_567
.LBB0_1032:                             # %cond.store946
	j	.LBB0_568
.LBB0_863:                              # %cond.store958
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 256(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 2
	bnez	a0, .LBB0_864
	j	.LBB0_574
.LBB0_864:                              # %cond.store961
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 260(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 4
	bnez	a0, .LBB0_865
	j	.LBB0_575
.LBB0_865:                              # %cond.store964
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 264(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 8
	bnez	a0, .LBB0_866
	j	.LBB0_576
.LBB0_866:                              # %cond.store967
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 268(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 16
	bnez	a0, .LBB0_867
	j	.LBB0_577
.LBB0_867:                              # %cond.store970
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 272(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1320(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 32
	bnez	a0, .LBB0_868
	j	.LBB0_578
.LBB0_868:                              # %cond.store973
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 276(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1200(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 64
	bnez	a0, .LBB0_869
	j	.LBB0_579
.LBB0_869:                              # %cond.store976
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 280(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 128
	bnez	a0, .LBB0_870
	j	.LBB0_580
.LBB0_870:                              # %cond.store979
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 284(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -1536
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 256
	bnez	a0, .LBB0_871
	j	.LBB0_581
.LBB0_871:                              # %cond.store982
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 288(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 512
	bnez	a0, .LBB0_872
	j	.LBB0_582
.LBB0_872:                              # %cond.store985
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 292(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 1024
	bnez	a0, .LBB0_873
	j	.LBB0_583
.LBB0_873:                              # %cond.store988
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 296(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 52
	bltz	a0, .LBB0_874
	j	.LBB0_584
.LBB0_874:                              # %cond.store991
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 300(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 51
	bltz	a0, .LBB0_875
	j	.LBB0_585
.LBB0_875:                              # %cond.store994
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 304(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 50
	bltz	a0, .LBB0_876
	j	.LBB0_586
.LBB0_876:                              # %cond.store997
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 308(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s7)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 49
	bgez	a0, .LBB0_1033
	j	.LBB0_587
.LBB0_1033:                             # %cond.store997
	j	.LBB0_588
.LBB0_877:                              # %cond.store1006
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 320(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 46
	bltz	a0, .LBB0_878
	j	.LBB0_592
.LBB0_878:                              # %cond.store1009
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 324(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 45
	bltz	a0, .LBB0_879
	j	.LBB0_593
.LBB0_879:                              # %cond.store1012
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 328(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 44
	bltz	a0, .LBB0_880
	j	.LBB0_594
.LBB0_880:                              # %cond.store1015
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 332(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 43
	lui	a1, 4
	addi	a1, a1, -576
	add	s6, sp, a1
	bltz	a0, .LBB0_881
	j	.LBB0_595
.LBB0_881:                              # %cond.store1018
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 336(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 42
	bltz	a0, .LBB0_882
	j	.LBB0_596
.LBB0_882:                              # %cond.store1021
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 340(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 41
	bltz	a0, .LBB0_883
	j	.LBB0_597
.LBB0_883:                              # %cond.store1024
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 344(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 40
	bltz	a0, .LBB0_884
	j	.LBB0_598
.LBB0_884:                              # %cond.store1027
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 348(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 39
	bltz	a0, .LBB0_885
	j	.LBB0_599
.LBB0_885:                              # %cond.store1030
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 352(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 38
	bltz	a0, .LBB0_886
	j	.LBB0_600
.LBB0_886:                              # %cond.store1033
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 356(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 37
	bltz	a0, .LBB0_887
	j	.LBB0_601
.LBB0_887:                              # %cond.store1036
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 360(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 36
	bltz	a0, .LBB0_888
	j	.LBB0_602
.LBB0_888:                              # %cond.store1039
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 364(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 512
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 35
	bltz	a0, .LBB0_889
	j	.LBB0_603
.LBB0_889:                              # %cond.store1042
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 368(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 34
	bltz	a0, .LBB0_890
	j	.LBB0_604
.LBB0_890:                              # %cond.store1045
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 372(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 33
	bgez	a0, .LBB0_1034
	j	.LBB0_605
.LBB0_1034:                             # %cond.store1045
	j	.LBB0_606
.LBB0_891:                              # %cond.store1054
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 384(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 30
	bltz	a0, .LBB0_892
	j	.LBB0_610
.LBB0_892:                              # %cond.store1057
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 388(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 29
	bltz	a0, .LBB0_893
	j	.LBB0_611
.LBB0_893:                              # %cond.store1060
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 392(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 28
	bltz	a0, .LBB0_894
	j	.LBB0_612
.LBB0_894:                              # %cond.store1063
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 396(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 27
	bltz	a0, .LBB0_895
	j	.LBB0_613
.LBB0_895:                              # %cond.store1066
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 400(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 26
	bltz	a0, .LBB0_896
	j	.LBB0_614
.LBB0_896:                              # %cond.store1069
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 404(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 25
	bltz	a0, .LBB0_897
	j	.LBB0_615
.LBB0_897:                              # %cond.store1072
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 408(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 24
	bltz	a0, .LBB0_898
	j	.LBB0_616
.LBB0_898:                              # %cond.store1075
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 412(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 31
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 23
	bltz	a0, .LBB0_899
	j	.LBB0_617
.LBB0_899:                              # %cond.store1078
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 416(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 22
	lui	a1, 3
	addi	a1, a1, 1384
	add	s6, sp, a1
	bltz	a0, .LBB0_900
	j	.LBB0_618
.LBB0_900:                              # %cond.store1081
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 420(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 21
	bltz	a0, .LBB0_901
	j	.LBB0_619
.LBB0_901:                              # %cond.store1084
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 424(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 20
	bltz	a0, .LBB0_902
	j	.LBB0_620
.LBB0_902:                              # %cond.store1087
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 428(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 15
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 19
	bltz	a0, .LBB0_903
	j	.LBB0_621
.LBB0_903:                              # %cond.store1090
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 432(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 18
	bltz	a0, .LBB0_904
	j	.LBB0_622
.LBB0_904:                              # %cond.store1093
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 436(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 17
	bgez	a0, .LBB0_1035
	j	.LBB0_623
.LBB0_1035:                             # %cond.store1093
	j	.LBB0_624
.LBB0_905:                              # %cond.store1102
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 448(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 14
	bltz	a0, .LBB0_906
	j	.LBB0_628
.LBB0_906:                              # %cond.store1105
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 452(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 13
	bltz	a0, .LBB0_907
	j	.LBB0_629
.LBB0_907:                              # %cond.store1108
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 456(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 12
	bltz	a0, .LBB0_908
	j	.LBB0_630
.LBB0_908:                              # %cond.store1111
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 460(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 11
	bltz	a0, .LBB0_909
	j	.LBB0_631
.LBB0_909:                              # %cond.store1114
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 464(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 10
	bltz	a0, .LBB0_910
	j	.LBB0_632
.LBB0_910:                              # %cond.store1117
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 468(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 9
	bltz	a0, .LBB0_911
	j	.LBB0_633
.LBB0_911:                              # %cond.store1120
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 472(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 8
	bltz	a0, .LBB0_912
	j	.LBB0_634
.LBB0_912:                              # %cond.store1123
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 476(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 7
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 7
	bltz	a0, .LBB0_913
	j	.LBB0_635
.LBB0_913:                              # %cond.store1126
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 480(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 6
	bltz	a0, .LBB0_914
	j	.LBB0_636
.LBB0_914:                              # %cond.store1129
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 484(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 5
	bltz	a0, .LBB0_915
	j	.LBB0_637
.LBB0_915:                              # %cond.store1132
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 488(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 4
	bgez	a0, .LBB0_1036
	j	.LBB0_638
.LBB0_1036:                             # %cond.store1132
	j	.LBB0_639
.LBB0_916:                              # %cond.store1150
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 512(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 2
	bnez	a0, .LBB0_917
	j	.LBB0_649
.LBB0_917:                              # %cond.store1153
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 516(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 4
	bnez	a0, .LBB0_918
	j	.LBB0_650
.LBB0_918:                              # %cond.store1156
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 520(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 8
	bnez	a0, .LBB0_919
	j	.LBB0_651
.LBB0_919:                              # %cond.store1159
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 524(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 16
	bnez	a0, .LBB0_920
	j	.LBB0_652
.LBB0_920:                              # %cond.store1162
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 528(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 32
	bnez	a0, .LBB0_921
	j	.LBB0_653
.LBB0_921:                              # %cond.store1165
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 532(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 64
	bnez	a0, .LBB0_922
	j	.LBB0_654
.LBB0_922:                              # %cond.store1168
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 536(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 128
	bnez	a0, .LBB0_923
	j	.LBB0_655
.LBB0_923:                              # %cond.store1171
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 540(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 256
	bnez	a0, .LBB0_924
	j	.LBB0_656
.LBB0_924:                              # %cond.store1174
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 544(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 512
	bnez	a0, .LBB0_925
	j	.LBB0_657
.LBB0_925:                              # %cond.store1177
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 548(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 1024
	bnez	a0, .LBB0_926
	j	.LBB0_658
.LBB0_926:                              # %cond.store1180
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 552(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 52
	bltz	a0, .LBB0_927
	j	.LBB0_659
.LBB0_927:                              # %cond.store1183
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 556(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 51
	bltz	a0, .LBB0_928
	j	.LBB0_660
.LBB0_928:                              # %cond.store1186
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 560(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 50
	bltz	a0, .LBB0_929
	j	.LBB0_661
.LBB0_929:                              # %cond.store1189
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 564(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 49
	bgez	a0, .LBB0_1037
	j	.LBB0_662
.LBB0_1037:                             # %cond.store1189
	j	.LBB0_663
.LBB0_930:                              # %cond.store1198
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 576(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 46
	bltz	a0, .LBB0_931
	j	.LBB0_667
.LBB0_931:                              # %cond.store1201
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 580(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 45
	bltz	a0, .LBB0_932
	j	.LBB0_668
.LBB0_932:                              # %cond.store1204
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 584(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 44
	bltz	a0, .LBB0_933
	j	.LBB0_669
.LBB0_933:                              # %cond.store1207
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 588(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 43
	bltz	a0, .LBB0_934
	j	.LBB0_670
.LBB0_934:                              # %cond.store1210
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 592(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 42
	bltz	a0, .LBB0_935
	j	.LBB0_671
.LBB0_935:                              # %cond.store1213
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 596(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 41
	lui	a1, 2
	addi	a1, a1, 1232
	add	s4, sp, a1
	bltz	a0, .LBB0_936
	j	.LBB0_672
.LBB0_936:                              # %cond.store1216
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 600(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 40
	bltz	a0, .LBB0_937
	j	.LBB0_673
.LBB0_937:                              # %cond.store1219
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 604(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 39
	bltz	a0, .LBB0_938
	j	.LBB0_674
.LBB0_938:                              # %cond.store1222
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 608(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 38
	bltz	a0, .LBB0_939
	j	.LBB0_675
.LBB0_939:                              # %cond.store1225
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 612(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 37
	bltz	a0, .LBB0_940
	j	.LBB0_676
.LBB0_940:                              # %cond.store1228
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 616(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 36
	bltz	a0, .LBB0_941
	j	.LBB0_677
.LBB0_941:                              # %cond.store1231
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 620(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 35
	bltz	a0, .LBB0_942
	j	.LBB0_678
.LBB0_942:                              # %cond.store1234
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 624(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 34
	bltz	a0, .LBB0_943
	j	.LBB0_679
.LBB0_943:                              # %cond.store1237
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 628(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 33
	bgez	a0, .LBB0_1038
	j	.LBB0_680
.LBB0_1038:                             # %cond.store1237
	j	.LBB0_681
.LBB0_944:                              # %cond.store1246
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 640(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 30
	bltz	a0, .LBB0_945
	j	.LBB0_685
.LBB0_945:                              # %cond.store1249
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 644(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 29
	bltz	a0, .LBB0_946
	j	.LBB0_686
.LBB0_946:                              # %cond.store1252
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 648(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 28
	bltz	a0, .LBB0_947
	j	.LBB0_687
.LBB0_947:                              # %cond.store1255
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 652(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 27
	bltz	a0, .LBB0_948
	j	.LBB0_688
.LBB0_948:                              # %cond.store1258
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 656(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 26
	bltz	a0, .LBB0_949
	j	.LBB0_689
.LBB0_949:                              # %cond.store1261
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 660(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 25
	bltz	a0, .LBB0_950
	j	.LBB0_690
.LBB0_950:                              # %cond.store1264
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 664(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 24
	bltz	a0, .LBB0_951
	j	.LBB0_691
.LBB0_951:                              # %cond.store1267
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 668(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 23
	bltz	a0, .LBB0_952
	j	.LBB0_692
.LBB0_952:                              # %cond.store1270
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 672(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 22
	bltz	a0, .LBB0_953
	j	.LBB0_693
.LBB0_953:                              # %cond.store1273
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 676(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 21
	bltz	a0, .LBB0_954
	j	.LBB0_694
.LBB0_954:                              # %cond.store1276
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 680(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 20
	lui	a1, 2
	addi	a1, a1, -904
	add	s5, sp, a1
	bltz	a0, .LBB0_955
	j	.LBB0_695
.LBB0_955:                              # %cond.store1279
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 684(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 19
	bltz	a0, .LBB0_956
	j	.LBB0_696
.LBB0_956:                              # %cond.store1282
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 688(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 18
	bltz	a0, .LBB0_957
	j	.LBB0_697
.LBB0_957:                              # %cond.store1285
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 692(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 17
	bgez	a0, .LBB0_1039
	j	.LBB0_698
.LBB0_1039:                             # %cond.store1285
	j	.LBB0_699
.LBB0_958:                              # %cond.store1294
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 704(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 14
	bltz	a0, .LBB0_959
	j	.LBB0_703
.LBB0_959:                              # %cond.store1297
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 708(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 13
	bltz	a0, .LBB0_960
	j	.LBB0_704
.LBB0_960:                              # %cond.store1300
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 712(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 12
	bltz	a0, .LBB0_961
	j	.LBB0_705
.LBB0_961:                              # %cond.store1303
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 716(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 11
	bltz	a0, .LBB0_962
	j	.LBB0_706
.LBB0_962:                              # %cond.store1306
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 720(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1320(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 10
	bltz	a0, .LBB0_963
	j	.LBB0_707
.LBB0_963:                              # %cond.store1309
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 724(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1200(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 9
	bltz	a0, .LBB0_964
	j	.LBB0_708
.LBB0_964:                              # %cond.store1312
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 728(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 8
	bltz	a0, .LBB0_965
	j	.LBB0_709
.LBB0_965:                              # %cond.store1315
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 732(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 7
	bltz	a0, .LBB0_966
	j	.LBB0_710
.LBB0_966:                              # %cond.store1318
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 736(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 6
	bltz	a0, .LBB0_967
	j	.LBB0_711
.LBB0_967:                              # %cond.store1321
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 740(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 31
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 5
	bltz	a0, .LBB0_968
	j	.LBB0_712
.LBB0_968:                              # %cond.store1324
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 744(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 4
	bltz	a0, .LBB0_969
	j	.LBB0_713
.LBB0_969:                              # %cond.store1327
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 748(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 15
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 3
	bltz	a0, .LBB0_970
	j	.LBB0_714
.LBB0_970:                              # %cond.store1330
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 752(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 2
	bgez	a0, .LBB0_1040
	j	.LBB0_715
.LBB0_1040:                             # %cond.store1330
	j	.LBB0_716
.LBB0_971:                              # %cond.store1342
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 768(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 2
	bnez	a0, .LBB0_972
	j	.LBB0_722
.LBB0_972:                              # %cond.store1345
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 772(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 4
	bnez	a0, .LBB0_973
	j	.LBB0_723
.LBB0_973:                              # %cond.store1348
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 776(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 8
	bnez	a0, .LBB0_974
	j	.LBB0_724
.LBB0_974:                              # %cond.store1351
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 780(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 16
	lui	a1, 1
	addi	a1, a1, 1080
	add	s3, sp, a1
	bnez	a0, .LBB0_975
	j	.LBB0_725
.LBB0_975:                              # %cond.store1354
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 784(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 32
	bnez	a0, .LBB0_976
	j	.LBB0_726
.LBB0_976:                              # %cond.store1357
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 788(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 27
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 64
	bnez	a0, .LBB0_977
	j	.LBB0_727
.LBB0_977:                              # %cond.store1360
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 792(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 128
	bnez	a0, .LBB0_978
	j	.LBB0_728
.LBB0_978:                              # %cond.store1363
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 796(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 13
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 256
	bnez	a0, .LBB0_979
	j	.LBB0_729
.LBB0_979:                              # %cond.store1366
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 800(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 512
	bnez	a0, .LBB0_980
	j	.LBB0_730
.LBB0_980:                              # %cond.store1369
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 804(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 1024
	bnez	a0, .LBB0_981
	j	.LBB0_731
.LBB0_981:                              # %cond.store1372
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 808(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 52
	bltz	a0, .LBB0_982
	j	.LBB0_732
.LBB0_982:                              # %cond.store1375
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 812(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 3
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 51
	bltz	a0, .LBB0_983
	j	.LBB0_733
.LBB0_983:                              # %cond.store1378
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 816(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 50
	bltz	a0, .LBB0_984
	j	.LBB0_734
.LBB0_984:                              # %cond.store1381
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 820(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 49
	bgez	a0, .LBB0_1041
	j	.LBB0_735
.LBB0_1041:                             # %cond.store1381
	j	.LBB0_736
.LBB0_985:                              # %cond.store1390
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 832(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 46
	bltz	a0, .LBB0_986
	j	.LBB0_740
.LBB0_986:                              # %cond.store1393
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 836(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 45
	bltz	a0, .LBB0_987
	j	.LBB0_741
.LBB0_987:                              # %cond.store1396
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 840(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 44
	bltz	a0, .LBB0_988
	j	.LBB0_742
.LBB0_988:                              # %cond.store1399
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 844(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 43
	bltz	a0, .LBB0_989
	j	.LBB0_743
.LBB0_989:                              # %cond.store1402
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 848(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 42
	bltz	a0, .LBB0_990
	j	.LBB0_744
.LBB0_990:                              # %cond.store1405
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 852(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 41
	bltz	a0, .LBB0_991
	j	.LBB0_745
.LBB0_991:                              # %cond.store1408
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 856(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 40
	bltz	a0, .LBB0_992
	j	.LBB0_746
.LBB0_992:                              # %cond.store1411
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 860(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 39
	addi	s3, sp, 2047
	addi	s3, s3, 993
	bltz	a0, .LBB0_993
	j	.LBB0_747
.LBB0_993:                              # %cond.store1414
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 864(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 38
	bltz	a0, .LBB0_994
	j	.LBB0_748
.LBB0_994:                              # %cond.store1417
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 868(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 37
	bltz	a0, .LBB0_995
	j	.LBB0_749
.LBB0_995:                              # %cond.store1420
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 872(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 36
	bltz	a0, .LBB0_996
	j	.LBB0_750
.LBB0_996:                              # %cond.store1423
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 876(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 35
	bltz	a0, .LBB0_997
	j	.LBB0_751
.LBB0_997:                              # %cond.store1426
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 880(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 34
	bltz	a0, .LBB0_998
	j	.LBB0_752
.LBB0_998:                              # %cond.store1429
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 884(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 33
	bgez	a0, .LBB0_1042
	j	.LBB0_753
.LBB0_1042:                             # %cond.store1429
	j	.LBB0_754
.LBB0_999:                              # %cond.store1438
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 896(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 30
	bltz	a0, .LBB0_1000
	j	.LBB0_758
.LBB0_1000:                             # %cond.store1441
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 900(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 29
	bltz	a0, .LBB0_1001
	j	.LBB0_759
.LBB0_1001:                             # %cond.store1444
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 904(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 28
	bltz	a0, .LBB0_1002
	j	.LBB0_760
.LBB0_1002:                             # %cond.store1447
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 908(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 27
	bltz	a0, .LBB0_1003
	j	.LBB0_761
.LBB0_1003:                             # %cond.store1450
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 912(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1921
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 26
	bltz	a0, .LBB0_1004
	j	.LBB0_762
.LBB0_1004:                             # %cond.store1453
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 916(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1793
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 25
	bltz	a0, .LBB0_1005
	j	.LBB0_763
.LBB0_1005:                             # %cond.store1456
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 920(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1665
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 24
	bltz	a0, .LBB0_1006
	j	.LBB0_764
.LBB0_1006:                             # %cond.store1459
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 924(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1537
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 23
	bltz	a0, .LBB0_1007
	j	.LBB0_765
.LBB0_1007:                             # %cond.store1462
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 928(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1409
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 22
	bltz	a0, .LBB0_1008
	j	.LBB0_766
.LBB0_1008:                             # %cond.store1465
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 932(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1281
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 21
	bltz	a0, .LBB0_1009
	j	.LBB0_767
.LBB0_1009:                             # %cond.store1468
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 936(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1153
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 20
	bltz	a0, .LBB0_1010
	j	.LBB0_768
.LBB0_1010:                             # %cond.store1471
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 940(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1025
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 19
	bltz	a0, .LBB0_1011
	j	.LBB0_769
.LBB0_1011:                             # %cond.store1474
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 944(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 897
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 18
	bltz	a0, .LBB0_1012
	j	.LBB0_770
.LBB0_1012:                             # %cond.store1477
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 948(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 769
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1176(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 17
	bgez	a0, .LBB0_1043
	j	.LBB0_771
.LBB0_1043:                             # %cond.store1477
	j	.LBB0_772
.LBB0_1013:                             # %cond.store1486
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 960(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 14
	bltz	a0, .LBB0_1014
	j	.LBB0_776
.LBB0_1014:                             # %cond.store1489
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 964(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 13
	bltz	a0, .LBB0_1015
	j	.LBB0_777
.LBB0_1015:                             # %cond.store1492
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 968(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 12
	bltz	a0, .LBB0_1016
	j	.LBB0_778
.LBB0_1016:                             # %cond.store1495
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 972(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 11
	bltz	a0, .LBB0_1017
	j	.LBB0_779
.LBB0_1017:                             # %cond.store1498
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 976(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 385
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1632(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 10
	bltz	a0, .LBB0_1018
	j	.LBB0_780
.LBB0_1018:                             # %cond.store1501
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 980(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 257
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1752(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 9
	bltz	a0, .LBB0_1019
	j	.LBB0_781
.LBB0_1019:                             # %cond.store1504
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 984(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 129
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1872(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 8
	bltz	a0, .LBB0_1020
	j	.LBB0_782
.LBB0_1020:                             # %cond.store1507
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 988(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1992(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 7
	bltz	a0, .LBB0_1021
	j	.LBB0_783
.LBB0_1021:                             # %cond.store1510
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 992(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1920
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1984(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 6
	bltz	a0, .LBB0_1022
	j	.LBB0_784
.LBB0_1022:                             # %cond.store1513
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 996(sp)                    # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1792
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1864(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 5
	bltz	a0, .LBB0_1023
	j	.LBB0_785
.LBB0_1023:                             # %cond.store1516
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 1000(sp)                   # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1664
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1744(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 4
	bltz	a0, .LBB0_1024
	j	.LBB0_786
.LBB0_1024:                             # %cond.store1519
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 1004(sp)                   # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1536
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1624(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 3
	bltz	a0, .LBB0_1025
	j	.LBB0_787
.LBB0_1025:                             # %cond.store1522
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 1008(sp)                   # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1408
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1504(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 2
	bltz	a0, .LBB0_1026
	j	.LBB0_788
.LBB0_1026:                             # %cond.store1525
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 1012(sp)                   # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1280
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1384(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 1
	bltz	a0, .LBB0_1027
	j	.LBB0_789
.LBB0_1027:                             # %cond.store1528
	.loc	1 0 44                          # k135114449651120.py:0:44
	flw	fa0, 1016(sp)                   # 4-byte Folded Reload
	.loc	1 10 44                         # k135114449651120.py:10:44
	call	__truncsfbf2
	addi	a0, sp, 1152
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 12
	addi	a2, a2, 1120
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1264(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	bgez	s4, .LBB0_1044
	j	.LBB0_790
.LBB0_1044:                             # %cond.store1528
	j	.LBB0_791
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused_cat_neg_slice_transpose_view_2, .Lfunc_end0-triton_poi_fused_cat_neg_slice_transpose_view_2
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
	.asciz	"k135114449651120.py"           # string offset=7 ; k135114449651120.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
