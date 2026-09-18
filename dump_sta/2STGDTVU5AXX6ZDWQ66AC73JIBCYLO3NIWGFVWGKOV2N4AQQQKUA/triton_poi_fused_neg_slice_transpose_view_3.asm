	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused_neg_slice_transpose_view_3 # -- Begin function triton_poi_fused_neg_slice_transpose_view_3
	.p2align	2
	.type	triton_poi_fused_neg_slice_transpose_view_3,@function
triton_poi_fused_neg_slice_transpose_view_3: # @triton_poi_fused_neg_slice_transpose_view_3
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294379184.py"
	.loc	1 2 0                           # k135114294379184.py:2:0
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
	lui	a2, 6
	addi	a2, a2, -112
	sub	sp, sp, a2
	csrr	a2, vlenb
	li	a4, 120
	mul	a2, a2, a4
	sub	sp, sp, a2
	andi	sp, sp, -128
	mv	t1, a1
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114294379184.py:4:33
	slli	a5, a3, 7
	li	a1, 32
	lui	a2, 599186
	.loc	1 5 23                          # k135114294379184.py:5:23
	vsetvli	zero, a1, e32, m8, ta, ma
	vid.v	v8
	addi	a4, a2, 1171
	vor.vx	v0, v8, a5
	csrr	a2, vlenb
	li	a3, 88
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 6
	addi	a3, a3, 1728
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 9 19                          # k135114294379184.py:9:19
	vmulh.vx	v16, v0, a4
	vadd.vv	v16, v16, v0
	vsra.vi	v16, v16, 8
	vsrl.vi	v24, v16, 31
	vadd.vv	v16, v16, v24
	.loc	1 5 23                          # k135114294379184.py:5:23
	vmv.v.x	v24, a5
	.loc	1 8 21                          # k135114294379184.py:8:21
	vsra.vi	v24, v24, 31
	vsrl.vi	v8, v24, 27
	csrr	a2, vlenb
	li	a3, 80
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 6
	addi	a3, a3, 1728
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vadd.vv	v24, v0, v8
	vsra.vi	v8, v24, 5
	csrr	a2, vlenb
	li	a3, 48
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 6
	addi	a3, a3, 1728
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 8 27 is_stmt 0                # k135114294379184.py:8:27
	vmulh.vx	v0, v8, a4
	vadd.vv	v0, v0, v8
	vsra.vi	v0, v0, 3
	vsrl.vi	v8, v0, 31
	vadd.vv	v8, v0, v8
	li	a3, -32
	li	a2, 14
	.loc	1 7 19 is_stmt 1                # k135114294379184.py:7:19
	vand.vx	v24, v24, a3
	csrr	a6, vlenb
	li	a7, 88
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 1728
	add	a6, a6, a7
	vl8r.v	v0, (a6)                        # vscale x 64-byte Folded Reload
	vsub.vv	v24, v0, v24
	.loc	1 11 43                         # k135114294379184.py:11:43
	vsll.vi	v16, v16, 6
	csrr	a6, vlenb
	slli	a6, a6, 4
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 1728
	add	a6, a6, a7
	vs8r.v	v24, (a6)                       # vscale x 64-byte Folded Spill
	.loc	1 11 35 is_stmt 0               # k135114294379184.py:11:35
	vadd.vv	v16, v24, v16
	csrr	a6, vlenb
	li	a7, 48
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 1728
	add	a6, a6, a7
	vl8r.v	v24, (a6)                       # vscale x 64-byte Folded Reload
	.loc	1 8 27 is_stmt 1                # k135114294379184.py:8:27
	vnmsub.vx	v8, a2, v24
	.loc	1 11 52                         # k135114294379184.py:11:52
	vsll.vi	v8, v8, 7
	.loc	1 11 40 is_stmt 0               # k135114294379184.py:11:40
	vadd.vv	v16, v16, v8
	li	a6, 96
	li	a7, 64
	li	t0, 896
	vid.v	v0
	.loc	1 5 23 is_stmt 1                # k135114294379184.py:5:23
	vadd.vx	v8, v0, a6
	vor.vx	v24, v8, a5
	vadd.vx	v8, v0, a7
	vor.vx	v0, v8, a5
	csrr	a6, vlenb
	li	a7, 112
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 1728
	add	a6, a6, a7
	vs8r.v	v24, (a6)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114294379184.py:6:21
	vmslt.vx	v8, v24, t0
	csrr	a6, vlenb
	li	a7, 104
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 1728
	add	a6, a6, a7
	vs8r.v	v0, (a6)                        # vscale x 64-byte Folded Spill
	vmslt.vx	v9, v0, t0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	csrr	a6, vlenb
	li	a7, 96
	mul	a6, a6, a7
	add	a6, sp, a6
	lui	a7, 6
	addi	a7, a7, 1728
	add	a6, a6, a7
	vs1r.v	v9, (a6)                        # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294379184.py:5:23
	vsetvli	zero, a1, e32, m8, ta, ma
	vid.v	v8
	vadd.vx	v8, v8, a1
	vor.vx	v0, v8, a5
	csrr	a5, vlenb
	li	a6, 88
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 6
	addi	a6, a6, 1728
	add	a5, a5, a6
	vl8r.v	v24, (a5)                       # vscale x 64-byte Folded Reload
	.loc	1 6 21                          # k135114294379184.py:6:21
	vmslt.vx	v9, v24, t0
	vmslt.vx	v8, v0, t0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	csrr	a5, vlenb
	li	a6, 96
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 6
	addi	a6, a6, 1728
	add	a5, a5, a6
	vl1r.v	v8, (a5)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v9, v8, 8
	csrr	a5, vlenb
	li	a6, 40
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 6
	addi	a6, a6, 1728
	add	a5, a5, a6
	vs1r.v	v9, (a5)                        # vscale x 8-byte Folded Spill
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	a5, v9
	.loc	1 11 48 is_stmt 0               # k135114294379184.py:11:48
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vx	v16, v16, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	andi	a6, a5, 1
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v16
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v8, a0
	beqz	a6, .LBB0_2
# %bb.1:                                # %cond.load
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs10, zero
	fmv.w.x	fs8, a6
	fmv.s	ft0, fs10
	fsw	fs10, 808(sp)                   # 4-byte Folded Spill
	fmv.s	ft1, fs10
	fsw	fs10, 824(sp)                   # 4-byte Folded Spill
	fmv.s	ft3, fs10
	fsw	fs10, 848(sp)                   # 4-byte Folded Spill
	fmv.s	ft2, fs10
	fsw	fs10, 872(sp)                   # 4-byte Folded Spill
	fmv.s	ft4, fs10
	fsw	fs10, 896(sp)                   # 4-byte Folded Spill
	fmv.s	ft5, fs10
	fsw	fs10, 920(sp)                   # 4-byte Folded Spill
	fmv.s	ft6, fs10
	fsw	fs10, 952(sp)                   # 4-byte Folded Spill
	fmv.s	ft7, fs10
	fsw	fs10, 976(sp)                   # 4-byte Folded Spill
	fmv.s	fs1, fs10
	fsw	fs10, 1008(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1000(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1016(sp)                  # 4-byte Folded Spill
	fmv.s	fa7, fs10
	fsw	fs10, 1024(sp)                  # 4-byte Folded Spill
	fmv.s	ft8, fs10
	fsw	fs10, 1032(sp)                  # 4-byte Folded Spill
	fmv.s	ft9, fs10
	fsw	fs10, 1040(sp)                  # 4-byte Folded Spill
	fmv.s	fa0, fs10
	fsw	fs10, 1048(sp)                  # 4-byte Folded Spill
	fmv.s	ft10, fs10
	fsw	fs10, 1056(sp)                  # 4-byte Folded Spill
	fmv.s	ft11, fs10
	fmv.s	fa1, fs10
	fsw	fs10, 448(sp)                   # 4-byte Folded Spill
	fsw	fs10, 456(sp)                   # 4-byte Folded Spill
	fsw	fs10, 464(sp)                   # 4-byte Folded Spill
	fsw	fs10, 472(sp)                   # 4-byte Folded Spill
	fsw	fs10, 480(sp)                   # 4-byte Folded Spill
	fsw	fs10, 488(sp)                   # 4-byte Folded Spill
	fsw	fs10, 496(sp)                   # 4-byte Folded Spill
	fsw	fs10, 504(sp)                   # 4-byte Folded Spill
	fsw	fs10, 512(sp)                   # 4-byte Folded Spill
	fsw	fs10, 520(sp)                   # 4-byte Folded Spill
	fsw	fs10, 528(sp)                   # 4-byte Folded Spill
	fsw	fs10, 536(sp)                   # 4-byte Folded Spill
	fsw	fs10, 544(sp)                   # 4-byte Folded Spill
	fsw	fs10, 552(sp)                   # 4-byte Folded Spill
	fsw	fs10, 560(sp)                   # 4-byte Folded Spill
	fsw	fs10, 568(sp)                   # 4-byte Folded Spill
	fsw	fs10, 576(sp)                   # 4-byte Folded Spill
	fsw	fs10, 584(sp)                   # 4-byte Folded Spill
	fsw	fs10, 592(sp)                   # 4-byte Folded Spill
	fsw	fs10, 600(sp)                   # 4-byte Folded Spill
	fsw	fs10, 608(sp)                   # 4-byte Folded Spill
	fsw	fs10, 616(sp)                   # 4-byte Folded Spill
	fsw	fs10, 624(sp)                   # 4-byte Folded Spill
	fsw	fs10, 632(sp)                   # 4-byte Folded Spill
	fsw	fs10, 640(sp)                   # 4-byte Folded Spill
	fsw	fs10, 648(sp)                   # 4-byte Folded Spill
	fsw	fs10, 656(sp)                   # 4-byte Folded Spill
	fsw	fs10, 664(sp)                   # 4-byte Folded Spill
	fsw	fs10, 672(sp)                   # 4-byte Folded Spill
	fsw	fs10, 680(sp)                   # 4-byte Folded Spill
	fsw	fs10, 1088(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1064(sp)                  # 4-byte Folded Spill
	fmv.s	fa5, fs10
	fsw	fs10, 1072(sp)                  # 4-byte Folded Spill
	fsw	fs10, 368(sp)                   # 4-byte Folded Spill
	fsw	fs10, 376(sp)                   # 4-byte Folded Spill
	fsw	fs10, 384(sp)                   # 4-byte Folded Spill
	fsw	fs10, 392(sp)                   # 4-byte Folded Spill
	fsw	fs10, 400(sp)                   # 4-byte Folded Spill
	fsw	fs10, 408(sp)                   # 4-byte Folded Spill
	fsw	fs10, 416(sp)                   # 4-byte Folded Spill
	fsw	fs10, 424(sp)                   # 4-byte Folded Spill
	fsw	fs10, 432(sp)                   # 4-byte Folded Spill
	fmv.s	fa3, fs10
	fsw	fs10, 1096(sp)                  # 4-byte Folded Spill
	fmv.s	fa4, fs10
	fmv.s	fa2, fs10
	fsw	fs10, 1104(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1112(sp)                  # 4-byte Folded Spill
	fmv.s	fs3, fs10
	fmv.s	fs6, fs10
	fmv.s	fs7, fs10
	fmv.s	fs9, fs10
	fmv.s	fs11, fs10
	fmv.s	fs0, fs10
	fmv.s	fs4, fs10
	fmv.s	fs5, fs10
	fsw	fs10, 1120(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1128(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1136(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1080(sp)                  # 4-byte Folded Spill
	fmv.s	fs2, fs10
	fsw	fs10, 688(sp)                   # 4-byte Folded Spill
	fsw	fs10, 696(sp)                   # 4-byte Folded Spill
	fsw	fs10, 704(sp)                   # 4-byte Folded Spill
	fsw	fs10, 712(sp)                   # 4-byte Folded Spill
	fsw	fs10, 720(sp)                   # 4-byte Folded Spill
	fsw	fs10, 728(sp)                   # 4-byte Folded Spill
	fsw	fs10, 736(sp)                   # 4-byte Folded Spill
	fsw	fs10, 744(sp)                   # 4-byte Folded Spill
	fsw	fs10, 752(sp)                   # 4-byte Folded Spill
	fsw	fs10, 760(sp)                   # 4-byte Folded Spill
	fsw	fs10, 768(sp)                   # 4-byte Folded Spill
	fsw	fs10, 776(sp)                   # 4-byte Folded Spill
	fsw	fs10, 784(sp)                   # 4-byte Folded Spill
	fsw	fs10, 792(sp)                   # 4-byte Folded Spill
	fsw	fs10, 800(sp)                   # 4-byte Folded Spill
	fsw	fs10, 816(sp)                   # 4-byte Folded Spill
	fsw	fs10, 832(sp)                   # 4-byte Folded Spill
	fsw	fs10, 840(sp)                   # 4-byte Folded Spill
	fsw	fs10, 856(sp)                   # 4-byte Folded Spill
	fsw	fs10, 864(sp)                   # 4-byte Folded Spill
	fsw	fs10, 880(sp)                   # 4-byte Folded Spill
	fsw	fs10, 888(sp)                   # 4-byte Folded Spill
	fsw	fs10, 904(sp)                   # 4-byte Folded Spill
	fsw	fs10, 912(sp)                   # 4-byte Folded Spill
	fsw	fs10, 928(sp)                   # 4-byte Folded Spill
	fsw	fs10, 936(sp)                   # 4-byte Folded Spill
	fsw	fs10, 944(sp)                   # 4-byte Folded Spill
	fsw	fs10, 960(sp)                   # 4-byte Folded Spill
	fsw	fs10, 968(sp)                   # 4-byte Folded Spill
	fsw	fs10, 984(sp)                   # 4-byte Folded Spill
	fsw	fs10, 992(sp)                   # 4-byte Folded Spill
	fmv.s	fa6, fs10
	andi	a6, a5, 2
	bnez	a6, .LBB0_3
	j	.LBB0_4
.LBB0_2:
	.loc	1 0 57                          # k135114294379184.py:0:57
	fmv.w.x	fs8, zero
	fmv.s	fs10, fs8
	fmv.s	ft0, fs8
	fsw	fs8, 808(sp)                    # 4-byte Folded Spill
	fmv.s	ft1, fs8
	fsw	fs8, 824(sp)                    # 4-byte Folded Spill
	fmv.s	ft3, fs8
	fsw	fs8, 848(sp)                    # 4-byte Folded Spill
	fmv.s	ft2, fs8
	fsw	fs8, 872(sp)                    # 4-byte Folded Spill
	fmv.s	ft4, fs8
	fsw	fs8, 896(sp)                    # 4-byte Folded Spill
	fmv.s	ft5, fs8
	fsw	fs8, 920(sp)                    # 4-byte Folded Spill
	fmv.s	ft6, fs8
	fsw	fs8, 952(sp)                    # 4-byte Folded Spill
	fmv.s	ft7, fs8
	fsw	fs8, 976(sp)                    # 4-byte Folded Spill
	fmv.s	fs1, fs8
	fsw	fs8, 1008(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1000(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1016(sp)                   # 4-byte Folded Spill
	fmv.s	fa7, fs8
	fsw	fs8, 1024(sp)                   # 4-byte Folded Spill
	fmv.s	ft8, fs8
	fsw	fs8, 1032(sp)                   # 4-byte Folded Spill
	fmv.s	ft9, fs8
	fsw	fs8, 1040(sp)                   # 4-byte Folded Spill
	fmv.s	fa0, fs8
	fsw	fs8, 1048(sp)                   # 4-byte Folded Spill
	fmv.s	ft10, fs8
	fsw	fs8, 1056(sp)                   # 4-byte Folded Spill
	fmv.s	ft11, fs8
	fmv.s	fa1, fs8
	fsw	fs8, 448(sp)                    # 4-byte Folded Spill
	fsw	fs8, 456(sp)                    # 4-byte Folded Spill
	fsw	fs8, 464(sp)                    # 4-byte Folded Spill
	fsw	fs8, 472(sp)                    # 4-byte Folded Spill
	fsw	fs8, 480(sp)                    # 4-byte Folded Spill
	fsw	fs8, 488(sp)                    # 4-byte Folded Spill
	fsw	fs8, 496(sp)                    # 4-byte Folded Spill
	fsw	fs8, 504(sp)                    # 4-byte Folded Spill
	fsw	fs8, 512(sp)                    # 4-byte Folded Spill
	fsw	fs8, 520(sp)                    # 4-byte Folded Spill
	fsw	fs8, 528(sp)                    # 4-byte Folded Spill
	fsw	fs8, 536(sp)                    # 4-byte Folded Spill
	fsw	fs8, 544(sp)                    # 4-byte Folded Spill
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
	fsw	fs8, 1088(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1064(sp)                   # 4-byte Folded Spill
	fmv.s	fa5, fs8
	fsw	fs8, 1072(sp)                   # 4-byte Folded Spill
	fsw	fs8, 368(sp)                    # 4-byte Folded Spill
	fsw	fs8, 376(sp)                    # 4-byte Folded Spill
	fsw	fs8, 384(sp)                    # 4-byte Folded Spill
	fsw	fs8, 392(sp)                    # 4-byte Folded Spill
	fsw	fs8, 400(sp)                    # 4-byte Folded Spill
	fsw	fs8, 408(sp)                    # 4-byte Folded Spill
	fsw	fs8, 416(sp)                    # 4-byte Folded Spill
	fsw	fs8, 424(sp)                    # 4-byte Folded Spill
	fsw	fs8, 432(sp)                    # 4-byte Folded Spill
	fmv.s	fa3, fs8
	fsw	fs8, 1096(sp)                   # 4-byte Folded Spill
	fmv.s	fa4, fs8
	fmv.s	fa2, fs8
	fsw	fs8, 1104(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1112(sp)                   # 4-byte Folded Spill
	fmv.s	fs3, fs8
	fmv.s	fs6, fs8
	fmv.s	fs7, fs8
	fmv.s	fs9, fs8
	fmv.s	fs11, fs8
	fmv.s	fs0, fs8
	fmv.s	fs4, fs8
	fmv.s	fs5, fs8
	fsw	fs8, 1120(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1128(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1136(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1080(sp)                   # 4-byte Folded Spill
	fmv.s	fs2, fs8
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
	fsw	fs8, 816(sp)                    # 4-byte Folded Spill
	fsw	fs8, 832(sp)                    # 4-byte Folded Spill
	fsw	fs8, 840(sp)                    # 4-byte Folded Spill
	fsw	fs8, 856(sp)                    # 4-byte Folded Spill
	fsw	fs8, 864(sp)                    # 4-byte Folded Spill
	fsw	fs8, 880(sp)                    # 4-byte Folded Spill
	fsw	fs8, 888(sp)                    # 4-byte Folded Spill
	fsw	fs8, 904(sp)                    # 4-byte Folded Spill
	fsw	fs8, 912(sp)                    # 4-byte Folded Spill
	fsw	fs8, 928(sp)                    # 4-byte Folded Spill
	fsw	fs8, 936(sp)                    # 4-byte Folded Spill
	fsw	fs8, 944(sp)                    # 4-byte Folded Spill
	fsw	fs8, 960(sp)                    # 4-byte Folded Spill
	fsw	fs8, 968(sp)                    # 4-byte Folded Spill
	fsw	fs8, 984(sp)                    # 4-byte Folded Spill
	fsw	fs8, 992(sp)                    # 4-byte Folded Spill
	fmv.s	fa6, fs8
	.loc	1 11 57                         # k135114294379184.py:11:57
	andi	a6, a5, 2
	beqz	a6, .LBB0_4
.LBB0_3:                                # %cond.load1
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	vmv.x.s	a6, v24
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fs10, a6
.LBB0_4:                                # %else2
	andi	a6, a5, 4
	bnez	a6, .LBB0_40
# %bb.5:                                # %else5
	andi	a6, a5, 8
	fsw	ft0, 224(sp)                    # 4-byte Folded Spill
	bnez	a6, .LBB0_41
.LBB0_6:                                # %else8
	andi	a7, a5, 16
	lui	a6, 6
	addi	a6, a6, -88
	add	a6, sp, a6
	fmv.s	ft0, fa0
	bnez	a7, .LBB0_42
.LBB0_7:                                # %else11
	andi	a7, a5, 32
	fsw	ft1, 232(sp)                    # 4-byte Folded Spill
	bnez	a7, .LBB0_43
.LBB0_8:                                # %else14
	andi	a7, a5, 64
	fmv.s	ft1, fa1
	bnez	a7, .LBB0_44
.LBB0_9:                                # %else17
	.loc	1 0 57                          # k135114294379184.py:0:57
	fmv.s	fa1, ft10
	.loc	1 11 57                         # k135114294379184.py:11:57
	andi	a7, a5, 128
	fsw	ft3, 240(sp)                    # 4-byte Folded Spill
	bnez	a7, .LBB0_45
.LBB0_10:                               # %else20
	andi	a7, a5, 256
	fmv.s	ft3, fs3
	bnez	a7, .LBB0_46
.LBB0_11:                               # %else23
	andi	a7, a5, 512
	fsw	ft2, 248(sp)                    # 4-byte Folded Spill
	bnez	a7, .LBB0_47
.LBB0_12:                               # %else26
	andi	a7, a5, 1024
	fmv.s	ft2, ft11
	bnez	a7, .LBB0_48
.LBB0_13:                               # %else29
	slli	a7, a5, 52
	fsw	ft4, 256(sp)                    # 4-byte Folded Spill
	bltz	a7, .LBB0_49
.LBB0_14:                               # %else32
	slli	a7, a5, 51
	fmv.s	ft4, fs6
	bltz	a7, .LBB0_50
.LBB0_15:                               # %else35
	slli	a7, a5, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	fsw	ft5, 264(sp)                    # 4-byte Folded Spill
	bltz	a7, .LBB0_51
.LBB0_16:                               # %else38
	slli	a7, a5, 49
	fmv.s	ft5, fs7
	bltz	a7, .LBB0_52
.LBB0_17:                               # %else41
	slli	a7, a5, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	fsw	ft6, 272(sp)                    # 4-byte Folded Spill
	bltz	a7, .LBB0_53
.LBB0_18:                               # %else44
	slli	a7, a5, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v24, a0
	fmv.s	ft6, fs9
	bltz	a7, .LBB0_54
.LBB0_19:                               # %else47
	slli	a7, a5, 46
	fsw	ft7, 280(sp)                    # 4-byte Folded Spill
	bltz	a7, .LBB0_55
.LBB0_20:                               # %else50
	slli	a7, a5, 45
	fmv.s	ft7, fs11
	bgez	a7, .LBB0_22
.LBB0_21:                               # %cond.load52
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 2
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs1, a7
.LBB0_22:                               # %else53
	slli	a7, a5, 44
	csrr	t0, vlenb
	li	t2, 80
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v0, v8
	fsw	fs1, 288(sp)                    # 4-byte Folded Spill
	bgez	a7, .LBB0_24
# %bb.23:                               # %cond.load55
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 3
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs1, a7
	fsw	fs1, 1008(sp)                   # 4-byte Folded Spill
.LBB0_24:                               # %else56
	slli	a7, a5, 43
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v8, v24, 5
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vs8r.v	v16, (t0)                       # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vs8r.v	v24, (t0)                       # vscale x 64-byte Folded Spill
	bgez	a7, .LBB0_26
# %bb.25:                               # %cond.load58
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 120(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs1, a7
	fsw	fs1, 1000(sp)                   # 4-byte Folded Spill
.LBB0_26:                               # %else59
	.loc	1 0 57                          # k135114294379184.py:0:57
	fmv.s	fs1, fs2
	vsetvli	zero, a1, e32, m8, ta, ma
	vmulh.vx	v24, v8, a4
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a5, 42
	vmulh.vx	v16, v0, a4
	csrr	t0, vlenb
	li	t2, 72
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vs8r.v	v0, (t0)                        # vscale x 64-byte Folded Spill
	fsw	fa6, 220(sp)                    # 4-byte Folded Spill
	bgez	a7, .LBB0_28
# %bb.27:                               # %cond.load61
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -128
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	csrr	a7, vlenb
	li	t0, 72
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	ld	a6, 0(a6)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa6, a6
	fsw	fa6, 1016(sp)                   # 4-byte Folded Spill
.LBB0_28:                               # %else62
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v24, v8
	vadd.vv	v0, v16, v0
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a5, 41
	lui	a6, 5
	addi	a6, a6, 1872
	add	a6, sp, a6
	fmv.s	fa6, fs4
	bgez	a7, .LBB0_30
# %bb.29:                               # %cond.load64
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -256
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 2016(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa7, a7
.LBB0_30:                               # %else65
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v24, v24, 3
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a5, 40
	vsra.vi	v0, v0, 8
	csrr	t0, vlenb
	li	t2, 56
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	fsw	fa7, 304(sp)                    # 4-byte Folded Spill
	bgez	a7, .LBB0_32
# %bb.31:                               # %cond.load67
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -384
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1896(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa7, a7
	fsw	fa7, 1024(sp)                   # 4-byte Folded Spill
.LBB0_32:                               # %else68
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vsrl.vi	v16, v24, 31
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a5, 39
	vsrl.vi	v8, v0, 31
	fmv.s	fa7, fs5
	bgez	a7, .LBB0_34
# %bb.33:                               # %cond.load70
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -512
	add	a7, sp, a7
	csrr	t0, vlenb
	slli	t0, t0, 5
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vs8r.v	v16, (t0)                       # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	csrr	a7, vlenb
	slli	a7, a7, 5
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vl8r.v	v16, (a7)                       # vscale x 64-byte Folded Reload
	ld	a7, 1776(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft8, a7
.LBB0_34:                               # %else71
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v24, v16
	vadd.vv	v0, v0, v8
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a5, 38
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a3
	fsw	ft8, 312(sp)                    # 4-byte Folded Spill
	bgez	a7, .LBB0_36
# %bb.35:                               # %cond.load73
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -640
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 1656(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft8, a7
	fsw	ft8, 1032(sp)                   # 4-byte Folded Spill
.LBB0_36:                               # %else74
	.loc	1 0 57                          # k135114294379184.py:0:57
	csrr	a7, vlenb
	li	t0, 72
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vl8r.v	v16, (a7)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v16, v16, v8
	csrr	a7, vlenb
	li	t0, 56
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	vnmsub.vx	v24, a2, v8
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a5, 37
	vsll.vi	v8, v0, 6
	fmv.s	ft8, fs0
	bgez	a7, .LBB0_38
# %bb.37:                               # %cond.load76
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -768
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a7)
	ld	a7, 1536(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft9, a7
.LBB0_38:                               # %else77
	.loc	1 0 57                          # k135114294379184.py:0:57
	csrr	a7, vlenb
	li	t0, 24
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vs8r.v	v16, (a7)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v16, v8
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a5, 36
	vsll.vi	v8, v24, 7
	fsw	ft9, 320(sp)                    # 4-byte Folded Spill
	bgez	a7, .LBB0_56
# %bb.39:                               # %cond.load79
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -896
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 1416(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft9, a7
	fsw	ft9, 1040(sp)                   # 4-byte Folded Spill
	slli	a7, a5, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	flw	ft9, 1128(sp)                   # 4-byte Folded Reload
	bltz	a7, .LBB0_57
	j	.LBB0_58
.LBB0_40:                               # %cond.load4
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v8, 2
	vmv.x.s	a6, v24
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	andi	a6, a5, 8
	fsw	ft0, 224(sp)                    # 4-byte Folded Spill
	beqz	a6, .LBB0_6
.LBB0_41:                               # %cond.load7
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v8, 3
	vmv.x.s	a6, v24
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 808(sp)                    # 4-byte Folded Spill
	andi	a7, a5, 16
	lui	a6, 6
	addi	a6, a6, -88
	add	a6, sp, a6
	fmv.s	ft0, fa0
	beqz	a7, .LBB0_7
.LBB0_42:                               # %cond.load10
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 1536
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1656(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft1, a7
	andi	a7, a5, 32
	fsw	ft1, 232(sp)                    # 4-byte Folded Spill
	beqz	a7, .LBB0_8
.LBB0_43:                               # %cond.load13
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 1408
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1536(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft1, a7
	fsw	ft1, 824(sp)                    # 4-byte Folded Spill
	andi	a7, a5, 64
	fmv.s	ft1, fa1
	beqz	a7, .LBB0_9
.LBB0_44:                               # %cond.load16
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 1280
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1416(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft3, a7
	fmv.s	fa1, ft10
	andi	a7, a5, 128
	fsw	ft3, 240(sp)                    # 4-byte Folded Spill
	beqz	a7, .LBB0_10
.LBB0_45:                               # %cond.load19
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 1152
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1296(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft3, a7
	fsw	ft3, 848(sp)                    # 4-byte Folded Spill
	andi	a7, a5, 256
	fmv.s	ft3, fs3
	beqz	a7, .LBB0_11
.LBB0_46:                               # %cond.load22
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a7, 25
	slli	a7, a7, 10
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1176(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft2, a7
	andi	a7, a5, 512
	fsw	ft2, 248(sp)                    # 4-byte Folded Spill
	beqz	a7, .LBB0_12
.LBB0_47:                               # %cond.load25
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 896
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1056(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft2, a7
	fsw	ft2, 872(sp)                    # 4-byte Folded Spill
	andi	a7, a5, 1024
	fmv.s	ft2, ft11
	beqz	a7, .LBB0_13
.LBB0_48:                               # %cond.load28
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 768
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 936(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft4, a7
	slli	a7, a5, 52
	fsw	ft4, 256(sp)                    # 4-byte Folded Spill
	bgez	a7, .LBB0_14
.LBB0_49:                               # %cond.load31
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 640
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 816(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft4, a7
	fsw	ft4, 896(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 51
	fmv.s	ft4, fs6
	bgez	a7, .LBB0_15
.LBB0_50:                               # %cond.load34
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 512
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 696(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft5, a7
	slli	a7, a5, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	fsw	ft5, 264(sp)                    # 4-byte Folded Spill
	bgez	a7, .LBB0_16
.LBB0_51:                               # %cond.load37
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 384
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 576(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft5, a7
	fsw	ft5, 920(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 49
	fmv.s	ft5, fs7
	bgez	a7, .LBB0_17
.LBB0_52:                               # %cond.load40
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 256
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 456(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft6, a7
	slli	a7, a5, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	fsw	ft6, 272(sp)                    # 4-byte Folded Spill
	bgez	a7, .LBB0_18
.LBB0_53:                               # %cond.load43
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, 128
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vse64.v	v8, (a7)
	ld	a7, 336(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft6, a7
	fsw	ft6, 952(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v24, a0
	fmv.s	ft6, fs9
	bgez	a7, .LBB0_19
.LBB0_54:                               # %cond.load46
	vmv.x.s	a7, v16
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft7, a7
	slli	a7, a5, 46
	fsw	ft7, 280(sp)                    # 4-byte Folded Spill
	bgez	a7, .LBB0_20
.LBB0_55:                               # %cond.load49
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v16, 1
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft7, a7
	fsw	ft7, 976(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 45
	fmv.s	ft7, fs11
	bltz	a7, .LBB0_21
	j	.LBB0_22
.LBB0_56:
	.loc	1 0 57                          # k135114294379184.py:0:57
	csrr	a7, vlenb
	li	t0, 96
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vl8r.v	v24, (a7)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a5, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	flw	ft9, 1128(sp)                   # 4-byte Folded Reload
	bgez	a7, .LBB0_58
.LBB0_57:                               # %cond.load82
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a7, 23
	slli	a7, a7, 10
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 1296(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft0, a7
.LBB0_58:                               # %else83
	.loc	1 0 57                          # k135114294379184.py:0:57
	flw	fa0, 1088(sp)                   # 4-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a5, 34
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vx	v16, v8, a1
	fsw	ft0, 328(sp)                    # 4-byte Folded Spill
	bltz	a7, .LBB0_97
# %bb.59:                               # %else86
	slli	a7, a5, 33
	flw	ft10, 1120(sp)                  # 4-byte Folded Reload
	bltz	a7, .LBB0_98
.LBB0_60:                               # %else89
	slli	a7, a5, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	fsw	fa1, 336(sp)                    # 4-byte Folded Spill
	bltz	a7, .LBB0_99
.LBB0_61:                               # %else92
	slli	a7, a5, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, a0
	flw	ft11, 1136(sp)                  # 4-byte Folded Reload
	bltz	a7, .LBB0_100
.LBB0_62:                               # %else95
	slli	a7, a5, 30
	bltz	a7, .LBB0_101
.LBB0_63:                               # %else98
	slli	a7, a5, 29
	bltz	a7, .LBB0_102
.LBB0_64:                               # %else101
	slli	a7, a5, 28
	bltz	a7, .LBB0_103
.LBB0_65:                               # %else104
	slli	a7, a5, 27
	bltz	a7, .LBB0_104
.LBB0_66:                               # %else107
	slli	a7, a5, 26
	bltz	a7, .LBB0_105
.LBB0_67:                               # %else110
	.loc	1 0 57                          # k135114294379184.py:0:57
	flw	fa1, 1096(sp)                   # 4-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a5, 25
	bltz	a7, .LBB0_106
.LBB0_68:                               # %else113
	slli	a7, a5, 24
	bltz	a7, .LBB0_107
.LBB0_69:                               # %else116
	slli	a7, a5, 23
	bltz	a7, .LBB0_108
.LBB0_70:                               # %else119
	slli	a7, a5, 22
	bltz	a7, .LBB0_109
.LBB0_71:                               # %else122
	slli	a7, a5, 21
	bltz	a7, .LBB0_110
.LBB0_72:                               # %else125
	slli	a6, a5, 20
	lui	a7, 5
	addi	a7, a7, -264
	add	a7, sp, a7
	bltz	a6, .LBB0_111
.LBB0_73:                               # %else128
	slli	a6, a5, 19
	bltz	a6, .LBB0_112
.LBB0_74:                               # %else131
	slli	a6, a5, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bltz	a6, .LBB0_113
.LBB0_75:                               # %else134
	slli	a6, a5, 17
	bltz	a6, .LBB0_114
.LBB0_76:                               # %else137
	slli	a6, a5, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bltz	a6, .LBB0_115
.LBB0_77:                               # %else140
	slli	a6, a5, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v24, a0
	bltz	a6, .LBB0_116
.LBB0_78:                               # %else143
	slli	a6, a5, 14
	bltz	a6, .LBB0_117
.LBB0_79:                               # %else146
	slli	a6, a5, 13
	bgez	a6, .LBB0_81
.LBB0_80:                               # %cond.load148
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 576(sp)                    # 4-byte Folded Spill
.LBB0_81:                               # %else149
	slli	a6, a5, 12
	csrr	t0, vlenb
	li	t2, 104
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	csrr	t0, vlenb
	li	t2, 80
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v8, v16
	bgez	a6, .LBB0_83
# %bb.82:                               # %cond.load151
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 584(sp)                    # 4-byte Folded Spill
.LBB0_83:                               # %else152
	slli	a6, a5, 11
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v8, v16, 5
	csrr	t0, vlenb
	slli	t0, t0, 5
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vs8r.v	v16, (t0)                       # vscale x 64-byte Folded Spill
	bgez	a6, .LBB0_85
# %bb.84:                               # %cond.load154
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a6, 21
	slli	a6, a6, 10
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1320(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 592(sp)                    # 4-byte Folded Spill
.LBB0_85:                               # %else155
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vmulh.vx	v24, v8, a4
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a6, a5, 10
	csrr	t0, vlenb
	li	t2, 104
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	vmulh.vx	v16, v16, a4
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vs8r.v	v0, (t0)                        # vscale x 64-byte Folded Spill
	bgez	a6, .LBB0_87
# %bb.86:                               # %cond.load157
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 896
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1200(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 600(sp)                    # 4-byte Folded Spill
.LBB0_87:                               # %else158
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v24, v8
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a6, a5, 9
	csrr	t0, vlenb
	li	t2, 104
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v16, v16, v0
	bgez	a6, .LBB0_89
# %bb.88:                               # %cond.load160
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 768
	add	a6, sp, a6
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1080(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 608(sp)                    # 4-byte Folded Spill
.LBB0_89:                               # %else161
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v24, v24, 3
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a6, a5, 8
	vsra.vi	v0, v16, 8
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	bgez	a6, .LBB0_91
# %bb.90:                               # %cond.load163
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 640
	add	a6, sp, a6
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a6)
	ld	a6, 960(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 616(sp)                    # 4-byte Folded Spill
.LBB0_91:                               # %else164
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vsrl.vi	v8, v24, 31
	csrr	a6, vlenb
	slli	a6, a6, 3
	add	a6, sp, a6
	lui	t0, 6
	addi	t0, t0, 1728
	add	a6, a6, t0
	vs8r.v	v8, (a6)                        # vscale x 64-byte Folded Spill
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a6, a5, 7
	vsrl.vi	v8, v0, 31
	lui	t0, 6
	addi	t0, t0, 1728
	add	t0, sp, t0
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	bgez	a6, .LBB0_93
# %bb.92:                               # %cond.load166
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 512
	add	a6, sp, a6
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a6)
	ld	a6, 840(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 624(sp)                    # 4-byte Folded Spill
.LBB0_93:                               # %else167
	.loc	1 0 57                          # k135114294379184.py:0:57
	csrr	a6, vlenb
	slli	a6, a6, 5
	add	a6, sp, a6
	lui	t0, 6
	addi	t0, t0, 1728
	add	a6, a6, t0
	vl8r.v	v16, (a6)                       # vscale x 64-byte Folded Reload
	csrr	a6, vlenb
	slli	a6, a6, 3
	add	a6, sp, a6
	lui	t0, 6
	addi	t0, t0, 1728
	add	a6, a6, t0
	vl8r.v	v8, (a6)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v24, v8
	lui	a6, 6
	addi	a6, a6, 1728
	add	a6, sp, a6
	vl8r.v	v8, (a6)                        # vscale x 64-byte Folded Reload
	vadd.vv	v0, v0, v8
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a6, a5, 6
	vand.vx	v8, v16, a3
	bgez	a6, .LBB0_95
# %bb.94:                               # %cond.load169
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 384
	add	a6, sp, a6
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a6)
	ld	a6, 720(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 632(sp)                    # 4-byte Folded Spill
.LBB0_95:                               # %else170
	.loc	1 0 57                          # k135114294379184.py:0:57
	csrr	a6, vlenb
	li	t0, 104
	mul	a6, a6, t0
	add	a6, sp, a6
	lui	t0, 6
	addi	t0, t0, 1728
	add	a6, a6, t0
	vl8r.v	v16, (a6)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v16, v16, v8
	csrr	a6, vlenb
	slli	a6, a6, 6
	add	a6, sp, a6
	lui	t0, 6
	addi	t0, t0, 1728
	add	a6, a6, t0
	vl8r.v	v8, (a6)                        # vscale x 64-byte Folded Reload
	vnmsub.vx	v24, a2, v8
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a6, a5, 5
	vsll.vi	v8, v0, 6
	bgez	a6, .LBB0_118
# %bb.96:                               # %cond.load172
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 256
	add	a6, sp, a6
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 600(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 640(sp)                    # 4-byte Folded Spill
	j	.LBB0_119
.LBB0_97:                               # %cond.load85
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -1152
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 1176(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft10, a7
	fsw	ft10, 1048(sp)                  # 4-byte Folded Spill
	slli	a7, a5, 33
	flw	ft10, 1120(sp)                  # 4-byte Folded Reload
	bgez	a7, .LBB0_60
.LBB0_98:                               # %cond.load88
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -1280
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 1056(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa1, a7
	slli	a7, a5, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	fsw	fa1, 336(sp)                    # 4-byte Folded Spill
	bgez	a7, .LBB0_61
.LBB0_99:                               # %cond.load91
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -1408
	add	a7, sp, a7
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vse64.v	v8, (a7)
	ld	a7, 936(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft11, a7
	fsw	ft11, 1056(sp)                  # 4-byte Folded Spill
	slli	a7, a5, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, a0
	flw	ft11, 1136(sp)                  # 4-byte Folded Reload
	bgez	a7, .LBB0_62
.LBB0_100:                              # %cond.load94
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft2, a7
	slli	a7, a5, 30
	bgez	a7, .LBB0_63
.LBB0_101:                              # %cond.load97
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	vmv.x.s	a7, v24
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft1, a7
	slli	a7, a5, 29
	bgez	a7, .LBB0_64
.LBB0_102:                              # %cond.load100
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v8, 2
	vmv.x.s	a7, v24
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa1, a7
	fsw	fa1, 448(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 28
	bgez	a7, .LBB0_65
.LBB0_103:                              # %cond.load103
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v8, 3
	vmv.x.s	a7, v24
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa1, a7
	fsw	fa1, 456(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 27
	bgez	a7, .LBB0_66
.LBB0_104:                              # %cond.load106
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -1536
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 720(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa1, a7
	fsw	fa1, 464(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 26
	bgez	a7, .LBB0_67
.LBB0_105:                              # %cond.load109
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -1664
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 600(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa1, a7
	fsw	fa1, 472(sp)                    # 4-byte Folded Spill
	flw	fa1, 1096(sp)                   # 4-byte Folded Reload
	slli	a7, a5, 25
	bgez	a7, .LBB0_68
.LBB0_106:                              # %cond.load112
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -1792
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 480(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft0, a7
	fsw	ft0, 480(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 24
	bgez	a7, .LBB0_69
.LBB0_107:                              # %cond.load115
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 6
	addi	a7, a7, -1920
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 360(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft0, a7
	fsw	ft0, 488(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 23
	bgez	a7, .LBB0_70
.LBB0_108:                              # %cond.load118
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a7, 11
	slli	a7, a7, 11
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 240(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft0, a7
	fsw	ft0, 496(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 22
	bgez	a7, .LBB0_71
.LBB0_109:                              # %cond.load121
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, 1920
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 120(a6)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft0, a7
	fsw	ft0, 504(sp)                    # 4-byte Folded Spill
	slli	a7, a5, 21
	bgez	a7, .LBB0_72
.LBB0_110:                              # %cond.load124
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, 1792
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a6, 0(a6)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 512(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 20
	lui	a7, 5
	addi	a7, a7, -264
	add	a7, sp, a7
	bgez	a6, .LBB0_73
.LBB0_111:                              # %cond.load127
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 1664
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a6)
	ld	a6, 2016(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 520(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 19
	bgez	a6, .LBB0_74
.LBB0_112:                              # %cond.load130
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 1536
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a6)
	ld	a6, 1896(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 528(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bgez	a6, .LBB0_75
.LBB0_113:                              # %cond.load133
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 1408
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a6)
	ld	a6, 1776(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 536(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 17
	bgez	a6, .LBB0_76
.LBB0_114:                              # %cond.load136
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 1280
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a6)
	ld	a6, 1656(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 544(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bgez	a6, .LBB0_77
.LBB0_115:                              # %cond.load139
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 1152
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vse64.v	v8, (a6)
	ld	a6, 1536(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 552(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v24, a0
	bgez	a6, .LBB0_78
.LBB0_116:                              # %cond.load142
	vmv.x.s	a6, v0
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 560(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 14
	bgez	a6, .LBB0_79
.LBB0_117:                              # %cond.load145
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	a6, v8
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 568(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 13
	bltz	a6, .LBB0_80
	j	.LBB0_81
.LBB0_118:
	.loc	1 0 57                          # k135114294379184.py:0:57
	csrr	a6, vlenb
	li	t0, 96
	mul	a6, a6, t0
	add	a6, sp, a6
	lui	t0, 6
	addi	t0, t0, 1728
	add	a6, a6, t0
	vl8r.v	v0, (a6)                        # vscale x 64-byte Folded Reload
.LBB0_119:                              # %else173
	csrr	a6, vlenb
	slli	a6, a6, 5
	add	a6, sp, a6
	lui	t0, 6
	addi	t0, t0, 1728
	add	a6, a6, t0
	vs8r.v	v16, (a6)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v16, v8
	.loc	1 11 57 is_stmt 1               # k135114294379184.py:11:57
	slli	a6, a5, 4
	vsll.vi	v8, v24, 7
	bltz	a6, .LBB0_163
# %bb.120:                              # %else176
	slli	a6, a5, 3
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	bltz	a6, .LBB0_164
.LBB0_121:                              # %else179
	slli	a6, a5, 2
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vx	v16, v8, a1
	bgez	a6, .LBB0_123
.LBB0_122:                              # %cond.load181
	.loc	1 0 57 is_stmt 0                # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, -128
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 240(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 664(sp)                    # 4-byte Folded Spill
.LBB0_123:                              # %else182
	slli	a6, a5, 1
	csrr	t0, vlenb
	li	t2, 40
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl1r.v	v8, (t0)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	bltz	a6, .LBB0_165
# %bb.124:                              # %else185
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	a6, v8
	bltz	a5, .LBB0_166
.LBB0_125:                              # %else188
	.loc	1 0 57                          # k135114294379184.py:0:57
	flw	ft0, 1112(sp)                   # 4-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	andi	a5, a6, 1
	vadd.vx	v8, v24, a0
	bnez	a5, .LBB0_167
.LBB0_126:                              # %else191
	andi	a5, a6, 2
	fsw	fa0, 1088(sp)                   # 4-byte Folded Spill
	bnez	a5, .LBB0_168
.LBB0_127:                              # %else194
	andi	a5, a6, 4
	flw	fa0, 1104(sp)                   # 4-byte Folded Reload
	bnez	a5, .LBB0_169
.LBB0_128:                              # %else197
	andi	a5, a6, 8
	fsw	fa5, 344(sp)                    # 4-byte Folded Spill
	bnez	a5, .LBB0_170
.LBB0_129:                              # %else200
	andi	a7, a6, 16
	lui	a5, 4
	addi	a5, a5, 1600
	add	a5, sp, a5
	flw	fs0, 368(sp)                    # 4-byte Folded Reload
	bnez	a7, .LBB0_171
.LBB0_130:                              # %else203
	.loc	1 0 57                          # k135114294379184.py:0:57
	flw	fs11, 376(sp)                   # 4-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	andi	a7, a6, 32
	flw	fs9, 384(sp)                    # 4-byte Folded Reload
	bnez	a7, .LBB0_172
.LBB0_131:                              # %else206
	.loc	1 0 57                          # k135114294379184.py:0:57
	flw	fs7, 392(sp)                    # 4-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	andi	a7, a6, 64
	flw	fs6, 400(sp)                    # 4-byte Folded Reload
	bnez	a7, .LBB0_173
.LBB0_132:                              # %else209
	.loc	1 0 57                          # k135114294379184.py:0:57
	flw	fs3, 408(sp)                    # 4-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	andi	a7, a6, 128
	flw	fs5, 416(sp)                    # 4-byte Folded Reload
	bnez	a7, .LBB0_174
.LBB0_133:                              # %else212
	.loc	1 0 57                          # k135114294379184.py:0:57
	flw	fs4, 424(sp)                    # 4-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	andi	a7, a6, 256
	flw	fs2, 432(sp)                    # 4-byte Folded Reload
	bnez	a7, .LBB0_175
.LBB0_134:                              # %else215
	andi	a7, a6, 512
	bnez	a7, .LBB0_176
.LBB0_135:                              # %else218
	andi	a7, a6, 1024
	bnez	a7, .LBB0_177
.LBB0_136:                              # %else221
	slli	a7, a6, 52
	bltz	a7, .LBB0_178
.LBB0_137:                              # %else224
	slli	a7, a6, 51
	bltz	a7, .LBB0_179
.LBB0_138:                              # %else227
	slli	a7, a6, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bltz	a7, .LBB0_180
.LBB0_139:                              # %else230
	slli	a7, a6, 49
	bltz	a7, .LBB0_181
.LBB0_140:                              # %else233
	slli	a7, a6, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bltz	a7, .LBB0_182
.LBB0_141:                              # %else236
	slli	a7, a6, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v24, a0
	bltz	a7, .LBB0_183
.LBB0_142:                              # %else239
	slli	a7, a6, 46
	bltz	a7, .LBB0_184
.LBB0_143:                              # %else242
	slli	a7, a6, 45
	bgez	a7, .LBB0_145
.LBB0_144:                              # %cond.load244
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 2
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft0, a7
.LBB0_145:                              # %else245
	slli	a7, a6, 44
	csrr	t0, vlenb
	li	t2, 112
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v8, (t0)                        # vscale x 64-byte Folded Reload
	csrr	t0, vlenb
	li	t2, 80
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v16, v8, v16
	bgez	a7, .LBB0_147
# %bb.146:                              # %cond.load247
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v24, 3
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft3, a7
.LBB0_147:                              # %else248
	slli	a7, a6, 43
	csrr	t0, vlenb
	li	t2, 40
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vs8r.v	v16, (t0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v16, v16, 5
	csrr	t0, vlenb
	li	t2, 96
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vs8r.v	v24, (t0)                       # vscale x 64-byte Folded Spill
	bgez	a7, .LBB0_149
# %bb.148:                              # %cond.load250
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a7, 9
	slli	a7, a7, 11
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a7)
	ld	a7, 480(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft4, a7
.LBB0_149:                              # %else251
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vmulh.vx	v8, v16, a4
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a7, a6, 42
	csrr	t0, vlenb
	li	t2, 112
	mul	t0, t0, t2
	add	t0, sp, t0
	lui	t2, 6
	addi	t2, t2, 1728
	add	t0, t0, t2
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	vmulh.vx	v24, v24, a4
	bgez	a7, .LBB0_151
# %bb.150:                              # %cond.load253
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a4, 4
	addi	a4, a4, 1920
	add	a4, sp, a4
	csrr	a7, vlenb
	li	t0, 96
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a4)
	ld	a4, 360(a5)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	ft5, a4
.LBB0_151:                              # %else254
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a4, a6, 41
	csrr	a7, vlenb
	li	t0, 112
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	vadd.vv	v0, v24, v0
	bgez	a4, .LBB0_153
# %bb.152:                              # %cond.load256
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a4, 4
	addi	a4, a4, 1792
	add	a4, sp, a4
	csrr	a7, vlenb
	li	t0, 96
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vl8r.v	v24, (a7)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a4)
	ld	a4, 240(a5)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	ft6, a4
.LBB0_153:                              # %else257
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vsra.vi	v24, v8, 3
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a4, a6, 40
	vsra.vi	v0, v0, 8
	csrr	a7, vlenb
	li	t0, 80
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vs8r.v	v16, (a7)                       # vscale x 64-byte Folded Spill
	bgez	a4, .LBB0_155
# %bb.154:                              # %cond.load259
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a4, 4
	addi	a4, a4, 1664
	add	a4, sp, a4
	csrr	a7, vlenb
	li	t0, 96
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a4)
	ld	a4, 120(a5)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	ft7, a4
.LBB0_155:                              # %else260
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vsrl.vi	v8, v24, 31
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a4, a6, 39
	vsrl.vi	v16, v0, 31
	bgez	a4, .LBB0_157
# %bb.156:                              # %cond.load262
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a4, 4
	addi	a4, a4, 1536
	add	a4, sp, a4
	csrr	a7, vlenb
	slli	a7, a7, 3
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	csrr	a7, vlenb
	li	t0, 96
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 6
	addi	t0, t0, 1728
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a4)
	csrr	a4, vlenb
	slli	a4, a4, 3
	add	a4, sp, a4
	lui	a7, 6
	addi	a7, a7, 1728
	add	a4, a4, a7
	vl8r.v	v8, (a4)                        # vscale x 64-byte Folded Reload
	ld	a4, 0(a5)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	ft8, a4
.LBB0_157:                              # %else263
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v24, v24, v8
	csrr	a4, vlenb
	li	a5, 40
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 6
	addi	a5, a5, 1728
	add	a4, a4, a5
	vl8r.v	v8, (a4)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a3
	vadd.vv	v16, v0, v16
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a4, a6, 38
	lui	a3, 4
	addi	a3, a3, -536
	add	a3, sp, a3
	bgez	a4, .LBB0_159
# %bb.158:                              # %cond.load265
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a4, 4
	addi	a4, a4, 1408
	add	a4, sp, a4
	csrr	a5, vlenb
	li	a7, 96
	mul	a5, a5, a7
	add	a5, sp, a5
	lui	a7, 6
	addi	a7, a7, 1728
	add	a5, a5, a7
	vl8r.v	v0, (a5)                        # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a4)
	ld	a4, 2016(a3)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa6, a4
.LBB0_159:                              # %else266
	.loc	1 0 57                          # k135114294379184.py:0:57
	csrr	a4, vlenb
	li	a5, 112
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 6
	addi	a5, a5, 1728
	add	a4, a4, a5
	vl8r.v	v0, (a4)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a1, e32, m8, ta, ma
	vsub.vv	v0, v0, v8
	csrr	a4, vlenb
	li	a5, 80
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 6
	addi	a5, a5, 1728
	add	a4, a4, a5
	vl8r.v	v8, (a4)                        # vscale x 64-byte Folded Reload
	vnmsub.vx	v24, a2, v8
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a2, a6, 37
	vsll.vi	v8, v16, 6
	bgez	a2, .LBB0_161
# %bb.160:                              # %cond.load268
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a2, 4
	addi	a2, a2, 1280
	add	a2, sp, a2
	csrr	a4, vlenb
	li	a5, 96
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 6
	addi	a5, a5, 1728
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a2)
	ld	a2, 1896(a3)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fa7, a2
.LBB0_161:                              # %else269
	.loc	1 0 57                          # k135114294379184.py:0:57
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v0, v8
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a2, a6, 36
	vsll.vi	v16, v24, 7
	bgez	a2, .LBB0_185
# %bb.162:                              # %cond.load271
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a2, 4
	addi	a2, a2, 1152
	add	a2, sp, a2
	csrr	a4, vlenb
	li	a5, 96
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 6
	addi	a5, a5, 1728
	add	a4, a4, a5
	vl8r.v	v24, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 1776(a3)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	ft10, a2
	slli	a2, a6, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bltz	a2, .LBB0_186
	j	.LBB0_187
.LBB0_163:                              # %cond.load175
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, 128
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 480(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 648(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 3
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a6, .LBB0_121
.LBB0_164:                              # %cond.load178
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 360(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 656(sp)                    # 4-byte Folded Spill
	slli	a6, a5, 2
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vx	v16, v8, a1
	bltz	a6, .LBB0_122
	j	.LBB0_123
.LBB0_165:                              # %cond.load184
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a6, 5
	addi	a6, a6, -256
	add	a6, sp, a6
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 120(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft0, a6
	fsw	ft0, 672(sp)                    # 4-byte Folded Spill
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	a6, v8
	bgez	a5, .LBB0_125
.LBB0_166:                              # %cond.load187
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a5, 5
	addi	a5, a5, -384
	add	a5, sp, a5
	.loc	1 11 57                         # k135114294379184.py:11:57
	vse64.v	v0, (a5)
	ld	a5, 0(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	ft0, a5
	fsw	ft0, 680(sp)                    # 4-byte Folded Spill
	flw	ft0, 1112(sp)                   # 4-byte Folded Reload
	andi	a5, a6, 1
	vadd.vx	v8, v24, a0
	beqz	a5, .LBB0_126
.LBB0_167:                              # %cond.load190
	vmv.x.s	a5, v8
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa0, a5
	andi	a5, a6, 2
	fsw	fa0, 1088(sp)                   # 4-byte Folded Spill
	beqz	a5, .LBB0_127
.LBB0_168:                              # %cond.load193
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	vmv.x.s	a5, v24
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa0, a5
	fsw	fa0, 1064(sp)                   # 4-byte Folded Spill
	andi	a5, a6, 4
	flw	fa0, 1104(sp)                   # 4-byte Folded Reload
	beqz	a5, .LBB0_128
.LBB0_169:                              # %cond.load196
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v8, 2
	vmv.x.s	a5, v24
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	andi	a5, a6, 8
	fsw	fa5, 344(sp)                    # 4-byte Folded Spill
	beqz	a5, .LBB0_129
.LBB0_170:                              # %cond.load199
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v8, 3
	vmv.x.s	a5, v24
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa5, a5
	fsw	fa5, 1072(sp)                   # 4-byte Folded Spill
	andi	a7, a6, 16
	lui	a5, 4
	addi	a5, a5, 1600
	add	a5, sp, a5
	flw	fs0, 368(sp)                    # 4-byte Folded Reload
	beqz	a7, .LBB0_130
.LBB0_171:                              # %cond.load202
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -512
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 2016(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs0, a7
	flw	fs11, 376(sp)                   # 4-byte Folded Reload
	andi	a7, a6, 32
	flw	fs9, 384(sp)                    # 4-byte Folded Reload
	beqz	a7, .LBB0_131
.LBB0_172:                              # %cond.load205
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -640
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1896(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs11, a7
	flw	fs7, 392(sp)                    # 4-byte Folded Reload
	andi	a7, a6, 64
	flw	fs6, 400(sp)                    # 4-byte Folded Reload
	beqz	a7, .LBB0_132
.LBB0_173:                              # %cond.load208
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -768
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1776(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs9, a7
	flw	fs3, 408(sp)                    # 4-byte Folded Reload
	andi	a7, a6, 128
	flw	fs5, 416(sp)                    # 4-byte Folded Reload
	beqz	a7, .LBB0_133
.LBB0_174:                              # %cond.load211
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -896
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1656(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs7, a7
	flw	fs4, 424(sp)                    # 4-byte Folded Reload
	andi	a7, a6, 256
	flw	fs2, 432(sp)                    # 4-byte Folded Reload
	beqz	a7, .LBB0_134
.LBB0_175:                              # %cond.load214
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a7, 19
	slli	a7, a7, 10
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1536(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs6, a7
	andi	a7, a6, 512
	beqz	a7, .LBB0_135
.LBB0_176:                              # %cond.load217
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -1152
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1416(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs3, a7
	andi	a7, a6, 1024
	beqz	a7, .LBB0_136
.LBB0_177:                              # %cond.load220
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -1280
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1296(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs5, a7
	slli	a7, a6, 52
	bgez	a7, .LBB0_137
.LBB0_178:                              # %cond.load223
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -1408
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1176(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs4, a7
	slli	a7, a6, 51
	bgez	a7, .LBB0_138
.LBB0_179:                              # %cond.load226
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -1536
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 1056(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs2, a7
	slli	a7, a6, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bgez	a7, .LBB0_139
.LBB0_180:                              # %cond.load229
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -1664
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 936(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa3, a7
	slli	a7, a6, 49
	bgez	a7, .LBB0_140
.LBB0_181:                              # %cond.load232
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -1792
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a7)
	ld	a7, 816(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa1, a7
	slli	a7, a6, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bgez	a7, .LBB0_141
.LBB0_182:                              # %cond.load235
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a7, 5
	addi	a7, a7, -1920
	add	a7, sp, a7
	.loc	1 11 57                         # k135114294379184.py:11:57
	vse64.v	v8, (a7)
	ld	a7, 696(a5)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa4, a7
	slli	a7, a6, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v24, v24, a0
	bgez	a7, .LBB0_142
.LBB0_183:                              # %cond.load238
	vmv.x.s	a7, v24
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa2, a7
	slli	a7, a6, 46
	bgez	a7, .LBB0_143
.LBB0_184:                              # %cond.load241
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v24, 1
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa0, a7
	slli	a7, a6, 45
	bltz	a7, .LBB0_144
	j	.LBB0_145
.LBB0_185:
	.loc	1 0 57                          # k135114294379184.py:0:57
	csrr	a2, vlenb
	li	a4, 96
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 6
	addi	a4, a4, 1728
	add	a2, a2, a4
	vl8r.v	v24, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 11 57                         # k135114294379184.py:11:57
	slli	a2, a6, 35
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a2, .LBB0_187
.LBB0_186:                              # %cond.load274
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a2, 17
	slli	a2, a2, 10
	add	a2, sp, a2
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a2)
	ld	a2, 1656(a3)
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	ft9, a2
.LBB0_187:                              # %else275
	slli	a2, a6, 34
	vsetvli	zero, a1, e32, m8, ta, ma
	vadd.vx	v16, v8, a1
	bltz	a2, .LBB0_224
# %bb.188:                              # %else278
	slli	a1, a6, 33
	bltz	a1, .LBB0_225
.LBB0_189:                              # %else281
	slli	a1, a6, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v16
	bltz	a1, .LBB0_226
.LBB0_190:                              # %else284
	slli	a1, a6, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v8, a0
	bltz	a1, .LBB0_227
.LBB0_191:                              # %else287
	slli	a1, a6, 30
	bltz	a1, .LBB0_228
.LBB0_192:                              # %else290
	slli	a1, a6, 29
	bltz	a1, .LBB0_229
.LBB0_193:                              # %else293
	slli	a1, a6, 28
	bltz	a1, .LBB0_230
.LBB0_194:                              # %else296
	slli	a1, a6, 27
	bltz	a1, .LBB0_231
.LBB0_195:                              # %else299
	slli	a1, a6, 26
	bltz	a1, .LBB0_232
.LBB0_196:                              # %else302
	slli	a1, a6, 25
	bltz	a1, .LBB0_233
.LBB0_197:                              # %else305
	slli	a1, a6, 24
	bltz	a1, .LBB0_234
.LBB0_198:                              # %else308
	slli	a1, a6, 23
	bltz	a1, .LBB0_235
.LBB0_199:                              # %else311
	slli	a1, a6, 22
	bltz	a1, .LBB0_236
.LBB0_200:                              # %else314
	slli	a1, a6, 21
	bltz	a1, .LBB0_237
.LBB0_201:                              # %else317
	slli	a1, a6, 20
	bltz	a1, .LBB0_238
.LBB0_202:                              # %else320
	slli	a1, a6, 19
	bltz	a1, .LBB0_239
.LBB0_203:                              # %else323
	slli	a1, a6, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bltz	a1, .LBB0_240
.LBB0_204:                              # %else326
	slli	a1, a6, 17
	lui	a2, 3
	addi	a2, a2, 1448
	add	s10, sp, a2
	bltz	a1, .LBB0_241
.LBB0_205:                              # %else329
	slli	a1, a6, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bltz	a1, .LBB0_242
.LBB0_206:                              # %else332
	slli	a1, a6, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, a0
	bltz	a1, .LBB0_243
.LBB0_207:                              # %else335
	slli	a0, a6, 14
	bltz	a0, .LBB0_244
.LBB0_208:                              # %else338
	slli	a0, a6, 13
	bltz	a0, .LBB0_245
.LBB0_209:                              # %else341
	slli	a0, a6, 12
	bltz	a0, .LBB0_246
.LBB0_210:                              # %else344
	slli	a0, a6, 11
	bltz	a0, .LBB0_247
.LBB0_211:                              # %else347
	slli	a0, a6, 10
	bltz	a0, .LBB0_248
.LBB0_212:                              # %else350
	slli	a0, a6, 9
	bltz	a0, .LBB0_249
.LBB0_213:                              # %else353
	slli	a0, a6, 8
	bltz	a0, .LBB0_250
.LBB0_214:                              # %else356
	slli	a0, a6, 7
	bltz	a0, .LBB0_251
.LBB0_215:                              # %else359
	slli	a0, a6, 6
	bltz	a0, .LBB0_252
.LBB0_216:                              # %else362
	slli	a0, a6, 5
	bltz	a0, .LBB0_253
.LBB0_217:                              # %else365
	slli	a0, a6, 4
	bltz	a0, .LBB0_254
.LBB0_218:                              # %else368
	slli	a0, a6, 3
	bltz	a0, .LBB0_255
.LBB0_219:                              # %else371
	slli	a0, a6, 2
	bltz	a0, .LBB0_256
.LBB0_220:                              # %else374
	slli	a0, a6, 1
	bgez	a0, .LBB0_222
.LBB0_221:                              # %cond.load376
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a0, 3
	addi	a0, a0, 1792
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 456(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 992(sp)                    # 4-byte Folded Spill
.LBB0_222:                              # %else377
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a0, 3
	addi	a0, a0, -600
	add	s4, sp, a0
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	sd	t1, 1144(sp)                    # 8-byte Folded Spill
	fsw	ft2, 440(sp)                    # 4-byte Folded Spill
	fsw	fs1, 432(sp)                    # 4-byte Folded Spill
	fsw	ft11, 1136(sp)                  # 4-byte Folded Spill
	fsw	ft9, 1128(sp)                   # 4-byte Folded Spill
	fsw	ft10, 1120(sp)                  # 4-byte Folded Spill
	fsw	fa7, 424(sp)                    # 4-byte Folded Spill
	fsw	fa6, 416(sp)                    # 4-byte Folded Spill
	fsw	ft8, 408(sp)                    # 4-byte Folded Spill
	fsw	ft7, 400(sp)                    # 4-byte Folded Spill
	fsw	ft6, 392(sp)                    # 4-byte Folded Spill
	fsw	ft5, 384(sp)                    # 4-byte Folded Spill
	fsw	ft4, 376(sp)                    # 4-byte Folded Spill
	fsw	ft3, 368(sp)                    # 4-byte Folded Spill
	fsw	ft0, 1112(sp)                   # 4-byte Folded Spill
	fsw	fa0, 1104(sp)                   # 4-byte Folded Spill
	fsw	fa2, 360(sp)                    # 4-byte Folded Spill
	fsw	fa4, 352(sp)                    # 4-byte Folded Spill
	fsw	fa1, 1096(sp)                   # 4-byte Folded Spill
	fsw	fa3, 296(sp)                    # 4-byte Folded Spill
	.loc	1 11 57                         # k135114294379184.py:11:57
	bgez	a6, .LBB0_257
# %bb.223:                              # %cond.load379
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a0, 3
	addi	a0, a0, 1664
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 336(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fs1, a0
	j	.LBB0_258
.LBB0_224:                              # %cond.load277
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, 896
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1536(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	ft11, a1
	slli	a1, a6, 33
	bgez	a1, .LBB0_189
.LBB0_225:                              # %cond.load280
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, 768
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a1)
	ld	a1, 1416(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 1080(sp)                   # 4-byte Folded Spill
	slli	a1, a6, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v16
	bgez	a1, .LBB0_190
.LBB0_226:                              # %cond.load283
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, 640
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vse64.v	v24, (a1)
	ld	a1, 1296(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fs1, a1
	slli	a1, a6, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v8, a0
	bgez	a1, .LBB0_191
.LBB0_227:                              # %cond.load286
	vmv.x.s	a1, v8
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 688(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 30
	bgez	a1, .LBB0_192
.LBB0_228:                              # %cond.load289
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	vmv.x.s	a1, v24
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 696(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 29
	bgez	a1, .LBB0_193
.LBB0_229:                              # %cond.load292
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v8, 2
	vmv.x.s	a1, v24
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 704(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 28
	bgez	a1, .LBB0_194
.LBB0_230:                              # %cond.load295
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v8, 3
	vmv.x.s	a1, v24
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 712(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 27
	bgez	a1, .LBB0_195
.LBB0_231:                              # %cond.load298
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, 512
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 1080(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 720(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 26
	bgez	a1, .LBB0_196
.LBB0_232:                              # %cond.load301
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, 384
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 960(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 728(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 25
	bgez	a1, .LBB0_197
.LBB0_233:                              # %cond.load304
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, 256
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 840(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 736(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 24
	bgez	a1, .LBB0_198
.LBB0_234:                              # %cond.load307
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, 128
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 720(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 744(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 23
	bgez	a1, .LBB0_199
.LBB0_235:                              # %cond.load310
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 600(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 752(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 22
	bgez	a1, .LBB0_200
.LBB0_236:                              # %cond.load313
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, -128
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 480(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 760(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 21
	bgez	a1, .LBB0_201
.LBB0_237:                              # %cond.load316
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, -256
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 360(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 768(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 20
	bgez	a1, .LBB0_202
.LBB0_238:                              # %cond.load319
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, -384
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 240(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 776(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 19
	bgez	a1, .LBB0_203
.LBB0_239:                              # %cond.load322
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a1, 31
	slli	a1, a1, 9
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 120(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 784(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	bgez	a1, .LBB0_204
.LBB0_240:                              # %cond.load325
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, -640
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 0(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 792(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 17
	lui	a2, 3
	addi	a2, a2, 1448
	add	s10, sp, a2
	bgez	a1, .LBB0_205
.LBB0_241:                              # %cond.load328
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, -768
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 1992(s10)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 800(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v16
	bgez	a1, .LBB0_206
.LBB0_242:                              # %cond.load331
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a1, 4
	addi	a1, a1, -896
	add	a1, sp, a1
	.loc	1 11 57                         # k135114294379184.py:11:57
	vse64.v	v8, (a1)
	ld	a1, 1872(s10)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 816(sp)                    # 4-byte Folded Spill
	slli	a1, a6, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, a0
	bgez	a1, .LBB0_207
.LBB0_243:                              # %cond.load334
	vmv.x.s	a0, v8
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 832(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 14
	bgez	a0, .LBB0_208
.LBB0_244:                              # %cond.load337
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 840(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 13
	bgez	a0, .LBB0_209
.LBB0_245:                              # %cond.load340
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 2
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 856(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 12
	bgez	a0, .LBB0_210
.LBB0_246:                              # %cond.load343
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 3
	vmv.x.s	a0, v16
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 864(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 11
	bgez	a0, .LBB0_211
.LBB0_247:                              # %cond.load346
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a0, 15
	slli	a0, a0, 10
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 880(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 10
	bgez	a0, .LBB0_212
.LBB0_248:                              # %cond.load349
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a0, 4
	addi	a0, a0, -1152
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 888(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 9
	bgez	a0, .LBB0_213
.LBB0_249:                              # %cond.load352
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a0, 4
	addi	a0, a0, -1280
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 904(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 8
	bgez	a0, .LBB0_214
.LBB0_250:                              # %cond.load355
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a0, 4
	addi	a0, a0, -1408
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 912(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 7
	bgez	a0, .LBB0_215
.LBB0_251:                              # %cond.load358
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a0, 29
	slli	a0, a0, 9
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 928(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 6
	bgez	a0, .LBB0_216
.LBB0_252:                              # %cond.load361
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a0, 4
	addi	a0, a0, -1664
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 936(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 5
	bgez	a0, .LBB0_217
.LBB0_253:                              # %cond.load364
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a0, 4
	addi	a0, a0, -1792
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 944(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 4
	bgez	a0, .LBB0_218
.LBB0_254:                              # %cond.load367
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a0, 4
	addi	a0, a0, -1920
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 960(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 3
	bgez	a0, .LBB0_219
.LBB0_255:                              # %cond.load370
	.loc	1 0 57                          # k135114294379184.py:0:57
	li	a0, 7
	slli	a0, a0, 11
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 968(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 2
	bgez	a0, .LBB0_220
.LBB0_256:                              # %cond.load373
	.loc	1 0 57                          # k135114294379184.py:0:57
	lui	a0, 3
	addi	a0, a0, 1920
	add	a0, sp, a0
	.loc	1 11 57                         # k135114294379184.py:11:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s10)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fa5, a0
	fsw	fa5, 984(sp)                    # 4-byte Folded Spill
	slli	a0, a6, 1
	bltz	a0, .LBB0_221
	j	.LBB0_222
.LBB0_257:
	.loc	1 0 57                          # k135114294379184.py:0:57
	flw	fs1, 220(sp)                    # 4-byte Folded Reload
.LBB0_258:                              # %else380
	li	a1, 32
	li	a0, 896
	csrr	a2, vlenb
	li	a3, 112
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 6
	addi	a3, a3, 1728
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 6 21 is_stmt 1                # k135114294379184.py:6:21
	vsetvli	zero, a1, e32, m8, ta, ma
	vmslt.vx	v8, v16, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v9, v16, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v10, v16, a0
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v11, v16, a0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	vslideup.vi	v11, v10, 4
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v11, v9, 8
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs1r.v	v11, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 11 67                         # k135114294379184.py:11:67
	fmv.s	fa0, ft1
	call	__truncsfbf2
	fsw	fa0, 220(sp)                    # 4-byte Folded Spill
	flw	fa0, 448(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 448(sp)                    # 4-byte Folded Spill
	flw	fa0, 456(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 456(sp)                    # 4-byte Folded Spill
	flw	fa0, 464(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 464(sp)                    # 4-byte Folded Spill
	flw	fa0, 472(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 472(sp)                    # 4-byte Folded Spill
	flw	fa0, 480(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 480(sp)                    # 4-byte Folded Spill
	flw	fa0, 488(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 488(sp)                    # 4-byte Folded Spill
	flw	fa0, 496(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 496(sp)                    # 4-byte Folded Spill
	flw	fa0, 504(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 504(sp)                    # 4-byte Folded Spill
	flw	fa0, 512(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 512(sp)                    # 4-byte Folded Spill
	flw	fa0, 520(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 520(sp)                    # 4-byte Folded Spill
	flw	fa0, 528(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 528(sp)                    # 4-byte Folded Spill
	flw	fa0, 536(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 536(sp)                    # 4-byte Folded Spill
	flw	fa0, 544(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 544(sp)                    # 4-byte Folded Spill
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
	fmv.s	fa0, fs8
	call	__truncsfbf2
	fsw	fa0, 208(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs10
	call	__truncsfbf2
	fsw	fa0, 200(sp)                    # 4-byte Folded Spill
	flw	fa0, 224(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 224(sp)                    # 4-byte Folded Spill
	flw	fa0, 808(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 808(sp)                    # 4-byte Folded Spill
	flw	fa0, 232(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 232(sp)                    # 4-byte Folded Spill
	flw	fa0, 824(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 824(sp)                    # 4-byte Folded Spill
	flw	fa0, 240(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 240(sp)                    # 4-byte Folded Spill
	flw	fa0, 848(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 848(sp)                    # 4-byte Folded Spill
	flw	fa0, 248(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 248(sp)                    # 4-byte Folded Spill
	flw	fa0, 872(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 872(sp)                    # 4-byte Folded Spill
	flw	fa0, 256(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 256(sp)                    # 4-byte Folded Spill
	flw	fa0, 896(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 896(sp)                    # 4-byte Folded Spill
	flw	fa0, 264(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 264(sp)                    # 4-byte Folded Spill
	flw	fa0, 920(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 920(sp)                    # 4-byte Folded Spill
	flw	fa0, 272(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 272(sp)                    # 4-byte Folded Spill
	flw	fa0, 952(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 952(sp)                    # 4-byte Folded Spill
	flw	fa0, 280(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 280(sp)                    # 4-byte Folded Spill
	flw	fa0, 976(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 976(sp)                    # 4-byte Folded Spill
	flw	fa0, 288(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 288(sp)                    # 4-byte Folded Spill
	flw	fa0, 1008(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1008(sp)                   # 4-byte Folded Spill
	flw	fa0, 1000(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1000(sp)                   # 4-byte Folded Spill
	flw	fa0, 1016(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1016(sp)                   # 4-byte Folded Spill
	flw	fa0, 304(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 304(sp)                    # 4-byte Folded Spill
	flw	fa0, 1024(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1024(sp)                   # 4-byte Folded Spill
	flw	fa0, 312(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 312(sp)                    # 4-byte Folded Spill
	flw	fa0, 1032(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1032(sp)                   # 4-byte Folded Spill
	flw	fa0, 320(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 320(sp)                    # 4-byte Folded Spill
	flw	fa0, 1040(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1040(sp)                   # 4-byte Folded Spill
	flw	fa0, 328(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 328(sp)                    # 4-byte Folded Spill
	flw	fa0, 1048(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1048(sp)                   # 4-byte Folded Spill
	flw	fa0, 336(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 336(sp)                    # 4-byte Folded Spill
	flw	fa0, 1056(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1056(sp)                   # 4-byte Folded Spill
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
	flw	fa0, 816(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 816(sp)                    # 4-byte Folded Spill
	flw	fa0, 832(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 832(sp)                    # 4-byte Folded Spill
	flw	fa0, 840(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 840(sp)                    # 4-byte Folded Spill
	flw	fa0, 856(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 856(sp)                    # 4-byte Folded Spill
	flw	fa0, 864(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 864(sp)                    # 4-byte Folded Spill
	flw	fa0, 880(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 880(sp)                    # 4-byte Folded Spill
	flw	fa0, 888(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 888(sp)                    # 4-byte Folded Spill
	flw	fa0, 904(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 904(sp)                    # 4-byte Folded Spill
	flw	fa0, 912(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 912(sp)                    # 4-byte Folded Spill
	flw	fa0, 928(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 192(sp)                    # 4-byte Folded Spill
	flw	fa0, 936(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 184(sp)                    # 4-byte Folded Spill
	flw	fa0, 944(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 176(sp)                    # 4-byte Folded Spill
	flw	fa0, 960(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 168(sp)                    # 4-byte Folded Spill
	flw	fa0, 968(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 160(sp)                    # 4-byte Folded Spill
	flw	fa0, 984(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 152(sp)                    # 4-byte Folded Spill
	flw	fa0, 992(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 144(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs1
	call	__truncsfbf2
	fsw	fa0, 120(sp)                    # 4-byte Folded Spill
	flw	fa0, 1088(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 128(sp)                    # 4-byte Folded Spill
	flw	fa0, 1064(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 136(sp)                    # 4-byte Folded Spill
	flw	fa0, 344(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 112(sp)                    # 4-byte Folded Spill
	flw	fa0, 1072(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 344(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs0
	call	__truncsfbf2
	fsw	fa0, 104(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs11
	call	__truncsfbf2
	fsw	fa0, 96(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs9
	call	__truncsfbf2
	fsw	fa0, 92(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs7
	call	__truncsfbf2
	fsw	fa0, 88(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs6
	call	__truncsfbf2
	fsw	fa0, 84(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs3
	call	__truncsfbf2
	fsw	fa0, 80(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs5
	call	__truncsfbf2
	fsw	fa0, 76(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs4
	call	__truncsfbf2
	fsw	fa0, 72(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs2
	call	__truncsfbf2
	fsw	fa0, 68(sp)                     # 4-byte Folded Spill
	flw	fa0, 296(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 296(sp)                    # 4-byte Folded Spill
	flw	fa0, 1096(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 64(sp)                     # 4-byte Folded Spill
	flw	fa0, 352(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 928(sp)                    # 4-byte Folded Spill
	flw	fa0, 360(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 936(sp)                    # 4-byte Folded Spill
	flw	fa0, 1104(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 944(sp)                    # 4-byte Folded Spill
	flw	fa0, 1112(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 960(sp)                    # 4-byte Folded Spill
	flw	fa0, 368(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs3, fa0
	flw	fa0, 376(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs8, fa0
	flw	fa0, 384(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs10, fa0
	flw	fa0, 392(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs2, fa0
	flw	fa0, 400(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs1, fa0
	flw	fa0, 408(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs0, fa0
	flw	fa0, 416(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs11, fa0
	flw	fa0, 424(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs9, fa0
	flw	fa0, 1120(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs7, fa0
	flw	fa0, 1128(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs6, fa0
	flw	fa0, 1136(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs5, fa0
	flw	fa0, 1080(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs4, fa0
	flw	fa0, 432(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	sd	a0, 1136(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs4
	sd	a0, 1128(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs5
	sd	a0, 1120(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs6
	sd	a0, 1112(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs7
	sd	a0, 1104(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs9
	sd	a0, 1096(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs11
	sd	a0, 1088(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs0
	sd	a0, 1080(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs1
	sd	a0, 1072(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs2
	sd	a0, 1064(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs10
	sd	a0, 992(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs8
	sd	a0, 984(sp)                     # 8-byte Folded Spill
	fmv.x.w	a0, fs3
	sd	a0, 968(sp)                     # 8-byte Folded Spill
	flw	fa5, 960(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 960(sp)                     # 8-byte Folded Spill
	flw	fa5, 944(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 944(sp)                     # 8-byte Folded Spill
	flw	fa5, 936(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 936(sp)                     # 8-byte Folded Spill
	flw	fa5, 928(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 928(sp)                     # 8-byte Folded Spill
	flw	fa5, 64(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 432(sp)                     # 8-byte Folded Spill
	flw	fa5, 296(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 424(sp)                     # 8-byte Folded Spill
	flw	fa5, 68(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 416(sp)                     # 8-byte Folded Spill
	flw	fa5, 72(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 408(sp)                     # 8-byte Folded Spill
	flw	fa5, 76(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 400(sp)                     # 8-byte Folded Spill
	flw	fa5, 80(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 392(sp)                     # 8-byte Folded Spill
	flw	fa5, 84(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 384(sp)                     # 8-byte Folded Spill
	flw	fa5, 88(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 376(sp)                     # 8-byte Folded Spill
	flw	fa5, 92(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 368(sp)                     # 8-byte Folded Spill
	flw	fa5, 96(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 360(sp)                     # 8-byte Folded Spill
	flw	fa5, 104(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 352(sp)                     # 8-byte Folded Spill
	flw	fa5, 344(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 344(sp)                     # 8-byte Folded Spill
	flw	fa5, 112(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 296(sp)                     # 8-byte Folded Spill
	flw	fa5, 136(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	flw	fa5, 128(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	flw	fa5, 120(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	flw	fa5, 144(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 144(sp)                     # 8-byte Folded Spill
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
	flw	fa5, 912(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 912(sp)                     # 8-byte Folded Spill
	flw	fa5, 904(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 904(sp)                     # 8-byte Folded Spill
	flw	fa5, 888(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 888(sp)                     # 8-byte Folded Spill
	flw	fa5, 880(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 880(sp)                     # 8-byte Folded Spill
	flw	fa5, 864(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 864(sp)                     # 8-byte Folded Spill
	flw	fa5, 856(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 856(sp)                     # 8-byte Folded Spill
	flw	fa5, 840(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 840(sp)                     # 8-byte Folded Spill
	flw	fa5, 832(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 832(sp)                     # 8-byte Folded Spill
	flw	fa5, 816(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 816(sp)                     # 8-byte Folded Spill
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
	flw	fa5, 1056(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1056(sp)                    # 8-byte Folded Spill
	flw	fa5, 336(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 336(sp)                     # 8-byte Folded Spill
	flw	fa5, 1048(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1048(sp)                    # 8-byte Folded Spill
	flw	fa5, 328(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 328(sp)                     # 8-byte Folded Spill
	flw	fa5, 1040(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1040(sp)                    # 8-byte Folded Spill
	flw	fa5, 320(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 320(sp)                     # 8-byte Folded Spill
	flw	fa5, 1032(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1032(sp)                    # 8-byte Folded Spill
	flw	fa5, 312(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 312(sp)                     # 8-byte Folded Spill
	flw	fa5, 1024(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1024(sp)                    # 8-byte Folded Spill
	flw	fa5, 304(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 304(sp)                     # 8-byte Folded Spill
	flw	fa5, 1016(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1016(sp)                    # 8-byte Folded Spill
	flw	fa5, 1000(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1000(sp)                    # 8-byte Folded Spill
	flw	fa5, 1008(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1008(sp)                    # 8-byte Folded Spill
	flw	fa5, 288(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 288(sp)                     # 8-byte Folded Spill
	flw	fa5, 976(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 976(sp)                     # 8-byte Folded Spill
	flw	fa5, 280(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 280(sp)                     # 8-byte Folded Spill
	flw	fa5, 952(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 952(sp)                     # 8-byte Folded Spill
	flw	fa5, 272(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 272(sp)                     # 8-byte Folded Spill
	flw	fa5, 920(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 920(sp)                     # 8-byte Folded Spill
	flw	fa5, 264(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 264(sp)                     # 8-byte Folded Spill
	flw	fa5, 896(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 896(sp)                     # 8-byte Folded Spill
	flw	fa5, 256(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 256(sp)                     # 8-byte Folded Spill
	flw	fa5, 872(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 872(sp)                     # 8-byte Folded Spill
	flw	fa5, 248(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 248(sp)                     # 8-byte Folded Spill
	flw	fa5, 848(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 848(sp)                     # 8-byte Folded Spill
	flw	fa5, 240(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 240(sp)                     # 8-byte Folded Spill
	flw	fa5, 824(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 824(sp)                     # 8-byte Folded Spill
	flw	fa5, 232(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 232(sp)                     # 8-byte Folded Spill
	flw	fa5, 808(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 808(sp)                     # 8-byte Folded Spill
	flw	fa5, 224(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 224(sp)                     # 8-byte Folded Spill
	flw	fa5, 200(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	flw	fa5, 208(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	flw	fa5, 680(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	flw	fa5, 672(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	flw	fa5, 664(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 96(sp)                      # 8-byte Folded Spill
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
	sd	a0, 584(sp)                     # 8-byte Folded Spill
	flw	fa5, 576(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 576(sp)                     # 8-byte Folded Spill
	flw	fa5, 568(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 568(sp)                     # 8-byte Folded Spill
	flw	fa5, 560(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 560(sp)                     # 8-byte Folded Spill
	flw	fa5, 552(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 552(sp)                     # 8-byte Folded Spill
	flw	fa5, 544(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 544(sp)                     # 8-byte Folded Spill
	flw	fa5, 536(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 536(sp)                     # 8-byte Folded Spill
	flw	fa5, 528(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 528(sp)                     # 8-byte Folded Spill
	flw	fa5, 520(sp)                    # 4-byte Folded Reload
	fmv.x.w	s11, fa5
	flw	fa5, 512(sp)                    # 4-byte Folded Reload
	fmv.x.w	s2, fa5
	flw	fa5, 504(sp)                    # 4-byte Folded Reload
	fmv.x.w	s3, fa5
	flw	fa5, 496(sp)                    # 4-byte Folded Reload
	fmv.x.w	s6, fa5
	flw	fa5, 488(sp)                    # 4-byte Folded Reload
	fmv.x.w	s7, fa5
	flw	fa5, 480(sp)                    # 4-byte Folded Reload
	fmv.x.w	s8, fa5
	flw	fa5, 472(sp)                    # 4-byte Folded Reload
	fmv.x.w	s5, fa5
	flw	fa5, 464(sp)                    # 4-byte Folded Reload
	fmv.x.w	s9, fa5
	flw	fa5, 456(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 680(sp)                     # 8-byte Folded Spill
	flw	fa5, 448(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 672(sp)                     # 8-byte Folded Spill
	flw	fa5, 220(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 664(sp)                     # 8-byte Folded Spill
	flw	fa0, 440(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	sh	s9, 1824(s4)
	sh	s5, 1826(s4)
	sh	s8, 1828(s4)
	sh	s7, 1830(s4)
	sh	s6, 1832(s4)
	sh	s3, 1834(s4)
	sh	s2, 1836(s4)
	sh	s11, 1838(s4)
	ld	a0, 528(sp)                     # 8-byte Folded Reload
	sh	a0, 1840(s4)
	ld	a0, 536(sp)                     # 8-byte Folded Reload
	sh	a0, 1842(s4)
	ld	a0, 544(sp)                     # 8-byte Folded Reload
	sh	a0, 1844(s4)
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	sh	a0, 1846(s4)
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	sh	a0, 1848(s4)
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	sh	a0, 1850(s4)
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	sh	a0, 1852(s4)
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	sh	a0, 1854(s4)
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	sh	a0, 1856(s4)
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	sh	a0, 1858(s4)
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	sh	a0, 1860(s4)
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	sh	a0, 1862(s4)
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	sh	a0, 1864(s4)
	ld	a0, 632(sp)                     # 8-byte Folded Reload
	sh	a0, 1866(s4)
	ld	a0, 640(sp)                     # 8-byte Folded Reload
	sh	a0, 1868(s4)
	ld	a0, 648(sp)                     # 8-byte Folded Reload
	sh	a0, 1870(s4)
	ld	a0, 656(sp)                     # 8-byte Folded Reload
	sh	a0, 1872(s4)
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	sh	a0, 1874(s4)
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	sh	a0, 1876(s4)
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	sh	a0, 1878(s4)
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	sh	a0, 1752(s4)
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	sh	a0, 1754(s4)
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	sh	a0, 1756(s4)
	ld	a0, 808(sp)                     # 8-byte Folded Reload
	sh	a0, 1758(s4)
	ld	a0, 232(sp)                     # 8-byte Folded Reload
	sh	a0, 1760(s4)
	ld	a0, 824(sp)                     # 8-byte Folded Reload
	sh	a0, 1762(s4)
	ld	a0, 240(sp)                     # 8-byte Folded Reload
	sh	a0, 1764(s4)
	ld	a0, 848(sp)                     # 8-byte Folded Reload
	sh	a0, 1766(s4)
	ld	a0, 248(sp)                     # 8-byte Folded Reload
	sh	a0, 1768(s4)
	ld	a0, 872(sp)                     # 8-byte Folded Reload
	sh	a0, 1770(s4)
	ld	a0, 256(sp)                     # 8-byte Folded Reload
	sh	a0, 1772(s4)
	ld	a0, 896(sp)                     # 8-byte Folded Reload
	sh	a0, 1774(s4)
	ld	a0, 264(sp)                     # 8-byte Folded Reload
	sh	a0, 1776(s4)
	ld	a0, 920(sp)                     # 8-byte Folded Reload
	sh	a0, 1778(s4)
	ld	a0, 272(sp)                     # 8-byte Folded Reload
	sh	a0, 1780(s4)
	ld	a0, 952(sp)                     # 8-byte Folded Reload
	sh	a0, 1782(s4)
	ld	a0, 280(sp)                     # 8-byte Folded Reload
	sh	a0, 1784(s4)
	ld	a0, 976(sp)                     # 8-byte Folded Reload
	sh	a0, 1786(s4)
	ld	a0, 288(sp)                     # 8-byte Folded Reload
	sh	a0, 1788(s4)
	ld	a0, 1008(sp)                    # 8-byte Folded Reload
	sh	a0, 1790(s4)
	ld	a0, 1000(sp)                    # 8-byte Folded Reload
	sh	a0, 1792(s4)
	ld	a0, 1016(sp)                    # 8-byte Folded Reload
	sh	a0, 1794(s4)
	ld	a0, 304(sp)                     # 8-byte Folded Reload
	sh	a0, 1796(s4)
	ld	a0, 1024(sp)                    # 8-byte Folded Reload
	sh	a0, 1798(s4)
	ld	a0, 312(sp)                     # 8-byte Folded Reload
	sh	a0, 1800(s4)
	ld	a0, 1032(sp)                    # 8-byte Folded Reload
	sh	a0, 1802(s4)
	ld	a0, 320(sp)                     # 8-byte Folded Reload
	sh	a0, 1804(s4)
	ld	a0, 1040(sp)                    # 8-byte Folded Reload
	sh	a0, 1806(s4)
	ld	a0, 328(sp)                     # 8-byte Folded Reload
	sh	a0, 1808(s4)
	ld	a0, 1048(sp)                    # 8-byte Folded Reload
	sh	a0, 1810(s4)
	ld	a0, 336(sp)                     # 8-byte Folded Reload
	sh	a0, 1812(s4)
	ld	a0, 1056(sp)                    # 8-byte Folded Reload
	sh	a0, 1814(s4)
	ld	a0, 664(sp)                     # 8-byte Folded Reload
	sh	a0, 1818(s4)
	ld	a0, 672(sp)                     # 8-byte Folded Reload
	sh	a0, 1820(s4)
	ld	a0, 680(sp)                     # 8-byte Folded Reload
	sh	a0, 1822(s4)
	fmv.x.w	a0, fa0
	sh	a0, 1816(s4)
	ld	a0, 688(sp)                     # 8-byte Folded Reload
	sh	a0, 24(s10)
	ld	a0, 696(sp)                     # 8-byte Folded Reload
	sh	a0, 26(s10)
	ld	a0, 704(sp)                     # 8-byte Folded Reload
	sh	a0, 28(s10)
	ld	a0, 712(sp)                     # 8-byte Folded Reload
	sh	a0, 30(s10)
	ld	a0, 720(sp)                     # 8-byte Folded Reload
	sh	a0, 32(s10)
	ld	a0, 728(sp)                     # 8-byte Folded Reload
	sh	a0, 34(s10)
	ld	a0, 736(sp)                     # 8-byte Folded Reload
	sh	a0, 36(s10)
	ld	a0, 744(sp)                     # 8-byte Folded Reload
	sh	a0, 38(s10)
	ld	a0, 752(sp)                     # 8-byte Folded Reload
	sh	a0, 40(s10)
	ld	a0, 760(sp)                     # 8-byte Folded Reload
	sh	a0, 42(s10)
	ld	a0, 768(sp)                     # 8-byte Folded Reload
	sh	a0, 44(s10)
	ld	a0, 776(sp)                     # 8-byte Folded Reload
	sh	a0, 46(s10)
	ld	a0, 784(sp)                     # 8-byte Folded Reload
	sh	a0, 48(s10)
	ld	a0, 792(sp)                     # 8-byte Folded Reload
	sh	a0, 50(s10)
	ld	a0, 800(sp)                     # 8-byte Folded Reload
	sh	a0, 52(s10)
	ld	a0, 816(sp)                     # 8-byte Folded Reload
	sh	a0, 54(s10)
	ld	a0, 832(sp)                     # 8-byte Folded Reload
	sh	a0, 56(s10)
	ld	a0, 840(sp)                     # 8-byte Folded Reload
	sh	a0, 58(s10)
	ld	a0, 856(sp)                     # 8-byte Folded Reload
	sh	a0, 60(s10)
	ld	a0, 864(sp)                     # 8-byte Folded Reload
	sh	a0, 62(s10)
	ld	a0, 880(sp)                     # 8-byte Folded Reload
	sh	a0, 64(s10)
	ld	a0, 888(sp)                     # 8-byte Folded Reload
	sh	a0, 66(s10)
	ld	a0, 904(sp)                     # 8-byte Folded Reload
	sh	a0, 68(s10)
	ld	a0, 912(sp)                     # 8-byte Folded Reload
	sh	a0, 70(s10)
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	sh	a0, 72(s10)
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	sh	a0, 74(s10)
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	sh	a0, 76(s10)
	ld	a0, 168(sp)                     # 8-byte Folded Reload
	sh	a0, 78(s10)
	ld	a0, 160(sp)                     # 8-byte Folded Reload
	sh	a0, 80(s10)
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	sh	a0, 82(s10)
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	sh	a0, 84(s10)
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	sh	a0, 86(s10)
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	sh	a0, 2008(s4)
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	sh	a0, 2010(s4)
	ld	a0, 296(sp)                     # 8-byte Folded Reload
	sh	a0, 2012(s4)
	ld	a0, 344(sp)                     # 8-byte Folded Reload
	sh	a0, 2014(s4)
	ld	a0, 352(sp)                     # 8-byte Folded Reload
	sh	a0, 2016(s4)
	ld	a0, 360(sp)                     # 8-byte Folded Reload
	sh	a0, 2018(s4)
	ld	a0, 368(sp)                     # 8-byte Folded Reload
	sh	a0, 2020(s4)
	ld	a0, 376(sp)                     # 8-byte Folded Reload
	sh	a0, 2022(s4)
	lui	a0, 3
	addi	a0, a0, 1216
	add	a3, sp, a0
	ld	a0, 384(sp)                     # 8-byte Folded Reload
	sh	a0, 2024(s4)
	lui	a0, 3
	addi	a0, a0, 1152
	add	a4, sp, a0
	ld	a0, 392(sp)                     # 8-byte Folded Reload
	sh	a0, 2026(s4)
	lui	a0, 3
	addi	a0, a0, 1472
	add	a5, sp, a0
	ld	a0, 400(sp)                     # 8-byte Folded Reload
	sh	a0, 2028(s4)
	lui	a0, 3
	addi	a0, a0, 1408
	add	a6, sp, a0
	fmv.w.x	fa5, zero
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	li	a7, 32
	.loc	1 13 33                         # k135114294379184.py:13:33
	vsetvli	zero, a7, e32, m8, ta, ma
	vsll.vi	v8, v8, 6
	ld	a0, 408(sp)                     # 8-byte Folded Reload
	.loc	1 11 67                         # k135114294379184.py:11:67
	sh	a0, 2030(s4)
	li	a0, 64
	ld	a1, 416(sp)                     # 8-byte Folded Reload
	sh	a1, 2032(s4)
	lui	a1, 3
	addi	a1, a1, 1280
	add	a2, sp, a1
	ld	a1, 424(sp)                     # 8-byte Folded Reload
	sh	a1, 2034(s4)
	li	a1, 27
	slli	a1, a1, 9
	add	a1, sp, a1
	ld	t0, 432(sp)                     # 8-byte Folded Reload
	sh	t0, 2036(s4)
	ld	t0, 928(sp)                     # 8-byte Folded Reload
	sh	t0, 2038(s4)
	ld	t0, 936(sp)                     # 8-byte Folded Reload
	sh	t0, 2040(s4)
	ld	t0, 944(sp)                     # 8-byte Folded Reload
	sh	t0, 2042(s4)
	ld	t0, 960(sp)                     # 8-byte Folded Reload
	sh	t0, 2044(s4)
	ld	t0, 968(sp)                     # 8-byte Folded Reload
	sh	t0, 2046(s4)
	ld	t0, 984(sp)                     # 8-byte Folded Reload
	sh	t0, 0(s10)
	ld	t0, 992(sp)                     # 8-byte Folded Reload
	sh	t0, 2(s10)
	ld	t0, 1064(sp)                    # 8-byte Folded Reload
	sh	t0, 4(s10)
	ld	t0, 1072(sp)                    # 8-byte Folded Reload
	sh	t0, 6(s10)
	ld	t0, 1080(sp)                    # 8-byte Folded Reload
	sh	t0, 8(s10)
	ld	t0, 1088(sp)                    # 8-byte Folded Reload
	sh	t0, 10(s10)
	ld	t0, 1096(sp)                    # 8-byte Folded Reload
	sh	t0, 12(s10)
	ld	t0, 1104(sp)                    # 8-byte Folded Reload
	sh	t0, 14(s10)
	ld	t0, 1112(sp)                    # 8-byte Folded Reload
	sh	t0, 16(s10)
	ld	t0, 1120(sp)                    # 8-byte Folded Reload
	sh	t0, 18(s10)
	ld	t0, 1128(sp)                    # 8-byte Folded Reload
	sh	t0, 20(s10)
	ld	t0, 1136(sp)                    # 8-byte Folded Reload
	sh	t0, 22(s10)
	csrr	t0, vlenb
	slli	t0, t0, 4
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 1728
	add	t0, t0, t1
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 13 30                         # k135114294379184.py:13:30
	vadd.vv	v8, v8, v16
	csrr	t0, vlenb
	li	t1, 72
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 6
	addi	t1, t1, 1728
	add	t0, t0, t1
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	.loc	1 11 67                         # k135114294379184.py:11:67
	vle16.v	v16, (a5)
	vle16.v	v24, (a6)
	vle16.v	v0, (a3)
	vle16.v	v4, (a4)
	vzext.vf2	v8, v16
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	vsll.vi	v8, v8, 16
	.loc	1 12 12                         # k135114294379184.py:12:12
	vfrsub.vf	v8, v8, fa5
	vfrsub.vf	v16, v16, fa5
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v24, v16, a7
	.loc	1 11 67                         # k135114294379184.py:11:67
	vsetvli	zero, a7, e32, m8, ta, ma
	li	a3, 32
	vzext.vf2	v8, v0
	vzext.vf2	v16, v4
	vsll.vi	v16, v16, 16
	vsll.vi	v8, v8, 16
	.loc	1 12 12                         # k135114294379184.py:12:12
	vfrsub.vf	v8, v8, fa5
	vfrsub.vf	v16, v16, fa5
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v0, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v0, v16, a3
	csrr	a3, vlenb
	li	a4, 72
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 6
	addi	a4, a4, 1728
	add	a3, a3, a4
	vl8r.v	v16, (a3)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v16
	vmv2r.v	v16, v0
	csrr	a3, vlenb
	li	a4, 104
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 6
	addi	a4, a4, 1728
	add	a3, a3, a4
	vs8r.v	v16, (a3)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vse16.v	v0, (a2)
	vmv2r.v	v16, v24
	csrr	a0, vlenb
	li	a2, 112
	mul	a0, a0, a2
	add	a0, sp, a0
	lui	a2, 6
	addi	a2, a2, 1728
	add	a0, a0, a2
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vse16.v	v24, (a1)
	lh	a0, 144(s10)
	sd	a0, 856(sp)                     # 8-byte Folded Spill
	lh	a0, 146(s10)
	sd	a0, 864(sp)                     # 8-byte Folded Spill
	lh	a0, 148(s10)
	sd	a0, 872(sp)                     # 8-byte Folded Spill
	lh	a0, 150(s10)
	sd	a0, 888(sp)                     # 8-byte Folded Spill
	lh	a0, 136(s10)
	sd	a0, 824(sp)                     # 8-byte Folded Spill
	lh	a0, 138(s10)
	sd	a0, 832(sp)                     # 8-byte Folded Spill
	lh	a0, 140(s10)
	sd	a0, 840(sp)                     # 8-byte Folded Spill
	lh	a0, 142(s10)
	sd	a0, 848(sp)                     # 8-byte Folded Spill
	lh	a0, 128(s10)
	sd	a0, 792(sp)                     # 8-byte Folded Spill
	lh	a0, 130(s10)
	sd	a0, 800(sp)                     # 8-byte Folded Spill
	lh	a0, 132(s10)
	sd	a0, 808(sp)                     # 8-byte Folded Spill
	lh	a0, 134(s10)
	sd	a0, 816(sp)                     # 8-byte Folded Spill
	lh	a0, 120(s10)
	sd	a0, 760(sp)                     # 8-byte Folded Spill
	lh	a0, 122(s10)
	sd	a0, 768(sp)                     # 8-byte Folded Spill
	lh	a0, 124(s10)
	sd	a0, 776(sp)                     # 8-byte Folded Spill
	lh	a0, 126(s10)
	sd	a0, 784(sp)                     # 8-byte Folded Spill
	lh	a0, 176(s10)
	sd	a0, 984(sp)                     # 8-byte Folded Spill
	lh	a0, 178(s10)
	sd	a0, 992(sp)                     # 8-byte Folded Spill
	lh	a0, 180(s10)
	sd	a0, 1000(sp)                    # 8-byte Folded Spill
	lh	a0, 182(s10)
	sd	a0, 1016(sp)                    # 8-byte Folded Spill
	lh	a0, 168(s10)
	sd	a0, 952(sp)                     # 8-byte Folded Spill
	lh	a0, 170(s10)
	sd	a0, 960(sp)                     # 8-byte Folded Spill
	lh	a0, 172(s10)
	sd	a0, 968(sp)                     # 8-byte Folded Spill
	lh	a0, 174(s10)
	sd	a0, 976(sp)                     # 8-byte Folded Spill
	lh	a0, 160(s10)
	sd	a0, 920(sp)                     # 8-byte Folded Spill
	lh	a0, 162(s10)
	sd	a0, 928(sp)                     # 8-byte Folded Spill
	lh	a0, 164(s10)
	sd	a0, 936(sp)                     # 8-byte Folded Spill
	lh	a0, 166(s10)
	sd	a0, 944(sp)                     # 8-byte Folded Spill
	lh	a0, 152(s10)
	sd	a0, 880(sp)                     # 8-byte Folded Spill
	lh	a0, 154(s10)
	sd	a0, 896(sp)                     # 8-byte Folded Spill
	lh	a0, 156(s10)
	sd	a0, 904(sp)                     # 8-byte Folded Spill
	lh	a0, 158(s10)
	sd	a0, 912(sp)                     # 8-byte Folded Spill
	lh	a0, 208(s10)
	sd	a0, 1112(sp)                    # 8-byte Folded Spill
	lh	a0, 210(s10)
	sd	a0, 1120(sp)                    # 8-byte Folded Spill
	lh	a0, 212(s10)
	sd	a0, 1128(sp)                    # 8-byte Folded Spill
	lh	a0, 214(s10)
	sd	a0, 1136(sp)                    # 8-byte Folded Spill
	lh	a0, 200(s10)
	sd	a0, 1080(sp)                    # 8-byte Folded Spill
	lh	a0, 202(s10)
	sd	a0, 1088(sp)                    # 8-byte Folded Spill
	lh	a0, 204(s10)
	sd	a0, 1096(sp)                    # 8-byte Folded Spill
	lh	a0, 206(s10)
	sd	a0, 1104(sp)                    # 8-byte Folded Spill
	lh	a0, 192(s10)
	sd	a0, 1048(sp)                    # 8-byte Folded Spill
	lh	a0, 194(s10)
	sd	a0, 1056(sp)                    # 8-byte Folded Spill
	lh	a0, 196(s10)
	sd	a0, 1064(sp)                    # 8-byte Folded Spill
	lh	a0, 198(s10)
	sd	a0, 1072(sp)                    # 8-byte Folded Spill
	lh	a0, 184(s10)
	sd	a0, 1008(sp)                    # 8-byte Folded Spill
	lh	a0, 186(s10)
	sd	a0, 1024(sp)                    # 8-byte Folded Spill
	lh	a0, 188(s10)
	sd	a0, 1032(sp)                    # 8-byte Folded Spill
	lh	a0, 190(s10)
	sd	a0, 1040(sp)                    # 8-byte Folded Spill
	lh	a0, 1936(s4)
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	lh	a0, 1938(s4)
	sd	a0, 480(sp)                     # 8-byte Folded Spill
	lh	a0, 1940(s4)
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	lh	a0, 1942(s4)
	sd	a0, 504(sp)                     # 8-byte Folded Spill
	lh	a0, 1928(s4)
	sd	a0, 440(sp)                     # 8-byte Folded Spill
	lh	a0, 1930(s4)
	sd	a0, 448(sp)                     # 8-byte Folded Spill
	lh	a0, 1932(s4)
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	lh	a0, 1934(s4)
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	lh	s2, 1920(s4)
	lh	s9, 1922(s4)
	lh	s8, 1924(s4)
	lh	s7, 1926(s4)
	lh	s5, 1912(s4)
	lh	s3, 1914(s4)
	lh	s11, 1916(s4)
	lh	s10, 1918(s4)
	lh	a0, 1968(s4)
	sd	a0, 600(sp)                     # 8-byte Folded Spill
	lh	a0, 1970(s4)
	sd	a0, 608(sp)                     # 8-byte Folded Spill
	lh	a0, 1972(s4)
	sd	a0, 616(sp)                     # 8-byte Folded Spill
	lh	a0, 1974(s4)
	sd	a0, 632(sp)                     # 8-byte Folded Spill
	lh	a0, 1960(s4)
	sd	a0, 568(sp)                     # 8-byte Folded Spill
	lh	a0, 1962(s4)
	sd	a0, 576(sp)                     # 8-byte Folded Spill
	lh	a0, 1964(s4)
	sd	a0, 584(sp)                     # 8-byte Folded Spill
	lh	a0, 1966(s4)
	sd	a0, 592(sp)                     # 8-byte Folded Spill
	lh	a0, 1952(s4)
	sd	a0, 536(sp)                     # 8-byte Folded Spill
	lh	a0, 1954(s4)
	sd	a0, 544(sp)                     # 8-byte Folded Spill
	lh	a0, 1956(s4)
	sd	a0, 552(sp)                     # 8-byte Folded Spill
	lh	a0, 1958(s4)
	sd	a0, 560(sp)                     # 8-byte Folded Spill
	lh	a0, 1944(s4)
	sd	a0, 496(sp)                     # 8-byte Folded Spill
	lh	a0, 1946(s4)
	sd	a0, 512(sp)                     # 8-byte Folded Spill
	lh	a0, 1948(s4)
	sd	a0, 520(sp)                     # 8-byte Folded Spill
	lh	a0, 1950(s4)
	sd	a0, 528(sp)                     # 8-byte Folded Spill
	lh	a0, 2000(s4)
	sd	a0, 728(sp)                     # 8-byte Folded Spill
	lh	a0, 2002(s4)
	sd	a0, 736(sp)                     # 8-byte Folded Spill
	lh	a0, 2004(s4)
	sd	a0, 744(sp)                     # 8-byte Folded Spill
	lh	a0, 2006(s4)
	sd	a0, 752(sp)                     # 8-byte Folded Spill
	lh	a0, 1992(s4)
	sd	a0, 696(sp)                     # 8-byte Folded Spill
	lh	a0, 1994(s4)
	sd	a0, 704(sp)                     # 8-byte Folded Spill
	lh	a0, 1996(s4)
	sd	a0, 712(sp)                     # 8-byte Folded Spill
	lh	a0, 1998(s4)
	sd	a0, 720(sp)                     # 8-byte Folded Spill
	lh	a0, 1984(s4)
	sd	a0, 664(sp)                     # 8-byte Folded Spill
	lh	a0, 1986(s4)
	sd	a0, 672(sp)                     # 8-byte Folded Spill
	lh	a0, 1988(s4)
	sd	a0, 680(sp)                     # 8-byte Folded Spill
	lh	a0, 1990(s4)
	sd	a0, 688(sp)                     # 8-byte Folded Spill
	lh	a0, 1976(s4)
	sd	a0, 624(sp)                     # 8-byte Folded Spill
	lh	a0, 1978(s4)
	sd	a0, 640(sp)                     # 8-byte Folded Spill
	lh	a0, 1980(s4)
	sd	a0, 648(sp)                     # 8-byte Folded Spill
	lh	a0, 1982(s4)
	sd	a0, 656(sp)                     # 8-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	add	a0, a0, a1
	ld	s6, 1728(a0)                    # 8-byte Folded Reload
	andi	a0, s6, 1
	vsetivli	zero, 16, e64, m8, ta, ma
	ld	a1, 1144(sp)                    # 8-byte Folded Reload
	vadd.vx	v8, v8, a1
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_259
	j	.LBB0_419
.LBB0_259:                              # %else384
	andi	a0, s6, 2
	beqz	a0, .LBB0_260
	j	.LBB0_420
.LBB0_260:                              # %else387
	andi	a0, s6, 4
	beqz	a0, .LBB0_261
	j	.LBB0_421
.LBB0_261:                              # %else390
	andi	a0, s6, 8
	beqz	a0, .LBB0_262
	j	.LBB0_422
.LBB0_262:                              # %else393
	andi	a0, s6, 16
	beqz	a0, .LBB0_263
	j	.LBB0_423
.LBB0_263:                              # %else396
	andi	a0, s6, 32
	beqz	a0, .LBB0_264
	j	.LBB0_424
.LBB0_264:                              # %else399
	andi	a0, s6, 64
	beqz	a0, .LBB0_265
	j	.LBB0_425
.LBB0_265:                              # %else402
	andi	a0, s6, 128
	beqz	a0, .LBB0_266
	j	.LBB0_426
.LBB0_266:                              # %else405
	andi	a0, s6, 256
	beqz	a0, .LBB0_267
	j	.LBB0_427
.LBB0_267:                              # %else408
	andi	a0, s6, 512
	beqz	a0, .LBB0_268
	j	.LBB0_428
.LBB0_268:                              # %else411
	andi	a0, s6, 1024
	beqz	a0, .LBB0_269
	j	.LBB0_429
.LBB0_269:                              # %else414
	slli	a0, s6, 52
	bgez	a0, .LBB0_270
	j	.LBB0_430
.LBB0_270:                              # %else417
	slli	a0, s6, 51
	bgez	a0, .LBB0_272
.LBB0_271:                              # %cond.store418
	.loc	1 0 44 is_stmt 0                # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_272:                              # %else420
	slli	a0, s6, 50
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_274
# %bb.273:                              # %cond.store421
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_274:                              # %else423
	slli	a0, s6, 49
	bgez	a0, .LBB0_276
# %bb.275:                              # %cond.store424
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 456(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_276:                              # %else426
	slli	a0, s6, 48
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_278
# %bb.277:                              # %cond.store427
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 336(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_278:                              # %else429
	slli	a0, s6, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	ld	a1, 1144(sp)                    # 8-byte Folded Reload
	vadd.vx	v8, v16, a1
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_279
	j	.LBB0_431
.LBB0_279:                              # %else432
	slli	a0, s6, 46
	bgez	a0, .LBB0_280
	j	.LBB0_432
.LBB0_280:                              # %else435
	slli	a0, s6, 45
	li	s5, 32
	bgez	a0, .LBB0_281
	j	.LBB0_433
.LBB0_281:                              # %else438
	slli	a0, s6, 44
	ld	s3, 1144(sp)                    # 8-byte Folded Reload
	bgez	a0, .LBB0_282
	j	.LBB0_434
.LBB0_282:                              # %else441
	slli	a0, s6, 43
	bgez	a0, .LBB0_283
	j	.LBB0_435
.LBB0_283:                              # %else444
	slli	a0, s6, 42
	bgez	a0, .LBB0_284
	j	.LBB0_436
.LBB0_284:                              # %else447
	slli	a0, s6, 41
	lui	a1, 2
	addi	a1, a1, 1360
	add	s2, sp, a1
	bgez	a0, .LBB0_285
	j	.LBB0_437
.LBB0_285:                              # %else450
	slli	a0, s6, 40
	bgez	a0, .LBB0_286
	j	.LBB0_438
.LBB0_286:                              # %else453
	slli	a0, s6, 39
	bgez	a0, .LBB0_287
	j	.LBB0_439
.LBB0_287:                              # %else456
	slli	a0, s6, 38
	bgez	a0, .LBB0_288
	j	.LBB0_440
.LBB0_288:                              # %else459
	slli	a0, s6, 37
	bgez	a0, .LBB0_289
	j	.LBB0_441
.LBB0_289:                              # %else462
	slli	a0, s6, 36
	bgez	a0, .LBB0_291
.LBB0_290:                              # %cond.store463
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 464(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_291:                              # %else465
	slli	a0, s6, 35
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s5, e32, m8, ta, ma
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_293
# %bb.292:                              # %cond.store466
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 472(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 21
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_293:                              # %else468
	slli	a0, s6, 34
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s5, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_295
# %bb.294:                              # %cond.store469
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 480(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_295:                              # %else471
	slli	a0, s6, 33
	bgez	a0, .LBB0_297
# %bb.296:                              # %cond.store472
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_297:                              # %else474
	slli	a0, s6, 32
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_299
# %bb.298:                              # %cond.store475
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 504(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_299:                              # %else477
	slli	a0, s6, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s3
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_300
	j	.LBB0_442
.LBB0_300:                              # %else480
	slli	a0, s6, 30
	bgez	a0, .LBB0_301
	j	.LBB0_443
.LBB0_301:                              # %else483
	slli	a0, s6, 29
	bgez	a0, .LBB0_302
	j	.LBB0_444
.LBB0_302:                              # %else486
	slli	a0, s6, 28
	bgez	a0, .LBB0_303
	j	.LBB0_445
.LBB0_303:                              # %else489
	slli	a0, s6, 27
	bgez	a0, .LBB0_304
	j	.LBB0_446
.LBB0_304:                              # %else492
	slli	a0, s6, 26
	bgez	a0, .LBB0_305
	j	.LBB0_447
.LBB0_305:                              # %else495
	slli	a0, s6, 25
	bgez	a0, .LBB0_306
	j	.LBB0_448
.LBB0_306:                              # %else498
	slli	a0, s6, 24
	bgez	a0, .LBB0_307
	j	.LBB0_449
.LBB0_307:                              # %else501
	slli	a0, s6, 23
	bgez	a0, .LBB0_308
	j	.LBB0_450
.LBB0_308:                              # %else504
	slli	a0, s6, 22
	bgez	a0, .LBB0_309
	j	.LBB0_451
.LBB0_309:                              # %else507
	slli	a0, s6, 21
	bgez	a0, .LBB0_310
	j	.LBB0_452
.LBB0_310:                              # %else510
	slli	a0, s6, 20
	lui	a1, 2
	addi	a1, a1, -776
	add	s2, sp, a1
	bgez	a0, .LBB0_311
	j	.LBB0_453
.LBB0_311:                              # %else513
	slli	a0, s6, 19
	bgez	a0, .LBB0_313
.LBB0_312:                              # %cond.store514
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_313:                              # %else516
	slli	a0, s6, 18
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_315
# %bb.314:                              # %cond.store517
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_315:                              # %else519
	slli	a0, s6, 17
	bgez	a0, .LBB0_317
# %bb.316:                              # %cond.store520
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_317:                              # %else522
	slli	a0, s6, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_319
# %bb.318:                              # %cond.store523
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 632(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_319:                              # %else525
	slli	a0, s6, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s3
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_320
	j	.LBB0_454
.LBB0_320:                              # %else528
	slli	a0, s6, 14
	bgez	a0, .LBB0_321
	j	.LBB0_455
.LBB0_321:                              # %else531
	slli	a0, s6, 13
	bgez	a0, .LBB0_322
	j	.LBB0_456
.LBB0_322:                              # %else534
	slli	a0, s6, 12
	bgez	a0, .LBB0_323
	j	.LBB0_457
.LBB0_323:                              # %else537
	slli	a0, s6, 11
	bgez	a0, .LBB0_324
	j	.LBB0_458
.LBB0_324:                              # %else540
	slli	a0, s6, 10
	bgez	a0, .LBB0_325
	j	.LBB0_459
.LBB0_325:                              # %else543
	slli	a0, s6, 9
	bgez	a0, .LBB0_326
	j	.LBB0_460
.LBB0_326:                              # %else546
	slli	a0, s6, 8
	bgez	a0, .LBB0_327
	j	.LBB0_461
.LBB0_327:                              # %else549
	slli	a0, s6, 7
	bgez	a0, .LBB0_328
	j	.LBB0_462
.LBB0_328:                              # %else552
	slli	a0, s6, 6
	bgez	a0, .LBB0_329
	j	.LBB0_463
.LBB0_329:                              # %else555
	slli	a0, s6, 5
	bgez	a0, .LBB0_330
	j	.LBB0_464
.LBB0_330:                              # %else558
	slli	a0, s6, 4
	bgez	a0, .LBB0_332
.LBB0_331:                              # %cond.store559
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 720(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_332:                              # %else561
	slli	a0, s6, 3
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s5, e32, m8, ta, ma
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_334
# %bb.333:                              # %cond.store562
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 728(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 15
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_334:                              # %else564
	slli	a0, s6, 2
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s5, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_336
# %bb.335:                              # %cond.store565
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 736(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_336:                              # %else567
	slli	a0, s6, 1
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl1r.v	v8, (a1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	bgez	a0, .LBB0_338
# %bb.337:                              # %cond.store568
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 744(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs1r.v	v24, (a0)                       # vscale x 8-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl1r.v	v24, (a0)                       # vscale x 8-byte Folded Reload
	li	a0, 29
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_338:                              # %else570
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	s4, v24
	bgez	s6, .LBB0_340
# %bb.339:                              # %cond.store571
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 752(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_340:                              # %else573
	andi	a0, s4, 1
	vadd.vx	v8, v16, s3
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_341
	j	.LBB0_465
.LBB0_341:                              # %else576
	andi	a0, s4, 2
	beqz	a0, .LBB0_342
	j	.LBB0_466
.LBB0_342:                              # %else579
	andi	a0, s4, 4
	beqz	a0, .LBB0_343
	j	.LBB0_467
.LBB0_343:                              # %else582
	andi	a0, s4, 8
	beqz	a0, .LBB0_344
	j	.LBB0_468
.LBB0_344:                              # %else585
	andi	a0, s4, 16
	lui	a1, 1
	addi	a1, a1, 1208
	add	s2, sp, a1
	beqz	a0, .LBB0_345
	j	.LBB0_469
.LBB0_345:                              # %else588
	andi	a0, s4, 32
	beqz	a0, .LBB0_346
	j	.LBB0_470
.LBB0_346:                              # %else591
	andi	a0, s4, 64
	beqz	a0, .LBB0_347
	j	.LBB0_471
.LBB0_347:                              # %else594
	andi	a0, s4, 128
	beqz	a0, .LBB0_348
	j	.LBB0_472
.LBB0_348:                              # %else597
	andi	a0, s4, 256
	beqz	a0, .LBB0_349
	j	.LBB0_473
.LBB0_349:                              # %else600
	andi	a0, s4, 512
	beqz	a0, .LBB0_350
	j	.LBB0_474
.LBB0_350:                              # %else603
	andi	a0, s4, 1024
	beqz	a0, .LBB0_351
	j	.LBB0_475
.LBB0_351:                              # %else606
	slli	a0, s4, 52
	bgez	a0, .LBB0_352
	j	.LBB0_476
.LBB0_352:                              # %else609
	slli	a0, s4, 51
	bgez	a0, .LBB0_354
.LBB0_353:                              # %cond.store610
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 3
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_354:                              # %else612
	slli	a0, s4, 50
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_356
# %bb.355:                              # %cond.store613
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_356:                              # %else615
	slli	a0, s4, 49
	bgez	a0, .LBB0_358
# %bb.357:                              # %cond.store616
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_358:                              # %else618
	slli	a0, s4, 48
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_360
# %bb.359:                              # %cond.store619
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_360:                              # %else621
	slli	a0, s4, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s3
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_361
	j	.LBB0_477
.LBB0_361:                              # %else624
	slli	a0, s4, 46
	bgez	a0, .LBB0_362
	j	.LBB0_478
.LBB0_362:                              # %else627
	slli	a0, s4, 45
	bgez	a0, .LBB0_363
	j	.LBB0_479
.LBB0_363:                              # %else630
	slli	a0, s4, 44
	bgez	a0, .LBB0_364
	j	.LBB0_480
.LBB0_364:                              # %else633
	slli	a0, s4, 43
	bgez	a0, .LBB0_365
	j	.LBB0_481
.LBB0_365:                              # %else636
	slli	a0, s4, 42
	bgez	a0, .LBB0_366
	j	.LBB0_482
.LBB0_366:                              # %else639
	slli	a0, s4, 41
	bgez	a0, .LBB0_367
	j	.LBB0_483
.LBB0_367:                              # %else642
	slli	a0, s4, 40
	bgez	a0, .LBB0_368
	j	.LBB0_484
.LBB0_368:                              # %else645
	slli	a0, s4, 39
	addi	s2, sp, 2047
	addi	s2, s2, 1121
	bgez	a0, .LBB0_369
	j	.LBB0_485
.LBB0_369:                              # %else648
	slli	a0, s4, 38
	bgez	a0, .LBB0_370
	j	.LBB0_486
.LBB0_370:                              # %else651
	slli	a0, s4, 37
	bgez	a0, .LBB0_371
	j	.LBB0_487
.LBB0_371:                              # %else654
	slli	a0, s4, 36
	bgez	a0, .LBB0_373
.LBB0_372:                              # %cond.store655
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 848(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_373:                              # %else657
	slli	a0, s4, 35
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s5, e32, m8, ta, ma
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_375
# %bb.374:                              # %cond.store658
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 856(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 9
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_375:                              # %else660
	slli	a0, s4, 34
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s5, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_377
# %bb.376:                              # %cond.store661
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 864(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_377:                              # %else663
	slli	a0, s4, 33
	bgez	a0, .LBB0_379
# %bb.378:                              # %cond.store664
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 872(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_379:                              # %else666
	slli	a0, s4, 32
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_381
# %bb.380:                              # %cond.store667
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 888(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_381:                              # %else669
	slli	a0, s4, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s3
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_382
	j	.LBB0_488
.LBB0_382:                              # %else672
	slli	a0, s4, 30
	bgez	a0, .LBB0_383
	j	.LBB0_489
.LBB0_383:                              # %else675
	slli	a0, s4, 29
	bgez	a0, .LBB0_384
	j	.LBB0_490
.LBB0_384:                              # %else678
	slli	a0, s4, 28
	bgez	a0, .LBB0_385
	j	.LBB0_491
.LBB0_385:                              # %else681
	slli	a0, s4, 27
	bgez	a0, .LBB0_386
	j	.LBB0_492
.LBB0_386:                              # %else684
	slli	a0, s4, 26
	bgez	a0, .LBB0_387
	j	.LBB0_493
.LBB0_387:                              # %else687
	slli	a0, s4, 25
	bgez	a0, .LBB0_388
	j	.LBB0_494
.LBB0_388:                              # %else690
	slli	a0, s4, 24
	bgez	a0, .LBB0_389
	j	.LBB0_495
.LBB0_389:                              # %else693
	slli	a0, s4, 23
	bgez	a0, .LBB0_390
	j	.LBB0_496
.LBB0_390:                              # %else696
	slli	a0, s4, 22
	bgez	a0, .LBB0_391
	j	.LBB0_497
.LBB0_391:                              # %else699
	slli	a0, s4, 21
	bgez	a0, .LBB0_392
	j	.LBB0_498
.LBB0_392:                              # %else702
	slli	a0, s4, 20
	bgez	a0, .LBB0_393
	j	.LBB0_499
.LBB0_393:                              # %else705
	slli	a0, s4, 19
	bgez	a0, .LBB0_395
.LBB0_394:                              # %cond.store706
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 984(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1025
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_395:                              # %else708
	slli	a0, s4, 18
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_397
# %bb.396:                              # %cond.store709
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 992(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 897
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1048(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_397:                              # %else711
	slli	a0, s4, 17
	bgez	a0, .LBB0_399
# %bb.398:                              # %cond.store712
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1000(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 769
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1168(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_399:                              # %else714
	slli	a0, s4, 16
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v8, v8
	bgez	a0, .LBB0_401
# %bb.400:                              # %cond.store715
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1016(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 641
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1288(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_401:                              # %else717
	slli	a0, s4, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v16, s3
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_402
	j	.LBB0_500
.LBB0_402:                              # %else720
	slli	a0, s4, 14
	bgez	a0, .LBB0_403
	j	.LBB0_501
.LBB0_403:                              # %else723
	slli	a0, s4, 13
	bgez	a0, .LBB0_404
	j	.LBB0_502
.LBB0_404:                              # %else726
	slli	a0, s4, 12
	bgez	a0, .LBB0_405
	j	.LBB0_503
.LBB0_405:                              # %else729
	slli	a0, s4, 11
	bgez	a0, .LBB0_406
	j	.LBB0_504
.LBB0_406:                              # %else732
	slli	a0, s4, 10
	bgez	a0, .LBB0_407
	j	.LBB0_505
.LBB0_407:                              # %else735
	slli	a0, s4, 9
	bgez	a0, .LBB0_408
	j	.LBB0_506
.LBB0_408:                              # %else738
	slli	a0, s4, 8
	bgez	a0, .LBB0_409
	j	.LBB0_507
.LBB0_409:                              # %else741
	slli	a0, s4, 7
	bgez	a0, .LBB0_410
	j	.LBB0_508
.LBB0_410:                              # %else744
	slli	a0, s4, 6
	bgez	a0, .LBB0_411
	j	.LBB0_509
.LBB0_411:                              # %else747
	slli	a0, s4, 5
	bgez	a0, .LBB0_412
	j	.LBB0_510
.LBB0_412:                              # %else750
	slli	a0, s4, 4
	bgez	a0, .LBB0_413
	j	.LBB0_511
.LBB0_413:                              # %else753
	slli	a0, s4, 3
	bgez	a0, .LBB0_414
	j	.LBB0_512
.LBB0_414:                              # %else756
	slli	a0, s4, 2
	bgez	a0, .LBB0_415
	j	.LBB0_513
.LBB0_415:                              # %else759
	slli	a0, s4, 1
	bgez	a0, .LBB0_416
	j	.LBB0_514
.LBB0_416:                              # %else762
	bgez	s4, .LBB0_418
.LBB0_417:                              # %cond.store763
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1136(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1152
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1272(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_418:                              # %else765
	.loc	1 13 4 epilogue_begin           # k135114294379184.py:13:4
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
.LBB0_419:                              # %cond.store
	.cfi_restore_state
	.loc	1 0 4                           # k135114294379184.py:0:4
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 2
	bnez	a0, .LBB0_420
	j	.LBB0_260
.LBB0_420:                              # %cond.store385
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 4
	bnez	a0, .LBB0_421
	j	.LBB0_261
.LBB0_421:                              # %cond.store388
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 8
	bnez	a0, .LBB0_422
	j	.LBB0_262
.LBB0_422:                              # %cond.store391
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 16
	bnez	a0, .LBB0_423
	j	.LBB0_263
.LBB0_423:                              # %cond.store394
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 13
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 32
	bnez	a0, .LBB0_424
	j	.LBB0_264
.LBB0_424:                              # %cond.store397
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 64
	bnez	a0, .LBB0_425
	j	.LBB0_265
.LBB0_425:                              # %cond.store400
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 128
	bnez	a0, .LBB0_426
	j	.LBB0_266
.LBB0_426:                              # %cond.store403
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 256
	bnez	a0, .LBB0_427
	j	.LBB0_267
.LBB0_427:                              # %cond.store406
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 512
	bnez	a0, .LBB0_428
	j	.LBB0_268
.LBB0_428:                              # %cond.store409
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 1024
	bnez	a0, .LBB0_429
	j	.LBB0_269
.LBB0_429:                              # %cond.store412
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 52
	bltz	a0, .LBB0_430
	j	.LBB0_270
.LBB0_430:                              # %cond.store415
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 44                         # k135114294379184.py:13:44
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 51
	bgez	a0, .LBB0_515
	j	.LBB0_271
.LBB0_515:                              # %cond.store415
	j	.LBB0_272
.LBB0_431:                              # %cond.store430
	slli	s5, s5, 16
	fmv.w.x	fa0, s5
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 46
	bltz	a0, .LBB0_432
	j	.LBB0_280
.LBB0_432:                              # %cond.store433
	slli	s3, s3, 16
	fmv.w.x	fa0, s3
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 45
	li	s5, 32
	bltz	a0, .LBB0_433
	j	.LBB0_281
.LBB0_433:                              # %cond.store436
	slli	s11, s11, 16
	fmv.w.x	fa0, s11
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 44
	ld	s3, 1144(sp)                    # 8-byte Folded Reload
	bltz	a0, .LBB0_434
	j	.LBB0_282
.LBB0_434:                              # %cond.store439
	slli	s10, s10, 16
	fmv.w.x	fa0, s10
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 43
	bltz	a0, .LBB0_435
	j	.LBB0_283
.LBB0_435:                              # %cond.store442
	slli	s2, s2, 16
	fmv.w.x	fa0, s2
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 42
	bltz	a0, .LBB0_436
	j	.LBB0_284
.LBB0_436:                              # %cond.store445
	slli	s9, s9, 16
	fmv.w.x	fa0, s9
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 41
	lui	a1, 2
	addi	a1, a1, 1360
	add	s2, sp, a1
	bltz	a0, .LBB0_437
	j	.LBB0_285
.LBB0_437:                              # %cond.store448
	.loc	1 0 44                          # k135114294379184.py:0:44
	slli	s8, s8, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, s8
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 40
	bltz	a0, .LBB0_438
	j	.LBB0_286
.LBB0_438:                              # %cond.store451
	.loc	1 0 44                          # k135114294379184.py:0:44
	slli	s7, s7, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, s7
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 39
	bltz	a0, .LBB0_439
	j	.LBB0_287
.LBB0_439:                              # %cond.store454
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 440(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 38
	bltz	a0, .LBB0_440
	j	.LBB0_288
.LBB0_440:                              # %cond.store457
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 448(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 37
	bltz	a0, .LBB0_441
	j	.LBB0_289
.LBB0_441:                              # %cond.store460
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 456(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 36
	bgez	a0, .LBB0_516
	j	.LBB0_290
.LBB0_516:                              # %cond.store460
	j	.LBB0_291
.LBB0_442:                              # %cond.store478
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 496(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 30
	bltz	a0, .LBB0_443
	j	.LBB0_301
.LBB0_443:                              # %cond.store481
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 512(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 29
	bltz	a0, .LBB0_444
	j	.LBB0_302
.LBB0_444:                              # %cond.store484
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 520(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 28
	bltz	a0, .LBB0_445
	j	.LBB0_303
.LBB0_445:                              # %cond.store487
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 528(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 27
	bltz	a0, .LBB0_446
	j	.LBB0_304
.LBB0_446:                              # %cond.store490
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 536(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 26
	bltz	a0, .LBB0_447
	j	.LBB0_305
.LBB0_447:                              # %cond.store493
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 544(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 25
	bltz	a0, .LBB0_448
	j	.LBB0_306
.LBB0_448:                              # %cond.store496
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 24
	bltz	a0, .LBB0_449
	j	.LBB0_307
.LBB0_449:                              # %cond.store499
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 23
	bltz	a0, .LBB0_450
	j	.LBB0_308
.LBB0_450:                              # %cond.store502
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 22
	bltz	a0, .LBB0_451
	j	.LBB0_309
.LBB0_451:                              # %cond.store505
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 21
	bltz	a0, .LBB0_452
	j	.LBB0_310
.LBB0_452:                              # %cond.store508
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 20
	lui	a1, 2
	addi	a1, a1, -776
	add	s2, sp, a1
	bltz	a0, .LBB0_453
	j	.LBB0_311
.LBB0_453:                              # %cond.store511
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 19
	bgez	a0, .LBB0_517
	j	.LBB0_312
.LBB0_517:                              # %cond.store511
	j	.LBB0_313
.LBB0_454:                              # %cond.store526
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 14
	bltz	a0, .LBB0_455
	j	.LBB0_321
.LBB0_455:                              # %cond.store529
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 640(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 13
	bltz	a0, .LBB0_456
	j	.LBB0_322
.LBB0_456:                              # %cond.store532
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 648(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 12
	bltz	a0, .LBB0_457
	j	.LBB0_323
.LBB0_457:                              # %cond.store535
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 656(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 11
	bltz	a0, .LBB0_458
	j	.LBB0_324
.LBB0_458:                              # %cond.store538
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 664(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1320(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 10
	bltz	a0, .LBB0_459
	j	.LBB0_325
.LBB0_459:                              # %cond.store541
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 672(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1200(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 9
	bltz	a0, .LBB0_460
	j	.LBB0_326
.LBB0_460:                              # %cond.store544
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 680(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 8
	bltz	a0, .LBB0_461
	j	.LBB0_327
.LBB0_461:                              # %cond.store547
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 688(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 7
	bltz	a0, .LBB0_462
	j	.LBB0_328
.LBB0_462:                              # %cond.store550
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 696(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 6
	bltz	a0, .LBB0_463
	j	.LBB0_329
.LBB0_463:                              # %cond.store553
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 704(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 5
	bltz	a0, .LBB0_464
	j	.LBB0_330
.LBB0_464:                              # %cond.store556
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 712(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 31
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 4
	bgez	a0, .LBB0_518
	j	.LBB0_331
.LBB0_518:                              # %cond.store556
	j	.LBB0_332
.LBB0_465:                              # %cond.store574
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 2
	bnez	a0, .LBB0_466
	j	.LBB0_342
.LBB0_466:                              # %cond.store577
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 4
	bnez	a0, .LBB0_467
	j	.LBB0_343
.LBB0_467:                              # %cond.store580
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 8
	bnez	a0, .LBB0_468
	j	.LBB0_344
.LBB0_468:                              # %cond.store583
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 16
	lui	a1, 1
	addi	a1, a1, 1208
	add	s2, sp, a1
	bnez	a0, .LBB0_469
	j	.LBB0_345
.LBB0_469:                              # %cond.store586
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 7
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 32
	bnez	a0, .LBB0_470
	j	.LBB0_346
.LBB0_470:                              # %cond.store589
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 64
	bnez	a0, .LBB0_471
	j	.LBB0_347
.LBB0_471:                              # %cond.store592
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 27
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 128
	bnez	a0, .LBB0_472
	j	.LBB0_348
.LBB0_472:                              # %cond.store595
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 256
	bnez	a0, .LBB0_473
	j	.LBB0_349
.LBB0_473:                              # %cond.store598
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 13
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 512
	bnez	a0, .LBB0_474
	j	.LBB0_350
.LBB0_474:                              # %cond.store601
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 1024
	bnez	a0, .LBB0_475
	j	.LBB0_351
.LBB0_475:                              # %cond.store604
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 52
	bltz	a0, .LBB0_476
	j	.LBB0_352
.LBB0_476:                              # %cond.store607
	.loc	1 0 44                          # k135114294379184.py:0:44
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 6
	addi	a1, a1, 1728
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 51
	bgez	a0, .LBB0_519
	j	.LBB0_353
.LBB0_519:                              # %cond.store607
	j	.LBB0_354
.LBB0_477:                              # %cond.store622
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 760(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 46
	bltz	a0, .LBB0_478
	j	.LBB0_362
.LBB0_478:                              # %cond.store625
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 768(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 45
	bltz	a0, .LBB0_479
	j	.LBB0_363
.LBB0_479:                              # %cond.store628
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 776(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 44
	bltz	a0, .LBB0_480
	j	.LBB0_364
.LBB0_480:                              # %cond.store631
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 784(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 43
	bltz	a0, .LBB0_481
	j	.LBB0_365
.LBB0_481:                              # %cond.store634
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 792(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 42
	bltz	a0, .LBB0_482
	j	.LBB0_366
.LBB0_482:                              # %cond.store637
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 800(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 41
	bltz	a0, .LBB0_483
	j	.LBB0_367
.LBB0_483:                              # %cond.store640
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 808(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 40
	bltz	a0, .LBB0_484
	j	.LBB0_368
.LBB0_484:                              # %cond.store643
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 816(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 39
	addi	s2, sp, 2047
	addi	s2, s2, 1121
	bltz	a0, .LBB0_485
	j	.LBB0_369
.LBB0_485:                              # %cond.store646
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 824(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 38
	bltz	a0, .LBB0_486
	j	.LBB0_370
.LBB0_486:                              # %cond.store649
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 832(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 37
	bltz	a0, .LBB0_487
	j	.LBB0_371
.LBB0_487:                              # %cond.store652
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 840(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 36
	bgez	a0, .LBB0_520
	j	.LBB0_372
.LBB0_520:                              # %cond.store652
	j	.LBB0_373
.LBB0_488:                              # %cond.store670
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 880(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 30
	bltz	a0, .LBB0_489
	j	.LBB0_383
.LBB0_489:                              # %cond.store673
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 896(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 29
	bltz	a0, .LBB0_490
	j	.LBB0_384
.LBB0_490:                              # %cond.store676
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 904(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 28
	bltz	a0, .LBB0_491
	j	.LBB0_385
.LBB0_491:                              # %cond.store679
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 912(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 27
	bltz	a0, .LBB0_492
	j	.LBB0_386
.LBB0_492:                              # %cond.store682
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 920(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 26
	bltz	a0, .LBB0_493
	j	.LBB0_387
.LBB0_493:                              # %cond.store685
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 928(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1921
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 25
	bltz	a0, .LBB0_494
	j	.LBB0_388
.LBB0_494:                              # %cond.store688
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 936(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1793
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 24
	bltz	a0, .LBB0_495
	j	.LBB0_389
.LBB0_495:                              # %cond.store691
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 944(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1665
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 23
	bltz	a0, .LBB0_496
	j	.LBB0_390
.LBB0_496:                              # %cond.store694
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 952(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1537
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 22
	bltz	a0, .LBB0_497
	j	.LBB0_391
.LBB0_497:                              # %cond.store697
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 960(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1409
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 21
	bltz	a0, .LBB0_498
	j	.LBB0_392
.LBB0_498:                              # %cond.store700
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 968(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1281
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 20
	bltz	a0, .LBB0_499
	j	.LBB0_393
.LBB0_499:                              # %cond.store703
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 976(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1153
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 19
	bgez	a0, .LBB0_521
	j	.LBB0_394
.LBB0_521:                              # %cond.store703
	j	.LBB0_395
.LBB0_500:                              # %cond.store718
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1008(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 14
	bltz	a0, .LBB0_501
	j	.LBB0_403
.LBB0_501:                              # %cond.store721
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1024(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 13
	bltz	a0, .LBB0_502
	j	.LBB0_404
.LBB0_502:                              # %cond.store724
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1032(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 12
	bltz	a0, .LBB0_503
	j	.LBB0_405
.LBB0_503:                              # %cond.store727
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1040(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 11
	bltz	a0, .LBB0_504
	j	.LBB0_406
.LBB0_504:                              # %cond.store730
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1048(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 513
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1504(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 10
	bltz	a0, .LBB0_505
	j	.LBB0_407
.LBB0_505:                              # %cond.store733
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1056(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 385
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1624(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 9
	bltz	a0, .LBB0_506
	j	.LBB0_408
.LBB0_506:                              # %cond.store736
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1064(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 257
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1744(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 8
	bltz	a0, .LBB0_507
	j	.LBB0_409
.LBB0_507:                              # %cond.store739
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1072(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 129
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1864(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 7
	bltz	a0, .LBB0_508
	j	.LBB0_410
.LBB0_508:                              # %cond.store742
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1080(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1984(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 6
	bltz	a0, .LBB0_509
	j	.LBB0_411
.LBB0_509:                              # %cond.store745
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1088(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1920
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1992(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 5
	bltz	a0, .LBB0_510
	j	.LBB0_412
.LBB0_510:                              # %cond.store748
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1096(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1792
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1872(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 4
	bltz	a0, .LBB0_511
	j	.LBB0_413
.LBB0_511:                              # %cond.store751
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1104(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1664
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1752(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 3
	bltz	a0, .LBB0_512
	j	.LBB0_414
.LBB0_512:                              # %cond.store754
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1112(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1536
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1632(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 2
	bltz	a0, .LBB0_513
	j	.LBB0_415
.LBB0_513:                              # %cond.store757
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1120(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1408
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1512(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 1
	bltz	a0, .LBB0_514
	j	.LBB0_416
.LBB0_514:                              # %cond.store760
	.loc	1 0 44                          # k135114294379184.py:0:44
	ld	a0, 1128(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 13 44                         # k135114294379184.py:13:44
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1280
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1728
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1392(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	bgez	s4, .LBB0_522
	j	.LBB0_417
.LBB0_522:                              # %cond.store760
	j	.LBB0_418
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
	.asciz	"k135114294379184.py"           # string offset=7 ; k135114294379184.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
