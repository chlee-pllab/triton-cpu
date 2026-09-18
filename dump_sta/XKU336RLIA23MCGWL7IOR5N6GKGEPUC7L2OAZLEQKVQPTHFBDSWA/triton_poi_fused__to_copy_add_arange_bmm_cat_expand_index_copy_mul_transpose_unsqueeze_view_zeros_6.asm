	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_zeros_6 # -- Begin function triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_zeros_6
	.p2align	2
	.type	triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_zeros_6,@function
triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_zeros_6: # @triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_zeros_6
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294089648.py"
	.loc	1 2 0                           # k135114294089648.py:2:0
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
	lui	a5, 6
	addi	a5, a5, 1424
	sub	sp, sp, a5
	csrr	a5, vlenb
	li	a7, 209
	mul	a5, a5, a7
	sub	sp, sp, a5
	andi	sp, sp, -128
	mv	t2, a4
	sd	a3, 432(sp)                     # 8-byte Folded Spill
	sd	a2, 416(sp)                     # 8-byte Folded Spill
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114294089648.py:4:33
	slliw	t1, a6, 7
	li	a3, 32
	lui	a2, 599186
	.loc	1 5 23                          # k135114294089648.py:5:23
	vsetvli	zero, a3, e32, m8, ta, ma
	vid.v	v8
	addi	a6, a2, 1171
	vor.vx	v16, v8, t1
	csrr	a2, vlenb
	li	a4, 193
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 7
	addi	a4, a4, -832
	add	a2, a2, a4
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 9 19                          # k135114294089648.py:9:19
	vmulh.vx	v8, v16, a6
	vadd.vv	v8, v8, v16
	vsra.vi	v8, v8, 9
	vsrl.vi	v24, v8, 31
	vadd.vv	v8, v8, v24
	csrr	a2, vlenb
	li	a4, 81
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 7
	addi	a4, a4, -832
	add	a2, a2, a4
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 5 23                          # k135114294089648.py:5:23
	vmv.v.x	v8, t1
	.loc	1 7 21                          # k135114294089648.py:7:21
	vsra.vi	v8, v8, 31
	vsrl.vi	v8, v8, 26
	csrr	a2, vlenb
	li	a4, 169
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 7
	addi	a4, a4, -832
	add	a2, a2, a4
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v16, v8
	vsra.vi	v24, v8, 6
	.loc	1 7 27 is_stmt 0                # k135114294089648.py:7:27
	vmulh.vx	v0, v24, a6
	vadd.vv	v0, v0, v24
	vsra.vi	v0, v0, 3
	vsrl.vi	v16, v0, 31
	vadd.vv	v0, v0, v16
	li	a4, 14
	vnmsub.vx	v0, a4, v24
	li	a5, -64
	.loc	1 8 19 is_stmt 1                # k135114294089648.py:8:19
	vand.vx	v8, v8, a5
	csrr	a2, vlenb
	li	a7, 193
	mul	a2, a2, a7
	add	a2, sp, a2
	lui	a7, 7
	addi	a7, a7, -832
	add	a2, a2, a7
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	vsub.vv	v16, v16, v8
	csrr	a2, vlenb
	li	a7, 81
	mul	a2, a2, a7
	add	a2, sp, a2
	lui	a7, 7
	addi	a7, a7, -832
	add	a2, a2, a7
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 13 38                         # k135114294089648.py:13:38
	vsll.vi	v8, v8, 6
	csrr	a2, vlenb
	li	a7, 89
	mul	a2, a2, a7
	add	a2, sp, a2
	lui	a7, 7
	addi	a7, a7, -832
	add	a2, a2, a7
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 13 35 is_stmt 0               # k135114294089648.py:13:35
	vadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	a7, 105
	mul	a2, a2, a7
	add	a2, sp, a2
	lui	a7, 7
	addi	a7, a7, -832
	add	a2, a2, a7
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 13 47                         # k135114294089648.py:13:47
	vsll.vi	v16, v0, 7
	.loc	1 13 43                         # k135114294089648.py:13:43
	vadd.vv	v24, v8, v16
	li	a2, 96
	li	a7, 64
	li	t0, 1792
	vid.v	v0
	.loc	1 5 23 is_stmt 1                # k135114294089648.py:5:23
	vadd.vx	v8, v0, a2
	vor.vx	v16, v8, t1
	vadd.vx	v8, v0, a7
	vor.vx	v0, v8, t1
	csrr	a2, vlenb
	li	a7, 201
	mul	a2, a2, a7
	add	a2, sp, a2
	lui	a7, 7
	addi	a7, a7, -832
	add	a2, a2, a7
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114294089648.py:6:21
	vmslt.vx	v8, v16, t0
	csrr	a2, vlenb
	li	a7, 185
	mul	a2, a2, a7
	add	a2, sp, a2
	lui	a7, 7
	addi	a7, a7, -832
	add	a2, a2, a7
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vmslt.vx	v9, v0, t0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	csrr	a2, vlenb
	li	a7, 177
	mul	a2, a2, a7
	add	a2, sp, a2
	lui	a7, 7
	addi	a7, a7, -832
	add	a2, a2, a7
	vs1r.v	v9, (a2)                        # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294089648.py:5:23
	vsetvli	zero, a3, e32, m8, ta, ma
	vid.v	v8
	vadd.vx	v8, v8, a3
	sd	t1, 424(sp)                     # 8-byte Folded Spill
	vor.vx	v16, v8, t1
	csrr	a2, vlenb
	li	a7, 193
	mul	a2, a2, a7
	add	a2, sp, a2
	lui	a7, 7
	addi	a7, a7, -832
	add	a2, a2, a7
	vl8r.v	v0, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 6 21                          # k135114294089648.py:6:21
	vmslt.vx	v9, v0, t0
	vmv.v.v	v0, v16
	vmslt.vx	v8, v16, t0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v8, 4
	csrr	a2, vlenb
	li	a7, 177
	mul	a2, a2, a7
	add	a2, sp, a2
	lui	a7, 7
	addi	a7, a7, -832
	add	a2, a2, a7
	vl1r.v	v8, (a2)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v9, v8, 8
	csrr	a2, vlenb
	slli	a7, a2, 7
	add	a2, a7, a2
	add	a2, sp, a2
	lui	a7, 7
	addi	a7, a7, -832
	add	a2, a2, a7
	vs1r.v	v9, (a2)                        # vscale x 8-byte Folded Spill
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	a7, v9
	andi	a2, a7, 1
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v8, v24, v24
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v8, a1
	beqz	a2, .LBB0_2
# %bb.1:                                # %cond.load
	vmv.x.s	a2, v16
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs10, zero
	fmv.w.x	fs8, a2
	fsw	fs10, 800(sp)                   # 4-byte Folded Spill
	fmv.s	ft0, fs10
	fsw	fs10, 816(sp)                   # 4-byte Folded Spill
	fmv.s	ft1, fs10
	fsw	fs10, 832(sp)                   # 4-byte Folded Spill
	fmv.s	ft2, fs10
	fsw	fs10, 864(sp)                   # 4-byte Folded Spill
	fmv.s	ft3, fs10
	fsw	fs10, 888(sp)                   # 4-byte Folded Spill
	fmv.s	ft4, fs10
	fsw	fs10, 912(sp)                   # 4-byte Folded Spill
	fmv.s	ft5, fs10
	fsw	fs10, 936(sp)                   # 4-byte Folded Spill
	fmv.s	ft6, fs10
	fsw	fs10, 968(sp)                   # 4-byte Folded Spill
	fmv.s	ft7, fs10
	fsw	fs10, 992(sp)                   # 4-byte Folded Spill
	fmv.s	fs1, fs10
	fsw	fs10, 1008(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1016(sp)                  # 4-byte Folded Spill
	fmv.s	fs0, fs10
	fsw	fs10, 1024(sp)                  # 4-byte Folded Spill
	fmv.s	ft8, fs10
	fsw	fs10, 1032(sp)                  # 4-byte Folded Spill
	fmv.s	fa7, fs10
	fsw	fs10, 1040(sp)                  # 4-byte Folded Spill
	fmv.s	ft11, fs10
	fsw	fs10, 1048(sp)                  # 4-byte Folded Spill
	fmv.s	fa2, fs10
	fsw	fs10, 1056(sp)                  # 4-byte Folded Spill
	fmv.s	ft9, fs10
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
	fsw	fs10, 336(sp)                   # 4-byte Folded Spill
	fsw	fs10, 344(sp)                   # 4-byte Folded Spill
	fsw	fs10, 352(sp)                   # 4-byte Folded Spill
	fsw	fs10, 360(sp)                   # 4-byte Folded Spill
	fsw	fs10, 368(sp)                   # 4-byte Folded Spill
	fsw	fs10, 376(sp)                   # 4-byte Folded Spill
	fsw	fs10, 384(sp)                   # 4-byte Folded Spill
	fsw	fs10, 392(sp)                   # 4-byte Folded Spill
	fsw	fs10, 400(sp)                   # 4-byte Folded Spill
	fmv.s	fa3, fs10
	fsw	fs10, 1096(sp)                  # 4-byte Folded Spill
	fmv.s	fs2, fs10
	fmv.s	fa4, fs10
	fmv.s	fs5, fs10
	fmv.s	fs3, fs10
	fmv.s	fs6, fs10
	fmv.s	fs7, fs10
	fmv.s	fs11, fs10
	fmv.s	fs9, fs10
	fmv.s	fs4, fs10
	fsw	fs10, 1104(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1112(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1120(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1128(sp)                  # 4-byte Folded Spill
	fmv.s	ft10, fs10
	fsw	fs10, 1136(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1080(sp)                  # 4-byte Folded Spill
	fsw	fs10, 1144(sp)                  # 4-byte Folded Spill
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
	fsw	fs10, 808(sp)                   # 4-byte Folded Spill
	fsw	fs10, 824(sp)                   # 4-byte Folded Spill
	fsw	fs10, 840(sp)                   # 4-byte Folded Spill
	fsw	fs10, 848(sp)                   # 4-byte Folded Spill
	fsw	fs10, 856(sp)                   # 4-byte Folded Spill
	fsw	fs10, 872(sp)                   # 4-byte Folded Spill
	fsw	fs10, 880(sp)                   # 4-byte Folded Spill
	fsw	fs10, 896(sp)                   # 4-byte Folded Spill
	fsw	fs10, 904(sp)                   # 4-byte Folded Spill
	fsw	fs10, 920(sp)                   # 4-byte Folded Spill
	fsw	fs10, 928(sp)                   # 4-byte Folded Spill
	fsw	fs10, 944(sp)                   # 4-byte Folded Spill
	fsw	fs10, 952(sp)                   # 4-byte Folded Spill
	fsw	fs10, 960(sp)                   # 4-byte Folded Spill
	fsw	fs10, 976(sp)                   # 4-byte Folded Spill
	fsw	fs10, 984(sp)                   # 4-byte Folded Spill
	fsw	fs10, 1000(sp)                  # 4-byte Folded Spill
	fmv.s	fa6, fs10
	andi	a2, a7, 2
	bnez	a2, .LBB0_3
	j	.LBB0_4
.LBB0_2:
	.loc	1 0 52 is_stmt 0                # k135114294089648.py:0:52
	fmv.w.x	fs8, zero
	fmv.s	fs10, fs8
	fsw	fs8, 800(sp)                    # 4-byte Folded Spill
	fmv.s	ft0, fs8
	fsw	fs8, 816(sp)                    # 4-byte Folded Spill
	fmv.s	ft1, fs8
	fsw	fs8, 832(sp)                    # 4-byte Folded Spill
	fmv.s	ft2, fs8
	fsw	fs8, 864(sp)                    # 4-byte Folded Spill
	fmv.s	ft3, fs8
	fsw	fs8, 888(sp)                    # 4-byte Folded Spill
	fmv.s	ft4, fs8
	fsw	fs8, 912(sp)                    # 4-byte Folded Spill
	fmv.s	ft5, fs8
	fsw	fs8, 936(sp)                    # 4-byte Folded Spill
	fmv.s	ft6, fs8
	fsw	fs8, 968(sp)                    # 4-byte Folded Spill
	fmv.s	ft7, fs8
	fsw	fs8, 992(sp)                    # 4-byte Folded Spill
	fmv.s	fs1, fs8
	fsw	fs8, 1008(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1016(sp)                   # 4-byte Folded Spill
	fmv.s	fs0, fs8
	fsw	fs8, 1024(sp)                   # 4-byte Folded Spill
	fmv.s	ft8, fs8
	fsw	fs8, 1032(sp)                   # 4-byte Folded Spill
	fmv.s	fa7, fs8
	fsw	fs8, 1040(sp)                   # 4-byte Folded Spill
	fmv.s	ft11, fs8
	fsw	fs8, 1048(sp)                   # 4-byte Folded Spill
	fmv.s	fa2, fs8
	fsw	fs8, 1056(sp)                   # 4-byte Folded Spill
	fmv.s	ft9, fs8
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
	fsw	fs8, 336(sp)                    # 4-byte Folded Spill
	fsw	fs8, 344(sp)                    # 4-byte Folded Spill
	fsw	fs8, 352(sp)                    # 4-byte Folded Spill
	fsw	fs8, 360(sp)                    # 4-byte Folded Spill
	fsw	fs8, 368(sp)                    # 4-byte Folded Spill
	fsw	fs8, 376(sp)                    # 4-byte Folded Spill
	fsw	fs8, 384(sp)                    # 4-byte Folded Spill
	fsw	fs8, 392(sp)                    # 4-byte Folded Spill
	fsw	fs8, 400(sp)                    # 4-byte Folded Spill
	fmv.s	fa3, fs8
	fsw	fs8, 1096(sp)                   # 4-byte Folded Spill
	fmv.s	fs2, fs8
	fmv.s	fa4, fs8
	fmv.s	fs5, fs8
	fmv.s	fs3, fs8
	fmv.s	fs6, fs8
	fmv.s	fs7, fs8
	fmv.s	fs11, fs8
	fmv.s	fs9, fs8
	fmv.s	fs4, fs8
	fsw	fs8, 1104(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1112(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1120(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1128(sp)                   # 4-byte Folded Spill
	fmv.s	ft10, fs8
	fsw	fs8, 1136(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1080(sp)                   # 4-byte Folded Spill
	fsw	fs8, 1144(sp)                   # 4-byte Folded Spill
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
	fsw	fs8, 808(sp)                    # 4-byte Folded Spill
	fsw	fs8, 824(sp)                    # 4-byte Folded Spill
	fsw	fs8, 840(sp)                    # 4-byte Folded Spill
	fsw	fs8, 848(sp)                    # 4-byte Folded Spill
	fsw	fs8, 856(sp)                    # 4-byte Folded Spill
	fsw	fs8, 872(sp)                    # 4-byte Folded Spill
	fsw	fs8, 880(sp)                    # 4-byte Folded Spill
	fsw	fs8, 896(sp)                    # 4-byte Folded Spill
	fsw	fs8, 904(sp)                    # 4-byte Folded Spill
	fsw	fs8, 920(sp)                    # 4-byte Folded Spill
	fsw	fs8, 928(sp)                    # 4-byte Folded Spill
	fsw	fs8, 944(sp)                    # 4-byte Folded Spill
	fsw	fs8, 952(sp)                    # 4-byte Folded Spill
	fsw	fs8, 960(sp)                    # 4-byte Folded Spill
	fsw	fs8, 976(sp)                    # 4-byte Folded Spill
	fsw	fs8, 984(sp)                    # 4-byte Folded Spill
	fsw	fs8, 1000(sp)                   # 4-byte Folded Spill
	fmv.s	fa6, fs8
	.loc	1 13 52                         # k135114294089648.py:13:52
	andi	a2, a7, 2
	beqz	a2, .LBB0_4
.LBB0_3:                                # %cond.load1
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v16, 1
	vmv.x.s	a2, v8
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	fs10, a2
.LBB0_4:                                # %else2
	andi	a2, a7, 4
	bnez	a2, .LBB0_32
# %bb.5:                                # %else5
	andi	a2, a7, 8
	beqz	a2, .LBB0_7
.LBB0_6:                                # %cond.load7
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 3
	vmv.x.s	a2, v8
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.w.x	ft0, a2
.LBB0_7:                                # %else8
	.loc	1 0 52                          # k135114294089648.py:0:52
	lwu	a2, 0(a0)
	lw	a0, 4(a0)
	.loc	1 13 52                         # k135114294089648.py:13:52
	andi	t1, a7, 16
	lui	t0, 6
	addi	t0, t0, 2024
	add	t0, sp, t0
	fsw	ft0, 200(sp)                    # 4-byte Folded Spill
	bnez	t1, .LBB0_33
# %bb.8:                                # %else11
	andi	t1, a7, 32
	fmv.s	ft0, fa1
	bnez	t1, .LBB0_34
.LBB0_9:                                # %else14
	.loc	1 0 52                          # k135114294089648.py:0:52
	fmv.s	fa1, ft9
	.loc	1 13 52                         # k135114294089648.py:13:52
	andi	t1, a7, 64
	fmv.s	ft9, fa7
	fsw	ft1, 208(sp)                    # 4-byte Folded Spill
	bnez	t1, .LBB0_35
.LBB0_10:                               # %else17
	andi	t1, a7, 128
	fmv.s	ft1, fs5
	bnez	t1, .LBB0_36
.LBB0_11:                               # %else20
	andi	t1, a7, 256
	fsw	ft2, 216(sp)                    # 4-byte Folded Spill
	bnez	t1, .LBB0_37
.LBB0_12:                               # %else23
	andi	t1, a7, 512
	fmv.s	ft2, fs3
	bnez	t1, .LBB0_38
.LBB0_13:                               # %else26
	andi	t1, a7, 1024
	fsw	ft3, 224(sp)                    # 4-byte Folded Spill
	bnez	t1, .LBB0_39
.LBB0_14:                               # %else29
	slli	t1, a7, 52
	fmv.s	ft3, fs6
	bltz	t1, .LBB0_40
.LBB0_15:                               # %else32
	slli	t1, a7, 51
	fsw	ft4, 232(sp)                    # 4-byte Folded Spill
	bltz	t1, .LBB0_41
.LBB0_16:                               # %else35
	slli	t1, a7, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v24, 16
	fmv.s	ft4, fs7
	bltz	t1, .LBB0_42
.LBB0_17:                               # %else38
	slli	t1, a7, 49
	lui	t0, 6
	addi	t0, t0, -88
	add	t0, sp, t0
	fsw	ft5, 240(sp)                    # 4-byte Folded Spill
	bltz	t1, .LBB0_43
.LBB0_18:                               # %else41
	slli	t1, a7, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	fmv.s	ft5, fs9
	bltz	t1, .LBB0_44
.LBB0_19:                               # %else44
	slli	t1, a7, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v24, a1
	fsw	ft6, 248(sp)                    # 4-byte Folded Spill
	bltz	t1, .LBB0_45
.LBB0_20:                               # %else47
	slli	t1, a7, 46
	fmv.s	ft6, fs11
	bltz	t1, .LBB0_46
.LBB0_21:                               # %else50
	slli	t1, a7, 45
	fsw	ft7, 256(sp)                    # 4-byte Folded Spill
	bltz	t1, .LBB0_47
.LBB0_22:                               # %else53
	slli	t1, a7, 44
	fmv.s	ft7, fs4
	bgez	t1, .LBB0_24
.LBB0_23:                               # %cond.load55
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 3
	vmv.x.s	t1, v8
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fs1, t1
.LBB0_24:                               # %else56
	slli	t1, a7, 43
	csrr	t3, vlenb
	li	t4, 169
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v8, (t3)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v0, v8
	fsw	fs1, 264(sp)                    # 4-byte Folded Spill
	bgez	t1, .LBB0_26
# %bb.25:                               # %cond.load58
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 1536
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1656(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fs1, t1
	fsw	fs1, 1008(sp)                   # 4-byte Folded Spill
.LBB0_26:                               # %else59
	slli	t1, a7, 42
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v24, v8, 6
	csrr	t3, vlenb
	li	t4, 161
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v24, (t3)                       # vscale x 64-byte Folded Spill
	csrr	t3, vlenb
	li	t4, 153
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v8, (t3)                        # vscale x 64-byte Folded Spill
	fmv.s	fs1, ft10
	fsw	fa6, 188(sp)                    # 4-byte Folded Spill
	bgez	t1, .LBB0_28
# %bb.27:                               # %cond.load61
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 1408
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1536(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa6, t1
	fsw	fa6, 1016(sp)                   # 4-byte Folded Spill
.LBB0_28:                               # %else62
	.loc	1 0 52                          # k135114294089648.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vmv8r.v	v24, v16
	csrr	t1, vlenb
	li	t3, 161
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v8, (t1)                        # vscale x 64-byte Folded Reload
	vmulh.vx	v8, v8, a6
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 41
	vmulh.vx	v16, v0, a6
	csrr	t3, vlenb
	li	t4, 145
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v16, (t3)                       # vscale x 64-byte Folded Spill
	flw	fa6, 1104(sp)                   # 4-byte Folded Reload
	vmv8r.v	v16, v0
	bgez	t1, .LBB0_30
# %bb.29:                               # %cond.load64
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 1280
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t1)
	ld	t1, 1416(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fs0, t1
.LBB0_30:                               # %else65
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	t1, vlenb
	li	t3, 161
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v0, (t1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v0
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 40
	vmv8r.v	v0, v16
	csrr	t3, vlenb
	li	t4, 145
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	vadd.vv	v16, v16, v0
	csrr	t3, vlenb
	li	t4, 145
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v16, (t3)                       # vscale x 64-byte Folded Spill
	fsw	fs0, 280(sp)                    # 4-byte Folded Spill
	bgez	t1, .LBB0_48
# %bb.31:                               # %cond.load67
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 1152
	add	t1, sp, t1
	vmv8r.v	v16, v24
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t1)
	ld	t1, 1296(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa7, t1
	fsw	fa7, 1024(sp)                   # 4-byte Folded Spill
	j	.LBB0_49
.LBB0_32:                               # %cond.load4
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 2
	vmv.x.s	a2, v8
	lhu	a2, 0(a2)
	slli	a2, a2, 16
	fmv.s	fa0, ft8
	fmv.s	ft8, ft11
	fmv.s	ft11, fa2
	fmv.w.x	fa2, a2
	fsw	fa2, 800(sp)                    # 4-byte Folded Spill
	fmv.s	fa2, ft11
	fmv.s	ft11, ft8
	fmv.s	ft8, fa0
	andi	a2, a7, 8
	bnez	a2, .LBB0_6
	j	.LBB0_7
.LBB0_33:                               # %cond.load10
	.loc	1 0 52                          # k135114294089648.py:0:52
	li	t1, 27
	slli	t1, t1, 10
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1080(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft0, t1
	fsw	ft0, 816(sp)                    # 4-byte Folded Spill
	andi	t1, a7, 32
	fmv.s	ft0, fa1
	beqz	t1, .LBB0_9
.LBB0_34:                               # %cond.load13
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 7
	addi	t1, t1, -1152
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 960(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft1, t1
	fmv.s	fa1, ft9
	andi	t1, a7, 64
	fmv.s	ft9, fa7
	fsw	ft1, 208(sp)                    # 4-byte Folded Spill
	beqz	t1, .LBB0_10
.LBB0_35:                               # %cond.load16
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 7
	addi	t1, t1, -1280
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 840(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft1, t1
	fsw	ft1, 832(sp)                    # 4-byte Folded Spill
	andi	t1, a7, 128
	fmv.s	ft1, fs5
	beqz	t1, .LBB0_11
.LBB0_36:                               # %cond.load19
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 7
	addi	t1, t1, -1408
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 720(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft2, t1
	andi	t1, a7, 256
	fsw	ft2, 216(sp)                    # 4-byte Folded Spill
	beqz	t1, .LBB0_12
.LBB0_37:                               # %cond.load22
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 7
	addi	t1, t1, -1536
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 600(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft2, t1
	fsw	ft2, 864(sp)                    # 4-byte Folded Spill
	andi	t1, a7, 512
	fmv.s	ft2, fs3
	beqz	t1, .LBB0_13
.LBB0_38:                               # %cond.load25
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 7
	addi	t1, t1, -1664
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 480(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft3, t1
	andi	t1, a7, 1024
	fsw	ft3, 224(sp)                    # 4-byte Folded Spill
	beqz	t1, .LBB0_14
.LBB0_39:                               # %cond.load28
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 7
	addi	t1, t1, -1792
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 360(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft3, t1
	fsw	ft3, 888(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 52
	fmv.s	ft3, fs6
	bgez	t1, .LBB0_15
.LBB0_40:                               # %cond.load31
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 7
	addi	t1, t1, -1920
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 240(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft4, t1
	slli	t1, a7, 51
	fsw	ft4, 232(sp)                    # 4-byte Folded Spill
	bgez	t1, .LBB0_16
.LBB0_41:                               # %cond.load34
	.loc	1 0 52                          # k135114294089648.py:0:52
	li	t1, 13
	slli	t1, t1, 11
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 120(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft4, t1
	fsw	ft4, 912(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v24, 16
	fmv.s	ft4, fs7
	bgez	t1, .LBB0_17
.LBB0_42:                               # %cond.load37
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 1920
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t0, 0(t0)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	ft5, t0
	slli	t1, a7, 49
	lui	t0, 6
	addi	t0, t0, -88
	add	t0, sp, t0
	fsw	ft5, 240(sp)                    # 4-byte Folded Spill
	bgez	t1, .LBB0_18
.LBB0_43:                               # %cond.load40
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 1792
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1992(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft5, t1
	fsw	ft5, 936(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	fmv.s	ft5, fs9
	bgez	t1, .LBB0_19
.LBB0_44:                               # %cond.load43
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 1664
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vse64.v	v16, (t1)
	ld	t1, 1872(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft6, t1
	slli	t1, a7, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v24, a1
	fsw	ft6, 248(sp)                    # 4-byte Folded Spill
	bgez	t1, .LBB0_20
.LBB0_45:                               # %cond.load46
	vmv.x.s	t1, v16
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft6, t1
	fsw	ft6, 968(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 46
	fmv.s	ft6, fs11
	bgez	t1, .LBB0_21
.LBB0_46:                               # %cond.load49
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v16, 1
	vmv.x.s	t1, v8
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft7, t1
	slli	t1, a7, 45
	fsw	ft7, 256(sp)                    # 4-byte Folded Spill
	bgez	t1, .LBB0_22
.LBB0_47:                               # %cond.load52
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 2
	vmv.x.s	t1, v8
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft7, t1
	fsw	ft7, 992(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 44
	fmv.s	ft7, fs4
	bltz	t1, .LBB0_23
	j	.LBB0_24
.LBB0_48:
	.loc	1 0 52                          # k135114294089648.py:0:52
	vmv8r.v	v16, v24
.LBB0_49:                               # %else68
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v8, v8, 3
	.loc	1 13 52 is_stmt 1               # k135114294089648.py:13:52
	slli	t1, a7, 39
	csrr	t3, vlenb
	li	t4, 145
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v24, (t3)                       # vscale x 64-byte Folded Reload
	vsra.vi	v24, v24, 9
	csrr	t3, vlenb
	li	t4, 137
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v0, (t3)                        # vscale x 64-byte Folded Spill
	csrr	t3, vlenb
	li	t4, 177
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v16, (t3)                       # vscale x 64-byte Folded Spill
	flw	fa7, 1120(sp)                   # 4-byte Folded Reload
	bgez	t1, .LBB0_51
# %bb.50:                               # %cond.load70
	.loc	1 0 52 is_stmt 0                # k135114294089648.py:0:52
	li	t1, 25
	slli	t1, t1, 10
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1176(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft8, t1
.LBB0_51:                               # %else71
	.loc	1 0 52                          # k135114294089648.py:0:52
	flw	fa0, 1088(sp)                   # 4-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vsrl.vi	v0, v8, 31
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 38
	vsrl.vi	v16, v24, 31
	fsw	ft8, 288(sp)                    # 4-byte Folded Spill
	bgez	t1, .LBB0_53
# %bb.52:                               # %cond.load73
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 896
	add	t1, sp, t1
	csrr	t3, vlenb
	li	t4, 145
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v0, (t3)                        # vscale x 64-byte Folded Spill
	csrr	t3, vlenb
	li	t4, 177
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v0, (t3)                        # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t1)
	csrr	t1, vlenb
	li	t3, 145
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v0, (t1)                        # vscale x 64-byte Folded Reload
	ld	t1, 1056(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft8, t1
	fsw	ft8, 1032(sp)                   # 4-byte Folded Spill
.LBB0_53:                               # %else74
	.loc	1 0 52                          # k135114294089648.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v0
	csrr	t1, vlenb
	li	t3, 145
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vs8r.v	v8, (t1)                        # vscale x 64-byte Folded Spill
	vadd.vv	v16, v24, v16
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 37
	csrr	t3, vlenb
	li	t4, 153
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v8, (t3)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a5
	csrr	t3, vlenb
	li	t4, 73
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v16, (t3)                       # vscale x 64-byte Folded Spill
	flw	ft8, 1128(sp)                   # 4-byte Folded Reload
	bgez	t1, .LBB0_55
# %bb.54:                               # %cond.load76
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 768
	add	t1, sp, t1
	csrr	t3, vlenb
	li	t4, 177
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 936(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft9, t1
.LBB0_55:                               # %else77
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	t1, vlenb
	li	t3, 137
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v0, (t1)                        # vscale x 64-byte Folded Reload
	csrr	t1, vlenb
	li	t3, 145
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	csrr	t1, vlenb
	li	t3, 161
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v24, (t1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vnmsub.vx	v16, a4, v24
	csrr	t1, vlenb
	li	t3, 145
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	vsub.vv	v8, v0, v8
	csrr	t1, vlenb
	li	t3, 113
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vs8r.v	v8, (t1)                        # vscale x 64-byte Folded Spill
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 36
	csrr	t3, vlenb
	li	t4, 73
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v8, (t3)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 6
	fsw	ft9, 296(sp)                    # 4-byte Folded Spill
	bgez	t1, .LBB0_57
# %bb.56:                               # %cond.load79
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 640
	add	t1, sp, t1
	csrr	t3, vlenb
	li	t4, 177
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 816(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft9, t1
	fsw	ft9, 1040(sp)                   # 4-byte Folded Spill
.LBB0_57:                               # %else80
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	t1, vlenb
	li	t3, 113
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 35
	csrr	t3, vlenb
	li	t4, 145
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 7
	flw	ft9, 1112(sp)                   # 4-byte Folded Reload
	bgez	t1, .LBB0_59
# %bb.58:                               # %cond.load82
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 512
	add	t1, sp, t1
	csrr	t3, vlenb
	li	t4, 177
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v24, (t3)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (t1)
	ld	t1, 696(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft11, t1
.LBB0_59:                               # %else83
	slli	t1, a7, 34
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v24, v8, v16
	fsw	ft11, 304(sp)                   # 4-byte Folded Spill
	bgez	t1, .LBB0_61
# %bb.60:                               # %cond.load85
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 384
	add	t1, sp, t1
	csrr	t3, vlenb
	li	t4, 177
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 576(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft10, t1
	fsw	ft10, 1048(sp)                  # 4-byte Folded Spill
	flw	ft10, 1136(sp)                  # 4-byte Folded Reload
	slli	t1, a7, 33
	bltz	t1, .LBB0_62
	j	.LBB0_63
.LBB0_61:
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	t1, vlenb
	li	t3, 177
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	flw	ft10, 1136(sp)                  # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 33
	bgez	t1, .LBB0_63
.LBB0_62:                               # %cond.load88
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 256
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 456(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
.LBB0_63:                               # %else89
	slli	t1, a7, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v24, v24
	fsw	fa2, 312(sp)                    # 4-byte Folded Spill
	bltz	t1, .LBB0_101
# %bb.64:                               # %else92
	.loc	1 0 52                          # k135114294089648.py:0:52
	flw	ft11, 1144(sp)                  # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v8, a1
	bltz	t1, .LBB0_102
.LBB0_65:                               # %else95
	slli	t1, a7, 30
	bltz	t1, .LBB0_103
.LBB0_66:                               # %else98
	slli	t1, a7, 29
	bltz	t1, .LBB0_104
.LBB0_67:                               # %else101
	slli	t1, a7, 28
	bltz	t1, .LBB0_105
.LBB0_68:                               # %else104
	slli	t1, a7, 27
	bltz	t1, .LBB0_106
.LBB0_69:                               # %else107
	slli	t1, a7, 26
	bltz	t1, .LBB0_107
.LBB0_70:                               # %else110
	slli	t1, a7, 25
	lui	t0, 5
	addi	t0, t0, 1872
	add	t0, sp, t0
	bltz	t1, .LBB0_108
.LBB0_71:                               # %else113
	slli	t1, a7, 24
	bltz	t1, .LBB0_109
.LBB0_72:                               # %else116
	slli	t1, a7, 23
	bltz	t1, .LBB0_110
.LBB0_73:                               # %else119
	slli	t1, a7, 22
	bltz	t1, .LBB0_111
.LBB0_74:                               # %else122
	slli	t1, a7, 21
	bltz	t1, .LBB0_112
.LBB0_75:                               # %else125
	slli	t1, a7, 20
	bltz	t1, .LBB0_113
.LBB0_76:                               # %else128
	slli	t1, a7, 19
	bltz	t1, .LBB0_114
.LBB0_77:                               # %else131
	slli	t1, a7, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v24, 16
	bltz	t1, .LBB0_115
.LBB0_78:                               # %else134
	slli	t1, a7, 17
	bltz	t1, .LBB0_116
.LBB0_79:                               # %else137
	slli	t1, a7, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	bltz	t1, .LBB0_117
.LBB0_80:                               # %else140
	slli	t1, a7, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v24, a1
	bltz	t1, .LBB0_118
.LBB0_81:                               # %else143
	slli	t1, a7, 14
	bltz	t1, .LBB0_119
.LBB0_82:                               # %else146
	slli	t1, a7, 13
	bltz	t1, .LBB0_120
.LBB0_83:                               # %else149
	slli	t1, a7, 12
	bgez	t1, .LBB0_85
.LBB0_84:                               # %cond.load151
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	t1, v8
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 584(sp)                    # 4-byte Folded Spill
.LBB0_85:                               # %else152
	slli	t1, a7, 11
	csrr	t3, vlenb
	li	t4, 185
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v8, (t3)                        # vscale x 64-byte Folded Reload
	csrr	t3, vlenb
	li	t4, 169
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	t1, .LBB0_87
# %bb.86:                               # %cond.load154
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -1536
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t1)
	ld	t1, 720(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 592(sp)                    # 4-byte Folded Spill
.LBB0_87:                               # %else155
	slli	t1, a7, 10
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v16, v8, 6
	csrr	t3, vlenb
	li	t4, 177
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v16, (t3)                       # vscale x 64-byte Folded Spill
	csrr	t3, vlenb
	li	t4, 121
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v8, (t3)                        # vscale x 64-byte Folded Spill
	bgez	t1, .LBB0_89
# %bb.88:                               # %cond.load157
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -1664
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t1)
	ld	t1, 600(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 600(sp)                    # 4-byte Folded Spill
.LBB0_89:                               # %else158
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	t1, vlenb
	li	t3, 177
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v8, (t1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vmulh.vx	v8, v8, a6
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 9
	csrr	t3, vlenb
	li	t4, 185
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	vmulh.vx	v24, v16, a6
	bgez	t1, .LBB0_91
# %bb.90:                               # %cond.load160
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -1792
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t1)
	ld	t1, 480(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 608(sp)                    # 4-byte Folded Spill
.LBB0_91:                               # %else161
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	t1, vlenb
	li	t3, 177
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 8
	csrr	t3, vlenb
	li	t4, 185
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	vadd.vv	v24, v24, v16
	bgez	t1, .LBB0_93
# %bb.92:                               # %cond.load163
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -1920
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t1)
	ld	t1, 360(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 616(sp)                    # 4-byte Folded Spill
.LBB0_93:                               # %else164
	.loc	1 0 52                          # k135114294089648.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v8, v8, 3
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 7
	vsra.vi	v24, v24, 9
	csrr	t3, vlenb
	li	t4, 161
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v0, (t3)                        # vscale x 64-byte Folded Spill
	bgez	t1, .LBB0_95
# %bb.94:                               # %cond.load166
	.loc	1 0 52                          # k135114294089648.py:0:52
	li	t1, 11
	slli	t1, t1, 11
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t1)
	ld	t1, 240(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 624(sp)                    # 4-byte Folded Spill
.LBB0_95:                               # %else167
	.loc	1 0 52                          # k135114294089648.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vsrl.vi	v0, v8, 31
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 6
	vsrl.vi	v16, v24, 31
	bgez	t1, .LBB0_97
# %bb.96:                               # %cond.load169
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 5
	addi	t1, t1, 1920
	add	t1, sp, t1
	csrr	t3, vlenb
	li	t4, 153
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v0, (t3)                        # vscale x 64-byte Folded Spill
	csrr	t3, vlenb
	li	t4, 161
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v0, (t3)                        # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t1)
	csrr	t1, vlenb
	li	t3, 153
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v0, (t1)                        # vscale x 64-byte Folded Reload
	ld	t1, 120(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 632(sp)                    # 4-byte Folded Spill
.LBB0_97:                               # %else170
	.loc	1 0 52                          # k135114294089648.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v0
	csrr	t1, vlenb
	li	t3, 153
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vs8r.v	v8, (t1)                        # vscale x 64-byte Folded Spill
	vadd.vv	v24, v24, v16
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, a7, 5
	csrr	t3, vlenb
	li	t4, 121
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v8, (t3)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a5
	bgez	t1, .LBB0_99
# %bb.98:                               # %cond.load172
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 5
	addi	t1, t1, 1792
	add	t1, sp, t1
	csrr	t3, vlenb
	li	t4, 161
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t0, 0(t0)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 640(sp)                    # 4-byte Folded Spill
.LBB0_99:                               # %else173
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	t0, vlenb
	li	t1, 153
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 7
	addi	t1, t1, -832
	add	t0, t0, t1
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	csrr	t0, vlenb
	li	t1, 177
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 7
	addi	t1, t1, -832
	add	t0, t0, t1
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vnmsub.vx	v16, a4, v0
	csrr	t0, vlenb
	li	t1, 153
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 7
	addi	t1, t1, -832
	add	t0, t0, t1
	vs8r.v	v16, (t0)                       # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t1, 185
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 7
	addi	t1, t1, -832
	add	t0, t0, t1
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	vsub.vv	v8, v16, v8
	csrr	t0, vlenb
	li	t1, 121
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 7
	addi	t1, t1, -832
	add	t0, t0, t1
	vs8r.v	v8, (t0)                        # vscale x 64-byte Folded Spill
	vsll.vi	v8, v24, 6
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t0, a7, 4
	lui	t1, 5
	addi	t1, t1, -264
	add	t1, sp, t1
	bgez	t0, .LBB0_121
# %bb.100:                              # %cond.load175
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t0, 5
	addi	t0, t0, 1664
	add	t0, sp, t0
	csrr	t3, vlenb
	li	t4, 161
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v0, (t3)                        # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 2016(t1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 648(sp)                    # 4-byte Folded Spill
	j	.LBB0_122
.LBB0_101:                              # %cond.load91
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, 128
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vse64.v	v16, (t1)
	ld	t1, 336(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft11, t1
	fsw	ft11, 1056(sp)                  # 4-byte Folded Spill
	flw	ft11, 1144(sp)                  # 4-byte Folded Reload
	slli	t1, a7, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v8, a1
	bgez	t1, .LBB0_65
.LBB0_102:                              # %cond.load94
	vmv.x.s	t1, v16
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa1, t1
	slli	t1, a7, 30
	bgez	t1, .LBB0_66
.LBB0_103:                              # %cond.load97
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v16, 1
	vmv.x.s	t1, v8
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft0, t1
	slli	t1, a7, 29
	bgez	t1, .LBB0_67
.LBB0_104:                              # %cond.load100
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 2
	vmv.x.s	t1, v8
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 448(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 28
	bgez	t1, .LBB0_68
.LBB0_105:                              # %cond.load103
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 3
	vmv.x.s	t1, v8
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 456(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 27
	bgez	t1, .LBB0_69
.LBB0_106:                              # %cond.load106
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 120(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 464(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 26
	bgez	t1, .LBB0_70
.LBB0_107:                              # %cond.load109
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -128
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t0, 0(t0)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 472(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 25
	lui	t0, 5
	addi	t0, t0, 1872
	add	t0, sp, t0
	bgez	t1, .LBB0_71
.LBB0_108:                              # %cond.load112
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -256
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 2016(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 480(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 24
	bgez	t1, .LBB0_72
.LBB0_109:                              # %cond.load115
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -384
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1896(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 488(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 23
	bgez	t1, .LBB0_73
.LBB0_110:                              # %cond.load118
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -512
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1776(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 496(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 22
	bgez	t1, .LBB0_74
.LBB0_111:                              # %cond.load121
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -640
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1656(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 504(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 21
	bgez	t1, .LBB0_75
.LBB0_112:                              # %cond.load124
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -768
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1536(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 512(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 20
	bgez	t1, .LBB0_76
.LBB0_113:                              # %cond.load127
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -896
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1416(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 520(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 19
	bgez	t1, .LBB0_77
.LBB0_114:                              # %cond.load130
	.loc	1 0 52                          # k135114294089648.py:0:52
	li	t1, 23
	slli	t1, t1, 10
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1296(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 528(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v24, 16
	bgez	t1, .LBB0_78
.LBB0_115:                              # %cond.load133
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -1152
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1176(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 536(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 17
	bgez	t1, .LBB0_79
.LBB0_116:                              # %cond.load136
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -1280
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (t1)
	ld	t1, 1056(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 544(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	bgez	t1, .LBB0_80
.LBB0_117:                              # %cond.load139
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 6
	addi	t1, t1, -1408
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vse64.v	v16, (t1)
	ld	t1, 936(t0)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 552(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v24, a1
	bgez	t1, .LBB0_81
.LBB0_118:                              # %cond.load142
	vmv.x.s	t1, v0
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 560(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 14
	bgez	t1, .LBB0_82
.LBB0_119:                              # %cond.load145
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	t1, v8
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 568(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 13
	bgez	t1, .LBB0_83
.LBB0_120:                              # %cond.load148
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	t1, v8
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	fa2, t1
	fsw	fa2, 576(sp)                    # 4-byte Folded Spill
	slli	t1, a7, 12
	bltz	t1, .LBB0_84
	j	.LBB0_85
.LBB0_121:
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	t0, vlenb
	li	t3, 161
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 7
	addi	t3, t3, -832
	add	t0, t0, t3
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
.LBB0_122:                              # %else176
	csrr	t0, vlenb
	li	t3, 121
	mul	t0, t0, t3
	add	t0, sp, t0
	lui	t3, 7
	addi	t3, t3, -832
	add	t0, t0, t3
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 13 52 is_stmt 1               # k135114294089648.py:13:52
	slli	t0, a7, 3
	csrr	t3, vlenb
	li	t4, 153
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 7
	csrr	t3, vlenb
	slli	t3, t3, 6
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v24, (t3)                       # vscale x 64-byte Folded Spill
	bgez	t0, .LBB0_124
# %bb.123:                              # %cond.load178
	.loc	1 0 52 is_stmt 0                # k135114294089648.py:0:52
	lui	t0, 5
	addi	t0, t0, 1536
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1896(t1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 656(sp)                    # 4-byte Folded Spill
.LBB0_124:                              # %else179
	slli	t0, a7, 2
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v24, v8, v16
	bgez	t0, .LBB0_126
# %bb.125:                              # %cond.load181
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t0, 5
	addi	t0, t0, 1408
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1776(t1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 664(sp)                    # 4-byte Folded Spill
.LBB0_126:                              # %else182
	slli	t0, a7, 1
	csrr	t3, vlenb
	slli	t4, t3, 7
	add	t3, t4, t3
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl1r.v	v8, (t3)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	bltz	t0, .LBB0_201
# %bb.127:                              # %else185
	.loc	1 0 52                          # k135114294089648.py:0:52
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v24, v24
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	t0, v16
	bltz	a7, .LBB0_202
.LBB0_128:                              # %else188
	.loc	1 0 52                          # k135114294089648.py:0:52
	fmv.s	fa2, fs2
	.loc	1 13 52                         # k135114294089648.py:13:52
	andi	a7, t0, 1
	vadd.vx	v16, v8, a1
	bnez	a7, .LBB0_203
.LBB0_129:                              # %else191
	andi	a7, t0, 2
	fsw	fa0, 1088(sp)                   # 4-byte Folded Spill
	bnez	a7, .LBB0_204
.LBB0_130:                              # %else194
	andi	a7, t0, 4
	flw	fa0, 1096(sp)                   # 4-byte Folded Reload
	bnez	a7, .LBB0_205
.LBB0_131:                              # %else197
	andi	a7, t0, 8
	fsw	fa5, 320(sp)                    # 4-byte Folded Spill
	bnez	a7, .LBB0_206
.LBB0_132:                              # %else200
	andi	a7, t0, 16
	flw	fs0, 336(sp)                    # 4-byte Folded Reload
	bnez	a7, .LBB0_207
.LBB0_133:                              # %else203
	.loc	1 0 52                          # k135114294089648.py:0:52
	flw	fs11, 344(sp)                   # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	andi	a7, t0, 32
	flw	fs9, 352(sp)                    # 4-byte Folded Reload
	bnez	a7, .LBB0_208
.LBB0_134:                              # %else206
	.loc	1 0 52                          # k135114294089648.py:0:52
	flw	fs7, 360(sp)                    # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	andi	a7, t0, 64
	flw	fs6, 368(sp)                    # 4-byte Folded Reload
	bnez	a7, .LBB0_209
.LBB0_135:                              # %else209
	.loc	1 0 52                          # k135114294089648.py:0:52
	flw	fs3, 376(sp)                    # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	andi	a7, t0, 128
	flw	fs5, 384(sp)                    # 4-byte Folded Reload
	bnez	a7, .LBB0_210
.LBB0_136:                              # %else212
	.loc	1 0 52                          # k135114294089648.py:0:52
	flw	fs4, 392(sp)                    # 4-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	andi	a7, t0, 256
	flw	fs2, 400(sp)                    # 4-byte Folded Reload
	bnez	a7, .LBB0_211
.LBB0_137:                              # %else215
	andi	a7, t0, 512
	bnez	a7, .LBB0_212
.LBB0_138:                              # %else218
	andi	a7, t0, 1024
	bnez	a7, .LBB0_213
.LBB0_139:                              # %else221
	slli	a7, t0, 52
	bltz	a7, .LBB0_214
.LBB0_140:                              # %else224
	slli	a7, t0, 51
	bltz	a7, .LBB0_215
.LBB0_141:                              # %else227
	slli	a7, t0, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v24, 16
	bltz	a7, .LBB0_216
.LBB0_142:                              # %else230
	slli	a7, t0, 49
	bltz	a7, .LBB0_217
.LBB0_143:                              # %else233
	slli	a7, t0, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	bltz	a7, .LBB0_218
.LBB0_144:                              # %else236
	slli	a7, t0, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v24, a1
	bltz	a7, .LBB0_219
.LBB0_145:                              # %else239
	slli	a7, t0, 46
	bltz	a7, .LBB0_220
.LBB0_146:                              # %else242
	slli	a7, t0, 45
	bltz	a7, .LBB0_221
.LBB0_147:                              # %else245
	slli	a7, t0, 44
	bgez	a7, .LBB0_149
.LBB0_148:                              # %cond.load247
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 3
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft3, a7
.LBB0_149:                              # %else248
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	a7, vlenb
	li	t1, 201
	mul	a7, a7, t1
	add	a7, sp, a7
	lui	t1, 7
	addi	t1, t1, -832
	add	a7, a7, t1
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	csrr	a7, vlenb
	li	t1, 169
	mul	a7, a7, t1
	add	a7, sp, a7
	lui	t1, 7
	addi	t1, t1, -832
	add	a7, a7, t1
	vl8r.v	v16, (a7)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, t0, 43
	lui	a7, 4
	addi	a7, a7, 1600
	add	a7, sp, a7
	bgez	t1, .LBB0_151
# %bb.150:                              # %cond.load250
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 5
	addi	t1, t1, -512
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t1)
	ld	t1, 2016(a7)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft4, t1
.LBB0_151:                              # %else251
	slli	t1, t0, 42
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v16, v8, 6
	csrr	t3, vlenb
	li	t4, 169
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v16, (t3)                       # vscale x 64-byte Folded Spill
	csrr	t3, vlenb
	slli	t4, t3, 7
	add	t3, t4, t3
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vs8r.v	v8, (t3)                        # vscale x 64-byte Folded Spill
	bgez	t1, .LBB0_153
# %bb.152:                              # %cond.load253
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t1, 5
	addi	t1, t1, -640
	add	t1, sp, t1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t1)
	ld	t1, 1896(a7)
	lhu	t1, 0(t1)
	slli	t1, t1, 16
	fmv.w.x	ft6, t1
.LBB0_153:                              # %else254
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	t1, vlenb
	li	t3, 169
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v8, (t1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vmulh.vx	v8, v8, a6
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	t1, t0, 41
	csrr	t3, vlenb
	li	t4, 201
	mul	t3, t3, t4
	add	t3, sp, t3
	lui	t4, 7
	addi	t4, t4, -832
	add	t3, t3, t4
	vl8r.v	v16, (t3)                       # vscale x 64-byte Folded Reload
	vmulh.vx	v24, v16, a6
	bgez	t1, .LBB0_155
# %bb.154:                              # %cond.load256
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a6, 5
	addi	a6, a6, -768
	add	a6, sp, a6
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1776(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft5, a6
.LBB0_155:                              # %else257
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	a6, vlenb
	li	t1, 169
	mul	a6, a6, t1
	add	a6, sp, a6
	lui	t1, 7
	addi	t1, t1, -832
	add	a6, a6, t1
	vl8r.v	v16, (a6)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	a6, t0, 40
	csrr	t1, vlenb
	li	t3, 201
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vadd.vv	v24, v24, v16
	csrr	t1, vlenb
	li	t3, 177
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vs8r.v	v0, (t1)                        # vscale x 64-byte Folded Spill
	bltz	a6, .LBB0_222
# %bb.156:                              # %else260
	.loc	1 0 52                          # k135114294089648.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v0, v8, 3
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	a6, t0, 39
	vsra.vi	v8, v24, 9
	bltz	a6, .LBB0_223
.LBB0_157:                              # %else263
	.loc	1 0 52                          # k135114294089648.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vsrl.vi	v24, v0, 31
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	a6, t0, 38
	vsrl.vi	v16, v8, 31
	bgez	a6, .LBB0_159
.LBB0_158:                              # %cond.load265
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a6, 5
	addi	a6, a6, -1152
	add	a6, sp, a6
	csrr	t1, vlenb
	li	t3, 161
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vs8r.v	v24, (t1)                       # vscale x 64-byte Folded Spill
	csrr	t1, vlenb
	li	t3, 177
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v24, (t1)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v24, (a6)
	csrr	a6, vlenb
	li	t1, 161
	mul	a6, a6, t1
	add	a6, sp, a6
	lui	t1, 7
	addi	t1, t1, -832
	add	a6, a6, t1
	vl8r.v	v24, (a6)                       # vscale x 64-byte Folded Reload
	ld	a6, 1416(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft9, a6
.LBB0_159:                              # %else266
	.loc	1 0 52                          # k135114294089648.py:0:52
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v24, v0, v24
	csrr	a6, vlenb
	li	t1, 161
	mul	a6, a6, t1
	add	a6, sp, a6
	lui	t1, 7
	addi	t1, t1, -832
	add	a6, a6, t1
	vs8r.v	v24, (a6)                       # vscale x 64-byte Folded Spill
	vadd.vv	v8, v8, v16
	csrr	a6, vlenb
	li	t1, 97
	mul	a6, a6, t1
	add	a6, sp, a6
	lui	t1, 7
	addi	t1, t1, -832
	add	a6, a6, t1
	vs8r.v	v8, (a6)                        # vscale x 64-byte Folded Spill
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	a6, t0, 37
	csrr	t1, vlenb
	slli	t3, t1, 7
	add	t1, t3, t1
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v8, (t1)                        # vscale x 64-byte Folded Reload
	vand.vx	v8, v8, a5
	bgez	a6, .LBB0_161
# %bb.160:                              # %cond.load268
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a5, 5
	addi	a5, a5, -1280
	add	a5, sp, a5
	csrr	a6, vlenb
	li	t1, 177
	mul	a6, a6, t1
	add	a6, sp, a6
	lui	t1, 7
	addi	t1, t1, -832
	add	a6, a6, t1
	vl8r.v	v16, (a6)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a5)
	ld	a5, 1296(a7)
	lhu	a5, 0(a5)
	slli	a5, a5, 16
	fmv.w.x	fa7, a5
.LBB0_161:                              # %else269
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	a5, vlenb
	li	a6, 137
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 7
	addi	a6, a6, -832
	add	a5, a5, a6
	vl8r.v	v0, (a5)                        # vscale x 64-byte Folded Reload
	csrr	a5, vlenb
	li	a6, 161
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 7
	addi	a6, a6, -832
	add	a5, a5, a6
	vl8r.v	v16, (a5)                       # vscale x 64-byte Folded Reload
	csrr	a5, vlenb
	li	a6, 169
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 7
	addi	a6, a6, -832
	add	a5, a5, a6
	vl8r.v	v24, (a5)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vnmsub.vx	v16, a4, v24
	csrr	a4, vlenb
	li	a5, 161
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	csrr	a4, vlenb
	li	a5, 201
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	vsub.vv	v8, v16, v8
	csrr	a4, vlenb
	slli	a5, a4, 7
	add	a4, a5, a4
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v8, (a4)                        # vscale x 64-byte Folded Spill
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	a4, t0, 36
	csrr	a5, vlenb
	li	a6, 97
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 7
	addi	a6, a6, -832
	add	a5, a5, a6
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 6
	bgez	a4, .LBB0_163
# %bb.162:                              # %cond.load271
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a4, 5
	addi	a4, a4, -1408
	add	a4, sp, a4
	csrr	a5, vlenb
	li	a6, 177
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 7
	addi	a6, a6, -832
	add	a5, a5, a6
	vl8r.v	v16, (a5)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a4)
	ld	a4, 1176(a7)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	ft8, a4
.LBB0_163:                              # %else272
	.loc	1 0 52                          # k135114294089648.py:0:52
	csrr	a4, vlenb
	slli	a5, a4, 7
	add	a4, a5, a4
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	a4, t0, 35
	csrr	a5, vlenb
	li	a6, 161
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 7
	addi	a6, a6, -832
	add	a5, a5, a6
	vl8r.v	v16, (a5)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 7
	bltz	a4, .LBB0_224
# %bb.164:                              # %else275
	slli	a4, t0, 34
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bltz	a4, .LBB0_225
.LBB0_165:                              # %else278
	slli	a3, t0, 33
	bltz	a3, .LBB0_226
.LBB0_166:                              # %else281
	slli	a3, t0, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	bltz	a3, .LBB0_227
.LBB0_167:                              # %else284
	slli	a3, t0, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v24, a1
	bltz	a3, .LBB0_228
.LBB0_168:                              # %else287
	slli	a3, t0, 30
	bltz	a3, .LBB0_229
.LBB0_169:                              # %else290
	slli	a3, t0, 29
	bltz	a3, .LBB0_230
.LBB0_170:                              # %else293
	slli	a3, t0, 28
	bltz	a3, .LBB0_231
.LBB0_171:                              # %else296
	slli	a3, t0, 27
	bltz	a3, .LBB0_232
.LBB0_172:                              # %else299
	slli	a3, t0, 26
	bltz	a3, .LBB0_233
.LBB0_173:                              # %else302
	slli	a3, t0, 25
	bltz	a3, .LBB0_234
.LBB0_174:                              # %else305
	slli	a3, t0, 24
	bltz	a3, .LBB0_235
.LBB0_175:                              # %else308
	slli	a3, t0, 23
	bltz	a3, .LBB0_236
.LBB0_176:                              # %else311
	slli	a4, t0, 22
	lui	a3, 4
	addi	a3, a3, -536
	add	a3, sp, a3
	bltz	a4, .LBB0_237
.LBB0_177:                              # %else314
	slli	a4, t0, 21
	bltz	a4, .LBB0_238
.LBB0_178:                              # %else317
	slli	a4, t0, 20
	bltz	a4, .LBB0_239
.LBB0_179:                              # %else320
	slli	a4, t0, 19
	bltz	a4, .LBB0_240
.LBB0_180:                              # %else323
	slli	a4, t0, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	bltz	a4, .LBB0_241
.LBB0_181:                              # %else326
	slli	a4, t0, 17
	bltz	a4, .LBB0_242
.LBB0_182:                              # %else329
	slli	a4, t0, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	bltz	a4, .LBB0_243
.LBB0_183:                              # %else332
	slli	a4, t0, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, a1
	bltz	a4, .LBB0_244
.LBB0_184:                              # %else335
	slli	a1, t0, 14
	bltz	a1, .LBB0_245
.LBB0_185:                              # %else338
	slli	a1, t0, 13
	bltz	a1, .LBB0_246
.LBB0_186:                              # %else341
	slli	a1, t0, 12
	bltz	a1, .LBB0_247
.LBB0_187:                              # %else344
	slli	a1, t0, 11
	bltz	a1, .LBB0_248
.LBB0_188:                              # %else347
	slli	a1, t0, 10
	bltz	a1, .LBB0_249
.LBB0_189:                              # %else350
	slli	a1, t0, 9
	bltz	a1, .LBB0_250
.LBB0_190:                              # %else353
	slli	a1, t0, 8
	bltz	a1, .LBB0_251
.LBB0_191:                              # %else356
	slli	a1, t0, 7
	bltz	a1, .LBB0_252
.LBB0_192:                              # %else359
	slli	a1, t0, 6
	bltz	a1, .LBB0_253
.LBB0_193:                              # %else362
	slli	a1, t0, 5
	bltz	a1, .LBB0_254
.LBB0_194:                              # %else365
	slli	a1, t0, 4
	bltz	a1, .LBB0_255
.LBB0_195:                              # %else368
	slli	a1, t0, 3
	bltz	a1, .LBB0_256
.LBB0_196:                              # %else371
	slli	a1, t0, 2
	bltz	a1, .LBB0_257
.LBB0_197:                              # %else374
	.loc	1 0 52                          # k135114294089648.py:0:52
	slli	a0, a0, 32
	.loc	1 13 52                         # k135114294089648.py:13:52
	slli	a1, t0, 1
	lui	a3, 3
	addi	a3, a3, 1448
	add	s8, sp, a3
	bgez	a1, .LBB0_199
.LBB0_198:                              # %cond.load376
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a1, 4
	addi	a1, a1, -768
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 1992(s8)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 1000(sp)                   # 4-byte Folded Spill
.LBB0_199:                              # %else377
	.loc	1 0 0                           # k135114294089648.py:0
	or	a0, a0, a2
	sd	a0, 192(sp)                     # 8-byte Folded Spill
	sd	t2, 440(sp)                     # 8-byte Folded Spill
	fsw	fa1, 412(sp)                    # 4-byte Folded Spill
	fsw	ft11, 1144(sp)                  # 4-byte Folded Spill
	fsw	ft10, 1136(sp)                  # 4-byte Folded Spill
	fsw	fs1, 400(sp)                    # 4-byte Folded Spill
	fsw	ft8, 1128(sp)                   # 4-byte Folded Spill
	fsw	fa7, 1120(sp)                   # 4-byte Folded Spill
	fsw	ft9, 1112(sp)                   # 4-byte Folded Spill
	fsw	fa6, 1104(sp)                   # 4-byte Folded Spill
	fsw	ft7, 392(sp)                    # 4-byte Folded Spill
	fsw	ft5, 384(sp)                    # 4-byte Folded Spill
	fsw	ft6, 376(sp)                    # 4-byte Folded Spill
	fsw	ft4, 368(sp)                    # 4-byte Folded Spill
	fsw	ft3, 360(sp)                    # 4-byte Folded Spill
	fsw	ft2, 352(sp)                    # 4-byte Folded Spill
	fsw	ft1, 344(sp)                    # 4-byte Folded Spill
	fsw	fa4, 336(sp)                    # 4-byte Folded Spill
	fsw	fa2, 328(sp)                    # 4-byte Folded Spill
	fsw	fa0, 1096(sp)                   # 4-byte Folded Spill
	fsw	fa3, 272(sp)                    # 4-byte Folded Spill
	.loc	1 13 52                         # k135114294089648.py:13:52
	bgez	t0, .LBB0_258
# %bb.200:                              # %cond.load379
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a0, 4
	addi	a0, a0, -896
	add	a0, sp, a0
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1872(s8)
	lhu	a0, 0(a0)
	slli	a0, a0, 16
	fmv.w.x	fs1, a0
	j	.LBB0_259
.LBB0_201:                              # %cond.load184
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	t0, 5
	addi	t0, t0, 1280
	add	t0, sp, t0
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (t0)
	ld	t0, 1656(t1)
	lhu	t0, 0(t0)
	slli	t0, t0, 16
	fmv.w.x	fa2, t0
	fsw	fa2, 672(sp)                    # 4-byte Folded Spill
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v24, v24
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.x.s	t0, v16
	bgez	a7, .LBB0_128
.LBB0_202:                              # %cond.load187
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, 1152
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vse64.v	v0, (a7)
	ld	a7, 1536(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa2, a7
	fsw	fa2, 680(sp)                    # 4-byte Folded Spill
	fmv.s	fa2, fs2
	andi	a7, t0, 1
	vadd.vx	v16, v8, a1
	beqz	a7, .LBB0_129
.LBB0_203:                              # %cond.load190
	vmv.x.s	a7, v16
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa0, a7
	andi	a7, t0, 2
	fsw	fa0, 1088(sp)                   # 4-byte Folded Spill
	beqz	a7, .LBB0_130
.LBB0_204:                              # %cond.load193
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v16, 1
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa0, a7
	fsw	fa0, 1064(sp)                   # 4-byte Folded Spill
	andi	a7, t0, 4
	flw	fa0, 1096(sp)                   # 4-byte Folded Reload
	beqz	a7, .LBB0_131
.LBB0_205:                              # %cond.load196
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 2
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	andi	a7, t0, 8
	fsw	fa5, 320(sp)                    # 4-byte Folded Spill
	beqz	a7, .LBB0_132
.LBB0_206:                              # %cond.load199
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v16, 3
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa5, a7
	fsw	fa5, 1072(sp)                   # 4-byte Folded Spill
	andi	a7, t0, 16
	flw	fs0, 336(sp)                    # 4-byte Folded Reload
	beqz	a7, .LBB0_133
.LBB0_207:                              # %cond.load202
	.loc	1 0 52                          # k135114294089648.py:0:52
	li	a7, 21
	slli	a7, a7, 10
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 1320(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs0, a7
	flw	fs11, 344(sp)                   # 4-byte Folded Reload
	andi	a7, t0, 32
	flw	fs9, 352(sp)                    # 4-byte Folded Reload
	beqz	a7, .LBB0_134
.LBB0_208:                              # %cond.load205
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, 896
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 1200(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs11, a7
	flw	fs7, 360(sp)                    # 4-byte Folded Reload
	andi	a7, t0, 64
	flw	fs6, 368(sp)                    # 4-byte Folded Reload
	beqz	a7, .LBB0_135
.LBB0_209:                              # %cond.load208
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, 768
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 1080(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs9, a7
	flw	fs3, 376(sp)                    # 4-byte Folded Reload
	andi	a7, t0, 128
	flw	fs5, 384(sp)                    # 4-byte Folded Reload
	beqz	a7, .LBB0_136
.LBB0_210:                              # %cond.load211
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, 640
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 960(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs7, a7
	flw	fs4, 392(sp)                    # 4-byte Folded Reload
	andi	a7, t0, 256
	flw	fs2, 400(sp)                    # 4-byte Folded Reload
	beqz	a7, .LBB0_137
.LBB0_211:                              # %cond.load214
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, 512
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 840(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs6, a7
	andi	a7, t0, 512
	beqz	a7, .LBB0_138
.LBB0_212:                              # %cond.load217
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, 384
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 720(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs3, a7
	andi	a7, t0, 1024
	beqz	a7, .LBB0_139
.LBB0_213:                              # %cond.load220
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, 256
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 600(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs5, a7
	slli	a7, t0, 52
	bgez	a7, .LBB0_140
.LBB0_214:                              # %cond.load223
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, 128
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 480(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs4, a7
	slli	a7, t0, 51
	bgez	a7, .LBB0_141
.LBB0_215:                              # %cond.load226
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 360(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fs2, a7
	slli	a7, t0, 50
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v24, 16
	bgez	a7, .LBB0_142
.LBB0_216:                              # %cond.load229
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, -128
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 240(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa3, a7
	slli	a7, t0, 49
	bgez	a7, .LBB0_143
.LBB0_217:                              # %cond.load232
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, -256
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a7)
	ld	a7, 120(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa0, a7
	slli	a7, t0, 48
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	bgez	a7, .LBB0_144
.LBB0_218:                              # %cond.load235
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a7, 5
	addi	a7, a7, -384
	add	a7, sp, a7
	.loc	1 13 52                         # k135114294089648.py:13:52
	vse64.v	v16, (a7)
	ld	a7, 0(t1)
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa2, a7
	slli	a7, t0, 47
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v0, v24, a1
	bgez	a7, .LBB0_145
.LBB0_219:                              # %cond.load238
	vmv.x.s	a7, v0
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	fa4, a7
	slli	a7, t0, 46
	bgez	a7, .LBB0_146
.LBB0_220:                              # %cond.load241
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v0, 1
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft1, a7
	slli	a7, t0, 45
	bgez	a7, .LBB0_147
.LBB0_221:                              # %cond.load244
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v0, 2
	vmv.x.s	a7, v8
	lhu	a7, 0(a7)
	slli	a7, a7, 16
	fmv.w.x	ft2, a7
	slli	a7, t0, 44
	bltz	a7, .LBB0_148
	j	.LBB0_149
.LBB0_222:                              # %cond.load259
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a6, 5
	addi	a6, a6, -896
	add	a6, sp, a6
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a6)
	ld	a6, 1656(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	ft7, a6
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v0, v8, 3
	slli	a6, t0, 39
	vsra.vi	v8, v24, 9
	bgez	a6, .LBB0_157
.LBB0_223:                              # %cond.load262
	.loc	1 0 52                          # k135114294089648.py:0:52
	li	a6, 19
	slli	a6, a6, 10
	add	a6, sp, a6
	csrr	t1, vlenb
	li	t3, 177
	mul	t1, t1, t3
	add	t1, sp, t1
	lui	t3, 7
	addi	t3, t3, -832
	add	t1, t1, t3
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a6)
	ld	a6, 1536(a7)
	lhu	a6, 0(a6)
	slli	a6, a6, 16
	fmv.w.x	fa6, a6
	vsetvli	zero, a3, e32, m8, ta, ma
	vsrl.vi	v24, v0, 31
	slli	a6, t0, 38
	vsrl.vi	v16, v8, 31
	bltz	a6, .LBB0_158
	j	.LBB0_159
.LBB0_224:                              # %cond.load274
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a4, 5
	addi	a4, a4, -1536
	add	a4, sp, a4
	vmv8r.v	v24, v0
	csrr	a5, vlenb
	li	a6, 177
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 7
	addi	a6, a6, -832
	add	a5, a5, a6
	vl8r.v	v0, (a5)                        # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v0, (a4)
	vmv8r.v	v0, v24
	ld	a4, 1056(a7)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fs1, a4
	slli	a4, t0, 34
	vsetvli	zero, a3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a4, .LBB0_165
.LBB0_225:                              # %cond.load277
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a3, 5
	addi	a3, a3, -1664
	add	a3, sp, a3
	csrr	a4, vlenb
	li	a5, 177
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a3)
	ld	a3, 936(a7)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	ft10, a3
	slli	a3, t0, 33
	bgez	a3, .LBB0_166
.LBB0_226:                              # %cond.load280
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a3, 5
	addi	a3, a3, -1792
	add	a3, sp, a3
	csrr	a4, vlenb
	li	a5, 177
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a3)
	ld	a3, 816(a7)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 1080(sp)                   # 4-byte Folded Spill
	slli	a3, t0, 32
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	bgez	a3, .LBB0_167
.LBB0_227:                              # %cond.load283
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a3, 5
	addi	a3, a3, -1920
	add	a3, sp, a3
	csrr	a4, vlenb
	li	a5, 177
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 13 52                         # k135114294089648.py:13:52
	vse64.v	v16, (a3)
	ld	a3, 696(a7)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	ft11, a3
	slli	a3, t0, 31
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v24, a1
	bgez	a3, .LBB0_168
.LBB0_228:                              # %cond.load286
	vmv.x.s	a3, v16
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 688(sp)                    # 4-byte Folded Spill
	slli	a3, t0, 30
	bgez	a3, .LBB0_169
.LBB0_229:                              # %cond.load289
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v16, 1
	vmv.x.s	a3, v24
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 696(sp)                    # 4-byte Folded Spill
	slli	a3, t0, 29
	bgez	a3, .LBB0_170
.LBB0_230:                              # %cond.load292
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v16, 2
	vmv.x.s	a3, v24
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 704(sp)                    # 4-byte Folded Spill
	slli	a3, t0, 28
	bgez	a3, .LBB0_171
.LBB0_231:                              # %cond.load295
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v24, v16, 3
	vmv.x.s	a3, v24
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 712(sp)                    # 4-byte Folded Spill
	slli	a3, t0, 27
	bgez	a3, .LBB0_172
.LBB0_232:                              # %cond.load298
	.loc	1 0 52                          # k135114294089648.py:0:52
	li	a3, 9
	slli	a3, a3, 11
	add	a3, sp, a3
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a3)
	ld	a3, 480(a7)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 720(sp)                    # 4-byte Folded Spill
	slli	a3, t0, 26
	bgez	a3, .LBB0_173
.LBB0_233:                              # %cond.load301
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a3, 4
	addi	a3, a3, 1920
	add	a3, sp, a3
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a3)
	ld	a3, 360(a7)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 728(sp)                    # 4-byte Folded Spill
	slli	a3, t0, 25
	bgez	a3, .LBB0_174
.LBB0_234:                              # %cond.load304
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a3, 4
	addi	a3, a3, 1792
	add	a3, sp, a3
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a3)
	ld	a3, 240(a7)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 736(sp)                    # 4-byte Folded Spill
	slli	a3, t0, 24
	bgez	a3, .LBB0_175
.LBB0_235:                              # %cond.load307
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a3, 4
	addi	a3, a3, 1664
	add	a3, sp, a3
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a3)
	ld	a3, 120(a7)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 744(sp)                    # 4-byte Folded Spill
	slli	a3, t0, 23
	bgez	a3, .LBB0_176
.LBB0_236:                              # %cond.load310
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a3, 4
	addi	a3, a3, 1536
	add	a3, sp, a3
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a3)
	ld	a3, 0(a7)
	lhu	a3, 0(a3)
	slli	a3, a3, 16
	fmv.w.x	fa5, a3
	fsw	fa5, 752(sp)                    # 4-byte Folded Spill
	slli	a4, t0, 22
	lui	a3, 4
	addi	a3, a3, -536
	add	a3, sp, a3
	bgez	a4, .LBB0_177
.LBB0_237:                              # %cond.load313
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a4, 4
	addi	a4, a4, 1408
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a4)
	ld	a4, 2016(a3)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 760(sp)                    # 4-byte Folded Spill
	slli	a4, t0, 21
	bgez	a4, .LBB0_178
.LBB0_238:                              # %cond.load316
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a4, 4
	addi	a4, a4, 1280
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a4)
	ld	a4, 1896(a3)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 768(sp)                    # 4-byte Folded Spill
	slli	a4, t0, 20
	bgez	a4, .LBB0_179
.LBB0_239:                              # %cond.load319
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a4, 4
	addi	a4, a4, 1152
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a4)
	ld	a4, 1776(a3)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 776(sp)                    # 4-byte Folded Spill
	slli	a4, t0, 19
	bgez	a4, .LBB0_180
.LBB0_240:                              # %cond.load322
	.loc	1 0 52                          # k135114294089648.py:0:52
	li	a4, 17
	slli	a4, a4, 10
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a4)
	ld	a4, 1656(a3)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 784(sp)                    # 4-byte Folded Spill
	slli	a4, t0, 18
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	bgez	a4, .LBB0_181
.LBB0_241:                              # %cond.load325
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a4, 4
	addi	a4, a4, 896
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a4)
	ld	a4, 1536(a3)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 792(sp)                    # 4-byte Folded Spill
	slli	a4, t0, 17
	bgez	a4, .LBB0_182
.LBB0_242:                              # %cond.load328
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a4, 4
	addi	a4, a4, 768
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v16, (a4)
	ld	a4, 1416(a3)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 808(sp)                    # 4-byte Folded Spill
	slli	a4, t0, 16
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v8
	bgez	a4, .LBB0_183
.LBB0_243:                              # %cond.load331
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a4, 4
	addi	a4, a4, 640
	add	a4, sp, a4
	.loc	1 13 52                         # k135114294089648.py:13:52
	vse64.v	v16, (a4)
	ld	a4, 1296(a3)
	lhu	a4, 0(a4)
	slli	a4, a4, 16
	fmv.w.x	fa5, a4
	fsw	fa5, 824(sp)                    # 4-byte Folded Spill
	slli	a4, t0, 15
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, a1
	bgez	a4, .LBB0_184
.LBB0_244:                              # %cond.load334
	vmv.x.s	a1, v8
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 840(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 14
	bgez	a1, .LBB0_185
.LBB0_245:                              # %cond.load337
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v16, v8, 1
	vmv.x.s	a1, v16
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 848(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 13
	bgez	a1, .LBB0_186
.LBB0_246:                              # %cond.load340
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 2
	vmv.x.s	a1, v16
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 856(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 12
	bgez	a1, .LBB0_187
.LBB0_247:                              # %cond.load343
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v16, v8, 3
	vmv.x.s	a1, v16
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 872(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 11
	bgez	a1, .LBB0_188
.LBB0_248:                              # %cond.load346
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a1, 4
	addi	a1, a1, 512
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 1080(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 880(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 10
	bgez	a1, .LBB0_189
.LBB0_249:                              # %cond.load349
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a1, 4
	addi	a1, a1, 384
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 960(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 896(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 9
	bgez	a1, .LBB0_190
.LBB0_250:                              # %cond.load352
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a1, 4
	addi	a1, a1, 256
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 840(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 904(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 8
	bgez	a1, .LBB0_191
.LBB0_251:                              # %cond.load355
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a1, 4
	addi	a1, a1, 128
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 720(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 920(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 7
	bgez	a1, .LBB0_192
.LBB0_252:                              # %cond.load358
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a1, 4
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 600(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 928(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 6
	bgez	a1, .LBB0_193
.LBB0_253:                              # %cond.load361
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a1, 4
	addi	a1, a1, -128
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 480(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 944(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 5
	bgez	a1, .LBB0_194
.LBB0_254:                              # %cond.load364
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a1, 4
	addi	a1, a1, -256
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 360(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 952(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 4
	bgez	a1, .LBB0_195
.LBB0_255:                              # %cond.load367
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a1, 4
	addi	a1, a1, -384
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 240(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 960(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 3
	bgez	a1, .LBB0_196
.LBB0_256:                              # %cond.load370
	.loc	1 0 52                          # k135114294089648.py:0:52
	li	a1, 31
	slli	a1, a1, 9
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 120(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 976(sp)                    # 4-byte Folded Spill
	slli	a1, t0, 2
	bgez	a1, .LBB0_197
.LBB0_257:                              # %cond.load373
	.loc	1 0 52                          # k135114294089648.py:0:52
	lui	a1, 4
	addi	a1, a1, -640
	add	a1, sp, a1
	.loc	1 13 52                         # k135114294089648.py:13:52
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a1)
	ld	a1, 0(a3)
	lhu	a1, 0(a1)
	slli	a1, a1, 16
	fmv.w.x	fa5, a1
	fsw	fa5, 984(sp)                    # 4-byte Folded Spill
	slli	a0, a0, 32
	slli	a1, t0, 1
	lui	a3, 3
	addi	a3, a3, 1448
	add	s8, sp, a3
	bltz	a1, .LBB0_198
	j	.LBB0_199
.LBB0_258:
	.loc	1 0 52                          # k135114294089648.py:0:52
	flw	fs1, 188(sp)                    # 4-byte Folded Reload
.LBB0_259:                              # %else380
	li	a1, 32
	li	a0, 1792
	csrr	a2, vlenb
	li	a3, 201
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 7
	addi	a3, a3, -832
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 6 21 is_stmt 1                # k135114294089648.py:6:21
	vsetvli	zero, a1, e32, m8, ta, ma
	vmslt.vx	v11, v16, a0
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs1r.v	v11, (a1)                       # vscale x 8-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v9, v16, a0
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs1r.v	v9, (a1)                        # vscale x 8-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vmslt.vx	v8, v16, a0
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs1r.v	v8, (a1)                        # vscale x 8-byte Folded Spill
	vmslt.vx	v10, v0, a0
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs1r.v	v10, (a0)                       # vscale x 8-byte Folded Spill
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v8, v10, 4
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs1r.v	v8, (a0)                        # vscale x 8-byte Folded Spill
	vslideup.vi	v9, v11, 4
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs1r.v	v9, (a0)                        # vscale x 8-byte Folded Spill
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v8, v9, 8
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs1r.v	v8, (a0)                        # vscale x 8-byte Folded Spill
	.loc	1 13 62                         # k135114294089648.py:13:62
	fmv.s	fa0, ft0
	call	__truncsfbf2
	fsw	fa0, 188(sp)                    # 4-byte Folded Spill
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
	fsw	fa0, 176(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs10
	call	__truncsfbf2
	fsw	fa0, 168(sp)                    # 4-byte Folded Spill
	flw	fa0, 800(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 800(sp)                    # 4-byte Folded Spill
	flw	fa0, 200(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 200(sp)                    # 4-byte Folded Spill
	flw	fa0, 816(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 816(sp)                    # 4-byte Folded Spill
	flw	fa0, 208(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 208(sp)                    # 4-byte Folded Spill
	flw	fa0, 832(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 832(sp)                    # 4-byte Folded Spill
	flw	fa0, 216(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 216(sp)                    # 4-byte Folded Spill
	flw	fa0, 864(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 864(sp)                    # 4-byte Folded Spill
	flw	fa0, 224(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 224(sp)                    # 4-byte Folded Spill
	flw	fa0, 888(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 888(sp)                    # 4-byte Folded Spill
	flw	fa0, 232(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 232(sp)                    # 4-byte Folded Spill
	flw	fa0, 912(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 912(sp)                    # 4-byte Folded Spill
	flw	fa0, 240(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 240(sp)                    # 4-byte Folded Spill
	flw	fa0, 936(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 936(sp)                    # 4-byte Folded Spill
	flw	fa0, 248(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 248(sp)                    # 4-byte Folded Spill
	flw	fa0, 968(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 968(sp)                    # 4-byte Folded Spill
	flw	fa0, 256(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 256(sp)                    # 4-byte Folded Spill
	flw	fa0, 992(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 992(sp)                    # 4-byte Folded Spill
	flw	fa0, 264(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 264(sp)                    # 4-byte Folded Spill
	flw	fa0, 1008(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1008(sp)                   # 4-byte Folded Spill
	flw	fa0, 1016(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1016(sp)                   # 4-byte Folded Spill
	flw	fa0, 280(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 280(sp)                    # 4-byte Folded Spill
	flw	fa0, 1024(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1024(sp)                   # 4-byte Folded Spill
	flw	fa0, 288(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 288(sp)                    # 4-byte Folded Spill
	flw	fa0, 1032(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1032(sp)                   # 4-byte Folded Spill
	flw	fa0, 296(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 296(sp)                    # 4-byte Folded Spill
	flw	fa0, 1040(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1040(sp)                   # 4-byte Folded Spill
	flw	fa0, 304(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 304(sp)                    # 4-byte Folded Spill
	flw	fa0, 1048(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 1048(sp)                   # 4-byte Folded Spill
	flw	fa0, 312(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 312(sp)                    # 4-byte Folded Spill
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
	flw	fa0, 808(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 808(sp)                    # 4-byte Folded Spill
	flw	fa0, 824(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 824(sp)                    # 4-byte Folded Spill
	flw	fa0, 840(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 840(sp)                    # 4-byte Folded Spill
	flw	fa0, 848(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 848(sp)                    # 4-byte Folded Spill
	flw	fa0, 856(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 856(sp)                    # 4-byte Folded Spill
	flw	fa0, 872(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 872(sp)                    # 4-byte Folded Spill
	flw	fa0, 880(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 880(sp)                    # 4-byte Folded Spill
	flw	fa0, 896(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 896(sp)                    # 4-byte Folded Spill
	flw	fa0, 904(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 904(sp)                    # 4-byte Folded Spill
	flw	fa0, 920(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 920(sp)                    # 4-byte Folded Spill
	flw	fa0, 928(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 928(sp)                    # 4-byte Folded Spill
	flw	fa0, 944(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 944(sp)                    # 4-byte Folded Spill
	flw	fa0, 952(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 160(sp)                    # 4-byte Folded Spill
	flw	fa0, 960(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 152(sp)                    # 4-byte Folded Spill
	flw	fa0, 976(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 144(sp)                    # 4-byte Folded Spill
	flw	fa0, 984(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 136(sp)                    # 4-byte Folded Spill
	flw	fa0, 1000(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 128(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs1
	call	__truncsfbf2
	fsw	fa0, 96(sp)                     # 4-byte Folded Spill
	flw	fa0, 1088(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 88(sp)                     # 4-byte Folded Spill
	flw	fa0, 1064(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 104(sp)                    # 4-byte Folded Spill
	flw	fa0, 320(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 112(sp)                    # 4-byte Folded Spill
	flw	fa0, 1072(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 120(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs0
	call	__truncsfbf2
	fsw	fa0, 80(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs11
	call	__truncsfbf2
	fsw	fa0, 320(sp)                    # 4-byte Folded Spill
	fmv.s	fa0, fs9
	call	__truncsfbf2
	fsw	fa0, 72(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs7
	call	__truncsfbf2
	fsw	fa0, 64(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs6
	call	__truncsfbf2
	fsw	fa0, 60(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs3
	call	__truncsfbf2
	fsw	fa0, 56(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs5
	call	__truncsfbf2
	fsw	fa0, 52(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs4
	call	__truncsfbf2
	fsw	fa0, 48(sp)                     # 4-byte Folded Spill
	fmv.s	fa0, fs2
	call	__truncsfbf2
	fsw	fa0, 44(sp)                     # 4-byte Folded Spill
	flw	fa0, 272(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 272(sp)                    # 4-byte Folded Spill
	flw	fa0, 1096(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 40(sp)                     # 4-byte Folded Spill
	flw	fa0, 328(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 328(sp)                    # 4-byte Folded Spill
	flw	fa0, 336(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 952(sp)                    # 4-byte Folded Spill
	flw	fa0, 344(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 960(sp)                    # 4-byte Folded Spill
	flw	fa0, 352(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fsw	fa0, 976(sp)                    # 4-byte Folded Spill
	flw	fa0, 360(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs3, fa0
	flw	fa0, 368(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs8, fa0
	flw	fa0, 376(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs10, fa0
	flw	fa0, 384(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs2, fa0
	flw	fa0, 392(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs1, fa0
	flw	fa0, 1104(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs0, fa0
	flw	fa0, 1112(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs11, fa0
	flw	fa0, 1120(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs9, fa0
	flw	fa0, 1128(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs7, fa0
	flw	fa0, 400(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs6, fa0
	flw	fa0, 1136(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs5, fa0
	flw	fa0, 1080(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.s	fs4, fa0
	flw	fa0, 1144(sp)                   # 4-byte Folded Reload
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	sd	a0, 1144(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs4
	sd	a0, 1136(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs5
	sd	a0, 1128(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs6
	sd	a0, 1120(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs7
	sd	a0, 1112(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs9
	sd	a0, 1104(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs11
	sd	a0, 1096(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs0
	sd	a0, 1088(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs1
	sd	a0, 1080(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs2
	sd	a0, 1072(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs10
	sd	a0, 1064(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs8
	sd	a0, 1000(sp)                    # 8-byte Folded Spill
	fmv.x.w	a0, fs3
	sd	a0, 984(sp)                     # 8-byte Folded Spill
	flw	fa5, 976(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 976(sp)                     # 8-byte Folded Spill
	flw	fa5, 960(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 960(sp)                     # 8-byte Folded Spill
	flw	fa5, 952(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 952(sp)                     # 8-byte Folded Spill
	flw	fa5, 328(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 400(sp)                     # 8-byte Folded Spill
	flw	fa5, 40(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 392(sp)                     # 8-byte Folded Spill
	flw	fa5, 272(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 384(sp)                     # 8-byte Folded Spill
	flw	fa5, 44(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 376(sp)                     # 8-byte Folded Spill
	flw	fa5, 48(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 368(sp)                     # 8-byte Folded Spill
	flw	fa5, 52(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 360(sp)                     # 8-byte Folded Spill
	flw	fa5, 56(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 352(sp)                     # 8-byte Folded Spill
	flw	fa5, 60(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 344(sp)                     # 8-byte Folded Spill
	flw	fa5, 64(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 336(sp)                     # 8-byte Folded Spill
	flw	fa5, 72(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 328(sp)                     # 8-byte Folded Spill
	flw	fa5, 320(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 320(sp)                     # 8-byte Folded Spill
	flw	fa5, 80(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 272(sp)                     # 8-byte Folded Spill
	flw	fa5, 120(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	flw	fa5, 112(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	flw	fa5, 104(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	flw	fa5, 88(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	flw	fa5, 96(sp)                     # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	flw	fa5, 128(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	flw	fa5, 136(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	flw	fa5, 144(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	flw	fa5, 152(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	flw	fa5, 160(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 160(sp)                     # 8-byte Folded Spill
	flw	fa5, 944(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 944(sp)                     # 8-byte Folded Spill
	flw	fa5, 928(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 928(sp)                     # 8-byte Folded Spill
	flw	fa5, 920(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 920(sp)                     # 8-byte Folded Spill
	flw	fa5, 904(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 904(sp)                     # 8-byte Folded Spill
	flw	fa5, 896(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 896(sp)                     # 8-byte Folded Spill
	flw	fa5, 880(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 880(sp)                     # 8-byte Folded Spill
	flw	fa5, 872(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 872(sp)                     # 8-byte Folded Spill
	flw	fa5, 856(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 856(sp)                     # 8-byte Folded Spill
	flw	fa5, 848(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 848(sp)                     # 8-byte Folded Spill
	flw	fa5, 840(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 840(sp)                     # 8-byte Folded Spill
	flw	fa5, 824(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 824(sp)                     # 8-byte Folded Spill
	flw	fa5, 808(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 808(sp)                     # 8-byte Folded Spill
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
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	flw	fa5, 312(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 312(sp)                     # 8-byte Folded Spill
	flw	fa5, 1048(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	flw	fa5, 304(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 304(sp)                     # 8-byte Folded Spill
	flw	fa5, 1040(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	flw	fa5, 296(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 296(sp)                     # 8-byte Folded Spill
	flw	fa5, 1032(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1032(sp)                    # 8-byte Folded Spill
	flw	fa5, 288(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 288(sp)                     # 8-byte Folded Spill
	flw	fa5, 1024(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1024(sp)                    # 8-byte Folded Spill
	flw	fa5, 280(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 280(sp)                     # 8-byte Folded Spill
	flw	fa5, 1016(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1016(sp)                    # 8-byte Folded Spill
	flw	fa5, 1008(sp)                   # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1008(sp)                    # 8-byte Folded Spill
	flw	fa5, 264(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 264(sp)                     # 8-byte Folded Spill
	flw	fa5, 992(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 992(sp)                     # 8-byte Folded Spill
	flw	fa5, 256(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 256(sp)                     # 8-byte Folded Spill
	flw	fa5, 968(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 968(sp)                     # 8-byte Folded Spill
	flw	fa5, 248(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 248(sp)                     # 8-byte Folded Spill
	flw	fa5, 936(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 936(sp)                     # 8-byte Folded Spill
	flw	fa5, 240(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 240(sp)                     # 8-byte Folded Spill
	flw	fa5, 912(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 912(sp)                     # 8-byte Folded Spill
	flw	fa5, 232(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 232(sp)                     # 8-byte Folded Spill
	flw	fa5, 888(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 888(sp)                     # 8-byte Folded Spill
	flw	fa5, 224(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 224(sp)                     # 8-byte Folded Spill
	flw	fa5, 864(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 864(sp)                     # 8-byte Folded Spill
	flw	fa5, 216(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	flw	fa5, 832(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 832(sp)                     # 8-byte Folded Spill
	flw	fa5, 208(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	flw	fa5, 816(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 816(sp)                     # 8-byte Folded Spill
	flw	fa5, 200(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	flw	fa5, 800(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 800(sp)                     # 8-byte Folded Spill
	flw	fa5, 168(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	flw	fa5, 176(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 176(sp)                     # 8-byte Folded Spill
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
	fmv.x.w	s3, fa5
	flw	fa5, 520(sp)                    # 4-byte Folded Reload
	fmv.x.w	s2, fa5
	flw	fa5, 512(sp)                    # 4-byte Folded Reload
	fmv.x.w	s7, fa5
	flw	fa5, 504(sp)                    # 4-byte Folded Reload
	fmv.x.w	s4, fa5
	flw	fa5, 496(sp)                    # 4-byte Folded Reload
	fmv.x.w	s5, fa5
	flw	fa5, 488(sp)                    # 4-byte Folded Reload
	fmv.x.w	s6, fa5
	flw	fa5, 480(sp)                    # 4-byte Folded Reload
	fmv.x.w	s9, fa5
	flw	fa5, 472(sp)                    # 4-byte Folded Reload
	fmv.x.w	s10, fa5
	flw	fa5, 464(sp)                    # 4-byte Folded Reload
	fmv.x.w	s11, fa5
	flw	fa5, 456(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1056(sp)                    # 8-byte Folded Spill
	flw	fa5, 448(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1048(sp)                    # 8-byte Folded Spill
	flw	fa5, 188(sp)                    # 4-byte Folded Reload
	fmv.x.w	a0, fa5
	sd	a0, 1040(sp)                    # 8-byte Folded Spill
	flw	fa0, 412(sp)                    # 4-byte Folded Reload
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	sh	s11, 1312(s8)
	sh	s10, 1314(s8)
	sh	s9, 1316(s8)
	sh	s6, 1318(s8)
	sh	s5, 1320(s8)
	sh	s4, 1322(s8)
	sh	s7, 1324(s8)
	sh	s2, 1326(s8)
	sh	s3, 1328(s8)
	ld	a0, 536(sp)                     # 8-byte Folded Reload
	sh	a0, 1330(s8)
	ld	a0, 544(sp)                     # 8-byte Folded Reload
	sh	a0, 1332(s8)
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	sh	a0, 1334(s8)
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	sh	a0, 1336(s8)
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	sh	a0, 1338(s8)
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	sh	a0, 1340(s8)
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	sh	a0, 1342(s8)
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	sh	a0, 1344(s8)
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	sh	a0, 1346(s8)
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	sh	a0, 1348(s8)
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	sh	a0, 1350(s8)
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	sh	a0, 1352(s8)
	ld	a0, 632(sp)                     # 8-byte Folded Reload
	sh	a0, 1354(s8)
	ld	a0, 640(sp)                     # 8-byte Folded Reload
	sh	a0, 1356(s8)
	ld	a0, 648(sp)                     # 8-byte Folded Reload
	sh	a0, 1358(s8)
	ld	a0, 656(sp)                     # 8-byte Folded Reload
	sh	a0, 1360(s8)
	ld	a0, 664(sp)                     # 8-byte Folded Reload
	sh	a0, 1362(s8)
	ld	a0, 672(sp)                     # 8-byte Folded Reload
	sh	a0, 1364(s8)
	ld	a0, 680(sp)                     # 8-byte Folded Reload
	sh	a0, 1366(s8)
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	sh	a0, 1240(s8)
	ld	a0, 168(sp)                     # 8-byte Folded Reload
	sh	a0, 1242(s8)
	ld	a0, 800(sp)                     # 8-byte Folded Reload
	sh	a0, 1244(s8)
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	sh	a0, 1246(s8)
	ld	a0, 816(sp)                     # 8-byte Folded Reload
	sh	a0, 1248(s8)
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	sh	a0, 1250(s8)
	ld	a0, 832(sp)                     # 8-byte Folded Reload
	sh	a0, 1252(s8)
	ld	a0, 216(sp)                     # 8-byte Folded Reload
	sh	a0, 1254(s8)
	ld	a0, 864(sp)                     # 8-byte Folded Reload
	sh	a0, 1256(s8)
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	sh	a0, 1258(s8)
	ld	a0, 888(sp)                     # 8-byte Folded Reload
	sh	a0, 1260(s8)
	ld	a0, 232(sp)                     # 8-byte Folded Reload
	sh	a0, 1262(s8)
	ld	a0, 912(sp)                     # 8-byte Folded Reload
	sh	a0, 1264(s8)
	ld	a0, 240(sp)                     # 8-byte Folded Reload
	sh	a0, 1266(s8)
	ld	a0, 936(sp)                     # 8-byte Folded Reload
	sh	a0, 1268(s8)
	ld	a0, 248(sp)                     # 8-byte Folded Reload
	sh	a0, 1270(s8)
	ld	a0, 968(sp)                     # 8-byte Folded Reload
	sh	a0, 1272(s8)
	ld	a0, 256(sp)                     # 8-byte Folded Reload
	sh	a0, 1274(s8)
	ld	a0, 992(sp)                     # 8-byte Folded Reload
	sh	a0, 1276(s8)
	ld	a0, 264(sp)                     # 8-byte Folded Reload
	sh	a0, 1278(s8)
	ld	a0, 1008(sp)                    # 8-byte Folded Reload
	sh	a0, 1280(s8)
	ld	a0, 1016(sp)                    # 8-byte Folded Reload
	sh	a0, 1282(s8)
	ld	a0, 280(sp)                     # 8-byte Folded Reload
	sh	a0, 1284(s8)
	ld	a0, 1024(sp)                    # 8-byte Folded Reload
	sh	a0, 1286(s8)
	ld	a0, 288(sp)                     # 8-byte Folded Reload
	sh	a0, 1288(s8)
	ld	a0, 1032(sp)                    # 8-byte Folded Reload
	sh	a0, 1290(s8)
	ld	a0, 296(sp)                     # 8-byte Folded Reload
	sh	a0, 1292(s8)
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	sh	a0, 1294(s8)
	ld	a0, 304(sp)                     # 8-byte Folded Reload
	sh	a0, 1296(s8)
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	sh	a0, 1298(s8)
	ld	a0, 312(sp)                     # 8-byte Folded Reload
	sh	a0, 1300(s8)
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	sh	a0, 1302(s8)
	ld	a0, 688(sp)                     # 8-byte Folded Reload
	sh	a0, 1560(s8)
	ld	a0, 696(sp)                     # 8-byte Folded Reload
	sh	a0, 1562(s8)
	ld	a0, 704(sp)                     # 8-byte Folded Reload
	sh	a0, 1564(s8)
	ld	a0, 712(sp)                     # 8-byte Folded Reload
	sh	a0, 1566(s8)
	ld	a0, 720(sp)                     # 8-byte Folded Reload
	sh	a0, 1568(s8)
	ld	a0, 728(sp)                     # 8-byte Folded Reload
	sh	a0, 1570(s8)
	ld	a0, 736(sp)                     # 8-byte Folded Reload
	sh	a0, 1572(s8)
	ld	a0, 744(sp)                     # 8-byte Folded Reload
	sh	a0, 1574(s8)
	ld	a0, 752(sp)                     # 8-byte Folded Reload
	sh	a0, 1576(s8)
	ld	a0, 760(sp)                     # 8-byte Folded Reload
	sh	a0, 1578(s8)
	ld	a0, 768(sp)                     # 8-byte Folded Reload
	sh	a0, 1580(s8)
	ld	a0, 776(sp)                     # 8-byte Folded Reload
	sh	a0, 1582(s8)
	ld	a0, 784(sp)                     # 8-byte Folded Reload
	sh	a0, 1584(s8)
	ld	a0, 792(sp)                     # 8-byte Folded Reload
	sh	a0, 1586(s8)
	ld	a0, 808(sp)                     # 8-byte Folded Reload
	sh	a0, 1588(s8)
	ld	a0, 824(sp)                     # 8-byte Folded Reload
	sh	a0, 1590(s8)
	ld	a0, 840(sp)                     # 8-byte Folded Reload
	sh	a0, 1592(s8)
	ld	a0, 848(sp)                     # 8-byte Folded Reload
	sh	a0, 1594(s8)
	ld	a0, 856(sp)                     # 8-byte Folded Reload
	sh	a0, 1596(s8)
	ld	a0, 872(sp)                     # 8-byte Folded Reload
	sh	a0, 1598(s8)
	ld	a0, 880(sp)                     # 8-byte Folded Reload
	sh	a0, 1600(s8)
	ld	a0, 896(sp)                     # 8-byte Folded Reload
	sh	a0, 1602(s8)
	ld	a0, 904(sp)                     # 8-byte Folded Reload
	sh	a0, 1604(s8)
	ld	a0, 920(sp)                     # 8-byte Folded Reload
	sh	a0, 1606(s8)
	ld	a0, 928(sp)                     # 8-byte Folded Reload
	sh	a0, 1608(s8)
	ld	a0, 944(sp)                     # 8-byte Folded Reload
	sh	a0, 1610(s8)
	ld	a0, 160(sp)                     # 8-byte Folded Reload
	sh	a0, 1612(s8)
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	sh	a0, 1614(s8)
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	sh	a0, 1616(s8)
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	sh	a0, 1618(s8)
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	sh	a0, 1620(s8)
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	sh	a0, 1622(s8)
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	sh	a0, 1496(s8)
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	sh	a0, 1498(s8)
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	sh	a0, 1500(s8)
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	sh	a0, 1502(s8)
	ld	a0, 272(sp)                     # 8-byte Folded Reload
	sh	a0, 1504(s8)
	ld	a0, 320(sp)                     # 8-byte Folded Reload
	sh	a0, 1506(s8)
	ld	a0, 328(sp)                     # 8-byte Folded Reload
	sh	a0, 1508(s8)
	ld	a0, 336(sp)                     # 8-byte Folded Reload
	sh	a0, 1510(s8)
	ld	a0, 344(sp)                     # 8-byte Folded Reload
	sh	a0, 1512(s8)
	ld	a0, 352(sp)                     # 8-byte Folded Reload
	sh	a0, 1514(s8)
	ld	a0, 360(sp)                     # 8-byte Folded Reload
	sh	a0, 1516(s8)
	ld	a0, 368(sp)                     # 8-byte Folded Reload
	sh	a0, 1518(s8)
	ld	a0, 376(sp)                     # 8-byte Folded Reload
	sh	a0, 1520(s8)
	ld	a0, 384(sp)                     # 8-byte Folded Reload
	sh	a0, 1522(s8)
	ld	a0, 392(sp)                     # 8-byte Folded Reload
	sh	a0, 1524(s8)
	ld	a0, 400(sp)                     # 8-byte Folded Reload
	sh	a0, 1526(s8)
	li	a0, -32
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	li	a3, 32
	.loc	1 14 37                         # k135114294089648.py:14:37
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v8, v16, 31
	vsrl.vi	v8, v8, 27
	vadd.vv	v8, v16, v8
	vand.vx	v8, v8, a0
	vsub.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 13 62                         # k135114294089648.py:13:62
	fmv.x.w	a2, fa0
	ld	a1, 952(sp)                     # 8-byte Folded Reload
	sh	a1, 1528(s8)
	ld	a1, 960(sp)                     # 8-byte Folded Reload
	sh	a1, 1530(s8)
	ld	a1, 976(sp)                     # 8-byte Folded Reload
	sh	a1, 1532(s8)
	ld	a1, 984(sp)                     # 8-byte Folded Reload
	sh	a1, 1534(s8)
	ld	a1, 1000(sp)                    # 8-byte Folded Reload
	sh	a1, 1536(s8)
	ld	a1, 1064(sp)                    # 8-byte Folded Reload
	sh	a1, 1538(s8)
	ld	a1, 1072(sp)                    # 8-byte Folded Reload
	sh	a1, 1540(s8)
	ld	a1, 1080(sp)                    # 8-byte Folded Reload
	sh	a1, 1542(s8)
	ld	a1, 1088(sp)                    # 8-byte Folded Reload
	sh	a1, 1544(s8)
	ld	a1, 1096(sp)                    # 8-byte Folded Reload
	sh	a1, 1546(s8)
	ld	a1, 1104(sp)                    # 8-byte Folded Reload
	sh	a1, 1548(s8)
	ld	a1, 1112(sp)                    # 8-byte Folded Reload
	sh	a1, 1550(s8)
	ld	a1, 1120(sp)                    # 8-byte Folded Reload
	sh	a1, 1552(s8)
	ld	a1, 1128(sp)                    # 8-byte Folded Reload
	sh	a1, 1554(s8)
	ld	a1, 1136(sp)                    # 8-byte Folded Reload
	sh	a1, 1556(s8)
	ld	a1, 1144(sp)                    # 8-byte Folded Reload
	sh	a1, 1558(s8)
	li	a1, 4
	sh	a2, 1304(s8)
	ld	a2, 1040(sp)                    # 8-byte Folded Reload
	sh	a2, 1306(s8)
	ld	a2, 1048(sp)                    # 8-byte Folded Reload
	sh	a2, 1308(s8)
	.loc	1 14 43                         # k135114294089648.py:14:43
	vsetivli	zero, 16, e32, m4, ta, mu
	vmv.v.i	v8, 0
	ld	a2, 1056(sp)                    # 8-byte Folded Reload
	.loc	1 13 62                         # k135114294089648.py:13:62
	sh	a2, 1310(s8)
	.loc	1 14 43                         # k135114294089648.py:14:43
	vmv.v.i	v16, 0
	csrr	a2, vlenb
	li	a4, 185
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 7
	addi	a4, a4, -832
	add	a2, a2, a4
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	csrr	a2, vlenb
	li	a4, 177
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 7
	addi	a4, a4, -832
	add	a2, a2, a4
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	vwmulsu.vx	v24, v16, a1
	csrr	a2, vlenb
	li	a4, 185
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 7
	addi	a4, a4, -832
	add	a2, a2, a4
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	ld	a2, 416(sp)                     # 8-byte Folded Reload
	vluxei64.v	v16, (a2), v24, v0.t
	csrr	a4, vlenb
	li	a5, 185
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	csrr	a4, vlenb
	li	a5, 177
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v16, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v0, v0, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v16, v24, a1
	vmv4r.v	v24, v8
	vluxei64.v	v24, (a2), v16, v0.t
	csrr	a4, vlenb
	li	a5, 201
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v0, (a4)                        # vscale x 64-byte Folded Reload
	.loc	1 14 37 is_stmt 0               # k135114294089648.py:14:37
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v0, v0, 31
	vsrl.vi	v0, v0, 27
	csrr	a4, vlenb
	li	a5, 201
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	vadd.vv	v0, v16, v0
	vand.vx	v0, v0, a0
	csrr	a4, vlenb
	li	a5, 201
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	vsub.vv	v0, v16, v0
	csrr	a4, vlenb
	slli	a4, a4, 4
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v0, (a4)                        # vscale x 64-byte Folded Spill
	.loc	1 14 43                         # k135114294089648.py:14:43
	csrr	a4, vlenb
	li	a5, 185
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	vslideup.vi	v16, v24, 16
	csrr	a4, vlenb
	li	a5, 185
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	vmv4r.v	v16, v8
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v24, v0, a1
	csrr	a4, vlenb
	li	a5, 201
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v24, (a4)                       # vscale x 64-byte Folded Spill
	csrr	a4, vlenb
	li	a5, 24
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl1r.v	v12, (a4)                       # vscale x 8-byte Folded Reload
	vmv1r.v	v0, v12
	csrr	a4, vlenb
	li	a5, 201
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v24, (a4)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v16, (a2), v24, v0.t
	csrr	a4, vlenb
	li	a5, 177
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	csrr	a4, vlenb
	li	a5, 153
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v0, (a4)                        # vscale x 64-byte Folded Reload
	.loc	1 23 20 is_stmt 1               # k135114294089648.py:23:20
	vsetvli	zero, a3, e32, m8, ta, ma
	vfcvt.f.x.v	v0, v0
	csrr	a4, vlenb
	li	a5, 185
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 25 20                         # k135114294089648.py:25:20
	vfmul.vv	v16, v16, v0
	csrr	a4, vlenb
	li	a5, 169
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	csrr	a4, vlenb
	slli	a4, a4, 4
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 14 43                         # k135114294089648.py:14:43
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v16, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v0, v12, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v24, v16, a1
	vmv4r.v	v16, v8
	vluxei64.v	v16, (a2), v24, v0.t
	csrr	a4, vlenb
	li	a5, 193
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v0, (a4)                        # vscale x 64-byte Folded Reload
	.loc	1 14 37 is_stmt 0               # k135114294089648.py:14:37
	vsetvli	zero, a3, e32, m8, ta, ma
	li	a3, 32
	vsra.vi	v0, v0, 31
	vsrl.vi	v0, v0, 27
	csrr	a4, vlenb
	li	a5, 193
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v24, (a4)                       # vscale x 64-byte Folded Reload
	vadd.vv	v0, v24, v0
	vand.vx	v0, v0, a0
	csrr	a4, vlenb
	li	a5, 193
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v24, (a4)                       # vscale x 64-byte Folded Reload
	vsub.vv	v0, v24, v0
	csrr	a4, vlenb
	li	a5, 193
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v0, (a4)                        # vscale x 64-byte Folded Spill
	.loc	1 14 43                         # k135114294089648.py:14:43
	csrr	a4, vlenb
	li	a5, 177
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v24, (a4)                       # vscale x 64-byte Folded Reload
	vslideup.vi	v24, v16, 16
	csrr	a4, vlenb
	li	a5, 177
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v24, (a4)                       # vscale x 64-byte Folded Spill
	vmv4r.v	v16, v8
	vmv.v.v	v24, v0
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v0, v24, a1
	csrr	a4, vlenb
	li	a5, 185
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v0, (a4)                        # vscale x 64-byte Folded Spill
	csrr	a4, vlenb
	slli	a4, a4, 5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl1r.v	v12, (a4)                       # vscale x 8-byte Folded Reload
	vmv1r.v	v0, v12
	csrr	a4, vlenb
	li	a5, 185
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v24, (a4)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v16, (a2), v24, v0.t
	csrr	a4, vlenb
	li	a5, 201
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	csrr	a4, vlenb
	li	a5, 161
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v0, (a4)                        # vscale x 64-byte Folded Reload
	.loc	1 23 20 is_stmt 1               # k135114294089648.py:23:20
	vsetvli	zero, a3, e32, m8, ta, ma
	vfcvt.f.x.v	v0, v0
	csrr	a4, vlenb
	li	a5, 177
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 25 20                         # k135114294089648.py:25:20
	vfmul.vv	v16, v16, v0
	csrr	a4, vlenb
	li	a5, 177
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	csrr	a4, vlenb
	li	a5, 193
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 14 43                         # k135114294089648.py:14:43
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v16, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v0, v12, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v16, v24, a1
	vmv4r.v	v24, v8
	vluxei64.v	v24, (a2), v16, v0.t
	csrr	a4, vlenb
	li	a5, 137
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v0, (a4)                        # vscale x 64-byte Folded Reload
	.loc	1 14 37 is_stmt 0               # k135114294089648.py:14:37
	vsetvli	zero, a3, e32, m8, ta, ma
	vsra.vi	v0, v0, 31
	vsrl.vi	v0, v0, 27
	csrr	a4, vlenb
	li	a5, 137
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -832
	add	a4, a4, a5
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	vadd.vv	v0, v16, v0
	vand.vx	v0, v0, a0
	csrr	a0, vlenb
	li	a4, 137
	mul	a0, a0, a4
	add	a0, sp, a0
	lui	a4, 7
	addi	a4, a4, -832
	add	a0, a0, a4
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsub.vv	v0, v16, v0
	csrr	a0, vlenb
	li	a4, 193
	mul	a0, a0, a4
	add	a0, sp, a0
	lui	a4, 7
	addi	a4, a4, -832
	add	a0, a0, a4
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 14 43                         # k135114294089648.py:14:43
	csrr	a0, vlenb
	li	a4, 201
	mul	a0, a0, a4
	add	a0, sp, a0
	lui	a4, 7
	addi	a4, a4, -832
	add	a0, a0, a4
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vslideup.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a4, 201
	mul	a0, a0, a4
	add	a0, sp, a0
	lui	a4, 7
	addi	a4, a4, -832
	add	a0, a0, a4
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vmv4r.v	v24, v8
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v16, v0, a1
	csrr	a0, vlenb
	li	a4, 40
	mul	a0, a0, a4
	add	a0, sp, a0
	lui	a4, 7
	addi	a4, a4, -832
	add	a0, a0, a4
	vl1r.v	v12, (a0)                       # vscale x 8-byte Folded Reload
	vmv1r.v	v0, v12
	vluxei64.v	v24, (a2), v16, v0.t
	csrr	a0, vlenb
	li	a4, 193
	mul	a0, a0, a4
	add	a0, sp, a0
	lui	a4, 7
	addi	a4, a4, -832
	add	a0, a0, a4
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v0, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v12, v12, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v0, v16, a1
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vmv1r.v	v0, v12
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v8, (a2), v16, v0.t
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 23 20 is_stmt 1               # k135114294089648.py:23:20
	vsetvli	zero, a3, e32, m8, ta, ma
	vfcvt.f.x.v	v0, v0
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 20                         # k135114294089648.py:25:20
	vfmul.vv	v16, v16, v0
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 14 43                         # k135114294089648.py:14:43
	vslideup.vi	v24, v8, 16
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 23 20                         # k135114294089648.py:23:20
	vfcvt.f.x.v	v8, v8
	.loc	1 25 20                         # k135114294089648.py:25:20
	vfmul.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	ld	a0, 424(sp)                     # 8-byte Folded Reload
	.loc	1 15 31                         # k135114294089648.py:15:31
	slli	a0, a0, 1
	li	s7, 64
	.loc	1 15 36 is_stmt 0               # k135114294089648.py:15:36
	vsetvli	zero, s7, e16, m8, ta, mu
	vmv.v.i	v24, 0
	ld	a1, 432(sp)                     # 8-byte Folded Reload
	.loc	1 15 31                         # k135114294089648.py:15:31
	add	a0, a1, a0
	.loc	1 15 36                         # k135114294089648.py:15:36
	vmv.v.i	v8, 0
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vle16.v	v8, (a0), v0.t
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	addi	a0, a0, 128
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vle16.v	v24, (a0), v0.t
	li	a1, 56
	li	a0, 128
	.loc	1 12 33 is_stmt 1               # k135114294089648.py:12:33
	vsetivli	zero, 16, e64, m8, ta, ma
	ld	a2, 192(sp)                     # 8-byte Folded Reload
	vmv.v.x	v8, a2
	csrr	a2, vlenb
	li	a3, 201
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 7
	addi	a3, a3, -832
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	csrr	a2, vlenb
	li	a3, 105
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 7
	addi	a3, a3, -832
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	csrr	a2, vlenb
	li	a3, 201
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 7
	addi	a3, a3, -832
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114294089648.py:17:18
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v0, v8, v16
	vmv8r.v	v16, v0
	.loc	1 21 32                         # k135114294089648.py:21:32
	vsetvli	zero, zero, e64, m8, ta, ma
	li	s6, 56
	vsrl.vx	v0, v0, a1
	vand.vx	v0, v0, a0
	vadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	li	a0, 32
	.loc	1 15 46                         # k135114294089648.py:15:46
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v0, v8, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a0
	li	s3, 32
	lui	a0, 4
	addi	a0, a0, -1344
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1152
	add	a1, sp, a1
	.loc	1 13 62                         # k135114294089648.py:13:62
	vsetvli	zero, s3, e32, m8, ta, ma
	vle16.v	v28, (a1)
	lui	a1, 4
	addi	a1, a1, -1408
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1088
	add	a2, sp, a2
	vle16.v	v4, (a2)
	vzext.vf2	v8, v28
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 56
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 7
	addi	a3, a3, -832
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v28, (a1)
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs4r.v	v8, (a0)                        # vscale x 32-byte Folded Spill
	vzext.vf2	v8, v4
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vzext.vf2	v8, v28
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl4r.v	v28, (a0)                       # vscale x 32-byte Folded Reload
	vzext.vf2	v8, v28
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 15 46                         # k135114294089648.py:15:46
	vzext.vf2	v8, v0
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vzext.vf2	v16, v24
	vsll.vi	v8, v8, 16
	lui	a0, 7
	addi	a0, a0, -832
	add	a0, sp, a0
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	lui	a0, 3
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 26 24                         # k135114294089648.py:26:24
	vse32.v	v8, (a0)
	flw	fa0, 84(s8)
	fsw	fa0, 1144(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 852(s8)
	flw	fa0, 80(s8)
	fsw	fa0, 1136(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 848(s8)
	flw	fa0, 76(s8)
	fsw	fa0, 1128(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 844(s8)
	flw	fa0, 72(s8)
	fsw	fa0, 1120(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 840(s8)
	flw	fa0, 68(s8)
	fsw	fa0, 1112(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 836(s8)
	flw	fa0, 64(s8)
	fsw	fa0, 1104(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 832(s8)
	flw	fa0, 60(s8)
	fsw	fa0, 1096(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 828(s8)
	flw	fa0, 56(s8)
	fsw	fa0, 1088(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 824(s8)
	flw	fa0, 52(s8)
	fsw	fa0, 1080(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 820(s8)
	flw	fa0, 48(s8)
	fsw	fa0, 1072(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 816(s8)
	flw	fa0, 44(s8)
	fsw	fa0, 1064(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 812(s8)
	flw	fa0, 40(s8)
	fsw	fa0, 1056(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 808(s8)
	flw	fa0, 36(s8)
	fsw	fa0, 1048(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 804(s8)
	flw	fa0, 32(s8)
	fsw	fa0, 1040(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 800(s8)
	flw	fa0, 28(s8)
	fsw	fa0, 1032(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 796(s8)
	flw	fa0, 24(s8)
	fsw	fa0, 1024(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 792(s8)
	flw	fa0, 20(s8)
	fsw	fa0, 1016(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 788(s8)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 1008(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 728(s8)
	flw	fa0, 16(s8)
	fsw	fa0, 1000(sp)                   # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 784(s8)
	flw	fa0, 12(s8)
	fsw	fa0, 992(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 780(s8)
	flw	fa0, 8(s8)
	fsw	fa0, 984(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 776(s8)
	flw	fa0, 4(s8)
	fsw	fa0, 976(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 772(s8)
	flw	fa0, 0(s8)
	fsw	fa0, 968(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 768(s8)
	lui	a0, 3
	addi	a0, a0, -600
	add	s2, sp, a0
	flw	fa0, 2044(s2)
	fsw	fa0, 960(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 764(s8)
	flw	fa0, 2040(s2)
	fsw	fa0, 952(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 760(s8)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 944(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 740(s8)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 936(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 736(s8)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 928(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 732(s8)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 920(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 756(s8)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 912(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 752(s8)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 904(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 748(s8)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 896(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 744(s8)
	li	a0, 7
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 724(s8)
	fsw	fa0, 888(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1108(s8)
	flw	fa0, 720(s8)
	fsw	fa0, 880(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1104(s8)
	flw	fa0, 716(s8)
	fsw	fa0, 872(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1100(s8)
	flw	fa0, 712(s8)
	fsw	fa0, 864(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1096(s8)
	flw	fa0, 708(s8)
	fsw	fa0, 856(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1092(s8)
	flw	fa0, 704(s8)
	fsw	fa0, 848(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1088(s8)
	flw	fa0, 700(s8)
	fsw	fa0, 840(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1084(s8)
	flw	fa0, 696(s8)
	fsw	fa0, 832(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1080(s8)
	flw	fa0, 692(s8)
	fsw	fa0, 824(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1076(s8)
	flw	fa0, 688(s8)
	fsw	fa0, 816(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1072(s8)
	flw	fa0, 684(s8)
	fsw	fa0, 808(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1068(s8)
	flw	fa0, 680(s8)
	fsw	fa0, 800(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1064(s8)
	flw	fa0, 676(s8)
	fsw	fa0, 792(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1060(s8)
	flw	fa0, 672(s8)
	fsw	fa0, 784(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1056(s8)
	flw	fa0, 668(s8)
	fsw	fa0, 776(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1052(s8)
	flw	fa0, 664(s8)
	fsw	fa0, 768(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1048(s8)
	flw	fa0, 660(s8)
	fsw	fa0, 760(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1044(s8)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 752(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 984(s8)
	flw	fa0, 656(s8)
	fsw	fa0, 744(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1040(s8)
	flw	fa0, 652(s8)
	fsw	fa0, 736(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1036(s8)
	flw	fa0, 648(s8)
	fsw	fa0, 728(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1032(s8)
	flw	fa0, 644(s8)
	fsw	fa0, 720(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1028(s8)
	flw	fa0, 640(s8)
	fsw	fa0, 712(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1024(s8)
	flw	fa0, 636(s8)
	fsw	fa0, 704(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1020(s8)
	flw	fa0, 632(s8)
	fsw	fa0, 696(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1016(s8)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 688(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 996(s8)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 680(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 992(s8)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 672(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 988(s8)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 664(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1012(s8)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 656(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1008(s8)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 648(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1004(s8)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 640(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1000(s8)
	lui	a0, 3
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 1876(s2)
	fsw	fa0, 632(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 212(s8)
	flw	fa0, 1872(s2)
	fsw	fa0, 624(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 208(s8)
	flw	fa0, 1868(s2)
	fsw	fa0, 616(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 204(s8)
	flw	fa0, 1864(s2)
	fsw	fa0, 608(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 200(s8)
	flw	fa0, 1860(s2)
	fsw	fa0, 600(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 196(s8)
	flw	fa0, 1856(s2)
	fsw	fa0, 592(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 192(s8)
	flw	fa0, 1852(s2)
	fsw	fa0, 584(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 188(s8)
	flw	fa0, 1848(s2)
	fsw	fa0, 576(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 184(s8)
	flw	fa0, 1844(s2)
	fsw	fa0, 568(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 180(s8)
	flw	fa0, 1840(s2)
	fsw	fa0, 560(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 176(s8)
	flw	fa0, 1836(s2)
	fsw	fa0, 552(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 172(s8)
	flw	fa0, 1832(s2)
	fsw	fa0, 544(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 168(s8)
	flw	fa0, 1828(s2)
	fsw	fa0, 536(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 164(s8)
	flw	fa0, 1824(s2)
	fsw	fa0, 528(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 160(s8)
	flw	fa0, 1820(s2)
	fsw	fa0, 520(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 156(s8)
	flw	fa0, 1816(s2)
	fsw	fa0, 512(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 152(s8)
	flw	fa0, 1812(s2)
	fsw	fa0, 504(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 148(s8)
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 496(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 88(s8)
	flw	fa0, 1808(s2)
	fsw	fa0, 488(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 144(s8)
	flw	fa0, 1804(s2)
	fsw	fa0, 480(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 140(s8)
	flw	fa0, 1800(s2)
	fsw	fa0, 472(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 136(s8)
	flw	fa0, 1796(s2)
	fsw	fa0, 464(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 132(s8)
	flw	fa0, 1792(s2)
	fsw	fa0, 456(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 128(s8)
	flw	fa0, 1788(s2)
	fsw	fa0, 448(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 124(s8)
	flw	fa0, 1784(s2)
	fsw	fa0, 432(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 120(s8)
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 424(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 100(s8)
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 416(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 96(s8)
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 412(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 92(s8)
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 400(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 116(s8)
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 392(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 112(s8)
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 384(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 108(s8)
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 376(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 104(s8)
	lui	a0, 3
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 2004(s2)
	fsw	fa0, 368(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 468(s8)
	flw	fa0, 2000(s2)
	fsw	fa0, 360(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 464(s8)
	flw	fa0, 1996(s2)
	fsw	fa0, 352(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 460(s8)
	flw	fa0, 1992(s2)
	fsw	fa0, 344(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 456(s8)
	flw	fa0, 1988(s2)
	fsw	fa0, 336(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 452(s8)
	flw	fa0, 1984(s2)
	fsw	fa0, 328(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 448(s8)
	flw	fa0, 1980(s2)
	fsw	fa0, 320(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 444(s8)
	flw	fa0, 1976(s2)
	fsw	fa0, 312(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 440(s8)
	flw	fa0, 1972(s2)
	fsw	fa0, 304(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 436(s8)
	flw	fa0, 1968(s2)
	fsw	fa0, 296(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 432(s8)
	flw	fa0, 1964(s2)
	fsw	fa0, 288(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 428(s8)
	flw	fa0, 1960(s2)
	fsw	fa0, 280(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 424(s8)
	flw	fa0, 1956(s2)
	fsw	fa0, 272(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 420(s8)
	flw	fa0, 1952(s2)
	fsw	fa0, 264(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 416(s8)
	flw	fa0, 1948(s2)
	fsw	fa0, 256(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 412(s8)
	flw	fa0, 1944(s2)
	fsw	fa0, 248(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 408(s8)
	flw	fa0, 1940(s2)
	fsw	fa0, 240(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 404(s8)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 232(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 344(s8)
	flw	fa0, 1936(s2)
	fsw	fa0, 224(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 400(s8)
	flw	fa0, 1932(s2)
	fsw	fa0, 216(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 396(s8)
	flw	fs8, 1928(s2)
	fmv.s	fa0, fs8
	call	cosf
	fsw	fa0, 392(s8)
	flw	fs9, 1924(s2)
	fmv.s	fa0, fs9
	call	cosf
	fsw	fa0, 388(s8)
	flw	fs10, 1920(s2)
	fmv.s	fa0, fs10
	call	cosf
	fsw	fa0, 384(s8)
	flw	fs11, 1916(s2)
	fmv.s	fa0, fs11
	call	cosf
	fsw	fa0, 380(s8)
	flw	fs0, 1912(s2)
	fmv.s	fa0, fs0
	call	cosf
	fsw	fa0, 376(s8)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fs1, v8
	fmv.s	fa0, fs1
	call	cosf
	fsw	fa0, 356(s8)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fs2, v8
	fmv.s	fa0, fs2
	call	cosf
	fsw	fa0, 352(s8)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fs3, v8
	fmv.s	fa0, fs3
	call	cosf
	fsw	fa0, 348(s8)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fs4, v8
	fmv.s	fa0, fs4
	call	cosf
	fsw	fa0, 372(s8)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fs5, v8
	fmv.s	fa0, fs5
	call	cosf
	fsw	fa0, 368(s8)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fs6, v8
	fmv.s	fa0, fs6
	call	cosf
	fsw	fa0, 364(s8)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fs7, v8
	fmv.s	fa0, fs7
	call	cosf
	fsw	fa0, 360(s8)
	lui	a0, 4
	addi	a0, a0, -1920
	add	a0, sp, a0
	vsetvli	zero, s3, e32, m8, ta, ma
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	lui	a0, 4
	addi	a0, a0, -1664
	add	a0, sp, a0
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	li	a0, 27
	slli	a0, a0, 9
	add	a0, sp, a0
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	lui	a0, 3
	addi	a0, a0, 1792
	add	a0, sp, a0
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	flw	fa0, 368(sp)                    # 4-byte Folded Reload
	.loc	1 29 24                         # k135114294089648.py:29:24
	call	sinf
	fsw	fa0, 596(s8)
	flw	fa0, 360(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 592(s8)
	flw	fa0, 352(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 588(s8)
	flw	fa0, 344(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 584(s8)
	flw	fa0, 336(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 580(s8)
	flw	fa0, 328(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 576(s8)
	flw	fa0, 320(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 572(s8)
	flw	fa0, 312(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 568(s8)
	flw	fa0, 304(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 564(s8)
	flw	fa0, 296(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 560(s8)
	flw	fa0, 288(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 556(s8)
	flw	fa0, 280(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 552(s8)
	flw	fa0, 272(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 548(s8)
	flw	fa0, 264(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 544(s8)
	flw	fa0, 256(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 540(s8)
	flw	fa0, 248(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 536(s8)
	flw	fa0, 240(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 532(s8)
	flw	fa0, 232(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 472(s8)
	flw	fa0, 224(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 528(s8)
	flw	fa0, 216(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 524(s8)
	fmv.s	fa0, fs8
	call	sinf
	fsw	fa0, 520(s8)
	fmv.s	fa0, fs9
	call	sinf
	fsw	fa0, 516(s8)
	fmv.s	fa0, fs10
	call	sinf
	fsw	fa0, 512(s8)
	fmv.s	fa0, fs11
	call	sinf
	fsw	fa0, 508(s8)
	fmv.s	fa0, fs0
	call	sinf
	fsw	fa0, 504(s8)
	fmv.s	fa0, fs1
	call	sinf
	fsw	fa0, 484(s8)
	fmv.s	fa0, fs2
	call	sinf
	fsw	fa0, 480(s8)
	fmv.s	fa0, fs3
	call	sinf
	fsw	fa0, 476(s8)
	fmv.s	fa0, fs4
	call	sinf
	fsw	fa0, 500(s8)
	fmv.s	fa0, fs5
	call	sinf
	fsw	fa0, 496(s8)
	fmv.s	fa0, fs6
	call	sinf
	fsw	fa0, 492(s8)
	fmv.s	fa0, fs7
	call	sinf
	fsw	fa0, 488(s8)
	flw	fa0, 632(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 340(s8)
	flw	fa0, 624(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 336(s8)
	flw	fa0, 616(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 332(s8)
	flw	fa0, 608(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 328(s8)
	flw	fa0, 600(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 324(s8)
	flw	fa0, 592(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 320(s8)
	flw	fa0, 584(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 316(s8)
	flw	fa0, 576(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 312(s8)
	flw	fa0, 568(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 308(s8)
	flw	fa0, 560(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 304(s8)
	flw	fa0, 552(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 300(s8)
	flw	fa0, 544(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 296(s8)
	flw	fa0, 536(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 292(s8)
	flw	fa0, 528(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 288(s8)
	flw	fa0, 520(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 284(s8)
	flw	fa0, 512(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 280(s8)
	flw	fa0, 504(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 276(s8)
	flw	fa0, 496(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 216(s8)
	flw	fa0, 488(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 272(s8)
	flw	fa0, 480(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 268(s8)
	flw	fa0, 472(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 264(s8)
	flw	fa0, 464(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 260(s8)
	flw	fa0, 456(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 256(s8)
	flw	fa0, 448(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 252(s8)
	flw	fa0, 432(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 248(s8)
	flw	fa0, 424(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 228(s8)
	flw	fa0, 416(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 224(s8)
	flw	fa0, 412(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 220(s8)
	flw	fa0, 400(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 244(s8)
	flw	fa0, 392(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 240(s8)
	flw	fa0, 384(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 236(s8)
	flw	fa0, 376(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 232(s8)
	flw	fa0, 888(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1236(s8)
	flw	fa0, 880(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1232(s8)
	flw	fa0, 872(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1228(s8)
	flw	fa0, 864(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1224(s8)
	flw	fa0, 856(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1220(s8)
	flw	fa0, 848(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1216(s8)
	flw	fa0, 840(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1212(s8)
	flw	fa0, 832(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1208(s8)
	flw	fa0, 824(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1204(s8)
	flw	fa0, 816(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1200(s8)
	flw	fa0, 808(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1196(s8)
	flw	fa0, 800(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1192(s8)
	flw	fa0, 792(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1188(s8)
	flw	fa0, 784(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1184(s8)
	flw	fa0, 776(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1180(s8)
	flw	fa0, 768(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1176(s8)
	flw	fa0, 760(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1172(s8)
	flw	fa0, 752(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1112(s8)
	flw	fa0, 744(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1168(s8)
	flw	fa0, 736(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1164(s8)
	flw	fa0, 728(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1160(s8)
	flw	fa0, 720(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1156(s8)
	flw	fa0, 712(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1152(s8)
	flw	fa0, 704(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1148(s8)
	flw	fa0, 696(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1144(s8)
	flw	fa0, 688(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1124(s8)
	flw	fa0, 680(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1120(s8)
	flw	fa0, 672(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1116(s8)
	flw	fa0, 664(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1140(s8)
	flw	fa0, 656(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1136(s8)
	flw	fa0, 648(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1132(s8)
	flw	fa0, 640(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1128(s8)
	flw	fa0, 1144(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 980(s8)
	flw	fa0, 1136(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 976(s8)
	flw	fa0, 1128(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 972(s8)
	flw	fa0, 1120(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 968(s8)
	flw	fa0, 1112(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 964(s8)
	flw	fa0, 1104(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 960(s8)
	flw	fa0, 1096(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 956(s8)
	flw	fa0, 1088(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 952(s8)
	flw	fa0, 1080(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 948(s8)
	flw	fa0, 1072(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 944(s8)
	flw	fa0, 1064(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 940(s8)
	flw	fa0, 1056(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 936(s8)
	flw	fa0, 1048(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 932(s8)
	flw	fa0, 1040(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 928(s8)
	flw	fa0, 1032(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 924(s8)
	flw	fa0, 1024(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 920(s8)
	flw	fa0, 1016(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 916(s8)
	flw	fa0, 1008(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 856(s8)
	flw	fa0, 1000(sp)                   # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 912(s8)
	flw	fa0, 992(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 908(s8)
	flw	fa0, 984(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 904(s8)
	flw	fa0, 976(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 900(s8)
	flw	fa0, 968(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 896(s8)
	flw	fa0, 960(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 892(s8)
	flw	fa0, 952(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 888(s8)
	flw	fa0, 944(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 868(s8)
	flw	fa0, 936(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 864(s8)
	flw	fa0, 928(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 860(s8)
	flw	fa0, 920(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 884(s8)
	flw	fa0, 912(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 880(s8)
	flw	fa0, 904(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 876(s8)
	flw	fa0, 896(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 872(s8)
	lui	a0, 3
	addi	a0, a0, 1920
	add	a0, sp, a0
	vsetvli	zero, s3, e32, m8, ta, ma
	vle32.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, 1664
	add	a0, sp, a0
	vle32.v	v24, (a0)
	lui	a0, 7
	addi	a0, a0, -832
	add	a0, sp, a0
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 31 20                         # k135114294089648.py:31:20
	vfmul.vv	v0, v8, v16
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 20                         # k135114294089648.py:32:20
	vfmacc.vv	v0, v8, v16
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 31 20                         # k135114294089648.py:31:20
	vfmul.vv	v0, v24, v8
	li	a0, 29
	slli	a0, a0, 9
	add	a0, sp, a0
	.loc	1 29 24                         # k135114294089648.py:29:24
	vle32.v	v8, (a0)
	lui	a0, 4
	addi	a0, a0, -1792
	add	a0, sp, a0
	vle32.v	v16, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 20                         # k135114294089648.py:32:20
	vfmacc.vv	v0, v16, v24
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 31 20                         # k135114294089648.py:31:20
	vfmul.vv	v24, v8, v16
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 20                         # k135114294089648.py:32:20
	vfmacc.vv	v24, v8, v16
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 31 20                         # k135114294089648.py:31:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 20                         # k135114294089648.py:32:20
	vfmacc.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 33 33                         # k135114294089648.py:33:33
	vsetivli	zero, 16, e64, m8, ta, ma
	vsll.vi	v16, v16, 6
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 33 45 is_stmt 0               # k135114294089648.py:33:45
	vsetvli	zero, s3, e32, m8, ta, ma
	vsll.vi	v0, v16, 13
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 33 30                         # k135114294089648.py:33:30
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v16, v0, v24
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 33 40                         # k135114294089648.py:33:40
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	vsetvli	zero, s3, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v8, v24, 16
	vsetvli	zero, s7, e16, m8, ta, ma
	vslideup.vx	v16, v8, s3
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e16, m4, ta, ma
	vnsrl.wi	v24, v8, 16
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vnsrl.wi	v8, v0, 16
	vsetvli	zero, s7, e16, m8, ta, ma
	vslideup.vx	v24, v8, s3
	lui	a0, 4
	addi	a0, a0, -1280
	add	a0, sp, a0
	li	a1, 15
	slli	a1, a1, 10
	add	a1, sp, a1
	vmv2r.v	v8, v24
	csrr	a2, vlenb
	li	a3, 185
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 7
	addi	a3, a3, -832
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vse16.v	v24, (a0)
	vmv2r.v	v8, v16
	csrr	a0, vlenb
	li	a2, 193
	mul	a0, a0, a2
	add	a0, sp, a0
	lui	a2, 7
	addi	a2, a2, -832
	add	a0, a0, a2
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vse16.v	v16, (a1)
	lh	a0, 1424(s8)
	sd	a0, 480(sp)                     # 8-byte Folded Spill
	lh	a0, 1426(s8)
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	lh	a0, 1428(s8)
	sd	a0, 496(sp)                     # 8-byte Folded Spill
	lh	a0, 1430(s8)
	sd	a0, 512(sp)                     # 8-byte Folded Spill
	lh	a0, 1416(s8)
	sd	a0, 448(sp)                     # 8-byte Folded Spill
	lh	a0, 1418(s8)
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	lh	a0, 1420(s8)
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	lh	a0, 1422(s8)
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	lh	s9, 1408(s8)
	lh	s2, 1410(s8)
	lh	s7, 1412(s8)
	lh	a0, 1414(s8)
	sd	a0, 432(sp)                     # 8-byte Folded Spill
	lh	s4, 1400(s8)
	lh	s3, 1402(s8)
	lh	s11, 1404(s8)
	lh	s10, 1406(s8)
	lh	a0, 1456(s8)
	sd	a0, 608(sp)                     # 8-byte Folded Spill
	lh	a0, 1458(s8)
	sd	a0, 616(sp)                     # 8-byte Folded Spill
	lh	a0, 1460(s8)
	sd	a0, 624(sp)                     # 8-byte Folded Spill
	lh	a0, 1462(s8)
	sd	a0, 640(sp)                     # 8-byte Folded Spill
	lh	a0, 1448(s8)
	sd	a0, 576(sp)                     # 8-byte Folded Spill
	lh	a0, 1450(s8)
	sd	a0, 584(sp)                     # 8-byte Folded Spill
	lh	a0, 1452(s8)
	sd	a0, 592(sp)                     # 8-byte Folded Spill
	lh	a0, 1454(s8)
	sd	a0, 600(sp)                     # 8-byte Folded Spill
	lh	a0, 1440(s8)
	sd	a0, 544(sp)                     # 8-byte Folded Spill
	lh	a0, 1442(s8)
	sd	a0, 552(sp)                     # 8-byte Folded Spill
	lh	a0, 1444(s8)
	sd	a0, 560(sp)                     # 8-byte Folded Spill
	lh	a0, 1446(s8)
	sd	a0, 568(sp)                     # 8-byte Folded Spill
	lh	a0, 1432(s8)
	sd	a0, 504(sp)                     # 8-byte Folded Spill
	lh	a0, 1434(s8)
	sd	a0, 520(sp)                     # 8-byte Folded Spill
	lh	a0, 1436(s8)
	sd	a0, 528(sp)                     # 8-byte Folded Spill
	lh	a0, 1438(s8)
	sd	a0, 536(sp)                     # 8-byte Folded Spill
	lh	a0, 1488(s8)
	sd	a0, 736(sp)                     # 8-byte Folded Spill
	lh	a0, 1490(s8)
	sd	a0, 744(sp)                     # 8-byte Folded Spill
	lh	a0, 1492(s8)
	sd	a0, 752(sp)                     # 8-byte Folded Spill
	lh	a0, 1494(s8)
	sd	a0, 760(sp)                     # 8-byte Folded Spill
	lh	a0, 1480(s8)
	sd	a0, 704(sp)                     # 8-byte Folded Spill
	lh	a0, 1482(s8)
	sd	a0, 712(sp)                     # 8-byte Folded Spill
	lh	a0, 1484(s8)
	sd	a0, 720(sp)                     # 8-byte Folded Spill
	lh	a0, 1486(s8)
	sd	a0, 728(sp)                     # 8-byte Folded Spill
	lh	a0, 1472(s8)
	sd	a0, 672(sp)                     # 8-byte Folded Spill
	lh	a0, 1474(s8)
	sd	a0, 680(sp)                     # 8-byte Folded Spill
	lh	a0, 1476(s8)
	sd	a0, 688(sp)                     # 8-byte Folded Spill
	lh	a0, 1478(s8)
	sd	a0, 696(sp)                     # 8-byte Folded Spill
	lh	a0, 1464(s8)
	sd	a0, 632(sp)                     # 8-byte Folded Spill
	lh	a0, 1466(s8)
	sd	a0, 648(sp)                     # 8-byte Folded Spill
	lh	a0, 1468(s8)
	sd	a0, 656(sp)                     # 8-byte Folded Spill
	lh	a0, 1470(s8)
	sd	a0, 664(sp)                     # 8-byte Folded Spill
	lh	a0, 1680(s8)
	sd	a0, 864(sp)                     # 8-byte Folded Spill
	lh	a0, 1682(s8)
	sd	a0, 872(sp)                     # 8-byte Folded Spill
	lh	a0, 1684(s8)
	sd	a0, 880(sp)                     # 8-byte Folded Spill
	lh	a0, 1686(s8)
	sd	a0, 896(sp)                     # 8-byte Folded Spill
	lh	a0, 1672(s8)
	sd	a0, 832(sp)                     # 8-byte Folded Spill
	lh	a0, 1674(s8)
	sd	a0, 840(sp)                     # 8-byte Folded Spill
	lh	a0, 1676(s8)
	sd	a0, 848(sp)                     # 8-byte Folded Spill
	lh	a0, 1678(s8)
	sd	a0, 856(sp)                     # 8-byte Folded Spill
	lh	a0, 1664(s8)
	sd	a0, 800(sp)                     # 8-byte Folded Spill
	lh	a0, 1666(s8)
	sd	a0, 808(sp)                     # 8-byte Folded Spill
	lh	a0, 1668(s8)
	sd	a0, 816(sp)                     # 8-byte Folded Spill
	lh	a0, 1670(s8)
	sd	a0, 824(sp)                     # 8-byte Folded Spill
	lh	a0, 1656(s8)
	sd	a0, 768(sp)                     # 8-byte Folded Spill
	lh	a0, 1658(s8)
	sd	a0, 776(sp)                     # 8-byte Folded Spill
	lh	a0, 1660(s8)
	sd	a0, 784(sp)                     # 8-byte Folded Spill
	lh	a0, 1662(s8)
	sd	a0, 792(sp)                     # 8-byte Folded Spill
	lh	a0, 1712(s8)
	sd	a0, 992(sp)                     # 8-byte Folded Spill
	lh	a0, 1714(s8)
	sd	a0, 1000(sp)                    # 8-byte Folded Spill
	lh	a0, 1716(s8)
	sd	a0, 1008(sp)                    # 8-byte Folded Spill
	lh	a0, 1718(s8)
	sd	a0, 1024(sp)                    # 8-byte Folded Spill
	lh	a0, 1704(s8)
	sd	a0, 960(sp)                     # 8-byte Folded Spill
	lh	a0, 1706(s8)
	sd	a0, 968(sp)                     # 8-byte Folded Spill
	lh	a0, 1708(s8)
	sd	a0, 976(sp)                     # 8-byte Folded Spill
	lh	a0, 1710(s8)
	sd	a0, 984(sp)                     # 8-byte Folded Spill
	lh	a0, 1696(s8)
	sd	a0, 928(sp)                     # 8-byte Folded Spill
	lh	a0, 1698(s8)
	sd	a0, 936(sp)                     # 8-byte Folded Spill
	lh	a0, 1700(s8)
	sd	a0, 944(sp)                     # 8-byte Folded Spill
	lh	a0, 1702(s8)
	sd	a0, 952(sp)                     # 8-byte Folded Spill
	lh	a0, 1688(s8)
	sd	a0, 888(sp)                     # 8-byte Folded Spill
	lh	a0, 1690(s8)
	sd	a0, 904(sp)                     # 8-byte Folded Spill
	lh	a0, 1692(s8)
	sd	a0, 912(sp)                     # 8-byte Folded Spill
	lh	a0, 1694(s8)
	sd	a0, 920(sp)                     # 8-byte Folded Spill
	lh	a0, 1744(s8)
	sd	a0, 1120(sp)                    # 8-byte Folded Spill
	lh	a0, 1746(s8)
	sd	a0, 1128(sp)                    # 8-byte Folded Spill
	lh	a0, 1748(s8)
	sd	a0, 1136(sp)                    # 8-byte Folded Spill
	lh	a0, 1750(s8)
	sd	a0, 1144(sp)                    # 8-byte Folded Spill
	lh	a0, 1736(s8)
	sd	a0, 1088(sp)                    # 8-byte Folded Spill
	lh	a0, 1738(s8)
	sd	a0, 1096(sp)                    # 8-byte Folded Spill
	lh	a0, 1740(s8)
	sd	a0, 1104(sp)                    # 8-byte Folded Spill
	lh	a0, 1742(s8)
	sd	a0, 1112(sp)                    # 8-byte Folded Spill
	lh	a0, 1728(s8)
	sd	a0, 1056(sp)                    # 8-byte Folded Spill
	lh	a0, 1730(s8)
	sd	a0, 1064(sp)                    # 8-byte Folded Spill
	lh	a0, 1732(s8)
	sd	a0, 1072(sp)                    # 8-byte Folded Spill
	lh	a0, 1734(s8)
	sd	a0, 1080(sp)                    # 8-byte Folded Spill
	lh	a0, 1720(s8)
	sd	a0, 1016(sp)                    # 8-byte Folded Spill
	lh	a0, 1722(s8)
	sd	a0, 1032(sp)                    # 8-byte Folded Spill
	lh	a0, 1724(s8)
	sd	a0, 1040(sp)                    # 8-byte Folded Spill
	lh	a0, 1726(s8)
	sd	a0, 1048(sp)                    # 8-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	add	a0, a0, a1
	ld	s8, -832(a0)                    # 8-byte Folded Reload
	andi	a0, s8, 1
	ld	s5, 440(sp)                     # 8-byte Folded Reload
	vadd.vx	v8, v8, s5
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_260
	j	.LBB0_441
.LBB0_260:                              # %else384
	andi	a0, s8, 2
	beqz	a0, .LBB0_261
	j	.LBB0_442
.LBB0_261:                              # %else387
	andi	a0, s8, 4
	beqz	a0, .LBB0_262
	j	.LBB0_443
.LBB0_262:                              # %else390
	andi	a0, s8, 8
	beqz	a0, .LBB0_263
	j	.LBB0_444
.LBB0_263:                              # %else393
	andi	a0, s8, 16
	beqz	a0, .LBB0_264
	j	.LBB0_445
.LBB0_264:                              # %else396
	andi	a0, s8, 32
	beqz	a0, .LBB0_265
	j	.LBB0_446
.LBB0_265:                              # %else399
	andi	a0, s8, 64
	beqz	a0, .LBB0_267
.LBB0_266:                              # %cond.store400
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
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
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 1416(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_267:                              # %else402
	andi	a0, s8, 128
	csrr	a1, vlenb
	li	a2, 105
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 0 0                           # k135114294089648.py:0
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 105
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	beqz	a0, .LBB0_269
# %bb.268:                              # %cond.store403
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
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
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 1296(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_269:                              # %else405
	andi	a0, s8, 256
	beqz	a0, .LBB0_271
# %bb.270:                              # %cond.store406
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
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
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 1176(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_271:                              # %else408
	andi	a0, s8, 512
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 105
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 0 0                           # k135114294089648.py:0
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v16, v8, v24
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	beqz	a0, .LBB0_273
# %bb.272:                              # %cond.store409
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 1056(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_273:                              # %else411
	andi	a0, s8, 1024
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 0 0                           # k135114294089648.py:0
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s6
	.loc	1 33 57                         # k135114294089648.py:33:57
	beqz	a0, .LBB0_275
# %bb.274:                              # %cond.store412
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 936(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_275:                              # %else414
	.loc	1 0 0                           # k135114294089648.py:0
	li	a0, 128
	vand.vx	v16, v8, a0
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 0                          # k135114294089648.py:33
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s8, 52
	csrr	a1, vlenb
	li	a2, 137
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 33 0                          # k135114294089648.py:33
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 137
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	bgez	a0, .LBB0_277
# %bb.276:                              # %cond.store415
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 816(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_277:                              # %else417
	slli	a0, s8, 51
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 0 0                           # k135114294089648.py:0
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	.loc	1 33 57                         # k135114294089648.py:33:57
	bgez	a0, .LBB0_279
# %bb.278:                              # %cond.store418
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 696(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_279:                              # %else420
	.loc	1 33 0                          # k135114294089648.py:33
	vsll.vi	v24, v8, 6
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s8, 50
	csrr	a1, vlenb
	li	a2, 105
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 137
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 33 0                          # k135114294089648.py:33
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v0, v16, v8
	.loc	1 33 57                         # k135114294089648.py:33:57
	bgez	a0, .LBB0_280
	j	.LBB0_447
.LBB0_280:                              # %else423
	slli	a0, s8, 49
	.loc	1 33 0                          # k135114294089648.py:33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	.loc	1 33 57                         # k135114294089648.py:33:57
	bgez	a0, .LBB0_281
	j	.LBB0_448
.LBB0_281:                              # %else426
	slli	a0, s8, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_283
.LBB0_282:                              # %cond.store427
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 0                          # k135114294089648.py:33
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	.loc	1 33 57                         # k135114294089648.py:33:57
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 336(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_283:                              # %else429
	slli	a0, s8, 47
	vadd.vx	v8, v8, s5
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_284
	j	.LBB0_449
.LBB0_284:                              # %else432
	slli	a0, s8, 46
	bgez	a0, .LBB0_285
	j	.LBB0_450
.LBB0_285:                              # %else435
	slli	a0, s8, 45
	li	s4, 128
	bgez	a0, .LBB0_286
	j	.LBB0_451
.LBB0_286:                              # %else438
	slli	a0, s8, 44
	li	s3, 32
	bgez	a0, .LBB0_287
	j	.LBB0_452
.LBB0_287:                              # %else441
	slli	a0, s8, 43
	bgez	a0, .LBB0_288
	j	.LBB0_453
.LBB0_288:                              # %else444
	slli	a0, s8, 42
	bgez	a0, .LBB0_289
	j	.LBB0_454
.LBB0_289:                              # %else447
	slli	a0, s8, 41
	lui	a1, 2
	addi	a1, a1, 1360
	add	s2, sp, a1
	bgez	a0, .LBB0_290
	j	.LBB0_455
.LBB0_290:                              # %else450
	slli	a0, s8, 40
	bgez	a0, .LBB0_291
	j	.LBB0_456
.LBB0_291:                              # %else453
	slli	a0, s8, 39
	li	s7, 56
	bgez	a0, .LBB0_293
.LBB0_292:                              # %cond.store454
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 448(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_293:                              # %else456
	slli	a0, s8, 38
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v24, v16, v8
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_295
# %bb.294:                              # %cond.store457
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 456(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_295:                              # %else459
	slli	a0, s8, 37
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v16, v8, s7
	bgez	a0, .LBB0_297
# %bb.296:                              # %cond.store460
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 464(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_297:                              # %else462
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 73
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsll.vi	v8, v8, 13
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s8, 36
	vsetivli	zero, 16, e64, m8, ta, ma
	vand.vx	v16, v16, s4
	bgez	a0, .LBB0_299
# %bb.298:                              # %cond.store463
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 472(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_299:                              # %else465
	slli	a0, s8, 35
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_301
# %bb.300:                              # %cond.store466
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 480(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1296(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_301:                              # %else468
	.loc	1 0 57                          # k135114294089648.py:0:57
	vsll.vi	v24, v8, 6
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s8, 34
	csrr	a1, vlenb
	li	a2, 113
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v0, v16, v8
	bgez	a0, .LBB0_302
	j	.LBB0_457
.LBB0_302:                              # %else471
	slli	a0, s8, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	bgez	a0, .LBB0_303
	j	.LBB0_458
.LBB0_303:                              # %else474
	slli	a0, s8, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_305
.LBB0_304:                              # %cond.store475
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 512(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 936(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_305:                              # %else477
	slli	a0, s8, 31
	vadd.vx	v8, v8, s5
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_306
	j	.LBB0_459
.LBB0_306:                              # %else480
	slli	a0, s8, 30
	bgez	a0, .LBB0_307
	j	.LBB0_460
.LBB0_307:                              # %else483
	slli	a0, s8, 29
	bgez	a0, .LBB0_308
	j	.LBB0_461
.LBB0_308:                              # %else486
	slli	a0, s8, 28
	bgez	a0, .LBB0_309
	j	.LBB0_462
.LBB0_309:                              # %else489
	slli	a0, s8, 27
	bgez	a0, .LBB0_310
	j	.LBB0_463
.LBB0_310:                              # %else492
	slli	a0, s8, 26
	bgez	a0, .LBB0_311
	j	.LBB0_464
.LBB0_311:                              # %else495
	slli	a0, s8, 25
	bgez	a0, .LBB0_313
.LBB0_312:                              # %cond.store496
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_313:                              # %else498
	slli	a0, s8, 24
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_315
# %bb.314:                              # %cond.store499
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_315:                              # %else501
	slli	a0, s8, 23
	bgez	a0, .LBB0_317
# %bb.316:                              # %cond.store502
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_317:                              # %else504
	slli	a0, s8, 22
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v16, v8, v24
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_319
# %bb.318:                              # %cond.store505
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_319:                              # %else507
	slli	a0, s8, 21
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s7
	bgez	a0, .LBB0_321
# %bb.320:                              # %cond.store508
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_321:                              # %else510
	.loc	1 0 57                          # k135114294089648.py:0:57
	vand.vx	v16, v8, s4
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s8, 20
	lui	a1, 2
	addi	a1, a1, -776
	add	s2, sp, a1
	bgez	a0, .LBB0_323
# %bb.322:                              # %cond.store511
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_323:                              # %else513
	slli	a0, s8, 19
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_325
# %bb.324:                              # %cond.store514
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_325:                              # %else516
	.loc	1 0 57                          # k135114294089648.py:0:57
	vsll.vi	v24, v8, 6
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s8, 18
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v0, v16, v8
	bgez	a0, .LBB0_326
	j	.LBB0_465
.LBB0_326:                              # %else519
	slli	a0, s8, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	bgez	a0, .LBB0_327
	j	.LBB0_466
.LBB0_327:                              # %else522
	slli	a0, s8, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_329
.LBB0_328:                              # %cond.store523
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 640(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_329:                              # %else525
	slli	a0, s8, 15
	vadd.vx	v8, v8, s5
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_330
	j	.LBB0_467
.LBB0_330:                              # %else528
	slli	a0, s8, 14
	bgez	a0, .LBB0_331
	j	.LBB0_468
.LBB0_331:                              # %else531
	slli	a0, s8, 13
	bgez	a0, .LBB0_332
	j	.LBB0_469
.LBB0_332:                              # %else534
	slli	a0, s8, 12
	bgez	a0, .LBB0_333
	j	.LBB0_470
.LBB0_333:                              # %else537
	slli	a0, s8, 11
	bgez	a0, .LBB0_334
	j	.LBB0_471
.LBB0_334:                              # %else540
	slli	a0, s8, 10
	bgez	a0, .LBB0_335
	j	.LBB0_472
.LBB0_335:                              # %else543
	slli	a0, s8, 9
	bgez	a0, .LBB0_336
	j	.LBB0_473
.LBB0_336:                              # %else546
	slli	a0, s8, 8
	bgez	a0, .LBB0_337
	j	.LBB0_474
.LBB0_337:                              # %else549
	slli	a0, s8, 7
	bgez	a0, .LBB0_339
.LBB0_338:                              # %cond.store550
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 704(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_339:                              # %else552
	slli	a0, s8, 6
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v24, v16, v8
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_341
# %bb.340:                              # %cond.store553
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 712(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_341:                              # %else555
	slli	a0, s8, 5
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v16, v8, s7
	bgez	a0, .LBB0_343
# %bb.342:                              # %cond.store556
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 720(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 31
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_343:                              # %else558
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsll.vi	v8, v8, 13
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s8, 4
	vsetivli	zero, 16, e64, m8, ta, ma
	vand.vx	v16, v16, s4
	bgez	a0, .LBB0_345
# %bb.344:                              # %cond.store559
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 728(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_345:                              # %else561
	slli	a0, s8, 3
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_347
# %bb.346:                              # %cond.store562
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 736(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	li	a0, 15
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_347:                              # %else564
	.loc	1 0 57                          # k135114294089648.py:0:57
	vsll.vi	v24, v8, 6
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s8, 2
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v0, v16, v8
	bgez	a0, .LBB0_349
# %bb.348:                              # %cond.store565
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 744(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_349:                              # %else567
	.loc	1 0 57                          # k135114294089648.py:0:57
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v16, v0, v24
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s8, 1
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl1r.v	v8, (a1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	bgez	a0, .LBB0_351
# %bb.350:                              # %cond.store568
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 752(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs1r.v	v24, (a0)                       # vscale x 8-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl1r.v	v24, (a0)                       # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 29
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_351:                              # %else570
	.loc	1 0 57                          # k135114294089648.py:0:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	.loc	1 33 57                         # k135114294089648.py:33:57
	vmv.x.s	s6, v24
	bgez	s8, .LBB0_353
# %bb.352:                              # %cond.store571
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 760(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_353:                              # %else573
	andi	a0, s6, 1
	vadd.vx	v8, v8, s5
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_354
	j	.LBB0_475
.LBB0_354:                              # %else576
	andi	a0, s6, 2
	beqz	a0, .LBB0_355
	j	.LBB0_476
.LBB0_355:                              # %else579
	andi	a0, s6, 4
	beqz	a0, .LBB0_356
	j	.LBB0_477
.LBB0_356:                              # %else582
	andi	a0, s6, 8
	beqz	a0, .LBB0_357
	j	.LBB0_478
.LBB0_357:                              # %else585
	andi	a0, s6, 16
	lui	a1, 1
	addi	a1, a1, 1208
	add	s2, sp, a1
	beqz	a0, .LBB0_358
	j	.LBB0_479
.LBB0_358:                              # %else588
	andi	a0, s6, 32
	beqz	a0, .LBB0_359
	j	.LBB0_480
.LBB0_359:                              # %else591
	andi	a0, s6, 64
	beqz	a0, .LBB0_361
.LBB0_360:                              # %cond.store592
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 27
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_361:                              # %else594
	andi	a0, s6, 128
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_363
# %bb.362:                              # %cond.store595
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_363:                              # %else597
	andi	a0, s6, 256
	beqz	a0, .LBB0_365
# %bb.364:                              # %cond.store598
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 13
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_365:                              # %else600
	andi	a0, s6, 512
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v16, v8, v24
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_367
# %bb.366:                              # %cond.store601
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_367:                              # %else603
	andi	a0, s6, 1024
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s7
	beqz	a0, .LBB0_369
# %bb.368:                              # %cond.store604
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1176(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_369:                              # %else606
	.loc	1 0 57                          # k135114294089648.py:0:57
	vand.vx	v16, v8, s4
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s6, 52
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_371
# %bb.370:                              # %cond.store607
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_371:                              # %else609
	slli	a0, s6, 51
	csrr	a1, vlenb
	li	a2, 169
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_373
# %bb.372:                              # %cond.store610
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 3
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 936(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_373:                              # %else612
	.loc	1 0 57                          # k135114294089648.py:0:57
	vsll.vi	v24, v8, 6
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s6, 50
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v0, v16, v8
	bgez	a0, .LBB0_374
	j	.LBB0_481
.LBB0_374:                              # %else615
	slli	a0, s6, 49
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	bgez	a0, .LBB0_375
	j	.LBB0_482
.LBB0_375:                              # %else618
	slli	a0, s6, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_377
.LBB0_376:                              # %cond.store619
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 576(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_377:                              # %else621
	slli	a0, s6, 47
	vadd.vx	v8, v8, s5
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_378
	j	.LBB0_483
.LBB0_378:                              # %else624
	slli	a0, s6, 46
	bgez	a0, .LBB0_379
	j	.LBB0_484
.LBB0_379:                              # %else627
	slli	a0, s6, 45
	bgez	a0, .LBB0_380
	j	.LBB0_485
.LBB0_380:                              # %else630
	slli	a0, s6, 44
	bgez	a0, .LBB0_381
	j	.LBB0_486
.LBB0_381:                              # %else633
	slli	a0, s6, 43
	bgez	a0, .LBB0_382
	j	.LBB0_487
.LBB0_382:                              # %else636
	slli	a0, s6, 42
	bgez	a0, .LBB0_383
	j	.LBB0_488
.LBB0_383:                              # %else639
	slli	a0, s6, 41
	bgez	a0, .LBB0_384
	j	.LBB0_489
.LBB0_384:                              # %else642
	slli	a0, s6, 40
	bgez	a0, .LBB0_385
	j	.LBB0_490
.LBB0_385:                              # %else645
	slli	a0, s6, 39
	addi	s2, sp, 2047
	addi	s2, s2, 1121
	bgez	a0, .LBB0_387
.LBB0_386:                              # %cond.store646
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 832(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_387:                              # %else648
	slli	a0, s6, 38
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v24, v16, v8
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_389
# %bb.388:                              # %cond.store649
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 840(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_389:                              # %else651
	slli	a0, s6, 37
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v16, v8, s7
	bgez	a0, .LBB0_391
# %bb.390:                              # %cond.store652
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 848(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 19
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_391:                              # %else654
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsll.vi	v8, v8, 13
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s6, 36
	vsetivli	zero, 16, e64, m8, ta, ma
	vand.vx	v16, v16, s4
	bgez	a0, .LBB0_393
# %bb.392:                              # %cond.store655
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 856(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_393:                              # %else657
	slli	a0, s6, 35
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_395
# %bb.394:                              # %cond.store658
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 864(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_395:                              # %else660
	.loc	1 0 57                          # k135114294089648.py:0:57
	vsll.vi	v24, v8, 6
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s6, 34
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v0, v16, v8
	bgez	a0, .LBB0_396
	j	.LBB0_491
.LBB0_396:                              # %else663
	slli	a0, s6, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	bgez	a0, .LBB0_397
	j	.LBB0_492
.LBB0_397:                              # %else666
	slli	a0, s6, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_399
.LBB0_398:                              # %cond.store667
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 896(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1176(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_399:                              # %else669
	slli	a0, s6, 31
	vadd.vx	v8, v8, s5
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_400
	j	.LBB0_493
.LBB0_400:                              # %else672
	slli	a0, s6, 30
	bgez	a0, .LBB0_401
	j	.LBB0_494
.LBB0_401:                              # %else675
	slli	a0, s6, 29
	bgez	a0, .LBB0_402
	j	.LBB0_495
.LBB0_402:                              # %else678
	slli	a0, s6, 28
	bgez	a0, .LBB0_403
	j	.LBB0_496
.LBB0_403:                              # %else681
	slli	a0, s6, 27
	bgez	a0, .LBB0_404
	j	.LBB0_497
.LBB0_404:                              # %else684
	slli	a0, s6, 26
	bgez	a0, .LBB0_405
	j	.LBB0_498
.LBB0_405:                              # %else687
	slli	a0, s6, 25
	bgez	a0, .LBB0_407
.LBB0_406:                              # %cond.store688
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 944(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1793
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_407:                              # %else690
	slli	a0, s6, 24
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_409
# %bb.408:                              # %cond.store691
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 952(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1665
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_409:                              # %else693
	slli	a0, s6, 23
	bgez	a0, .LBB0_411
# %bb.410:                              # %cond.store694
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 960(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1537
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_411:                              # %else696
	slli	a0, s6, 22
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_413
# %bb.412:                              # %cond.store697
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 968(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1409
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_413:                              # %else699
	slli	a0, s6, 21
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s7
	bgez	a0, .LBB0_415
# %bb.414:                              # %cond.store700
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 976(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1281
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_415:                              # %else702
	.loc	1 0 57                          # k135114294089648.py:0:57
	vand.vx	v16, v8, s4
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s6, 20
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_417
# %bb.416:                              # %cond.store703
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 984(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 1153
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_417:                              # %else705
	slli	a0, s6, 19
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_419
# %bb.418:                              # %cond.store706
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 992(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1025
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_419:                              # %else708
	.loc	1 0 57                          # k135114294089648.py:0:57
	vsll.vi	v24, v8, 6
	.loc	1 33 57                         # k135114294089648.py:33:57
	slli	a0, s6, 18
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.vv	v0, v16, v8
	bgez	a0, .LBB0_420
	j	.LBB0_499
.LBB0_420:                              # %else711
	slli	a0, s6, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	bgez	a0, .LBB0_421
	j	.LBB0_500
.LBB0_421:                              # %else714
	slli	a0, s6, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_423
.LBB0_422:                              # %cond.store715
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1024(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 641
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1288(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_423:                              # %else717
	slli	a0, s6, 15
	vadd.vx	v8, v8, s5
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_424
	j	.LBB0_501
.LBB0_424:                              # %else720
	slli	a0, s6, 14
	bgez	a0, .LBB0_425
	j	.LBB0_502
.LBB0_425:                              # %else723
	slli	a0, s6, 13
	bgez	a0, .LBB0_426
	j	.LBB0_503
.LBB0_426:                              # %else726
	slli	a0, s6, 12
	bgez	a0, .LBB0_427
	j	.LBB0_504
.LBB0_427:                              # %else729
	slli	a0, s6, 11
	bgez	a0, .LBB0_428
	j	.LBB0_505
.LBB0_428:                              # %else732
	slli	a0, s6, 10
	bgez	a0, .LBB0_429
	j	.LBB0_506
.LBB0_429:                              # %else735
	slli	a0, s6, 9
	bgez	a0, .LBB0_430
	j	.LBB0_507
.LBB0_430:                              # %else738
	slli	a0, s6, 8
	bgez	a0, .LBB0_431
	j	.LBB0_508
.LBB0_431:                              # %else741
	slli	a0, s6, 7
	bgez	a0, .LBB0_432
	j	.LBB0_509
.LBB0_432:                              # %else744
	slli	a0, s6, 6
	bgez	a0, .LBB0_433
	j	.LBB0_510
.LBB0_433:                              # %else747
	slli	a0, s6, 5
	bgez	a0, .LBB0_434
	j	.LBB0_511
.LBB0_434:                              # %else750
	slli	a0, s6, 4
	bgez	a0, .LBB0_435
	j	.LBB0_512
.LBB0_435:                              # %else753
	slli	a0, s6, 3
	bgez	a0, .LBB0_436
	j	.LBB0_513
.LBB0_436:                              # %else756
	slli	a0, s6, 2
	bgez	a0, .LBB0_437
	j	.LBB0_514
.LBB0_437:                              # %else759
	slli	a0, s6, 1
	bgez	a0, .LBB0_438
	j	.LBB0_515
.LBB0_438:                              # %else762
	bgez	s6, .LBB0_440
.LBB0_439:                              # %cond.store763
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1144(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1152
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1272(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_440:                              # %else765
	.loc	1 33 4 epilogue_begin           # k135114294089648.py:33:4
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
.LBB0_441:                              # %cond.store
	.cfi_restore_state
	.loc	1 0 4                           # k135114294089648.py:0:4
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s8, 2
	bnez	a0, .LBB0_442
	j	.LBB0_261
.LBB0_442:                              # %cond.store385
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s8, 4
	bnez	a0, .LBB0_443
	j	.LBB0_262
.LBB0_443:                              # %cond.store388
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s8, 8
	bnez	a0, .LBB0_444
	j	.LBB0_263
.LBB0_444:                              # %cond.store391
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s8, 16
	bnez	a0, .LBB0_445
	j	.LBB0_264
.LBB0_445:                              # %cond.store394
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
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
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 1656(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s8, 32
	bnez	a0, .LBB0_446
	j	.LBB0_265
.LBB0_446:                              # %cond.store397
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
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
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 1536(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s8, 64
	beqz	a0, .LBB0_516
	j	.LBB0_266
.LBB0_516:                              # %cond.store397
	j	.LBB0_267
.LBB0_447:                              # %cond.store421
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 57                         # k135114294089648.py:33:57
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 576(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 49
	.loc	1 33 0                          # k135114294089648.py:33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	.loc	1 33 57                         # k135114294089648.py:33:57
	bltz	a0, .LBB0_448
	j	.LBB0_281
.LBB0_448:                              # %cond.store424
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 33 0                          # k135114294089648.py:33
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	.loc	1 33 57                         # k135114294089648.py:33:57
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 177
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 456(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_517
	j	.LBB0_282
.LBB0_517:                              # %cond.store424
	j	.LBB0_283
.LBB0_449:                              # %cond.store430
	slli	s4, s4, 16
	fmv.w.x	fa0, s4
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 46
	bltz	a0, .LBB0_450
	j	.LBB0_285
.LBB0_450:                              # %cond.store433
	slli	s3, s3, 16
	fmv.w.x	fa0, s3
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 45
	li	s4, 128
	bltz	a0, .LBB0_451
	j	.LBB0_286
.LBB0_451:                              # %cond.store436
	slli	s11, s11, 16
	fmv.w.x	fa0, s11
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 44
	li	s3, 32
	bltz	a0, .LBB0_452
	j	.LBB0_287
.LBB0_452:                              # %cond.store439
	slli	s10, s10, 16
	fmv.w.x	fa0, s10
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 43
	bltz	a0, .LBB0_453
	j	.LBB0_288
.LBB0_453:                              # %cond.store442
	slli	s9, s9, 16
	fmv.w.x	fa0, s9
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 120(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 42
	bltz	a0, .LBB0_454
	j	.LBB0_289
.LBB0_454:                              # %cond.store445
	slli	s2, s2, 16
	fmv.w.x	fa0, s2
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 3
	addi	a0, a0, -600
	add	a0, sp, a0
	ld	a0, 0(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 41
	lui	a1, 2
	addi	a1, a1, 1360
	add	s2, sp, a1
	bltz	a0, .LBB0_455
	j	.LBB0_290
.LBB0_455:                              # %cond.store448
	.loc	1 0 57                          # k135114294089648.py:0:57
	slli	s7, s7, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, s7
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 40
	bltz	a0, .LBB0_456
	j	.LBB0_291
.LBB0_456:                              # %cond.store451
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 432(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 39
	li	s7, 56
	bgez	a0, .LBB0_518
	j	.LBB0_292
.LBB0_518:                              # %cond.store451
	j	.LBB0_293
.LBB0_457:                              # %cond.store469
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	bltz	a0, .LBB0_458
	j	.LBB0_303
.LBB0_458:                              # %cond.store472
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 496(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1056(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_519
	j	.LBB0_304
.LBB0_519:                              # %cond.store472
	j	.LBB0_305
.LBB0_459:                              # %cond.store478
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 504(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 30
	bltz	a0, .LBB0_460
	j	.LBB0_307
.LBB0_460:                              # %cond.store481
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 520(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 29
	bltz	a0, .LBB0_461
	j	.LBB0_308
.LBB0_461:                              # %cond.store484
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 528(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 28
	bltz	a0, .LBB0_462
	j	.LBB0_309
.LBB0_462:                              # %cond.store487
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 536(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 27
	bltz	a0, .LBB0_463
	j	.LBB0_310
.LBB0_463:                              # %cond.store490
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 544(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 26
	bltz	a0, .LBB0_464
	j	.LBB0_311
.LBB0_464:                              # %cond.store493
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 25
	bgez	a0, .LBB0_520
	j	.LBB0_312
.LBB0_520:                              # %cond.store493
	j	.LBB0_313
.LBB0_465:                              # %cond.store517
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	bltz	a0, .LBB0_466
	j	.LBB0_327
.LBB0_466:                              # %cond.store520
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_521
	j	.LBB0_328
.LBB0_521:                              # %cond.store520
	j	.LBB0_329
.LBB0_467:                              # %cond.store526
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 632(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 14
	bltz	a0, .LBB0_468
	j	.LBB0_331
.LBB0_468:                              # %cond.store529
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 648(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 13
	bltz	a0, .LBB0_469
	j	.LBB0_332
.LBB0_469:                              # %cond.store532
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 656(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 12
	bltz	a0, .LBB0_470
	j	.LBB0_333
.LBB0_470:                              # %cond.store535
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 664(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 11
	bltz	a0, .LBB0_471
	j	.LBB0_334
.LBB0_471:                              # %cond.store538
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 672(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1320(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 10
	bltz	a0, .LBB0_472
	j	.LBB0_335
.LBB0_472:                              # %cond.store541
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 680(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1200(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 9
	bltz	a0, .LBB0_473
	j	.LBB0_336
.LBB0_473:                              # %cond.store544
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 688(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 8
	bltz	a0, .LBB0_474
	j	.LBB0_337
.LBB0_474:                              # %cond.store547
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 696(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 7
	bgez	a0, .LBB0_522
	j	.LBB0_338
.LBB0_522:                              # %cond.store547
	j	.LBB0_339
.LBB0_475:                              # %cond.store574
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 2
	bnez	a0, .LBB0_476
	j	.LBB0_355
.LBB0_476:                              # %cond.store577
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 4
	bnez	a0, .LBB0_477
	j	.LBB0_356
.LBB0_477:                              # %cond.store580
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 8
	bnez	a0, .LBB0_478
	j	.LBB0_357
.LBB0_478:                              # %cond.store583
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 16
	lui	a1, 1
	addi	a1, a1, 1208
	add	s2, sp, a1
	bnez	a0, .LBB0_479
	j	.LBB0_358
.LBB0_479:                              # %cond.store586
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 7
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 32
	bnez	a0, .LBB0_480
	j	.LBB0_359
.LBB0_480:                              # %cond.store589
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 64
	beqz	a0, .LBB0_523
	j	.LBB0_360
.LBB0_523:                              # %cond.store589
	j	.LBB0_361
.LBB0_481:                              # %cond.store613
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 49
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	bltz	a0, .LBB0_482
	j	.LBB0_375
.LBB0_482:                              # %cond.store616
	.loc	1 0 57                          # k135114294089648.py:0:57
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 185
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 696(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_524
	j	.LBB0_376
.LBB0_524:                              # %cond.store616
	j	.LBB0_377
.LBB0_483:                              # %cond.store622
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 768(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 46
	bltz	a0, .LBB0_484
	j	.LBB0_379
.LBB0_484:                              # %cond.store625
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 776(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 45
	bltz	a0, .LBB0_485
	j	.LBB0_380
.LBB0_485:                              # %cond.store628
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 784(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 44
	bltz	a0, .LBB0_486
	j	.LBB0_381
.LBB0_486:                              # %cond.store631
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 792(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 43
	bltz	a0, .LBB0_487
	j	.LBB0_382
.LBB0_487:                              # %cond.store634
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 800(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 42
	bltz	a0, .LBB0_488
	j	.LBB0_383
.LBB0_488:                              # %cond.store637
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 808(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 41
	bltz	a0, .LBB0_489
	j	.LBB0_384
.LBB0_489:                              # %cond.store640
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 816(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 40
	bltz	a0, .LBB0_490
	j	.LBB0_385
.LBB0_490:                              # %cond.store643
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 824(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 39
	addi	s2, sp, 2047
	addi	s2, s2, 1121
	bgez	a0, .LBB0_525
	j	.LBB0_386
.LBB0_525:                              # %cond.store643
	j	.LBB0_387
.LBB0_491:                              # %cond.store661
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 872(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	bltz	a0, .LBB0_492
	j	.LBB0_397
.LBB0_492:                              # %cond.store664
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 880(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1296(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_526
	j	.LBB0_398
.LBB0_526:                              # %cond.store664
	j	.LBB0_399
.LBB0_493:                              # %cond.store670
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 888(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 30
	bltz	a0, .LBB0_494
	j	.LBB0_401
.LBB0_494:                              # %cond.store673
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 904(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 29
	bltz	a0, .LBB0_495
	j	.LBB0_402
.LBB0_495:                              # %cond.store676
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 912(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 28
	bltz	a0, .LBB0_496
	j	.LBB0_403
.LBB0_496:                              # %cond.store679
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 920(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 27
	bltz	a0, .LBB0_497
	j	.LBB0_404
.LBB0_497:                              # %cond.store682
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 928(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 26
	bltz	a0, .LBB0_498
	j	.LBB0_405
.LBB0_498:                              # %cond.store685
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 936(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1921
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 25
	bgez	a0, .LBB0_527
	j	.LBB0_406
.LBB0_527:                              # %cond.store685
	j	.LBB0_407
.LBB0_499:                              # %cond.store709
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1000(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 897
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1048(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v0, v24
	bltz	a0, .LBB0_500
	j	.LBB0_421
.LBB0_500:                              # %cond.store712
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1008(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 769
	csrr	a1, vlenb
	li	a2, 193
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -832
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1168(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_528
	j	.LBB0_422
.LBB0_528:                              # %cond.store712
	j	.LBB0_423
.LBB0_501:                              # %cond.store718
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1016(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 14
	bltz	a0, .LBB0_502
	j	.LBB0_425
.LBB0_502:                              # %cond.store721
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1032(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 13
	bltz	a0, .LBB0_503
	j	.LBB0_426
.LBB0_503:                              # %cond.store724
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1040(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 12
	bltz	a0, .LBB0_504
	j	.LBB0_427
.LBB0_504:                              # %cond.store727
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1048(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 11
	bltz	a0, .LBB0_505
	j	.LBB0_428
.LBB0_505:                              # %cond.store730
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1056(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 513
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1504(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 10
	bltz	a0, .LBB0_506
	j	.LBB0_429
.LBB0_506:                              # %cond.store733
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1064(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 385
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1624(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 9
	bltz	a0, .LBB0_507
	j	.LBB0_430
.LBB0_507:                              # %cond.store736
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1072(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 257
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1744(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 8
	bltz	a0, .LBB0_508
	j	.LBB0_431
.LBB0_508:                              # %cond.store739
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1080(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 129
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1864(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 7
	bltz	a0, .LBB0_509
	j	.LBB0_432
.LBB0_509:                              # %cond.store742
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1088(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1984(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 6
	bltz	a0, .LBB0_510
	j	.LBB0_433
.LBB0_510:                              # %cond.store745
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1096(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1920
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1992(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 5
	bltz	a0, .LBB0_511
	j	.LBB0_434
.LBB0_511:                              # %cond.store748
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1104(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1792
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1872(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 4
	bltz	a0, .LBB0_512
	j	.LBB0_435
.LBB0_512:                              # %cond.store751
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1112(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1664
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1752(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 3
	bltz	a0, .LBB0_513
	j	.LBB0_436
.LBB0_513:                              # %cond.store754
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1120(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1536
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1632(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 2
	bltz	a0, .LBB0_514
	j	.LBB0_437
.LBB0_514:                              # %cond.store757
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1128(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1408
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1512(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 1
	bltz	a0, .LBB0_515
	j	.LBB0_438
.LBB0_515:                              # %cond.store760
	.loc	1 0 57                          # k135114294089648.py:0:57
	ld	a0, 1136(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 33 57                         # k135114294089648.py:33:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1280
	csrr	a1, vlenb
	li	a2, 201
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -832
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1392(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	bgez	s6, .LBB0_529
	j	.LBB0_439
.LBB0_529:                              # %cond.store760
	j	.LBB0_440
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_zeros_6, .Lfunc_end0-triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_zeros_6
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
	.asciz	"k135114294089648.py"           # string offset=7 ; k135114294089648.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
