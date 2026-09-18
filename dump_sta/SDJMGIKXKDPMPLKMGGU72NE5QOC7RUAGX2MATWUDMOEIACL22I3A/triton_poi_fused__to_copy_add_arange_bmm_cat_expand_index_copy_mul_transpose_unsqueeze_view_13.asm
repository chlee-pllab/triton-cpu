	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_13 # -- Begin function triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_13
	.p2align	2
	.type	triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_13,@function
triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_13: # @triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_13
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114449662640.py"
	.loc	1 2 0                           # k135114449662640.py:2:0
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
	lui	a6, 3
	addi	a6, a6, 912
	sub	sp, sp, a6
	csrr	a6, vlenb
	li	t0, 160
	mul	a6, a6, t0
	sub	sp, sp, a6
	andi	sp, sp, -128
	mv	s5, a5
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114449662640.py:4:33
	slliw	a7, a7, 7
	li	s3, 32
	li	s4, 64
	li	a6, 96
	li	t0, 128
	.loc	1 5 23                          # k135114449662640.py:5:23
	vsetvli	zero, s3, e32, m8, ta, ma
	vid.v	v8
	.loc	1 12 30                         # k135114449662640.py:12:30
	slli	a5, a7, 1
	.loc	1 5 23                          # k135114449662640.py:5:23
	vor.vx	v16, v8, a7
	csrr	t1, vlenb
	slli	t1, t1, 7
	add	t1, sp, t1
	lui	t2, 4
	addi	t2, t2, -1344
	add	t1, t1, t2
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	vadd.vx	v16, v8, s4
	vadd.vx	v24, v8, a6
	vadd.vx	v8, v8, s3
	.loc	1 12 30                         # k135114449662640.py:12:30
	add	a1, a1, a5
	.loc	1 5 23                          # k135114449662640.py:5:23
	vor.vx	v0, v16, a7
	csrr	a6, vlenb
	li	t1, 104
	mul	a6, a6, t1
	add	a6, sp, a6
	lui	t1, 4
	addi	t1, t1, -1344
	add	a6, a6, t1
	vs8r.v	v0, (a6)                        # vscale x 64-byte Folded Spill
	vor.vx	v24, v24, a7
	csrr	a6, vlenb
	li	t1, 120
	mul	a6, a6, t1
	add	a6, sp, a6
	lui	t1, 4
	addi	t1, t1, -1344
	add	a6, a6, t1
	vs8r.v	v24, (a6)                       # vscale x 64-byte Folded Spill
	vor.vx	v8, v8, a7
	csrr	a6, vlenb
	li	t1, 96
	mul	a6, a6, t1
	add	a6, sp, a6
	lui	t1, 4
	addi	t1, t1, -1344
	add	a6, a6, t1
	vs8r.v	v8, (a6)                        # vscale x 64-byte Folded Spill
	.loc	1 12 35                         # k135114449662640.py:12:35
	addi	a6, a1, 128
	.loc	1 6 21                          # k135114449662640.py:6:21
	vmslt.vx	v16, v0, t0
	csrr	t1, vlenb
	li	t2, 56
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 4
	addi	t2, t2, -1344
	add	t1, t1, t2
	vs1r.v	v16, (t1)                       # vscale x 8-byte Folded Spill
	vmslt.vx	v17, v24, t0
	csrr	t1, vlenb
	li	t2, 144
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 4
	addi	t2, t2, -1344
	add	t1, t1, t2
	vs1r.v	v17, (t1)                       # vscale x 8-byte Folded Spill
	vmv1r.v	v0, v16
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v0, v17, 4
	csrr	t1, vlenb
	li	t2, 72
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 4
	addi	t2, t2, -1344
	add	t1, t1, t2
	vs1r.v	v0, (t1)                        # vscale x 8-byte Folded Spill
	csrr	t1, vlenb
	slli	t1, t1, 7
	add	t1, sp, t1
	lui	t2, 4
	addi	t2, t2, -1344
	add	t1, t1, t2
	vl8r.v	v24, (t1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vmslt.vx	v17, v24, t0
	csrr	t1, vlenb
	li	t2, 136
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 4
	addi	t2, t2, -1344
	add	t1, t1, t2
	vs1r.v	v17, (t1)                       # vscale x 8-byte Folded Spill
	vmslt.vx	v16, v8, t0
	csrr	t0, vlenb
	li	t1, 112
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 4
	addi	t1, t1, -1344
	add	t0, t0, t1
	vs1r.v	v16, (t0)                       # vscale x 8-byte Folded Spill
	vmv1r.v	v24, v17
	.loc	1 12 35                         # k135114449662640.py:12:35
	vsetvli	zero, s4, e16, m8, ta, ma
	vmv.v.i	v8, 0
	.loc	1 6 21                          # k135114449662640.py:6:21
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v24, v16, 4
	csrr	t0, vlenb
	slli	t0, t0, 6
	add	t0, sp, t0
	lui	t1, 4
	addi	t1, t1, -1344
	add	t0, t0, t1
	vs1r.v	v24, (t0)                       # vscale x 8-byte Folded Spill
	.loc	1 12 35                         # k135114449662640.py:12:35
	vsetvli	zero, s4, e16, m8, ta, mu
	vle16.v	v8, (a6), v0.t
	vmv.v.i	v16, 0
	vmv1r.v	v0, v24
	vle16.v	v16, (a1), v0.t
	li	a6, -32
	li	a1, 4
	.loc	1 5 23                          # k135114449662640.py:5:23
	vsetvli	zero, s3, e32, m8, ta, ma
	vmv.v.x	v24, a7
	.loc	1 9 19                          # k135114449662640.py:9:19
	vsra.vi	v24, v24, 31
	csrr	a7, vlenb
	li	t0, 80
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	.loc	1 13 37                         # k135114449662640.py:13:37
	vsrl.vi	v24, v24, 27
	csrr	a7, vlenb
	li	t0, 152
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	csrr	a7, vlenb
	slli	a7, a7, 7
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	csrr	a7, vlenb
	li	t0, 152
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vl8r.v	v24, (a7)                       # vscale x 64-byte Folded Reload
	vadd.vv	v24, v0, v24
	vand.vx	v24, v24, a6
	vsub.vv	v24, v0, v24
	.loc	1 12 45                         # k135114449662640.py:12:45
	vzext.vf2	v0, v8
	csrr	a7, vlenb
	li	t0, 48
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vs8r.v	v0, (a7)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s3, e16, m8, ta, ma
	vslidedown.vx	v8, v8, s3
	csrr	a7, vlenb
	li	t0, 40
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	.loc	1 13 43                         # k135114449662640.py:13:43
	vsetivli	zero, 16, e32, m4, ta, ma
	vmv.v.i	v8, 0
	.loc	1 12 45                         # k135114449662640.py:12:45
	vsetvli	zero, s3, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a7, vlenb
	slli	a7, a7, 5
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vs8r.v	v0, (a7)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s3, e16, m8, ta, ma
	vslidedown.vx	v16, v16, s3
	csrr	a7, vlenb
	li	t0, 24
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vs8r.v	v16, (a7)                       # vscale x 64-byte Folded Spill
	.loc	1 13 43                         # k135114449662640.py:13:43
	vmv4r.v	v16, v8
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v8, v24, a1
	csrr	a7, vlenb
	li	t0, 136
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vl1r.v	v0, (a7)                        # vscale x 8-byte Folded Reload
	vluxei64.v	v16, (a2), v8, v0.t
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v24, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v0, v0, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v8, v24, a1
	csrr	a7, vlenb
	li	t0, 136
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vs8r.v	v8, (a7)                        # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	csrr	a7, vlenb
	li	t0, 136
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 4
	addi	t0, t0, -1344
	add	a7, a7, t0
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	vluxei64.v	v24, (a2), v8, v0.t
	.loc	1 14 20                         # k135114449662640.py:14:20
	lw	a7, 4(a3)
	.loc	1 13 43                         # k135114449662640.py:13:43
	vsetvli	zero, s3, e32, m8, ta, ma
	vslideup.vi	v16, v24, 16
	.loc	1 14 20                         # k135114449662640.py:14:20
	lwu	a3, 0(a3)
	slli	a7, a7, 32
	csrr	t0, vlenb
	li	t1, 152
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 4
	addi	t1, t1, -1344
	add	t0, t0, t1
	vl8r.v	v24, (t0)                       # vscale x 64-byte Folded Reload
	csrr	t0, vlenb
	li	t1, 120
	mul	t0, t0, t1
	add	t0, sp, t0
	lui	t1, 4
	addi	t1, t1, -1344
	add	t0, t0, t1
	vl8r.v	v0, (t0)                        # vscale x 64-byte Folded Reload
	.loc	1 13 37                         # k135114449662640.py:13:37
	vadd.vv	v24, v0, v24
	vand.vx	v24, v24, a6
	.loc	1 14 20                         # k135114449662640.py:14:20
	or	a3, a7, a3
	.loc	1 13 37                         # k135114449662640.py:13:37
	vsub.vv	v24, v0, v24
	.loc	1 25 21                         # k135114449662640.py:25:21
	fcvt.s.l	fa5, a3
	.loc	1 26 20                         # k135114449662640.py:26:20
	vfmul.vf	v16, v16, fa5
	csrr	a3, vlenb
	li	a7, 136
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vs8r.v	v16, (a3)                       # vscale x 64-byte Folded Spill
	.loc	1 13 43                         # k135114449662640.py:13:43
	vsetivli	zero, 16, e32, m4, ta, mu
	vmv.v.i	v16, 0
	vwmulsu.vx	v8, v24, a1
	csrr	a3, vlenb
	li	a7, 144
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vl1r.v	v0, (a3)                        # vscale x 8-byte Folded Reload
	vluxei64.v	v16, (a2), v8, v0.t
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v24, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v0, v0, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v24, v8, a1
	csrr	a3, vlenb
	li	a7, 144
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vs8r.v	v24, (a3)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	csrr	a3, vlenb
	li	a7, 144
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vl8r.v	v8, (a3)                        # vscale x 64-byte Folded Reload
	vluxei64.v	v24, (a2), v8, v0.t
	vsetvli	zero, s3, e32, m8, ta, ma
	vslideup.vi	v16, v24, 16
	csrr	a3, vlenb
	li	a7, 104
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vl8r.v	v0, (a3)                        # vscale x 64-byte Folded Reload
	csrr	a3, vlenb
	li	a7, 152
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vl8r.v	v24, (a3)                       # vscale x 64-byte Folded Reload
	.loc	1 13 37 is_stmt 0               # k135114449662640.py:13:37
	vadd.vv	v24, v0, v24
	vand.vx	v24, v24, a6
	vsub.vv	v0, v0, v24
	csrr	a3, vlenb
	slli	a3, a3, 4
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vs8r.v	v0, (a3)                        # vscale x 64-byte Folded Spill
	.loc	1 26 20 is_stmt 1               # k135114449662640.py:26:20
	vfmul.vf	v16, v16, fa5
	csrr	a3, vlenb
	li	a7, 144
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vs8r.v	v16, (a3)                       # vscale x 64-byte Folded Spill
	vsetivli	zero, 16, e32, m4, ta, mu
	vmv.v.i	v8, 0
	.loc	1 13 43                         # k135114449662640.py:13:43
	vmv.v.i	v24, 0
	vwmulsu.vx	v16, v0, a1
	csrr	a3, vlenb
	li	a7, 56
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vl1r.v	v0, (a3)                        # vscale x 8-byte Folded Reload
	vluxei64.v	v24, (a2), v16, v0.t
	csrr	a3, vlenb
	li	a7, 88
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vs8r.v	v24, (a3)                       # vscale x 64-byte Folded Spill
	csrr	a3, vlenb
	slli	a3, a3, 4
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vl8r.v	v16, (a3)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v16, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v0, v0, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v16, v24, a1
	csrr	a3, vlenb
	li	a7, 56
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vs8r.v	v16, (a3)                       # vscale x 64-byte Folded Spill
	vmv4r.v	v16, v8
	csrr	a3, vlenb
	li	a7, 56
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vl8r.v	v24, (a3)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v16, (a2), v24, v0.t
	csrr	a3, vlenb
	li	a7, 88
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vl8r.v	v24, (a3)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vslideup.vi	v24, v16, 16
	csrr	a3, vlenb
	li	a7, 88
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vs8r.v	v24, (a3)                       # vscale x 64-byte Folded Spill
	csrr	a3, vlenb
	li	a7, 96
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vl8r.v	v0, (a3)                        # vscale x 64-byte Folded Reload
	csrr	a3, vlenb
	li	a7, 152
	mul	a3, a3, a7
	add	a3, sp, a3
	lui	a7, 4
	addi	a7, a7, -1344
	add	a3, a3, a7
	vl8r.v	v16, (a3)                       # vscale x 64-byte Folded Reload
	.loc	1 13 37 is_stmt 0               # k135114449662640.py:13:37
	vadd.vv	v16, v0, v16
	vand.vx	v16, v16, a6
	vsub.vv	v16, v0, v16
	csrr	a3, vlenb
	li	a6, 152
	mul	a3, a3, a6
	add	a3, sp, a3
	lui	a6, 4
	addi	a6, a6, -1344
	add	a3, a3, a6
	vs8r.v	v16, (a3)                       # vscale x 64-byte Folded Spill
	.loc	1 13 43                         # k135114449662640.py:13:43
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v24, v16, a1
	csrr	a3, vlenb
	li	a6, 56
	mul	a3, a3, a6
	add	a3, sp, a3
	lui	a6, 4
	addi	a6, a6, -1344
	add	a3, a3, a6
	vs8r.v	v24, (a3)                       # vscale x 64-byte Folded Spill
	vmv4r.v	v16, v8
	csrr	a3, vlenb
	li	a6, 112
	mul	a3, a3, a6
	add	a3, sp, a3
	lui	a6, 4
	addi	a6, a6, -1344
	add	a3, a3, a6
	vl1r.v	v12, (a3)                       # vscale x 8-byte Folded Reload
	vmv1r.v	v0, v12
	csrr	a3, vlenb
	li	a6, 56
	mul	a3, a3, a6
	add	a3, sp, a3
	lui	a6, 4
	addi	a6, a6, -1344
	add	a3, a3, a6
	vl8r.v	v24, (a3)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v16, (a2), v24, v0.t
	csrr	a3, vlenb
	li	a6, 152
	mul	a3, a3, a6
	add	a3, sp, a3
	lui	a6, 4
	addi	a6, a6, -1344
	add	a3, a3, a6
	vl8r.v	v0, (a3)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v0, v0, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v12, v12, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v24, v0, a1
	vmv1r.v	v0, v12
	vluxei64.v	v8, (a2), v24, v0.t
	vsetvli	zero, s3, e32, m8, ta, ma
	vslideup.vi	v16, v8, 16
	li	a1, -64
	csrr	a2, vlenb
	li	a3, 80
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 9 19 is_stmt 1                # k135114449662640.py:9:19
	vsrl.vi	v8, v8, 26
	csrr	a2, vlenb
	li	a3, 112
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	csrr	a2, vlenb
	slli	a2, a2, 7
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v0, (a2)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v0, v8
	csrr	a2, vlenb
	li	a3, 56
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 8 19                          # k135114449662640.py:8:19
	vand.vx	v8, v8, a1
	vsub.vv	v8, v0, v8
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 16 31                         # k135114449662640.py:16:31
	add	a4, a4, a5
	.loc	1 16 36 is_stmt 0               # k135114449662640.py:16:36
	addi	a1, a4, 128
	csrr	a2, vlenb
	li	a3, 88
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 26 20 is_stmt 1               # k135114449662640.py:26:20
	vfmul.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 152
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s4, e16, m8, ta, mu
	vmv.v.i	v24, 0
	.loc	1 16 36                         # k135114449662640.py:16:36
	vmv.v.i	v8, 0
	csrr	a2, vlenb
	li	a3, 72
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vle16.v	v8, (a1), v0.t
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vle16.v	v24, (a4), v0.t
	vmv.v.v	v0, v24
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl1r.v	v24, (a1)                       # vscale x 8-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl1r.v	v25, (a1)                       # vscale x 8-byte Folded Reload
	.loc	1 6 21                          # k135114449662640.py:6:21
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v24, v25, 8
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs1r.v	v24, (a1)                       # vscale x 8-byte Folded Spill
	.loc	1 26 20                         # k135114449662640.py:26:20
	vsetvli	zero, s3, e32, m8, ta, ma
	vfmul.vf	v16, v16, fa5
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 16 46                         # k135114449662640.py:16:46
	vsetvli	zero, s3, e16, m8, ta, ma
	vslidedown.vx	v16, v8, s3
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s3, e32, m8, ta, ma
	vzext.vf2	v16, v8
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s3, e16, m8, ta, ma
	vslidedown.vx	v24, v0, s3
	vsetvli	zero, s3, e32, m8, ta, ma
	vzext.vf2	v8, v0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a1, sp, a1
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 10 19                         # k135114449662640.py:10:19
	lw	a1, 4(a0)
	lwu	a0, 0(a0)
	lui	a2, 3
	addi	a2, a2, 1192
	add	s9, sp, a2
	lui	a2, 3
	addi	a2, a2, -856
	add	s8, sp, a2
	slli	a2, a1, 32
	.loc	1 22 32                         # k135114449662640.py:22:32
	srliw	a1, a1, 31
	.loc	1 10 19                         # k135114449662640.py:10:19
	or	a0, a2, a0
	.loc	1 22 32                         # k135114449662640.py:22:32
	slli	a1, a1, 7
	csrr	a2, vlenb
	li	a3, 40
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 12 45                         # k135114449662640.py:12:45
	vzext.vf2	v8, v16
	csrr	a2, vlenb
	li	a3, 48
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 16
	csrr	a2, vlenb
	li	a3, 40
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	csrr	a2, vlenb
	li	a3, 24
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v0, (a2)                        # vscale x 64-byte Folded Reload
	vzext.vf2	v16, v0
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 48
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	csrr	a2, vlenb
	slli	a2, a2, 5
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	slli	a2, a2, 6
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vsll.vi	v8, v16, 16
	csrr	a2, vlenb
	li	a3, 72
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 16 46                         # k135114449662640.py:16:46
	vzext.vf2	v8, v16
	csrr	a2, vlenb
	slli	a2, a2, 4
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 16
	csrr	a2, vlenb
	slli	a2, a2, 4
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	vzext.vf2	v16, v24
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	lui	a2, 4
	addi	a2, a2, -1344
	add	a2, sp, a2
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 24
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vsll.vi	v8, v16, 16
	csrr	a2, vlenb
	slli	a2, a2, 5
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 22 32                         # k135114449662640.py:22:32
	add	s11, a1, a0
	lui	a0, 3
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 27 24                         # k135114449662640.py:27:24
	vse32.v	v8, (a0)
	flw	fa0, 84(s9)
	fsw	fa0, 888(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 852(s9)
	flw	fa0, 80(s9)
	fsw	fa0, 880(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 848(s9)
	flw	fa0, 76(s9)
	fsw	fa0, 872(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 844(s9)
	flw	fa0, 72(s9)
	fsw	fa0, 864(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 840(s9)
	flw	fa0, 68(s9)
	fsw	fa0, 856(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 836(s9)
	flw	fa0, 64(s9)
	fsw	fa0, 848(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 832(s9)
	flw	fa0, 60(s9)
	fsw	fa0, 840(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 828(s9)
	flw	fa0, 56(s9)
	fsw	fa0, 832(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 824(s9)
	flw	fa0, 52(s9)
	fsw	fa0, 824(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 820(s9)
	flw	fa0, 48(s9)
	fsw	fa0, 816(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 816(s9)
	flw	fa0, 44(s9)
	fsw	fa0, 808(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 812(s9)
	flw	fa0, 40(s9)
	fsw	fa0, 800(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 808(s9)
	flw	fa0, 36(s9)
	fsw	fa0, 792(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 804(s9)
	flw	fa0, 32(s9)
	fsw	fa0, 784(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 800(s9)
	flw	fa0, 28(s9)
	fsw	fa0, 776(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 796(s9)
	flw	fa0, 24(s9)
	fsw	fa0, 768(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 792(s9)
	flw	fa0, 20(s9)
	fsw	fa0, 760(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 788(s9)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 752(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 728(s9)
	flw	fa0, 16(s9)
	fsw	fa0, 744(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 784(s9)
	flw	fa0, 12(s9)
	fsw	fa0, 736(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 780(s9)
	flw	fa0, 8(s9)
	fsw	fa0, 728(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 776(s9)
	flw	fa0, 4(s9)
	fsw	fa0, 720(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 772(s9)
	flw	fa0, 0(s9)
	fsw	fa0, 712(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 768(s9)
	flw	fa0, 2044(s8)
	fsw	fa0, 704(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 764(s9)
	flw	fa0, 2040(s8)
	fsw	fa0, 696(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 760(s9)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 688(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 740(s9)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 680(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 736(s9)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 672(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 732(s9)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 664(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 756(s9)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 656(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 752(s9)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 648(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 748(s9)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 640(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 744(s9)
	lui	a0, 3
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 212(s9)
	fsw	fa0, 376(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1108(s9)
	flw	fa0, 208(s9)
	fsw	fa0, 368(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1104(s9)
	flw	fa0, 204(s9)
	fsw	fa0, 360(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1100(s9)
	flw	fa0, 200(s9)
	fsw	fa0, 352(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1096(s9)
	flw	fa0, 196(s9)
	fsw	fa0, 344(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1092(s9)
	flw	fa0, 192(s9)
	fsw	fa0, 336(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1088(s9)
	flw	fa0, 188(s9)
	fsw	fa0, 328(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1084(s9)
	flw	fa0, 184(s9)
	fsw	fa0, 320(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1080(s9)
	flw	fa0, 180(s9)
	fsw	fa0, 312(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1076(s9)
	flw	fa0, 176(s9)
	fsw	fa0, 304(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1072(s9)
	flw	fa0, 172(s9)
	fsw	fa0, 296(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1068(s9)
	flw	fa0, 168(s9)
	fsw	fa0, 288(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1064(s9)
	flw	fa0, 164(s9)
	fsw	fa0, 280(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1060(s9)
	flw	fa0, 160(s9)
	fsw	fa0, 272(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1056(s9)
	flw	fa0, 156(s9)
	fsw	fa0, 264(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1052(s9)
	flw	fa0, 152(s9)
	fsw	fa0, 256(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1048(s9)
	flw	fa0, 148(s9)
	fsw	fa0, 248(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1044(s9)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 240(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 984(s9)
	flw	fa0, 144(s9)
	fsw	fa0, 232(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1040(s9)
	flw	fa0, 140(s9)
	fsw	fa0, 224(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1036(s9)
	flw	fa0, 136(s9)
	fsw	fa0, 216(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1032(s9)
	flw	fa0, 132(s9)
	fsw	fa0, 208(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1028(s9)
	flw	fa0, 128(s9)
	fsw	fa0, 200(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1024(s9)
	flw	fa0, 124(s9)
	fsw	fa0, 192(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1020(s9)
	flw	fa0, 120(s9)
	fsw	fa0, 184(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1016(s9)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 176(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 996(s9)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 168(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 992(s9)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 164(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 988(s9)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 160(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1012(s9)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 156(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1008(s9)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 152(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1004(s9)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 148(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1000(s9)
	lui	a0, 3
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 1876(s8)
	fsw	fa0, 632(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 340(s9)
	flw	fa0, 1872(s8)
	fsw	fa0, 624(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 336(s9)
	flw	fa0, 1868(s8)
	fsw	fa0, 616(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 332(s9)
	flw	fa0, 1864(s8)
	fsw	fa0, 608(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 328(s9)
	flw	fa0, 1860(s8)
	fsw	fa0, 600(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 324(s9)
	flw	fa0, 1856(s8)
	fsw	fa0, 592(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 320(s9)
	flw	fa0, 1852(s8)
	fsw	fa0, 584(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 316(s9)
	flw	fa0, 1848(s8)
	fsw	fa0, 576(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 312(s9)
	flw	fa0, 1844(s8)
	fsw	fa0, 568(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 308(s9)
	flw	fa0, 1840(s8)
	fsw	fa0, 560(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 304(s9)
	flw	fa0, 1836(s8)
	fsw	fa0, 552(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 300(s9)
	flw	fa0, 1832(s8)
	fsw	fa0, 544(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 296(s9)
	flw	fa0, 1828(s8)
	fsw	fa0, 536(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 292(s9)
	flw	fa0, 1824(s8)
	fsw	fa0, 528(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 288(s9)
	flw	fa0, 1820(s8)
	fsw	fa0, 520(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 284(s9)
	flw	fa0, 1816(s8)
	fsw	fa0, 512(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 280(s9)
	flw	fa0, 1812(s8)
	fsw	fa0, 504(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 276(s9)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 496(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 216(s9)
	flw	fa0, 1808(s8)
	fsw	fa0, 488(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 272(s9)
	flw	fa0, 1804(s8)
	fsw	fa0, 480(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 268(s9)
	flw	fa0, 1800(s8)
	fsw	fa0, 472(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 264(s9)
	flw	fa0, 1796(s8)
	fsw	fa0, 464(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 260(s9)
	flw	fa0, 1792(s8)
	fsw	fa0, 456(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 256(s9)
	flw	fa0, 1788(s8)
	fsw	fa0, 448(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 252(s9)
	flw	fa0, 1784(s8)
	fsw	fa0, 440(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 248(s9)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 432(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 228(s9)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 424(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 224(s9)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 416(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 220(s9)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 408(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 244(s9)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 400(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 240(s9)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 392(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 236(s9)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 384(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 232(s9)
	li	a0, 13
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 2004(s8)
	fsw	fa0, 144(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 596(s9)
	flw	fa0, 2000(s8)
	fsw	fa0, 140(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 592(s9)
	flw	fa0, 1996(s8)
	fsw	fa0, 136(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 588(s9)
	flw	fa0, 1992(s8)
	fsw	fa0, 132(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 584(s9)
	flw	fa0, 1988(s8)
	fsw	fa0, 128(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 580(s9)
	flw	fa0, 1984(s8)
	fsw	fa0, 124(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 576(s9)
	flw	fa0, 1980(s8)
	fsw	fa0, 120(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 572(s9)
	flw	fa0, 1976(s8)
	fsw	fa0, 116(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 568(s9)
	flw	fa0, 1972(s8)
	fsw	fa0, 112(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 564(s9)
	flw	fa0, 1968(s8)
	fsw	fa0, 108(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 560(s9)
	flw	fa0, 1964(s8)
	fsw	fa0, 104(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 556(s9)
	flw	fa0, 1960(s8)
	fsw	fa0, 100(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 552(s9)
	flw	fa0, 1956(s8)
	fsw	fa0, 96(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 548(s9)
	flw	fa0, 1952(s8)
	fsw	fa0, 92(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 544(s9)
	flw	fa0, 1948(s8)
	fsw	fa0, 88(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 540(s9)
	flw	fa0, 1944(s8)
	fsw	fa0, 84(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 536(s9)
	flw	fa0, 1940(s8)
	fsw	fa0, 80(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 532(s9)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 76(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 472(s9)
	flw	fa0, 1936(s8)
	fsw	fa0, 72(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 528(s9)
	flw	fa0, 1932(s8)
	fsw	fa0, 68(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 524(s9)
	flw	fs8, 1928(s8)
	fmv.s	fa0, fs8
	call	cosf
	fsw	fa0, 520(s9)
	flw	fs9, 1924(s8)
	fmv.s	fa0, fs9
	call	cosf
	fsw	fa0, 516(s9)
	flw	fs10, 1920(s8)
	fmv.s	fa0, fs10
	call	cosf
	fsw	fa0, 512(s9)
	flw	fs11, 1916(s8)
	fmv.s	fa0, fs11
	call	cosf
	fsw	fa0, 508(s9)
	flw	fs0, 1912(s8)
	fmv.s	fa0, fs0
	call	cosf
	fsw	fa0, 504(s9)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fs1, v8
	fmv.s	fa0, fs1
	call	cosf
	fsw	fa0, 484(s9)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fs2, v8
	fmv.s	fa0, fs2
	call	cosf
	fsw	fa0, 480(s9)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fs3, v8
	fmv.s	fa0, fs3
	call	cosf
	fsw	fa0, 476(s9)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fs4, v8
	fmv.s	fa0, fs4
	call	cosf
	fsw	fa0, 500(s9)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fs5, v8
	fmv.s	fa0, fs5
	call	cosf
	fsw	fa0, 496(s9)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fs6, v8
	fmv.s	fa0, fs6
	call	cosf
	fsw	fa0, 492(s9)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fs7, v8
	fmv.s	fa0, fs7
	call	cosf
	fsw	fa0, 488(s9)
	lui	a0, 3
	addi	a0, a0, 1920
	add	a0, sp, a0
	vsetvli	zero, s3, e32, m8, ta, ma
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	lui	a0, 4
	addi	a0, a0, -1920
	add	a0, sp, a0
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	lui	a0, 3
	addi	a0, a0, 1408
	add	a0, sp, a0
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	lui	a0, 3
	addi	a0, a0, 1664
	add	a0, sp, a0
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	flw	fa0, 376(sp)                    # 4-byte Folded Reload
	.loc	1 30 24                         # k135114449662640.py:30:24
	call	sinf
	fsw	fa0, 1236(s9)
	flw	fa0, 368(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1232(s9)
	flw	fa0, 360(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1228(s9)
	flw	fa0, 352(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1224(s9)
	flw	fa0, 344(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1220(s9)
	flw	fa0, 336(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1216(s9)
	flw	fa0, 328(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1212(s9)
	flw	fa0, 320(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1208(s9)
	flw	fa0, 312(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1204(s9)
	flw	fa0, 304(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1200(s9)
	flw	fa0, 296(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1196(s9)
	flw	fa0, 288(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1192(s9)
	flw	fa0, 280(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1188(s9)
	flw	fa0, 272(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1184(s9)
	flw	fa0, 264(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1180(s9)
	flw	fa0, 256(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1176(s9)
	flw	fa0, 248(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1172(s9)
	flw	fa0, 240(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1112(s9)
	flw	fa0, 232(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1168(s9)
	flw	fa0, 224(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1164(s9)
	flw	fa0, 216(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1160(s9)
	flw	fa0, 208(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1156(s9)
	flw	fa0, 200(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1152(s9)
	flw	fa0, 192(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1148(s9)
	flw	fa0, 184(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1144(s9)
	flw	fa0, 176(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1124(s9)
	flw	fa0, 168(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1120(s9)
	flw	fa0, 164(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1116(s9)
	flw	fa0, 160(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1140(s9)
	flw	fa0, 156(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1136(s9)
	flw	fa0, 152(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1132(s9)
	flw	fa0, 148(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1128(s9)
	flw	fa0, 888(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 980(s9)
	flw	fa0, 880(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 976(s9)
	flw	fa0, 872(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 972(s9)
	flw	fa0, 864(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 968(s9)
	flw	fa0, 856(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 964(s9)
	flw	fa0, 848(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 960(s9)
	flw	fa0, 840(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 956(s9)
	flw	fa0, 832(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 952(s9)
	flw	fa0, 824(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 948(s9)
	flw	fa0, 816(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 944(s9)
	flw	fa0, 808(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 940(s9)
	flw	fa0, 800(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 936(s9)
	flw	fa0, 792(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 932(s9)
	flw	fa0, 784(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 928(s9)
	flw	fa0, 776(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 924(s9)
	flw	fa0, 768(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 920(s9)
	flw	fa0, 760(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 916(s9)
	flw	fa0, 752(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 856(s9)
	flw	fa0, 744(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 912(s9)
	flw	fa0, 736(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 908(s9)
	flw	fa0, 728(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 904(s9)
	flw	fa0, 720(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 900(s9)
	flw	fa0, 712(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 896(s9)
	flw	fa0, 704(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 892(s9)
	flw	fa0, 696(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 888(s9)
	flw	fa0, 688(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 868(s9)
	flw	fa0, 680(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 864(s9)
	flw	fa0, 672(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 860(s9)
	flw	fa0, 664(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 884(s9)
	flw	fa0, 656(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 880(s9)
	flw	fa0, 648(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 876(s9)
	flw	fa0, 640(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 872(s9)
	flw	fa0, 144(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 724(s9)
	flw	fa0, 140(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 720(s9)
	flw	fa0, 136(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 716(s9)
	flw	fa0, 132(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 712(s9)
	flw	fa0, 128(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 708(s9)
	flw	fa0, 124(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 704(s9)
	flw	fa0, 120(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 700(s9)
	flw	fa0, 116(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 696(s9)
	flw	fa0, 112(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 692(s9)
	flw	fa0, 108(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 688(s9)
	flw	fa0, 104(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 684(s9)
	flw	fa0, 100(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 680(s9)
	flw	fa0, 96(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 676(s9)
	flw	fa0, 92(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 672(s9)
	flw	fa0, 88(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 668(s9)
	flw	fa0, 84(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 664(s9)
	flw	fa0, 80(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 660(s9)
	flw	fa0, 76(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 600(s9)
	flw	fa0, 72(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 656(s9)
	flw	fa0, 68(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 652(s9)
	fmv.s	fa0, fs8
	call	sinf
	fsw	fa0, 648(s9)
	fmv.s	fa0, fs9
	call	sinf
	fsw	fa0, 644(s9)
	fmv.s	fa0, fs10
	call	sinf
	fsw	fa0, 640(s9)
	fmv.s	fa0, fs11
	call	sinf
	fsw	fa0, 636(s9)
	fmv.s	fa0, fs0
	call	sinf
	fsw	fa0, 632(s9)
	fmv.s	fa0, fs1
	call	sinf
	fsw	fa0, 612(s9)
	fmv.s	fa0, fs2
	call	sinf
	fsw	fa0, 608(s9)
	fmv.s	fa0, fs3
	call	sinf
	fsw	fa0, 604(s9)
	fmv.s	fa0, fs4
	call	sinf
	fsw	fa0, 628(s9)
	fmv.s	fa0, fs5
	call	sinf
	fsw	fa0, 624(s9)
	fmv.s	fa0, fs6
	call	sinf
	fsw	fa0, 620(s9)
	fmv.s	fa0, fs7
	call	sinf
	fsw	fa0, 616(s9)
	flw	fa0, 632(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 468(s9)
	flw	fa0, 624(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 464(s9)
	flw	fa0, 616(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 460(s9)
	flw	fa0, 608(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 456(s9)
	flw	fa0, 600(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 452(s9)
	flw	fa0, 592(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 448(s9)
	flw	fa0, 584(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 444(s9)
	flw	fa0, 576(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 440(s9)
	flw	fa0, 568(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 436(s9)
	flw	fa0, 560(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 432(s9)
	flw	fa0, 552(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 428(s9)
	flw	fa0, 544(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 424(s9)
	flw	fa0, 536(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 420(s9)
	flw	fa0, 528(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 416(s9)
	flw	fa0, 520(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 412(s9)
	flw	fa0, 512(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 408(s9)
	flw	fa0, 504(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 404(s9)
	flw	fa0, 496(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 344(s9)
	flw	fa0, 488(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 400(s9)
	flw	fa0, 480(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 396(s9)
	flw	fa0, 472(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 392(s9)
	flw	fa0, 464(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 388(s9)
	flw	fa0, 456(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 384(s9)
	flw	fa0, 448(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 380(s9)
	flw	fa0, 440(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 376(s9)
	flw	fa0, 432(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 356(s9)
	flw	fa0, 424(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 352(s9)
	flw	fa0, 416(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 348(s9)
	flw	fa0, 408(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 372(s9)
	flw	fa0, 400(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 368(s9)
	flw	fa0, 392(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 364(s9)
	flw	fa0, 384(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 360(s9)
	lui	a0, 3
	addi	a0, a0, 1792
	add	a0, sp, a0
	vsetvli	zero, s3, e32, m8, ta, ma
	vle32.v	v8, (a0)
	li	a0, 27
	slli	a0, a0, 9
	add	a0, sp, a0
	vle32.v	v24, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 20                         # k135114449662640.py:32:20
	vfmul.vv	v0, v8, v16
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 33 20                         # k135114449662640.py:33:20
	vfmacc.vv	v0, v8, v16
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 32 20                         # k135114449662640.py:32:20
	vfmul.vv	v0, v24, v8
	lui	a0, 4
	addi	a0, a0, -1792
	add	a0, sp, a0
	.loc	1 30 24                         # k135114449662640.py:30:24
	vle32.v	v24, (a0)
	li	a0, 7
	slli	a0, a0, 11
	add	a0, sp, a0
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 33 20                         # k135114449662640.py:33:20
	vfmacc.vv	v0, v8, v16
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 32 20                         # k135114449662640.py:32:20
	vfmul.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 33 20                         # k135114449662640.py:33:20
	vfmacc.vv	v24, v8, v16
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 20                         # k135114449662640.py:32:20
	vfmul.vv	v24, v16, v8
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 33 20                         # k135114449662640.py:33:20
	vfmacc.vv	v24, v8, v16
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v8, v0, 16
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v0, v16, 16
	vsetvli	zero, s4, e16, m8, ta, ma
	vslideup.vx	v0, v8, s3
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 34 33 is_stmt 0               # k135114449662640.py:34:33
	slli	s11, s11, 6
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 45                         # k135114449662640.py:34:45
	vsetvli	zero, s3, e32, m8, ta, ma
	vsll.vi	v8, v8, 7
	lui	a2, 1048574
	lui	a0, 4
	addi	a0, a0, -1664
	add	a0, sp, a0
	li	a1, 29
	slli	a1, a1, 9
	add	a1, sp, a1
	vand.vx	v8, v8, a2
	csrr	a2, vlenb
	slli	a2, a2, 7
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v8, v24, 16
	csrr	a2, vlenb
	li	a3, 144
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, s4, e16, m8, ta, ma
	vslideup.vx	v24, v8, s3
	csrr	a2, vlenb
	li	a3, 80
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	csrr	a2, vlenb
	slli	a2, a2, 7
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v0, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 34 30                         # k135114449662640.py:34:30
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v0, v16
	.loc	1 34 40                         # k135114449662640.py:34:40
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v8, s11
	csrr	a2, vlenb
	li	a3, 152
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 4
	addi	a3, a3, -1344
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetvli	zero, s4, e16, m8, ta, ma
	vse16.v	v16, (a0)
	vmv2r.v	v16, v24
	csrr	a0, vlenb
	li	a2, 144
	mul	a0, a0, a2
	add	a0, sp, a0
	lui	a2, 4
	addi	a2, a2, -1344
	add	a0, a0, a2
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vse16.v	v24, (a1)
	lh	a0, 1296(s9)
	sd	a0, 608(sp)                     # 8-byte Folded Spill
	lh	a0, 1298(s9)
	sd	a0, 616(sp)                     # 8-byte Folded Spill
	lh	a0, 1300(s9)
	sd	a0, 624(sp)                     # 8-byte Folded Spill
	lh	a0, 1302(s9)
	sd	a0, 648(sp)                     # 8-byte Folded Spill
	lh	a0, 1288(s9)
	sd	a0, 576(sp)                     # 8-byte Folded Spill
	lh	a0, 1290(s9)
	sd	a0, 584(sp)                     # 8-byte Folded Spill
	lh	a0, 1292(s9)
	sd	a0, 592(sp)                     # 8-byte Folded Spill
	lh	a0, 1294(s9)
	sd	a0, 600(sp)                     # 8-byte Folded Spill
	lh	a0, 1280(s9)
	sd	a0, 544(sp)                     # 8-byte Folded Spill
	lh	a0, 1282(s9)
	sd	a0, 552(sp)                     # 8-byte Folded Spill
	lh	a0, 1284(s9)
	sd	a0, 560(sp)                     # 8-byte Folded Spill
	lh	a0, 1286(s9)
	sd	a0, 568(sp)                     # 8-byte Folded Spill
	lh	a0, 1272(s9)
	sd	a0, 512(sp)                     # 8-byte Folded Spill
	lh	a0, 1274(s9)
	sd	a0, 520(sp)                     # 8-byte Folded Spill
	lh	a0, 1276(s9)
	sd	a0, 528(sp)                     # 8-byte Folded Spill
	lh	a0, 1278(s9)
	sd	a0, 536(sp)                     # 8-byte Folded Spill
	lh	a0, 1424(s9)
	sd	a0, 224(sp)                     # 8-byte Folded Spill
	lh	a0, 1426(s9)
	sd	a0, 232(sp)                     # 8-byte Folded Spill
	lh	a0, 1428(s9)
	sd	a0, 240(sp)                     # 8-byte Folded Spill
	lh	a0, 1430(s9)
	sd	a0, 264(sp)                     # 8-byte Folded Spill
	lh	a0, 1416(s9)
	sd	a0, 192(sp)                     # 8-byte Folded Spill
	lh	a0, 1418(s9)
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	lh	a0, 1420(s9)
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	lh	a0, 1422(s9)
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	lh	s2, 1408(s9)
	lh	a0, 1410(s9)
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	lh	a0, 1412(s9)
	sd	a0, 176(sp)                     # 8-byte Folded Spill
	lh	a0, 1414(s9)
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	lh	s10, 1400(s9)
	lh	s6, 1402(s9)
	lh	s7, 1404(s9)
	lh	s4, 1406(s9)
	lh	a0, 1328(s9)
	sd	a0, 736(sp)                     # 8-byte Folded Spill
	lh	a0, 1330(s9)
	sd	a0, 744(sp)                     # 8-byte Folded Spill
	lh	a0, 1332(s9)
	sd	a0, 752(sp)                     # 8-byte Folded Spill
	lh	a0, 1334(s9)
	sd	a0, 776(sp)                     # 8-byte Folded Spill
	lh	a0, 1320(s9)
	sd	a0, 704(sp)                     # 8-byte Folded Spill
	lh	a0, 1322(s9)
	sd	a0, 712(sp)                     # 8-byte Folded Spill
	lh	a0, 1324(s9)
	sd	a0, 720(sp)                     # 8-byte Folded Spill
	lh	a0, 1326(s9)
	sd	a0, 728(sp)                     # 8-byte Folded Spill
	lh	a0, 1312(s9)
	sd	a0, 672(sp)                     # 8-byte Folded Spill
	lh	a0, 1314(s9)
	sd	a0, 680(sp)                     # 8-byte Folded Spill
	lh	a0, 1316(s9)
	sd	a0, 688(sp)                     # 8-byte Folded Spill
	lh	a0, 1318(s9)
	sd	a0, 696(sp)                     # 8-byte Folded Spill
	lh	a0, 1304(s9)
	sd	a0, 632(sp)                     # 8-byte Folded Spill
	lh	a0, 1306(s9)
	sd	a0, 640(sp)                     # 8-byte Folded Spill
	lh	a0, 1308(s9)
	sd	a0, 656(sp)                     # 8-byte Folded Spill
	lh	a0, 1310(s9)
	sd	a0, 664(sp)                     # 8-byte Folded Spill
	lh	a0, 1456(s9)
	sd	a0, 352(sp)                     # 8-byte Folded Spill
	lh	a0, 1458(s9)
	sd	a0, 360(sp)                     # 8-byte Folded Spill
	lh	a0, 1460(s9)
	sd	a0, 368(sp)                     # 8-byte Folded Spill
	lh	a0, 1462(s9)
	sd	a0, 392(sp)                     # 8-byte Folded Spill
	lh	a0, 1448(s9)
	sd	a0, 320(sp)                     # 8-byte Folded Spill
	lh	a0, 1450(s9)
	sd	a0, 328(sp)                     # 8-byte Folded Spill
	lh	a0, 1452(s9)
	sd	a0, 336(sp)                     # 8-byte Folded Spill
	lh	a0, 1454(s9)
	sd	a0, 344(sp)                     # 8-byte Folded Spill
	lh	a0, 1440(s9)
	sd	a0, 288(sp)                     # 8-byte Folded Spill
	lh	a0, 1442(s9)
	sd	a0, 296(sp)                     # 8-byte Folded Spill
	lh	a0, 1444(s9)
	sd	a0, 304(sp)                     # 8-byte Folded Spill
	lh	a0, 1446(s9)
	sd	a0, 312(sp)                     # 8-byte Folded Spill
	lh	a0, 1432(s9)
	sd	a0, 248(sp)                     # 8-byte Folded Spill
	lh	a0, 1434(s9)
	sd	a0, 256(sp)                     # 8-byte Folded Spill
	lh	a0, 1436(s9)
	sd	a0, 272(sp)                     # 8-byte Folded Spill
	lh	a0, 1438(s9)
	sd	a0, 280(sp)                     # 8-byte Folded Spill
	lh	a0, 1360(s9)
	sd	a0, 864(sp)                     # 8-byte Folded Spill
	lh	a0, 1362(s9)
	sd	a0, 872(sp)                     # 8-byte Folded Spill
	lh	a0, 1364(s9)
	sd	a0, 880(sp)                     # 8-byte Folded Spill
	lh	a0, 1366(s9)
	sd	a0, 888(sp)                     # 8-byte Folded Spill
	lh	a0, 1352(s9)
	sd	a0, 832(sp)                     # 8-byte Folded Spill
	lh	a0, 1354(s9)
	sd	a0, 840(sp)                     # 8-byte Folded Spill
	lh	a0, 1356(s9)
	sd	a0, 848(sp)                     # 8-byte Folded Spill
	lh	a0, 1358(s9)
	sd	a0, 856(sp)                     # 8-byte Folded Spill
	lh	a0, 1344(s9)
	sd	a0, 800(sp)                     # 8-byte Folded Spill
	lh	a0, 1346(s9)
	sd	a0, 808(sp)                     # 8-byte Folded Spill
	lh	a0, 1348(s9)
	sd	a0, 816(sp)                     # 8-byte Folded Spill
	lh	a0, 1350(s9)
	sd	a0, 824(sp)                     # 8-byte Folded Spill
	lh	a0, 1336(s9)
	sd	a0, 760(sp)                     # 8-byte Folded Spill
	lh	a0, 1338(s9)
	sd	a0, 768(sp)                     # 8-byte Folded Spill
	lh	a0, 1340(s9)
	sd	a0, 784(sp)                     # 8-byte Folded Spill
	lh	a0, 1342(s9)
	sd	a0, 792(sp)                     # 8-byte Folded Spill
	lh	a0, 1488(s9)
	sd	a0, 480(sp)                     # 8-byte Folded Spill
	lh	a0, 1490(s9)
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	lh	a0, 1492(s9)
	sd	a0, 496(sp)                     # 8-byte Folded Spill
	lh	a0, 1494(s9)
	sd	a0, 504(sp)                     # 8-byte Folded Spill
	lh	a0, 1480(s9)
	sd	a0, 448(sp)                     # 8-byte Folded Spill
	lh	a0, 1482(s9)
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	lh	a0, 1484(s9)
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	lh	a0, 1486(s9)
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	lh	a0, 1472(s9)
	sd	a0, 416(sp)                     # 8-byte Folded Spill
	lh	a0, 1474(s9)
	sd	a0, 424(sp)                     # 8-byte Folded Spill
	lh	a0, 1476(s9)
	sd	a0, 432(sp)                     # 8-byte Folded Spill
	lh	a0, 1478(s9)
	sd	a0, 440(sp)                     # 8-byte Folded Spill
	lh	a0, 1464(s9)
	sd	a0, 376(sp)                     # 8-byte Folded Spill
	lh	a0, 1466(s9)
	sd	a0, 384(sp)                     # 8-byte Folded Spill
	lh	a0, 1468(s9)
	sd	a0, 400(sp)                     # 8-byte Folded Spill
	lh	a0, 1470(s9)
	sd	a0, 408(sp)                     # 8-byte Folded Spill
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	add	a0, a0, a1
	ld	s9, -1344(a0)                   # 8-byte Folded Reload
	andi	a0, s9, 1
	vadd.vx	v8, v8, s5
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_1
	j	.LBB0_166
.LBB0_1:                                # %else
	andi	a0, s9, 2
	beqz	a0, .LBB0_2
	j	.LBB0_167
.LBB0_2:                                # %else2
	andi	a0, s9, 4
	beqz	a0, .LBB0_3
	j	.LBB0_168
.LBB0_3:                                # %else4
	andi	a0, s9, 8
	beqz	a0, .LBB0_4
	j	.LBB0_169
.LBB0_4:                                # %else6
	andi	a0, s9, 16
	beqz	a0, .LBB0_5
	j	.LBB0_170
.LBB0_5:                                # %else8
	andi	a0, s9, 32
	beqz	a0, .LBB0_6
	j	.LBB0_171
.LBB0_6:                                # %else10
	andi	a0, s9, 64
	beqz	a0, .LBB0_7
	j	.LBB0_172
.LBB0_7:                                # %else12
	andi	a0, s9, 128
	beqz	a0, .LBB0_8
	j	.LBB0_173
.LBB0_8:                                # %else14
	andi	a0, s9, 256
	beqz	a0, .LBB0_9
	j	.LBB0_174
.LBB0_9:                                # %else16
	andi	a0, s9, 512
	beqz	a0, .LBB0_10
	j	.LBB0_175
.LBB0_10:                               # %else18
	andi	a0, s9, 1024
	beqz	a0, .LBB0_12
.LBB0_11:                               # %cond.store19
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_12:                               # %else20
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 0                          # k135114449662640.py:34
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s9, 52
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 34 0                          # k135114449662640.py:34
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 34 57                         # k135114449662640.py:34:57
	bgez	a0, .LBB0_14
# %bb.13:                               # %cond.store21
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_14:                               # %else22
	slli	a0, s9, 51
	bgez	a0, .LBB0_16
# %bb.15:                               # %cond.store23
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_16:                               # %else24
	slli	a0, s9, 50
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 0                          # k135114449662640.py:34
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	.loc	1 34 57                         # k135114449662640.py:34:57
	bgez	a0, .LBB0_17
	j	.LBB0_176
.LBB0_17:                               # %else26
	slli	a0, s9, 49
	.loc	1 34 0                          # k135114449662640.py:34
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	.loc	1 34 57                         # k135114449662640.py:34:57
	bgez	a0, .LBB0_18
	j	.LBB0_177
.LBB0_18:                               # %else28
	slli	a0, s9, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_20
.LBB0_19:                               # %cond.store29
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 0                          # k135114449662640.py:34
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	.loc	1 34 57                         # k135114449662640.py:34:57
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 336(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_20:                               # %else30
	slli	a0, s9, 47
	vadd.vx	v8, v8, s5
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_21
	j	.LBB0_178
.LBB0_21:                               # %else32
	slli	a0, s9, 46
	bgez	a0, .LBB0_22
	j	.LBB0_179
.LBB0_22:                               # %else34
	slli	a0, s9, 45
	bgez	a0, .LBB0_23
	j	.LBB0_180
.LBB0_23:                               # %else36
	slli	a0, s9, 44
	lui	s6, 1048574
	bgez	a0, .LBB0_24
	j	.LBB0_181
.LBB0_24:                               # %else38
	slli	a0, s9, 43
	bgez	a0, .LBB0_25
	j	.LBB0_182
.LBB0_25:                               # %else40
	slli	a0, s9, 42
	mv	s4, s5
	bgez	a0, .LBB0_26
	j	.LBB0_183
.LBB0_26:                               # %else42
	slli	a0, s9, 41
	lui	a1, 2
	addi	a1, a1, 1104
	add	s2, sp, a1
	bgez	a0, .LBB0_27
	j	.LBB0_184
.LBB0_27:                               # %else44
	slli	a0, s9, 40
	li	s5, -64
	bgez	a0, .LBB0_28
	j	.LBB0_185
.LBB0_28:                               # %else46
	slli	a0, s9, 39
	bgez	a0, .LBB0_30
.LBB0_29:                               # %cond.store47
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_30:                               # %else48
	slli	a0, s9, 38
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_32
# %bb.31:                               # %cond.store49
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_32:                               # %else50
	.loc	1 0 57                          # k135114449662640.py:0:57
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v16, v8, s5
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s9, 37
	vsll.vi	v24, v8, 7
	bgez	a0, .LBB0_34
# %bb.33:                               # %cond.store51
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 21
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_34:                               # %else52
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s9, 36
	vand.vx	v8, v24, s6
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_36
# %bb.35:                               # %cond.store53
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 216(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_36:                               # %else54
	slli	a0, s9, 35
	bgez	a0, .LBB0_38
# %bb.37:                               # %cond.store55
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_38:                               # %else56
	slli	a0, s9, 34
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	bgez	a0, .LBB0_39
	j	.LBB0_186
.LBB0_39:                               # %else58
	slli	a0, s9, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	bgez	a0, .LBB0_40
	j	.LBB0_187
.LBB0_40:                               # %else60
	slli	a0, s9, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_42
.LBB0_41:                               # %cond.store61
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 264(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 936(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_42:                               # %else62
	slli	a0, s9, 31
	vadd.vx	v8, v8, s4
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_43
	j	.LBB0_188
.LBB0_43:                               # %else64
	slli	a0, s9, 30
	bgez	a0, .LBB0_44
	j	.LBB0_189
.LBB0_44:                               # %else66
	slli	a0, s9, 29
	bgez	a0, .LBB0_45
	j	.LBB0_190
.LBB0_45:                               # %else68
	slli	a0, s9, 28
	bgez	a0, .LBB0_46
	j	.LBB0_191
.LBB0_46:                               # %else70
	slli	a0, s9, 27
	bgez	a0, .LBB0_47
	j	.LBB0_192
.LBB0_47:                               # %else72
	slli	a0, s9, 26
	bgez	a0, .LBB0_48
	j	.LBB0_193
.LBB0_48:                               # %else74
	slli	a0, s9, 25
	bgez	a0, .LBB0_49
	j	.LBB0_194
.LBB0_49:                               # %else76
	slli	a0, s9, 24
	bgez	a0, .LBB0_50
	j	.LBB0_195
.LBB0_50:                               # %else78
	slli	a0, s9, 23
	bgez	a0, .LBB0_51
	j	.LBB0_196
.LBB0_51:                               # %else80
	slli	a0, s9, 22
	bgez	a0, .LBB0_52
	j	.LBB0_197
.LBB0_52:                               # %else82
	slli	a0, s9, 21
	bgez	a0, .LBB0_54
.LBB0_53:                               # %cond.store83
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 336(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_54:                               # %else84
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s9, 20
	lui	a1, 2
	addi	a1, a1, -1032
	add	s2, sp, a1
	bgez	a0, .LBB0_56
# %bb.55:                               # %cond.store85
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 344(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_56:                               # %else86
	slli	a0, s9, 19
	bgez	a0, .LBB0_58
# %bb.57:                               # %cond.store87
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 352(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_58:                               # %else88
	slli	a0, s9, 18
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	bgez	a0, .LBB0_59
	j	.LBB0_198
.LBB0_59:                               # %else90
	slli	a0, s9, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	bgez	a0, .LBB0_60
	j	.LBB0_199
.LBB0_60:                               # %else92
	slli	a0, s9, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_62
.LBB0_61:                               # %cond.store93
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 392(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_62:                               # %else94
	slli	a0, s9, 15
	vadd.vx	v8, v8, s4
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_63
	j	.LBB0_200
.LBB0_63:                               # %else96
	slli	a0, s9, 14
	bgez	a0, .LBB0_64
	j	.LBB0_201
.LBB0_64:                               # %else98
	slli	a0, s9, 13
	bgez	a0, .LBB0_65
	j	.LBB0_202
.LBB0_65:                               # %else100
	slli	a0, s9, 12
	bgez	a0, .LBB0_66
	j	.LBB0_203
.LBB0_66:                               # %else102
	slli	a0, s9, 11
	bgez	a0, .LBB0_67
	j	.LBB0_204
.LBB0_67:                               # %else104
	slli	a0, s9, 10
	bgez	a0, .LBB0_68
	j	.LBB0_205
.LBB0_68:                               # %else106
	slli	a0, s9, 9
	bgez	a0, .LBB0_69
	j	.LBB0_206
.LBB0_69:                               # %else108
	slli	a0, s9, 8
	bgez	a0, .LBB0_70
	j	.LBB0_207
.LBB0_70:                               # %else110
	slli	a0, s9, 7
	bgez	a0, .LBB0_72
.LBB0_71:                               # %cond.store111
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 448(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 31
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_72:                               # %else112
	slli	a0, s9, 6
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_74
# %bb.73:                               # %cond.store113
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 456(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 720(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_74:                               # %else114
	.loc	1 0 57                          # k135114449662640.py:0:57
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v16, v8, s5
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s9, 5
	vsll.vi	v24, v8, 7
	bgez	a0, .LBB0_76
# %bb.75:                               # %cond.store115
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 464(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 15
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_76:                               # %else116
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s9, 4
	vand.vx	v8, v24, s6
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_78
# %bb.77:                               # %cond.store117
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 472(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_78:                               # %else118
	slli	a0, s9, 3
	bgez	a0, .LBB0_80
# %bb.79:                               # %cond.store119
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 480(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 29
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_80:                               # %else120
	slli	a0, s9, 2
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	bgez	a0, .LBB0_82
# %bb.81:                               # %cond.store121
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_82:                               # %else122
	.loc	1 0 57                          # k135114449662640.py:0:57
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v24, s11
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s9, 1
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl1r.v	v8, (a1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	bgez	a0, .LBB0_84
# %bb.83:                               # %cond.store123
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 496(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs1r.v	v24, (a0)                       # vscale x 8-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl1r.v	v24, (a0)                       # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 7
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_84:                               # %else124
	.loc	1 0 57                          # k135114449662640.py:0:57
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	.loc	1 34 57                         # k135114449662640.py:34:57
	vmv.x.s	s8, v24
	bgez	s9, .LBB0_86
# %bb.85:                               # %cond.store125
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 504(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_86:                               # %else126
	andi	a0, s8, 1
	vadd.vx	v8, v8, s4
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_87
	j	.LBB0_208
.LBB0_87:                               # %else128
	andi	a0, s8, 2
	beqz	a0, .LBB0_88
	j	.LBB0_209
.LBB0_88:                               # %else130
	andi	a0, s8, 4
	beqz	a0, .LBB0_89
	j	.LBB0_210
.LBB0_89:                               # %else132
	andi	a0, s8, 8
	beqz	a0, .LBB0_90
	j	.LBB0_211
.LBB0_90:                               # %else134
	andi	a0, s8, 16
	lui	a1, 1
	addi	a1, a1, 952
	add	s2, sp, a1
	beqz	a0, .LBB0_91
	j	.LBB0_212
.LBB0_91:                               # %else136
	andi	a0, s8, 32
	beqz	a0, .LBB0_92
	j	.LBB0_213
.LBB0_92:                               # %else138
	andi	a0, s8, 64
	beqz	a0, .LBB0_93
	j	.LBB0_214
.LBB0_93:                               # %else140
	andi	a0, s8, 128
	beqz	a0, .LBB0_94
	j	.LBB0_215
.LBB0_94:                               # %else142
	andi	a0, s8, 256
	beqz	a0, .LBB0_95
	j	.LBB0_216
.LBB0_95:                               # %else144
	andi	a0, s8, 512
	beqz	a0, .LBB0_96
	j	.LBB0_217
.LBB0_96:                               # %else146
	andi	a0, s8, 1024
	beqz	a0, .LBB0_98
.LBB0_97:                               # %cond.store147
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 3
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_98:                               # %else148
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s8, 52
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_100
# %bb.99:                               # %cond.store149
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_100:                              # %else150
	slli	a0, s8, 51
	bgez	a0, .LBB0_102
# %bb.101:                              # %cond.store151
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_102:                              # %else152
	slli	a0, s8, 50
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	bgez	a0, .LBB0_103
	j	.LBB0_218
.LBB0_103:                              # %else154
	slli	a0, s8, 49
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	bgez	a0, .LBB0_104
	j	.LBB0_219
.LBB0_104:                              # %else156
	slli	a0, s8, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_106
.LBB0_105:                              # %cond.store157
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 576(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_106:                              # %else158
	slli	a0, s8, 47
	vadd.vx	v8, v8, s4
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_107
	j	.LBB0_220
.LBB0_107:                              # %else160
	slli	a0, s8, 46
	bgez	a0, .LBB0_108
	j	.LBB0_221
.LBB0_108:                              # %else162
	slli	a0, s8, 45
	bgez	a0, .LBB0_109
	j	.LBB0_222
.LBB0_109:                              # %else164
	slli	a0, s8, 44
	bgez	a0, .LBB0_110
	j	.LBB0_223
.LBB0_110:                              # %else166
	slli	a0, s8, 43
	bgez	a0, .LBB0_111
	j	.LBB0_224
.LBB0_111:                              # %else168
	slli	a0, s8, 42
	bgez	a0, .LBB0_112
	j	.LBB0_225
.LBB0_112:                              # %else170
	slli	a0, s8, 41
	bgez	a0, .LBB0_113
	j	.LBB0_226
.LBB0_113:                              # %else172
	slli	a0, s8, 40
	bgez	a0, .LBB0_114
	j	.LBB0_227
.LBB0_114:                              # %else174
	slli	a0, s8, 39
	addi	s2, sp, 2047
	addi	s2, s2, 865
	bgez	a0, .LBB0_116
.LBB0_115:                              # %cond.store175
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_116:                              # %else176
	slli	a0, s8, 38
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a0, .LBB0_118
# %bb.117:                              # %cond.store177
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_118:                              # %else178
	.loc	1 0 57                          # k135114449662640.py:0:57
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v16, v8, s5
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s8, 37
	vsll.vi	v8, v8, 7
	bgez	a0, .LBB0_120
# %bb.119:                              # %cond.store179
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 9
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_120:                              # %else180
	.loc	1 0 57                          # k135114449662640.py:0:57
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v8, v8, s6
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s8, 36
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsub.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_122
# %bb.121:                              # %cond.store181
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_122:                              # %else182
	slli	a0, s8, 35
	bgez	a0, .LBB0_124
# %bb.123:                              # %cond.store183
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_124:                              # %else184
	slli	a0, s8, 34
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v16
	bgez	a0, .LBB0_125
	j	.LBB0_228
.LBB0_125:                              # %else186
	slli	a0, s8, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	bgez	a0, .LBB0_126
	j	.LBB0_229
.LBB0_126:                              # %else188
	slli	a0, s8, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_128
.LBB0_127:                              # %cond.store189
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 648(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1921
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1176(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_128:                              # %else190
	slli	a0, s8, 31
	vadd.vx	v8, v8, s4
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_129
	j	.LBB0_230
.LBB0_129:                              # %else192
	slli	a0, s8, 30
	bgez	a0, .LBB0_130
	j	.LBB0_231
.LBB0_130:                              # %else194
	slli	a0, s8, 29
	bgez	a0, .LBB0_131
	j	.LBB0_232
.LBB0_131:                              # %else196
	slli	a0, s8, 28
	bgez	a0, .LBB0_132
	j	.LBB0_233
.LBB0_132:                              # %else198
	slli	a0, s8, 27
	bgez	a0, .LBB0_133
	j	.LBB0_234
.LBB0_133:                              # %else200
	slli	a0, s8, 26
	bgez	a0, .LBB0_134
	j	.LBB0_235
.LBB0_134:                              # %else202
	slli	a0, s8, 25
	bgez	a0, .LBB0_135
	j	.LBB0_236
.LBB0_135:                              # %else204
	slli	a0, s8, 24
	bgez	a0, .LBB0_136
	j	.LBB0_237
.LBB0_136:                              # %else206
	slli	a0, s8, 23
	bgez	a0, .LBB0_137
	j	.LBB0_238
.LBB0_137:                              # %else208
	slli	a0, s8, 22
	bgez	a0, .LBB0_138
	j	.LBB0_239
.LBB0_138:                              # %else210
	slli	a0, s8, 21
	bgez	a0, .LBB0_140
.LBB0_139:                              # %cond.store211
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 720(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1025
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_140:                              # %else212
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	a0, s8, 20
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_142
# %bb.141:                              # %cond.store213
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 728(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 897
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_142:                              # %else214
	slli	a0, s8, 19
	bgez	a0, .LBB0_144
# %bb.143:                              # %cond.store215
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 736(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 769
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_144:                              # %else216
	slli	a0, s8, 18
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	bgez	a0, .LBB0_145
	j	.LBB0_240
.LBB0_145:                              # %else218
	slli	a0, s8, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	bgez	a0, .LBB0_146
	j	.LBB0_241
.LBB0_146:                              # %else220
	slli	a0, s8, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_148
.LBB0_147:                              # %cond.store221
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 776(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 385
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1544(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_148:                              # %else222
	slli	a0, s8, 15
	vadd.vx	v8, v8, s4
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_149
	j	.LBB0_242
.LBB0_149:                              # %else224
	slli	a0, s8, 14
	bgez	a0, .LBB0_150
	j	.LBB0_243
.LBB0_150:                              # %else226
	slli	a0, s8, 13
	bgez	a0, .LBB0_151
	j	.LBB0_244
.LBB0_151:                              # %else228
	slli	a0, s8, 12
	bgez	a0, .LBB0_152
	j	.LBB0_245
.LBB0_152:                              # %else230
	slli	a0, s8, 11
	bgez	a0, .LBB0_153
	j	.LBB0_246
.LBB0_153:                              # %else232
	slli	a0, s8, 10
	bgez	a0, .LBB0_154
	j	.LBB0_247
.LBB0_154:                              # %else234
	slli	a0, s8, 9
	bgez	a0, .LBB0_155
	j	.LBB0_248
.LBB0_155:                              # %else236
	slli	a0, s8, 8
	bgez	a0, .LBB0_156
	j	.LBB0_249
.LBB0_156:                              # %else238
	slli	a0, s8, 7
	bgez	a0, .LBB0_157
	j	.LBB0_250
.LBB0_157:                              # %else240
	slli	a0, s8, 6
	bgez	a0, .LBB0_158
	j	.LBB0_251
.LBB0_158:                              # %else242
	slli	a0, s8, 5
	bgez	a0, .LBB0_159
	j	.LBB0_252
.LBB0_159:                              # %else244
	slli	a0, s8, 4
	bgez	a0, .LBB0_160
	j	.LBB0_253
.LBB0_160:                              # %else246
	slli	a0, s8, 3
	bgez	a0, .LBB0_161
	j	.LBB0_254
.LBB0_161:                              # %else248
	slli	a0, s8, 2
	bgez	a0, .LBB0_162
	j	.LBB0_255
.LBB0_162:                              # %else250
	slli	a0, s8, 1
	bgez	a0, .LBB0_163
	j	.LBB0_256
.LBB0_163:                              # %else252
	bgez	s8, .LBB0_165
.LBB0_164:                              # %cond.store253
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 888(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 896
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1016(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_165:                              # %else254
	.loc	1 34 4 epilogue_begin           # k135114449662640.py:34:4
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
.LBB0_166:                              # %cond.store
	.cfi_restore_state
	.loc	1 0 4                           # k135114449662640.py:0:4
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s9, 2
	bnez	a0, .LBB0_167
	j	.LBB0_2
.LBB0_167:                              # %cond.store1
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s9, 4
	bnez	a0, .LBB0_168
	j	.LBB0_3
.LBB0_168:                              # %cond.store3
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s9, 8
	bnez	a0, .LBB0_169
	j	.LBB0_4
.LBB0_169:                              # %cond.store5
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s9, 16
	bnez	a0, .LBB0_170
	j	.LBB0_5
.LBB0_170:                              # %cond.store7
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s9, 32
	bnez	a0, .LBB0_171
	j	.LBB0_6
.LBB0_171:                              # %cond.store9
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s9, 64
	bnez	a0, .LBB0_172
	j	.LBB0_7
.LBB0_172:                              # %cond.store11
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s9, 128
	bnez	a0, .LBB0_173
	j	.LBB0_8
.LBB0_173:                              # %cond.store13
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s9, 256
	bnez	a0, .LBB0_174
	j	.LBB0_9
.LBB0_174:                              # %cond.store15
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s9, 512
	bnez	a0, .LBB0_175
	j	.LBB0_10
.LBB0_175:                              # %cond.store17
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s9, 1024
	beqz	a0, .LBB0_257
	j	.LBB0_11
.LBB0_257:                              # %cond.store17
	j	.LBB0_12
.LBB0_176:                              # %cond.store25
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 49
	.loc	1 34 0                          # k135114449662640.py:34
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	.loc	1 34 57                         # k135114449662640.py:34:57
	bltz	a0, .LBB0_177
	j	.LBB0_18
.LBB0_177:                              # %cond.store27
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 34 0                          # k135114449662640.py:34
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	.loc	1 34 57                         # k135114449662640.py:34:57
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 456(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_258
	j	.LBB0_19
.LBB0_258:                              # %cond.store27
	j	.LBB0_20
.LBB0_178:                              # %cond.store31
	slli	s10, s10, 16
	fmv.w.x	fa0, s10
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 46
	bltz	a0, .LBB0_179
	j	.LBB0_22
.LBB0_179:                              # %cond.store33
	slli	s6, s6, 16
	fmv.w.x	fa0, s6
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 45
	bltz	a0, .LBB0_180
	j	.LBB0_23
.LBB0_180:                              # %cond.store35
	slli	s7, s7, 16
	fmv.w.x	fa0, s7
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 44
	lui	s6, 1048574
	bltz	a0, .LBB0_181
	j	.LBB0_24
.LBB0_181:                              # %cond.store37
	slli	s4, s4, 16
	fmv.w.x	fa0, s4
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 43
	bltz	a0, .LBB0_182
	j	.LBB0_25
.LBB0_182:                              # %cond.store39
	slli	s2, s2, 16
	fmv.w.x	fa0, s2
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 42
	mv	s4, s5
	bltz	a0, .LBB0_183
	j	.LBB0_26
.LBB0_183:                              # %cond.store41
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	s5, 168(sp)                     # 8-byte Folded Reload
	.loc	1 34 57                         # k135114449662640.py:34:57
	slli	s5, s5, 16
	fmv.w.x	fa0, s5
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s8)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 41
	lui	a1, 2
	addi	a1, a1, 1104
	add	s2, sp, a1
	bltz	a0, .LBB0_184
	j	.LBB0_27
.LBB0_184:                              # %cond.store43
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 40
	li	s5, -64
	bltz	a0, .LBB0_185
	j	.LBB0_28
.LBB0_185:                              # %cond.store45
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 39
	bgez	a0, .LBB0_259
	j	.LBB0_29
.LBB0_259:                              # %cond.store45
	j	.LBB0_30
.LBB0_186:                              # %cond.store57
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 232(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	bltz	a0, .LBB0_187
	j	.LBB0_40
.LBB0_187:                              # %cond.store59
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 240(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1056(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_260
	j	.LBB0_41
.LBB0_260:                              # %cond.store59
	j	.LBB0_42
.LBB0_188:                              # %cond.store63
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 248(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 30
	bltz	a0, .LBB0_189
	j	.LBB0_44
.LBB0_189:                              # %cond.store65
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 256(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 29
	bltz	a0, .LBB0_190
	j	.LBB0_45
.LBB0_190:                              # %cond.store67
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 272(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 28
	bltz	a0, .LBB0_191
	j	.LBB0_46
.LBB0_191:                              # %cond.store69
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 280(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 27
	bltz	a0, .LBB0_192
	j	.LBB0_47
.LBB0_192:                              # %cond.store71
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 288(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 26
	bltz	a0, .LBB0_193
	j	.LBB0_48
.LBB0_193:                              # %cond.store73
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 296(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 25
	bltz	a0, .LBB0_194
	j	.LBB0_49
.LBB0_194:                              # %cond.store75
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 304(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 24
	bltz	a0, .LBB0_195
	j	.LBB0_50
.LBB0_195:                              # %cond.store77
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 312(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 23
	bltz	a0, .LBB0_196
	j	.LBB0_51
.LBB0_196:                              # %cond.store79
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 320(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 22
	bltz	a0, .LBB0_197
	j	.LBB0_52
.LBB0_197:                              # %cond.store81
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 328(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 21
	bgez	a0, .LBB0_261
	j	.LBB0_53
.LBB0_261:                              # %cond.store81
	j	.LBB0_54
.LBB0_198:                              # %cond.store89
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 360(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	bltz	a0, .LBB0_199
	j	.LBB0_60
.LBB0_199:                              # %cond.store91
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 368(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_262
	j	.LBB0_61
.LBB0_262:                              # %cond.store91
	j	.LBB0_62
.LBB0_200:                              # %cond.store95
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 376(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 14
	bltz	a0, .LBB0_201
	j	.LBB0_64
.LBB0_201:                              # %cond.store97
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 384(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 13
	bltz	a0, .LBB0_202
	j	.LBB0_65
.LBB0_202:                              # %cond.store99
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 400(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 12
	bltz	a0, .LBB0_203
	j	.LBB0_66
.LBB0_203:                              # %cond.store101
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 408(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s9, 11
	bltz	a0, .LBB0_204
	j	.LBB0_67
.LBB0_204:                              # %cond.store103
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 416(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1320(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 10
	bltz	a0, .LBB0_205
	j	.LBB0_68
.LBB0_205:                              # %cond.store105
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 424(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1200(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 9
	bltz	a0, .LBB0_206
	j	.LBB0_69
.LBB0_206:                              # %cond.store107
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 432(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 8
	bltz	a0, .LBB0_207
	j	.LBB0_70
.LBB0_207:                              # %cond.store109
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 440(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s9, 7
	bgez	a0, .LBB0_263
	j	.LBB0_71
.LBB0_263:                              # %cond.store109
	j	.LBB0_72
.LBB0_208:                              # %cond.store127
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s8, 2
	bnez	a0, .LBB0_209
	j	.LBB0_88
.LBB0_209:                              # %cond.store129
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s8, 4
	bnez	a0, .LBB0_210
	j	.LBB0_89
.LBB0_210:                              # %cond.store131
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s8, 8
	bnez	a0, .LBB0_211
	j	.LBB0_90
.LBB0_211:                              # %cond.store133
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s8, 16
	lui	a1, 1
	addi	a1, a1, 952
	add	s2, sp, a1
	bnez	a0, .LBB0_212
	j	.LBB0_91
.LBB0_212:                              # %cond.store135
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 27
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s8, 32
	bnez	a0, .LBB0_213
	j	.LBB0_92
.LBB0_213:                              # %cond.store137
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s8, 64
	bnez	a0, .LBB0_214
	j	.LBB0_93
.LBB0_214:                              # %cond.store139
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 13
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s8, 128
	bnez	a0, .LBB0_215
	j	.LBB0_94
.LBB0_215:                              # %cond.store141
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s8, 256
	bnez	a0, .LBB0_216
	j	.LBB0_95
.LBB0_216:                              # %cond.store143
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s8, 512
	bnez	a0, .LBB0_217
	j	.LBB0_96
.LBB0_217:                              # %cond.store145
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s8, 1024
	beqz	a0, .LBB0_264
	j	.LBB0_97
.LBB0_264:                              # %cond.store145
	j	.LBB0_98
.LBB0_218:                              # %cond.store153
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 49
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	bltz	a0, .LBB0_219
	j	.LBB0_104
.LBB0_219:                              # %cond.store155
	.loc	1 0 57                          # k135114449662640.py:0:57
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 696(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_265
	j	.LBB0_105
.LBB0_265:                              # %cond.store155
	j	.LBB0_106
.LBB0_220:                              # %cond.store159
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 512(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 46
	bltz	a0, .LBB0_221
	j	.LBB0_108
.LBB0_221:                              # %cond.store161
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 520(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 45
	bltz	a0, .LBB0_222
	j	.LBB0_109
.LBB0_222:                              # %cond.store163
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 528(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 44
	bltz	a0, .LBB0_223
	j	.LBB0_110
.LBB0_223:                              # %cond.store165
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 536(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 43
	bltz	a0, .LBB0_224
	j	.LBB0_111
.LBB0_224:                              # %cond.store167
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 544(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 42
	bltz	a0, .LBB0_225
	j	.LBB0_112
.LBB0_225:                              # %cond.store169
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 41
	bltz	a0, .LBB0_226
	j	.LBB0_113
.LBB0_226:                              # %cond.store171
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 40
	bltz	a0, .LBB0_227
	j	.LBB0_114
.LBB0_227:                              # %cond.store173
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 39
	addi	s2, sp, 2047
	addi	s2, s2, 865
	bgez	a0, .LBB0_266
	j	.LBB0_115
.LBB0_266:                              # %cond.store173
	j	.LBB0_116
.LBB0_228:                              # %cond.store185
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	bltz	a0, .LBB0_229
	j	.LBB0_126
.LBB0_229:                              # %cond.store187
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	lui	a0, 1
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1296(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_267
	j	.LBB0_127
.LBB0_267:                              # %cond.store187
	j	.LBB0_128
.LBB0_230:                              # %cond.store191
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 632(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 30
	bltz	a0, .LBB0_231
	j	.LBB0_130
.LBB0_231:                              # %cond.store193
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 640(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 29
	bltz	a0, .LBB0_232
	j	.LBB0_131
.LBB0_232:                              # %cond.store195
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 656(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 28
	bltz	a0, .LBB0_233
	j	.LBB0_132
.LBB0_233:                              # %cond.store197
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 664(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 27
	bltz	a0, .LBB0_234
	j	.LBB0_133
.LBB0_234:                              # %cond.store199
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 672(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1793
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 26
	bltz	a0, .LBB0_235
	j	.LBB0_134
.LBB0_235:                              # %cond.store201
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 680(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1665
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 25
	bltz	a0, .LBB0_236
	j	.LBB0_135
.LBB0_236:                              # %cond.store203
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 688(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1537
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 24
	bltz	a0, .LBB0_237
	j	.LBB0_136
.LBB0_237:                              # %cond.store205
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 696(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1409
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 23
	bltz	a0, .LBB0_238
	j	.LBB0_137
.LBB0_238:                              # %cond.store207
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 704(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1281
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 22
	bltz	a0, .LBB0_239
	j	.LBB0_138
.LBB0_239:                              # %cond.store209
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 712(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1153
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s2)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 21
	bgez	a0, .LBB0_268
	j	.LBB0_139
.LBB0_268:                              # %cond.store209
	j	.LBB0_140
.LBB0_240:                              # %cond.store217
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 744(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 641
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1304(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s11
	bltz	a0, .LBB0_241
	j	.LBB0_146
.LBB0_241:                              # %cond.store219
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 752(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 513
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 4
	addi	a1, a1, -1344
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1424(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_269
	j	.LBB0_147
.LBB0_269:                              # %cond.store219
	j	.LBB0_148
.LBB0_242:                              # %cond.store223
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 760(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 14
	bltz	a0, .LBB0_243
	j	.LBB0_150
.LBB0_243:                              # %cond.store225
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 768(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 13
	bltz	a0, .LBB0_244
	j	.LBB0_151
.LBB0_244:                              # %cond.store227
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 784(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 12
	bltz	a0, .LBB0_245
	j	.LBB0_152
.LBB0_245:                              # %cond.store229
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 792(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s8, 11
	bltz	a0, .LBB0_246
	j	.LBB0_153
.LBB0_246:                              # %cond.store231
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 800(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 257
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1760(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 10
	bltz	a0, .LBB0_247
	j	.LBB0_154
.LBB0_247:                              # %cond.store233
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 808(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 129
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1880(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 9
	bltz	a0, .LBB0_248
	j	.LBB0_155
.LBB0_248:                              # %cond.store235
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 816(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -2000(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 8
	bltz	a0, .LBB0_249
	j	.LBB0_156
.LBB0_249:                              # %cond.store237
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 824(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1920
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1976(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 7
	bltz	a0, .LBB0_250
	j	.LBB0_157
.LBB0_250:                              # %cond.store239
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 832(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1792
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1856(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 6
	bltz	a0, .LBB0_251
	j	.LBB0_158
.LBB0_251:                              # %cond.store241
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 840(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1664
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1736(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 5
	bltz	a0, .LBB0_252
	j	.LBB0_159
.LBB0_252:                              # %cond.store243
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 848(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1536
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1616(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 4
	bltz	a0, .LBB0_253
	j	.LBB0_160
.LBB0_253:                              # %cond.store245
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 856(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1408
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1496(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 3
	bltz	a0, .LBB0_254
	j	.LBB0_161
.LBB0_254:                              # %cond.store247
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 864(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1280
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1376(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 2
	bltz	a0, .LBB0_255
	j	.LBB0_162
.LBB0_255:                              # %cond.store249
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 872(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1152
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1256(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s8, 1
	bltz	a0, .LBB0_256
	j	.LBB0_163
.LBB0_256:                              # %cond.store251
	.loc	1 0 57                          # k135114449662640.py:0:57
	ld	a0, 880(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	.loc	1 34 57                         # k135114449662640.py:34:57
	fmv.w.x	fa0, a0
	call	__truncsfbf2
	addi	a0, sp, 1024
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 4
	addi	a2, a2, -1344
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1136(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	bgez	s8, .LBB0_270
	j	.LBB0_164
.LBB0_270:                              # %cond.store251
	j	.LBB0_165
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_13, .Lfunc_end0-triton_poi_fused__to_copy_add_arange_bmm_cat_expand_index_copy_mul_transpose_unsqueeze_view_13
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
	.asciz	"k135114449662640.py"           # string offset=7 ; k135114449662640.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
