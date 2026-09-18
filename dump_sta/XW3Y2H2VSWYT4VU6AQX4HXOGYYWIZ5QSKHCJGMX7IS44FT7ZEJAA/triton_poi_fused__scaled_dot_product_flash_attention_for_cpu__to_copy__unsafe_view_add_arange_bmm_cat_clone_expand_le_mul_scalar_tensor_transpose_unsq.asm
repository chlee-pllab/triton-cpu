	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_7 # -- Begin function triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_7
	.p2align	2
	.type	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_7,@function
triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_7: # @triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_7
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135112023917040.py"
	.loc	1 2 0                           # k135112023917040.py:2:0
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
	fsd	fs0, 1976(sp)                   # 8-byte Folded Spill
	fsd	fs1, 1968(sp)                   # 8-byte Folded Spill
	fsd	fs2, 1960(sp)                   # 8-byte Folded Spill
	fsd	fs3, 1952(sp)                   # 8-byte Folded Spill
	fsd	fs4, 1944(sp)                   # 8-byte Folded Spill
	fsd	fs5, 1936(sp)                   # 8-byte Folded Spill
	fsd	fs6, 1928(sp)                   # 8-byte Folded Spill
	fsd	fs7, 1920(sp)                   # 8-byte Folded Spill
	fsd	fs8, 1912(sp)                   # 8-byte Folded Spill
	fsd	fs9, 1904(sp)                   # 8-byte Folded Spill
	fsd	fs10, 1896(sp)                  # 8-byte Folded Spill
	fsd	fs11, 1888(sp)                  # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s2, -24
	.cfi_offset s3, -32
	.cfi_offset s4, -40
	.cfi_offset s5, -48
	.cfi_offset fs0, -56
	.cfi_offset fs1, -64
	.cfi_offset fs2, -72
	.cfi_offset fs3, -80
	.cfi_offset fs4, -88
	.cfi_offset fs5, -96
	.cfi_offset fs6, -104
	.cfi_offset fs7, -112
	.cfi_offset fs8, -120
	.cfi_offset fs9, -128
	.cfi_offset fs10, -136
	.cfi_offset fs11, -144
	addi	s0, sp, 2032
	.cfi_def_cfa s0, 0
	addi	sp, sp, -272
	csrr	a4, vlenb
	li	a6, 113
	mul	a4, a4, a6
	sub	sp, sp, a4
	andi	sp, sp, -128
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135112023917040.py:4:33
	slliw	a5, a5, 7
	li	s2, 32
	li	s3, 64
	li	a6, 96
	li	a7, 896
	.loc	1 5 23                          # k135112023917040.py:5:23
	vsetvli	zero, s2, e32, m8, ta, ma
	vid.v	v16
	.loc	1 8 34                          # k135112023917040.py:8:34
	slli	a4, a5, 1
	.loc	1 5 23                          # k135112023917040.py:5:23
	vadd.vx	v8, v16, s3
	vadd.vx	v24, v16, a6
	.loc	1 8 34                          # k135112023917040.py:8:34
	add	s4, a0, a4
	.loc	1 5 23                          # k135112023917040.py:5:23
	vor.vx	v0, v8, a5
	csrr	a0, vlenb
	li	a6, 24
	mul	a0, a0, a6
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vor.vx	v24, v24, a5
	csrr	a0, vlenb
	li	a6, 89
	mul	a0, a0, a6
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135112023917040.py:6:21
	vmslt.vx	v8, v0, a7
	csrr	a0, vlenb
	li	a6, 81
	mul	a0, a0, a6
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs1r.v	v8, (a0)                        # vscale x 8-byte Folded Spill
	vmslt.vx	v9, v24, a7
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs1r.v	v9, (a0)                        # vscale x 8-byte Folded Spill
	vmv1r.v	v0, v8
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v0, v9, 4
	csrr	a0, vlenb
	li	a6, 73
	mul	a0, a0, a6
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs1r.v	v0, (a0)                        # vscale x 8-byte Folded Spill
	.loc	1 8 39                          # k135112023917040.py:8:39
	vsetvli	zero, s3, e16, m8, ta, mu
	vmv.v.i	v8, 0
	addi	s5, s4, 128
	vle16.v	v8, (s5), v0.t
	li	a0, -32
	.loc	1 5 23                          # k135112023917040.py:5:23
	vsetvli	zero, s2, e32, m8, ta, ma
	vor.vx	v24, v16, a5
	vadd.vx	v16, v16, s2
	vmv.v.x	v0, a5
	.loc	1 9 36                          # k135112023917040.py:9:36
	vsra.vi	v0, v0, 31
	.loc	1 5 23                          # k135112023917040.py:5:23
	vor.vx	v16, v16, a5
	csrr	a5, vlenb
	slli	a5, a5, 4
	add	a5, sp, a5
	addi	a5, a5, 2047
	addi	a5, a5, 113
	vs8r.v	v16, (a5)                       # vscale x 64-byte Folded Spill
	.loc	1 9 36                          # k135112023917040.py:9:36
	vsrl.vi	v0, v0, 27
	csrr	a5, vlenb
	li	a6, 105
	mul	a5, a5, a6
	add	a5, sp, a5
	addi	a5, a5, 2047
	addi	a5, a5, 113
	vs8r.v	v0, (a5)                        # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135112023917040.py:6:21
	vmslt.vx	v0, v24, a7
	addi	a5, sp, 2047
	addi	a5, a5, 113
	vs1r.v	v0, (a5)                        # vscale x 8-byte Folded Spill
	vmslt.vx	v7, v16, a7
	csrr	a5, vlenb
	li	a6, 40
	mul	a5, a5, a6
	add	a5, sp, a5
	addi	a5, a5, 2047
	addi	a5, a5, 113
	vs1r.v	v7, (a5)                        # vscale x 8-byte Folded Spill
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v0, v7, 4
	csrr	a5, vlenb
	li	a6, 72
	mul	a5, a5, a6
	add	a5, sp, a5
	addi	a5, a5, 2047
	addi	a5, a5, 113
	vs1r.v	v0, (a5)                        # vscale x 8-byte Folded Spill
	csrr	a5, vlenb
	li	a6, 105
	mul	a5, a5, a6
	add	a5, sp, a5
	addi	a5, a5, 2047
	addi	a5, a5, 113
	vl8r.v	v16, (a5)                       # vscale x 64-byte Folded Reload
	.loc	1 9 36                          # k135112023917040.py:9:36
	vsetvli	zero, s2, e32, m8, ta, ma
	vadd.vv	v16, v24, v16
	vand.vx	v16, v16, a0
	vsub.vv	v16, v24, v16
	csrr	a5, vlenb
	li	a6, 97
	mul	a5, a5, a6
	add	a5, sp, a5
	addi	a5, a5, 2047
	addi	a5, a5, 113
	vs8r.v	v16, (a5)                       # vscale x 64-byte Folded Spill
	.loc	1 8 39                          # k135112023917040.py:8:39
	vsetvli	zero, s3, e16, m8, ta, mu
	vmv.v.i	v16, 0
	vle16.v	v16, (s4), v0.t
	li	a5, 4
	.loc	1 8 49 is_stmt 0                # k135112023917040.py:8:49
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v24, v8
	csrr	a6, vlenb
	slli	a6, a6, 6
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 113
	vs8r.v	v24, (a6)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, s2
	csrr	a6, vlenb
	li	a7, 48
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 113
	vs8r.v	v8, (a6)                        # vscale x 64-byte Folded Spill
	.loc	1 9 42 is_stmt 1                # k135112023917040.py:9:42
	vsetivli	zero, 16, e32, m4, ta, ma
	vmv.v.i	v24, 0
	.loc	1 8 49                          # k135112023917040.py:8:49
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v8, v16
	csrr	a6, vlenb
	li	a7, 56
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 113
	vs8r.v	v8, (a6)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v8, v16, s2
	csrr	a6, vlenb
	slli	a6, a6, 5
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 113
	vs8r.v	v8, (a6)                        # vscale x 64-byte Folded Spill
	.loc	1 9 42                          # k135112023917040.py:9:42
	vmv4r.v	v16, v24
	csrr	a6, vlenb
	li	a7, 97
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 113
	vl8r.v	v0, (a6)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v8, v0, a5
	addi	a6, sp, 2047
	addi	a6, a6, 113
	vl1r.v	v0, (a6)                        # vscale x 8-byte Folded Reload
	vluxei64.v	v16, (a1), v8, v0.t
	csrr	a6, vlenb
	li	a7, 97
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 113
	vl8r.v	v8, (a6)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v0, v0, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v24, v8, a5
	csrr	a6, vlenb
	li	a7, 97
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 113
	vs8r.v	v24, (a6)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v8, 0
	csrr	a6, vlenb
	li	a7, 97
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 113
	vl8r.v	v24, (a6)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v8, (a1), v24, v0.t
	vsetvli	zero, s2, e32, m8, ta, ma
	vslideup.vi	v16, v8, 16
	.loc	1 10 19                         # k135112023917040.py:10:19
	lwu	a6, 0(a2)
	lw	a2, 4(a2)
	csrr	a7, vlenb
	li	t0, 105
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 2047
	addi	a7, a7, 113
	vl8r.v	v8, (a7)                        # vscale x 64-byte Folded Reload
	csrr	a7, vlenb
	li	t0, 89
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 2047
	addi	a7, a7, 113
	vl8r.v	v0, (a7)                        # vscale x 64-byte Folded Reload
	.loc	1 9 36                          # k135112023917040.py:9:36
	vadd.vv	v8, v0, v8
	vand.vx	v8, v8, a0
	vsub.vv	v8, v0, v8
	.loc	1 10 19                         # k135112023917040.py:10:19
	slli	a2, a2, 32
	or	a2, a2, a6
	.loc	1 15 19                         # k135112023917040.py:15:19
	fcvt.s.l	fa5, a2
	.loc	1 16 18                         # k135112023917040.py:16:18
	vfmul.vf	v16, v16, fa5
	csrr	a2, vlenb
	li	a6, 89
	mul	a2, a2, a6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 9 42                          # k135112023917040.py:9:42
	vsetivli	zero, 16, e32, m4, ta, mu
	vmv.v.i	v16, 0
	vwmulsu.vx	v24, v8, a5
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vluxei64.v	v16, (a1), v24, v0.t
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v8, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v0, v0, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v8, v24, a5
	csrr	a2, vlenb
	li	a6, 97
	mul	a2, a2, a6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vmv.v.i	v8, 0
	csrr	a2, vlenb
	li	a6, 97
	mul	a2, a2, a6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vl8r.v	v24, (a2)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v8, (a1), v24, v0.t
	vsetvli	zero, s2, e32, m8, ta, ma
	vslideup.vi	v16, v8, 16
	csrr	a2, vlenb
	li	a6, 24
	mul	a2, a2, a6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vl8r.v	v0, (a2)                        # vscale x 64-byte Folded Reload
	csrr	a2, vlenb
	li	a6, 105
	mul	a2, a2, a6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 9 36 is_stmt 0                # k135112023917040.py:9:36
	vadd.vv	v8, v0, v8
	vand.vx	v8, v8, a0
	vsub.vv	v8, v0, v8
	.loc	1 16 18 is_stmt 1               # k135112023917040.py:16:18
	vfmul.vf	v16, v16, fa5
	csrr	a2, vlenb
	li	a6, 97
	mul	a2, a2, a6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 9 42                          # k135112023917040.py:9:42
	vsetivli	zero, 16, e32, m4, ta, mu
	vmv.v.i	v16, 0
	vwmulsu.vx	v24, v8, a5
	csrr	a2, vlenb
	li	a6, 81
	mul	a2, a2, a6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vluxei64.v	v16, (a1), v24, v0.t
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v24, v8, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v0, v0, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v8, v24, a5
	csrr	a2, vlenb
	li	a6, 81
	mul	a2, a2, a6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vmv.v.i	v8, 0
	csrr	a2, vlenb
	li	a6, 81
	mul	a2, a2, a6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vl8r.v	v24, (a2)                       # vscale x 64-byte Folded Reload
	vluxei64.v	v8, (a1), v24, v0.t
	vsetvli	zero, s2, e32, m8, ta, ma
	vslideup.vi	v16, v8, 16
	csrr	a2, vlenb
	li	a6, 105
	mul	a2, a2, a6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	csrr	a2, vlenb
	slli	a2, a2, 4
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 113
	vl8r.v	v0, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 9 36 is_stmt 0                # k135112023917040.py:9:36
	vadd.vv	v8, v0, v8
	vand.vx	v8, v8, a0
	vsub.vv	v8, v0, v8
	csrr	a0, vlenb
	li	a2, 24
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 16 18 is_stmt 1               # k135112023917040.py:16:18
	vfmul.vf	v16, v16, fa5
	csrr	a0, vlenb
	li	a2, 105
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vsetivli	zero, 16, e32, m4, ta, mu
	vmv.v.i	v24, 0
	.loc	1 9 42                          # k135112023917040.py:9:42
	vmv.v.i	v16, 0
	vwmulsu.vx	v0, v8, a5
	csrr	a0, vlenb
	li	a2, 81
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a2, 40
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl1r.v	v20, (a0)                       # vscale x 8-byte Folded Reload
	vmv1r.v	v0, v20
	csrr	a0, vlenb
	li	a2, 81
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vluxei64.v	v16, (a1), v8, v0.t
	csrr	a0, vlenb
	li	a2, 24
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v0, v8, 16
	vsetivli	zero, 2, e8, mf4, ta, ma
	vslidedown.vi	v20, v20, 2
	vsetivli	zero, 16, e32, m4, ta, mu
	vwmulsu.vx	v8, v0, a5
	csrr	a0, vlenb
	li	a2, 81
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vmv1r.v	v0, v20
	csrr	a0, vlenb
	li	a2, 81
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vluxei64.v	v24, (a1), v8, v0.t
	.loc	1 12 31                         # k135112023917040.py:12:31
	add	a3, a3, a4
	.loc	1 12 36 is_stmt 0               # k135112023917040.py:12:36
	addi	a0, a3, 128
	vsetvli	zero, s3, e16, m8, ta, mu
	vmv.v.i	v8, 0
	csrr	a1, vlenb
	li	a2, 73
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 2047
	addi	a1, a1, 113
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vle16.v	v8, (a0), v0.t
	.loc	1 9 42 is_stmt 1                # k135112023917040.py:9:42
	vsetvli	zero, s2, e32, m8, ta, ma
	vslideup.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vmv.v.i	v24, 0
	.loc	1 12 36                         # k135112023917040.py:12:36
	vle16.v	v24, (a3), v0.t
	.loc	1 16 18                         # k135112023917040.py:16:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfmul.vf	v16, v16, fa5
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 46                         # k135112023917040.py:12:46
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v16, v8, s2
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vmv8r.v	v8, v24
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v24, s2
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135112023917040.py:8:49
	vzext.vf2	v8, v16
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vzext.vf2	v16, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 12 46                         # k135112023917040.py:12:46
	vzext.vf2	v8, v16
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vzext.vf2	v16, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 768
	csrr	a1, vlenb
	li	a2, 81
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 2047
	addi	a1, a1, 113
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 23                         # k135112023917040.py:17:23
	vse32.v	v8, (a0)
	flw	fa0, 892(sp)
	fsw	fa0, 508(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1660(sp)
	flw	fa0, 888(sp)
	fsw	fa0, 504(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1656(sp)
	flw	fa0, 884(sp)
	fsw	fa0, 500(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1652(sp)
	flw	fa0, 880(sp)
	fsw	fa0, 496(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1648(sp)
	flw	fa0, 876(sp)
	fsw	fa0, 492(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1644(sp)
	flw	fa0, 872(sp)
	fsw	fa0, 488(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1640(sp)
	flw	fa0, 868(sp)
	fsw	fa0, 484(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1636(sp)
	flw	fa0, 864(sp)
	fsw	fa0, 480(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1632(sp)
	flw	fa0, 860(sp)
	fsw	fa0, 476(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1628(sp)
	flw	fa0, 856(sp)
	fsw	fa0, 472(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1624(sp)
	flw	fa0, 852(sp)
	fsw	fa0, 468(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1620(sp)
	flw	fa0, 848(sp)
	fsw	fa0, 464(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1616(sp)
	flw	fa0, 844(sp)
	fsw	fa0, 460(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1612(sp)
	flw	fa0, 840(sp)
	fsw	fa0, 456(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1608(sp)
	flw	fa0, 836(sp)
	fsw	fa0, 452(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1604(sp)
	flw	fa0, 832(sp)
	fsw	fa0, 448(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1600(sp)
	flw	fa0, 828(sp)
	fsw	fa0, 444(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1596(sp)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 440(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1536(sp)
	flw	fa0, 824(sp)
	fsw	fa0, 436(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1592(sp)
	flw	fa0, 820(sp)
	fsw	fa0, 432(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1588(sp)
	flw	fa0, 816(sp)
	fsw	fa0, 428(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1584(sp)
	flw	fa0, 812(sp)
	fsw	fa0, 424(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1580(sp)
	flw	fa0, 808(sp)
	fsw	fa0, 420(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1576(sp)
	flw	fa0, 804(sp)
	fsw	fa0, 416(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1572(sp)
	flw	fa0, 800(sp)
	fsw	fa0, 412(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1568(sp)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 408(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1548(sp)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 404(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1544(sp)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 400(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1540(sp)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 396(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1564(sp)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 392(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1560(sp)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 388(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1556(sp)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 384(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1552(sp)
	addi	a0, sp, 896
	csrr	a1, vlenb
	li	a2, 89
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 2047
	addi	a1, a1, 113
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s2, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 1020(sp)
	fsw	fa0, 252(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1916(sp)
	flw	fa0, 1016(sp)
	fsw	fa0, 248(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1912(sp)
	flw	fa0, 1012(sp)
	fsw	fa0, 244(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1908(sp)
	flw	fa0, 1008(sp)
	fsw	fa0, 240(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1904(sp)
	flw	fa0, 1004(sp)
	fsw	fa0, 236(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1900(sp)
	flw	fa0, 1000(sp)
	fsw	fa0, 232(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1896(sp)
	flw	fa0, 996(sp)
	fsw	fa0, 228(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1892(sp)
	flw	fa0, 992(sp)
	fsw	fa0, 224(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1888(sp)
	flw	fa0, 988(sp)
	fsw	fa0, 220(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1884(sp)
	flw	fa0, 984(sp)
	fsw	fa0, 216(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1880(sp)
	flw	fa0, 980(sp)
	fsw	fa0, 212(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1876(sp)
	flw	fa0, 976(sp)
	fsw	fa0, 208(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1872(sp)
	flw	fa0, 972(sp)
	fsw	fa0, 204(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1868(sp)
	flw	fa0, 968(sp)
	fsw	fa0, 200(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1864(sp)
	flw	fa0, 964(sp)
	fsw	fa0, 196(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1860(sp)
	flw	fa0, 960(sp)
	fsw	fa0, 192(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1856(sp)
	flw	fa0, 956(sp)
	fsw	fa0, 188(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1852(sp)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 184(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1792(sp)
	flw	fa0, 952(sp)
	fsw	fa0, 180(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1848(sp)
	flw	fa0, 948(sp)
	fsw	fa0, 176(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1844(sp)
	flw	fa0, 944(sp)
	fsw	fa0, 172(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1840(sp)
	flw	fa0, 940(sp)
	fsw	fa0, 168(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1836(sp)
	flw	fa0, 936(sp)
	fsw	fa0, 164(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1832(sp)
	flw	fa0, 932(sp)
	fsw	fa0, 160(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1828(sp)
	flw	fa0, 928(sp)
	fsw	fa0, 156(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1824(sp)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 152(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1804(sp)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 148(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1800(sp)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 144(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1796(sp)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 140(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1820(sp)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 136(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1816(sp)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 132(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1812(sp)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 128(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1808(sp)
	addi	a0, sp, 512
	csrr	a1, vlenb
	li	a2, 97
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 2047
	addi	a1, a1, 113
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s2, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 636(sp)
	fsw	fa0, 380(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1148(sp)
	flw	fa0, 632(sp)
	fsw	fa0, 376(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1144(sp)
	flw	fa0, 628(sp)
	fsw	fa0, 372(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1140(sp)
	flw	fa0, 624(sp)
	fsw	fa0, 368(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1136(sp)
	flw	fa0, 620(sp)
	fsw	fa0, 364(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1132(sp)
	flw	fa0, 616(sp)
	fsw	fa0, 360(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1128(sp)
	flw	fa0, 612(sp)
	fsw	fa0, 356(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1124(sp)
	flw	fa0, 608(sp)
	fsw	fa0, 352(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1120(sp)
	flw	fa0, 604(sp)
	fsw	fa0, 348(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1116(sp)
	flw	fa0, 600(sp)
	fsw	fa0, 344(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1112(sp)
	flw	fa0, 596(sp)
	fsw	fa0, 340(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1108(sp)
	flw	fa0, 592(sp)
	fsw	fa0, 336(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1104(sp)
	flw	fa0, 588(sp)
	fsw	fa0, 332(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1100(sp)
	flw	fa0, 584(sp)
	fsw	fa0, 328(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1096(sp)
	flw	fa0, 580(sp)
	fsw	fa0, 324(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1092(sp)
	flw	fa0, 576(sp)
	fsw	fa0, 320(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1088(sp)
	flw	fa0, 572(sp)
	fsw	fa0, 316(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1084(sp)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 312(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1024(sp)
	flw	fa0, 568(sp)
	fsw	fa0, 308(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1080(sp)
	flw	fa0, 564(sp)
	fsw	fa0, 304(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1076(sp)
	flw	fa0, 560(sp)
	fsw	fa0, 300(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1072(sp)
	flw	fa0, 556(sp)
	fsw	fa0, 296(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1068(sp)
	flw	fa0, 552(sp)
	fsw	fa0, 292(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1064(sp)
	flw	fa0, 548(sp)
	fsw	fa0, 288(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1060(sp)
	flw	fa0, 544(sp)
	fsw	fa0, 284(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1056(sp)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	fsw	fa0, 280(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1036(sp)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	fsw	fa0, 276(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1032(sp)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	fsw	fa0, 272(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1028(sp)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	fsw	fa0, 268(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1052(sp)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	fsw	fa0, 264(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1048(sp)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	fsw	fa0, 260(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1044(sp)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	fsw	fa0, 256(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1040(sp)
	addi	a0, sp, 640
	csrr	a1, vlenb
	li	a2, 105
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 2047
	addi	a1, a1, 113
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s2, e32, m8, ta, ma
	vse32.v	v8, (a0)
	flw	fa0, 764(sp)
	fsw	fa0, 124(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1404(sp)
	flw	fa0, 760(sp)
	fsw	fa0, 120(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1400(sp)
	flw	fa0, 756(sp)
	fsw	fa0, 116(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1396(sp)
	flw	fa0, 752(sp)
	fsw	fa0, 112(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1392(sp)
	flw	fa0, 748(sp)
	fsw	fa0, 108(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1388(sp)
	flw	fa0, 744(sp)
	fsw	fa0, 104(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1384(sp)
	flw	fa0, 740(sp)
	fsw	fa0, 100(sp)                    # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1380(sp)
	flw	fa0, 736(sp)
	fsw	fa0, 96(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1376(sp)
	flw	fa0, 732(sp)
	fsw	fa0, 92(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1372(sp)
	flw	fa0, 728(sp)
	fsw	fa0, 88(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1368(sp)
	flw	fa0, 724(sp)
	fsw	fa0, 84(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1364(sp)
	flw	fa0, 720(sp)
	fsw	fa0, 80(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1360(sp)
	flw	fa0, 716(sp)
	fsw	fa0, 76(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1356(sp)
	flw	fa0, 712(sp)
	fsw	fa0, 72(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1352(sp)
	flw	fa0, 708(sp)
	fsw	fa0, 68(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1348(sp)
	flw	fa0, 704(sp)
	fsw	fa0, 64(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1344(sp)
	flw	fa0, 700(sp)
	fsw	fa0, 60(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1340(sp)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	fsw	fa0, 56(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1280(sp)
	flw	fa0, 696(sp)
	fsw	fa0, 52(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1336(sp)
	flw	fa0, 692(sp)
	fsw	fa0, 48(sp)                     # 4-byte Folded Spill
	call	cosf
	fsw	fa0, 1332(sp)
	flw	fs8, 688(sp)
	fmv.s	fa0, fs8
	call	cosf
	fsw	fa0, 1328(sp)
	flw	fs9, 684(sp)
	fmv.s	fa0, fs9
	call	cosf
	fsw	fa0, 1324(sp)
	flw	fs10, 680(sp)
	fmv.s	fa0, fs10
	call	cosf
	fsw	fa0, 1320(sp)
	flw	fs11, 676(sp)
	fmv.s	fa0, fs11
	call	cosf
	fsw	fa0, 1316(sp)
	flw	fs0, 672(sp)
	fmv.s	fa0, fs0
	call	cosf
	fsw	fa0, 1312(sp)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fs1, v8
	fmv.s	fa0, fs1
	call	cosf
	fsw	fa0, 1292(sp)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fs2, v8
	fmv.s	fa0, fs2
	call	cosf
	fsw	fa0, 1288(sp)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fs3, v8
	fmv.s	fa0, fs3
	call	cosf
	fsw	fa0, 1284(sp)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fs4, v8
	fmv.s	fa0, fs4
	call	cosf
	fsw	fa0, 1308(sp)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fs5, v8
	fmv.s	fa0, fs5
	call	cosf
	fsw	fa0, 1304(sp)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fs6, v8
	fmv.s	fa0, fs6
	call	cosf
	fsw	fa0, 1300(sp)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fs7, v8
	fmv.s	fa0, fs7
	call	cosf
	fsw	fa0, 1296(sp)
	addi	a0, sp, 1536
	vsetvli	zero, s2, e32, m8, ta, ma
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 1792
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 1024
	vle32.v	v8, (a0)
	addi	a0, sp, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 1280
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	flw	fa0, 252(sp)                    # 4-byte Folded Reload
	.loc	1 20 24                         # k135112023917040.py:20:24
	call	sinf
	fsw	fa0, 2044(sp)
	flw	fa0, 248(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2040(sp)
	flw	fa0, 244(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2036(sp)
	flw	fa0, 240(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2032(sp)
	flw	fa0, 236(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2028(sp)
	flw	fa0, 232(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2024(sp)
	flw	fa0, 228(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2020(sp)
	flw	fa0, 224(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2016(sp)
	flw	fa0, 220(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2012(sp)
	flw	fa0, 216(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2008(sp)
	flw	fa0, 212(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2004(sp)
	flw	fa0, 208(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 2000(sp)
	flw	fa0, 204(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1996(sp)
	flw	fa0, 200(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1992(sp)
	flw	fa0, 196(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1988(sp)
	flw	fa0, 192(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1984(sp)
	flw	fa0, 188(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1980(sp)
	flw	fa0, 184(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1920(sp)
	flw	fa0, 180(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1976(sp)
	flw	fa0, 176(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1972(sp)
	flw	fa0, 172(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1968(sp)
	flw	fa0, 168(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1964(sp)
	flw	fa0, 164(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1960(sp)
	flw	fa0, 160(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1956(sp)
	flw	fa0, 156(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1952(sp)
	flw	fa0, 152(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1932(sp)
	flw	fa0, 148(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1928(sp)
	flw	fa0, 144(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1924(sp)
	flw	fa0, 140(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1948(sp)
	flw	fa0, 136(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1944(sp)
	flw	fa0, 132(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1940(sp)
	flw	fa0, 128(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1936(sp)
	flw	fa0, 508(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1788(sp)
	flw	fa0, 504(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1784(sp)
	flw	fa0, 500(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1780(sp)
	flw	fa0, 496(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1776(sp)
	flw	fa0, 492(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1772(sp)
	flw	fa0, 488(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1768(sp)
	flw	fa0, 484(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1764(sp)
	flw	fa0, 480(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1760(sp)
	flw	fa0, 476(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1756(sp)
	flw	fa0, 472(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1752(sp)
	flw	fa0, 468(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1748(sp)
	flw	fa0, 464(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1744(sp)
	flw	fa0, 460(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1740(sp)
	flw	fa0, 456(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1736(sp)
	flw	fa0, 452(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1732(sp)
	flw	fa0, 448(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1728(sp)
	flw	fa0, 444(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1724(sp)
	flw	fa0, 440(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1664(sp)
	flw	fa0, 436(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1720(sp)
	flw	fa0, 432(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1716(sp)
	flw	fa0, 428(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1712(sp)
	flw	fa0, 424(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1708(sp)
	flw	fa0, 420(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1704(sp)
	flw	fa0, 416(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1700(sp)
	flw	fa0, 412(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1696(sp)
	flw	fa0, 408(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1676(sp)
	flw	fa0, 404(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1672(sp)
	flw	fa0, 400(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1668(sp)
	flw	fa0, 396(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1692(sp)
	flw	fa0, 392(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1688(sp)
	flw	fa0, 388(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1684(sp)
	flw	fa0, 384(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1680(sp)
	flw	fa0, 124(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1532(sp)
	flw	fa0, 120(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1528(sp)
	flw	fa0, 116(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1524(sp)
	flw	fa0, 112(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1520(sp)
	flw	fa0, 108(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1516(sp)
	flw	fa0, 104(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1512(sp)
	flw	fa0, 100(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1508(sp)
	flw	fa0, 96(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1504(sp)
	flw	fa0, 92(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1500(sp)
	flw	fa0, 88(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1496(sp)
	flw	fa0, 84(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1492(sp)
	flw	fa0, 80(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1488(sp)
	flw	fa0, 76(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1484(sp)
	flw	fa0, 72(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1480(sp)
	flw	fa0, 68(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1476(sp)
	flw	fa0, 64(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1472(sp)
	flw	fa0, 60(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1468(sp)
	flw	fa0, 56(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1408(sp)
	flw	fa0, 52(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1464(sp)
	flw	fa0, 48(sp)                     # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1460(sp)
	fmv.s	fa0, fs8
	call	sinf
	fsw	fa0, 1456(sp)
	fmv.s	fa0, fs9
	call	sinf
	fsw	fa0, 1452(sp)
	fmv.s	fa0, fs10
	call	sinf
	fsw	fa0, 1448(sp)
	fmv.s	fa0, fs11
	call	sinf
	fsw	fa0, 1444(sp)
	fmv.s	fa0, fs0
	call	sinf
	fsw	fa0, 1440(sp)
	fmv.s	fa0, fs1
	call	sinf
	fsw	fa0, 1420(sp)
	fmv.s	fa0, fs2
	call	sinf
	fsw	fa0, 1416(sp)
	fmv.s	fa0, fs3
	call	sinf
	fsw	fa0, 1412(sp)
	fmv.s	fa0, fs4
	call	sinf
	fsw	fa0, 1436(sp)
	fmv.s	fa0, fs5
	call	sinf
	fsw	fa0, 1432(sp)
	fmv.s	fa0, fs6
	call	sinf
	fsw	fa0, 1428(sp)
	fmv.s	fa0, fs7
	call	sinf
	fsw	fa0, 1424(sp)
	flw	fa0, 380(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1276(sp)
	flw	fa0, 376(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1272(sp)
	flw	fa0, 372(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1268(sp)
	flw	fa0, 368(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1264(sp)
	flw	fa0, 364(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1260(sp)
	flw	fa0, 360(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1256(sp)
	flw	fa0, 356(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1252(sp)
	flw	fa0, 352(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1248(sp)
	flw	fa0, 348(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1244(sp)
	flw	fa0, 344(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1240(sp)
	flw	fa0, 340(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1236(sp)
	flw	fa0, 336(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1232(sp)
	flw	fa0, 332(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1228(sp)
	flw	fa0, 328(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1224(sp)
	flw	fa0, 324(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1220(sp)
	flw	fa0, 320(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1216(sp)
	flw	fa0, 316(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1212(sp)
	flw	fa0, 312(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1152(sp)
	flw	fa0, 308(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1208(sp)
	flw	fa0, 304(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1204(sp)
	flw	fa0, 300(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1200(sp)
	flw	fa0, 296(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1196(sp)
	flw	fa0, 292(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1192(sp)
	flw	fa0, 288(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1188(sp)
	flw	fa0, 284(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1184(sp)
	flw	fa0, 280(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1164(sp)
	flw	fa0, 276(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1160(sp)
	flw	fa0, 272(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1156(sp)
	flw	fa0, 268(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1180(sp)
	flw	fa0, 264(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1176(sp)
	flw	fa0, 260(sp)                    # 4-byte Folded Reload
	call	sinf
	fsw	fa0, 1172(sp)
	flw	fa0, 256(sp)                    # 4-byte Folded Reload
	call	sinf
	csrr	a0, vlenb
	li	a1, 73
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	fsw	fa0, 1168(sp)
	addi	a0, sp, 1408
	vsetvli	zero, s2, e32, m8, ta, ma
	vle32.v	v8, (a0)
	addi	a0, sp, 1152
	vle32.v	v16, (a0)
	csrr	a0, vlenb
	li	a1, 73
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 20                         # k135112023917040.py:22:20
	vfmul.vv	v8, v8, v24
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 23 20                         # k135112023917040.py:23:20
	vfmacc.vv	v16, v8, v24
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 73
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 20                         # k135112023917040.py:22:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 1920
	.loc	1 20 24                         # k135112023917040.py:20:24
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 73
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 1664
	vle32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 23 20                         # k135112023917040.py:23:20
	vfmacc.vv	v24, v8, v16
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 73
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 20                         # k135112023917040.py:22:20
	vfmul.vv	v24, v16, v8
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 23 20                         # k135112023917040.py:23:20
	vfmacc.vv	v24, v8, v16
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 20                         # k135112023917040.py:22:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 23 20                         # k135112023917040.py:23:20
	vfmacc.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 40                         # k135112023917040.py:24:40
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v8, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v24, v16, s2
	vse16.v	v24, (s5), v0.t
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s2, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v8, v24, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v8, v16, s2
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 113
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vse16.v	v8, (s4), v0.t
	.loc	1 24 4 epilogue_begin is_stmt 0 # k135112023917040.py:24:4
	addi	sp, s0, -2032
	.cfi_def_cfa sp, 2032
	ld	ra, 2024(sp)                    # 8-byte Folded Reload
	ld	s0, 2016(sp)                    # 8-byte Folded Reload
	ld	s2, 2008(sp)                    # 8-byte Folded Reload
	ld	s3, 2000(sp)                    # 8-byte Folded Reload
	ld	s4, 1992(sp)                    # 8-byte Folded Reload
	ld	s5, 1984(sp)                    # 8-byte Folded Reload
	fld	fs0, 1976(sp)                   # 8-byte Folded Reload
	fld	fs1, 1968(sp)                   # 8-byte Folded Reload
	fld	fs2, 1960(sp)                   # 8-byte Folded Reload
	fld	fs3, 1952(sp)                   # 8-byte Folded Reload
	fld	fs4, 1944(sp)                   # 8-byte Folded Reload
	fld	fs5, 1936(sp)                   # 8-byte Folded Reload
	fld	fs6, 1928(sp)                   # 8-byte Folded Reload
	fld	fs7, 1920(sp)                   # 8-byte Folded Reload
	fld	fs8, 1912(sp)                   # 8-byte Folded Reload
	fld	fs9, 1904(sp)                   # 8-byte Folded Reload
	fld	fs10, 1896(sp)                  # 8-byte Folded Reload
	fld	fs11, 1888(sp)                  # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
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
	.size	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_7, .Lfunc_end0-triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_7
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
	.asciz	"k135112023917040.py"           # string offset=7 ; k135112023917040.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
