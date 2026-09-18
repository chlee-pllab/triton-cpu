	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused__unsafe_view_mul_silu_11 # -- Begin function triton_poi_fused__unsafe_view_mul_silu_11
	.p2align	2
	.type	triton_poi_fused__unsafe_view_mul_silu_11,@function
triton_poi_fused__unsafe_view_mul_silu_11: # @triton_poi_fused__unsafe_view_mul_silu_11
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114449656112.py"
	.loc	1 2 0                           # k135114449656112.py:2:0
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
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s2, -24
	.cfi_offset s3, -32
	.cfi_offset s4, -40
	.cfi_offset s5, -48
	.cfi_offset s6, -56
	.cfi_offset s7, -64
	.cfi_offset s8, -72
	addi	s0, sp, 2032
	.cfi_def_cfa s0, 0
	addi	sp, sp, -288
	csrr	a2, vlenb
	li	a4, 196
	mul	a2, a2, a4
	sub	sp, sp, a2
	andi	sp, sp, -128
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114449656112.py:4:33
	slliw	a3, a3, 8
	li	s2, 32
	li	s3, 64
	li	a4, 96
	li	a5, 128
	li	a2, 160
	li	a6, 192
	.loc	1 5 23                          # k135114449656112.py:5:23
	vsetvli	zero, s2, e32, m8, ta, ma
	vid.v	v16
	vadd.vx	v8, v16, a2
	csrr	a2, vlenb
	li	a7, 180
	mul	a2, a2, a7
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	li	a2, 224
	vadd.vx	v24, v16, a2
	li	a2, 19
	slli	a7, a2, 8
	vor.vx	v0, v16, a3
	vor.vx	v24, v24, a3
	.loc	1 6 21                          # k135114449656112.py:6:21
	vmslt.vx	v8, v0, a7
	csrr	a2, vlenb
	slli	t0, a2, 7
	add	a2, t0, a2
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs1r.v	v8, (a2)                        # vscale x 8-byte Folded Spill
	vmslt.vx	v8, v24, a7
	csrr	a2, vlenb
	li	t0, 188
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs1r.v	v8, (a2)                        # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114449656112.py:5:23
	vadd.vx	v24, v16, a6
	.loc	1 8 34                          # k135114449656112.py:8:34
	slli	a2, a3, 1
	.loc	1 5 23                          # k135114449656112.py:5:23
	vor.vx	v24, v24, a3
	.loc	1 6 21                          # k135114449656112.py:6:21
	vmslt.vx	v8, v24, a7
	csrr	a6, vlenb
	li	t0, 130
	mul	a6, a6, t0
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 193
	vs1r.v	v8, (a6)                        # vscale x 8-byte Folded Spill
	.loc	1 8 39                          # k135114449656112.py:8:39
	vsetvli	zero, s3, e16, m8, ta, ma
	vmv.v.i	v0, 0
	csrr	a6, vlenb
	li	t0, 180
	mul	a6, a6, t0
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 193
	vl8r.v	v8, (a6)                        # vscale x 64-byte Folded Reload
	.loc	1 5 23                          # k135114449656112.py:5:23
	vsetvli	zero, s2, e32, m8, ta, ma
	vor.vx	v8, v8, a3
	.loc	1 6 21                          # k135114449656112.py:6:21
	vmslt.vx	v24, v8, a7
	csrr	a6, vlenb
	li	t0, 180
	mul	a6, a6, t0
	add	a6, sp, a6
	addi	a6, a6, 2047
	addi	a6, a6, 193
	vs1r.v	v24, (a6)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114449656112.py:5:23
	vadd.vx	v8, v16, a5
	vor.vx	v8, v8, a3
	.loc	1 6 21                          # k135114449656112.py:6:21
	vmslt.vx	v24, v8, a7
	csrr	a5, vlenb
	li	a6, 131
	mul	a5, a5, a6
	add	a5, sp, a5
	addi	a5, a5, 2047
	addi	a5, a5, 193
	vs1r.v	v24, (a5)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114449656112.py:5:23
	vadd.vx	v8, v16, a4
	vor.vx	v8, v8, a3
	.loc	1 6 21                          # k135114449656112.py:6:21
	vmslt.vx	v24, v8, a7
	csrr	a4, vlenb
	li	a5, 172
	mul	a4, a4, a5
	add	a4, sp, a4
	addi	a4, a4, 2047
	addi	a4, a4, 193
	vs1r.v	v24, (a4)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114449656112.py:5:23
	vadd.vx	v8, v16, s3
	vadd.vx	v24, v16, s2
	.loc	1 8 34                          # k135114449656112.py:8:34
	add	s7, a0, a2
	.loc	1 5 23                          # k135114449656112.py:5:23
	vor.vx	v8, v8, a3
	.loc	1 6 21                          # k135114449656112.py:6:21
	vmslt.vx	v16, v8, a7
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs1r.v	v16, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 8 39                          # k135114449656112.py:8:39
	vmv8r.v	v16, v0
	.loc	1 5 23                          # k135114449656112.py:5:23
	vor.vx	v8, v24, a3
	.loc	1 6 21                          # k135114449656112.py:6:21
	vmslt.vx	v24, v8, a7
	csrr	a0, vlenb
	li	a3, 164
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs1r.v	v24, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 8 39                          # k135114449656112.py:8:39
	vmv8r.v	v24, v0
	addi	s4, s7, 128
	addi	s5, s7, 256
	addi	s6, s7, 384
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v3, (a0)                        # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a3, 172
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v8, (a0)                        # vscale x 8-byte Folded Reload
	.loc	1 6 21                          # k135114449656112.py:6:21
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v3, v8, 4
	csrr	a0, vlenb
	li	a3, 130
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v7, (a0)                        # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a3, 188
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v8, (a0)                        # vscale x 8-byte Folded Reload
	vslideup.vi	v7, v8, 4
	csrr	a0, vlenb
	li	a3, 130
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs1r.v	v7, (a0)                        # vscale x 8-byte Folded Spill
	.loc	1 8 39                          # k135114449656112.py:8:39
	vmv1r.v	v0, v3
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs1r.v	v3, (a0)                        # vscale x 8-byte Folded Spill
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v16, (s4), v0.t
	csrr	a0, vlenb
	li	a3, 131
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a3, 180
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v8, (a0)                        # vscale x 8-byte Folded Reload
	.loc	1 6 21                          # k135114449656112.py:6:21
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v0, v8, 4
	csrr	a0, vlenb
	li	a3, 131
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs1r.v	v0, (a0)                        # vscale x 8-byte Folded Spill
	csrr	a0, vlenb
	slli	a3, a0, 7
	add	a0, a3, a0
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v6, (a0)                        # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a3, 164
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v8, (a0)                        # vscale x 8-byte Folded Reload
	vslideup.vi	v6, v8, 4
	csrr	a0, vlenb
	slli	a3, a0, 7
	add	a0, a3, a0
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs1r.v	v6, (a0)                        # vscale x 8-byte Folded Spill
	.loc	1 8 39                          # k135114449656112.py:8:39
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v24, (s5), v0.t
	.loc	1 8 49 is_stmt 0                # k135114449656112.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v8, v16, s2
	csrr	a0, vlenb
	li	a3, 156
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v8, v16
	csrr	a0, vlenb
	li	a3, 140
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 8 39                          # k135114449656112.py:8:39
	vsetvli	zero, s3, e16, m8, ta, mu
	vmv.v.i	v8, 0
	vmv1r.v	v0, v7
	vle16.v	v8, (s6), v0.t
	.loc	1 8 49                          # k135114449656112.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, s2
	csrr	a0, vlenb
	li	a3, 172
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v16, v24
	csrr	a0, vlenb
	li	a3, 148
	mul	a0, a0, a3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 8 39                          # k135114449656112.py:8:39
	vsetvli	zero, s3, e16, m8, ta, mu
	vmv.v.i	v16, 0
	vmv1r.v	v0, v6
	vle16.v	v16, (s7), v0.t
	.loc	1 9 30 is_stmt 1                # k135114449656112.py:9:30
	add	a1, a1, a2
	.loc	1 9 35 is_stmt 0                # k135114449656112.py:9:35
	addi	a0, a1, 128
	.loc	1 8 49 is_stmt 1                # k135114449656112.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v24, v8, s2
	csrr	a2, vlenb
	li	a3, 188
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v24, v8
	csrr	a2, vlenb
	li	a3, 164
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s3, e16, m8, ta, mu
	vmv.v.i	v24, 0
	.loc	1 9 35                          # k135114449656112.py:9:35
	vmv.v.i	v8, 0
	vmv1r.v	v0, v3
	vle16.v	v8, (a0), v0.t
	addi	a0, a1, 256
	.loc	1 8 49                          # k135114449656112.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v16, s2
	csrr	a2, vlenb
	li	a3, 120
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a2, vlenb
	li	a3, 180
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 9 35                          # k135114449656112.py:9:35
	vmv8r.v	v16, v24
	csrr	a2, vlenb
	li	a3, 131
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v16, (a0), v0.t
	addi	a0, a1, 384
	.loc	1 9 45 is_stmt 0                # k135114449656112.py:9:45
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v8, s2
	csrr	a2, vlenb
	li	a3, 72
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	csrr	a2, vlenb
	li	a3, 88
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vmv8r.v	v8, v24
	csrr	a2, vlenb
	li	a3, 130
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	.loc	1 9 35                          # k135114449656112.py:9:35
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v24, (a0), v0.t
	csrr	a0, vlenb
	slli	a2, a0, 7
	add	a0, a2, a0
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vle16.v	v8, (a1), v0.t
	.loc	1 9 45                          # k135114449656112.py:9:45
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v16, s2
	addi	a0, sp, 2047
	addi	a0, a0, 193
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, s2
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v16, v24
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v16, v8, s2
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	s8, sp, 2047
	addi	s8, s8, 1
	fmv.w.x	fa5, zero
	addi	a0, sp, 512
	addi	a1, sp, 384
	csrr	a2, vlenb
	li	a3, 140
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49 is_stmt 1                # k135114449656112.py:8:49
	vsll.vi	v16, v8, 16
	csrr	a2, vlenb
	li	a3, 96
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114449656112.py:11:12
	vfrsub.vf	v16, v16, fa5
	csrr	a2, vlenb
	li	a3, 132
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114449656112.py:12:25
	vse32.v	v16, (a0)
	addi	a0, sp, 768
	csrr	a2, vlenb
	li	a3, 156
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114449656112.py:8:49
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a2, vlenb
	li	a3, 104
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114449656112.py:11:12
	vfrsub.vf	v16, v16, fa5
	csrr	a2, vlenb
	li	a3, 140
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114449656112.py:12:25
	vse32.v	v16, (a1)
	addi	a1, sp, 640
	csrr	a2, vlenb
	li	a3, 148
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114449656112.py:8:49
	vsll.vi	v16, v8, 16
	csrr	a2, vlenb
	li	a3, 80
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114449656112.py:11:12
	vfrsub.vf	v16, v16, fa5
	csrr	a2, vlenb
	li	a3, 148
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114449656112.py:12:25
	vse32.v	v16, (a0)
	addi	a0, sp, 1024
	csrr	a2, vlenb
	li	a3, 172
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114449656112.py:8:49
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a2, vlenb
	slli	a2, a2, 6
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114449656112.py:11:12
	vfrsub.vf	v16, v16, fa5
	csrr	a2, vlenb
	li	a3, 156
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114449656112.py:12:25
	vse32.v	v16, (a1)
	addi	a1, sp, 896
	csrr	a2, vlenb
	li	a3, 164
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114449656112.py:8:49
	vsll.vi	v16, v8, 16
	csrr	a2, vlenb
	li	a3, 48
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114449656112.py:11:12
	vfrsub.vf	v16, v16, fa5
	csrr	a2, vlenb
	li	a3, 164
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114449656112.py:12:25
	vse32.v	v16, (a0)
	addi	a0, sp, 256
	csrr	a2, vlenb
	li	a3, 188
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114449656112.py:8:49
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a2, vlenb
	slli	a2, a2, 5
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114449656112.py:11:12
	vfrsub.vf	v16, v16, fa5
	csrr	a2, vlenb
	li	a3, 172
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114449656112.py:12:25
	vse32.v	v16, (a1)
	addi	a1, sp, 128
	csrr	a2, vlenb
	li	a3, 120
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114449656112.py:8:49
	vzext.vf2	v16, v8
	csrr	a2, vlenb
	li	a3, 180
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	vsll.vi	v24, v8, 16
	csrr	a2, vlenb
	li	a3, 120
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	vsll.vi	v16, v16, 16
	csrr	a2, vlenb
	li	a3, 112
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	csrr	a2, vlenb
	li	a3, 72
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 9 45                          # k135114449656112.py:9:45
	vzext.vf2	v16, v8
	csrr	a2, vlenb
	li	a3, 88
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	vsll.vi	v24, v8, 16
	csrr	a2, vlenb
	li	a3, 88
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, sp, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	vzext.vf2	v24, v8
	vsll.vi	v8, v16, 16
	csrr	a2, vlenb
	li	a3, 72
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	csrr	a2, vlenb
	li	a3, 56
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 56
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	csrr	a2, vlenb
	li	a3, 120
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 11 12                         # k135114449656112.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 180
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 2047
	addi	a2, a2, 193
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114449656112.py:12:25
	vse32.v	v8, (a0)
	csrr	a0, vlenb
	li	a2, 40
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 9 45                          # k135114449656112.py:9:45
	vzext.vf2	v8, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a2, 40
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a2, 24
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a2, 24
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vzext.vf2	v16, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vsll.vi	v8, v16, 16
	addi	a0, sp, 2047
	addi	a0, a0, 193
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a2, 112
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 11 12                         # k135114449656112.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a0, vlenb
	li	a2, 188
	mul	a0, a0, a2
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114449656112.py:12:25
	vse32.v	v8, (a1)
	flw	fa0, 636(sp)
	call	expf
	fsw	fa0, 1660(sp)
	flw	fa0, 632(sp)
	call	expf
	fsw	fa0, 1656(sp)
	flw	fa0, 628(sp)
	call	expf
	fsw	fa0, 1652(sp)
	flw	fa0, 624(sp)
	call	expf
	fsw	fa0, 1648(sp)
	flw	fa0, 620(sp)
	call	expf
	fsw	fa0, 1644(sp)
	flw	fa0, 616(sp)
	call	expf
	fsw	fa0, 1640(sp)
	flw	fa0, 612(sp)
	call	expf
	fsw	fa0, 1636(sp)
	flw	fa0, 608(sp)
	call	expf
	fsw	fa0, 1632(sp)
	flw	fa0, 604(sp)
	call	expf
	fsw	fa0, 1628(sp)
	flw	fa0, 600(sp)
	call	expf
	fsw	fa0, 1624(sp)
	flw	fa0, 596(sp)
	call	expf
	fsw	fa0, 1620(sp)
	flw	fa0, 592(sp)
	call	expf
	fsw	fa0, 1616(sp)
	flw	fa0, 588(sp)
	call	expf
	fsw	fa0, 1612(sp)
	flw	fa0, 584(sp)
	call	expf
	fsw	fa0, 1608(sp)
	flw	fa0, 580(sp)
	call	expf
	fsw	fa0, 1604(sp)
	flw	fa0, 576(sp)
	call	expf
	fsw	fa0, 1600(sp)
	flw	fa0, 572(sp)
	call	expf
	fsw	fa0, 1596(sp)
	csrr	a0, vlenb
	li	a1, 132
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1536(sp)
	flw	fa0, 568(sp)
	call	expf
	fsw	fa0, 1592(sp)
	flw	fa0, 564(sp)
	call	expf
	fsw	fa0, 1588(sp)
	flw	fa0, 560(sp)
	call	expf
	fsw	fa0, 1584(sp)
	flw	fa0, 556(sp)
	call	expf
	fsw	fa0, 1580(sp)
	flw	fa0, 552(sp)
	call	expf
	fsw	fa0, 1576(sp)
	flw	fa0, 548(sp)
	call	expf
	fsw	fa0, 1572(sp)
	flw	fa0, 544(sp)
	call	expf
	fsw	fa0, 1568(sp)
	csrr	a0, vlenb
	li	a1, 132
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1548(sp)
	csrr	a0, vlenb
	li	a1, 132
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1544(sp)
	csrr	a0, vlenb
	li	a1, 132
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1540(sp)
	csrr	a0, vlenb
	li	a1, 132
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1564(sp)
	csrr	a0, vlenb
	li	a1, 132
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1560(sp)
	csrr	a0, vlenb
	li	a1, 132
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1556(sp)
	csrr	a0, vlenb
	li	a1, 132
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1552(sp)
	flw	fa0, 508(sp)
	call	expf
	fsw	fa0, 1532(sp)
	flw	fa0, 504(sp)
	call	expf
	fsw	fa0, 1528(sp)
	flw	fa0, 500(sp)
	call	expf
	fsw	fa0, 1524(sp)
	flw	fa0, 496(sp)
	call	expf
	fsw	fa0, 1520(sp)
	flw	fa0, 492(sp)
	call	expf
	fsw	fa0, 1516(sp)
	flw	fa0, 488(sp)
	call	expf
	fsw	fa0, 1512(sp)
	flw	fa0, 484(sp)
	call	expf
	fsw	fa0, 1508(sp)
	flw	fa0, 480(sp)
	call	expf
	fsw	fa0, 1504(sp)
	flw	fa0, 476(sp)
	call	expf
	fsw	fa0, 1500(sp)
	flw	fa0, 472(sp)
	call	expf
	fsw	fa0, 1496(sp)
	flw	fa0, 468(sp)
	call	expf
	fsw	fa0, 1492(sp)
	flw	fa0, 464(sp)
	call	expf
	fsw	fa0, 1488(sp)
	flw	fa0, 460(sp)
	call	expf
	fsw	fa0, 1484(sp)
	flw	fa0, 456(sp)
	call	expf
	fsw	fa0, 1480(sp)
	flw	fa0, 452(sp)
	call	expf
	fsw	fa0, 1476(sp)
	flw	fa0, 448(sp)
	call	expf
	fsw	fa0, 1472(sp)
	flw	fa0, 444(sp)
	call	expf
	fsw	fa0, 1468(sp)
	csrr	a0, vlenb
	li	a1, 140
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1408(sp)
	flw	fa0, 440(sp)
	call	expf
	fsw	fa0, 1464(sp)
	flw	fa0, 436(sp)
	call	expf
	fsw	fa0, 1460(sp)
	flw	fa0, 432(sp)
	call	expf
	fsw	fa0, 1456(sp)
	flw	fa0, 428(sp)
	call	expf
	fsw	fa0, 1452(sp)
	flw	fa0, 424(sp)
	call	expf
	fsw	fa0, 1448(sp)
	flw	fa0, 420(sp)
	call	expf
	fsw	fa0, 1444(sp)
	flw	fa0, 416(sp)
	call	expf
	fsw	fa0, 1440(sp)
	csrr	a0, vlenb
	li	a1, 140
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1420(sp)
	csrr	a0, vlenb
	li	a1, 140
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1416(sp)
	csrr	a0, vlenb
	li	a1, 140
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1412(sp)
	csrr	a0, vlenb
	li	a1, 140
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1436(sp)
	csrr	a0, vlenb
	li	a1, 140
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1432(sp)
	csrr	a0, vlenb
	li	a1, 140
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1428(sp)
	csrr	a0, vlenb
	li	a1, 140
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1424(sp)
	flw	fa0, 892(sp)
	call	expf
	fsw	fa0, 1916(sp)
	flw	fa0, 888(sp)
	call	expf
	fsw	fa0, 1912(sp)
	flw	fa0, 884(sp)
	call	expf
	fsw	fa0, 1908(sp)
	flw	fa0, 880(sp)
	call	expf
	fsw	fa0, 1904(sp)
	flw	fa0, 876(sp)
	call	expf
	fsw	fa0, 1900(sp)
	flw	fa0, 872(sp)
	call	expf
	fsw	fa0, 1896(sp)
	flw	fa0, 868(sp)
	call	expf
	fsw	fa0, 1892(sp)
	flw	fa0, 864(sp)
	call	expf
	fsw	fa0, 1888(sp)
	flw	fa0, 860(sp)
	call	expf
	fsw	fa0, 1884(sp)
	flw	fa0, 856(sp)
	call	expf
	fsw	fa0, 1880(sp)
	flw	fa0, 852(sp)
	call	expf
	fsw	fa0, 1876(sp)
	flw	fa0, 848(sp)
	call	expf
	fsw	fa0, 1872(sp)
	flw	fa0, 844(sp)
	call	expf
	fsw	fa0, 1868(sp)
	flw	fa0, 840(sp)
	call	expf
	fsw	fa0, 1864(sp)
	flw	fa0, 836(sp)
	call	expf
	fsw	fa0, 1860(sp)
	flw	fa0, 832(sp)
	call	expf
	fsw	fa0, 1856(sp)
	flw	fa0, 828(sp)
	call	expf
	fsw	fa0, 1852(sp)
	csrr	a0, vlenb
	li	a1, 148
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1792(sp)
	flw	fa0, 824(sp)
	call	expf
	fsw	fa0, 1848(sp)
	flw	fa0, 820(sp)
	call	expf
	fsw	fa0, 1844(sp)
	flw	fa0, 816(sp)
	call	expf
	fsw	fa0, 1840(sp)
	flw	fa0, 812(sp)
	call	expf
	fsw	fa0, 1836(sp)
	flw	fa0, 808(sp)
	call	expf
	fsw	fa0, 1832(sp)
	flw	fa0, 804(sp)
	call	expf
	fsw	fa0, 1828(sp)
	flw	fa0, 800(sp)
	call	expf
	fsw	fa0, 1824(sp)
	csrr	a0, vlenb
	li	a1, 148
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1804(sp)
	csrr	a0, vlenb
	li	a1, 148
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1800(sp)
	csrr	a0, vlenb
	li	a1, 148
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1796(sp)
	csrr	a0, vlenb
	li	a1, 148
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1820(sp)
	csrr	a0, vlenb
	li	a1, 148
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1816(sp)
	csrr	a0, vlenb
	li	a1, 148
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1812(sp)
	csrr	a0, vlenb
	li	a1, 148
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1808(sp)
	flw	fa0, 764(sp)
	call	expf
	fsw	fa0, 1788(sp)
	flw	fa0, 760(sp)
	call	expf
	fsw	fa0, 1784(sp)
	flw	fa0, 756(sp)
	call	expf
	fsw	fa0, 1780(sp)
	flw	fa0, 752(sp)
	call	expf
	fsw	fa0, 1776(sp)
	flw	fa0, 748(sp)
	call	expf
	fsw	fa0, 1772(sp)
	flw	fa0, 744(sp)
	call	expf
	fsw	fa0, 1768(sp)
	flw	fa0, 740(sp)
	call	expf
	fsw	fa0, 1764(sp)
	flw	fa0, 736(sp)
	call	expf
	fsw	fa0, 1760(sp)
	flw	fa0, 732(sp)
	call	expf
	fsw	fa0, 1756(sp)
	flw	fa0, 728(sp)
	call	expf
	fsw	fa0, 1752(sp)
	flw	fa0, 724(sp)
	call	expf
	fsw	fa0, 1748(sp)
	flw	fa0, 720(sp)
	call	expf
	fsw	fa0, 1744(sp)
	flw	fa0, 716(sp)
	call	expf
	fsw	fa0, 1740(sp)
	flw	fa0, 712(sp)
	call	expf
	fsw	fa0, 1736(sp)
	flw	fa0, 708(sp)
	call	expf
	fsw	fa0, 1732(sp)
	flw	fa0, 704(sp)
	call	expf
	fsw	fa0, 1728(sp)
	flw	fa0, 700(sp)
	call	expf
	fsw	fa0, 1724(sp)
	csrr	a0, vlenb
	li	a1, 156
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1664(sp)
	flw	fa0, 696(sp)
	call	expf
	fsw	fa0, 1720(sp)
	flw	fa0, 692(sp)
	call	expf
	fsw	fa0, 1716(sp)
	flw	fa0, 688(sp)
	call	expf
	fsw	fa0, 1712(sp)
	flw	fa0, 684(sp)
	call	expf
	fsw	fa0, 1708(sp)
	flw	fa0, 680(sp)
	call	expf
	fsw	fa0, 1704(sp)
	flw	fa0, 676(sp)
	call	expf
	fsw	fa0, 1700(sp)
	flw	fa0, 672(sp)
	call	expf
	fsw	fa0, 1696(sp)
	csrr	a0, vlenb
	li	a1, 156
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1676(sp)
	csrr	a0, vlenb
	li	a1, 156
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1672(sp)
	csrr	a0, vlenb
	li	a1, 156
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1668(sp)
	csrr	a0, vlenb
	li	a1, 156
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1692(sp)
	csrr	a0, vlenb
	li	a1, 156
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1688(sp)
	csrr	a0, vlenb
	li	a1, 156
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1684(sp)
	csrr	a0, vlenb
	li	a1, 156
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1680(sp)
	flw	fa0, 1148(sp)
	call	expf
	fsw	fa0, 124(s8)
	flw	fa0, 1144(sp)
	call	expf
	fsw	fa0, 120(s8)
	flw	fa0, 1140(sp)
	call	expf
	fsw	fa0, 116(s8)
	flw	fa0, 1136(sp)
	call	expf
	fsw	fa0, 112(s8)
	flw	fa0, 1132(sp)
	call	expf
	fsw	fa0, 108(s8)
	flw	fa0, 1128(sp)
	call	expf
	fsw	fa0, 104(s8)
	flw	fa0, 1124(sp)
	call	expf
	fsw	fa0, 100(s8)
	flw	fa0, 1120(sp)
	call	expf
	fsw	fa0, 96(s8)
	flw	fa0, 1116(sp)
	call	expf
	fsw	fa0, 92(s8)
	flw	fa0, 1112(sp)
	call	expf
	fsw	fa0, 88(s8)
	flw	fa0, 1108(sp)
	call	expf
	fsw	fa0, 84(s8)
	flw	fa0, 1104(sp)
	call	expf
	fsw	fa0, 80(s8)
	flw	fa0, 1100(sp)
	call	expf
	fsw	fa0, 76(s8)
	flw	fa0, 1096(sp)
	call	expf
	fsw	fa0, 72(s8)
	flw	fa0, 1092(sp)
	call	expf
	fsw	fa0, 68(s8)
	flw	fa0, 1088(sp)
	call	expf
	fsw	fa0, 64(s8)
	flw	fa0, 1084(sp)
	call	expf
	fsw	fa0, 60(s8)
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 0(s8)
	flw	fa0, 1080(sp)
	call	expf
	fsw	fa0, 56(s8)
	flw	fa0, 1076(sp)
	call	expf
	fsw	fa0, 52(s8)
	flw	fa0, 1072(sp)
	call	expf
	fsw	fa0, 48(s8)
	flw	fa0, 1068(sp)
	call	expf
	fsw	fa0, 44(s8)
	flw	fa0, 1064(sp)
	call	expf
	fsw	fa0, 40(s8)
	flw	fa0, 1060(sp)
	call	expf
	fsw	fa0, 36(s8)
	flw	fa0, 1056(sp)
	call	expf
	fsw	fa0, 32(s8)
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 12(s8)
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 8(s8)
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 4(s8)
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 28(s8)
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 24(s8)
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 20(s8)
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 16(s8)
	flw	fa0, 1020(sp)
	call	expf
	fsw	fa0, 2044(sp)
	flw	fa0, 1016(sp)
	call	expf
	fsw	fa0, 2040(sp)
	flw	fa0, 1012(sp)
	call	expf
	fsw	fa0, 2036(sp)
	flw	fa0, 1008(sp)
	call	expf
	fsw	fa0, 2032(sp)
	flw	fa0, 1004(sp)
	call	expf
	fsw	fa0, 2028(sp)
	flw	fa0, 1000(sp)
	call	expf
	fsw	fa0, 2024(sp)
	flw	fa0, 996(sp)
	call	expf
	fsw	fa0, 2020(sp)
	flw	fa0, 992(sp)
	call	expf
	fsw	fa0, 2016(sp)
	flw	fa0, 988(sp)
	call	expf
	fsw	fa0, 2012(sp)
	flw	fa0, 984(sp)
	call	expf
	fsw	fa0, 2008(sp)
	flw	fa0, 980(sp)
	call	expf
	fsw	fa0, 2004(sp)
	flw	fa0, 976(sp)
	call	expf
	fsw	fa0, 2000(sp)
	flw	fa0, 972(sp)
	call	expf
	fsw	fa0, 1996(sp)
	flw	fa0, 968(sp)
	call	expf
	fsw	fa0, 1992(sp)
	flw	fa0, 964(sp)
	call	expf
	fsw	fa0, 1988(sp)
	flw	fa0, 960(sp)
	call	expf
	fsw	fa0, 1984(sp)
	flw	fa0, 956(sp)
	call	expf
	fsw	fa0, 1980(sp)
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1920(sp)
	flw	fa0, 952(sp)
	call	expf
	fsw	fa0, 1976(sp)
	flw	fa0, 948(sp)
	call	expf
	fsw	fa0, 1972(sp)
	flw	fa0, 944(sp)
	call	expf
	fsw	fa0, 1968(sp)
	flw	fa0, 940(sp)
	call	expf
	fsw	fa0, 1964(sp)
	flw	fa0, 936(sp)
	call	expf
	fsw	fa0, 1960(sp)
	flw	fa0, 932(sp)
	call	expf
	fsw	fa0, 1956(sp)
	flw	fa0, 928(sp)
	call	expf
	fsw	fa0, 1952(sp)
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1932(sp)
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1928(sp)
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1924(sp)
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1948(sp)
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1944(sp)
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1940(sp)
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1936(sp)
	flw	fa0, 380(sp)
	call	expf
	fsw	fa0, 1404(sp)
	flw	fa0, 376(sp)
	call	expf
	fsw	fa0, 1400(sp)
	flw	fa0, 372(sp)
	call	expf
	fsw	fa0, 1396(sp)
	flw	fa0, 368(sp)
	call	expf
	fsw	fa0, 1392(sp)
	flw	fa0, 364(sp)
	call	expf
	fsw	fa0, 1388(sp)
	flw	fa0, 360(sp)
	call	expf
	fsw	fa0, 1384(sp)
	flw	fa0, 356(sp)
	call	expf
	fsw	fa0, 1380(sp)
	flw	fa0, 352(sp)
	call	expf
	fsw	fa0, 1376(sp)
	flw	fa0, 348(sp)
	call	expf
	fsw	fa0, 1372(sp)
	flw	fa0, 344(sp)
	call	expf
	fsw	fa0, 1368(sp)
	flw	fa0, 340(sp)
	call	expf
	fsw	fa0, 1364(sp)
	flw	fa0, 336(sp)
	call	expf
	fsw	fa0, 1360(sp)
	flw	fa0, 332(sp)
	call	expf
	fsw	fa0, 1356(sp)
	flw	fa0, 328(sp)
	call	expf
	fsw	fa0, 1352(sp)
	flw	fa0, 324(sp)
	call	expf
	fsw	fa0, 1348(sp)
	flw	fa0, 320(sp)
	call	expf
	fsw	fa0, 1344(sp)
	flw	fa0, 316(sp)
	call	expf
	fsw	fa0, 1340(sp)
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1280(sp)
	flw	fa0, 312(sp)
	call	expf
	fsw	fa0, 1336(sp)
	flw	fa0, 308(sp)
	call	expf
	fsw	fa0, 1332(sp)
	flw	fa0, 304(sp)
	call	expf
	fsw	fa0, 1328(sp)
	flw	fa0, 300(sp)
	call	expf
	fsw	fa0, 1324(sp)
	flw	fa0, 296(sp)
	call	expf
	fsw	fa0, 1320(sp)
	flw	fa0, 292(sp)
	call	expf
	fsw	fa0, 1316(sp)
	flw	fa0, 288(sp)
	call	expf
	fsw	fa0, 1312(sp)
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1292(sp)
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1288(sp)
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1284(sp)
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1308(sp)
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1304(sp)
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1300(sp)
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1296(sp)
	flw	fa0, 252(sp)
	call	expf
	fsw	fa0, 1276(sp)
	flw	fa0, 248(sp)
	call	expf
	fsw	fa0, 1272(sp)
	flw	fa0, 244(sp)
	call	expf
	fsw	fa0, 1268(sp)
	flw	fa0, 240(sp)
	call	expf
	fsw	fa0, 1264(sp)
	flw	fa0, 236(sp)
	call	expf
	fsw	fa0, 1260(sp)
	flw	fa0, 232(sp)
	call	expf
	fsw	fa0, 1256(sp)
	flw	fa0, 228(sp)
	call	expf
	fsw	fa0, 1252(sp)
	flw	fa0, 224(sp)
	call	expf
	fsw	fa0, 1248(sp)
	flw	fa0, 220(sp)
	call	expf
	fsw	fa0, 1244(sp)
	flw	fa0, 216(sp)
	call	expf
	fsw	fa0, 1240(sp)
	flw	fa0, 212(sp)
	call	expf
	fsw	fa0, 1236(sp)
	flw	fa0, 208(sp)
	call	expf
	fsw	fa0, 1232(sp)
	flw	fa0, 204(sp)
	call	expf
	fsw	fa0, 1228(sp)
	flw	fa0, 200(sp)
	call	expf
	fsw	fa0, 1224(sp)
	flw	fa0, 196(sp)
	call	expf
	fsw	fa0, 1220(sp)
	flw	fa0, 192(sp)
	call	expf
	fsw	fa0, 1216(sp)
	flw	fa0, 188(sp)
	call	expf
	fsw	fa0, 1212(sp)
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1152(sp)
	flw	fa0, 184(sp)
	call	expf
	fsw	fa0, 1208(sp)
	flw	fa0, 180(sp)
	call	expf
	fsw	fa0, 1204(sp)
	flw	fa0, 176(sp)
	call	expf
	fsw	fa0, 1200(sp)
	flw	fa0, 172(sp)
	call	expf
	fsw	fa0, 1196(sp)
	flw	fa0, 168(sp)
	call	expf
	fsw	fa0, 1192(sp)
	flw	fa0, 164(sp)
	call	expf
	fsw	fa0, 1188(sp)
	flw	fa0, 160(sp)
	call	expf
	fsw	fa0, 1184(sp)
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1164(sp)
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1160(sp)
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1156(sp)
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1180(sp)
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1176(sp)
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1172(sp)
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	fsw	fa0, 1168(sp)
	addi	a0, sp, 1280
	addi	a1, sp, 1152
	vsetvli	zero, s2, e32, m8, ta, ma
	vle32.v	v8, (a1)
	lui	a1, 260096
	vle32.v	v16, (a0)
	fmv.w.x	fa5, a1
	.loc	1 14 18                         # k135114449656112.py:14:18
	vfadd.vf	v8, v8, fa5
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114449656112.py:15:19
	vfdiv.vv	v8, v24, v8
	.loc	1 14 18                         # k135114449656112.py:14:18
	vfadd.vf	v16, v16, fa5
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114449656112.py:15:19
	vfdiv.vv	v16, v24, v16
	addi	a0, sp, 2047
	addi	a0, a0, 193
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114449656112.py:17:18
	vfmul.vv	v24, v8, v24
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v16, v8
	.loc	1 18 39                         # k135114449656112.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v8, v16, s2
	addi	a0, sp, 1536
	addi	a1, sp, 1408
	addi	a2, sp, 1792
	addi	a3, sp, 1664
	addi	a4, sp, 1920
	.loc	1 12 25                         # k135114449656112.py:12:25
	vsetvli	zero, s2, e32, m8, ta, ma
	vle32.v	v16, (a0)
	csrr	a0, vlenb
	li	a5, 188
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (a1)
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (a2)
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (a4)
	vle32.v	v24, (a3)
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v24, (s8)
	.loc	1 18 39                         # k135114449656112.py:18:39
	vsetvli	zero, s3, e16, m8, ta, ma
	vse16.v	v8, (s7), v0.t
	.loc	1 14 18                         # k135114449656112.py:14:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfadd.vf	v8, v16, fa5
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114449656112.py:15:19
	vfdiv.vv	v0, v16, v8
	.loc	1 14 18                         # k135114449656112.py:14:18
	vfadd.vf	v8, v24, fa5
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114449656112.py:15:19
	vfdiv.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114449656112.py:14:18
	vfadd.vf	v16, v16, fa5
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114449656112.py:15:19
	vfdiv.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114449656112.py:14:18
	vfadd.vf	v16, v16, fa5
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114449656112.py:15:19
	vfdiv.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114449656112.py:14:18
	vfadd.vf	v16, v16, fa5
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114449656112.py:15:19
	vfdiv.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114449656112.py:14:18
	vfadd.vf	v24, v16, fa5
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114449656112.py:15:19
	vfdiv.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114449656112.py:17:18
	vfmul.vv	v24, v0, v16
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v8, v8, v16
	.loc	1 18 39                         # k135114449656112.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	vnsrl.wi	v8, v24, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v16, v8, s2
	csrr	a0, vlenb
	li	a1, 130
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vse16.v	v16, (s6), v0.t
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 164
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114449656112.py:17:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 172
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v24, v16
	.loc	1 18 39                         # k135114449656112.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v24, v16, s2
	csrr	a0, vlenb
	li	a1, 131
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vse16.v	v24, (s5), v0.t
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 180
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114449656112.py:17:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 188
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v24, v16
	.loc	1 18 39                         # k135114449656112.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v24, v16, s2
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 2047
	addi	a0, a0, 193
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vse16.v	v24, (s4), v0.t
	.loc	1 18 4 epilogue_begin is_stmt 0 # k135114449656112.py:18:4
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
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
	.cfi_restore s6
	.cfi_restore s7
	.cfi_restore s8
	addi	sp, sp, 2032
	.cfi_def_cfa_offset 0
	ret
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused__unsafe_view_mul_silu_11, .Lfunc_end0-triton_poi_fused__unsafe_view_mul_silu_11
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
	.asciz	"k135114449656112.py"           # string offset=7 ; k135114449656112.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
