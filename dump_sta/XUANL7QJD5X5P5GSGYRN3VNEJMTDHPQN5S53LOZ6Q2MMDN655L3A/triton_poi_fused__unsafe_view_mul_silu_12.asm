	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused__unsafe_view_mul_silu_12 # -- Begin function triton_poi_fused__unsafe_view_mul_silu_12
	.p2align	2
	.type	triton_poi_fused__unsafe_view_mul_silu_12,@function
triton_poi_fused__unsafe_view_mul_silu_12: # @triton_poi_fused__unsafe_view_mul_silu_12
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294394832.py"
	.loc	1 2 0                           # k135114294394832.py:2:0
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
	addi	s0, sp, 2032
	.cfi_def_cfa s0, 0
	addi	sp, sp, -2048
	addi	sp, sp, -272
	csrr	a2, vlenb
	li	a4, 392
	mul	a2, a2, a4
	sub	sp, sp, a2
	andi	sp, sp, -128
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114294394832.py:4:33
	slliw	a3, a3, 9
	li	s2, 32
	li	s3, 64
	li	a4, 96
	li	a5, 128
	li	a7, 160
	li	t0, 192
	li	a2, 224
	li	t1, 256
	li	t2, 288
	li	t3, 320
	li	t4, 448
	.loc	1 5 23                          # k135114294394832.py:5:23
	vsetvli	zero, s2, e32, m8, ta, ma
	vid.v	v8
	vadd.vx	v16, v8, a2
	csrr	a2, vlenb
	li	a6, 360
	mul	a2, a2, a6
	add	a2, sp, a2
	lui	a6, 1
	addi	a6, a6, 160
	add	a2, a2, a6
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	li	a2, 480
	vadd.vx	v24, v8, a2
	lui	a6, 17
	addi	a6, a6, -1536
	vor.vx	v0, v8, a3
	vor.vx	v24, v24, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v16, v0, a6
	csrr	a2, vlenb
	li	t5, 263
	mul	a2, a2, t5
	add	a2, sp, a2
	lui	t5, 1
	addi	t5, t5, 160
	add	a2, a2, t5
	vs1r.v	v16, (a2)                       # vscale x 8-byte Folded Spill
	vmslt.vx	v16, v24, a6
	csrr	a2, vlenb
	li	t5, 384
	mul	a2, a2, t5
	add	a2, sp, a2
	lui	t5, 1
	addi	t5, t5, 160
	add	a2, a2, t5
	vs1r.v	v16, (a2)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v24, v8, t4
	li	a2, 416
	vor.vx	v24, v24, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v2, v24, a6
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v24, v8, a2
	li	a2, 384
	vor.vx	v24, v24, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v16, v24, a6
	csrr	t4, vlenb
	li	t5, 376
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 1
	addi	t5, t5, 160
	add	t4, t4, t5
	vs1r.v	v16, (t4)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v24, v8, a2
	li	a2, 352
	vor.vx	v24, v24, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v3, v24, a6
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v24, v8, a2
	.loc	1 8 34                          # k135114294394832.py:8:34
	slli	a2, a3, 1
	.loc	1 5 23                          # k135114294394832.py:5:23
	vor.vx	v24, v24, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v16, v24, a6
	csrr	t4, vlenb
	li	t5, 368
	mul	t4, t4, t5
	add	t4, sp, t4
	lui	t5, 1
	addi	t5, t5, 160
	add	t4, t4, t5
	vs1r.v	v16, (t4)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v24, v8, t3
	vor.vx	v24, v24, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v4, v24, a6
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v24, v8, t2
	vor.vx	v24, v24, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v16, v24, a6
	csrr	t2, vlenb
	li	t3, 352
	mul	t2, t2, t3
	add	t2, sp, t2
	lui	t3, 1
	addi	t3, t3, 160
	add	t2, t2, t3
	vs1r.v	v16, (t2)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v24, v8, t1
	csrr	t1, vlenb
	li	t2, 360
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 1
	addi	t2, t2, 160
	add	t1, t1, t2
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vor.vx	v16, v16, a3
	vor.vx	v24, v24, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v5, v24, a6
	vmslt.vx	v24, v16, a6
	csrr	t1, vlenb
	li	t2, 360
	mul	t1, t1, t2
	add	t1, sp, t1
	lui	t2, 1
	addi	t2, t2, 160
	add	t1, t1, t2
	vs1r.v	v24, (t1)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v16, v8, t0
	vor.vx	v16, v16, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v6, v16, a6
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v16, v8, a7
	vor.vx	v16, v16, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v24, v16, a6
	csrr	a7, vlenb
	li	t0, 344
	mul	a7, a7, t0
	add	a7, sp, a7
	lui	t0, 1
	addi	t0, t0, 160
	add	a7, a7, t0
	vs1r.v	v24, (a7)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v16, v8, a5
	vor.vx	v16, v16, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v7, v16, a6
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v16, v8, a4
	vor.vx	v16, v16, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v1, v16, a6
	.loc	1 5 23                          # k135114294394832.py:5:23
	vadd.vx	v16, v8, s3
	vor.vx	v16, v16, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v0, v16, a6
	.loc	1 8 39                          # k135114294394832.py:8:39
	vsetvli	zero, s3, e16, m8, ta, ma
	vmv.v.i	v24, 0
	.loc	1 5 23                          # k135114294394832.py:5:23
	vsetvli	zero, s2, e32, m8, ta, ma
	vadd.vx	v8, v8, s2
	.loc	1 8 34                          # k135114294394832.py:8:34
	add	s8, a0, a2
	.loc	1 5 23                          # k135114294394832.py:5:23
	vor.vx	v8, v8, a3
	.loc	1 6 21                          # k135114294394832.py:6:21
	vmslt.vx	v16, v8, a6
	.loc	1 8 39                          # k135114294394832.py:8:39
	vmv8r.v	v8, v24
	addi	a0, s8, 128
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	csrr	a3, vlenb
	li	a4, 384
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vl1r.v	v17, (a3)                       # vscale x 8-byte Folded Reload
	.loc	1 6 21                          # k135114294394832.py:6:21
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v2, v17, 4
	csrr	a3, vlenb
	li	a4, 258
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vs1r.v	v2, (a3)                        # vscale x 8-byte Folded Spill
	csrr	a3, vlenb
	li	a4, 376
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vl1r.v	v17, (a3)                       # vscale x 8-byte Folded Reload
	vslideup.vi	v3, v17, 4
	csrr	a3, vlenb
	li	a4, 259
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vs1r.v	v3, (a3)                        # vscale x 8-byte Folded Spill
	csrr	a3, vlenb
	li	a4, 368
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vl1r.v	v17, (a3)                       # vscale x 8-byte Folded Reload
	vslideup.vi	v4, v17, 4
	csrr	a3, vlenb
	li	a4, 260
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vs1r.v	v4, (a3)                        # vscale x 8-byte Folded Spill
	csrr	a3, vlenb
	li	a4, 352
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vl1r.v	v17, (a3)                       # vscale x 8-byte Folded Reload
	vslideup.vi	v5, v17, 4
	csrr	a3, vlenb
	li	a4, 261
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vs1r.v	v5, (a3)                        # vscale x 8-byte Folded Spill
	csrr	a3, vlenb
	li	a4, 360
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vl1r.v	v17, (a3)                       # vscale x 8-byte Folded Reload
	vslideup.vi	v6, v17, 4
	csrr	a3, vlenb
	li	a4, 262
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vs1r.v	v6, (a3)                        # vscale x 8-byte Folded Spill
	csrr	a3, vlenb
	li	a4, 344
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vl1r.v	v17, (a3)                       # vscale x 8-byte Folded Reload
	vslideup.vi	v7, v17, 4
	csrr	a3, vlenb
	slli	a3, a3, 8
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vs1r.v	v7, (a3)                        # vscale x 8-byte Folded Spill
	vslideup.vi	v0, v1, 4
	csrr	a3, vlenb
	slli	a4, a3, 8
	add	a3, a4, a3
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vs1r.v	v0, (a3)                        # vscale x 8-byte Folded Spill
	csrr	a3, vlenb
	li	a4, 263
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vl1r.v	v17, (a3)                       # vscale x 8-byte Folded Reload
	vslideup.vi	v17, v16, 4
	csrr	a3, vlenb
	li	a4, 263
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 1
	addi	a4, a4, 160
	add	a3, a3, a4
	vs1r.v	v17, (a3)                       # vscale x 8-byte Folded Spill
	.loc	1 8 39                          # k135114294394832.py:8:39
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v8, (a0), v0.t
	vmv8r.v	v16, v24
	addi	a0, s8, 256
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	vmv1r.v	v0, v7
	vle16.v	v16, (a0), v0.t
	addi	s6, s8, 384
	.loc	1 8 49 is_stmt 0                # k135114294394832.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v8, s2
	csrr	a0, vlenb
	li	a3, 288
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	csrr	a0, vlenb
	li	a3, 272
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 8 39                          # k135114294394832.py:8:39
	vmv8r.v	v8, v24
	csrr	a0, vlenb
	li	a3, 262
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v8, (s6), v0.t
	addi	s7, s8, 512
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v16, s2
	csrr	a0, vlenb
	li	a3, 304
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a0, vlenb
	li	a3, 280
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 8 39                          # k135114294394832.py:8:39
	vmv8r.v	v16, v24
	csrr	a0, vlenb
	li	a3, 261
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v16, (s7), v0.t
	addi	s9, s8, 640
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v8, s2
	csrr	a0, vlenb
	li	a3, 320
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	csrr	a0, vlenb
	li	a3, 296
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 8 39                          # k135114294394832.py:8:39
	vmv8r.v	v8, v24
	csrr	a0, vlenb
	li	a3, 260
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v8, (s9), v0.t
	csrr	a0, vlenb
	li	a3, 384
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	s10, s8, 768
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v8, v16, s2
	csrr	a0, vlenb
	li	a3, 336
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v8, v16
	csrr	a0, vlenb
	li	a3, 312
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 8 39                          # k135114294394832.py:8:39
	vmv8r.v	v8, v24
	csrr	a0, vlenb
	li	a3, 259
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v8, (s10), v0.t
	addi	s11, s8, 896
	csrr	a0, vlenb
	li	a3, 384
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v16, v0, s2
	csrr	a0, vlenb
	li	a3, 352
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v16, v0
	csrr	a0, vlenb
	li	a3, 328
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 8 39                          # k135114294394832.py:8:39
	vmv8r.v	v16, v24
	csrr	a0, vlenb
	li	a3, 258
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v16, (s11), v0.t
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v8, s2
	csrr	a0, vlenb
	li	a3, 368
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	csrr	a0, vlenb
	li	a3, 344
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 8 39                          # k135114294394832.py:8:39
	vmv8r.v	v8, v24
	csrr	a0, vlenb
	li	a3, 263
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v8, (s8), v0.t
	.loc	1 9 30 is_stmt 1                # k135114294394832.py:9:30
	add	a1, a1, a2
	.loc	1 9 35 is_stmt 0                # k135114294394832.py:9:35
	addi	a0, a1, 128
	.loc	1 8 49 is_stmt 1                # k135114294394832.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v16, s2
	csrr	a2, vlenb
	li	a3, 384
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a2, vlenb
	li	a3, 360
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 9 35                          # k135114294394832.py:9:35
	vmv8r.v	v16, v24
	csrr	a2, vlenb
	slli	a3, a2, 8
	add	a2, a3, a2
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v16, (a0), v0.t
	addi	a0, a1, 256
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v8, s2
	csrr	a2, vlenb
	li	a3, 248
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	csrr	a2, vlenb
	li	a3, 376
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 9 35                          # k135114294394832.py:9:35
	vmv8r.v	v8, v24
	csrr	a2, vlenb
	slli	a2, a2, 8
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v8, (a0), v0.t
	addi	a0, a1, 384
	.loc	1 9 45 is_stmt 0                # k135114294394832.py:9:45
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v16, s2
	csrr	a2, vlenb
	li	a3, 184
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a2, vlenb
	li	a3, 192
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 9 35                          # k135114294394832.py:9:35
	vmv8r.v	v16, v24
	csrr	a2, vlenb
	li	a3, 262
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v16, (a0), v0.t
	addi	a0, a1, 512
	.loc	1 9 45                          # k135114294394832.py:9:45
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v8, s2
	csrr	a2, vlenb
	li	a3, 168
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	csrr	a2, vlenb
	li	a3, 176
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 9 35                          # k135114294394832.py:9:35
	vmv8r.v	v8, v24
	csrr	a2, vlenb
	li	a3, 261
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v8, (a0), v0.t
	addi	a0, a1, 640
	.loc	1 9 45                          # k135114294394832.py:9:45
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v16, s2
	csrr	a2, vlenb
	li	a3, 136
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a2, vlenb
	li	a3, 144
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 9 35                          # k135114294394832.py:9:35
	vmv8r.v	v16, v24
	csrr	a2, vlenb
	li	a3, 260
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v16, (a0), v0.t
	addi	a0, a1, 768
	.loc	1 9 45                          # k135114294394832.py:9:45
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v8, s2
	csrr	a2, vlenb
	li	a3, 88
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	csrr	a2, vlenb
	li	a3, 104
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 9 35                          # k135114294394832.py:9:35
	vmv8r.v	v8, v24
	csrr	a2, vlenb
	li	a3, 259
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v8, (a0), v0.t
	addi	a0, a1, 896
	.loc	1 9 45                          # k135114294394832.py:9:45
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v16, s2
	csrr	a2, vlenb
	li	a3, 72
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a2, vlenb
	li	a3, 80
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 9 35                          # k135114294394832.py:9:35
	vmv8r.v	v16, v24
	csrr	a2, vlenb
	li	a3, 258
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, s3, e16, m8, ta, mu
	vle16.v	v16, (a0), v0.t
	csrr	a0, vlenb
	li	a2, 263
	mul	a0, a0, a2
	add	a0, sp, a0
	lui	a2, 1
	addi	a2, a2, 160
	add	a0, a0, a2
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vle16.v	v24, (a1), v0.t
	.loc	1 9 45                          # k135114294394832.py:9:45
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v0, v8, s2
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v8, v16, s2
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v8, v16
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, s2
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, s2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	lui	a0, 1
	addi	a0, a0, 32
	add	s5, sp, a0
	addi	s4, sp, 2047
	addi	s4, s4, 33
	fmv.w.x	fa5, zero
	addi	a0, sp, 512
	addi	a1, sp, 384
	csrr	a2, vlenb
	li	a3, 272
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49 is_stmt 1                # k135114294394832.py:8:49
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 232
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 264
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a0)
	addi	a0, sp, 768
	csrr	a2, vlenb
	li	a3, 288
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 240
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 272
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a1)
	addi	a1, sp, 640
	csrr	a2, vlenb
	li	a3, 280
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 224
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 280
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a0)
	addi	a0, sp, 1024
	csrr	a2, vlenb
	li	a3, 304
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 216
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 288
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a1)
	addi	a1, sp, 896
	csrr	a2, vlenb
	li	a3, 296
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 208
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 296
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a0)
	addi	a0, sp, 1280
	csrr	a2, vlenb
	li	a3, 320
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 200
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 304
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a1)
	addi	a1, sp, 1152
	csrr	a2, vlenb
	li	a3, 312
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 152
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 312
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a0)
	addi	a0, sp, 1536
	csrr	a2, vlenb
	li	a3, 336
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 160
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 320
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a1)
	addi	a1, sp, 1408
	csrr	a2, vlenb
	li	a3, 328
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	slli	a2, a2, 7
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 328
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a0)
	addi	a0, sp, 1792
	csrr	a2, vlenb
	li	a3, 352
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 120
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 336
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a1)
	addi	a1, sp, 1664
	csrr	a2, vlenb
	li	a3, 344
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 112
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 344
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a0)
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a2, vlenb
	li	a3, 368
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v16, (a2)                       # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	li	a3, 96
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 352
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a1)
	addi	a1, sp, 1920
	csrr	a2, vlenb
	li	a3, 360
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vl8r.v	v8, (a2)                        # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vsll.vi	v8, v8, 16
	csrr	a2, vlenb
	slli	a2, a2, 5
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a2, vlenb
	li	a3, 360
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 1
	addi	a3, a3, 160
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a0)
	addi	a2, sp, 256
	csrr	a0, vlenb
	li	a3, 384
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a3, 24
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v8, fa5
	csrr	a0, vlenb
	li	a3, 368
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 1
	addi	a3, a3, 160
	add	a0, a0, a3
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a1)
	addi	a0, sp, 128
	csrr	a1, vlenb
	li	a3, 248
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 8 49                          # k135114294394832.py:8:49
	vzext.vf2	v8, v16
	csrr	a1, vlenb
	li	a3, 376
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a3, 248
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsll.vi	v24, v8, 16
	lui	a1, 1
	addi	a1, a1, 160
	add	a1, sp, a1
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a3, 184
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 9 45                          # k135114294394832.py:9:45
	vzext.vf2	v8, v16
	csrr	a1, vlenb
	li	a3, 192
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a3, 192
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a3, 168
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vzext.vf2	v16, v0
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 184
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a3, 176
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 176
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a3, 136
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vzext.vf2	v8, v0
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a3, 168
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a3, 144
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a3, 144
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a3, 248
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v16, v16, fa5
	csrr	a1, vlenb
	li	a3, 376
	mul	a1, a1, a3
	add	a1, sp, a1
	lui	a3, 1
	addi	a3, a3, 160
	add	a1, a1, a3
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v16, (a2)
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 9 45                          # k135114294394832.py:9:45
	vzext.vf2	v16, v0
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vzext.vf2	v8, v0
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vzext.vf2	v16, v0
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vzext.vf2	v8, v0
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vzext.vf2	v16, v0
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vsll.vi	v8, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 11 12                         # k135114294394832.py:11:12
	vfrsub.vf	v8, v24, fa5
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 1
	addi	a2, a2, 160
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 12 25                         # k135114294394832.py:12:25
	vse32.v	v8, (a0)
	flw	fa0, 636(sp)
	call	expf
	fsw	fa0, 604(s4)
	flw	fa0, 632(sp)
	call	expf
	fsw	fa0, 600(s4)
	flw	fa0, 628(sp)
	call	expf
	fsw	fa0, 596(s4)
	flw	fa0, 624(sp)
	call	expf
	fsw	fa0, 592(s4)
	flw	fa0, 620(sp)
	call	expf
	fsw	fa0, 588(s4)
	flw	fa0, 616(sp)
	call	expf
	fsw	fa0, 584(s4)
	flw	fa0, 612(sp)
	call	expf
	fsw	fa0, 580(s4)
	flw	fa0, 608(sp)
	call	expf
	fsw	fa0, 576(s4)
	flw	fa0, 604(sp)
	call	expf
	fsw	fa0, 572(s4)
	flw	fa0, 600(sp)
	call	expf
	fsw	fa0, 568(s4)
	flw	fa0, 596(sp)
	call	expf
	fsw	fa0, 564(s4)
	flw	fa0, 592(sp)
	call	expf
	fsw	fa0, 560(s4)
	flw	fa0, 588(sp)
	call	expf
	fsw	fa0, 556(s4)
	flw	fa0, 584(sp)
	call	expf
	fsw	fa0, 552(s4)
	flw	fa0, 580(sp)
	call	expf
	fsw	fa0, 548(s4)
	flw	fa0, 576(sp)
	call	expf
	fsw	fa0, 544(s4)
	flw	fa0, 572(sp)
	call	expf
	fsw	fa0, 540(s4)
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 480(s4)
	flw	fa0, 568(sp)
	call	expf
	fsw	fa0, 536(s4)
	flw	fa0, 564(sp)
	call	expf
	fsw	fa0, 532(s4)
	flw	fa0, 560(sp)
	call	expf
	fsw	fa0, 528(s4)
	flw	fa0, 556(sp)
	call	expf
	fsw	fa0, 524(s4)
	flw	fa0, 552(sp)
	call	expf
	fsw	fa0, 520(s4)
	flw	fa0, 548(sp)
	call	expf
	fsw	fa0, 516(s4)
	flw	fa0, 544(sp)
	call	expf
	fsw	fa0, 512(s4)
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 492(s4)
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 488(s4)
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 484(s4)
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 508(s4)
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 504(s4)
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 500(s4)
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 496(s4)
	flw	fa0, 508(sp)
	call	expf
	fsw	fa0, 476(s4)
	flw	fa0, 504(sp)
	call	expf
	fsw	fa0, 472(s4)
	flw	fa0, 500(sp)
	call	expf
	fsw	fa0, 468(s4)
	flw	fa0, 496(sp)
	call	expf
	fsw	fa0, 464(s4)
	flw	fa0, 492(sp)
	call	expf
	fsw	fa0, 460(s4)
	flw	fa0, 488(sp)
	call	expf
	fsw	fa0, 456(s4)
	flw	fa0, 484(sp)
	call	expf
	fsw	fa0, 452(s4)
	flw	fa0, 480(sp)
	call	expf
	fsw	fa0, 448(s4)
	flw	fa0, 476(sp)
	call	expf
	fsw	fa0, 444(s4)
	flw	fa0, 472(sp)
	call	expf
	fsw	fa0, 440(s4)
	flw	fa0, 468(sp)
	call	expf
	fsw	fa0, 436(s4)
	flw	fa0, 464(sp)
	call	expf
	fsw	fa0, 432(s4)
	flw	fa0, 460(sp)
	call	expf
	fsw	fa0, 428(s4)
	flw	fa0, 456(sp)
	call	expf
	fsw	fa0, 424(s4)
	flw	fa0, 452(sp)
	call	expf
	fsw	fa0, 420(s4)
	flw	fa0, 448(sp)
	call	expf
	fsw	fa0, 416(s4)
	flw	fa0, 444(sp)
	call	expf
	fsw	fa0, 412(s4)
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 352(s4)
	flw	fa0, 440(sp)
	call	expf
	fsw	fa0, 408(s4)
	flw	fa0, 436(sp)
	call	expf
	fsw	fa0, 404(s4)
	flw	fa0, 432(sp)
	call	expf
	fsw	fa0, 400(s4)
	flw	fa0, 428(sp)
	call	expf
	fsw	fa0, 396(s4)
	flw	fa0, 424(sp)
	call	expf
	fsw	fa0, 392(s4)
	flw	fa0, 420(sp)
	call	expf
	fsw	fa0, 388(s4)
	flw	fa0, 416(sp)
	call	expf
	fsw	fa0, 384(s4)
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 364(s4)
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 360(s4)
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 356(s4)
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 380(s4)
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 376(s4)
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 372(s4)
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 368(s4)
	flw	fa0, 892(sp)
	call	expf
	fsw	fa0, 860(s4)
	flw	fa0, 888(sp)
	call	expf
	fsw	fa0, 856(s4)
	flw	fa0, 884(sp)
	call	expf
	fsw	fa0, 852(s4)
	flw	fa0, 880(sp)
	call	expf
	fsw	fa0, 848(s4)
	flw	fa0, 876(sp)
	call	expf
	fsw	fa0, 844(s4)
	flw	fa0, 872(sp)
	call	expf
	fsw	fa0, 840(s4)
	flw	fa0, 868(sp)
	call	expf
	fsw	fa0, 836(s4)
	flw	fa0, 864(sp)
	call	expf
	fsw	fa0, 832(s4)
	flw	fa0, 860(sp)
	call	expf
	fsw	fa0, 828(s4)
	flw	fa0, 856(sp)
	call	expf
	fsw	fa0, 824(s4)
	flw	fa0, 852(sp)
	call	expf
	fsw	fa0, 820(s4)
	flw	fa0, 848(sp)
	call	expf
	fsw	fa0, 816(s4)
	flw	fa0, 844(sp)
	call	expf
	fsw	fa0, 812(s4)
	flw	fa0, 840(sp)
	call	expf
	fsw	fa0, 808(s4)
	flw	fa0, 836(sp)
	call	expf
	fsw	fa0, 804(s4)
	flw	fa0, 832(sp)
	call	expf
	fsw	fa0, 800(s4)
	flw	fa0, 828(sp)
	call	expf
	fsw	fa0, 796(s4)
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 736(s4)
	flw	fa0, 824(sp)
	call	expf
	fsw	fa0, 792(s4)
	flw	fa0, 820(sp)
	call	expf
	fsw	fa0, 788(s4)
	flw	fa0, 816(sp)
	call	expf
	fsw	fa0, 784(s4)
	flw	fa0, 812(sp)
	call	expf
	fsw	fa0, 780(s4)
	flw	fa0, 808(sp)
	call	expf
	fsw	fa0, 776(s4)
	flw	fa0, 804(sp)
	call	expf
	fsw	fa0, 772(s4)
	flw	fa0, 800(sp)
	call	expf
	fsw	fa0, 768(s4)
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 748(s4)
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 744(s4)
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 740(s4)
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 764(s4)
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 760(s4)
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 756(s4)
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 752(s4)
	flw	fa0, 764(sp)
	call	expf
	fsw	fa0, 732(s4)
	flw	fa0, 760(sp)
	call	expf
	fsw	fa0, 728(s4)
	flw	fa0, 756(sp)
	call	expf
	fsw	fa0, 724(s4)
	flw	fa0, 752(sp)
	call	expf
	fsw	fa0, 720(s4)
	flw	fa0, 748(sp)
	call	expf
	fsw	fa0, 716(s4)
	flw	fa0, 744(sp)
	call	expf
	fsw	fa0, 712(s4)
	flw	fa0, 740(sp)
	call	expf
	fsw	fa0, 708(s4)
	flw	fa0, 736(sp)
	call	expf
	fsw	fa0, 704(s4)
	flw	fa0, 732(sp)
	call	expf
	fsw	fa0, 700(s4)
	flw	fa0, 728(sp)
	call	expf
	fsw	fa0, 696(s4)
	flw	fa0, 724(sp)
	call	expf
	fsw	fa0, 692(s4)
	flw	fa0, 720(sp)
	call	expf
	fsw	fa0, 688(s4)
	flw	fa0, 716(sp)
	call	expf
	fsw	fa0, 684(s4)
	flw	fa0, 712(sp)
	call	expf
	fsw	fa0, 680(s4)
	flw	fa0, 708(sp)
	call	expf
	fsw	fa0, 676(s4)
	flw	fa0, 704(sp)
	call	expf
	fsw	fa0, 672(s4)
	flw	fa0, 700(sp)
	call	expf
	fsw	fa0, 668(s4)
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 608(s4)
	flw	fa0, 696(sp)
	call	expf
	fsw	fa0, 664(s4)
	flw	fa0, 692(sp)
	call	expf
	fsw	fa0, 660(s4)
	flw	fa0, 688(sp)
	call	expf
	fsw	fa0, 656(s4)
	flw	fa0, 684(sp)
	call	expf
	fsw	fa0, 652(s4)
	flw	fa0, 680(sp)
	call	expf
	fsw	fa0, 648(s4)
	flw	fa0, 676(sp)
	call	expf
	fsw	fa0, 644(s4)
	flw	fa0, 672(sp)
	call	expf
	fsw	fa0, 640(s4)
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 620(s4)
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 616(s4)
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 612(s4)
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 636(s4)
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 632(s4)
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 628(s4)
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 624(s4)
	flw	fa0, 1148(sp)
	call	expf
	fsw	fa0, 1116(s4)
	flw	fa0, 1144(sp)
	call	expf
	fsw	fa0, 1112(s4)
	flw	fa0, 1140(sp)
	call	expf
	fsw	fa0, 1108(s4)
	flw	fa0, 1136(sp)
	call	expf
	fsw	fa0, 1104(s4)
	flw	fa0, 1132(sp)
	call	expf
	fsw	fa0, 1100(s4)
	flw	fa0, 1128(sp)
	call	expf
	fsw	fa0, 1096(s4)
	flw	fa0, 1124(sp)
	call	expf
	fsw	fa0, 1092(s4)
	flw	fa0, 1120(sp)
	call	expf
	fsw	fa0, 1088(s4)
	flw	fa0, 1116(sp)
	call	expf
	fsw	fa0, 1084(s4)
	flw	fa0, 1112(sp)
	call	expf
	fsw	fa0, 1080(s4)
	flw	fa0, 1108(sp)
	call	expf
	fsw	fa0, 1076(s4)
	flw	fa0, 1104(sp)
	call	expf
	fsw	fa0, 1072(s4)
	flw	fa0, 1100(sp)
	call	expf
	fsw	fa0, 1068(s4)
	flw	fa0, 1096(sp)
	call	expf
	fsw	fa0, 1064(s4)
	flw	fa0, 1092(sp)
	call	expf
	fsw	fa0, 1060(s4)
	flw	fa0, 1088(sp)
	call	expf
	fsw	fa0, 1056(s4)
	flw	fa0, 1084(sp)
	call	expf
	fsw	fa0, 1052(s4)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 992(s4)
	flw	fa0, 1080(sp)
	call	expf
	fsw	fa0, 1048(s4)
	flw	fa0, 1076(sp)
	call	expf
	fsw	fa0, 1044(s4)
	flw	fa0, 1072(sp)
	call	expf
	fsw	fa0, 1040(s4)
	flw	fa0, 1068(sp)
	call	expf
	fsw	fa0, 1036(s4)
	flw	fa0, 1064(sp)
	call	expf
	fsw	fa0, 1032(s4)
	flw	fa0, 1060(sp)
	call	expf
	fsw	fa0, 1028(s4)
	flw	fa0, 1056(sp)
	call	expf
	fsw	fa0, 1024(s4)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1004(s4)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1000(s4)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 996(s4)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1020(s4)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1016(s4)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1012(s4)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1008(s4)
	flw	fa0, 1020(sp)
	call	expf
	fsw	fa0, 988(s4)
	flw	fa0, 1016(sp)
	call	expf
	fsw	fa0, 984(s4)
	flw	fa0, 1012(sp)
	call	expf
	fsw	fa0, 980(s4)
	flw	fa0, 1008(sp)
	call	expf
	fsw	fa0, 976(s4)
	flw	fa0, 1004(sp)
	call	expf
	fsw	fa0, 972(s4)
	flw	fa0, 1000(sp)
	call	expf
	fsw	fa0, 968(s4)
	flw	fa0, 996(sp)
	call	expf
	fsw	fa0, 964(s4)
	flw	fa0, 992(sp)
	call	expf
	fsw	fa0, 960(s4)
	flw	fa0, 988(sp)
	call	expf
	fsw	fa0, 956(s4)
	flw	fa0, 984(sp)
	call	expf
	fsw	fa0, 952(s4)
	flw	fa0, 980(sp)
	call	expf
	fsw	fa0, 948(s4)
	flw	fa0, 976(sp)
	call	expf
	fsw	fa0, 944(s4)
	flw	fa0, 972(sp)
	call	expf
	fsw	fa0, 940(s4)
	flw	fa0, 968(sp)
	call	expf
	fsw	fa0, 936(s4)
	flw	fa0, 964(sp)
	call	expf
	fsw	fa0, 932(s4)
	flw	fa0, 960(sp)
	call	expf
	fsw	fa0, 928(s4)
	flw	fa0, 956(sp)
	call	expf
	fsw	fa0, 924(s4)
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 864(s4)
	flw	fa0, 952(sp)
	call	expf
	fsw	fa0, 920(s4)
	flw	fa0, 948(sp)
	call	expf
	fsw	fa0, 916(s4)
	flw	fa0, 944(sp)
	call	expf
	fsw	fa0, 912(s4)
	flw	fa0, 940(sp)
	call	expf
	fsw	fa0, 908(s4)
	flw	fa0, 936(sp)
	call	expf
	fsw	fa0, 904(s4)
	flw	fa0, 932(sp)
	call	expf
	fsw	fa0, 900(s4)
	flw	fa0, 928(sp)
	call	expf
	fsw	fa0, 896(s4)
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 876(s4)
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 872(s4)
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 868(s4)
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 892(s4)
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 888(s4)
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 884(s4)
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 880(s4)
	flw	fa0, 1404(sp)
	call	expf
	fsw	fa0, 1372(s4)
	flw	fa0, 1400(sp)
	call	expf
	fsw	fa0, 1368(s4)
	flw	fa0, 1396(sp)
	call	expf
	fsw	fa0, 1364(s4)
	flw	fa0, 1392(sp)
	call	expf
	fsw	fa0, 1360(s4)
	flw	fa0, 1388(sp)
	call	expf
	fsw	fa0, 1356(s4)
	flw	fa0, 1384(sp)
	call	expf
	fsw	fa0, 1352(s4)
	flw	fa0, 1380(sp)
	call	expf
	fsw	fa0, 1348(s4)
	flw	fa0, 1376(sp)
	call	expf
	fsw	fa0, 1344(s4)
	flw	fa0, 1372(sp)
	call	expf
	fsw	fa0, 1340(s4)
	flw	fa0, 1368(sp)
	call	expf
	fsw	fa0, 1336(s4)
	flw	fa0, 1364(sp)
	call	expf
	fsw	fa0, 1332(s4)
	flw	fa0, 1360(sp)
	call	expf
	fsw	fa0, 1328(s4)
	flw	fa0, 1356(sp)
	call	expf
	fsw	fa0, 1324(s4)
	flw	fa0, 1352(sp)
	call	expf
	fsw	fa0, 1320(s4)
	flw	fa0, 1348(sp)
	call	expf
	fsw	fa0, 1316(s4)
	flw	fa0, 1344(sp)
	call	expf
	fsw	fa0, 1312(s4)
	flw	fa0, 1340(sp)
	call	expf
	fsw	fa0, 1308(s4)
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1248(s4)
	flw	fa0, 1336(sp)
	call	expf
	fsw	fa0, 1304(s4)
	flw	fa0, 1332(sp)
	call	expf
	fsw	fa0, 1300(s4)
	flw	fa0, 1328(sp)
	call	expf
	fsw	fa0, 1296(s4)
	flw	fa0, 1324(sp)
	call	expf
	fsw	fa0, 1292(s4)
	flw	fa0, 1320(sp)
	call	expf
	fsw	fa0, 1288(s4)
	flw	fa0, 1316(sp)
	call	expf
	fsw	fa0, 1284(s4)
	flw	fa0, 1312(sp)
	call	expf
	fsw	fa0, 1280(s4)
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1260(s4)
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1256(s4)
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1252(s4)
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1276(s4)
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1272(s4)
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1268(s4)
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1264(s4)
	flw	fa0, 1276(sp)
	call	expf
	fsw	fa0, 1244(s4)
	flw	fa0, 1272(sp)
	call	expf
	fsw	fa0, 1240(s4)
	flw	fa0, 1268(sp)
	call	expf
	fsw	fa0, 1236(s4)
	flw	fa0, 1264(sp)
	call	expf
	fsw	fa0, 1232(s4)
	flw	fa0, 1260(sp)
	call	expf
	fsw	fa0, 1228(s4)
	flw	fa0, 1256(sp)
	call	expf
	fsw	fa0, 1224(s4)
	flw	fa0, 1252(sp)
	call	expf
	fsw	fa0, 1220(s4)
	flw	fa0, 1248(sp)
	call	expf
	fsw	fa0, 1216(s4)
	flw	fa0, 1244(sp)
	call	expf
	fsw	fa0, 1212(s4)
	flw	fa0, 1240(sp)
	call	expf
	fsw	fa0, 1208(s4)
	flw	fa0, 1236(sp)
	call	expf
	fsw	fa0, 1204(s4)
	flw	fa0, 1232(sp)
	call	expf
	fsw	fa0, 1200(s4)
	flw	fa0, 1228(sp)
	call	expf
	fsw	fa0, 1196(s4)
	flw	fa0, 1224(sp)
	call	expf
	fsw	fa0, 1192(s4)
	flw	fa0, 1220(sp)
	call	expf
	fsw	fa0, 1188(s4)
	flw	fa0, 1216(sp)
	call	expf
	fsw	fa0, 1184(s4)
	flw	fa0, 1212(sp)
	call	expf
	fsw	fa0, 1180(s4)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1120(s4)
	flw	fa0, 1208(sp)
	call	expf
	fsw	fa0, 1176(s4)
	flw	fa0, 1204(sp)
	call	expf
	fsw	fa0, 1172(s4)
	flw	fa0, 1200(sp)
	call	expf
	fsw	fa0, 1168(s4)
	flw	fa0, 1196(sp)
	call	expf
	fsw	fa0, 1164(s4)
	flw	fa0, 1192(sp)
	call	expf
	fsw	fa0, 1160(s4)
	flw	fa0, 1188(sp)
	call	expf
	fsw	fa0, 1156(s4)
	flw	fa0, 1184(sp)
	call	expf
	fsw	fa0, 1152(s4)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1132(s4)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1128(s4)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1124(s4)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1148(s4)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1144(s4)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1140(s4)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1136(s4)
	flw	fa0, 1660(sp)
	call	expf
	fsw	fa0, 1628(s4)
	flw	fa0, 1656(sp)
	call	expf
	fsw	fa0, 1624(s4)
	flw	fa0, 1652(sp)
	call	expf
	fsw	fa0, 1620(s4)
	flw	fa0, 1648(sp)
	call	expf
	fsw	fa0, 1616(s4)
	flw	fa0, 1644(sp)
	call	expf
	fsw	fa0, 1612(s4)
	flw	fa0, 1640(sp)
	call	expf
	fsw	fa0, 1608(s4)
	flw	fa0, 1636(sp)
	call	expf
	fsw	fa0, 1604(s4)
	flw	fa0, 1632(sp)
	call	expf
	fsw	fa0, 1600(s4)
	flw	fa0, 1628(sp)
	call	expf
	fsw	fa0, 1596(s4)
	flw	fa0, 1624(sp)
	call	expf
	fsw	fa0, 1592(s4)
	flw	fa0, 1620(sp)
	call	expf
	fsw	fa0, 1588(s4)
	flw	fa0, 1616(sp)
	call	expf
	fsw	fa0, 1584(s4)
	flw	fa0, 1612(sp)
	call	expf
	fsw	fa0, 1580(s4)
	flw	fa0, 1608(sp)
	call	expf
	fsw	fa0, 1576(s4)
	flw	fa0, 1604(sp)
	call	expf
	fsw	fa0, 1572(s4)
	flw	fa0, 1600(sp)
	call	expf
	fsw	fa0, 1568(s4)
	flw	fa0, 1596(sp)
	call	expf
	fsw	fa0, 1564(s4)
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1504(s4)
	flw	fa0, 1592(sp)
	call	expf
	fsw	fa0, 1560(s4)
	flw	fa0, 1588(sp)
	call	expf
	fsw	fa0, 1556(s4)
	flw	fa0, 1584(sp)
	call	expf
	fsw	fa0, 1552(s4)
	flw	fa0, 1580(sp)
	call	expf
	fsw	fa0, 1548(s4)
	flw	fa0, 1576(sp)
	call	expf
	fsw	fa0, 1544(s4)
	flw	fa0, 1572(sp)
	call	expf
	fsw	fa0, 1540(s4)
	flw	fa0, 1568(sp)
	call	expf
	fsw	fa0, 1536(s4)
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1516(s4)
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1512(s4)
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1508(s4)
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1532(s4)
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1528(s4)
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1524(s4)
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1520(s4)
	flw	fa0, 1532(sp)
	call	expf
	fsw	fa0, 1500(s4)
	flw	fa0, 1528(sp)
	call	expf
	fsw	fa0, 1496(s4)
	flw	fa0, 1524(sp)
	call	expf
	fsw	fa0, 1492(s4)
	flw	fa0, 1520(sp)
	call	expf
	fsw	fa0, 1488(s4)
	flw	fa0, 1516(sp)
	call	expf
	fsw	fa0, 1484(s4)
	flw	fa0, 1512(sp)
	call	expf
	fsw	fa0, 1480(s4)
	flw	fa0, 1508(sp)
	call	expf
	fsw	fa0, 1476(s4)
	flw	fa0, 1504(sp)
	call	expf
	fsw	fa0, 1472(s4)
	flw	fa0, 1500(sp)
	call	expf
	fsw	fa0, 1468(s4)
	flw	fa0, 1496(sp)
	call	expf
	fsw	fa0, 1464(s4)
	flw	fa0, 1492(sp)
	call	expf
	fsw	fa0, 1460(s4)
	flw	fa0, 1488(sp)
	call	expf
	fsw	fa0, 1456(s4)
	flw	fa0, 1484(sp)
	call	expf
	fsw	fa0, 1452(s4)
	flw	fa0, 1480(sp)
	call	expf
	fsw	fa0, 1448(s4)
	flw	fa0, 1476(sp)
	call	expf
	fsw	fa0, 1444(s4)
	flw	fa0, 1472(sp)
	call	expf
	fsw	fa0, 1440(s4)
	flw	fa0, 1468(sp)
	call	expf
	fsw	fa0, 1436(s4)
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1376(s4)
	flw	fa0, 1464(sp)
	call	expf
	fsw	fa0, 1432(s4)
	flw	fa0, 1460(sp)
	call	expf
	fsw	fa0, 1428(s4)
	flw	fa0, 1456(sp)
	call	expf
	fsw	fa0, 1424(s4)
	flw	fa0, 1452(sp)
	call	expf
	fsw	fa0, 1420(s4)
	flw	fa0, 1448(sp)
	call	expf
	fsw	fa0, 1416(s4)
	flw	fa0, 1444(sp)
	call	expf
	fsw	fa0, 1412(s4)
	flw	fa0, 1440(sp)
	call	expf
	fsw	fa0, 1408(s4)
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1388(s4)
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1384(s4)
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1380(s4)
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1404(s4)
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1400(s4)
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1396(s4)
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1392(s4)
	flw	fa0, 1916(sp)
	call	expf
	fsw	fa0, 1884(s4)
	flw	fa0, 1912(sp)
	call	expf
	fsw	fa0, 1880(s4)
	flw	fa0, 1908(sp)
	call	expf
	fsw	fa0, 1876(s4)
	flw	fa0, 1904(sp)
	call	expf
	fsw	fa0, 1872(s4)
	flw	fa0, 1900(sp)
	call	expf
	fsw	fa0, 1868(s4)
	flw	fa0, 1896(sp)
	call	expf
	fsw	fa0, 1864(s4)
	flw	fa0, 1892(sp)
	call	expf
	fsw	fa0, 1860(s4)
	flw	fa0, 1888(sp)
	call	expf
	fsw	fa0, 1856(s4)
	flw	fa0, 1884(sp)
	call	expf
	fsw	fa0, 1852(s4)
	flw	fa0, 1880(sp)
	call	expf
	fsw	fa0, 1848(s4)
	flw	fa0, 1876(sp)
	call	expf
	fsw	fa0, 1844(s4)
	flw	fa0, 1872(sp)
	call	expf
	fsw	fa0, 1840(s4)
	flw	fa0, 1868(sp)
	call	expf
	fsw	fa0, 1836(s4)
	flw	fa0, 1864(sp)
	call	expf
	fsw	fa0, 1832(s4)
	flw	fa0, 1860(sp)
	call	expf
	fsw	fa0, 1828(s4)
	flw	fa0, 1856(sp)
	call	expf
	fsw	fa0, 1824(s4)
	flw	fa0, 1852(sp)
	call	expf
	fsw	fa0, 1820(s4)
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1760(s4)
	flw	fa0, 1848(sp)
	call	expf
	fsw	fa0, 1816(s4)
	flw	fa0, 1844(sp)
	call	expf
	fsw	fa0, 1812(s4)
	flw	fa0, 1840(sp)
	call	expf
	fsw	fa0, 1808(s4)
	flw	fa0, 1836(sp)
	call	expf
	fsw	fa0, 1804(s4)
	flw	fa0, 1832(sp)
	call	expf
	fsw	fa0, 1800(s4)
	flw	fa0, 1828(sp)
	call	expf
	fsw	fa0, 1796(s4)
	flw	fa0, 1824(sp)
	call	expf
	fsw	fa0, 1792(s4)
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1772(s4)
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1768(s4)
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1764(s4)
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1788(s4)
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1784(s4)
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1780(s4)
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1776(s4)
	flw	fa0, 1788(sp)
	call	expf
	fsw	fa0, 1756(s4)
	flw	fa0, 1784(sp)
	call	expf
	fsw	fa0, 1752(s4)
	flw	fa0, 1780(sp)
	call	expf
	fsw	fa0, 1748(s4)
	flw	fa0, 1776(sp)
	call	expf
	fsw	fa0, 1744(s4)
	flw	fa0, 1772(sp)
	call	expf
	fsw	fa0, 1740(s4)
	flw	fa0, 1768(sp)
	call	expf
	fsw	fa0, 1736(s4)
	flw	fa0, 1764(sp)
	call	expf
	fsw	fa0, 1732(s4)
	flw	fa0, 1760(sp)
	call	expf
	fsw	fa0, 1728(s4)
	flw	fa0, 1756(sp)
	call	expf
	fsw	fa0, 1724(s4)
	flw	fa0, 1752(sp)
	call	expf
	fsw	fa0, 1720(s4)
	flw	fa0, 1748(sp)
	call	expf
	fsw	fa0, 1716(s4)
	flw	fa0, 1744(sp)
	call	expf
	fsw	fa0, 1712(s4)
	flw	fa0, 1740(sp)
	call	expf
	fsw	fa0, 1708(s4)
	flw	fa0, 1736(sp)
	call	expf
	fsw	fa0, 1704(s4)
	flw	fa0, 1732(sp)
	call	expf
	fsw	fa0, 1700(s4)
	flw	fa0, 1728(sp)
	call	expf
	fsw	fa0, 1696(s4)
	flw	fa0, 1724(sp)
	call	expf
	fsw	fa0, 1692(s4)
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1632(s4)
	flw	fa0, 1720(sp)
	call	expf
	fsw	fa0, 1688(s4)
	flw	fa0, 1716(sp)
	call	expf
	fsw	fa0, 1684(s4)
	flw	fa0, 1712(sp)
	call	expf
	fsw	fa0, 1680(s4)
	flw	fa0, 1708(sp)
	call	expf
	fsw	fa0, 1676(s4)
	flw	fa0, 1704(sp)
	call	expf
	fsw	fa0, 1672(s4)
	flw	fa0, 1700(sp)
	call	expf
	fsw	fa0, 1668(s4)
	flw	fa0, 1696(sp)
	call	expf
	fsw	fa0, 1664(s4)
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1644(s4)
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1640(s4)
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1636(s4)
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1660(s4)
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1656(s4)
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1652(s4)
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1648(s4)
	flw	fa0, 92(s4)
	call	expf
	fsw	fa0, 92(s5)
	flw	fa0, 88(s4)
	call	expf
	fsw	fa0, 88(s5)
	flw	fa0, 84(s4)
	call	expf
	fsw	fa0, 84(s5)
	flw	fa0, 80(s4)
	call	expf
	fsw	fa0, 80(s5)
	flw	fa0, 76(s4)
	call	expf
	fsw	fa0, 76(s5)
	flw	fa0, 72(s4)
	call	expf
	fsw	fa0, 72(s5)
	flw	fa0, 68(s4)
	call	expf
	fsw	fa0, 68(s5)
	flw	fa0, 64(s4)
	call	expf
	fsw	fa0, 64(s5)
	flw	fa0, 60(s4)
	call	expf
	fsw	fa0, 60(s5)
	flw	fa0, 56(s4)
	call	expf
	fsw	fa0, 56(s5)
	flw	fa0, 52(s4)
	call	expf
	fsw	fa0, 52(s5)
	flw	fa0, 48(s4)
	call	expf
	fsw	fa0, 48(s5)
	flw	fa0, 44(s4)
	call	expf
	fsw	fa0, 44(s5)
	flw	fa0, 40(s4)
	call	expf
	fsw	fa0, 40(s5)
	flw	fa0, 36(s4)
	call	expf
	fsw	fa0, 36(s5)
	flw	fa0, 32(s4)
	call	expf
	fsw	fa0, 32(s5)
	flw	fa0, 28(s4)
	call	expf
	fsw	fa0, 28(s5)
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 2016(s4)
	flw	fa0, 24(s4)
	call	expf
	fsw	fa0, 24(s5)
	flw	fa0, 20(s4)
	call	expf
	fsw	fa0, 20(s5)
	flw	fa0, 16(s4)
	call	expf
	fsw	fa0, 16(s5)
	flw	fa0, 12(s4)
	call	expf
	fsw	fa0, 12(s5)
	flw	fa0, 8(s4)
	call	expf
	fsw	fa0, 8(s5)
	flw	fa0, 4(s4)
	call	expf
	fsw	fa0, 4(s5)
	flw	fa0, 0(s4)
	call	expf
	fsw	fa0, 0(s5)
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 2028(s4)
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 2024(s4)
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 2020(s4)
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 2044(s4)
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 2040(s4)
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 2036(s4)
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 2032(s4)
	flw	fa0, 2044(sp)
	call	expf
	fsw	fa0, 2012(s4)
	flw	fa0, 2040(sp)
	call	expf
	fsw	fa0, 2008(s4)
	flw	fa0, 2036(sp)
	call	expf
	fsw	fa0, 2004(s4)
	flw	fa0, 2032(sp)
	call	expf
	fsw	fa0, 2000(s4)
	flw	fa0, 2028(sp)
	call	expf
	fsw	fa0, 1996(s4)
	flw	fa0, 2024(sp)
	call	expf
	fsw	fa0, 1992(s4)
	flw	fa0, 2020(sp)
	call	expf
	fsw	fa0, 1988(s4)
	flw	fa0, 2016(sp)
	call	expf
	fsw	fa0, 1984(s4)
	flw	fa0, 2012(sp)
	call	expf
	fsw	fa0, 1980(s4)
	flw	fa0, 2008(sp)
	call	expf
	fsw	fa0, 1976(s4)
	flw	fa0, 2004(sp)
	call	expf
	fsw	fa0, 1972(s4)
	flw	fa0, 2000(sp)
	call	expf
	fsw	fa0, 1968(s4)
	flw	fa0, 1996(sp)
	call	expf
	fsw	fa0, 1964(s4)
	flw	fa0, 1992(sp)
	call	expf
	fsw	fa0, 1960(s4)
	flw	fa0, 1988(sp)
	call	expf
	fsw	fa0, 1956(s4)
	flw	fa0, 1984(sp)
	call	expf
	fsw	fa0, 1952(s4)
	flw	fa0, 1980(sp)
	call	expf
	fsw	fa0, 1948(s4)
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1888(s4)
	flw	fa0, 1976(sp)
	call	expf
	fsw	fa0, 1944(s4)
	flw	fa0, 1972(sp)
	call	expf
	fsw	fa0, 1940(s4)
	flw	fa0, 1968(sp)
	call	expf
	fsw	fa0, 1936(s4)
	flw	fa0, 1964(sp)
	call	expf
	fsw	fa0, 1932(s4)
	flw	fa0, 1960(sp)
	call	expf
	fsw	fa0, 1928(s4)
	flw	fa0, 1956(sp)
	call	expf
	fsw	fa0, 1924(s4)
	flw	fa0, 1952(sp)
	call	expf
	fsw	fa0, 1920(s4)
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1900(s4)
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1896(s4)
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1892(s4)
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1916(s4)
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1912(s4)
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1908(s4)
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 1904(s4)
	flw	fa0, 380(sp)
	call	expf
	fsw	fa0, 348(s4)
	flw	fa0, 376(sp)
	call	expf
	fsw	fa0, 344(s4)
	flw	fa0, 372(sp)
	call	expf
	fsw	fa0, 340(s4)
	flw	fa0, 368(sp)
	call	expf
	fsw	fa0, 336(s4)
	flw	fa0, 364(sp)
	call	expf
	fsw	fa0, 332(s4)
	flw	fa0, 360(sp)
	call	expf
	fsw	fa0, 328(s4)
	flw	fa0, 356(sp)
	call	expf
	fsw	fa0, 324(s4)
	flw	fa0, 352(sp)
	call	expf
	fsw	fa0, 320(s4)
	flw	fa0, 348(sp)
	call	expf
	fsw	fa0, 316(s4)
	flw	fa0, 344(sp)
	call	expf
	fsw	fa0, 312(s4)
	flw	fa0, 340(sp)
	call	expf
	fsw	fa0, 308(s4)
	flw	fa0, 336(sp)
	call	expf
	fsw	fa0, 304(s4)
	flw	fa0, 332(sp)
	call	expf
	fsw	fa0, 300(s4)
	flw	fa0, 328(sp)
	call	expf
	fsw	fa0, 296(s4)
	flw	fa0, 324(sp)
	call	expf
	fsw	fa0, 292(s4)
	flw	fa0, 320(sp)
	call	expf
	fsw	fa0, 288(s4)
	flw	fa0, 316(sp)
	call	expf
	fsw	fa0, 284(s4)
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 224(s4)
	flw	fa0, 312(sp)
	call	expf
	fsw	fa0, 280(s4)
	flw	fa0, 308(sp)
	call	expf
	fsw	fa0, 276(s4)
	flw	fa0, 304(sp)
	call	expf
	fsw	fa0, 272(s4)
	flw	fa0, 300(sp)
	call	expf
	fsw	fa0, 268(s4)
	flw	fa0, 296(sp)
	call	expf
	fsw	fa0, 264(s4)
	flw	fa0, 292(sp)
	call	expf
	fsw	fa0, 260(s4)
	flw	fa0, 288(sp)
	call	expf
	fsw	fa0, 256(s4)
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 236(s4)
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 232(s4)
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 228(s4)
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 252(s4)
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 248(s4)
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 244(s4)
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 240(s4)
	flw	fa0, 252(sp)
	call	expf
	fsw	fa0, 220(s4)
	flw	fa0, 248(sp)
	call	expf
	fsw	fa0, 216(s4)
	flw	fa0, 244(sp)
	call	expf
	fsw	fa0, 212(s4)
	flw	fa0, 240(sp)
	call	expf
	fsw	fa0, 208(s4)
	flw	fa0, 236(sp)
	call	expf
	fsw	fa0, 204(s4)
	flw	fa0, 232(sp)
	call	expf
	fsw	fa0, 200(s4)
	flw	fa0, 228(sp)
	call	expf
	fsw	fa0, 196(s4)
	flw	fa0, 224(sp)
	call	expf
	fsw	fa0, 192(s4)
	flw	fa0, 220(sp)
	call	expf
	fsw	fa0, 188(s4)
	flw	fa0, 216(sp)
	call	expf
	fsw	fa0, 184(s4)
	flw	fa0, 212(sp)
	call	expf
	fsw	fa0, 180(s4)
	flw	fa0, 208(sp)
	call	expf
	fsw	fa0, 176(s4)
	flw	fa0, 204(sp)
	call	expf
	fsw	fa0, 172(s4)
	flw	fa0, 200(sp)
	call	expf
	fsw	fa0, 168(s4)
	flw	fa0, 196(sp)
	call	expf
	fsw	fa0, 164(s4)
	flw	fa0, 192(sp)
	call	expf
	fsw	fa0, 160(s4)
	flw	fa0, 188(sp)
	call	expf
	fsw	fa0, 156(s4)
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 96(s4)
	flw	fa0, 184(sp)
	call	expf
	fsw	fa0, 152(s4)
	flw	fa0, 180(sp)
	call	expf
	fsw	fa0, 148(s4)
	flw	fa0, 176(sp)
	call	expf
	fsw	fa0, 144(s4)
	flw	fa0, 172(sp)
	call	expf
	fsw	fa0, 140(s4)
	flw	fa0, 168(sp)
	call	expf
	fsw	fa0, 136(s4)
	flw	fa0, 164(sp)
	call	expf
	fsw	fa0, 132(s4)
	flw	fa0, 160(sp)
	call	expf
	fsw	fa0, 128(s4)
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 108(s4)
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 104(s4)
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 100(s4)
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 7
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 124(s4)
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 6
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 120(s4)
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 5
	vfmv.f.s	fa0, v8
	call	expf
	fsw	fa0, 116(s4)
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e32, m2, ta, ma
	vslidedown.vi	v8, v8, 4
	vfmv.f.s	fa0, v8
	call	expf
	csrr	a0, vlenb
	li	a1, 263
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	fsw	fa0, 112(s4)
	addi	a0, sp, 2047
	addi	a0, a0, 257
	addi	a1, sp, 2047
	addi	a1, a1, 129
	vsetvli	zero, s2, e32, m8, ta, ma
	vle32.v	v8, (a0)
	lui	a0, 260096
	vle32.v	v16, (a1)
	fmv.w.x	fa5, a0
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v8, v8, fa5
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v16, v16, fa5
	lui	a0, 1
	addi	a0, a0, 160
	add	a0, sp, a0
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	lui	a0, 1
	add	a0, sp, a0
	.loc	1 12 25                         # k135114294394832.py:12:25
	vle32.v	v24, (a0)
	addi	a0, sp, 2047
	addi	a0, a0, 1921
	vle32.v	v8, (a0)
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v24, v24, fa5
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v8, v8, fa5
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114294394832.py:17:18
	vfmul.vv	v16, v16, v8
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v24, v8
	.loc	1 18 39                         # k135114294394832.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v8, v24, 16
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v8, v24, s2
	addi	a0, sp, 2047
	addi	a0, a0, 513
	addi	a1, sp, 2047
	addi	a1, a1, 385
	addi	a2, sp, 2047
	addi	a2, a2, 769
	addi	a3, sp, 2047
	addi	a3, a3, 641
	addi	a4, sp, 2047
	addi	a4, a4, 1025
	addi	a5, sp, 2047
	addi	a5, a5, 897
	addi	a6, sp, 2047
	addi	a6, a6, 1281
	addi	a7, sp, 2047
	addi	a7, a7, 1153
	addi	t0, sp, 2047
	addi	t0, t0, 1537
	addi	t1, sp, 2047
	addi	t1, t1, 1409
	addi	t2, sp, 2047
	addi	t2, t2, 1793
	addi	t3, sp, 2047
	addi	t3, t3, 1665
	.loc	1 12 25                         # k135114294394832.py:12:25
	vsetvli	zero, s2, e32, m8, ta, ma
	vle32.v	v16, (a0)
	csrr	a0, vlenb
	li	t4, 368
	mul	a0, a0, t4
	add	a0, sp, a0
	lui	t4, 1
	addi	t4, t4, 160
	add	a0, a0, t4
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (a1)
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (a2)
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (a3)
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (a4)
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (a5)
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (a6)
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (a7)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (t0)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (t1)
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v16, (t2)
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle32.v	v24, (t3)
	.loc	1 18 39                         # k135114294394832.py:18:39
	vsetvli	zero, s3, e16, m8, ta, ma
	vse16.v	v8, (s8), v0.t
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114294394832.py:17:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfmul.vv	v16, v16, v8
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v0, v0, v8
	.loc	1 18 39                         # k135114294394832.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v8, v0, 16
	vnsrl.wi	v0, v16, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v8, v0, s2
	csrr	a0, vlenb
	li	a1, 258
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vse16.v	v8, (s11), v0.t
	.loc	1 14 18                         # k135114294394832.py:14:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfadd.vf	v8, v24, fa5
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v0, v16, v8
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v8, v8, fa5
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v16, v16, fa5
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v16, v16, fa5
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v16, v16, fa5
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v24, v16, fa5
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114294394832.py:17:18
	vfmul.vv	v24, v0, v16
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v8, v8, v16
	.loc	1 18 39                         # k135114294394832.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	vnsrl.wi	v8, v24, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v16, v8, s2
	csrr	a0, vlenb
	li	a1, 259
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vse16.v	v16, (s10), v0.t
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114294394832.py:17:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v24, v16
	.loc	1 18 39                         # k135114294394832.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v24, v16, s2
	csrr	a0, vlenb
	li	a1, 260
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vse16.v	v24, (s9), v0.t
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114294394832.py:17:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v24, v16
	.loc	1 18 39                         # k135114294394832.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v24, v16, s2
	csrr	a0, vlenb
	li	a1, 261
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vse16.v	v24, (s7), v0.t
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfadd.vf	v8, v8, fa5
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v0, v16, v8
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v8, v8, fa5
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v16, v16, fa5
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v16, v16, fa5
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v16, v16, fa5
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 14 18                         # k135114294394832.py:14:18
	vfadd.vf	v24, v16, fa5
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 15 19                         # k135114294394832.py:15:19
	vfdiv.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114294394832.py:17:18
	vfmul.vv	v24, v0, v16
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v8, v8, v16
	.loc	1 18 39                         # k135114294394832.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	vnsrl.wi	v8, v24, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v16, v8, s2
	csrr	a0, vlenb
	li	a1, 262
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vse16.v	v16, (s6), v0.t
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114294394832.py:17:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v24, v16
	.loc	1 18 39                         # k135114294394832.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v24, v16, s2
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	vse16.v	v24, (a0), v0.t
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 18                         # k135114294394832.py:17:18
	vsetvli	zero, s2, e32, m8, ta, ma
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v24, v16
	.loc	1 18 39                         # k135114294394832.py:18:39
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, s3, e16, m8, ta, ma
	vslideup.vx	v24, v16, s2
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 1
	addi	a1, a1, 160
	add	a0, a0, a1
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	vse16.v	v24, (a0), v0.t
	.loc	1 18 4 epilogue_begin is_stmt 0 # k135114294394832.py:18:4
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
	addi	sp, sp, 2032
	.cfi_def_cfa_offset 0
	ret
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused__unsafe_view_mul_silu_12, .Lfunc_end0-triton_poi_fused__unsafe_view_mul_silu_12
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
	.asciz	"k135114294394832.py"           # string offset=7 ; k135114294394832.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
