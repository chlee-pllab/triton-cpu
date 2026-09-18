	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18
.LCPI0_0:
	.byte	4                               # 0x4
	.byte	5                               # 0x5
	.byte	6                               # 0x6
	.byte	7                               # 0x7
	.byte	0                               # 0x0
	.byte	1                               # 0x1
	.byte	2                               # 0x2
	.byte	3                               # 0x3
	.byte	8                               # 0x8
	.byte	9                               # 0x9
	.byte	10                              # 0xa
	.byte	11                              # 0xb
	.byte	12                              # 0xc
	.byte	13                              # 0xd
	.byte	14                              # 0xe
	.byte	15                              # 0xf
.LCPI0_1:
	.byte	2                               # 0x2
	.byte	3                               # 0x3
	.byte	0                               # 0x0
	.byte	1                               # 0x1
	.byte	4                               # 0x4
	.byte	5                               # 0x5
	.byte	6                               # 0x6
	.byte	7                               # 0x7
	.byte	8                               # 0x8
	.byte	9                               # 0x9
	.byte	10                              # 0xa
	.byte	11                              # 0xb
	.byte	12                              # 0xc
	.byte	13                              # 0xd
	.byte	14                              # 0xe
	.byte	15                              # 0xf
	.text
	.globl	triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18
	.p2align	2
	.type	triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18,@function
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18: # @triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114449000512.py"
	.loc	1 2 0                           # k135114449000512.py:2:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -160
	.cfi_def_cfa_offset 160
	sd	ra, 152(sp)                     # 8-byte Folded Spill
	sd	s0, 144(sp)                     # 8-byte Folded Spill
	sd	s1, 136(sp)                     # 8-byte Folded Spill
	sd	s2, 128(sp)                     # 8-byte Folded Spill
	sd	s3, 120(sp)                     # 8-byte Folded Spill
	sd	s4, 112(sp)                     # 8-byte Folded Spill
	sd	s5, 104(sp)                     # 8-byte Folded Spill
	sd	s6, 96(sp)                      # 8-byte Folded Spill
	sd	s7, 88(sp)                      # 8-byte Folded Spill
	sd	s8, 80(sp)                      # 8-byte Folded Spill
	sd	s9, 72(sp)                      # 8-byte Folded Spill
	sd	s10, 64(sp)                     # 8-byte Folded Spill
	sd	s11, 56(sp)                     # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	.cfi_offset s9, -88
	.cfi_offset s10, -96
	.cfi_offset s11, -104
	csrr	a7, vlenb
	li	t0, 432
	mul	a7, a7, t0
	sub	sp, sp, a7
	.cfi_escape 0x0f, 0x0f, 0x72, 0x00, 0x11, 0xa0, 0x01, 0x22, 0x11, 0xb0, 0x03, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 160 + 432 * vlenb
	mv	t3, a6
	mv	t2, a5
	mv	t1, a0
	li	a7, 64
.Ltmp0:
	.loc	1 17 41 prologue_end            # k135114449000512.py:17:41
	addi	a5, a0, 1152
	sd	a5, 16(sp)                      # 8-byte Folded Spill
	addi	a0, a0, 1024
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	.loc	1 18 37                         # k135114449000512.py:18:37
	addi	t0, a1, 1024
	.loc	1 17 41                         # k135114449000512.py:17:41
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v16, (a0)
	csrr	a0, vlenb
	li	a6, 368
	mul	a0, a0, a6
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114449000512.py:18:37
	vle16.v	v24, (t0)
	csrr	a0, vlenb
	li	a6, 352
	mul	a0, a0, a6
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114449000512.py:17:41
	addi	a0, t1, 1536
	sd	a0, 24(sp)                      # 8-byte Folded Spill
	li	t0, 32
	.loc	1 17 95 is_stmt 0               # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v16
	.loc	1 18 91 is_stmt 1               # k135114449000512.py:18:91
	vzext.vf2	v16, v24
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a6, vlenb
	li	t4, 408
	mul	a6, a6, t4
	add	a6, sp, a6
	addi	a6, a6, 48
	vs8r.v	v8, (a6)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114449000512.py:17:41
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v16, (a0)
	csrr	a0, vlenb
	li	a6, 328
	mul	a0, a0, a6
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114449000512.py:18:37
	addi	t4, a1, 1152
	addi	t5, a1, 1536
	vle16.v	v0, (t5)
	csrr	a0, vlenb
	li	a6, 288
	mul	a0, a0, a6
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 17 41 is_stmt 0               # k135114449000512.py:17:41
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a5)
	csrr	a0, vlenb
	li	a5, 336
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114449000512.py:18:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114449000512.py:18:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (t4)
	csrr	a0, vlenb
	li	a5, 312
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 23 18 is_stmt 1               # k135114449000512.py:23:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 400
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114449000512.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 392
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114449000512.py:17:41
	addi	t4, t1, 1280
	addi	t5, t1, 1664
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v16, (t5)
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114449000512.py:18:37
	addi	t6, a1, 1280
	addi	s0, a1, 1664
	vle16.v	v0, (s0)
	csrr	a0, vlenb
	li	a5, 232
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 17 41 is_stmt 0               # k135114449000512.py:17:41
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (t4)
	csrr	a0, vlenb
	li	a5, 264
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114449000512.py:18:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114449000512.py:18:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (t6)
	csrr	a0, vlenb
	li	a5, 240
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 23 18 is_stmt 1               # k135114449000512.py:23:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 384
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114449000512.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 376
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114449000512.py:17:41
	addi	t6, t1, 384
	addi	s1, t1, 1408
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v16, (s1)
	csrr	a0, vlenb
	li	a5, 216
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114449000512.py:18:37
	addi	s0, a1, 384
	addi	s2, a1, 1408
	vle16.v	v0, (s2)
	csrr	a0, vlenb
	li	a5, 200
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 17 41 is_stmt 0               # k135114449000512.py:17:41
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (t6)
	csrr	a0, vlenb
	li	a5, 224
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114449000512.py:18:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114449000512.py:18:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (s0)
	csrr	a0, vlenb
	li	a5, 208
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 23 18 is_stmt 1               # k135114449000512.py:23:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 360
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114449000512.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 344
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114449000512.py:17:41
	addi	s2, t1, 640
	addi	s0, t1, 256
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v16, (s0)
	csrr	a0, vlenb
	li	a5, 176
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114449000512.py:18:37
	addi	s3, a1, 640
	addi	s4, a1, 256
	vle16.v	v0, (s4)
	csrr	a0, vlenb
	li	a5, 160
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 17 41 is_stmt 0               # k135114449000512.py:17:41
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (s2)
	csrr	a0, vlenb
	li	a5, 184
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114449000512.py:18:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114449000512.py:18:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (s3)
	csrr	a0, vlenb
	li	a5, 168
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 23 18 is_stmt 1               # k135114449000512.py:23:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 320
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114449000512.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 304
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114449000512.py:17:41
	addi	s4, t1, 128
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v16, (s4)
	csrr	a0, vlenb
	li	a5, 152
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114449000512.py:18:37
	addi	s3, a1, 128
	vle16.v	v24, (s3)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114449000512.py:17:41
	addi	s3, t1, 512
	.loc	1 17 95 is_stmt 0               # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 18 91 is_stmt 1               # k135114449000512.py:18:91
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 17 41                         # k135114449000512.py:17:41
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (s3)
	csrr	a0, vlenb
	li	a5, 136
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114449000512.py:23:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 272
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114449000512.py:18:37
	addi	s5, a1, 512
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (s5)
	csrr	a0, vlenb
	li	a5, 112
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 17 41 is_stmt 0               # k135114449000512.py:17:41
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (t1)
	csrr	a0, vlenb
	li	a5, 144
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114449000512.py:18:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114449000512.py:18:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a5, 120
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 23 18 is_stmt 1               # k135114449000512.py:23:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 280
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114449000512.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a5, 296
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114449000512.py:17:41
	addi	s5, t1, 768
	addi	s6, t1, 896
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v8, (s6)
	csrr	a0, vlenb
	li	a5, 104
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114449000512.py:18:37
	addi	s7, a1, 768
	addi	a1, a1, 896
	vle16.v	v0, (a1)
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v24, v16, 16
	.loc	1 17 41 is_stmt 0               # k135114449000512.py:17:41
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v8, (s5)
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114449000512.py:18:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v8, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114449000512.py:18:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v16, (s7)
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 23 18 is_stmt 1               # k135114449000512.py:23:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v24, v16
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v8, v24
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v24, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v24
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v16, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v0, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 23 18                         # k135114449000512.py:23:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 768
	csrr	a0, vlenb
	li	a5, 424
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	csrr	a0, vlenb
	li	a5, 416
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114449000512.py:18:91
	vslidedown.vx	v16, v16, t0
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114449000512.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114449000512.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114449000512.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 19 37                         # k135114449000512.py:19:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114449000512.py:23:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 896
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114449000512.py:19:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 19 37 is_stmt 0               # k135114449000512.py:19:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a2)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 512
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 128
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 19 37                         # k135114449000512.py:19:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 640
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 256
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 19 37                         # k135114449000512.py:19:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 384
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 1408
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 19 37                         # k135114449000512.py:19:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 1280
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 1664
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 19 37                         # k135114449000512.py:19:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 1152
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 1536
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vzext.vf2	v8, v24
	vsll.vi	v24, v8, 16
	.loc	1 19 37                         # k135114449000512.py:19:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v8, (a1)
	csrr	a0, vlenb
	li	a1, 392
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114449000512.py:19:37
	addi	a1, a2, 1024
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	.loc	1 19 91 is_stmt 0               # k135114449000512.py:19:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 400
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18 is_stmt 1               # k135114449000512.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114449000512.py:19:91
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 408
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v0, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 408
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 400
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 392
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 1024
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 20 37                         # k135114449000512.py:20:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114449000512.py:19:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114449000512.py:24:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 1536
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 20 37                         # k135114449000512.py:20:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v16, (a1)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 1152
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 1664
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 20 37                         # k135114449000512.py:20:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 1280
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 1408
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 20 37                         # k135114449000512.py:20:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 384
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 256
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 20 37                         # k135114449000512.py:20:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 640
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 128
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 20 37                         # k135114449000512.py:20:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 512
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a3)
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 896
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v24, v8, 16
	.loc	1 20 37                         # k135114449000512.py:20:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v8, (a1)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449000512.py:20:37
	addi	a1, a3, 768
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	.loc	1 20 91 is_stmt 0               # k135114449000512.py:20:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18 is_stmt 1               # k135114449000512.py:25:18
	vfadd.vv	v16, v16, v0
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91                         # k135114449000512.py:20:91
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v16, v0
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v0, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 392
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 400
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 768
	.loc	1 20 91                         # k135114449000512.py:20:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	.loc	1 21 37                         # k135114449000512.py:21:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 408
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114449000512.py:25:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 896
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v16, (a1)
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vfadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 392
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 91                         # k135114449000512.py:21:91
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 21 37 is_stmt 0               # k135114449000512.py:21:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a4)
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 512
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 128
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 21 37                         # k135114449000512.py:21:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 400
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 640
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 256
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 21 37                         # k135114449000512.py:21:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 384
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 408
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 1408
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 21 37                         # k135114449000512.py:21:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 1280
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 1664
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 21 37                         # k135114449000512.py:21:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v0, (a1)
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 1152
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v24, (a1)
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 1536
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v16, v8, 16
	.loc	1 21 37                         # k135114449000512.py:21:37
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v8, (a1)
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449000512.py:21:37
	addi	a1, a4, 1024
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v16, (a1)
	.loc	1 21 91 is_stmt 0               # k135114449000512.py:21:91
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449000512.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91                         # k135114449000512.py:21:91
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v0, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v8, v24, 16
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 91                         # k135114449000512.py:21:91
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449000512.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 400
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp1:
	.file	2 "/home/chlee/triton-cpu/python/triton/language" "standard.py"
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v8, v16, v16
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp2:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v16, v16, v16
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp3:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v16, v24, v24
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp4:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp5:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v8, v16, v16
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp6:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v16, v16, v16
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp7:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v16, v24, v24
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp8:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v16, v8, v8
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp9:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v16, v8, v8
.Ltmp10:
	.loc	1 30 37                         # k135114449000512.py:30:37
	vmv.v.i	v24, 0
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp11:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmadd.vv	v0, v0, v24
	vfadd.vv	v8, v16, v0
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp12:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v0, v8, v8
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp13:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v0, v8, v8
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp14:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp15:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v8, v16, v16
	vfadd.vv	v8, v8, v0
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp16:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp17:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v8, v16, v16
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp18:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v0, v16, v16
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp19:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v0, v16, v16
	vfadd.vv	v8, v0, v8
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp20:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v16, v8, v8
	csrr	a0, vlenb
	li	a1, 408
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp21:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v16, v8, v8
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmadd.vv	v8, v8, v24
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmadd.vv	v8, v8, v24
	csrr	a0, vlenb
	li	a1, 392
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vfmacc.vv	v24, v0, v0
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp22:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v16, v16, v16
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp23:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v16, v24, v24
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp24:
	.loc	1 28 19                         # k135114449000512.py:28:19
	vfmul.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp25:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfmacc.vv	v8, v0, v0
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
.Ltmp26:
	.loc	1 40 78                         # k135114449000512.py:40:78
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v8, v24, 16
	csrr	a0, vlenb
	li	a1, 408
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 400
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 392
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v0, 16
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp27:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v0, v24
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v0, v24
	vfadd.vv	v8, v16, v8
	vfadd.vv	v8, v0, v8
.Ltmp28:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114449000512.py:31:26 ]
	vslidedown.vi	v16, v8, 16
	vslideup.vi	v16, v8, 16
.Ltmp29:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfadd.vv	v8, v8, v16
.Lpcrel_hi0:
	auipc	a1, %pcrel_hi(.LCPI0_0)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi0)
.Ltmp30:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114449000512.py:31:26 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vle8.v	v16, (a1)
.Lpcrel_hi1:
	auipc	a1, %pcrel_hi(.LCPI0_1)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi1)
	vle8.v	v5, (a1)
	vsext.vf2	v6, v16
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v6
.Ltmp31:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp32:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114449000512.py:31:26 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vsext.vf2	v6, v5
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v6
.Ltmp33:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vsetvli	zero, t0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp34:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114449000512.py:31:26 ]
	vslidedown.vi	v16, v8, 2
.Ltmp35:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449000512.py:31:26 ] ]
	vfadd.vv	v8, v8, v16
.Ltmp36:
	.loc	1 31 29                         # k135114449000512.py:31:29
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v9, v8, 1
	vsetivli	zero, 1, e32, mf2, ta, ma
	vfadd.vv	v8, v8, v9
	.loc	1 22 38                         # k135114449000512.py:22:38
	addi	a1, t2, 128
	addi	a2, t2, 256
	addi	a3, t2, 384
	addi	a4, t2, 512
	lui	s7, 280064
	fmv.w.x	fa5, s7
	.loc	1 33 21                         # k135114449000512.py:33:21
	vfdiv.vf	v8, v8, fa5
	.loc	1 22 38                         # k135114449000512.py:22:38
	addi	s7, t2, 640
	addi	s8, t2, 768
	addi	s9, t2, 896
	addi	s10, t2, 1024
	addi	s11, t2, 1152
	lui	ra, 219235
	addi	ra, ra, 1981
	fmv.w.x	fa5, ra
	addi	ra, t2, 1280
	.loc	1 35 20                         # k135114449000512.py:35:20
	vfadd.vf	v8, v8, fa5
	.loc	1 36 28                         # k135114449000512.py:36:28
	vfsqrt.v	v7, v8
	.loc	1 22 38                         # k135114449000512.py:22:38
	addi	a0, t2, 1408
	addi	a5, t2, 1536
	addi	a6, t2, 1664
	vsetvli	zero, a7, e16, m8, ta, ma
	vle16.v	v8, (t2)
	csrr	t2, vlenb
	sd	t0, 8(sp)                       # 8-byte Folded Spill
	li	t0, 200
	mul	t2, t2, t0
	ld	t0, 8(sp)                       # 8-byte Folded Reload
	add	t2, sp, t2
	addi	t2, t2, 48
	vs8r.v	v8, (t2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a1)
	csrr	a1, vlenb
	li	t2, 192
	mul	a1, a1, t2
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a2)
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a3)
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a4)
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s7)
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s8)
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s9)
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s10)
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s11)
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (ra)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a0)
	addi	a0, sp, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vle16.v	v16, (a5)
	vle16.v	v8, (a6)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 40 78                         # k135114449000512.py:40:78
	vse16.v	v24, (t1)
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (t5)
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s1)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	vse16.v	v24, (a0)
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s6)
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s2)
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (t6)
	.loc	1 37 19                         # k135114449000512.py:37:19
	vsetvli	zero, a7, e32, m1, ta, ma
	vfmv.f.s	fa5, v7
	lui	a0, 260096
	fmv.w.x	fa4, a0
	fdiv.s	fa5, fa4, fa5
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 40 78                         # k135114449000512.py:40:78
	vsetvli	zero, a7, e16, m8, ta, ma
	vse16.v	v24, (s4)
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	vse16.v	v24, (a0)
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (t4)
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	vse16.v	v24, (a0)
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s5)
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s3)
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s0)
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v0, v0, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v8, v24, 16
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v24, v24, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v24, v24, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	addi	a0, sp, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 392
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 392
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 400
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 400
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 408
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 408
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v16, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v8, v24, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v0, v8
	vsll.vi	v8, v0, 16
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v0, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v8, v0, v8
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v24, v16, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449000512.py:22:92
	vsetvli	zero, t0, e16, m8, ta, ma
	vslidedown.vx	v0, v16, t0
	vsetvli	zero, t0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 37 19                         # k135114449000512.py:37:19
	vfmul.vf	v0, v0, fa5
	.loc	1 39 20                         # k135114449000512.py:39:20
	vfmul.vv	v16, v0, v16
	.loc	1 41 76                         # k135114449000512.py:41:76
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v0, v16, 16
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v16, v0, t0
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v8, v24, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v8, v16, t0
	csrr	a0, vlenb
	li	a1, 416
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 408
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v16, v8, t0
	csrr	a0, vlenb
	li	a1, 408
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 400
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v16, v8, t0
	csrr	a0, vlenb
	li	a1, 400
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 384
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 368
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 392
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 392
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 360
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 344
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 336
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v24, v8, t0
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, t0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v0, 16
	vsetvli	zero, a7, e16, m8, ta, ma
	vslideup.vx	v16, v8, t0
	addi	a0, t3, 1664
	vse16.v	v16, (a0)
	addi	a0, t3, 1536
	vse16.v	v24, (a0)
	addi	a0, t3, 1408
	csrr	a1, vlenb
	li	a2, 336
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t3, 1280
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t3, 1152
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t3, 1024
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t3, 896
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t3, 768
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t3, 640
	csrr	a1, vlenb
	li	a2, 368
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t3, 512
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t3, 384
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t3, 256
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t3, 128
	csrr	a1, vlenb
	li	a2, 416
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (t3)
	.loc	1 41 4 epilogue_begin is_stmt 0 # k135114449000512.py:41:4
	csrr	a0, vlenb
	li	a1, 432
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 160
	ld	ra, 152(sp)                     # 8-byte Folded Reload
	ld	s0, 144(sp)                     # 8-byte Folded Reload
	ld	s1, 136(sp)                     # 8-byte Folded Reload
	ld	s2, 128(sp)                     # 8-byte Folded Reload
	ld	s3, 120(sp)                     # 8-byte Folded Reload
	ld	s4, 112(sp)                     # 8-byte Folded Reload
	ld	s5, 104(sp)                     # 8-byte Folded Reload
	ld	s6, 96(sp)                      # 8-byte Folded Reload
	ld	s7, 88(sp)                      # 8-byte Folded Reload
	ld	s8, 80(sp)                      # 8-byte Folded Reload
	ld	s9, 72(sp)                      # 8-byte Folded Reload
	ld	s10, 64(sp)                     # 8-byte Folded Reload
	ld	s11, 56(sp)                     # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s1
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
	addi	sp, sp, 160
	.cfi_def_cfa_offset 0
	ret
.Ltmp37:
.Lfunc_end0:
	.size	triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18, .Lfunc_end0-triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
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
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	5                               # DW_FORM_data2
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
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
	.byte	1                               # Abbrev [1] 0xb:0x52 DW_TAG_compile_unit
	.word	.Linfo_string0                  # DW_AT_producer
	.half	2                               # DW_AT_language
	.word	.Linfo_string1                  # DW_AT_name
	.word	.Lline_table_start0             # DW_AT_stmt_list
	.word	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.word	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_inline
	.byte	3                               # Abbrev [3] 0x30:0x2c DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.word	42                              # DW_AT_abstract_origin
	.byte	4                               # Abbrev [4] 0x41:0x1a DW_TAG_inlined_subroutine
	.word	42                              # DW_AT_abstract_origin
	.word	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	1                               # DW_AT_call_file
	.byte	31                              # DW_AT_call_line
	.byte	26                              # DW_AT_call_column
	.byte	5                               # Abbrev [5] 0x4d:0xd DW_TAG_inlined_subroutine
	.word	42                              # DW_AT_abstract_origin
	.word	.Ldebug_ranges1                 # DW_AT_ranges
	.byte	2                               # DW_AT_call_file
	.half	293                             # DW_AT_call_line
	.byte	36                              # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        # string offset=0 ; triton
.Linfo_string1:
	.asciz	"k135114449000512.py"           # string offset=7 ; k135114449000512.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
.Linfo_string3:
	.asciz	"triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18" # string offset=74 ; triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
