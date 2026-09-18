	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14
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
	.globl	triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14
	.p2align	2
	.type	triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14,@function
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14: # @triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114449009392.py"
	.loc	1 2 0                           # k135114449009392.py:2:0
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
	li	t0, 433
	mul	a7, a7, t0
	sub	sp, sp, a7
	.cfi_escape 0x0f, 0x0f, 0x72, 0x00, 0x11, 0xa0, 0x01, 0x22, 0x11, 0xb1, 0x03, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 160 + 433 * vlenb
	mv	t6, a6
	mv	t3, a5
	mv	a7, a0
.Ltmp0:
	.loc	1 17 19 prologue_end            # k135114449009392.py:17:19
	lwu	t0, 0(a1)
	lw	t1, 4(a1)
	li	a6, 64
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	a0, a0, 768
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v8, (a7)
	csrr	a1, vlenb
	li	t2, 361
	mul	a1, a1, t2
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 17 19                         # k135114449009392.py:17:19
	slli	t1, t1, 32
	or	t0, t1, t0
	.loc	1 24 48                         # k135114449009392.py:24:48
	slli	t1, t0, 8
	slli	t0, t0, 11
	sub	s5, t0, t1
	add	s5, s5, a2
	vle16.v	v16, (s5)
	csrr	a1, vlenb
	li	a2, 409
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	li	a2, 32
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 24 102                        # k135114449009392.py:24:102
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	.loc	1 19 41                         # k135114449009392.py:19:41
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 425
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114449009392.py:25:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v24, v0
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	a0, a7, 896
	sd	a0, 16(sp)                      # 8-byte Folded Spill
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	t1, s5, 768
	csrr	a1, vlenb
	li	t0, 425
	mul	a1, a1, t0
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vzext.vf2	v24, v0
	.loc	1 24 48                         # k135114449009392.py:24:48
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (t1)
	csrr	a1, vlenb
	li	t0, 329
	mul	a1, a1, t0
	add	a1, sp, a1
	addi	a1, a1, 48
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e32, m8, ta, ma
	vsll.vi	v24, v24, 16
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	t1, s5, 896
	.loc	1 19 41                         # k135114449009392.py:19:41
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	.loc	1 24 48 is_stmt 0               # k135114449009392.py:24:48
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (t1)
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18 is_stmt 1               # k135114449009392.py:25:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 95                         # k135114449009392.py:19:95
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	a0, a7, 512
	sd	a0, 24(sp)                      # 8-byte Folded Spill
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (a0)
	csrr	a0, vlenb
	li	a1, 305
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	t2, s5, 512
	vle16.v	v8, (t2)
	csrr	a0, vlenb
	li	a1, 313
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	t2, a7, 128
	.loc	1 19 95 is_stmt 0               # k135114449009392.py:19:95
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v24, v24, 16
	.loc	1 24 102 is_stmt 1              # k135114449009392.py:24:102
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 19 41                         # k135114449009392.py:19:41
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v8, (t2)
	csrr	a0, vlenb
	li	a1, 289
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114449009392.py:25:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	t1, a7, 640
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	t4, s5, 128
	.loc	1 19 95                         # k135114449009392.py:19:95
	vzext.vf2	v24, v8
	.loc	1 24 48                         # k135114449009392.py:24:48
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (t4)
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e32, m8, ta, ma
	vsll.vi	v24, v24, 16
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	t4, s5, 640
	.loc	1 19 41                         # k135114449009392.py:19:41
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v8, (t1)
	csrr	a0, vlenb
	li	a1, 241
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	.loc	1 24 48 is_stmt 0               # k135114449009392.py:24:48
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (t4)
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18 is_stmt 1               # k135114449009392.py:25:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 95                         # k135114449009392.py:19:95
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 24 102                        # k135114449009392.py:24:102
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v8, v24, 16
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	t4, a7, 256
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (t4)
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	t5, s5, 256
	vle16.v	v24, (t5)
	csrr	a0, vlenb
	li	a1, 217
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	t5, a7, 384
	.loc	1 19 95 is_stmt 0               # k135114449009392.py:19:95
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 24 102 is_stmt 1              # k135114449009392.py:24:102
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 19 41                         # k135114449009392.py:19:41
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (t5)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114449009392.py:25:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	t0, a7, 1408
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	s0, s5, 384
	.loc	1 19 95                         # k135114449009392.py:19:95
	vzext.vf2	v8, v24
	.loc	1 24 48                         # k135114449009392.py:24:48
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (s0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e32, m8, ta, ma
	vsll.vi	v8, v8, 16
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	s0, s5, 1408
	.loc	1 19 41                         # k135114449009392.py:19:41
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (t0)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 24 48 is_stmt 0               # k135114449009392.py:24:48
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (s0)
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 25 18 is_stmt 1               # k135114449009392.py:25:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 95                         # k135114449009392.py:19:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 24 102                        # k135114449009392.py:24:102
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 297
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	s0, a7, 1280
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (s0)
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	s1, s5, 1280
	vle16.v	v24, (s1)
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	s1, a7, 1664
	.loc	1 19 95 is_stmt 0               # k135114449009392.py:19:95
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 24 102 is_stmt 1              # k135114449009392.py:24:102
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 19 41                         # k135114449009392.py:19:41
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (s1)
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114449009392.py:25:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 265
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	s2, a7, 1152
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	s3, s5, 1664
	.loc	1 19 95                         # k135114449009392.py:19:95
	vzext.vf2	v8, v24
	.loc	1 24 48                         # k135114449009392.py:24:48
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (s3)
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e32, m8, ta, ma
	vsll.vi	v8, v8, 16
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	s3, s5, 1152
	.loc	1 19 41                         # k135114449009392.py:19:41
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (s2)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 24 48 is_stmt 0               # k135114449009392.py:24:48
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (s3)
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 25 18 is_stmt 1               # k135114449009392.py:25:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 95                         # k135114449009392.py:19:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 24 102                        # k135114449009392.py:24:102
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 273
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	s3, a7, 1536
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (s3)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	s4, s5, 1536
	vle16.v	v16, (s4)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 41                         # k135114449009392.py:19:41
	addi	s4, a7, 1024
	.loc	1 19 95 is_stmt 0               # k135114449009392.py:19:95
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v24, v8, 16
	.loc	1 24 102 is_stmt 1              # k135114449009392.py:24:102
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 19 41                         # k135114449009392.py:19:41
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (s4)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114449009392.py:25:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 24 48                         # k135114449009392.py:24:48
	addi	s5, s5, 1024
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v8, (s5)
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 20 37                         # k135114449009392.py:20:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (a3)
	csrr	a0, vlenb
	li	a1, 233
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 91                         # k135114449009392.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 281
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 425
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 305
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 313
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 289
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 313
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 241
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 289
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 217
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 217
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 25 18                         # k135114449009392.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 1024
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 95                         # k135114449009392.py:19:95
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 24 102                        # k135114449009392.py:24:102
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 20 37                         # k135114449009392.py:20:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (s5)
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114449009392.py:25:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 1536
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (s5)
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 1152
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 20 37                         # k135114449009392.py:20:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (s5)
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 305
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 1664
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (s5)
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 273
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 273
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 1280
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 20 37                         # k135114449009392.py:20:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (s5)
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 1408
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (s5)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 265
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 241
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 384
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 20 37                         # k135114449009392.py:20:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (s5)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 297
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 297
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 256
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (s5)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 91                         # k135114449009392.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 640
	csrr	a0, vlenb
	li	a1, 233
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 20 37                         # k135114449009392.py:20:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (s5)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 128
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 20 37                         # k135114449009392.py:20:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (s5)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 512
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (s5)
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	s5, a3, 896
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vzext.vf2	v8, v24
	vsll.vi	v24, v8, 16
	.loc	1 20 37                         # k135114449009392.py:20:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v8, (s5)
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 37                         # k135114449009392.py:20:37
	addi	a3, a3, 768
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (a3)
	.loc	1 20 91 is_stmt 0               # k135114449009392.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18 is_stmt 1               # k135114449009392.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (a4)
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 281
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v16, v16, v0
	csrr	a0, vlenb
	li	a1, 425
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v0, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 265
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 233
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 217
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 289
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 217
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 313
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 768
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 21 37                         # k135114449009392.py:21:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (a3)
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 91                         # k135114449009392.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114449009392.py:26:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 896
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 21 37                         # k135114449009392.py:21:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (a3)
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 512
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (a3)
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 128
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 21 37                         # k135114449009392.py:21:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (a3)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 640
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (a3)
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 91                         # k135114449009392.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 256
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	.loc	1 21 37                         # k135114449009392.py:21:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (a3)
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 384
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (a3)
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 1408
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 21 37                         # k135114449009392.py:21:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (a3)
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 1280
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (a3)
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 297
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 289
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 1664
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 21 37                         # k135114449009392.py:21:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v0, (a3)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 241
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 313
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 1152
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v24, (a3)
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 281
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 1536
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vzext.vf2	v8, v24
	vsll.vi	v16, v8, 16
	.loc	1 21 37                         # k135114449009392.py:21:37
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v8, (a3)
	csrr	a0, vlenb
	li	a1, 273
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v16, v24, v16
	csrr	a0, vlenb
	li	a1, 297
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 37                         # k135114449009392.py:21:37
	addi	a3, a4, 1024
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v16, (a3)
	.loc	1 21 91 is_stmt 0               # k135114449009392.py:21:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 305
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18 is_stmt 1               # k135114449009392.py:27:18
	vfadd.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 273
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91                         # k135114449009392.py:21:91
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 305
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v0, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 217
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 217
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 241
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v8, v24, 16
	csrr	a0, vlenb
	li	a1, 233
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 233
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 91                         # k135114449009392.py:21:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 265
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114449009392.py:27:18
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 265
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 273
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp1:
	.file	2 "/home/chlee/triton-cpu/python/triton/language" "standard.py"
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v8, v16, v16
	csrr	a0, vlenb
	li	a1, 305
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp2:
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v16, v16, v16
	csrr	a0, vlenb
	li	a1, 425
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp3:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v16, v24, v24
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 281
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp4:
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp5:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v8, v16, v16
	csrr	a0, vlenb
	li	a1, 297
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp6:
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v16, v16, v16
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp7:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v16, v24, v24
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 289
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp8:
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v16, v8, v8
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp9:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v16, v8, v8
.Ltmp10:
	.loc	1 31 37                         # k135114449009392.py:31:37
	vmv.v.i	v24, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp11:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmadd.vv	v0, v0, v24
	vfadd.vv	v8, v16, v0
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp12:
	.loc	1 41 78                         # k135114449009392.py:41:78
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v0, v8, 16
	csrr	a0, vlenb
	li	a1, 425
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v0, a2
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 233
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 29 20                         # k135114449009392.py:29:20
	vsetvli	zero, a2, e32, m8, ta, ma
	vfmul.vv	v0, v8, v8
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp13:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v0, v8, v8
	csrr	a0, vlenb
	li	a1, 265
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp14:
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp15:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v8, v16, v16
	vfadd.vv	v8, v8, v0
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp16:
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp17:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v8, v16, v16
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp18:
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v0, v16, v16
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp19:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v0, v16, v16
	vfadd.vv	v8, v0, v8
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 313
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp20:
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v16, v8, v8
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp21:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v16, v8, v8
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmadd.vv	v8, v8, v24
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmadd.vv	v8, v8, v24
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vfmacc.vv	v24, v0, v0
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp22:
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v16, v16, v16
	csrr	a0, vlenb
	li	a1, 217
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp23:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v16, v24, v24
	vfadd.vv	v8, v16, v8
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 241
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp24:
	.loc	1 29 20                         # k135114449009392.py:29:20
	vfmul.vv	v8, v8, v8
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp25:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfmacc.vv	v8, v0, v0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
.Ltmp26:
	.loc	1 41 78                         # k135114449009392.py:41:78
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v8, v24, 16
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 265
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 305
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 313
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 233
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 273
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v0, 16
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 73
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 57
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 297
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 49
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 241
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 289
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	slli	a1, a0, 5
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 281
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 25
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp27:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v0, v24
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v0, v24
	vfadd.vv	v8, v16, v8
	vfadd.vv	v8, v0, v8
.Ltmp28:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114449009392.py:32:26 ]
	vslidedown.vi	v16, v8, 16
	vslideup.vi	v16, v8, 16
.Ltmp29:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfadd.vv	v8, v8, v16
.Lpcrel_hi0:
	auipc	a3, %pcrel_hi(.LCPI0_0)
	addi	a3, a3, %pcrel_lo(.Lpcrel_hi0)
.Ltmp30:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114449009392.py:32:26 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vle8.v	v16, (a3)
.Lpcrel_hi1:
	auipc	a3, %pcrel_hi(.LCPI0_1)
	addi	a3, a3, %pcrel_lo(.Lpcrel_hi1)
	vle8.v	v5, (a3)
	vsext.vf2	v6, v16
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v6
.Ltmp31:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp32:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114449009392.py:32:26 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vsext.vf2	v6, v5
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v6
.Ltmp33:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp34:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114449009392.py:32:26 ]
	vslidedown.vi	v16, v8, 2
.Ltmp35:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114449009392.py:32:26 ] ]
	vfadd.vv	v8, v8, v16
.Ltmp36:
	.loc	1 32 29                         # k135114449009392.py:32:29
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v9, v8, 1
	vsetivli	zero, 1, e32, mf2, ta, ma
	vfadd.vv	v8, v8, v9
	.loc	1 22 38                         # k135114449009392.py:22:38
	addi	a3, a5, 128
	addi	a4, a5, 256
	addi	s5, a5, 384
	addi	s6, a5, 512
	lui	s7, 280064
	fmv.w.x	fa5, s7
	.loc	1 34 21                         # k135114449009392.py:34:21
	vfdiv.vf	v8, v8, fa5
	.loc	1 22 38                         # k135114449009392.py:22:38
	addi	s7, a5, 640
	addi	s8, a5, 768
	addi	s9, a5, 896
	addi	s10, a5, 1024
	addi	s11, a5, 1152
	lui	ra, 219235
	addi	ra, ra, 1981
	fmv.w.x	fa5, ra
	addi	ra, a5, 1280
	.loc	1 36 20                         # k135114449009392.py:36:20
	vfadd.vf	v8, v8, fa5
	.loc	1 37 28                         # k135114449009392.py:37:28
	vfsqrt.v	v8, v8
	sd	a2, 8(sp)                       # 8-byte Folded Spill
	addi	a0, sp, 48
	vs1r.v	v8, (a0)                        # vscale x 8-byte Folded Spill
	.loc	1 22 38                         # k135114449009392.py:22:38
	addi	a0, a5, 1408
	addi	a1, a5, 1536
	addi	a5, a5, 1664
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v8, (t3)
	csrr	t3, vlenb
	li	a2, 201
	mul	t3, t3, a2
	ld	a2, 8(sp)                       # 8-byte Folded Reload
	add	t3, sp, t3
	addi	t3, t3, 48
	vs8r.v	v8, (t3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a3)
	csrr	a3, vlenb
	li	t3, 185
	mul	a3, a3, t3
	add	a3, sp, a3
	addi	a3, a3, 48
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a4)
	csrr	a3, vlenb
	li	a4, 177
	mul	a3, a3, a4
	add	a3, sp, a3
	addi	a3, a3, 48
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s5)
	csrr	a3, vlenb
	li	a4, 153
	mul	a3, a3, a4
	add	a3, sp, a3
	addi	a3, a3, 48
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s6)
	csrr	a3, vlenb
	slli	a4, a3, 7
	add	a3, a4, a3
	add	a3, sp, a3
	addi	a3, a3, 48
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s7)
	csrr	a3, vlenb
	li	a4, 105
	mul	a3, a3, a4
	add	a3, sp, a3
	addi	a3, a3, 48
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s8)
	csrr	a3, vlenb
	li	a4, 89
	mul	a3, a3, a4
	add	a3, sp, a3
	addi	a3, a3, 48
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s9)
	csrr	a3, vlenb
	slli	a4, a3, 6
	add	a3, a4, a3
	add	a3, sp, a3
	addi	a3, a3, 48
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s10)
	csrr	a3, vlenb
	li	a4, 41
	mul	a3, a3, a4
	add	a3, sp, a3
	addi	a3, a3, 48
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s11)
	csrr	a3, vlenb
	slli	a4, a3, 4
	add	a3, a4, a3
	add	a3, sp, a3
	addi	a3, a3, 48
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (ra)
	csrr	a3, vlenb
	slli	a4, a3, 3
	add	a3, a4, a3
	add	a3, sp, a3
	addi	a3, a3, 48
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vle16.v	v16, (a1)
	vle16.v	v8, (a5)
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 41 78                         # k135114449009392.py:41:78
	vse16.v	v0, (a7)
	csrr	a0, vlenb
	li	a1, 25
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s1)
	csrr	a0, vlenb
	slli	a1, a0, 5
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (t0)
	csrr	a0, vlenb
	li	a1, 49
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s2)
	csrr	a0, vlenb
	li	a1, 57
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	vse16.v	v24, (a0)
	csrr	a0, vlenb
	li	a1, 73
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (t1)
	csrr	a0, vlenb
	li	a1, 81
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (t5)
	.loc	1 38 20                         # k135114449009392.py:38:20
	flw	fa5, 48(sp)                     # 8-byte Folded Reload
	lui	a0, 260096
	fmv.w.x	fa4, a0
	fdiv.s	fa5, fa4, fa5
	csrr	a0, vlenb
	li	a1, 97
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 41 78                         # k135114449009392.py:41:78
	vse16.v	v24, (t2)
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s3)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s0)
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (s4)
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	vse16.v	v24, (a0)
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	vse16.v	v24, (a0)
	csrr	a0, vlenb
	li	a1, 169
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v24, (t4)
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a0, vlenb
	li	a1, 281
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v0, v0, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 281
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v8, v24, 16
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 273
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 273
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 233
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 233
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 289
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 289
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 241
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 241
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 3
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 313
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 313
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 4
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 297
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 297
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 41
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 305
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 305
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 265
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 265
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 6
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 89
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 177
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 217
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 185
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v8
	vsll.vi	v8, v0, 16
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v0, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v8, v0, v8
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	csrr	a0, vlenb
	li	a1, 425
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v24, v16, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v24, v24, v0
	csrr	a0, vlenb
	li	a1, 201
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 92                         # k135114449009392.py:22:92
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v0, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 38 20                         # k135114449009392.py:38:20
	vfmul.vf	v0, v0, fa5
	.loc	1 40 20                         # k135114449009392.py:40:20
	vfmul.vv	v16, v0, v16
	.loc	1 42 76                         # k135114449009392.py:42:76
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v0, v16, 16
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v0, a2
	csrr	a0, vlenb
	li	a1, 425
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v8, v24, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v8, v16, a2
	csrr	a0, vlenb
	li	a1, 417
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v8, a2
	csrr	a0, vlenb
	li	a1, 409
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 209
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v8, a2
	csrr	a0, vlenb
	li	a1, 401
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 193
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 385
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 393
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 265
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 305
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 8
	add	a0, a1, a0
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 297
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 249
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 313
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 241
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 289
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 233
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 273
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v8, a2
	csrr	a0, vlenb
	li	a1, 225
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a0, vlenb
	li	a1, 281
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v0, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v8, a2
	addi	a0, t6, 1664
	vse16.v	v16, (a0)
	addi	a0, t6, 1536
	vse16.v	v24, (a0)
	addi	a0, t6, 1408
	csrr	a1, vlenb
	li	a2, 337
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t6, 1280
	csrr	a1, vlenb
	li	a2, 345
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t6, 1152
	csrr	a1, vlenb
	li	a2, 353
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t6, 1024
	csrr	a1, vlenb
	li	a2, 361
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t6, 896
	csrr	a1, vlenb
	li	a2, 369
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t6, 768
	csrr	a1, vlenb
	li	a2, 393
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t6, 640
	csrr	a1, vlenb
	li	a2, 377
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t6, 512
	csrr	a1, vlenb
	li	a2, 385
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t6, 384
	csrr	a1, vlenb
	li	a2, 401
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t6, 256
	csrr	a1, vlenb
	li	a2, 409
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	addi	a0, t6, 128
	csrr	a1, vlenb
	li	a2, 417
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 48
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 425
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 48
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (t6)
	.loc	1 42 4 epilogue_begin is_stmt 0 # k135114449009392.py:42:4
	csrr	a0, vlenb
	li	a1, 433
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
	.size	triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14, .Lfunc_end0-triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14
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
	.byte	32                              # DW_AT_call_line
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
	.asciz	"k135114449009392.py"           # string offset=7 ; k135114449009392.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
.Linfo_string3:
	.asciz	"triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14" # string offset=74 ; triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
