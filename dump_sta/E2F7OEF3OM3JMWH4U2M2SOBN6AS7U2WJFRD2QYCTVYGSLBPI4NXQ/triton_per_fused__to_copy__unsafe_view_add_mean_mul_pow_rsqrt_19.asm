	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19
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
	.globl	triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19
	.p2align	2
	.type	triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19,@function
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19: # @triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114448401024.py"
	.loc	1 2 0                           # k135114448401024.py:2:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -48
	.cfi_def_cfa_offset 48
	sd	s0, 40(sp)                      # 8-byte Folded Spill
	sd	s1, 32(sp)                      # 8-byte Folded Spill
	sd	s2, 24(sp)                      # 8-byte Folded Spill
	sd	s3, 16(sp)                      # 8-byte Folded Spill
	.cfi_offset s0, -8
	.cfi_offset s1, -16
	.cfi_offset s2, -24
	.cfi_offset s3, -32
	csrr	a3, vlenb
	li	a4, 328
	mul	a3, a3, a4
	sub	sp, sp, a3
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0x30, 0x22, 0x11, 0xc8, 0x02, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 48 + 328 * vlenb
	li	a5, 64
.Ltmp0:
	.loc	1 17 41 prologue_end            # k135114448401024.py:17:41
	addi	a3, a0, 896
	addi	a4, a0, 768
	li	a6, 32
	.loc	1 18 37                         # k135114448401024.py:18:37
	addi	a7, a1, 896
	.loc	1 17 41                         # k135114448401024.py:17:41
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v16, (a4)
	sd	a2, 8(sp)                       # 8-byte Folded Spill
	csrr	a2, vlenb
	li	t0, 232
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114448401024.py:18:37
	addi	t0, a1, 768
	vle16.v	v0, (t0)
	csrr	a2, vlenb
	li	t0, 184
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114448401024.py:17:41
	vle16.v	v24, (a3)
	csrr	a2, vlenb
	li	t0, 240
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 17 95 is_stmt 0               # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 18 91 is_stmt 1               # k135114448401024.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114448401024.py:18:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v0, (a7)
	csrr	a2, vlenb
	li	a7, 216
	mul	a2, a2, a7
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 20 18 is_stmt 1               # k135114448401024.py:20:18
	vsetvli	zero, a6, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	a7, 280
	mul	a2, a2, a7
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	a7, 312
	mul	a2, a2, a7
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114448401024.py:17:41
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v16, (a0)
	csrr	a2, vlenb
	li	a7, 200
	mul	a2, a2, a7
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	addi	a7, a0, 512
	.loc	1 18 37                         # k135114448401024.py:18:37
	addi	t0, a1, 512
	vle16.v	v0, (a1)
	csrr	a2, vlenb
	li	t1, 168
	mul	a2, a2, t1
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 17 41 is_stmt 0               # k135114448401024.py:17:41
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v24, (a7)
	csrr	a2, vlenb
	li	t1, 208
	mul	a2, a2, t1
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114448401024.py:18:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114448401024.py:18:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v0, (t0)
	csrr	a2, vlenb
	li	t0, 176
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 20 18 is_stmt 1               # k135114448401024.py:20:18
	vsetvli	zero, a6, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	t0, 288
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	t0, 272
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114448401024.py:17:41
	addi	t1, a0, 640
	addi	t0, a0, 128
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v16, (t0)
	csrr	a2, vlenb
	li	t2, 144
	mul	a2, a2, t2
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114448401024.py:18:37
	addi	t2, a1, 640
	addi	t3, a1, 128
	vle16.v	v0, (t3)
	csrr	a2, vlenb
	li	t3, 120
	mul	a2, a2, t3
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 17 41 is_stmt 0               # k135114448401024.py:17:41
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v24, (t1)
	csrr	a2, vlenb
	li	t3, 152
	mul	a2, a2, t3
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114448401024.py:18:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114448401024.py:18:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v0, (t2)
	csrr	a2, vlenb
	li	t2, 136
	mul	a2, a2, t2
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 20 18 is_stmt 1               # k135114448401024.py:20:18
	vsetvli	zero, a6, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	t2, 320
	mul	a2, a2, t2
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	t2, 304
	mul	a2, a2, t2
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114448401024.py:17:41
	addi	t2, a0, 384
	addi	t3, a0, 256
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v16, (t3)
	csrr	a2, vlenb
	li	t4, 96
	mul	a2, a2, t4
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114448401024.py:18:37
	addi	t4, a1, 384
	addi	t5, a1, 256
	vle16.v	v0, (t5)
	csrr	a2, vlenb
	li	t5, 80
	mul	a2, a2, t5
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 17 41 is_stmt 0               # k135114448401024.py:17:41
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v24, (t2)
	csrr	a2, vlenb
	li	t5, 104
	mul	a2, a2, t5
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114448401024.py:18:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114448401024.py:18:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v0, (t4)
	csrr	a2, vlenb
	li	t4, 88
	mul	a2, a2, t4
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 20 18 is_stmt 1               # k135114448401024.py:20:18
	vsetvli	zero, a6, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	t4, 248
	mul	a2, a2, t4
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	t4, 296
	mul	a2, a2, t4
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114448401024.py:17:41
	addi	t5, a0, 1280
	addi	t4, a0, 1408
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v16, (t4)
	csrr	a2, vlenb
	slli	a2, a2, 6
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114448401024.py:18:37
	addi	t6, a1, 1280
	addi	s0, a1, 1408
	vle16.v	v0, (s0)
	csrr	a2, vlenb
	li	s0, 48
	mul	a2, a2, s0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 17 41 is_stmt 0               # k135114448401024.py:17:41
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v24, (t5)
	csrr	a2, vlenb
	li	s0, 72
	mul	a2, a2, s0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114448401024.py:18:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114448401024.py:18:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v0, (t6)
	csrr	a2, vlenb
	li	t6, 56
	mul	a2, a2, t6
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 20 18 is_stmt 1               # k135114448401024.py:20:18
	vsetvli	zero, a6, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	t6, 264
	mul	a2, a2, t6
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	t6, 192
	mul	a2, a2, t6
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114448401024.py:17:41
	addi	t6, a0, 1152
	addi	s0, a0, 1664
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v16, (s0)
	csrr	a2, vlenb
	slli	a2, a2, 5
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114448401024.py:18:37
	addi	s1, a1, 1152
	addi	s2, a1, 1664
	vle16.v	v0, (s2)
	csrr	a2, vlenb
	slli	a2, a2, 4
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 17 41 is_stmt 0               # k135114448401024.py:17:41
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v24, (t6)
	csrr	a2, vlenb
	li	s2, 40
	mul	a2, a2, s2
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114448401024.py:18:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 18 37 is_stmt 0               # k135114448401024.py:18:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v0, (s1)
	csrr	a2, vlenb
	li	s1, 24
	mul	a2, a2, s1
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v0, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 20 18 is_stmt 1               # k135114448401024.py:20:18
	vsetvli	zero, a6, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	li	s1, 224
	mul	a2, a2, s1
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 95                         # k135114448401024.py:17:95
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v8, v16
	csrr	a2, vlenb
	slli	a2, a2, 8
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 17 41                         # k135114448401024.py:17:41
	addi	s1, a0, 1024
	addi	s2, a0, 1536
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (s2)
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 18 37                         # k135114448401024.py:18:37
	addi	s3, a1, 1024
	addi	a1, a1, 1536
	vle16.v	v0, (a1)
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v16, v24, 16
	.loc	1 17 41 is_stmt 0               # k135114448401024.py:17:41
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (s1)
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	ld	a2, 8(sp)                       # 8-byte Folded Reload
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 18 91 is_stmt 1               # k135114448401024.py:18:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v0
	vsll.vi	v8, v24, 16
	.loc	1 18 37 is_stmt 0               # k135114448401024.py:18:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v24, (s3)
	addi	a1, sp, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 20 18 is_stmt 1               # k135114448401024.py:20:18
	vsetvli	zero, a6, e32, m8, ta, ma
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 112
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	s3, 160
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 232
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 184
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 184
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 240
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 216
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 232
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 200
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 168
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 200
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 208
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 176
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 168
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 144
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 120
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 240
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 152
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 136
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 216
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 96
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 80
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 144
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 104
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 88
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 208
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 48
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 176
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 72
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 56
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 120
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 136
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 40
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	csrr	a1, vlenb
	li	s3, 24
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 152
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v0, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 104
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 112
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	addi	a1, sp, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 91                         # k135114448401024.py:18:91
	vslidedown.vx	v16, v16, a6
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 18 91                         # k135114448401024.py:18:91
	vzext.vf2	v8, v16
	.loc	1 17 95                         # k135114448401024.py:17:95
	vsll.vi	v16, v24, 16
	.loc	1 18 91                         # k135114448401024.py:18:91
	vsll.vi	v8, v8, 16
	.loc	1 20 18                         # k135114448401024.py:20:18
	vfadd.vv	v0, v16, v8
	csrr	a1, vlenb
	li	s3, 96
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v8, v8, v8
	csrr	a1, vlenb
	li	s3, 272
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp1:
	.file	2 "/home/chlee/triton-cpu/python/triton/language" "standard.py"
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v8, v16, v16
	csrr	a1, vlenb
	li	s3, 160
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp2:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v16, v16, v16
	csrr	a1, vlenb
	li	s3, 288
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp3:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v16, v24, v24
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 112
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 224
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp4:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v8, v8, v8
	csrr	a1, vlenb
	li	s3, 304
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp5:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v8, v16, v16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp6:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v16, v16, v16
	csrr	a1, vlenb
	li	s3, 320
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp7:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v16, v24, v24
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	s3, 88
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 264
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp8:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v16, v8, v8
	csrr	a1, vlenb
	li	s3, 296
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp9:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v16, v8, v8
.Ltmp10:
	.loc	1 24 35                         # k135114448401024.py:24:35
	vmv.v.i	v8, 0
	csrr	a1, vlenb
	li	s3, 312
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp11:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmadd.vv	v24, v24, v8
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	s3, 80
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 104
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp12:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v16, v16, v16
	csrr	a1, vlenb
	li	s3, 168
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp13:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v16, v24, v24
.Ltmp14:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v24, v0, v0
	csrr	a1, vlenb
	li	s3, 200
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp15:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v24, v0, v0
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 72
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 136
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp16:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v16, v16, v16
	csrr	a1, vlenb
	li	s3, 216
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp17:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v16, v24, v24
	csrr	a1, vlenb
	li	s3, 152
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp18:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v24, v24, v24
	csrr	a1, vlenb
	li	s3, 240
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp19:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v24, v0, v0
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 192
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp20:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v24, v16, v16
	csrr	a1, vlenb
	li	s3, 248
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp21:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v24, v16, v16
	csrr	a1, vlenb
	li	s3, 232
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfmadd.vv	v16, v16, v8
	csrr	a1, vlenb
	li	s3, 48
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 184
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfmadd.vv	v16, v16, v8
	csrr	a1, vlenb
	li	s3, 280
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfmacc.vv	v8, v0, v0
	vfadd.vv	v8, v24, v8
	csrr	a1, vlenb
	li	s3, 56
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 120
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp22:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v24, v24, v24
	csrr	a1, vlenb
	li	s3, 144
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp23:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v24, v0, v0
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 176
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp24:
	.loc	1 22 18                         # k135114448401024.py:22:18
	vfmul.vv	v24, v24, v24
	csrr	a1, vlenb
	li	s3, 208
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp25:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfmacc.vv	v24, v0, v0
	csrr	a1, vlenb
	li	s3, 48
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v24, v8
	csrr	a1, vlenb
	li	s3, 88
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	s3, 80
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v8, v0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v8, v8, v24
	csrr	a1, vlenb
	li	s3, 88
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	s3, 72
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v8, v16
	csrr	a1, vlenb
	li	s3, 112
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	s3, 56
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v8, v8, v24
	vfadd.vv	v8, v8, v0
	csrr	a1, vlenb
	li	s3, 88
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v24
	vfadd.vv	v8, v8, v16
.Ltmp26:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114448401024.py:25:24 ]
	vslidedown.vi	v16, v8, 16
	vslideup.vi	v16, v8, 16
.Ltmp27:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfadd.vv	v8, v8, v16
.Lpcrel_hi0:
	auipc	a1, %pcrel_hi(.LCPI0_0)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi0)
.Ltmp28:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114448401024.py:25:24 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vle8.v	v16, (a1)
.Lpcrel_hi1:
	auipc	a1, %pcrel_hi(.LCPI0_1)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi1)
	vle8.v	v26, (a1)
	vsext.vf2	v24, v16
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v24
.Ltmp29:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vsetvli	zero, a6, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp30:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114448401024.py:25:24 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vsext.vf2	v24, v26
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v24
.Ltmp31:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vsetvli	zero, a6, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp32:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114448401024.py:25:24 ]
	vslidedown.vi	v16, v8, 2
.Ltmp33:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114448401024.py:25:24 ] ]
	vfadd.vv	v8, v8, v16
.Ltmp34:
	.loc	1 25 27                         # k135114448401024.py:25:27
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v9, v8, 1
	vsetivli	zero, 1, e32, mf2, ta, ma
	vfadd.vv	v8, v8, v9
	lui	a1, 280064
	fmv.w.x	fa5, a1
	.loc	1 27 20                         # k135114448401024.py:27:20
	vfdiv.vf	v8, v8, fa5
	lui	a1, 219235
	addi	a1, a1, 1981
	fmv.w.x	fa5, a1
	.loc	1 29 20                         # k135114448401024.py:29:20
	vfadd.vf	v8, v8, fa5
	.loc	1 30 28                         # k135114448401024.py:30:28
	vfsqrt.v	v8, v8
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmv.f.s	fa5, v8
	lui	a1, 260096
	fmv.w.x	fa4, a1
	fdiv.s	fa5, fa4, fa5
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 1536
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 1280
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 104
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 112
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 192
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 192
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 1024
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 120
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 120
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 160
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 160
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 768
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 96
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 104
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 280
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 280
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 512
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 184
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 184
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 272
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 272
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 256
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 168
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 168
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 248
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 248
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37 is_stmt 0               # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a2)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 144
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 144
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 288
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 288
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 1664
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 200
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 200
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 224
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 224
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 1408
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 136
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 136
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 264
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 264
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 1152
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 176
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 176
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 896
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 152
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 152
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 312
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 96
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 640
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 232
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 232
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	s3, 304
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	csrr	a1, vlenb
	li	s3, 304
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 384
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v24, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	s3, 216
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v0, v16, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v0, v24
	csrr	a1, vlenb
	li	s3, 216
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	csrr	a1, vlenb
	li	s3, 296
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v16, v16, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v16, v0
	csrr	a1, vlenb
	li	s3, 312
	mul	a1, a1, s3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 37                         # k135114448401024.py:19:37
	addi	a1, a2, 128
	.loc	1 19 91 is_stmt 0               # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v16, v8
	.loc	1 19 37                         # k135114448401024.py:19:37
	vsetvli	zero, a5, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e32, m8, ta, ma
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 31 19 is_stmt 1               # k135114448401024.py:31:19
	vfmul.vf	v24, v24, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v16, v24, v16
	.loc	1 19 91                         # k135114448401024.py:19:91
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v0, v0, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v24, v0, v24
	.loc	1 19 91                         # k135114448401024.py:19:91
	vsetvli	zero, a6, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a6
	vsetvli	zero, a6, e32, m8, ta, ma
	vzext.vf2	v0, v8
	vsll.vi	v8, v0, 16
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 31 19                         # k135114448401024.py:31:19
	vfmul.vf	v0, v0, fa5
	.loc	1 33 19                         # k135114448401024.py:33:19
	vfmul.vv	v8, v0, v8
	.loc	1 34 79                         # k135114448401024.py:34:79
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v0, v8, 16
	vnsrl.wi	v8, v24, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v8, v0, a6
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v16, v8, a6
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 216
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v16, v8, a6
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v16, v8, a6
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v24, v8, a6
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v24, v8, a6
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v24, v8, a6
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v24, v8, a6
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v24, v8, a6
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v24, v8, a6
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v24, v8, a6
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v24, v8, a6
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v24, v8, a6
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a6, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v0, 16
	vsetvli	zero, a5, e16, m8, ta, ma
	vslideup.vx	v16, v8, a6
	vse16.v	v16, (s2)
	vse16.v	v24, (t5)
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (s1)
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a4)
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a7)
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (t3)
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 16
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (s0)
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 16
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (t4)
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 16
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (t6)
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 16
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (a3)
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 16
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (t1)
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 16
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (t2)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 16
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vse16.v	v8, (t0)
	.loc	1 34 4 epilogue_begin is_stmt 0 # k135114448401024.py:34:4
	csrr	a0, vlenb
	li	a1, 328
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 48
	ld	s0, 40(sp)                      # 8-byte Folded Reload
	ld	s1, 32(sp)                      # 8-byte Folded Reload
	ld	s2, 24(sp)                      # 8-byte Folded Reload
	ld	s3, 16(sp)                      # 8-byte Folded Reload
	.cfi_restore s0
	.cfi_restore s1
	.cfi_restore s2
	.cfi_restore s3
	addi	sp, sp, 48
	.cfi_def_cfa_offset 0
	ret
.Ltmp35:
.Lfunc_end0:
	.size	triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19, .Lfunc_end0-triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19
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
	.byte	25                              # DW_AT_call_line
	.byte	24                              # DW_AT_call_column
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
	.quad	.Ltmp34-.Lfunc_begin0
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
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        # string offset=0 ; triton
.Linfo_string1:
	.asciz	"k135114448401024.py"           # string offset=7 ; k135114448401024.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
.Linfo_string3:
	.asciz	"triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19" # string offset=74 ; triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
