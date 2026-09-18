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
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294095840.py"
	.loc	1 2 0                           # k135114294095840.py:2:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -144
	.cfi_def_cfa_offset 144
	sd	ra, 136(sp)                     # 8-byte Folded Spill
	sd	s0, 128(sp)                     # 8-byte Folded Spill
	sd	s1, 120(sp)                     # 8-byte Folded Spill
	sd	s2, 112(sp)                     # 8-byte Folded Spill
	sd	s3, 104(sp)                     # 8-byte Folded Spill
	sd	s4, 96(sp)                      # 8-byte Folded Spill
	sd	s5, 88(sp)                      # 8-byte Folded Spill
	sd	s6, 80(sp)                      # 8-byte Folded Spill
	sd	s7, 72(sp)                      # 8-byte Folded Spill
	sd	s8, 64(sp)                      # 8-byte Folded Spill
	sd	s9, 56(sp)                      # 8-byte Folded Spill
	sd	s10, 48(sp)                     # 8-byte Folded Spill
	sd	s11, 40(sp)                     # 8-byte Folded Spill
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
	csrr	a3, vlenb
	li	a4, 377
	mul	a3, a3, a4
	sub	sp, sp, a3
	.cfi_escape 0x0f, 0x0f, 0x72, 0x00, 0x11, 0x90, 0x01, 0x22, 0x11, 0xf9, 0x02, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 144 + 377 * vlenb
	sext.w	a4, a5
.Ltmp0:
	.loc	1 18 45 prologue_end            # k135114294095840.py:18:45
	slli	a6, a5, 7
	slli	a5, a5, 10
	li	t0, 64
	.loc	1 10 21                         # k135114294095840.py:10:21
	slti	s3, a4, 14
	.loc	1 18 45                         # k135114294095840.py:18:45
	subw	s4, a5, a6
	.loc	1 18 50 is_stmt 0               # k135114294095840.py:18:50
	vsetvli	zero, t0, e16, m8, ta, ma
	vmv.v.i	v8, 0
	.loc	1 18 34                         # k135114294095840.py:18:34
	slli	s4, s4, 1
	.loc	1 18 50                         # k135114294095840.py:18:50
	vsetvli	zero, zero, e8, m4, ta, ma
	vmv.v.x	v4, s3
	vmv8r.v	v24, v8
	vmv8r.v	v16, v8
	.loc	1 18 34                         # k135114294095840.py:18:34
	add	a3, a0, s4
	.loc	1 18 50                         # k135114294095840.py:18:50
	vmsne.vi	v0, v4, 0
	csrr	a0, vlenb
	li	a4, 320
	mul	a0, a0, a4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs1r.v	v0, (a0)                        # vscale x 8-byte Folded Spill
	addi	a6, a3, 384
	addi	s0, a3, 1408
	addi	a0, a3, 256
	vsetvli	zero, zero, e16, m8, ta, mu
	vle16.v	v24, (a6), v0.t
	csrr	a4, vlenb
	li	a5, 312
	mul	a4, a4, a5
	add	a4, sp, a4
	addi	a4, a4, 32
	vs8r.v	v24, (a4)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (s0), v0.t
	csrr	a4, vlenb
	li	a5, 304
	mul	a4, a4, a5
	add	a4, sp, a4
	addi	a4, a4, 32
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	vle16.v	v16, (a0), v0.t
	csrr	a0, vlenb
	li	a4, 296
	mul	a0, a0, a4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	t5, a3, 1280
	vmv.v.i	v16, 0
	vle16.v	v16, (t5), v0.t
	csrr	a0, vlenb
	li	a4, 288
	mul	a0, a0, a4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	a4, a3, 640
	vmv.v.i	v16, 0
	vle16.v	v16, (a4), v0.t
	csrr	a0, vlenb
	li	a5, 280
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	s1, a3, 1664
	vmv.v.i	v16, 0
	vle16.v	v16, (s1), v0.t
	csrr	a0, vlenb
	li	a5, 248
	mul	a0, a0, a5
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	a5, a3, 128
	vmv.v.i	v16, 0
	vle16.v	v16, (a5), v0.t
	csrr	a0, vlenb
	li	a7, 224
	mul	a0, a0, a7
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	t6, a3, 1152
	vmv.v.i	v16, 0
	vle16.v	v16, (t6), v0.t
	csrr	a0, vlenb
	li	a7, 216
	mul	a0, a0, a7
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	t1, a3, 512
	vmv.v.i	v16, 0
	vle16.v	v16, (t1), v0.t
	csrr	a0, vlenb
	li	a7, 200
	mul	a0, a0, a7
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	s2, a3, 1536
	vmv.v.i	v16, 0
	vle16.v	v16, (s2), v0.t
	csrr	a0, vlenb
	li	a7, 184
	mul	a0, a0, a7
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	t4, a3, 1024
	vmv.v.i	v16, 0
	vle16.v	v16, (t4), v0.t
	csrr	a0, vlenb
	li	a7, 272
	mul	a0, a0, a7
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	vle16.v	v16, (a3), v0.t
	csrr	a0, vlenb
	li	a7, 168
	mul	a0, a0, a7
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	t3, a3, 896
	vmv.v.i	v16, 0
	vle16.v	v16, (t3), v0.t
	li	a7, 32
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	csrr	a0, vlenb
	li	t2, 353
	mul	a0, a0, t2
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a7
	csrr	a0, vlenb
	li	t2, 369
	mul	a0, a0, t2
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 18 50                         # k135114294095840.py:18:50
	addi	t2, a3, 768
	vmv8r.v	v16, v8
	vsetvli	zero, t0, e16, m8, ta, mu
	vle16.v	v16, (t2), v0.t
	.loc	1 19 30 is_stmt 1               # k135114294095840.py:19:30
	add	a1, a1, s4
	.loc	1 19 46 is_stmt 0               # k135114294095840.py:19:46
	addi	s4, a1, 384
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	li	s4, 264
	mul	a0, a0, s4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 1408
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 256
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	li	s4, 240
	mul	a0, a0, s4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 1280
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	li	s4, 232
	mul	a0, a0, s4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 640
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	li	s4, 208
	mul	a0, a0, s4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 1664
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	li	s4, 192
	mul	a0, a0, s4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 128
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	li	s4, 176
	mul	a0, a0, s4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 1152
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	li	s4, 160
	mul	a0, a0, s4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 512
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	li	s4, 152
	mul	a0, a0, s4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 1536
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	li	s4, 144
	mul	a0, a0, s4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 1024
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	csrr	a0, vlenb
	li	s4, 136
	mul	a0, a0, s4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, a1, 896
	vmv8r.v	v24, v8
	vle16.v	v24, (s4), v0.t
	addi	a1, a1, 768
	vle16.v	v8, (a1), v0.t
	.loc	1 18 112 is_stmt 1              # k135114294095840.py:18:112
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a7
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 19 108                        # k135114294095840.py:19:108
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v16, v24
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a7
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v8
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v8, v0
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v8, v0
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsll.vi	v8, v8, 16
	.loc	1 19 108                        # k135114294095840.py:19:108
	vsll.vi	v16, v16, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vsll.vi	v16, v16, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 25 43                         # k135114294095840.py:25:43
	vsetvli	zero, zero, e8, m2, ta, ma
	vmv.v.x	v8, s3
	vmsne.vi	v0, v8, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs1r.v	v0, (a0)                        # vscale x 8-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, zero, e32, m8, ta, mu
	vzext.vf2	v8, v16
	vsll.vi	v16, v8, 16
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v8, v24
	vsll.vi	v24, v8, 16
	.loc	1 25 43                         # k135114294095840.py:25:43
	vmv.v.i	v8, 0
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v24, 0
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	vmv.v.i	v8, 0
	vfmul.vv	v8, v16, v16, v0.t
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	vmv.v.i	v8, 0
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp1:
	.file	2 "/home/chlee/triton-cpu/python/triton/language" "standard.py"
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp2:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v24, v16, v24
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp3:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp4:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp5:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp6:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v24, v16, v24
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp7:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp8:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a7
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vslidedown.vx	v24, v8, a7
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e32, m8, ta, mu
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp9:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp10:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a7
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vslidedown.vx	v0, v8, a7
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e32, m8, ta, mu
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v24, v16, v24
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp11:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp12:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a7
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vslidedown.vx	v0, v8, a7
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e32, m8, ta, mu
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v24, v16, v24
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp13:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp14:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a7
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vslidedown.vx	v0, v8, a7
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e32, m8, ta, mu
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v24, v16, v24
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp15:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp16:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp17:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp18:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v24, v16, v24
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v8, 0
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp19:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v24, v16
	addi	a0, sp, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
.Ltmp20:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a7
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v24, a7
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vslidedown.vx	v0, v8, a7
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e32, m8, ta, mu
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v16, v24
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v24, v16, v24
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp21:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
.Ltmp22:
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v0, v8, a7
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vslidedown.vx	v16, v8, a7
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a7
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 18 112                        # k135114294095840.py:18:112
	vsetvli	zero, a7, e32, m8, ta, mu
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v8, v24, v16
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 18 112                        # k135114294095840.py:18:112
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294095840.py:19:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 21 18                         # k135114294095840.py:21:18
	vfadd.vv	v24, v16, v24
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 23 18                         # k135114294095840.py:23:18
	vmv.v.i	v16, 0
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp23:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	fmv.w.x	fa5, zero
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vfadd.vf	v24, v8, fa5
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v0
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v0, v16
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v0
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v16, v24
	vfadd.vv	v8, v8, v24
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vf	v24, v24, fa5
	addi	a0, sp, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v16, v24
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vf	v24, v24, fa5
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v0, v24
	vfadd.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v24
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vfadd.vf	v24, v24, fa5
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v0, v24
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v0, v24
	vfadd.vv	v16, v16, v24
	vfadd.vv	v8, v8, v16
.Ltmp24:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294095840.py:26:24 ]
	vslidedown.vi	v16, v8, 16
	vslideup.vi	v16, v8, 16
.Ltmp25:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v8, v16
.Lpcrel_hi0:
	auipc	a1, %pcrel_hi(.LCPI0_0)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi0)
.Ltmp26:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294095840.py:26:24 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vle8.v	v16, (a1)
.Lpcrel_hi1:
	auipc	a1, %pcrel_hi(.LCPI0_1)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi1)
	vle8.v	v26, (a1)
	vsext.vf2	v24, v16
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v24
.Ltmp27:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vsetvli	zero, a7, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp28:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294095840.py:26:24 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vsext.vf2	v24, v26
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v24
.Ltmp29:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vsetvli	zero, a7, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp30:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294095840.py:26:24 ]
	vslidedown.vi	v16, v8, 2
.Ltmp31:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294095840.py:26:24 ] ]
	vfadd.vv	v8, v8, v16
.Ltmp32:
	.loc	1 26 27                         # k135114294095840.py:26:27
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v9, v8, 1
	vsetivli	zero, 1, e32, mf2, ta, ma
	vfadd.vv	v8, v8, v9
	lui	a1, 280064
	fmv.w.x	fa5, a1
	.loc	1 28 20                         # k135114294095840.py:28:20
	vfdiv.vf	v8, v8, fa5
	lui	a1, 219235
	addi	a1, a1, 1981
	fmv.w.x	fa5, a1
	.loc	1 30 20                         # k135114294095840.py:30:20
	vfadd.vf	v8, v8, fa5
	.loc	1 31 28                         # k135114294095840.py:31:28
	vfsqrt.v	v8, v8
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmv.f.s	fa5, v8
	lui	a1, 260096
	fmv.w.x	fa4, a1
	fdiv.s	fa5, fa4, fa5
	.loc	1 20 37                         # k135114294095840.py:20:37
	addi	a1, a2, 1664
	vsetvli	zero, t0, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 20 90 is_stmt 0               # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	sd	a3, 8(sp)                       # 8-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19 is_stmt 1               # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v8, v24, 16
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v24, v8
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v8, 16
	vnsrl.wi	v8, v16, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v8, v24, a7
	.loc	1 20 37                         # k135114294095840.py:20:37
	addi	a1, a2, 128
	addi	s3, a2, 256
	addi	s4, a2, 384
	addi	s5, a2, 512
	addi	s6, a2, 640
	addi	s7, a2, 768
	addi	s8, a2, 896
	addi	s9, a2, 1024
	addi	s10, a2, 1152
	addi	s11, a2, 1280
	addi	ra, a2, 1408
	addi	a0, a2, 1536
	vle16.v	v16, (a2)
	csrr	a2, vlenb
	li	a3, 369
	mul	a2, a2, a3
	ld	a3, 8(sp)                       # 8-byte Folded Reload
	add	a2, sp, a2
	addi	a2, a2, 32
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (a1)
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 32
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (s3)
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 32
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (s4)
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 32
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (s5)
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 32
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (s6)
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 32
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (s7)
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 32
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (s8)
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 32
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (s9)
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 32
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (s10)
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 32
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v24, (a0)
	vle16.v	v16, (s11)
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (ra)
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	.loc	1 35 51                         # k135114294095840.py:35:51
	vse16.v	v8, (s1), v0.t
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v0, v0, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v0, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a0, vlenb
	li	a1, 152
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v0, v0, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v24, v0, v24
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v0, v24, 16
	vnsrl.wi	v24, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v24, v0, a7
	csrr	a0, vlenb
	li	a1, 320
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	vse16.v	v24, (s2), v0.t
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 192
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v24, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 232
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (s0), v0.t
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 208
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 240
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (t5), v0.t
	csrr	a0, vlenb
	slli	a0, a0, 7
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 184
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (t6), v0.t
	csrr	a0, vlenb
	li	a1, 136
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 321
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 272
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (t4), v0.t
	csrr	a0, vlenb
	li	a1, 144
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 337
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 329
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (t3), v0.t
	csrr	a0, vlenb
	li	a1, 160
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 345
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 120
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (t2), v0.t
	csrr	a0, vlenb
	li	a1, 176
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 216
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (a4), v0.t
	csrr	a0, vlenb
	slli	a0, a0, 8
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 112
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 200
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (t1), v0.t
	csrr	a0, vlenb
	li	a1, 264
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 312
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (a6), v0.t
	csrr	a0, vlenb
	li	a1, 288
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 280
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	addi	a0, a3, 256
	vse16.v	v16, (a0), v0.t
	csrr	a0, vlenb
	li	a1, 304
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 353
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 224
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (a5), v0.t
	csrr	a0, vlenb
	li	a1, 369
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 361
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v16, v16, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v8, v16, v8
	.loc	1 20 90                         # k135114294095840.py:20:90
	vsetvli	zero, a7, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a7
	vsetvli	zero, a7, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a0, vlenb
	li	a1, 168
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 32
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 32 19                         # k135114294095840.py:32:19
	vfmul.vf	v24, v24, fa5
	.loc	1 34 19                         # k135114294095840.py:34:19
	vfmul.vv	v16, v24, v16
	.loc	1 35 51                         # k135114294095840.py:35:51
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a7
	vse16.v	v16, (a3), v0.t
	.loc	1 35 4 epilogue_begin is_stmt 0 # k135114294095840.py:35:4
	csrr	a0, vlenb
	li	a1, 377
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 144
	ld	ra, 136(sp)                     # 8-byte Folded Reload
	ld	s0, 128(sp)                     # 8-byte Folded Reload
	ld	s1, 120(sp)                     # 8-byte Folded Reload
	ld	s2, 112(sp)                     # 8-byte Folded Reload
	ld	s3, 104(sp)                     # 8-byte Folded Reload
	ld	s4, 96(sp)                      # 8-byte Folded Reload
	ld	s5, 88(sp)                      # 8-byte Folded Reload
	ld	s6, 80(sp)                      # 8-byte Folded Reload
	ld	s7, 72(sp)                      # 8-byte Folded Reload
	ld	s8, 64(sp)                      # 8-byte Folded Reload
	ld	s9, 56(sp)                      # 8-byte Folded Reload
	ld	s10, 48(sp)                     # 8-byte Folded Reload
	ld	s11, 40(sp)                     # 8-byte Folded Reload
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
	addi	sp, sp, 144
	.cfi_def_cfa_offset 0
	ret
.Ltmp33:
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
	.byte	26                              # DW_AT_call_line
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
	.quad	.Ltmp32-.Lfunc_begin0
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
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        # string offset=0 ; triton
.Linfo_string1:
	.asciz	"k135114294095840.py"           # string offset=7 ; k135114294095840.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
.Linfo_string3:
	.asciz	"triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19" # string offset=74 ; triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
