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
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114299998752.py"
	.loc	1 2 0                           # k135114299998752.py:2:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -176
	.cfi_def_cfa_offset 176
	sd	ra, 168(sp)                     # 8-byte Folded Spill
	sd	s0, 160(sp)                     # 8-byte Folded Spill
	sd	s1, 152(sp)                     # 8-byte Folded Spill
	sd	s2, 144(sp)                     # 8-byte Folded Spill
	sd	s3, 136(sp)                     # 8-byte Folded Spill
	sd	s4, 128(sp)                     # 8-byte Folded Spill
	sd	s5, 120(sp)                     # 8-byte Folded Spill
	sd	s6, 112(sp)                     # 8-byte Folded Spill
	sd	s7, 104(sp)                     # 8-byte Folded Spill
	sd	s8, 96(sp)                      # 8-byte Folded Spill
	sd	s9, 88(sp)                      # 8-byte Folded Spill
	sd	s10, 80(sp)                     # 8-byte Folded Spill
	sd	s11, 72(sp)                     # 8-byte Folded Spill
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
	li	t0, 561
	mul	a7, a7, t0
	sub	sp, sp, a7
	.cfi_escape 0x0f, 0x0f, 0x72, 0x00, 0x11, 0xb0, 0x01, 0x22, 0x11, 0xb1, 0x04, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 176 + 561 * vlenb
	sd	a6, 48(sp)                      # 8-byte Folded Spill
	mv	t2, a5
	csrr	a5, vlenb
	li	a6, 561
	mul	a5, a5, a6
	add	a5, sp, a5
	lw	a7, 184(a5)
	vsetivli	zero, 16, e64, m8, ta, ma
	vid.v	v8
	li	t0, 13
	vadd.vv	v16, v8, v8
.Ltmp0:
	.loc	1 18 35 prologue_end            # k135114299998752.py:18:35
	blt	t0, a7, .LBB0_2
# %bb.1:
	.loc	1 18 30 is_stmt 0               # k135114299998752.py:18:30
	slli	t0, a7, 3
	add	a1, a1, t0
	.loc	1 18 35                         # k135114299998752.py:18:35
	lw	t0, 4(a1)
	lwu	a1, 0(a1)
	slli	t0, t0, 32
	or	a1, t0, a1
	.loc	1 24 41 is_stmt 1               # k135114299998752.py:24:41
	slli	t0, a1, 7
	slli	a1, a1, 10
	sub	a1, a1, t0
	.loc	1 24 37 is_stmt 0               # k135114299998752.py:24:37
	vmv.v.x	v8, a1
	vadd.vv	v8, v8, v8
	.loc	1 24 30                         # k135114299998752.py:24:30
	vor.vv	v16, v8, v16
.LBB0_2:
	.loc	1 0 30                          # k135114299998752.py:0:30
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 10 21 is_stmt 1               # k135114299998752.py:10:21
	slti	s7, a7, 14
	.loc	1 19 45                         # k135114299998752.py:19:45
	slli	t0, a7, 7
	slli	a7, a7, 10
	li	t1, 64
	subw	a7, a7, t0
	.loc	1 19 50 is_stmt 0               # k135114299998752.py:19:50
	vsetvli	zero, t1, e8, m4, ta, ma
	vmv.v.x	v16, s7
	vsetvli	zero, zero, e16, m8, ta, ma
	vmv.v.i	v8, 0
	.loc	1 19 34                         # k135114299998752.py:19:34
	slli	t0, a7, 1
	.loc	1 19 50                         # k135114299998752.py:19:50
	vsetvli	zero, zero, e8, m4, ta, ma
	vmsne.vi	v0, v16, 0
	csrr	a1, vlenb
	li	a5, 552
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs1r.v	v0, (a1)                        # vscale x 8-byte Folded Spill
	vmv8r.v	v16, v8
	.loc	1 19 34                         # k135114299998752.py:19:34
	add	s2, a0, t0
	.loc	1 19 50                         # k135114299998752.py:19:50
	addi	a0, s2, 384
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	addi	t6, s2, 1408
	vsetvli	zero, zero, e16, m8, ta, mu
	vle16.v	v16, (a0), v0.t
	csrr	a0, vlenb
	li	a1, 424
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	vle16.v	v16, (t6), v0.t
	sd	t6, 8(sp)                       # 8-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 456
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	a0, s2, 256
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	vmv.v.i	v16, 0
	vle16.v	v16, (a0), v0.t
	csrr	a0, vlenb
	li	a1, 440
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	s0, s2, 1280
	vmv.v.i	v16, 0
	vle16.v	v16, (s0), v0.t
	csrr	a0, vlenb
	li	a1, 432
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	t3, s2, 640
	vmv.v.i	v16, 0
	vle16.v	v16, (t3), v0.t
	csrr	a0, vlenb
	li	a1, 488
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	s5, s2, 1664
	vmv.v.i	v16, 0
	vle16.v	v16, (s5), v0.t
	csrr	a0, vlenb
	li	a1, 480
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	a0, s2, 128
	sd	a0, 24(sp)                      # 8-byte Folded Spill
	vmv.v.i	v16, 0
	vle16.v	v16, (a0), v0.t
	csrr	a0, vlenb
	li	a1, 376
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	s1, s2, 1152
	vmv.v.i	v16, 0
	vle16.v	v16, (s1), v0.t
	csrr	a0, vlenb
	li	a1, 296
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	t4, s2, 512
	vmv.v.i	v16, 0
	vle16.v	v16, (t4), v0.t
	csrr	a0, vlenb
	li	a1, 352
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	s6, s2, 1536
	vmv.v.i	v16, 0
	vle16.v	v16, (s6), v0.t
	csrr	a0, vlenb
	li	a1, 248
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	s3, s2, 1024
	vmv.v.i	v16, 0
	vle16.v	v16, (s3), v0.t
	csrr	a0, vlenb
	li	a1, 496
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	vle16.v	v16, (s2), v0.t
	csrr	a0, vlenb
	li	a1, 448
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	s4, s2, 896
	vmv.v.i	v16, 0
	vle16.v	v16, (s4), v0.t
	csrr	a0, vlenb
	li	a1, 553
	mul	a0, a0, a1
	add	a0, sp, a0
	addi	a0, a0, 64
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	addi	t5, s2, 768
	vmv.v.i	v16, 0
	vle16.v	v16, (t5), v0.t
	vmv.v.v	v0, v16
	li	a0, 32
	csrr	a1, vlenb
	li	a5, 553
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v24
	csrr	a1, vlenb
	li	a5, 520
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a5, 553
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a0
	csrr	a1, vlenb
	li	a5, 544
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	csrr	a1, vlenb
	li	a5, 528
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v0, a0
	csrr	a1, vlenb
	li	a5, 536
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 20 30 is_stmt 1               # k135114299998752.py:20:30
	add	a3, a3, t0
	.loc	1 20 46 is_stmt 0               # k135114299998752.py:20:46
	addi	s8, a3, 768
	vmv8r.v	v16, v8
	csrr	a1, vlenb
	li	a5, 552
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, t1, e16, m8, ta, mu
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 553
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 896
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 464
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a5, 184
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 1024
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 168
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 1536
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 240
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 512
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 200
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 1152
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 232
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 128
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 224
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 1664
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 272
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 640
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 1280
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 344
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 256
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 336
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	s8, a3, 1408
	vmv8r.v	v16, v8
	vle16.v	v16, (s8), v0.t
	csrr	a1, vlenb
	li	a5, 400
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a3, 384
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 392
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 21 30 is_stmt 1               # k135114299998752.py:21:30
	add	a4, a4, t0
	.loc	1 21 46 is_stmt 0               # k135114299998752.py:21:46
	addi	a3, a4, 384
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 384
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 1408
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 368
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 256
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 328
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 1280
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 320
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 640
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 216
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 1664
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 208
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 128
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 192
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 1152
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 176
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 512
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 144
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 1536
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 136
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 1024
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 112
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vmv8r.v	v16, v8
	vle16.v	v16, (a4), v0.t
	csrr	a1, vlenb
	li	a3, 104
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a4, 896
	vmv8r.v	v16, v8
	vle16.v	v16, (a3), v0.t
	vmv.v.v	v0, v16
	csrr	a1, vlenb
	li	a3, 553
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 108 is_stmt 1              # k135114299998752.py:20:108
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a0
	csrr	a1, vlenb
	li	a3, 504
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a3, 553
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	csrr	a1, vlenb
	li	a3, 472
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a3, 464
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v24, v16, a0
	csrr	a1, vlenb
	li	a3, 416
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	csrr	a1, vlenb
	li	a3, 408
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v24, v0
	csrr	a1, vlenb
	li	a3, 464
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v0, a0
	csrr	a1, vlenb
	li	a3, 553
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 24 48                         # k135114299998752.py:24:48
	vsetvli	zero, a0, e64, m1, ta, ma
	vmv.x.s	a3, v16
	.loc	1 21 46                         # k135114299998752.py:21:46
	addi	a4, a4, 768
	vmv8r.v	v16, v8
	csrr	a1, vlenb
	li	a5, 552
	mul	a1, a1, a5
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, t1, e16, m8, ta, mu
	vle16.v	v16, (a4), v0.t
	.loc	1 24 48                         # k135114299998752.py:24:48
	add	a2, a3, a2
	addi	a3, a2, 384
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 312
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 1408
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 304
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 256
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 280
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 1280
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 264
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 640
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 160
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 1664
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 152
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 128
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 1152
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 120
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 512
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 96
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 1536
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 88
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 1024
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	li	a3, 80
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv8r.v	v24, v8
	vle16.v	v24, (a2), v0.t
	csrr	a1, vlenb
	li	a3, 72
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a3, a2, 896
	vmv8r.v	v24, v8
	vmv1r.v	v7, v0
	vle16.v	v24, (a3), v0.t
	addi	a2, a2, 768
	vle16.v	v8, (a2), v0.t
	.loc	1 21 108                        # k135114299998752.py:21:108
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a0
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 24 110                        # k135114299998752.py:24:110
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v24
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v8, v16
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v8, v8, v24
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v8, v8, v0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vsll.vi	v0, v24, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v0, v16, v0
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 416
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v0, v16
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 472
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vsll.vi	v0, v16, 16
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v0, v16, v0
	csrr	a1, vlenb
	li	a2, 464
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vsll.vi	v24, v16, 16
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v24, v16
	vsll.vi	v24, v24, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v8, v24
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vsll.vi	v8, v8, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v0, v8
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 31 45                         # k135114299998752.py:31:45
	vsetvli	zero, zero, e8, m2, ta, ma
	vmv.v.x	v8, s7
	vmsne.vi	v0, v8, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs1r.v	v0, (a1)                        # vscale x 8-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vsetvli	zero, zero, e32, m8, ta, mu
	vzext.vf2	v8, v16
	vsll.vi	v16, v8, 16
	.loc	1 31 45                         # k135114299998752.py:31:45
	vmv.v.i	v8, 0
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	a2, 472
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	vmv.v.i	v8, 0
	vfmul.vv	v8, v16, v16, v0.t
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	vmv.v.i	v8, 0
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp1:
	.file	2 "/home/chlee/triton-cpu/python/triton/language" "standard.py"
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp2:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 464
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp3:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp4:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp5:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp6:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 216
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 416
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp7:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp8:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v0, v8, a0
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	addi	a1, sp, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vslidedown.vx	v16, v8, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v24, v16
	addi	a1, sp, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp9:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp10:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v0, v8, a0
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vslidedown.vx	v16, v8, a0
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp11:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp12:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v0, v8, a0
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vslidedown.vx	v16, v8, a0
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp13:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp14:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v0, v8, a0
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 216
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vslidedown.vx	v16, v8, a0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp15:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 216
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp16:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 336
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp17:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp18:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 368
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v8, 0
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp19:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp20:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v0, v16, a0
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v16, a0
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 336
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 336
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vslidedown.vx	v16, v8, a0
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 336
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp21:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp22:
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v8, v16, a0
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v24, v8, a0
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 368
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vslidedown.vx	v16, v8, a0
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 112                        # k135114299998752.py:19:112
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 112                        # k135114299998752.py:19:112
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 110                        # k135114299998752.py:24:110
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114299998752.py:25:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114299998752.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 26 18                         # k135114299998752.py:26:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114299998752.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 27 18                         # k135114299998752.py:27:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 29 19                         # k135114299998752.py:29:19
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp23:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	fmv.w.x	fa5, zero
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vf	v8, v8, fa5
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp24:
	.loc	1 41 50                         # k135114299998752.py:41:50
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	.loc	1 22 38                         # k135114299998752.py:22:38
	addi	a2, t2, 128
	addi	a3, t2, 256
	addi	a4, t2, 384
	addi	s7, t2, 512
	addi	s8, t2, 640
	addi	s9, t2, 768
	addi	s10, t2, 896
	addi	s11, t2, 1024
	addi	ra, t2, 1152
	addi	a1, t2, 1280
	addi	a5, t2, 1408
	addi	a6, t2, 1536
	addi	a7, t2, 1664
	vle16.v	v8, (t2)
	csrr	t2, vlenb
	li	t6, 553
	mul	t2, t2, t6
	ld	t6, 8(sp)                       # 8-byte Folded Reload
	add	t2, sp, t2
	addi	t2, t2, 64
	vs8r.v	v8, (t2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a2)
	csrr	a2, vlenb
	li	t2, 400
	mul	a2, a2, t2
	add	a2, sp, a2
	addi	a2, a2, 64
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a3)
	csrr	a2, vlenb
	li	a3, 392
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 64
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a4)
	csrr	a2, vlenb
	li	a3, 384
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 64
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s7)
	csrr	a2, vlenb
	li	a3, 368
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 64
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s8)
	csrr	a2, vlenb
	li	a3, 344
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 64
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s9)
	csrr	a2, vlenb
	li	a3, 336
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 64
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s10)
	csrr	a2, vlenb
	li	a3, 328
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 64
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s11)
	csrr	a2, vlenb
	li	a3, 320
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 64
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (ra)
	csrr	a2, vlenb
	li	a3, 312
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 64
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a1)
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a5)
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a6)
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a7)
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 552
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	.loc	1 41 50                         # k135114299998752.py:41:50
	vse16.v	v16, (s2), v0.t
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	vse16.v	v16, (s5), v0.t
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	vse16.v	v16, (s6), v0.t
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	vse16.v	v16, (t6), v0.t
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	vse16.v	v16, (s0), v0.t
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	vse16.v	v16, (s1), v0.t
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	vse16.v	v16, (s3), v0.t
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	vse16.v	v16, (s4), v0.t
	csrr	a1, vlenb
	li	a2, 472
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v24, v8, a0
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp25:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vsetvli	zero, a0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v0, v24
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 216
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v24, v0
	vfadd.vv	v8, v8, v0
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vf	v0, v0, fa5
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v24, v0
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vf	v0, v0, fa5
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v24, v0
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vf	v0, v0, fa5
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v24, v0
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v24, v0
	vfadd.vv	v16, v16, v0
	vfadd.vv	v8, v8, v16
.Ltmp26:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114299998752.py:32:26 ]
	vslidedown.vi	v16, v8, 16
	vslideup.vi	v16, v8, 16
.Ltmp27:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v8, v16
.Lpcrel_hi0:
	auipc	a1, %pcrel_hi(.LCPI0_0)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi0)
.Ltmp28:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114299998752.py:32:26 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vle8.v	v16, (a1)
.Lpcrel_hi1:
	auipc	a1, %pcrel_hi(.LCPI0_1)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi1)
	vle8.v	v5, (a1)
	vsext.vf2	v6, v16
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v6
.Ltmp29:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vsetvli	zero, a0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp30:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114299998752.py:32:26 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vsext.vf2	v6, v5
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v6
.Ltmp31:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vsetvli	zero, a0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp32:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114299998752.py:32:26 ]
	vslidedown.vi	v16, v8, 2
.Ltmp33:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114299998752.py:32:26 ] ]
	vfadd.vv	v8, v8, v16
.Ltmp34:
	.loc	1 32 29                         # k135114299998752.py:32:29
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v9, v8, 1
	vsetivli	zero, 1, e32, mf2, ta, ma
	vfadd.vv	v8, v8, v9
	lui	a1, 280064
	fmv.w.x	fa5, a1
	.loc	1 34 21                         # k135114299998752.py:34:21
	vfdiv.vf	v7, v8, fa5
	csrr	a1, vlenb
	li	a2, 552
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 41 50                         # k135114299998752.py:41:50
	vsetvli	zero, t1, e16, m8, ta, ma
	vse16.v	v8, (t5), v0.t
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 416
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	vmv1r.v	v25, v0
	vse16.v	v16, (t3), v0.t
	lui	a1, 219235
	addi	a1, a1, 1981
	fmv.w.x	fa5, a1
	.loc	1 36 20                         # k135114299998752.py:36:20
	vsetivli	zero, 1, e32, mf2, ta, ma
	vfadd.vf	v8, v7, fa5
	.loc	1 37 28                         # k135114299998752.py:37:28
	vfsqrt.v	v24, v8
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 41 50                         # k135114299998752.py:41:50
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 464
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v0, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	vmv1r.v	v0, v25
	vse16.v	v16, (t4), v0.t
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v0, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	vmv1r.v	v0, v25
	ld	a1, 32(sp)                      # 8-byte Folded Reload
	vse16.v	v16, (a1), v0.t
	.loc	1 38 19                         # k135114299998752.py:38:19
	vsetvli	zero, t1, e32, m1, ta, ma
	vfmv.f.s	fa5, v24
	lui	a1, 260096
	fmv.w.x	fa4, a1
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 41 50                         # k135114299998752.py:41:50
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	.loc	1 38 19                         # k135114299998752.py:38:19
	fdiv.s	fa5, fa4, fa5
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 41 50                         # k135114299998752.py:41:50
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	ld	a1, 40(sp)                      # 8-byte Folded Reload
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m4, ta, ma
	vnsrl.wi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v8, a0
	ld	a1, 24(sp)                      # 8-byte Folded Reload
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	ld	a6, 48(sp)                      # 8-byte Folded Reload
	.loc	1 42 25 is_stmt 0               # k135114299998752.py:42:25
	add	a6, a6, t0
	.loc	1 42 48                         # k135114299998752.py:42:48
	addi	a1, a6, 1664
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91 is_stmt 1               # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 1536
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 1408
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 1280
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 1152
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 1024
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 896
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 336
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 472
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 768
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 416
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 640
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 368
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 464
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 512
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 384
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 256
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a6, 128
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 553
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v16, v16, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114299998752.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 64
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 38 19                         # k135114299998752.py:38:19
	vfmul.vf	v24, v24, fa5
	.loc	1 40 20                         # k135114299998752.py:40:20
	vfmul.vv	v16, v24, v16
	.loc	1 42 48                         # k135114299998752.py:42:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, t1, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	vse16.v	v16, (a6), v0.t
	.loc	1 42 4 epilogue_begin is_stmt 0 # k135114299998752.py:42:4
	csrr	a0, vlenb
	li	a1, 561
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 176
	ld	ra, 168(sp)                     # 8-byte Folded Reload
	ld	s0, 160(sp)                     # 8-byte Folded Reload
	ld	s1, 152(sp)                     # 8-byte Folded Reload
	ld	s2, 144(sp)                     # 8-byte Folded Reload
	ld	s3, 136(sp)                     # 8-byte Folded Reload
	ld	s4, 128(sp)                     # 8-byte Folded Reload
	ld	s5, 120(sp)                     # 8-byte Folded Reload
	ld	s6, 112(sp)                     # 8-byte Folded Reload
	ld	s7, 104(sp)                     # 8-byte Folded Reload
	ld	s8, 96(sp)                      # 8-byte Folded Reload
	ld	s9, 88(sp)                      # 8-byte Folded Reload
	ld	s10, 80(sp)                     # 8-byte Folded Reload
	ld	s11, 72(sp)                     # 8-byte Folded Reload
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
	addi	sp, sp, 176
	.cfi_def_cfa_offset 0
	ret
.Ltmp35:
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
	.asciz	"k135114299998752.py"           # string offset=7 ; k135114299998752.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
.Linfo_string3:
	.asciz	"triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14" # string offset=74 ; triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_14
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
