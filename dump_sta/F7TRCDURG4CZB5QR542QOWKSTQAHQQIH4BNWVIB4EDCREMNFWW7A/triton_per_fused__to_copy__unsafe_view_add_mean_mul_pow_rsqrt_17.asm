	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17
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
	.globl	triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17
	.p2align	2
	.type	triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17,@function
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17: # @triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294084560.py"
	.loc	1 2 0                           # k135114294084560.py:2:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -32
	.cfi_def_cfa_offset 32
	sd	s0, 24(sp)                      # 8-byte Folded Spill
	sd	s1, 16(sp)                      # 8-byte Folded Spill
	.cfi_offset s0, -8
	.cfi_offset s1, -16
	csrr	a6, vlenb
	li	a7, 569
	mul	a6, a6, a7
	sub	sp, sp, a6
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0x20, 0x22, 0x11, 0xb9, 0x04, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 32 + 569 * vlenb
	csrr	a6, vlenb
	li	a7, 569
	mul	a6, a6, a7
	add	a6, sp, a6
	lw	a7, 32(a6)
	li	a6, 64
.Ltmp0:
	.loc	1 18 46 prologue_end            # k135114294084560.py:18:46
	vsetvli	zero, a6, e16, m8, ta, ma
	vmv.v.i	v8, 0
	vmv.v.i	v24, 0
	vmv.v.i	v16, 0
	.loc	1 10 21                         # k135114294084560.py:10:21
	slti	t0, a7, 14
	.loc	1 18 41                         # k135114294084560.py:18:41
	slli	t1, a7, 7
	slli	a7, a7, 10
	subw	a7, a7, t1
	.loc	1 18 46 is_stmt 0               # k135114294084560.py:18:46
	vsetvli	zero, zero, e8, m4, ta, ma
	vmv.v.x	v4, t0
	.loc	1 18 30                         # k135114294084560.py:18:30
	slli	a7, a7, 1
	.loc	1 18 46                         # k135114294084560.py:18:46
	vmsne.vi	v0, v4, 0
	csrr	t1, vlenb
	li	t2, 560
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs1r.v	v0, (t1)                        # vscale x 8-byte Folded Spill
	.loc	1 18 30                         # k135114294084560.py:18:30
	add	a0, a0, a7
	.loc	1 18 46                         # k135114294084560.py:18:46
	addi	t1, a0, 384
	addi	t2, a0, 1408
	addi	t3, a0, 256
	vsetvli	zero, zero, e16, m8, ta, mu
	vle16.v	v24, (t1), v0.t
	csrr	t1, vlenb
	li	t4, 536
	mul	t1, t1, t4
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v24, (t1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (t2), v0.t
	csrr	t1, vlenb
	li	t2, 544
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	vle16.v	v16, (t3), v0.t
	csrr	t1, vlenb
	li	t2, 528
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a0, 1280
	vmv.v.i	v16, 0
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 520
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a0, 640
	vmv.v.i	v16, 0
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 440
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a0, 1664
	vmv.v.i	v16, 0
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 432
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a0, 128
	vmv.v.i	v16, 0
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 392
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a0, 1152
	vmv.v.i	v16, 0
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 376
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a0, 512
	vmv.v.i	v16, 0
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 320
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a0, 1536
	vmv.v.i	v16, 0
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 312
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a0, 1024
	vmv.v.i	v16, 0
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 328
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	vle16.v	v16, (a0), v0.t
	csrr	t1, vlenb
	li	t2, 280
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a0, 896
	vmv.v.i	v16, 0
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 561
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	a0, a0, 768
	vmv.v.i	v16, 0
	vle16.v	v16, (a0), v0.t
	vmv.v.v	v0, v16
	li	a0, 32
	csrr	t1, vlenb
	li	t2, 561
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vl8r.v	v24, (t1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v24
	csrr	t1, vlenb
	li	t2, 480
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	csrr	t1, vlenb
	li	t2, 561
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vl8r.v	v16, (t1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a0
	csrr	t1, vlenb
	slli	t1, t1, 9
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v0
	csrr	t1, vlenb
	li	t2, 504
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v0, a0
	csrr	t1, vlenb
	li	t2, 552
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	.loc	1 19 30 is_stmt 1               # k135114294084560.py:19:30
	add	a1, a1, a7
	.loc	1 19 46 is_stmt 0               # k135114294084560.py:19:46
	addi	t1, a1, 384
	vmv8r.v	v16, v8
	csrr	t2, vlenb
	li	t3, 560
	mul	t2, t2, t3
	add	t2, sp, t2
	addi	t2, t2, 16
	vl1r.v	v0, (t2)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, a6, e16, m8, ta, mu
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 488
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 1408
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 496
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 256
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 456
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 1280
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 448
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 640
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 360
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 1664
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 352
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 128
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 304
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 1152
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 296
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 512
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 264
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 1536
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	slli	t1, t1, 8
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 1024
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 208
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	vmv8r.v	v16, v8
	vle16.v	v16, (a1), v0.t
	csrr	t1, vlenb
	li	t2, 192
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	t1, a1, 896
	vmv8r.v	v16, v8
	vle16.v	v16, (t1), v0.t
	csrr	t1, vlenb
	li	t2, 561
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vs8r.v	v16, (t1)                       # vscale x 64-byte Folded Spill
	addi	a1, a1, 768
	vmv8r.v	v16, v8
	vmv1r.v	v24, v0
	vle16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 472
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 20 30 is_stmt 1               # k135114294084560.py:20:30
	add	a2, a2, a7
	.loc	1 20 46 is_stmt 0               # k135114294084560.py:20:46
	addi	a1, a2, 768
	vmv8r.v	v16, v8
	vle16.v	v16, (a1), v0.t
	vmv.v.v	v0, v16
	csrr	a1, vlenb
	li	t1, 561
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108 is_stmt 1              # k135114294084560.py:19:108
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	csrr	a1, vlenb
	li	t1, 104
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	t1, 561
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v24, v16, a0
	csrr	a1, vlenb
	li	t1, 120
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	t1, 472
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	csrr	a1, vlenb
	li	t1, 112
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v24, v16, a0
	csrr	a1, vlenb
	li	t1, 464
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 20 108                        # k135114294084560.py:20:108
	vslidedown.vx	v24, v0, a0
	csrr	a1, vlenb
	li	t1, 561
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v0
	csrr	a1, vlenb
	li	t1, 472
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 20 46 is_stmt 0               # k135114294084560.py:20:46
	addi	a1, a2, 896
	vmv8r.v	v16, v8
	csrr	t1, vlenb
	li	t2, 560
	mul	t1, t1, t2
	add	t1, sp, t1
	addi	t1, t1, 16
	vl1r.v	v0, (t1)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, a6, e16, m8, ta, mu
	vle16.v	v16, (a1), v0.t
	vmv8r.v	v24, v8
	vle16.v	v24, (a2), v0.t
	csrr	a1, vlenb
	li	t1, 176
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 1024
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 160
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 1536
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 216
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 512
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 200
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 1152
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 248
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 128
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 240
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 1664
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 288
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 640
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 272
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 1280
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 384
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 256
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 368
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 1408
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	t1, 424
	mul	a1, a1, t1
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a2, 384
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 416
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 21 30 is_stmt 1               # k135114294084560.py:21:30
	add	a3, a3, a7
	.loc	1 21 46 is_stmt 0               # k135114294084560.py:21:46
	addi	a1, a3, 384
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 1408
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 256
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 1280
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 336
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 640
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 1664
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 128
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 1152
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 512
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 1536
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 1024
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv8r.v	v24, v8
	vle16.v	v24, (a3), v0.t
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	addi	a1, a3, 896
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	addi	a1, a3, 768
	vle16.v	v8, (a1), v0.t
	.loc	1 20 108 is_stmt 1              # k135114294084560.py:20:108
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v0, v16, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v24
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vsll.vi	v24, v16, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v8, v24
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v8, v8, v0
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 552
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	csrr	a1, vlenb
	li	a2, 464
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v0, v16
	csrr	a1, vlenb
	li	a2, 472
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v8, v24, v0
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vsll.vi	v16, v8, 16
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v16
	vsll.vi	v0, v0, 16
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v0, v8, v0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v8, v8, v24
	csrr	a1, vlenb
	li	a2, 472
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v24, v16
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v8, v24
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	a2, 464
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v16, v0, v16
	csrr	a1, vlenb
	li	a2, 552
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vsll.vi	v16, v8, 16
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v16, v8, v16
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vsll.vi	v16, v8, 16
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v16, v8, v16
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 472
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 472
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 29 44                         # k135114294084560.py:29:44
	vsetvli	zero, zero, e8, m2, ta, ma
	vmv.v.x	v8, t0
	vmsne.vi	v0, v8, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs1r.v	v0, (a1)                        # vscale x 8-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vsetvli	zero, zero, e32, m8, ta, mu
	vzext.vf2	v8, v16
	vsll.vi	v16, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v16, v8, v16
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	vmv.v.i	v8, 0
	vfmul.vv	v8, v16, v16, v0.t
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 472
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 552
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 464
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp1:
	.file	2 "/home/chlee/triton-cpu/python/triton/language" "standard.py"
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp2:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 216
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp3:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp4:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp5:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp6:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp7:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp8:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a0
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vslidedown.vx	v24, v8, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	addi	a1, sp, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 160
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 136
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	addi	a1, sp, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp9:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp10:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a0
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vslidedown.vx	v24, v8, a0
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 216
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 200
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 152
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 144
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 192
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 176
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp11:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp12:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a0
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 216
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vslidedown.vx	v24, v8, a0
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 184
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 168
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 216
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp13:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp14:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a0
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vslidedown.vx	v24, v8, a0
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 224
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp15:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp16:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 368
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 336
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v8, 0
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp17:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp18:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 416
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v8, 0
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp19:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp20:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a0
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v24, a0
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vslidedown.vx	v24, v8, a0
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 368
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 344
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 336
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 240
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 232
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v8, 0
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp21:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp22:
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v8, v16, a0
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vslidedown.vx	v16, v16, a0
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vslidedown.vx	v24, v8, a0
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 416
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a0
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 18 108                        # k135114294084560.py:18:108
	vsetvli	zero, a0, e32, m8, ta, mu
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 18 108                        # k135114294084560.py:18:108
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294084560.py:19:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294084560.py:23:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v24, v24, v0
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 20 108                        # k135114294084560.py:20:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 24 18                         # k135114294084560.py:24:18
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v0, v8
	vsll.vi	v0, v0, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v8, v16, v0
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 108                        # k135114294084560.py:21:108
	vzext.vf2	v16, v0
	vsll.vi	v16, v16, 16
	.loc	1 25 18                         # k135114294084560.py:25:18
	vfadd.vv	v24, v24, v16
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 27 18                         # k135114294084560.py:27:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp23:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	fmv.w.x	fa5, zero
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vf	v0, v8, fa5
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 208
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v24, v16
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 296
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 288
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a2, 272
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v16, v0
	vfadd.vv	v8, v8, v0
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vf	v0, v0, fa5
	csrr	a1, vlenb
	li	a2, 248
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v16, v0
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vf	v0, v0, fa5
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v0, v24, v0
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vf	v24, v24, fa5
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v0, v24
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v0, v24
	vfadd.vv	v16, v16, v24
	vfadd.vv	v8, v8, v16
.Ltmp24:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294084560.py:30:26 ]
	vslidedown.vi	v16, v8, 16
	vslideup.vi	v16, v8, 16
.Ltmp25:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v8, v16
.Lpcrel_hi0:
	auipc	a1, %pcrel_hi(.LCPI0_0)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi0)
.Ltmp26:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294084560.py:30:26 ]
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
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vsetvli	zero, a0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp28:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294084560.py:30:26 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vsext.vf2	v24, v26
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v24
.Ltmp29:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vsetvli	zero, a0, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp30:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294084560.py:30:26 ]
	vslidedown.vi	v16, v8, 2
.Ltmp31:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294084560.py:30:26 ] ]
	vfadd.vv	v8, v8, v16
.Ltmp32:
	.loc	1 30 29                         # k135114294084560.py:30:29
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v9, v8, 1
	vsetivli	zero, 1, e32, mf2, ta, ma
	vfadd.vv	v8, v8, v9
	lui	a1, 280064
	fmv.w.x	fa5, a1
	.loc	1 32 21                         # k135114294084560.py:32:21
	vfdiv.vf	v8, v8, fa5
	lui	a1, 219235
	addi	a1, a1, 1981
	fmv.w.x	fa5, a1
	.loc	1 34 20                         # k135114294084560.py:34:20
	vfadd.vf	v8, v8, fa5
	.loc	1 35 28                         # k135114294084560.py:35:28
	vfsqrt.v	v8, v8
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmv.f.s	fa5, v8
	lui	a1, 260096
	fmv.w.x	fa4, a1
	fdiv.s	fa5, fa4, fa5
	.loc	1 22 38                         # k135114294084560.py:22:38
	addi	a1, a4, 1664
	vsetvli	zero, a6, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 22 91 is_stmt 0               # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19 is_stmt 1               # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v24, v24, v16
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v8, v16, 16
	csrr	a1, vlenb
	li	a2, 352
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v0, v8, 16
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v0, a0
	.loc	1 22 38                         # k135114294084560.py:22:38
	addi	a1, a4, 128
	addi	a2, a4, 256
	addi	a3, a4, 384
	addi	t0, a4, 512
	addi	t1, a4, 640
	addi	t2, a4, 768
	addi	t3, a4, 896
	addi	t4, a4, 1024
	addi	t5, a4, 1152
	addi	t6, a4, 1280
	addi	s0, a4, 1408
	addi	s1, a4, 1536
	.loc	1 39 25                         # k135114294084560.py:39:25
	add	a5, a5, a7
	.loc	1 39 48 is_stmt 0               # k135114294084560.py:39:48
	addi	a7, a5, 1664
	.loc	1 22 38 is_stmt 1               # k135114294084560.py:22:38
	vle16.v	v8, (a4)
	csrr	a4, vlenb
	sd	s2, 8(sp)                       # 8-byte Folded Spill
	li	s2, 561
	mul	a4, a4, s2
	ld	s2, 8(sp)                       # 8-byte Folded Reload
	add	a4, sp, a4
	addi	a4, a4, 16
	vs8r.v	v8, (a4)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a1)
	csrr	a1, vlenb
	li	a4, 536
	mul	a1, a1, a4
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a2)
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (a3)
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (t0)
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (t1)
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (t2)
	csrr	a1, vlenb
	li	a2, 416
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (t3)
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (t4)
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (t5)
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v24, (s1)
	vle16.v	v8, (t6)
	csrr	a1, vlenb
	li	a2, 368
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vle16.v	v8, (s0)
	csrr	a1, vlenb
	li	a2, 560
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	.loc	1 39 48                         # k135114294084560.py:39:48
	vse16.v	v16, (a7), v0.t
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 88
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v0, v0, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v0, v16
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a1, vlenb
	li	a2, 264
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v0, v0, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v24, v0, v24
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v0, v24, 16
	vnsrl.wi	v24, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v24, v0, a0
	addi	a1, a5, 1536
	csrr	a2, vlenb
	li	a3, 560
	mul	a2, a2, a3
	add	a2, sp, a2
	addi	a2, a2, 16
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vse16.v	v24, (a1), v0.t
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a2, 304
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v8, v24, 16
	csrr	a1, vlenb
	li	a2, 424
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v24, v8
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v8, 16
	vnsrl.wi	v8, v16, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v8, v24, a0
	addi	a1, a5, 1408
	vse16.v	v8, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 368
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 360
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 520
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a5, 1280
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 384
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 96
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 312
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a5, 1152
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 400
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 464
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 328
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a5, 1024
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 408
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 480
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 472
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a5, 896
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 416
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 504
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 120
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a5, 768
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 448
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 376
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a5, 640
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 456
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 112
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 320
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a5, 512
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 488
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 432
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 544
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a5, 384
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 496
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 440
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 528
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a5, 256
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 536
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 9
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 392
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	addi	a1, a5, 128
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a2, 561
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 552
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v16, v16, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v8, v16, v8
	.loc	1 22 91                         # k135114294084560.py:22:91
	vsetvli	zero, a0, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a0
	vsetvli	zero, a0, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a2, 280
	mul	a1, a1, a2
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 36 19                         # k135114294084560.py:36:19
	vfmul.vf	v24, v24, fa5
	.loc	1 38 20                         # k135114294084560.py:38:20
	vfmul.vv	v16, v24, v16
	.loc	1 39 48                         # k135114294084560.py:39:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a6, e16, m8, ta, ma
	vslideup.vx	v16, v24, a0
	vse16.v	v16, (a5), v0.t
	.loc	1 39 4 epilogue_begin is_stmt 0 # k135114294084560.py:39:4
	csrr	a0, vlenb
	li	a1, 569
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 32
	ld	s0, 24(sp)                      # 8-byte Folded Reload
	ld	s1, 16(sp)                      # 8-byte Folded Reload
	.cfi_restore s0
	.cfi_restore s1
	addi	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
.Ltmp33:
.Lfunc_end0:
	.size	triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17, .Lfunc_end0-triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17
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
	.byte	30                              # DW_AT_call_line
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
	.asciz	"k135114294084560.py"           # string offset=7 ; k135114294084560.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
.Linfo_string3:
	.asciz	"triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17" # string offset=74 ; triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
