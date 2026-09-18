	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11
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
	.globl	triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11
	.p2align	2
	.type	triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11,@function
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11: # @triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294088064.py"
	.loc	1 2 0                           # k135114294088064.py:2:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -32
	.cfi_def_cfa_offset 32
	sd	s0, 24(sp)                      # 8-byte Folded Spill
	sd	s1, 16(sp)                      # 8-byte Folded Spill
	.cfi_offset s0, -8
	.cfi_offset s1, -16
	csrr	a5, vlenb
	li	a6, 385
	mul	a5, a5, a6
	sub	sp, sp, a5
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0x20, 0x22, 0x11, 0x81, 0x03, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 32 + 385 * vlenb
	sext.w	a5, a7
	vsetivli	zero, 16, e64, m8, ta, ma
	vid.v	v8
	li	a6, 13
	vadd.vv	v16, v8, v8
.Ltmp0:
	.loc	1 18 35 prologue_end            # k135114294088064.py:18:35
	blt	a6, a5, .LBB0_2
# %bb.1:
	.loc	1 18 30 is_stmt 0               # k135114294088064.py:18:30
	slli	a6, a5, 3
	add	a0, a0, a6
	.loc	1 18 35                         # k135114294088064.py:18:35
	lw	a6, 4(a0)
	lwu	a0, 0(a0)
	slli	a6, a6, 32
	or	a0, a6, a0
	.loc	1 22 41 is_stmt 1               # k135114294088064.py:22:41
	slli	a6, a0, 7
	slli	a0, a0, 10
	sub	a0, a0, a6
	.loc	1 22 37 is_stmt 0               # k135114294088064.py:22:37
	vmv.v.x	v8, a0
	vadd.vv	v8, v8, v8
	.loc	1 22 30                         # k135114294088064.py:22:30
	vor.vv	v16, v8, v16
.LBB0_2:
	.loc	1 0 30                          # k135114294088064.py:0:30
	csrr	a0, vlenb
	li	a6, 369
	mul	a0, a0, a6
	add	a0, sp, a0
	addi	a0, a0, 16
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 10 21 is_stmt 1               # k135114294088064.py:10:21
	slti	a6, a5, 14
	.loc	1 19 41                         # k135114294088064.py:19:41
	slli	a5, a7, 7
	slli	a7, a7, 10
	li	a0, 64
	subw	a5, a7, a5
	.loc	1 19 46 is_stmt 0               # k135114294088064.py:19:46
	vsetvli	zero, a0, e8, m4, ta, ma
	vmv.v.x	v20, a6
	vsetvli	zero, zero, e16, m8, ta, ma
	vmv.v.i	v8, 0
	.loc	1 19 30                         # k135114294088064.py:19:30
	slli	a5, a5, 1
	.loc	1 19 46                         # k135114294088064.py:19:46
	vsetvli	zero, zero, e8, m4, ta, ma
	vmsne.vi	v0, v20, 0
	csrr	a7, vlenb
	li	t0, 328
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs1r.v	v0, (a7)                        # vscale x 8-byte Folded Spill
	vmv8r.v	v24, v8
	.loc	1 19 30                         # k135114294088064.py:19:30
	add	a7, a2, a5
	.loc	1 19 46                         # k135114294088064.py:19:46
	addi	a2, a7, 384
	addi	t0, a7, 1408
	vsetvli	zero, zero, e16, m8, ta, mu
	vle16.v	v24, (a2), v0.t
	csrr	a2, vlenb
	li	t1, 288
	mul	a2, a2, t1
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	vle16.v	v24, (t0), v0.t
	csrr	a2, vlenb
	li	t0, 320
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, a7, 256
	vmv.v.i	v24, 0
	vle16.v	v24, (a2), v0.t
	csrr	a2, vlenb
	li	t0, 312
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, a7, 1280
	vmv.v.i	v24, 0
	vle16.v	v24, (a2), v0.t
	csrr	a2, vlenb
	li	t0, 304
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, a7, 640
	vmv.v.i	v24, 0
	vle16.v	v24, (a2), v0.t
	csrr	a2, vlenb
	li	t0, 272
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, a7, 1664
	vmv.v.i	v24, 0
	vle16.v	v24, (a2), v0.t
	csrr	a2, vlenb
	li	t0, 240
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, a7, 128
	vmv.v.i	v24, 0
	vle16.v	v24, (a2), v0.t
	csrr	a2, vlenb
	li	t0, 232
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, a7, 1152
	vmv.v.i	v24, 0
	vle16.v	v24, (a2), v0.t
	csrr	a2, vlenb
	li	t0, 224
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, a7, 512
	vmv.v.i	v24, 0
	vle16.v	v24, (a2), v0.t
	csrr	a2, vlenb
	li	t0, 216
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, a7, 1536
	vmv.v.i	v24, 0
	vle16.v	v24, (a2), v0.t
	csrr	a2, vlenb
	li	t0, 200
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, a7, 1024
	vmv.v.i	v24, 0
	vle16.v	v24, (a2), v0.t
	csrr	a2, vlenb
	li	t0, 296
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	vle16.v	v24, (a7), v0.t
	csrr	a2, vlenb
	li	t0, 176
	mul	a2, a2, t0
	add	a2, sp, a2
	addi	a2, a2, 16
	vs8r.v	v24, (a2)                       # vscale x 64-byte Folded Spill
	addi	a2, a7, 896
	vmv.v.i	v24, 0
	vle16.v	v24, (a2), v0.t
	li	a2, 32
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v24
	csrr	t0, vlenb
	li	t1, 361
	mul	t0, t0, t1
	add	t0, sp, t0
	addi	t0, t0, 16
	vs8r.v	v16, (t0)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	csrr	t0, vlenb
	li	t1, 377
	mul	t0, t0, t1
	add	t0, sp, t0
	addi	t0, t0, 16
	vs8r.v	v24, (t0)                       # vscale x 64-byte Folded Spill
	csrr	t0, vlenb
	li	t1, 369
	mul	t0, t0, t1
	add	t0, sp, t0
	addi	t0, t0, 16
	vl8r.v	v16, (t0)                       # vscale x 64-byte Folded Reload
	.loc	1 22 48 is_stmt 1               # k135114294088064.py:22:48
	vsetvli	zero, a2, e64, m1, ta, ma
	vmv.x.s	t0, v16
	.loc	1 19 46                         # k135114294088064.py:19:46
	addi	a7, a7, 768
	vmv8r.v	v16, v8
	vsetvli	zero, a0, e16, m8, ta, mu
	vle16.v	v16, (a7), v0.t
	.loc	1 22 48                         # k135114294088064.py:22:48
	add	a1, t0, a1
	addi	a7, a1, 384
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	li	t0, 280
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 1408
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	li	t0, 264
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 256
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	slli	a7, a7, 8
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 1280
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	li	t0, 248
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 640
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	li	t0, 208
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 1664
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	li	t0, 192
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 128
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	li	t0, 184
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 1152
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	li	t0, 168
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 512
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	li	t0, 160
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 1536
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	li	t0, 152
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 1024
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	csrr	a7, vlenb
	li	t0, 144
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	vmv8r.v	v24, v8
	vle16.v	v24, (a1), v0.t
	csrr	a7, vlenb
	li	t0, 136
	mul	a7, a7, t0
	add	a7, sp, a7
	addi	a7, a7, 16
	vs8r.v	v24, (a7)                       # vscale x 64-byte Folded Spill
	addi	a7, a1, 896
	vmv8r.v	v24, v8
	vle16.v	v24, (a7), v0.t
	addi	a1, a1, 768
	vle16.v	v8, (a1), v0.t
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v16
	csrr	a1, vlenb
	li	a7, 353
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v0, (a1)                        # vscale x 64-byte Folded Spill
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 22 110                        # k135114294088064.py:22:110
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v24
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	csrr	a1, vlenb
	li	a7, 337
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v8
	csrr	a1, vlenb
	li	a7, 112
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	li	a7, 120
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a7, 296
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a7, 144
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v8, v0
	csrr	a1, vlenb
	li	a7, 329
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a7, 176
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v8, v0
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a7, 136
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v0, v24
	vsll.vi	v0, v0, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v8, v0
	csrr	a1, vlenb
	li	a7, 369
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a7, 361
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsll.vi	v8, v8, 16
	.loc	1 22 110                        # k135114294088064.py:22:110
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a7, 345
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a7, 353
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a7, 112
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a7, 353
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a7, 377
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a7, 337
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a7, 337
	mul	a1, a1, a7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 27 43                         # k135114294088064.py:27:43
	vsetvli	zero, zero, e8, m2, ta, ma
	vmv.v.x	v8, a6
	vmsne.vi	v0, v8, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs1r.v	v0, (a1)                        # vscale x 8-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, zero, e32, m8, ta, mu
	vzext.vf2	v8, v16
	vsll.vi	v16, v8, 16
	csrr	a1, vlenb
	li	a6, 120
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v8, v24
	vsll.vi	v24, v8, 16
	.loc	1 27 43                         # k135114294088064.py:27:43
	vmv.v.i	v8, 0
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a6, 120
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a6, 353
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a6, 345
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
	csrr	a1, vlenb
	li	a6, 72
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v24, 0
	vmv.v.i	v8, 0
	vfmul.vv	v8, v16, v16, v0.t
	csrr	a1, vlenb
	li	a6, 80
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 337
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	csrr	a1, vlenb
	li	a6, 88
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v16, 0
	vmv.v.i	v8, 0
	csrr	a1, vlenb
	li	a6, 369
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a6, 329
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp1:
	.file	2 "/home/chlee/triton-cpu/python/triton/language" "standard.py"
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 48
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 216
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp2:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 160
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 200
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 152
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	li	a6, 96
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp3:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a6, 40
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 232
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp4:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 184
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a6, 361
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 224
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 168
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 104
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a6, 361
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp5:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 272
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp6:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 208
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 112
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 240
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 192
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	li	a6, 56
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp7:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a6, 24
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 296
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp8:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a2
	csrr	a1, vlenb
	li	a6, 176
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	li	a6, 176
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 144
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vslidedown.vx	v24, v8, a2
	csrr	a1, vlenb
	li	a6, 136
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	li	a6, 144
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e32, m8, ta, mu
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 296
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 176
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 144
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 176
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v8, v8, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a6, 296
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp9:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 144
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 216
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp10:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a2
	csrr	a1, vlenb
	li	a6, 200
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	li	a6, 200
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 160
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vslidedown.vx	v0, v8, a2
	csrr	a1, vlenb
	li	a6, 152
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	li	a6, 160
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e32, m8, ta, mu
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 216
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 200
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 160
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	li	a6, 160
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp11:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a6, 152
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 232
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp12:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a2
	csrr	a1, vlenb
	li	a6, 224
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	li	a6, 224
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 184
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vslidedown.vx	v0, v8, a2
	csrr	a1, vlenb
	li	a6, 168
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	li	a6, 200
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e32, m8, ta, mu
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 232
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 224
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 200
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	li	a6, 200
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp13:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a6, 136
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 272
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp14:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v8, a2
	csrr	a1, vlenb
	li	a6, 240
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	li	a6, 272
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 208
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vslidedown.vx	v0, v8, a2
	csrr	a1, vlenb
	li	a6, 192
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	li	a6, 240
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e32, m8, ta, mu
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 224
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 272
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 240
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	li	a6, 184
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp15:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 312
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp16:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 240
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 304
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 248
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	li	a6, 208
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp17:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 288
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
.Ltmp18:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 280
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 272
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 320
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 264
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v8
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	li	a6, 192
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	vmv.v.i	v8, 0
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	csrr	a1, vlenb
	li	a6, 272
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp19:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v24, v16
	addi	a1, sp, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 312
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp20:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	csrr	a1, vlenb
	li	a6, 304
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v24, a2
	csrr	a1, vlenb
	li	a6, 304
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vslidedown.vx	v0, v8, a2
	csrr	a1, vlenb
	li	a6, 248
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e32, m8, ta, mu
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v16, v24
	csrr	a1, vlenb
	li	a6, 312
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 304
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp21:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a6, 304
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 320
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
.Ltmp22:
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v16, a2
	csrr	a1, vlenb
	li	a6, 248
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 288
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v0, v8, a2
	csrr	a1, vlenb
	li	a6, 280
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vslidedown.vx	v16, v8, a2
	csrr	a1, vlenb
	li	a6, 264
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vx	v8, v8, a2
	csrr	a1, vlenb
	li	a6, 288
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 19 108                        # k135114294088064.py:19:108
	vsetvli	zero, a2, e32, m8, ta, mu
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v0, v16
	vsll.vi	v16, v0, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a6, 320
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 248
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 19 108                        # k135114294088064.py:19:108
	vzext.vf2	v16, v24
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 288
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 22 110                        # k135114294088064.py:22:110
	vzext.vf2	v24, v0
	vsll.vi	v24, v24, 16
	.loc	1 23 18                         # k135114294088064.py:23:18
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	li	a6, 168
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 25 18                         # k135114294088064.py:25:18
	vmv.v.i	v16, 0
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	vfmul.vv	v16, v24, v24, v0.t
	vmv.v.i	v24, 0
	vfmul.vv	v24, v8, v8, v0.t
.Ltmp23:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v24, v16
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	fmv.w.x	fa5, zero
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vf	v24, v8, fa5
	csrr	a1, vlenb
	li	a6, 48
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a6, 40
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a6, 24
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a6, 264
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 144
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a6, 152
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v0, v16
	csrr	a1, vlenb
	li	a6, 280
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	li	a6, 136
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v0
	csrr	a1, vlenb
	li	a6, 288
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v16, v24
	vfadd.vv	v8, v8, v24
	csrr	a1, vlenb
	li	a6, 72
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vf	v24, v24, fa5
	addi	a1, sp, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v16, v24
	csrr	a1, vlenb
	li	a6, 264
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a6, 80
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vf	v24, v24, fa5
	csrr	a1, vlenb
	li	a6, 304
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v0, v24
	vfadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a6, 280
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vv	v16, v16, v24
	csrr	a1, vlenb
	li	a6, 88
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vfadd.vf	v24, v24, fa5
	csrr	a1, vlenb
	li	a6, 377
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v0, v24
	csrr	a1, vlenb
	li	a6, 288
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	vfadd.vv	v24, v0, v24
	vfadd.vv	v16, v16, v24
	vfadd.vv	v8, v8, v16
.Ltmp24:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294088064.py:28:25 ]
	vslidedown.vi	v16, v8, 16
	vslideup.vi	v16, v8, 16
.Ltmp25:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v8, v16
.Lpcrel_hi0:
	auipc	a1, %pcrel_hi(.LCPI0_0)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi0)
.Ltmp26:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294088064.py:28:25 ]
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
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp28:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294088064.py:28:25 ]
	vsetivli	zero, 16, e16, m2, ta, ma
	vsext.vf2	v24, v26
	vsetvli	zero, zero, e64, m8, ta, ma
	vrgatherei16.vv	v16, v8, v24
.Ltmp29:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vsetvli	zero, a2, e32, m8, ta, ma
	vfadd.vv	v8, v8, v16
.Ltmp30:
	.loc	2 293 36                        # standard.py:293:36 @[ k135114294088064.py:28:25 ]
	vslidedown.vi	v16, v8, 2
.Ltmp31:
	.loc	2 263 15                        # standard.py:263:15 @[ standard.py:293:36 @[ k135114294088064.py:28:25 ] ]
	vfadd.vv	v8, v8, v16
.Ltmp32:
	.loc	1 28 28                         # k135114294088064.py:28:28
	vsetivli	zero, 1, e32, m1, ta, ma
	vslidedown.vi	v9, v8, 1
	vsetivli	zero, 1, e32, mf2, ta, ma
	vfadd.vv	v8, v8, v9
	lui	a1, 280064
	fmv.w.x	fa5, a1
	.loc	1 30 21                         # k135114294088064.py:30:21
	vfdiv.vf	v8, v8, fa5
	lui	a1, 219235
	addi	a1, a1, 1981
	fmv.w.x	fa5, a1
	.loc	1 32 20                         # k135114294088064.py:32:20
	vfadd.vf	v8, v8, fa5
	.loc	1 33 28                         # k135114294088064.py:33:28
	vfsqrt.v	v8, v8
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmv.f.s	fa5, v8
	lui	a1, 260096
	fmv.w.x	fa4, a1
	fdiv.s	fa5, fa4, fa5
	.loc	1 20 38                         # k135114294088064.py:20:38
	addi	a1, a3, 1664
	vsetvli	zero, a0, e16, m8, ta, ma
	vle16.v	v8, (a1)
	.loc	1 20 91 is_stmt 0               # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v16, v8
	vsll.vi	v16, v16, 16
	csrr	a1, vlenb
	li	a6, 56
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19 is_stmt 1               # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v8, v8, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v8
	vsll.vi	v8, v24, 16
	csrr	a1, vlenb
	li	a6, 184
	mul	a1, a1, a6
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v24, v8
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v8, 16
	vnsrl.wi	v8, v16, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v8, v24, a2
	.loc	1 20 38                         # k135114294088064.py:20:38
	addi	a1, a3, 128
	addi	a6, a3, 256
	addi	a7, a3, 384
	addi	t0, a3, 512
	addi	t1, a3, 640
	addi	t2, a3, 768
	addi	t3, a3, 896
	addi	t4, a3, 1024
	addi	t5, a3, 1152
	addi	t6, a3, 1280
	addi	s0, a3, 1408
	addi	s1, a3, 1536
	.loc	1 37 25                         # k135114294088064.py:37:25
	add	a4, a4, a5
	.loc	1 37 48 is_stmt 0               # k135114294088064.py:37:48
	addi	a5, a4, 1664
	.loc	1 20 38 is_stmt 1               # k135114294088064.py:20:38
	vle16.v	v16, (a3)
	csrr	a3, vlenb
	sd	s2, 8(sp)                       # 8-byte Folded Spill
	li	s2, 377
	mul	a3, a3, s2
	ld	s2, 8(sp)                       # 8-byte Folded Reload
	add	a3, sp, a3
	addi	a3, a3, 16
	vs8r.v	v16, (a3)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (a1)
	csrr	a1, vlenb
	li	a3, 304
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (a6)
	csrr	a1, vlenb
	li	a3, 288
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (a7)
	csrr	a1, vlenb
	li	a3, 280
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (t0)
	csrr	a1, vlenb
	li	a3, 264
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (t1)
	csrr	a1, vlenb
	li	a3, 248
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (t2)
	csrr	a1, vlenb
	li	a3, 184
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (t3)
	csrr	a1, vlenb
	li	a3, 152
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (t4)
	csrr	a1, vlenb
	li	a3, 144
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (t5)
	csrr	a1, vlenb
	li	a3, 136
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v24, (s1)
	vle16.v	v16, (t6)
	csrr	a1, vlenb
	li	a3, 88
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	vle16.v	v16, (s0)
	csrr	a1, vlenb
	li	a3, 328
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl1r.v	v0, (a1)                        # vscale x 8-byte Folded Reload
	.loc	1 37 48                         # k135114294088064.py:37:48
	vse16.v	v8, (a5), v0.t
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 96
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v0, v0, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v0, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v24, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v0, v24
	vsll.vi	v24, v0, 16
	csrr	a1, vlenb
	li	a3, 160
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v0, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v0, v0, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v24, v0, v24
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v0, v24, 16
	vnsrl.wi	v24, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v24, v0, a2
	addi	a1, a4, 1536
	csrr	a3, vlenb
	li	a5, 328
	mul	a3, a3, a5
	add	a3, sp, a3
	addi	a3, a3, 16
	vl1r.v	v0, (a3)                        # vscale x 8-byte Folded Reload
	vse16.v	v24, (a1), v0.t
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v16
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 192
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v24, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v16, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 168
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 1408
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 88
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 208
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	slli	a1, a1, 8
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 1280
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 136
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 104
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 200
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 1152
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 144
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 329
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 296
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 1024
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 152
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 345
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 337
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 896
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 184
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 353
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 120
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 768
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 248
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 112
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 224
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 640
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 264
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 7
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 216
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 512
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 280
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 272
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 320
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 384
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 288
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 240
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 312
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 256
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 304
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 361
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 232
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	addi	a1, a4, 128
	vse16.v	v16, (a1), v0.t
	csrr	a1, vlenb
	li	a3, 377
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v8, v24
	vsll.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a3, 369
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v16, v16, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v8, v16, v8
	.loc	1 20 91                         # k135114294088064.py:20:91
	vsetvli	zero, a2, e16, m8, ta, ma
	vslidedown.vx	v16, v24, a2
	vsetvli	zero, a2, e32, m8, ta, ma
	vzext.vf2	v24, v16
	vsll.vi	v16, v24, 16
	csrr	a1, vlenb
	li	a3, 176
	mul	a1, a1, a3
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 34 19                         # k135114294088064.py:34:19
	vfmul.vf	v24, v24, fa5
	.loc	1 36 20                         # k135114294088064.py:36:20
	vfmul.vv	v16, v24, v16
	.loc	1 37 48                         # k135114294088064.py:37:48
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v24, v16, 16
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, a0, e16, m8, ta, ma
	vslideup.vx	v16, v24, a2
	vse16.v	v16, (a4), v0.t
	.loc	1 37 4 epilogue_begin is_stmt 0 # k135114294088064.py:37:4
	csrr	a0, vlenb
	li	a1, 385
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
	.size	triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11, .Lfunc_end0-triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11
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
	.byte	28                              # DW_AT_call_line
	.byte	25                              # DW_AT_call_column
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
	.asciz	"k135114294088064.py"           # string offset=7 ; k135114294088064.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
.Linfo_string3:
	.asciz	"triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11" # string offset=74 ; triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
