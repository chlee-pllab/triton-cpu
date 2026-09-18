	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused_add_arange_index_copy_transpose_view_zeros_7 # -- Begin function triton_poi_fused_add_arange_index_copy_transpose_view_zeros_7
	.p2align	2
	.type	triton_poi_fused_add_arange_index_copy_transpose_view_zeros_7,@function
triton_poi_fused_add_arange_index_copy_transpose_view_zeros_7: # @triton_poi_fused_add_arange_index_copy_transpose_view_zeros_7
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294095936.py"
	.loc	1 2 0                           # k135114294095936.py:2:0
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
	.cfi_remember_state
	lui	a3, 6
	addi	a3, a3, 144
	sub	sp, sp, a3
	csrr	a3, vlenb
	li	a5, 169
	mul	a3, a3, a5
	sub	sp, sp, a3
	andi	sp, sp, -128
	mv	s2, a2
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114294095936.py:4:33
	slliw	a2, a4, 8
	li	s3, 32
	.loc	1 5 23                          # k135114294095936.py:5:23
	vsetvli	zero, s3, e32, m8, ta, ma
	vmv.v.x	v8, a2
	vid.v	v16
	vor.vx	v0, v16, a2
	.loc	1 7 19                          # k135114294095936.py:7:19
	vsra.vi	v8, v8, 31
	csrr	a3, vlenb
	li	a4, 161
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 7
	addi	a4, a4, -2016
	add	a3, a3, a4
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	.loc	1 10 21                         # k135114294095936.py:10:21
	vsrl.vi	v8, v8, 26
	csrr	a3, vlenb
	li	a4, 113
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 7
	addi	a4, a4, -2016
	add	a3, a3, a4
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vadd.vv	v16, v0, v8
	csrr	a3, vlenb
	li	a4, 153
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 7
	addi	a4, a4, -2016
	add	a3, a3, a4
	vs8r.v	v16, (a3)                       # vscale x 64-byte Folded Spill
	vsra.vi	v8, v16, 6
	.loc	1 10 27 is_stmt 0               # k135114294095936.py:10:27
	vsrl.vi	v24, v16, 31
	vadd.vv	v24, v8, v24
	vand.vi	v24, v24, -2
	vsub.vv	v8, v8, v24
	csrr	a3, vlenb
	li	a4, 145
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 7
	addi	a4, a4, -2016
	add	a3, a3, a4
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	.loc	1 11 19 is_stmt 1               # k135114294095936.py:11:19
	lwu	a3, 0(a0)
	lw	a0, 4(a0)
	li	a6, 128
	li	a4, 56
	csrr	a5, vlenb
	li	a7, 161
	mul	a5, a5, a7
	add	a5, sp, a5
	lui	a7, 7
	addi	a7, a7, -2016
	add	a5, a5, a7
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	.loc	1 7 19                          # k135114294095936.py:7:19
	vsrl.vi	v8, v8, 25
	.loc	1 11 19                         # k135114294095936.py:11:19
	slli	a0, a0, 32
	csrr	a5, vlenb
	li	a7, 105
	mul	a5, a5, a7
	add	a5, sp, a5
	lui	a7, 7
	addi	a7, a7, -2016
	add	a5, a5, a7
	vs8r.v	v8, (a5)                        # vscale x 64-byte Folded Spill
	.loc	1 7 19                          # k135114294095936.py:7:19
	vadd.vv	v8, v0, v8
	vsra.vi	v8, v8, 7
	.loc	1 11 19                         # k135114294095936.py:11:19
	or	a0, a0, a3
	.loc	1 12 33                         # k135114294095936.py:12:33
	vsetivli	zero, 16, e64, m8, ta, ma
	vmv.v.x	v24, a0
	csrr	a0, vlenb
	li	a3, 161
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 7
	addi	a3, a3, -2016
	add	a0, a0, a3
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a3, 7
	addi	a3, a3, -2016
	add	a0, a0, a3
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 15 18                         # k135114294095936.py:15:18
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v24, v8
	.loc	1 19 32                         # k135114294095936.py:19:32
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v24, v16, a4
	vand.vx	v24, v24, a6
	vadd.vv	v8, v24, v16
	csrr	a0, vlenb
	slli	a3, a0, 7
	add	a0, a3, a0
	add	a0, sp, a0
	lui	a3, 7
	addi	a3, a3, -2016
	add	a0, a0, a3
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	li	a0, -64
	csrr	a3, vlenb
	li	a4, 153
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 7
	addi	a4, a4, -2016
	add	a3, a3, a4
	vl8r.v	v8, (a3)                        # vscale x 64-byte Folded Reload
	.loc	1 9 19                          # k135114294095936.py:9:19
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v8, v8, a0
	vsub.vv	v8, v0, v8
	csrr	a0, vlenb
	li	a3, 145
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 7
	addi	a3, a3, -2016
	add	a0, a0, a3
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	.loc	1 21 45                         # k135114294095936.py:21:45
	vsll.vi	v16, v16, 13
	.loc	1 21 30 is_stmt 0               # k135114294095936.py:21:30
	vadd.vv	v8, v16, v8
	csrr	a0, vlenb
	li	a3, 24
	mul	a0, a0, a3
	add	a0, sp, a0
	lui	a3, 7
	addi	a3, a3, -2016
	add	a0, a0, a3
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	li	a3, 1792
	.loc	1 6 21 is_stmt 1                # k135114294095936.py:6:21
	vmslt.vx	v8, v0, a3
	li	a0, 64
	li	a4, 96
	vid.v	v0
	.loc	1 5 23                          # k135114294095936.py:5:23
	vadd.vx	v16, v0, a0
	vor.vx	v24, v16, a2
	vadd.vx	v16, v0, a4
	vor.vx	v16, v16, a2
	csrr	a4, vlenb
	li	a5, 56
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114294095936.py:6:21
	vmslt.vx	v10, v16, a3
	csrr	a4, vlenb
	li	a5, 40
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vs8r.v	v24, (a4)                       # vscale x 64-byte Folded Spill
	vmslt.vx	v9, v24, a3
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v9, v10, 4
	csrr	a4, vlenb
	li	a5, 137
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vs1r.v	v9, (a4)                        # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294095936.py:5:23
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vx	v16, v0, s3
	vor.vx	v16, v16, a2
	csrr	a4, vlenb
	slli	a4, a4, 5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114294095936.py:6:21
	vmslt.vx	v10, v16, a3
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v8, v10, 4
	csrr	a4, vlenb
	li	a5, 121
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vs1r.v	v8, (a4)                        # vscale x 8-byte Folded Spill
	.loc	1 13 30                         # k135114294095936.py:13:30
	slli	a4, a2, 1
	add	a1, a1, a4
	.loc	1 13 35 is_stmt 0               # k135114294095936.py:13:35
	vsetvli	zero, a0, e16, m8, ta, mu
	vmv.v.i	v24, 0
	addi	a4, a1, 128
	vmv.v.i	v8, 0
	csrr	a5, vlenb
	li	a7, 137
	mul	a5, a5, a7
	add	a5, sp, a5
	lui	a7, 7
	addi	a7, a7, -2016
	add	a5, a5, a7
	vl1r.v	v0, (a5)                        # vscale x 8-byte Folded Reload
	vle16.v	v8, (a4), v0.t
	csrr	a4, vlenb
	li	a5, 145
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vs8r.v	v8, (a4)                        # vscale x 64-byte Folded Spill
	li	a4, 192
	li	a5, 224
	vsetvli	zero, s3, e32, m8, ta, ma
	vid.v	v0
	vmv.v.v	v8, v0
	.loc	1 5 23 is_stmt 1                # k135114294095936.py:5:23
	vadd.vx	v16, v0, a4
	vor.vx	v0, v16, a2
	vadd.vx	v16, v8, a5
	vor.vx	v16, v16, a2
	csrr	a4, vlenb
	li	a5, 96
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114294095936.py:6:21
	vmslt.vx	v8, v16, a3
	csrr	a4, vlenb
	li	a5, 153
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vs1r.v	v8, (a4)                        # vscale x 8-byte Folded Spill
	csrr	a4, vlenb
	li	a5, 88
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vs8r.v	v0, (a4)                        # vscale x 64-byte Folded Spill
	vmslt.vx	v16, v0, a3
	csrr	a4, vlenb
	li	a5, 153
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vl1r.v	v8, (a4)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v16, v8, 4
	csrr	a4, vlenb
	li	a5, 72
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 7
	addi	a5, a5, -2016
	add	a4, a4, a5
	vs1r.v	v16, (a4)                       # vscale x 8-byte Folded Spill
	li	a4, 160
	vsetvli	zero, s3, e32, m8, ta, ma
	vid.v	v8
	.loc	1 5 23                          # k135114294095936.py:5:23
	vadd.vx	v16, v8, a6
	vadd.vx	v0, v8, a4
	vor.vx	v8, v16, a2
	vor.vx	v16, v0, a2
	csrr	a2, vlenb
	li	a4, 80
	mul	a2, a2, a4
	add	a2, sp, a2
	lui	a4, 7
	addi	a4, a4, -2016
	add	a2, a2, a4
	vs8r.v	v16, (a2)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114294095936.py:6:21
	vmslt.vx	v7, v16, a3
	csrr	a2, vlenb
	slli	a2, a2, 6
	add	a2, sp, a2
	lui	a4, 7
	addi	a4, a4, -2016
	add	a2, a2, a4
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	vmslt.vx	v0, v8, a3
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v0, v7, 4
	.loc	1 13 35                         # k135114294095936.py:13:35
	addi	a2, a1, 256
	vmv8r.v	v8, v24
	csrr	a3, vlenb
	li	a4, 104
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 7
	addi	a4, a4, -2016
	add	a3, a3, a4
	vs1r.v	v0, (a3)                        # vscale x 8-byte Folded Spill
	vsetvli	zero, a0, e16, m8, ta, mu
	vle16.v	v8, (a2), v0.t
	csrr	a2, vlenb
	li	a3, 153
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 7
	addi	a3, a3, -2016
	add	a2, a2, a3
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	addi	a2, a1, 384
	vmv8r.v	v16, v24
	csrr	a3, vlenb
	li	a4, 72
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 7
	addi	a4, a4, -2016
	add	a3, a3, a4
	vl1r.v	v0, (a3)                        # vscale x 8-byte Folded Reload
	vle16.v	v16, (a2), v0.t
	csrr	a2, vlenb
	li	a3, 121
	mul	a2, a2, a3
	add	a2, sp, a2
	lui	a3, 7
	addi	a3, a3, -2016
	add	a2, a2, a3
	vl1r.v	v0, (a2)                        # vscale x 8-byte Folded Reload
	vle16.v	v24, (a1), v0.t
	.loc	1 6 21                          # k135114294095936.py:6:21
	vmv1r.v	v8, v0
	csrr	a1, vlenb
	li	a2, 137
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl1r.v	v9, (a1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v8, v9, 8
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs1r.v	v8, (a1)                        # vscale x 8-byte Folded Spill
	lui	a1, 6
	addi	a1, a1, 360
	add	s11, sp, a1
	lui	a1, 6
	addi	a1, a1, 1536
	add	a1, sp, a1
	lui	a2, 6
	addi	a2, a2, 1920
	add	a2, sp, a2
	lui	a3, 6
	addi	a3, a3, 1792
	add	a3, sp, a3
	lui	a4, 6
	addi	a4, a4, 1664
	add	a4, sp, a4
	csrr	a5, vlenb
	slli	a6, a5, 7
	add	a5, a6, a5
	add	a5, sp, a5
	lui	a6, 7
	addi	a6, a6, -2016
	add	a5, a5, a6
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	.loc	1 21 33                         # k135114294095936.py:21:33
	vsetvli	zero, zero, e64, m8, ta, ma
	vsll.vi	v8, v8, 6
	csrr	a5, vlenb
	li	a6, 24
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 7
	addi	a6, a6, -2016
	add	a5, a5, a6
	vl8r.v	v0, (a5)                        # vscale x 64-byte Folded Reload
	.loc	1 21 40 is_stmt 0               # k135114294095936.py:21:40
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v8, v8, v0
	vmv2r.v	v0, v24
	csrr	a5, vlenb
	slli	a6, a5, 7
	add	a5, a6, a5
	add	a5, sp, a5
	lui	a6, 7
	addi	a6, a6, -2016
	add	a5, a5, a6
	vs8r.v	v0, (a5)                        # vscale x 64-byte Folded Spill
	.loc	1 13 35 is_stmt 1               # k135114294095936.py:13:35
	vsetvli	zero, a0, e16, m8, ta, ma
	vse16.v	v24, (a1)
	vmv2r.v	v24, v16
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	vse16.v	v16, (a2)
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v16, (a3)
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v16, (a4)
	.loc	1 21 56                         # k135114294095936.py:21:56
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	.loc	1 13 35                         # k135114294095936.py:13:35
	lh	a0, 1296(s11)
	sd	a0, 352(sp)                     # 8-byte Folded Spill
	lh	a0, 1298(s11)
	sd	a0, 360(sp)                     # 8-byte Folded Spill
	lh	a0, 1300(s11)
	sd	a0, 368(sp)                     # 8-byte Folded Spill
	lh	a0, 1302(s11)
	sd	a0, 376(sp)                     # 8-byte Folded Spill
	lh	a0, 1288(s11)
	sd	a0, 320(sp)                     # 8-byte Folded Spill
	lh	a0, 1290(s11)
	sd	a0, 328(sp)                     # 8-byte Folded Spill
	lh	a0, 1292(s11)
	sd	a0, 336(sp)                     # 8-byte Folded Spill
	lh	a0, 1294(s11)
	sd	a0, 344(sp)                     # 8-byte Folded Spill
	lh	a0, 1280(s11)
	sd	a0, 288(sp)                     # 8-byte Folded Spill
	lh	a0, 1282(s11)
	sd	a0, 296(sp)                     # 8-byte Folded Spill
	lh	a0, 1284(s11)
	sd	a0, 304(sp)                     # 8-byte Folded Spill
	lh	a0, 1286(s11)
	sd	a0, 312(sp)                     # 8-byte Folded Spill
	lh	a0, 1272(s11)
	sd	a0, 256(sp)                     # 8-byte Folded Spill
	lh	a0, 1274(s11)
	sd	a0, 264(sp)                     # 8-byte Folded Spill
	lh	a0, 1276(s11)
	sd	a0, 272(sp)                     # 8-byte Folded Spill
	lh	a0, 1278(s11)
	sd	a0, 280(sp)                     # 8-byte Folded Spill
	lh	a0, 1264(s11)
	sd	a0, 224(sp)                     # 8-byte Folded Spill
	lh	a0, 1266(s11)
	sd	a0, 232(sp)                     # 8-byte Folded Spill
	lh	a0, 1268(s11)
	sd	a0, 240(sp)                     # 8-byte Folded Spill
	lh	a0, 1270(s11)
	sd	a0, 248(sp)                     # 8-byte Folded Spill
	lh	a0, 1680(s11)
	sd	a0, 1504(sp)                    # 8-byte Folded Spill
	lh	a0, 1682(s11)
	sd	a0, 1512(sp)                    # 8-byte Folded Spill
	lh	a0, 1684(s11)
	sd	a0, 1520(sp)                    # 8-byte Folded Spill
	lh	a0, 1686(s11)
	sd	a0, 1528(sp)                    # 8-byte Folded Spill
	lh	a0, 1672(s11)
	sd	a0, 1472(sp)                    # 8-byte Folded Spill
	lh	a0, 1674(s11)
	sd	a0, 1480(sp)                    # 8-byte Folded Spill
	lh	a0, 1676(s11)
	sd	a0, 1488(sp)                    # 8-byte Folded Spill
	lh	a0, 1678(s11)
	sd	a0, 1496(sp)                    # 8-byte Folded Spill
	lh	a0, 1664(s11)
	sd	a0, 1440(sp)                    # 8-byte Folded Spill
	lh	a0, 1666(s11)
	sd	a0, 1448(sp)                    # 8-byte Folded Spill
	lh	a0, 1668(s11)
	sd	a0, 1456(sp)                    # 8-byte Folded Spill
	lh	a0, 1670(s11)
	sd	a0, 1464(sp)                    # 8-byte Folded Spill
	lh	a0, 1656(s11)
	sd	a0, 1408(sp)                    # 8-byte Folded Spill
	lh	a0, 1658(s11)
	sd	a0, 1416(sp)                    # 8-byte Folded Spill
	lh	a0, 1660(s11)
	sd	a0, 1424(sp)                    # 8-byte Folded Spill
	lh	a0, 1662(s11)
	sd	a0, 1432(sp)                    # 8-byte Folded Spill
	lh	a0, 1648(s11)
	sd	a0, 1376(sp)                    # 8-byte Folded Spill
	lh	a0, 1650(s11)
	sd	a0, 1384(sp)                    # 8-byte Folded Spill
	lh	a0, 1652(s11)
	sd	a0, 1392(sp)                    # 8-byte Folded Spill
	lh	a0, 1654(s11)
	sd	a0, 1400(sp)                    # 8-byte Folded Spill
	lh	a0, 1552(s11)
	sd	a0, 1120(sp)                    # 8-byte Folded Spill
	lh	a0, 1554(s11)
	sd	a0, 1128(sp)                    # 8-byte Folded Spill
	lh	a0, 1556(s11)
	sd	a0, 1136(sp)                    # 8-byte Folded Spill
	lh	a0, 1558(s11)
	sd	a0, 1144(sp)                    # 8-byte Folded Spill
	lh	a0, 1544(s11)
	sd	a0, 1088(sp)                    # 8-byte Folded Spill
	lh	a0, 1546(s11)
	sd	a0, 1096(sp)                    # 8-byte Folded Spill
	lh	a0, 1548(s11)
	sd	a0, 1104(sp)                    # 8-byte Folded Spill
	lh	a0, 1550(s11)
	sd	a0, 1112(sp)                    # 8-byte Folded Spill
	lh	a0, 1536(s11)
	sd	a0, 1056(sp)                    # 8-byte Folded Spill
	lh	a0, 1538(s11)
	sd	a0, 1064(sp)                    # 8-byte Folded Spill
	lh	a0, 1540(s11)
	sd	a0, 1072(sp)                    # 8-byte Folded Spill
	lh	a0, 1542(s11)
	sd	a0, 1080(sp)                    # 8-byte Folded Spill
	lh	a0, 1528(s11)
	sd	a0, 1024(sp)                    # 8-byte Folded Spill
	lh	a0, 1530(s11)
	sd	a0, 1032(sp)                    # 8-byte Folded Spill
	lh	a0, 1532(s11)
	sd	a0, 1040(sp)                    # 8-byte Folded Spill
	lh	a0, 1534(s11)
	sd	a0, 1048(sp)                    # 8-byte Folded Spill
	lh	a0, 1520(s11)
	sd	a0, 992(sp)                     # 8-byte Folded Spill
	lh	a0, 1522(s11)
	sd	a0, 1000(sp)                    # 8-byte Folded Spill
	lh	a0, 1524(s11)
	sd	a0, 1008(sp)                    # 8-byte Folded Spill
	lh	a0, 1526(s11)
	sd	a0, 1016(sp)                    # 8-byte Folded Spill
	lh	a0, 1424(s11)
	sd	a0, 736(sp)                     # 8-byte Folded Spill
	lh	a0, 1426(s11)
	sd	a0, 744(sp)                     # 8-byte Folded Spill
	lh	a0, 1428(s11)
	sd	a0, 752(sp)                     # 8-byte Folded Spill
	lh	a0, 1430(s11)
	sd	a0, 760(sp)                     # 8-byte Folded Spill
	lh	a0, 1416(s11)
	sd	a0, 704(sp)                     # 8-byte Folded Spill
	lh	a0, 1418(s11)
	sd	a0, 712(sp)                     # 8-byte Folded Spill
	lh	a0, 1420(s11)
	sd	a0, 720(sp)                     # 8-byte Folded Spill
	lh	a0, 1422(s11)
	sd	a0, 728(sp)                     # 8-byte Folded Spill
	lh	a0, 1408(s11)
	sd	a0, 672(sp)                     # 8-byte Folded Spill
	lh	a0, 1410(s11)
	sd	a0, 680(sp)                     # 8-byte Folded Spill
	lh	a0, 1412(s11)
	sd	a0, 688(sp)                     # 8-byte Folded Spill
	lh	a0, 1414(s11)
	sd	a0, 696(sp)                     # 8-byte Folded Spill
	lh	a0, 1400(s11)
	sd	a0, 640(sp)                     # 8-byte Folded Spill
	lh	a0, 1402(s11)
	sd	a0, 648(sp)                     # 8-byte Folded Spill
	lh	a0, 1404(s11)
	sd	a0, 656(sp)                     # 8-byte Folded Spill
	lh	a0, 1406(s11)
	sd	a0, 664(sp)                     # 8-byte Folded Spill
	lh	a0, 1392(s11)
	sd	a0, 608(sp)                     # 8-byte Folded Spill
	lh	a0, 1394(s11)
	sd	a0, 616(sp)                     # 8-byte Folded Spill
	lh	a0, 1396(s11)
	sd	a0, 624(sp)                     # 8-byte Folded Spill
	lh	a0, 1398(s11)
	sd	a0, 632(sp)                     # 8-byte Folded Spill
	lh	a0, 1256(s11)
	sd	a0, 192(sp)                     # 8-byte Folded Spill
	lh	a0, 1258(s11)
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	lh	a0, 1260(s11)
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	lh	a0, 1262(s11)
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	lh	a0, 1248(s11)
	sd	a0, 160(sp)                     # 8-byte Folded Spill
	lh	a0, 1250(s11)
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	lh	a0, 1252(s11)
	sd	a0, 176(sp)                     # 8-byte Folded Spill
	lh	a0, 1254(s11)
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	lh	a0, 1240(s11)
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	lh	a0, 1242(s11)
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lh	a0, 1244(s11)
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	lh	a0, 1246(s11)
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	lh	a0, 1232(s11)
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	lh	a0, 1234(s11)
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	lh	a0, 1236(s11)
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	lh	a0, 1238(s11)
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	lh	a0, 1640(s11)
	sd	a0, 1344(sp)                    # 8-byte Folded Spill
	lh	a0, 1642(s11)
	sd	a0, 1352(sp)                    # 8-byte Folded Spill
	lh	a0, 1644(s11)
	sd	a0, 1360(sp)                    # 8-byte Folded Spill
	lh	a0, 1646(s11)
	sd	a0, 1368(sp)                    # 8-byte Folded Spill
	lh	a0, 1632(s11)
	sd	a0, 1312(sp)                    # 8-byte Folded Spill
	lh	a0, 1634(s11)
	sd	a0, 1320(sp)                    # 8-byte Folded Spill
	lh	a0, 1636(s11)
	sd	a0, 1328(sp)                    # 8-byte Folded Spill
	lh	a0, 1638(s11)
	sd	a0, 1336(sp)                    # 8-byte Folded Spill
	lh	a0, 1624(s11)
	sd	a0, 1280(sp)                    # 8-byte Folded Spill
	lh	a0, 1626(s11)
	sd	a0, 1288(sp)                    # 8-byte Folded Spill
	lh	a0, 1628(s11)
	sd	a0, 1296(sp)                    # 8-byte Folded Spill
	lh	a0, 1630(s11)
	sd	a0, 1304(sp)                    # 8-byte Folded Spill
	lh	a0, 1616(s11)
	sd	a0, 1248(sp)                    # 8-byte Folded Spill
	lh	a0, 1618(s11)
	sd	a0, 1256(sp)                    # 8-byte Folded Spill
	lh	a0, 1620(s11)
	sd	a0, 1264(sp)                    # 8-byte Folded Spill
	lh	a0, 1622(s11)
	sd	a0, 1272(sp)                    # 8-byte Folded Spill
	lh	a0, 1512(s11)
	sd	a0, 960(sp)                     # 8-byte Folded Spill
	lh	a0, 1514(s11)
	sd	a0, 968(sp)                     # 8-byte Folded Spill
	lh	a0, 1516(s11)
	sd	a0, 976(sp)                     # 8-byte Folded Spill
	lh	a0, 1518(s11)
	sd	a0, 984(sp)                     # 8-byte Folded Spill
	lh	a0, 1504(s11)
	sd	a0, 928(sp)                     # 8-byte Folded Spill
	lh	a0, 1506(s11)
	sd	a0, 936(sp)                     # 8-byte Folded Spill
	lh	a0, 1508(s11)
	sd	a0, 944(sp)                     # 8-byte Folded Spill
	lh	a0, 1510(s11)
	sd	a0, 952(sp)                     # 8-byte Folded Spill
	lh	a0, 1496(s11)
	sd	a0, 896(sp)                     # 8-byte Folded Spill
	lh	a0, 1498(s11)
	sd	a0, 904(sp)                     # 8-byte Folded Spill
	lh	a0, 1500(s11)
	sd	a0, 912(sp)                     # 8-byte Folded Spill
	lh	a0, 1502(s11)
	sd	a0, 920(sp)                     # 8-byte Folded Spill
	lh	a0, 1488(s11)
	sd	a0, 864(sp)                     # 8-byte Folded Spill
	lh	a0, 1490(s11)
	sd	a0, 872(sp)                     # 8-byte Folded Spill
	lh	a0, 1492(s11)
	sd	a0, 880(sp)                     # 8-byte Folded Spill
	lh	a0, 1494(s11)
	sd	a0, 888(sp)                     # 8-byte Folded Spill
	lh	a0, 1384(s11)
	sd	a0, 576(sp)                     # 8-byte Folded Spill
	lh	a0, 1386(s11)
	sd	a0, 584(sp)                     # 8-byte Folded Spill
	lh	a0, 1388(s11)
	sd	a0, 592(sp)                     # 8-byte Folded Spill
	lh	a0, 1390(s11)
	sd	a0, 600(sp)                     # 8-byte Folded Spill
	lh	a0, 1376(s11)
	sd	a0, 544(sp)                     # 8-byte Folded Spill
	lh	a0, 1378(s11)
	sd	a0, 552(sp)                     # 8-byte Folded Spill
	lh	a0, 1380(s11)
	sd	a0, 560(sp)                     # 8-byte Folded Spill
	lh	a0, 1382(s11)
	sd	a0, 568(sp)                     # 8-byte Folded Spill
	lh	a0, 1368(s11)
	sd	a0, 512(sp)                     # 8-byte Folded Spill
	lh	a0, 1370(s11)
	sd	a0, 520(sp)                     # 8-byte Folded Spill
	lh	a0, 1372(s11)
	sd	a0, 528(sp)                     # 8-byte Folded Spill
	lh	a0, 1374(s11)
	sd	a0, 536(sp)                     # 8-byte Folded Spill
	lh	a0, 1360(s11)
	sd	a0, 480(sp)                     # 8-byte Folded Spill
	lh	a0, 1362(s11)
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	lh	a0, 1364(s11)
	sd	a0, 496(sp)                     # 8-byte Folded Spill
	lh	a0, 1366(s11)
	sd	a0, 504(sp)                     # 8-byte Folded Spill
	lh	a0, 1224(s11)
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	lh	a0, 1226(s11)
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	lh	a0, 1228(s11)
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	lh	a0, 1230(s11)
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	lh	s10, 1216(s11)
	lh	s4, 1218(s11)
	lh	a0, 1220(s11)
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	lh	a0, 1222(s11)
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	lh	s6, 1208(s11)
	lh	s9, 1210(s11)
	lh	s5, 1212(s11)
	lh	s8, 1214(s11)
	lh	a0, 1608(s11)
	sd	a0, 1216(sp)                    # 8-byte Folded Spill
	lh	a0, 1610(s11)
	sd	a0, 1224(sp)                    # 8-byte Folded Spill
	lh	a0, 1612(s11)
	sd	a0, 1232(sp)                    # 8-byte Folded Spill
	lh	a0, 1614(s11)
	sd	a0, 1240(sp)                    # 8-byte Folded Spill
	lh	a0, 1600(s11)
	sd	a0, 1184(sp)                    # 8-byte Folded Spill
	lh	a0, 1602(s11)
	sd	a0, 1192(sp)                    # 8-byte Folded Spill
	lh	a0, 1604(s11)
	sd	a0, 1200(sp)                    # 8-byte Folded Spill
	lh	a0, 1606(s11)
	sd	a0, 1208(sp)                    # 8-byte Folded Spill
	lh	a0, 1592(s11)
	sd	a0, 1152(sp)                    # 8-byte Folded Spill
	lh	a0, 1594(s11)
	sd	a0, 1160(sp)                    # 8-byte Folded Spill
	lh	a0, 1596(s11)
	sd	a0, 1168(sp)                    # 8-byte Folded Spill
	lh	a0, 1598(s11)
	sd	a0, 1176(sp)                    # 8-byte Folded Spill
	lh	a0, 1480(s11)
	sd	a0, 832(sp)                     # 8-byte Folded Spill
	lh	a0, 1482(s11)
	sd	a0, 840(sp)                     # 8-byte Folded Spill
	lh	a0, 1484(s11)
	sd	a0, 848(sp)                     # 8-byte Folded Spill
	lh	a0, 1486(s11)
	sd	a0, 856(sp)                     # 8-byte Folded Spill
	lh	a0, 1472(s11)
	sd	a0, 800(sp)                     # 8-byte Folded Spill
	lh	a0, 1474(s11)
	sd	a0, 808(sp)                     # 8-byte Folded Spill
	lh	a0, 1476(s11)
	sd	a0, 816(sp)                     # 8-byte Folded Spill
	lh	a0, 1478(s11)
	sd	a0, 824(sp)                     # 8-byte Folded Spill
	lh	a0, 1464(s11)
	sd	a0, 768(sp)                     # 8-byte Folded Spill
	lh	a0, 1466(s11)
	sd	a0, 776(sp)                     # 8-byte Folded Spill
	lh	a0, 1468(s11)
	sd	a0, 784(sp)                     # 8-byte Folded Spill
	lh	a0, 1470(s11)
	sd	a0, 792(sp)                     # 8-byte Folded Spill
	lh	a0, 1352(s11)
	sd	a0, 448(sp)                     # 8-byte Folded Spill
	lh	a0, 1354(s11)
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	lh	a0, 1356(s11)
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	lh	a0, 1358(s11)
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	lh	a0, 1344(s11)
	sd	a0, 416(sp)                     # 8-byte Folded Spill
	lh	a0, 1346(s11)
	sd	a0, 424(sp)                     # 8-byte Folded Spill
	lh	a0, 1348(s11)
	sd	a0, 432(sp)                     # 8-byte Folded Spill
	lh	a0, 1350(s11)
	sd	a0, 440(sp)                     # 8-byte Folded Spill
	lh	a0, 1336(s11)
	sd	a0, 384(sp)                     # 8-byte Folded Spill
	lh	a0, 1338(s11)
	sd	a0, 392(sp)                     # 8-byte Folded Spill
	lh	a0, 1340(s11)
	sd	a0, 400(sp)                     # 8-byte Folded Spill
	lh	a0, 1342(s11)
	sd	a0, 408(sp)                     # 8-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	add	a0, a0, a1
	ld	s7, -2016(a0)                   # 8-byte Folded Reload
	andi	a0, s7, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_1
	j	.LBB0_415
.LBB0_1:                                # %else
	andi	a0, s7, 2
	beqz	a0, .LBB0_2
	j	.LBB0_416
.LBB0_2:                                # %else2
	andi	a0, s7, 4
	beqz	a0, .LBB0_3
	j	.LBB0_417
.LBB0_3:                                # %else4
	andi	a0, s7, 8
	beqz	a0, .LBB0_4
	j	.LBB0_418
.LBB0_4:                                # %else6
	andi	a0, s7, 16
	beqz	a0, .LBB0_5
	j	.LBB0_419
.LBB0_5:                                # %else8
	andi	a0, s7, 32
	beqz	a0, .LBB0_6
	j	.LBB0_420
.LBB0_6:                                # %else10
	andi	a0, s7, 64
	beqz	a0, .LBB0_8
.LBB0_7:                                # %cond.store11
	.loc	1 0 56 is_stmt 0                # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_8:                                # %else12
	andi	a0, s7, 128
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 0 0                           # k135114294095936.py:0
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	beqz	a0, .LBB0_10
# %bb.9:                                # %cond.store13
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_10:                               # %else14
	andi	a0, s7, 256
	beqz	a0, .LBB0_12
# %bb.11:                               # %cond.store15
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_12:                               # %else16
	andi	a0, s7, 512
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 0 0                           # k135114294095936.py:0
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v16, v8, v24
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	beqz	a0, .LBB0_14
# %bb.13:                               # %cond.store17
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_14:                               # %else18
	andi	a0, s7, 1024
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 0 0                           # k135114294095936.py:0
	vsetvli	zero, zero, e64, m8, ta, ma
	li	a1, 56
	vsrl.vx	v8, v8, a1
	.loc	1 21 56                         # k135114294095936.py:21:56
	beqz	a0, .LBB0_16
# %bb.15:                               # %cond.store19
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 360(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_16:                               # %else20
	slli	a0, s7, 52
	.loc	1 0 0                           # k135114294095936.py:0
	li	a1, 128
	vand.vx	v16, v8, a1
	.loc	1 21 56                         # k135114294095936.py:21:56
	bgez	a0, .LBB0_18
# %bb.17:                               # %cond.store21
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 6
	addi	a0, a0, 512
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_18:                               # %else22
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v16, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 51
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 21 0                          # k135114294095936.py:21
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	bgez	a0, .LBB0_20
# %bb.19:                               # %cond.store23
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 6
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_20:                               # %else24
	slli	a0, s7, 50
	.loc	1 21 0                          # k135114294095936.py:21
	vsetivli	zero, 16, e64, m8, ta, ma
	vsll.vi	v8, v16, 6
	.loc	1 21 56                         # k135114294095936.py:21:56
	bgez	a0, .LBB0_22
# %bb.21:                               # %cond.store25
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_22:                               # %else26
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v8, v8, v16
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 49
	lui	a1, 6
	addi	a1, a1, -1752
	add	s11, sp, a1
	bgez	a0, .LBB0_24
# %bb.23:                               # %cond.store27
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1992(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_24:                               # %else28
	slli	a0, s7, 48
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_26
# %bb.25:                               # %cond.store29
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1872(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_26:                               # %else30
	slli	a0, s7, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_27
	j	.LBB0_421
.LBB0_27:                               # %else32
	slli	a0, s7, 46
	bgez	a0, .LBB0_28
	j	.LBB0_422
.LBB0_28:                               # %else34
	slli	a0, s7, 45
	bgez	a0, .LBB0_29
	j	.LBB0_423
.LBB0_29:                               # %else36
	slli	a0, s7, 44
	li	s9, 56
	bgez	a0, .LBB0_30
	j	.LBB0_424
.LBB0_30:                               # %else38
	slli	a0, s7, 43
	bgez	a0, .LBB0_31
	j	.LBB0_425
.LBB0_31:                               # %else40
	slli	a0, s7, 42
	li	s8, 128
	bgez	a0, .LBB0_33
.LBB0_32:                               # %cond.store41
	.loc	1 0 56                          # k135114294095936.py:0:56
	slli	s4, s4, 16
	fmv.w.x	fa0, s4
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_33:                               # %else42
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 41
	csrr	a1, vlenb
	li	a2, 113
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	li	s10, -64
	bgez	a0, .LBB0_35
# %bb.34:                               # %cond.store43
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 6
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_35:                               # %else44
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vsra.vi	v8, v24, 7
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsra.vi	v16, v8, 6
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 40
	vsrl.vi	v16, v8, 31
	bgez	a0, .LBB0_37
# %bb.36:                               # %cond.store45
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 6
	addi	a0, a0, -512
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_37:                               # %else46
	slli	a0, s7, 39
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a0, .LBB0_39
# %bb.38:                               # %cond.store47
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1176(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_39:                               # %else48
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vi	v0, v8, -2
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 38
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v24, v8, v16
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_41
# %bb.40:                               # %cond.store49
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	lui	a0, 7
	addi	a0, a0, -2016
	add	a0, sp, a0
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 7
	addi	a0, a0, -2016
	add	a0, sp, a0
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 6
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_41:                               # %else50
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v16, v8, s10
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsub.vv	v24, v8, v0
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 37
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vsrl.vx	v0, v8, s9
	bgez	a0, .LBB0_43
# %bb.42:                               # %cond.store51
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	lui	a0, 7
	addi	a0, a0, -2016
	add	a0, sp, a0
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 7
	addi	a0, a0, -2016
	add	a0, sp, a0
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 6
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_43:                               # %else52
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	vsll.vi	v24, v24, 13
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 36
	vsetivli	zero, 16, e64, m8, ta, ma
	vand.vx	v16, v0, s8
	bgez	a0, .LBB0_45
# %bb.44:                               # %cond.store53
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 23
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 816(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_45:                               # %else54
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v24, v8
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 35
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_47
# %bb.46:                               # %cond.store55
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 696(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_47:                               # %else56
	slli	a0, s7, 34
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_49
# %bb.48:                               # %cond.store57
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 6
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_49:                               # %else58
	slli	a0, s7, 33
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_51
# %bb.50:                               # %cond.store59
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 6
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 456(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_51:                               # %else60
	slli	a0, s7, 32
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_53
# %bb.52:                               # %cond.store61
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1536
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 336(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_53:                               # %else62
	slli	a0, s7, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_54
	j	.LBB0_426
.LBB0_54:                               # %else64
	slli	a0, s7, 30
	bgez	a0, .LBB0_55
	j	.LBB0_427
.LBB0_55:                               # %else66
	slli	a0, s7, 29
	bgez	a0, .LBB0_56
	j	.LBB0_428
.LBB0_56:                               # %else68
	slli	a0, s7, 28
	bgez	a0, .LBB0_57
	j	.LBB0_429
.LBB0_57:                               # %else70
	slli	a0, s7, 27
	bgez	a0, .LBB0_58
	j	.LBB0_430
.LBB0_58:                               # %else72
	slli	a0, s7, 26
	bgez	a0, .LBB0_59
	j	.LBB0_431
.LBB0_59:                               # %else74
	slli	a0, s7, 25
	lui	a1, 5
	addi	a1, a1, 208
	add	s4, sp, a1
	bgez	a0, .LBB0_61
.LBB0_60:                               # %cond.store75
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_61:                               # %else76
	slli	a0, s7, 24
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_63
# %bb.62:                               # %cond.store77
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_63:                               # %else78
	slli	a0, s7, 23
	bgez	a0, .LBB0_65
# %bb.64:                               # %cond.store79
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_65:                               # %else80
	slli	a0, s7, 22
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v16, v8, v24
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_67
# %bb.66:                               # %cond.store81
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_67:                               # %else82
	slli	a0, s7, 21
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s9
	bgez	a0, .LBB0_69
# %bb.68:                               # %cond.store83
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_69:                               # %else84
	slli	a0, s7, 20
	vand.vx	v16, v8, s8
	bgez	a0, .LBB0_71
# %bb.70:                               # %cond.store85
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 216(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, 1536
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_71:                               # %else86
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v16, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 19
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_73
# %bb.72:                               # %cond.store87
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_73:                               # %else88
	slli	a0, s7, 18
	vsetivli	zero, 16, e64, m8, ta, ma
	vsll.vi	v16, v16, 6
	bgez	a0, .LBB0_75
# %bb.74:                               # %cond.store89
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 232(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_75:                               # %else90
	slli	a0, s7, 17
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_77
# %bb.76:                               # %cond.store91
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 240(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_77:                               # %else92
	slli	a0, s7, 16
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_79
# %bb.78:                               # %cond.store93
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 248(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 936(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_79:                               # %else94
	slli	a0, s7, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_80
	j	.LBB0_432
.LBB0_80:                               # %else96
	slli	a0, s7, 14
	bgez	a0, .LBB0_81
	j	.LBB0_433
.LBB0_81:                               # %else98
	slli	a0, s7, 13
	bgez	a0, .LBB0_82
	j	.LBB0_434
.LBB0_82:                               # %else100
	slli	a0, s7, 12
	bgez	a0, .LBB0_83
	j	.LBB0_435
.LBB0_83:                               # %else102
	slli	a0, s7, 11
	bgez	a0, .LBB0_84
	j	.LBB0_436
.LBB0_84:                               # %else104
	slli	a0, s7, 10
	bgez	a0, .LBB0_86
.LBB0_85:                               # %cond.store105
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 296(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_86:                               # %else106
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 9
	csrr	a1, vlenb
	li	a2, 113
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_88
# %bb.87:                               # %cond.store107
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 304(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_88:                               # %else108
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vsra.vi	v8, v24, 7
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsra.vi	v16, v8, 6
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 8
	vsrl.vi	v16, v8, 31
	bgez	a0, .LBB0_90
# %bb.89:                               # %cond.store109
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 312(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, 512
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_90:                               # %else110
	slli	a0, s7, 7
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a0, .LBB0_92
# %bb.91:                               # %cond.store111
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 320(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 240(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_92:                               # %else112
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vi	v0, v8, -2
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 6
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v24, v8, v16
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_94
# %bb.93:                               # %cond.store113
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 328(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_94:                               # %else114
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v16, v8, s10
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsub.vv	v24, v8, v0
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 5
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vsrl.vx	v0, v8, s9
	bgez	a0, .LBB0_96
# %bb.95:                               # %cond.store115
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 336(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_96:                               # %else116
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	vsll.vi	v24, v24, 13
	vsetivli	zero, 16, e64, m8, ta, ma
	vand.vx	v16, v0, s8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 4
	lui	a1, 5
	addi	a1, a1, -1928
	add	s5, sp, a1
	bgez	a0, .LBB0_98
# %bb.97:                               # %cond.store117
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 344(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 2016(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_98:                               # %else118
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 3
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_100
# %bb.99:                               # %cond.store119
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 352(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1896(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_100:                              # %else120
	slli	a0, s7, 2
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_102
# %bb.101:                              # %cond.store121
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 360(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_102:                              # %else122
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s7, 1
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl1r.v	v8, (a1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	bgez	a0, .LBB0_104
# %bb.103:                              # %cond.store123
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 368(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs1r.v	v24, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl1r.v	v24, (a0)                       # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_104:                              # %else124
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	.loc	1 21 56                         # k135114294095936.py:21:56
	vmv.x.s	s4, v24
	bgez	s7, .LBB0_106
# %bb.105:                              # %cond.store125
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 376(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -512
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_106:                              # %else126
	andi	a0, s4, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_107
	j	.LBB0_437
.LBB0_107:                              # %else128
	andi	a0, s4, 2
	beqz	a0, .LBB0_108
	j	.LBB0_438
.LBB0_108:                              # %else130
	andi	a0, s4, 4
	beqz	a0, .LBB0_109
	j	.LBB0_439
.LBB0_109:                              # %else132
	andi	a0, s4, 8
	beqz	a0, .LBB0_110
	j	.LBB0_440
.LBB0_110:                              # %else134
	andi	a0, s4, 16
	beqz	a0, .LBB0_111
	j	.LBB0_441
.LBB0_111:                              # %else136
	andi	a0, s4, 32
	beqz	a0, .LBB0_112
	j	.LBB0_442
.LBB0_112:                              # %else138
	andi	a0, s4, 64
	beqz	a0, .LBB0_114
.LBB0_113:                              # %cond.store139
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_114:                              # %else140
	andi	a0, s4, 128
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_116
# %bb.115:                              # %cond.store141
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_116:                              # %else142
	andi	a0, s4, 256
	beqz	a0, .LBB0_118
# %bb.117:                              # %cond.store143
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_118:                              # %else144
	andi	a0, s4, 512
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v16, v8, v24
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_120
# %bb.119:                              # %cond.store145
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_120:                              # %else146
	andi	a0, s4, 1024
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s9
	beqz	a0, .LBB0_122
# %bb.121:                              # %cond.store147
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 600(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_122:                              # %else148
	slli	a0, s4, 52
	vand.vx	v16, v8, s8
	bgez	a0, .LBB0_124
# %bb.123:                              # %cond.store149
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, -1536
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_124:                              # %else150
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v16, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 51
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_126
# %bb.125:                              # %cond.store151
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_126:                              # %else152
	slli	a0, s4, 50
	vsetivli	zero, 16, e64, m8, ta, ma
	vsll.vi	v16, v16, 6
	bgez	a0, .LBB0_128
# %bb.127:                              # %cond.store153
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_128:                              # %else154
	slli	a0, s4, 49
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_130
# %bb.129:                              # %cond.store155
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 5
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_130:                              # %else156
	slli	a0, s4, 48
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_132
# %bb.131:                              # %cond.store157
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_132:                              # %else158
	slli	a0, s4, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_133
	j	.LBB0_443
.LBB0_133:                              # %else160
	slli	a0, s4, 46
	bgez	a0, .LBB0_134
	j	.LBB0_444
.LBB0_134:                              # %else162
	slli	a0, s4, 45
	bgez	a0, .LBB0_135
	j	.LBB0_445
.LBB0_135:                              # %else164
	slli	a0, s4, 44
	bgez	a0, .LBB0_136
	j	.LBB0_446
.LBB0_136:                              # %else166
	slli	a0, s4, 43
	lui	a1, 4
	addi	a1, a1, -64
	add	s5, sp, a1
	bgez	a0, .LBB0_137
	j	.LBB0_447
.LBB0_137:                              # %else168
	slli	a0, s4, 42
	bgez	a0, .LBB0_139
.LBB0_138:                              # %cond.store169
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 424(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_139:                              # %else170
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 41
	csrr	a1, vlenb
	li	a2, 113
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_141
# %bb.140:                              # %cond.store171
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 432(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_141:                              # %else172
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vsra.vi	v8, v24, 7
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsra.vi	v16, v8, 6
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 40
	vsrl.vi	v16, v8, 31
	bgez	a0, .LBB0_143
# %bb.142:                              # %cond.store173
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 440(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, 1536
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_143:                              # %else174
	slli	a0, s4, 39
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a0, .LBB0_145
# %bb.144:                              # %cond.store175
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 448(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_145:                              # %else176
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vi	v0, v8, -2
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 38
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v24, v8, v16
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_147
# %bb.146:                              # %cond.store177
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 456(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_147:                              # %else178
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v16, v8, s10
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsub.vv	v24, v8, v0
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 37
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vsrl.vx	v0, v8, s9
	bgez	a0, .LBB0_149
# %bb.148:                              # %cond.store179
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 464(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_149:                              # %else180
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	vsll.vi	v24, v24, 13
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 36
	vsetivli	zero, 16, e64, m8, ta, ma
	vand.vx	v16, v0, s8
	bgez	a0, .LBB0_151
# %bb.150:                              # %cond.store181
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 472(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 17
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1176(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_151:                              # %else182
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 35
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_153
# %bb.152:                              # %cond.store183
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 480(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1056(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_153:                              # %else184
	slli	a0, s4, 34
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_155
# %bb.154:                              # %cond.store185
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_155:                              # %else186
	slli	a0, s4, 33
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_157
# %bb.156:                              # %cond.store187
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 496(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_157:                              # %else188
	slli	a0, s4, 32
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_159
# %bb.158:                              # %cond.store189
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 504(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 512
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 696(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_159:                              # %else190
	slli	a0, s4, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_160
	j	.LBB0_448
.LBB0_160:                              # %else192
	slli	a0, s4, 30
	bgez	a0, .LBB0_161
	j	.LBB0_449
.LBB0_161:                              # %else194
	slli	a0, s4, 29
	bgez	a0, .LBB0_162
	j	.LBB0_450
.LBB0_162:                              # %else196
	slli	a0, s4, 28
	bgez	a0, .LBB0_163
	j	.LBB0_451
.LBB0_163:                              # %else198
	slli	a0, s4, 27
	bgez	a0, .LBB0_164
	j	.LBB0_452
.LBB0_164:                              # %else200
	slli	a0, s4, 26
	bgez	a0, .LBB0_165
	j	.LBB0_453
.LBB0_165:                              # %else202
	slli	a0, s4, 25
	bgez	a0, .LBB0_167
.LBB0_166:                              # %cond.store203
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_167:                              # %else204
	slli	a0, s4, 24
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_169
# %bb.168:                              # %cond.store205
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_169:                              # %else206
	slli	a0, s4, 23
	bgez	a0, .LBB0_171
# %bb.170:                              # %cond.store207
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_171:                              # %else208
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v16, v8, v24
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 22
	lui	a1, 3
	addi	a1, a1, 1896
	add	s5, sp, a1
	bgez	a0, .LBB0_173
# %bb.172:                              # %cond.store209
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_173:                              # %else210
	slli	a0, s4, 21
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s9
	bgez	a0, .LBB0_175
# %bb.174:                              # %cond.store211
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1896(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_175:                              # %else212
	slli	a0, s4, 20
	vand.vx	v16, v8, s8
	bgez	a0, .LBB0_177
# %bb.176:                              # %cond.store213
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 31
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_177:                              # %else214
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v16, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 19
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_179
# %bb.178:                              # %cond.store215
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_179:                              # %else216
	slli	a0, s4, 18
	vsetivli	zero, 16, e64, m8, ta, ma
	vsll.vi	v16, v16, 6
	bgez	a0, .LBB0_181
# %bb.180:                              # %cond.store217
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_181:                              # %else218
	slli	a0, s4, 17
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_183
# %bb.182:                              # %cond.store219
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_183:                              # %else220
	slli	a0, s4, 16
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_185
# %bb.184:                              # %cond.store221
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 632(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 15
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1296(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_185:                              # %else222
	slli	a0, s4, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_186
	j	.LBB0_454
.LBB0_186:                              # %else224
	slli	a0, s4, 14
	bgez	a0, .LBB0_187
	j	.LBB0_455
.LBB0_187:                              # %else226
	slli	a0, s4, 13
	bgez	a0, .LBB0_188
	j	.LBB0_456
.LBB0_188:                              # %else228
	slli	a0, s4, 12
	bgez	a0, .LBB0_189
	j	.LBB0_457
.LBB0_189:                              # %else230
	slli	a0, s4, 11
	bgez	a0, .LBB0_190
	j	.LBB0_458
.LBB0_190:                              # %else232
	slli	a0, s4, 10
	bgez	a0, .LBB0_192
.LBB0_191:                              # %cond.store233
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 680(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_192:                              # %else234
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 9
	csrr	a1, vlenb
	li	a2, 113
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_194
# %bb.193:                              # %cond.store235
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 688(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_194:                              # %else236
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vsra.vi	v8, v24, 7
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsra.vi	v16, v8, 6
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 8
	vsrl.vi	v16, v8, 31
	bgez	a0, .LBB0_196
# %bb.195:                              # %cond.store237
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 696(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 29
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_196:                              # %else238
	slli	a0, s4, 7
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a0, .LBB0_198
# %bb.197:                              # %cond.store239
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 704(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 600(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_198:                              # %else240
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vi	v0, v8, -2
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 6
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v24, v8, v16
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_200
# %bb.199:                              # %cond.store241
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 712(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_200:                              # %else242
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v16, v8, s10
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsub.vv	v24, v8, v0
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 5
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vsrl.vx	v0, v8, s9
	bgez	a0, .LBB0_202
# %bb.201:                              # %cond.store243
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 720(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 4
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_202:                              # %else244
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	vsll.vi	v24, v24, 13
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 4
	vsetivli	zero, 16, e64, m8, ta, ma
	vand.vx	v16, v0, s8
	bgez	a0, .LBB0_204
# %bb.203:                              # %cond.store245
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 728(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 7
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 240(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_204:                              # %else246
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v24, v8
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 3
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_206
# %bb.205:                              # %cond.store247
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 736(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_206:                              # %else248
	slli	a0, s4, 2
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_208
# %bb.207:                              # %cond.store249
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 744(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_208:                              # %else250
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl1r.v	v8, (a0)                        # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl1r.v	v9, (a0)                        # vscale x 8-byte Folded Reload
	vsetvli	zero, zero, e8, m1, ta, ma
	vslideup.vi	v8, v9, 8
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs1r.v	v8, (a0)                        # vscale x 8-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 1
	lui	a1, 3
	addi	a1, a1, -216
	add	s6, sp, a1
	bgez	a0, .LBB0_210
# %bb.209:                              # %cond.store251
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 752(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1992(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_210:                              # %else252
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	.loc	1 21 56                         # k135114294095936.py:21:56
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	add	a0, a0, a1
	ld	s5, -2016(a0)                   # 8-byte Folded Reload
	bgez	s4, .LBB0_212
# %bb.211:                              # %cond.store253
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 760(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 27
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1872(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_212:                              # %else254
	andi	a0, s5, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_213
	j	.LBB0_459
.LBB0_213:                              # %else256
	andi	a0, s5, 2
	beqz	a0, .LBB0_214
	j	.LBB0_460
.LBB0_214:                              # %else258
	andi	a0, s5, 4
	beqz	a0, .LBB0_215
	j	.LBB0_461
.LBB0_215:                              # %else260
	andi	a0, s5, 8
	beqz	a0, .LBB0_216
	j	.LBB0_462
.LBB0_216:                              # %else262
	andi	a0, s5, 16
	beqz	a0, .LBB0_217
	j	.LBB0_463
.LBB0_217:                              # %else264
	andi	a0, s5, 32
	beqz	a0, .LBB0_218
	j	.LBB0_464
.LBB0_218:                              # %else266
	andi	a0, s5, 64
	beqz	a0, .LBB0_220
.LBB0_219:                              # %cond.store267
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_220:                              # %else268
	andi	a0, s5, 128
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_222
# %bb.221:                              # %cond.store269
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 13
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_222:                              # %else270
	andi	a0, s5, 256
	beqz	a0, .LBB0_224
# %bb.223:                              # %cond.store271
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_224:                              # %else272
	andi	a0, s5, 512
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v16, v8, v24
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_226
# %bb.225:                              # %cond.store273
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_226:                              # %else274
	andi	a0, s5, 1024
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s9
	beqz	a0, .LBB0_228
# %bb.227:                              # %cond.store275
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 936(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_228:                              # %else276
	slli	a0, s5, 52
	vand.vx	v16, v8, s8
	bgez	a0, .LBB0_230
# %bb.229:                              # %cond.store277
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 25
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_230:                              # %else278
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v16, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 51
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_232
# %bb.231:                              # %cond.store279
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_232:                              # %else280
	slli	a0, s5, 50
	vsetivli	zero, 16, e64, m8, ta, ma
	vsll.vi	v16, v16, 6
	bgez	a0, .LBB0_234
# %bb.233:                              # %cond.store281
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_234:                              # %else282
	slli	a0, s5, 49
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_236
# %bb.235:                              # %cond.store283
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 456(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_236:                              # %else284
	slli	a0, s5, 48
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_238
# %bb.237:                              # %cond.store285
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 336(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_238:                              # %else286
	slli	a0, s5, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_239
	j	.LBB0_465
.LBB0_239:                              # %else288
	slli	a0, s5, 46
	bgez	a0, .LBB0_240
	j	.LBB0_466
.LBB0_240:                              # %else290
	slli	a0, s5, 45
	bgez	a0, .LBB0_241
	j	.LBB0_467
.LBB0_241:                              # %else292
	slli	a0, s5, 44
	bgez	a0, .LBB0_242
	j	.LBB0_468
.LBB0_242:                              # %else294
	slli	a0, s5, 43
	bgez	a0, .LBB0_243
	j	.LBB0_469
.LBB0_243:                              # %else296
	slli	a0, s5, 42
	bgez	a0, .LBB0_245
.LBB0_244:                              # %cond.store297
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 808(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_245:                              # %else298
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v16, v8
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 41
	lui	a1, 2
	addi	a1, a1, 1744
	add	s4, sp, a1
	bgez	a0, .LBB0_247
# %bb.246:                              # %cond.store299
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 816(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_247:                              # %else300
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vsra.vi	v8, v24, 7
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsra.vi	v16, v8, 6
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 40
	vsrl.vi	v16, v8, 31
	bgez	a0, .LBB0_249
# %bb.248:                              # %cond.store301
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 824(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 23
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_249:                              # %else302
	slli	a0, s5, 39
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a0, .LBB0_251
# %bb.250:                              # %cond.store303
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 832(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1776(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_251:                              # %else304
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vi	v0, v8, -2
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 38
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v24, v8, v16
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_253
# %bb.252:                              # %cond.store305
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 840(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_253:                              # %else306
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v16, v8, s10
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsub.vv	v24, v8, v0
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 37
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vsrl.vx	v0, v8, s9
	bgez	a0, .LBB0_255
# %bb.254:                              # %cond.store307
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 848(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_255:                              # %else308
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	vsll.vi	v24, v24, 13
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 36
	vsetivli	zero, 16, e64, m8, ta, ma
	vand.vx	v16, v0, s8
	bgez	a0, .LBB0_257
# %bb.256:                              # %cond.store309
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 856(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 11
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_257:                              # %else310
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v24, v8
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 35
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_259
# %bb.258:                              # %cond.store311
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 864(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_259:                              # %else312
	slli	a0, s5, 34
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_261
# %bb.260:                              # %cond.store313
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 872(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_261:                              # %else314
	slli	a0, s5, 33
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_263
# %bb.262:                              # %cond.store315
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 880(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_263:                              # %else316
	slli	a0, s5, 32
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_265
# %bb.264:                              # %cond.store317
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 888(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 936(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_265:                              # %else318
	slli	a0, s5, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_266
	j	.LBB0_470
.LBB0_266:                              # %else320
	slli	a0, s5, 30
	bgez	a0, .LBB0_267
	j	.LBB0_471
.LBB0_267:                              # %else322
	slli	a0, s5, 29
	bgez	a0, .LBB0_268
	j	.LBB0_472
.LBB0_268:                              # %else324
	slli	a0, s5, 28
	bgez	a0, .LBB0_269
	j	.LBB0_473
.LBB0_269:                              # %else326
	slli	a0, s5, 27
	bgez	a0, .LBB0_270
	j	.LBB0_474
.LBB0_270:                              # %else328
	slli	a0, s5, 26
	bgez	a0, .LBB0_271
	j	.LBB0_475
.LBB0_271:                              # %else330
	slli	a0, s5, 25
	bgez	a0, .LBB0_273
.LBB0_272:                              # %cond.store331
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 944(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_273:                              # %else332
	slli	a0, s5, 24
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_275
# %bb.274:                              # %cond.store333
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 952(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_275:                              # %else334
	slli	a0, s5, 23
	bgez	a0, .LBB0_277
# %bb.276:                              # %cond.store335
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 960(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_277:                              # %else336
	slli	a0, s5, 22
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v16, v8, v24
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_279
# %bb.278:                              # %cond.store337
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 968(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_279:                              # %else338
	slli	a0, s5, 21
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s9
	bgez	a0, .LBB0_281
# %bb.280:                              # %cond.store339
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 976(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_281:                              # %else340
	.loc	1 0 56                          # k135114294095936.py:0:56
	vand.vx	v16, v8, s8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 20
	lui	a1, 2
	addi	a1, a1, -392
	add	s6, sp, a1
	bgez	a0, .LBB0_283
# %bb.282:                              # %cond.store341
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 984(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 19
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_283:                              # %else342
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v16, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 19
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_285
# %bb.284:                              # %cond.store343
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 992(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_285:                              # %else344
	slli	a0, s5, 18
	vsetivli	zero, 16, e64, m8, ta, ma
	vsll.vi	v16, v16, 6
	bgez	a0, .LBB0_287
# %bb.286:                              # %cond.store345
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1000(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_287:                              # %else346
	slli	a0, s5, 17
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_289
# %bb.288:                              # %cond.store347
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1008(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_289:                              # %else348
	slli	a0, s5, 16
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_291
# %bb.290:                              # %cond.store349
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1016(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_291:                              # %else350
	slli	a0, s5, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_292
	j	.LBB0_476
.LBB0_292:                              # %else352
	slli	a0, s5, 14
	bgez	a0, .LBB0_293
	j	.LBB0_477
.LBB0_293:                              # %else354
	slli	a0, s5, 13
	bgez	a0, .LBB0_294
	j	.LBB0_478
.LBB0_294:                              # %else356
	slli	a0, s5, 12
	bgez	a0, .LBB0_295
	j	.LBB0_479
.LBB0_295:                              # %else358
	slli	a0, s5, 11
	bgez	a0, .LBB0_296
	j	.LBB0_480
.LBB0_296:                              # %else360
	slli	a0, s5, 10
	bgez	a0, .LBB0_298
.LBB0_297:                              # %cond.store361
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1064(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1200(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_298:                              # %else362
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v24, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 9
	csrr	a1, vlenb
	li	a2, 113
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vadd.vv	v8, v16, v8
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_300
# %bb.299:                              # %cond.store363
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1072(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_300:                              # %else364
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vsra.vi	v8, v24, 7
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsra.vi	v16, v8, 6
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 8
	vsrl.vi	v16, v8, 31
	bgez	a0, .LBB0_302
# %bb.301:                              # %cond.store365
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1080(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 17
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_302:                              # %else366
	slli	a0, s5, 7
	csrr	a1, vlenb
	li	a2, 80
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a0, .LBB0_304
# %bb.303:                              # %cond.store367
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1088(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 840(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_304:                              # %else368
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vi	v0, v8, -2
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 6
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v24, v8, v16
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_306
# %bb.305:                              # %cond.store369
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1096(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_306:                              # %else370
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v16, v8, s10
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsub.vv	v24, v8, v0
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 5
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vsrl.vx	v0, v8, s9
	bgez	a0, .LBB0_308
# %bb.307:                              # %cond.store371
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1104(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_308:                              # %else372
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	vsll.vi	v24, v24, 13
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 4
	vsetivli	zero, 16, e64, m8, ta, ma
	vand.vx	v16, v0, s8
	bgez	a0, .LBB0_310
# %bb.309:                              # %cond.store373
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1112(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 80
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 88
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 480(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_310:                              # %else374
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 3
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_312
# %bb.311:                              # %cond.store375
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1120(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 360(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_312:                              # %else376
	slli	a0, s5, 2
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_314
# %bb.313:                              # %cond.store377
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1128(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 31
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_314:                              # %else378
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s5, 1
	csrr	a1, vlenb
	li	a2, 104
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl1r.v	v8, (a1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	bgez	a0, .LBB0_316
# %bb.315:                              # %cond.store379
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1136(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs1r.v	v24, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 104
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl1r.v	v24, (a0)                       # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_316:                              # %else380
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	.loc	1 21 56                         # k135114294095936.py:21:56
	vmv.x.s	s4, v24
	bgez	s5, .LBB0_318
# %bb.317:                              # %cond.store381
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1144(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 15
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_318:                              # %else382
	andi	a0, s4, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_319
	j	.LBB0_481
.LBB0_319:                              # %else384
	andi	a0, s4, 2
	beqz	a0, .LBB0_320
	j	.LBB0_482
.LBB0_320:                              # %else386
	andi	a0, s4, 4
	beqz	a0, .LBB0_321
	j	.LBB0_483
.LBB0_321:                              # %else388
	andi	a0, s4, 8
	beqz	a0, .LBB0_322
	j	.LBB0_484
.LBB0_322:                              # %else390
	andi	a0, s4, 16
	lui	a1, 1
	addi	a1, a1, 1592
	add	s5, sp, a1
	beqz	a0, .LBB0_323
	j	.LBB0_485
.LBB0_323:                              # %else392
	andi	a0, s4, 32
	beqz	a0, .LBB0_324
	j	.LBB0_486
.LBB0_324:                              # %else394
	andi	a0, s4, 64
	beqz	a0, .LBB0_326
.LBB0_325:                              # %cond.store395
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_326:                              # %else396
	andi	a0, s4, 128
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_328
# %bb.327:                              # %cond.store397
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 7
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_328:                              # %else398
	andi	a0, s4, 256
	beqz	a0, .LBB0_330
# %bb.329:                              # %cond.store399
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_330:                              # %else400
	andi	a0, s4, 512
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v24, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v16, v8, v24
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v16, (a1)                       # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_332
# %bb.331:                              # %cond.store401
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 27
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_332:                              # %else402
	andi	a0, s4, 1024
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s9
	beqz	a0, .LBB0_334
# %bb.333:                              # %cond.store403
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1176(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_334:                              # %else404
	slli	a0, s4, 52
	vand.vx	v16, v8, s8
	bgez	a0, .LBB0_336
# %bb.335:                              # %cond.store405
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 13
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_336:                              # %else406
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v16, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 51
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_338
# %bb.337:                              # %cond.store407
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_338:                              # %else408
	slli	a0, s4, 50
	vsetivli	zero, 16, e64, m8, ta, ma
	vsll.vi	v16, v16, 6
	bgez	a0, .LBB0_340
# %bb.339:                              # %cond.store409
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 25
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_340:                              # %else410
	slli	a0, s4, 49
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_342
# %bb.341:                              # %cond.store411
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_342:                              # %else412
	slli	a0, s4, 48
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_344
# %bb.343:                              # %cond.store413
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 3
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 576(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_344:                              # %else414
	slli	a0, s4, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_345
	j	.LBB0_487
.LBB0_345:                              # %else416
	slli	a0, s4, 46
	bgez	a0, .LBB0_346
	j	.LBB0_488
.LBB0_346:                              # %else418
	slli	a0, s4, 45
	bgez	a0, .LBB0_347
	j	.LBB0_489
.LBB0_347:                              # %else420
	slli	a0, s4, 44
	bgez	a0, .LBB0_348
	j	.LBB0_490
.LBB0_348:                              # %else422
	slli	a0, s4, 43
	bgez	a0, .LBB0_349
	j	.LBB0_491
.LBB0_349:                              # %else424
	slli	a0, s4, 42
	bgez	a0, .LBB0_351
.LBB0_350:                              # %cond.store425
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1192(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_351:                              # %else426
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 105
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v24, v8, v16
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 41
	csrr	a1, vlenb
	li	a2, 113
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vadd.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 137
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_353
# %bb.352:                              # %cond.store427
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1200(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_353:                              # %else428
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vsra.vi	v8, v24, 7
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsra.vi	v16, v8, 6
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 40
	vsrl.vi	v16, v8, 31
	bgez	a0, .LBB0_355
# %bb.354:                              # %cond.store429
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1208(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 11
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_355:                              # %else430
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 39
	addi	s5, sp, 2047
	addi	s5, s5, 1505
	bgez	a0, .LBB0_357
# %bb.356:                              # %cond.store431
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1216(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 2016(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_357:                              # %else432
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vi	v0, v8, -2
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 38
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v24, v8, v16
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v24, (a1)                       # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_359
# %bb.358:                              # %cond.store433
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1224(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	li	a0, 21
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_359:                              # %else434
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vand.vx	v16, v8, s10
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsub.vv	v24, v8, v0
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 37
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vsrl.vx	v0, v8, s9
	bgez	a0, .LBB0_361
# %bb.360:                              # %cond.store435
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1232(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v0, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_361:                              # %else436
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 96
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s3, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	vsll.vi	v24, v24, 13
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 36
	vsetivli	zero, 16, e64, m8, ta, ma
	vand.vx	v16, v0, s8
	bgez	a0, .LBB0_363
# %bb.362:                              # %cond.store437
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1240(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 113
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 121
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 5
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1656(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_363:                              # %else438
	.loc	1 0 56                          # k135114294095936.py:0:56
	vsetvli	zero, s3, e32, m8, ta, ma
	vadd.vv	v8, v24, v8
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 35
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_365
# %bb.364:                              # %cond.store439
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1248(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_365:                              # %else440
	slli	a0, s4, 34
	vsll.vi	v16, v8, 6
	bgez	a0, .LBB0_367
# %bb.366:                              # %cond.store441
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1256(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 19
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_367:                              # %else442
	slli	a0, s4, 33
	csrr	a1, vlenb
	li	a2, 137
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_369
# %bb.368:                              # %cond.store443
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1264(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_369:                              # %else444
	slli	a0, s4, 32
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_371
# %bb.370:                              # %cond.store445
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1272(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1176(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_371:                              # %else446
	slli	a0, s4, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_372
	j	.LBB0_492
.LBB0_372:                              # %else448
	slli	a0, s4, 30
	bgez	a0, .LBB0_373
	j	.LBB0_493
.LBB0_373:                              # %else450
	slli	a0, s4, 29
	bgez	a0, .LBB0_374
	j	.LBB0_494
.LBB0_374:                              # %else452
	slli	a0, s4, 28
	bgez	a0, .LBB0_375
	j	.LBB0_495
.LBB0_375:                              # %else454
	slli	a0, s4, 27
	bgez	a0, .LBB0_376
	j	.LBB0_496
.LBB0_376:                              # %else456
	slli	a0, s4, 26
	bgez	a0, .LBB0_377
	j	.LBB0_497
.LBB0_377:                              # %else458
	slli	a0, s4, 25
	bgez	a0, .LBB0_379
.LBB0_378:                              # %cond.store459
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1328(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_379:                              # %else460
	slli	a0, s4, 24
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_381
# %bb.380:                              # %cond.store461
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1336(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 1
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_381:                              # %else462
	slli	a0, s4, 23
	bgez	a0, .LBB0_383
# %bb.382:                              # %cond.store463
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1344(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1921
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_383:                              # %else464
	slli	a0, s4, 22
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.wv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_385
# %bb.384:                              # %cond.store465
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1352(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1793
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_385:                              # %else466
	slli	a0, s4, 21
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e64, m8, ta, ma
	vsrl.vx	v8, v8, s9
	bgez	a0, .LBB0_387
# %bb.386:                              # %cond.store467
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1360(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1665
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 240(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_387:                              # %else468
	slli	a0, s4, 20
	vand.vx	v16, v8, s8
	bgez	a0, .LBB0_389
# %bb.388:                              # %cond.store469
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1368(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 1537
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_389:                              # %else470
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vadd.vv	v16, v16, v8
	.loc	1 21 56                         # k135114294095936.py:21:56
	slli	a0, s4, 19
	csrr	a1, vlenb
	li	a2, 137
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_391
# %bb.390:                              # %cond.store471
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1376(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 1409
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_391:                              # %else472
	slli	a0, s4, 18
	vsetivli	zero, 16, e64, m8, ta, ma
	vsll.vi	v16, v16, 6
	bgez	a0, .LBB0_393
# %bb.392:                              # %cond.store473
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1384(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 1281
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -664(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_393:                              # %else474
	slli	a0, s4, 17
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e32, m4, ta, ma
	vwadd.wv	v16, v16, v8
	bgez	a0, .LBB0_395
# %bb.394:                              # %cond.store475
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1392(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 1153
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -784(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_395:                              # %else476
	slli	a0, s4, 16
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	bgez	a0, .LBB0_397
# %bb.396:                              # %cond.store477
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1400(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1025
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 161
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -904(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_397:                              # %else478
	slli	a0, s4, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_398
	j	.LBB0_498
.LBB0_398:                              # %else480
	slli	a0, s4, 14
	bgez	a0, .LBB0_399
	j	.LBB0_499
.LBB0_399:                              # %else482
	slli	a0, s4, 13
	bgez	a0, .LBB0_400
	j	.LBB0_500
.LBB0_400:                              # %else484
	slli	a0, s4, 12
	bgez	a0, .LBB0_401
	j	.LBB0_501
.LBB0_401:                              # %else486
	slli	a0, s4, 11
	bgez	a0, .LBB0_402
	j	.LBB0_502
.LBB0_402:                              # %else488
	slli	a0, s4, 10
	bgez	a0, .LBB0_403
	j	.LBB0_503
.LBB0_403:                              # %else490
	slli	a0, s4, 9
	bgez	a0, .LBB0_404
	j	.LBB0_504
.LBB0_404:                              # %else492
	slli	a0, s4, 8
	bgez	a0, .LBB0_405
	j	.LBB0_505
.LBB0_405:                              # %else494
	slli	a0, s4, 7
	bgez	a0, .LBB0_406
	j	.LBB0_506
.LBB0_406:                              # %else496
	slli	a0, s4, 6
	bgez	a0, .LBB0_407
	j	.LBB0_507
.LBB0_407:                              # %else498
	slli	a0, s4, 5
	bgez	a0, .LBB0_408
	j	.LBB0_508
.LBB0_408:                              # %else500
	slli	a0, s4, 4
	bgez	a0, .LBB0_409
	j	.LBB0_509
.LBB0_409:                              # %else502
	slli	a0, s4, 3
	bgez	a0, .LBB0_410
	j	.LBB0_510
.LBB0_410:                              # %else504
	slli	a0, s4, 2
	bgez	a0, .LBB0_411
	j	.LBB0_511
.LBB0_411:                              # %else506
	slli	a0, s4, 1
	bgez	a0, .LBB0_412
	j	.LBB0_512
.LBB0_412:                              # %else508
	bgez	s4, .LBB0_414
.LBB0_413:                              # %cond.store509
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1528(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 1536
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_414:                              # %else510
	.loc	1 21 4 epilogue_begin           # k135114294095936.py:21:4
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
.LBB0_415:                              # %cond.store
	.cfi_restore_state
	.loc	1 0 4                           # k135114294095936.py:0:4
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s7, 2
	bnez	a0, .LBB0_416
	j	.LBB0_2
.LBB0_416:                              # %cond.store1
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s7, 4
	bnez	a0, .LBB0_417
	j	.LBB0_3
.LBB0_417:                              # %cond.store3
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s7, 8
	bnez	a0, .LBB0_418
	j	.LBB0_4
.LBB0_418:                              # %cond.store5
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s7, 16
	bnez	a0, .LBB0_419
	j	.LBB0_5
.LBB0_419:                              # %cond.store7
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s7, 32
	bnez	a0, .LBB0_420
	j	.LBB0_6
.LBB0_420:                              # %cond.store9
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	slli	a1, a0, 7
	add	a0, a1, a0
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 121
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s7, 64
	beqz	a0, .LBB0_513
	j	.LBB0_7
.LBB0_513:                              # %cond.store9
	j	.LBB0_8
.LBB0_421:                              # %cond.store31
	.loc	1 0 56                          # k135114294095936.py:0:56
	slli	s6, s6, 16
	fmv.w.x	fa0, s6
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 46
	bltz	a0, .LBB0_422
	j	.LBB0_28
.LBB0_422:                              # %cond.store33
	.loc	1 0 56                          # k135114294095936.py:0:56
	slli	s9, s9, 16
	fmv.w.x	fa0, s9
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 45
	bltz	a0, .LBB0_423
	j	.LBB0_29
.LBB0_423:                              # %cond.store35
	.loc	1 0 56                          # k135114294095936.py:0:56
	slli	s5, s5, 16
	fmv.w.x	fa0, s5
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 44
	li	s9, 56
	bltz	a0, .LBB0_424
	j	.LBB0_30
.LBB0_424:                              # %cond.store37
	.loc	1 0 56                          # k135114294095936.py:0:56
	slli	s8, s8, 16
	fmv.w.x	fa0, s8
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 43
	bltz	a0, .LBB0_425
	j	.LBB0_31
.LBB0_425:                              # %cond.store39
	.loc	1 0 56                          # k135114294095936.py:0:56
	slli	s10, s10, 16
	fmv.w.x	fa0, s10
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s7, 42
	li	s8, 128
	bgez	a0, .LBB0_514
	j	.LBB0_32
.LBB0_514:                              # %cond.store39
	j	.LBB0_33
.LBB0_426:                              # %cond.store63
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 30
	bltz	a0, .LBB0_427
	j	.LBB0_55
.LBB0_427:                              # %cond.store65
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 29
	bltz	a0, .LBB0_428
	j	.LBB0_56
.LBB0_428:                              # %cond.store67
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 28
	bltz	a0, .LBB0_429
	j	.LBB0_57
.LBB0_429:                              # %cond.store69
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 27
	bltz	a0, .LBB0_430
	j	.LBB0_58
.LBB0_430:                              # %cond.store71
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 160(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s7, 26
	bltz	a0, .LBB0_431
	j	.LBB0_59
.LBB0_431:                              # %cond.store73
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 168(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 6
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s11)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s7, 25
	lui	a1, 5
	addi	a1, a1, 208
	add	s4, sp, a1
	bgez	a0, .LBB0_515
	j	.LBB0_60
.LBB0_515:                              # %cond.store73
	j	.LBB0_61
.LBB0_432:                              # %cond.store95
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 256(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 14
	bltz	a0, .LBB0_433
	j	.LBB0_81
.LBB0_433:                              # %cond.store97
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 264(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 13
	bltz	a0, .LBB0_434
	j	.LBB0_82
.LBB0_434:                              # %cond.store99
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 272(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 12
	bltz	a0, .LBB0_435
	j	.LBB0_83
.LBB0_435:                              # %cond.store101
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 280(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s7, 11
	bltz	a0, .LBB0_436
	j	.LBB0_84
.LBB0_436:                              # %cond.store103
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 288(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s7, 10
	bgez	a0, .LBB0_516
	j	.LBB0_85
.LBB0_516:                              # %cond.store103
	j	.LBB0_86
.LBB0_437:                              # %cond.store127
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 2
	bnez	a0, .LBB0_438
	j	.LBB0_108
.LBB0_438:                              # %cond.store129
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 4
	bnez	a0, .LBB0_439
	j	.LBB0_109
.LBB0_439:                              # %cond.store131
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 8
	bnez	a0, .LBB0_440
	j	.LBB0_110
.LBB0_440:                              # %cond.store133
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 16
	bnez	a0, .LBB0_441
	j	.LBB0_111
.LBB0_441:                              # %cond.store135
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1320(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 32
	bnez	a0, .LBB0_442
	j	.LBB0_112
.LBB0_442:                              # %cond.store137
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 145
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 5
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a2, a1, 7
	add	a1, a2, a1
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1200(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 64
	beqz	a0, .LBB0_517
	j	.LBB0_113
.LBB0_517:                              # %cond.store137
	j	.LBB0_114
.LBB0_443:                              # %cond.store159
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 384(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 46
	bltz	a0, .LBB0_444
	j	.LBB0_134
.LBB0_444:                              # %cond.store161
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 392(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 45
	bltz	a0, .LBB0_445
	j	.LBB0_135
.LBB0_445:                              # %cond.store163
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 400(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 44
	bltz	a0, .LBB0_446
	j	.LBB0_136
.LBB0_446:                              # %cond.store165
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 408(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 43
	lui	a1, 4
	addi	a1, a1, -64
	add	s5, sp, a1
	bltz	a0, .LBB0_447
	j	.LBB0_137
.LBB0_447:                              # %cond.store167
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 416(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 42
	bgez	a0, .LBB0_518
	j	.LBB0_138
.LBB0_518:                              # %cond.store167
	j	.LBB0_139
.LBB0_448:                              # %cond.store191
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 512(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 30
	bltz	a0, .LBB0_449
	j	.LBB0_161
.LBB0_449:                              # %cond.store193
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 520(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 29
	bltz	a0, .LBB0_450
	j	.LBB0_162
.LBB0_450:                              # %cond.store195
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 528(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 28
	bltz	a0, .LBB0_451
	j	.LBB0_163
.LBB0_451:                              # %cond.store197
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 536(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 27
	bltz	a0, .LBB0_452
	j	.LBB0_164
.LBB0_452:                              # %cond.store199
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 544(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 26
	bltz	a0, .LBB0_453
	j	.LBB0_165
.LBB0_453:                              # %cond.store201
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 25
	bgez	a0, .LBB0_519
	j	.LBB0_166
.LBB0_519:                              # %cond.store201
	j	.LBB0_167
.LBB0_454:                              # %cond.store223
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 640(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 14
	bltz	a0, .LBB0_455
	j	.LBB0_187
.LBB0_455:                              # %cond.store225
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 648(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 13
	bltz	a0, .LBB0_456
	j	.LBB0_188
.LBB0_456:                              # %cond.store227
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 656(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 12
	bltz	a0, .LBB0_457
	j	.LBB0_189
.LBB0_457:                              # %cond.store229
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 664(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 11
	bltz	a0, .LBB0_458
	j	.LBB0_190
.LBB0_458:                              # %cond.store231
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 672(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 4
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 10
	bgez	a0, .LBB0_520
	j	.LBB0_191
.LBB0_520:                              # %cond.store231
	j	.LBB0_192
.LBB0_459:                              # %cond.store255
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 2
	bnez	a0, .LBB0_460
	j	.LBB0_214
.LBB0_460:                              # %cond.store257
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 4
	bnez	a0, .LBB0_461
	j	.LBB0_215
.LBB0_461:                              # %cond.store259
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 8
	bnez	a0, .LBB0_462
	j	.LBB0_216
.LBB0_462:                              # %cond.store261
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s5, 16
	bnez	a0, .LBB0_463
	j	.LBB0_217
.LBB0_463:                              # %cond.store263
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 32
	bnez	a0, .LBB0_464
	j	.LBB0_218
.LBB0_464:                              # %cond.store265
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 153
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 145
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s5, 64
	beqz	a0, .LBB0_521
	j	.LBB0_219
.LBB0_521:                              # %cond.store265
	j	.LBB0_220
.LBB0_465:                              # %cond.store287
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 768(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 46
	bltz	a0, .LBB0_466
	j	.LBB0_240
.LBB0_466:                              # %cond.store289
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 776(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 45
	bltz	a0, .LBB0_467
	j	.LBB0_241
.LBB0_467:                              # %cond.store291
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 784(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 44
	bltz	a0, .LBB0_468
	j	.LBB0_242
.LBB0_468:                              # %cond.store293
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 792(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 43
	bltz	a0, .LBB0_469
	j	.LBB0_243
.LBB0_469:                              # %cond.store295
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 800(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 42
	bgez	a0, .LBB0_522
	j	.LBB0_244
.LBB0_522:                              # %cond.store295
	j	.LBB0_245
.LBB0_470:                              # %cond.store319
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 896(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 30
	bltz	a0, .LBB0_471
	j	.LBB0_267
.LBB0_471:                              # %cond.store321
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 904(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 29
	bltz	a0, .LBB0_472
	j	.LBB0_268
.LBB0_472:                              # %cond.store323
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 912(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 28
	bltz	a0, .LBB0_473
	j	.LBB0_269
.LBB0_473:                              # %cond.store325
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 920(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 27
	bltz	a0, .LBB0_474
	j	.LBB0_270
.LBB0_474:                              # %cond.store327
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 928(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 26
	bltz	a0, .LBB0_475
	j	.LBB0_271
.LBB0_475:                              # %cond.store329
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 936(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 25
	bgez	a0, .LBB0_523
	j	.LBB0_272
.LBB0_523:                              # %cond.store329
	j	.LBB0_273
.LBB0_476:                              # %cond.store351
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1024(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 14
	bltz	a0, .LBB0_477
	j	.LBB0_293
.LBB0_477:                              # %cond.store353
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1032(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 13
	bltz	a0, .LBB0_478
	j	.LBB0_294
.LBB0_478:                              # %cond.store355
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1040(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 12
	bltz	a0, .LBB0_479
	j	.LBB0_295
.LBB0_479:                              # %cond.store357
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1048(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s5, 11
	bltz	a0, .LBB0_480
	j	.LBB0_296
.LBB0_480:                              # %cond.store359
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1056(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1320(s6)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s5, 10
	bgez	a0, .LBB0_524
	j	.LBB0_297
.LBB0_524:                              # %cond.store359
	j	.LBB0_298
.LBB0_481:                              # %cond.store383
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 2
	bnez	a0, .LBB0_482
	j	.LBB0_320
.LBB0_482:                              # %cond.store385
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 4
	bnez	a0, .LBB0_483
	j	.LBB0_321
.LBB0_483:                              # %cond.store387
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 8
	bnez	a0, .LBB0_484
	j	.LBB0_322
.LBB0_484:                              # %cond.store389
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s4, 16
	lui	a1, 1
	addi	a1, a1, 1592
	add	s5, sp, a1
	bnez	a0, .LBB0_485
	j	.LBB0_323
.LBB0_485:                              # %cond.store391
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 32
	bnez	a0, .LBB0_486
	j	.LBB0_324
.LBB0_486:                              # %cond.store393
	.loc	1 0 56                          # k135114294095936.py:0:56
	csrr	a0, vlenb
	li	a1, 137
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 7
	addi	a1, a1, -2016
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 29
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s4, 64
	beqz	a0, .LBB0_525
	j	.LBB0_325
.LBB0_525:                              # %cond.store393
	j	.LBB0_326
.LBB0_487:                              # %cond.store415
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1152(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 46
	bltz	a0, .LBB0_488
	j	.LBB0_346
.LBB0_488:                              # %cond.store417
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1160(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 45
	bltz	a0, .LBB0_489
	j	.LBB0_347
.LBB0_489:                              # %cond.store419
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1168(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 44
	bltz	a0, .LBB0_490
	j	.LBB0_348
.LBB0_490:                              # %cond.store421
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1176(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 43
	bltz	a0, .LBB0_491
	j	.LBB0_349
.LBB0_491:                              # %cond.store423
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1184(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 42
	bgez	a0, .LBB0_526
	j	.LBB0_350
.LBB0_526:                              # %cond.store423
	j	.LBB0_351
.LBB0_492:                              # %cond.store447
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1280(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 30
	bltz	a0, .LBB0_493
	j	.LBB0_373
.LBB0_493:                              # %cond.store449
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1288(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 29
	bltz	a0, .LBB0_494
	j	.LBB0_374
.LBB0_494:                              # %cond.store451
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1296(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 28
	bltz	a0, .LBB0_495
	j	.LBB0_375
.LBB0_495:                              # %cond.store453
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1304(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 27
	bltz	a0, .LBB0_496
	j	.LBB0_376
.LBB0_496:                              # %cond.store455
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1312(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 26
	bltz	a0, .LBB0_497
	j	.LBB0_377
.LBB0_497:                              # %cond.store457
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1320(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 153
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s5)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 25
	bgez	a0, .LBB0_527
	j	.LBB0_378
.LBB0_527:                              # %cond.store457
	j	.LBB0_379
.LBB0_498:                              # %cond.store479
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1408(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 14
	bltz	a0, .LBB0_499
	j	.LBB0_399
.LBB0_499:                              # %cond.store481
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1416(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 13
	bltz	a0, .LBB0_500
	j	.LBB0_400
.LBB0_500:                              # %cond.store483
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1424(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 12
	bltz	a0, .LBB0_501
	j	.LBB0_401
.LBB0_501:                              # %cond.store485
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1432(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s4, 11
	bltz	a0, .LBB0_502
	j	.LBB0_402
.LBB0_502:                              # %cond.store487
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1440(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 897
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1120(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 10
	bltz	a0, .LBB0_503
	j	.LBB0_403
.LBB0_503:                              # %cond.store489
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1448(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 769
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1240(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 9
	bltz	a0, .LBB0_504
	j	.LBB0_404
.LBB0_504:                              # %cond.store491
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1456(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 641
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1360(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 8
	bltz	a0, .LBB0_505
	j	.LBB0_405
.LBB0_505:                              # %cond.store493
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1464(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 513
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1480(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 7
	bltz	a0, .LBB0_506
	j	.LBB0_406
.LBB0_506:                              # %cond.store495
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1472(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 385
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1600(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 6
	bltz	a0, .LBB0_507
	j	.LBB0_407
.LBB0_507:                              # %cond.store497
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1480(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 257
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1720(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 5
	bltz	a0, .LBB0_508
	j	.LBB0_408
.LBB0_508:                              # %cond.store499
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1488(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 129
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1840(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 4
	bltz	a0, .LBB0_509
	j	.LBB0_409
.LBB0_509:                              # %cond.store501
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1496(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1960(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 3
	bltz	a0, .LBB0_510
	j	.LBB0_410
.LBB0_510:                              # %cond.store503
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1504(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 1920
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 2
	bltz	a0, .LBB0_511
	j	.LBB0_411
.LBB0_511:                              # %cond.store505
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1512(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 1792
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s4, 1
	bltz	a0, .LBB0_512
	j	.LBB0_412
.LBB0_512:                              # %cond.store507
	.loc	1 0 56                          # k135114294095936.py:0:56
	ld	a0, 1520(sp)                    # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 21 56                         # k135114294095936.py:21:56
	call	__truncsfbf2
	addi	a0, sp, 1664
	csrr	a1, vlenb
	li	a2, 161
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 7
	addi	a2, a2, -2016
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	bgez	s4, .LBB0_528
	j	.LBB0_413
.LBB0_528:                              # %cond.store507
	j	.LBB0_414
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused_add_arange_index_copy_transpose_view_zeros_7, .Lfunc_end0-triton_poi_fused_add_arange_index_copy_transpose_view_zeros_7
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
	.asciz	"k135114294095936.py"           # string offset=7 ; k135114294095936.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
