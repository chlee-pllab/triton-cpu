	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused_add_arange_index_copy_transpose_view_6 # -- Begin function triton_poi_fused_add_arange_index_copy_transpose_view_6
	.p2align	2
	.type	triton_poi_fused_add_arange_index_copy_transpose_view_6,@function
triton_poi_fused_add_arange_index_copy_transpose_view_6: # @triton_poi_fused_add_arange_index_copy_transpose_view_6
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114449654864.py"
	.loc	1 2 0                           # k135114449654864.py:2:0
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
	lui	a3, 3
	addi	a3, a3, -880
	sub	sp, sp, a3
	csrr	a3, vlenb
	li	a5, 80
	mul	a3, a3, a5
	sub	sp, sp, a3
	andi	sp, sp, -128
	mv	s2, a2
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114449654864.py:4:33
	slliw	a3, a4, 7
	li	s7, 32
	li	a4, 128
	li	a2, -64
	.loc	1 5 23                          # k135114449654864.py:5:23
	vsetvli	zero, s7, e32, m8, ta, ma
	vmv.v.x	v8, a3
	vid.v	v24
	vor.vx	v0, v24, a3
	.loc	1 9 19                          # k135114449654864.py:9:19
	vsra.vi	v8, v8, 31
	vsrl.vi	v8, v8, 26
	csrr	a5, vlenb
	li	a6, 48
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 1056
	add	a5, a5, a6
	vs8r.v	v8, (a5)                        # vscale x 64-byte Folded Spill
	vadd.vv	v8, v0, v8
	csrr	a5, vlenb
	li	a6, 56
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 1056
	add	a5, a5, a6
	vs8r.v	v8, (a5)                        # vscale x 64-byte Folded Spill
	.loc	1 8 19                          # k135114449654864.py:8:19
	vand.vx	v8, v8, a2
	vsub.vv	v8, v0, v8
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, sp, a2
	lui	a5, 3
	addi	a5, a5, 1056
	add	a2, a2, a5
	vs8r.v	v8, (a2)                        # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114449654864.py:6:21
	vmslt.vx	v8, v0, a4
	csrr	a2, vlenb
	li	a5, 72
	mul	a2, a2, a5
	add	a2, sp, a2
	lui	a5, 3
	addi	a5, a5, 1056
	add	a2, a2, a5
	vs1r.v	v8, (a2)                        # vscale x 8-byte Folded Spill
	li	a2, 64
	li	a5, 96
	.loc	1 5 23                          # k135114449654864.py:5:23
	vadd.vx	v0, v24, a2
	vor.vx	v8, v0, a3
	vadd.vx	v0, v24, a5
	vor.vx	v0, v0, a3
	csrr	a5, vlenb
	li	a6, 40
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 1056
	add	a5, a5, a6
	vs8r.v	v0, (a5)                        # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114449654864.py:6:21
	vmslt.vx	v16, v0, a4
	csrr	a5, vlenb
	slli	a5, a5, 6
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 1056
	add	a5, a5, a6
	vs1r.v	v16, (a5)                       # vscale x 8-byte Folded Spill
	csrr	a5, vlenb
	li	a6, 24
	mul	a5, a5, a6
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 1056
	add	a5, a5, a6
	vs8r.v	v8, (a5)                        # vscale x 64-byte Folded Spill
	vmslt.vx	v6, v8, a4
	csrr	a5, vlenb
	slli	a5, a5, 6
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 1056
	add	a5, a5, a6
	vl1r.v	v8, (a5)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v6, v8, 4
	.loc	1 5 23                          # k135114449654864.py:5:23
	vsetvli	zero, s7, e32, m8, ta, ma
	vadd.vx	v24, v24, s7
	vor.vx	v24, v24, a3
	csrr	a5, vlenb
	slli	a5, a5, 4
	add	a5, sp, a5
	lui	a6, 3
	addi	a6, a6, 1056
	add	a5, a5, a6
	vs8r.v	v24, (a5)                       # vscale x 64-byte Folded Spill
	.loc	1 6 21                          # k135114449654864.py:6:21
	vmslt.vx	v10, v24, a4
	csrr	a4, vlenb
	li	a5, 72
	mul	a4, a4, a5
	add	a4, sp, a4
	lui	a5, 3
	addi	a5, a5, 1056
	add	a4, a4, a5
	vl1r.v	v7, (a4)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v7, v10, 4
	.loc	1 12 30                         # k135114449654864.py:12:30
	slli	a3, a3, 1
	add	a1, a1, a3
	.loc	1 12 35 is_stmt 0               # k135114449654864.py:12:35
	vsetvli	zero, a2, e16, m8, ta, mu
	vmv.v.i	v24, 0
	addi	a3, a1, 128
	vmv.v.i	v8, 0
	vmv1r.v	v0, v6
	vle16.v	v8, (a3), v0.t
	csrr	a3, vlenb
	li	a4, 72
	mul	a3, a3, a4
	add	a3, sp, a3
	lui	a4, 3
	addi	a4, a4, 1056
	add	a3, a3, a4
	vs8r.v	v8, (a3)                        # vscale x 64-byte Folded Spill
	vmv1r.v	v0, v7
	vle16.v	v24, (a1), v0.t
	.loc	1 6 21 is_stmt 1                # k135114449654864.py:6:21
	vsetivli	zero, 16, e8, m1, ta, ma
	vslideup.vi	v7, v6, 8
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a3, 3
	addi	a3, a3, 1056
	add	a1, a1, a3
	vs1r.v	v7, (a1)                        # vscale x 8-byte Folded Spill
	lui	a1, 3
	addi	a1, a1, -984
	add	s4, sp, a1
	.loc	1 10 19                         # k135114449654864.py:10:19
	lwu	a1, 0(a0)
	lw	a0, 4(a0)
	lui	a3, 3
	addi	a3, a3, 768
	add	a3, sp, a3
	lui	a4, 3
	addi	a4, a4, 896
	add	a4, sp, a4
	lui	a6, 1048574
	slli	a5, a0, 32
	.loc	1 18 32                         # k135114449654864.py:18:32
	srliw	a0, a0, 31
	.loc	1 10 19                         # k135114449654864.py:10:19
	or	a1, a5, a1
	.loc	1 18 32                         # k135114449654864.py:18:32
	slli	a0, a0, 7
	csrr	a5, vlenb
	li	a7, 56
	mul	a5, a5, a7
	add	a5, sp, a5
	lui	a7, 3
	addi	a7, a7, 1056
	add	a5, a5, a7
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	.loc	1 20 45                         # k135114449654864.py:20:45
	vsetvli	zero, s7, e32, m8, ta, ma
	vsll.vi	v8, v8, 7
	.loc	1 18 32                         # k135114449654864.py:18:32
	add	a0, a0, a1
	.loc	1 20 45                         # k135114449654864.py:20:45
	vand.vx	v16, v8, a6
	.loc	1 20 33 is_stmt 0               # k135114449654864.py:20:33
	slli	s9, a0, 6
	lui	a0, 3
	addi	a0, a0, 1056
	add	a0, sp, a0
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v0, (a0)                        # vscale x 64-byte Folded Reload
	.loc	1 20 30                         # k135114449654864.py:20:30
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v8, v16, v0
	.loc	1 20 40                         # k135114449654864.py:20:40
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v8, s9
	.loc	1 20 56                         # k135114449654864.py:20:56
	vadd.vv	v8, v8, v8
	vmv2r.v	v16, v24
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 12 35 is_stmt 1               # k135114449654864.py:12:35
	vsetvli	zero, a2, e16, m8, ta, ma
	vse16.v	v24, (a3)
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	vse16.v	v16, (a4)
	lh	a0, 1872(s4)
	sd	a0, 352(sp)                     # 8-byte Folded Spill
	lh	a0, 1874(s4)
	sd	a0, 360(sp)                     # 8-byte Folded Spill
	lh	a0, 1876(s4)
	sd	a0, 368(sp)                     # 8-byte Folded Spill
	lh	a0, 1878(s4)
	sd	a0, 376(sp)                     # 8-byte Folded Spill
	lh	a0, 1864(s4)
	sd	a0, 320(sp)                     # 8-byte Folded Spill
	lh	a0, 1866(s4)
	sd	a0, 328(sp)                     # 8-byte Folded Spill
	lh	a0, 1868(s4)
	sd	a0, 336(sp)                     # 8-byte Folded Spill
	lh	a0, 1870(s4)
	sd	a0, 344(sp)                     # 8-byte Folded Spill
	lh	a0, 1856(s4)
	sd	a0, 288(sp)                     # 8-byte Folded Spill
	lh	a0, 1858(s4)
	sd	a0, 296(sp)                     # 8-byte Folded Spill
	lh	a0, 1860(s4)
	sd	a0, 304(sp)                     # 8-byte Folded Spill
	lh	a0, 1862(s4)
	sd	a0, 312(sp)                     # 8-byte Folded Spill
	lh	a0, 1848(s4)
	sd	a0, 256(sp)                     # 8-byte Folded Spill
	lh	a0, 1850(s4)
	sd	a0, 264(sp)                     # 8-byte Folded Spill
	lh	a0, 1852(s4)
	sd	a0, 272(sp)                     # 8-byte Folded Spill
	lh	a0, 1854(s4)
	sd	a0, 280(sp)                     # 8-byte Folded Spill
	lh	a0, 1840(s4)
	sd	a0, 224(sp)                     # 8-byte Folded Spill
	lh	a0, 1842(s4)
	sd	a0, 232(sp)                     # 8-byte Folded Spill
	lh	a0, 1844(s4)
	sd	a0, 240(sp)                     # 8-byte Folded Spill
	lh	a0, 1846(s4)
	sd	a0, 248(sp)                     # 8-byte Folded Spill
	lh	a0, 2000(s4)
	sd	a0, 736(sp)                     # 8-byte Folded Spill
	lh	a0, 2002(s4)
	sd	a0, 744(sp)                     # 8-byte Folded Spill
	lh	a0, 2004(s4)
	sd	a0, 752(sp)                     # 8-byte Folded Spill
	lh	a0, 2006(s4)
	sd	a0, 760(sp)                     # 8-byte Folded Spill
	lh	a0, 1992(s4)
	sd	a0, 704(sp)                     # 8-byte Folded Spill
	lh	a0, 1994(s4)
	sd	a0, 712(sp)                     # 8-byte Folded Spill
	lh	a0, 1996(s4)
	sd	a0, 720(sp)                     # 8-byte Folded Spill
	lh	a0, 1998(s4)
	sd	a0, 728(sp)                     # 8-byte Folded Spill
	lh	a0, 1984(s4)
	sd	a0, 672(sp)                     # 8-byte Folded Spill
	lh	a0, 1986(s4)
	sd	a0, 680(sp)                     # 8-byte Folded Spill
	lh	a0, 1988(s4)
	sd	a0, 688(sp)                     # 8-byte Folded Spill
	lh	a0, 1990(s4)
	sd	a0, 696(sp)                     # 8-byte Folded Spill
	lh	a0, 1976(s4)
	sd	a0, 640(sp)                     # 8-byte Folded Spill
	lh	a0, 1978(s4)
	sd	a0, 648(sp)                     # 8-byte Folded Spill
	lh	a0, 1980(s4)
	sd	a0, 656(sp)                     # 8-byte Folded Spill
	lh	a0, 1982(s4)
	sd	a0, 664(sp)                     # 8-byte Folded Spill
	lh	a0, 1968(s4)
	sd	a0, 608(sp)                     # 8-byte Folded Spill
	lh	a0, 1970(s4)
	sd	a0, 616(sp)                     # 8-byte Folded Spill
	lh	a0, 1972(s4)
	sd	a0, 624(sp)                     # 8-byte Folded Spill
	lh	a0, 1974(s4)
	sd	a0, 632(sp)                     # 8-byte Folded Spill
	lh	a0, 1832(s4)
	sd	a0, 192(sp)                     # 8-byte Folded Spill
	lh	a0, 1834(s4)
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	lh	a0, 1836(s4)
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	lh	a0, 1838(s4)
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	lh	a0, 1824(s4)
	sd	a0, 160(sp)                     # 8-byte Folded Spill
	lh	a0, 1826(s4)
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	lh	a0, 1828(s4)
	sd	a0, 176(sp)                     # 8-byte Folded Spill
	lh	a0, 1830(s4)
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	lh	a0, 1816(s4)
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	lh	a0, 1818(s4)
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	lh	a0, 1820(s4)
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	lh	a0, 1822(s4)
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	lh	a0, 1808(s4)
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	lh	a0, 1810(s4)
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	lh	a0, 1812(s4)
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	lh	a0, 1814(s4)
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	lh	a0, 1960(s4)
	sd	a0, 576(sp)                     # 8-byte Folded Spill
	lh	a0, 1962(s4)
	sd	a0, 584(sp)                     # 8-byte Folded Spill
	lh	a0, 1964(s4)
	sd	a0, 592(sp)                     # 8-byte Folded Spill
	lh	a0, 1966(s4)
	sd	a0, 600(sp)                     # 8-byte Folded Spill
	lh	a0, 1952(s4)
	sd	a0, 544(sp)                     # 8-byte Folded Spill
	lh	a0, 1954(s4)
	sd	a0, 552(sp)                     # 8-byte Folded Spill
	lh	a0, 1956(s4)
	sd	a0, 560(sp)                     # 8-byte Folded Spill
	lh	a0, 1958(s4)
	sd	a0, 568(sp)                     # 8-byte Folded Spill
	lh	a0, 1944(s4)
	sd	a0, 512(sp)                     # 8-byte Folded Spill
	lh	a0, 1946(s4)
	sd	a0, 520(sp)                     # 8-byte Folded Spill
	lh	a0, 1948(s4)
	sd	a0, 528(sp)                     # 8-byte Folded Spill
	lh	a0, 1950(s4)
	sd	a0, 536(sp)                     # 8-byte Folded Spill
	lh	a0, 1936(s4)
	sd	a0, 480(sp)                     # 8-byte Folded Spill
	lh	a0, 1938(s4)
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	lh	a0, 1940(s4)
	sd	a0, 496(sp)                     # 8-byte Folded Spill
	lh	a0, 1942(s4)
	sd	a0, 504(sp)                     # 8-byte Folded Spill
	lh	a0, 1800(s4)
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	lh	a0, 1802(s4)
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	lh	a0, 1804(s4)
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	lh	a0, 1806(s4)
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	lh	s8, 1792(s4)
	lh	a0, 1794(s4)
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	lh	a0, 1796(s4)
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	lh	a0, 1798(s4)
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	lh	s5, 1784(s4)
	lh	s11, 1786(s4)
	lh	s3, 1788(s4)
	lh	s10, 1790(s4)
	lh	a0, 1928(s4)
	sd	a0, 448(sp)                     # 8-byte Folded Spill
	lh	a0, 1930(s4)
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	lh	a0, 1932(s4)
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	lh	a0, 1934(s4)
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	lh	a0, 1920(s4)
	sd	a0, 416(sp)                     # 8-byte Folded Spill
	lh	a0, 1922(s4)
	sd	a0, 424(sp)                     # 8-byte Folded Spill
	lh	a0, 1924(s4)
	sd	a0, 432(sp)                     # 8-byte Folded Spill
	lh	a0, 1926(s4)
	sd	a0, 440(sp)                     # 8-byte Folded Spill
	lh	a0, 1912(s4)
	sd	a0, 384(sp)                     # 8-byte Folded Spill
	lh	a0, 1914(s4)
	sd	a0, 392(sp)                     # 8-byte Folded Spill
	lh	a0, 1916(s4)
	sd	a0, 400(sp)                     # 8-byte Folded Spill
	lh	a0, 1918(s4)
	sd	a0, 408(sp)                     # 8-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	add	a0, a0, a1
	ld	s6, 1056(a0)                    # 8-byte Folded Reload
	andi	a0, s6, 1
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_1
	j	.LBB0_166
.LBB0_1:                                # %else
	andi	a0, s6, 2
	beqz	a0, .LBB0_2
	j	.LBB0_167
.LBB0_2:                                # %else2
	andi	a0, s6, 4
	beqz	a0, .LBB0_3
	j	.LBB0_168
.LBB0_3:                                # %else4
	andi	a0, s6, 8
	beqz	a0, .LBB0_4
	j	.LBB0_169
.LBB0_4:                                # %else6
	andi	a0, s6, 16
	beqz	a0, .LBB0_5
	j	.LBB0_170
.LBB0_5:                                # %else8
	andi	a0, s6, 32
	beqz	a0, .LBB0_6
	j	.LBB0_171
.LBB0_6:                                # %else10
	andi	a0, s6, 64
	beqz	a0, .LBB0_7
	j	.LBB0_172
.LBB0_7:                                # %else12
	andi	a0, s6, 128
	beqz	a0, .LBB0_8
	j	.LBB0_173
.LBB0_8:                                # %else14
	andi	a0, s6, 256
	beqz	a0, .LBB0_9
	j	.LBB0_174
.LBB0_9:                                # %else16
	andi	a0, s6, 512
	beqz	a0, .LBB0_10
	j	.LBB0_175
.LBB0_10:                               # %else18
	andi	a0, s6, 1024
	beqz	a0, .LBB0_12
.LBB0_11:                               # %cond.store19
	.loc	1 0 56 is_stmt 0                # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_12:                               # %else20
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s6, 52
	lui	a1, 3
	addi	a1, a1, 1056
	add	a1, sp, a1
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	.loc	1 0 0                           # k135114449654864.py:0
	vslidedown.vi	v8, v8, 16
	lui	a1, 3
	addi	a1, a1, 1056
	add	a1, sp, a1
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	bgez	a0, .LBB0_14
# %bb.13:                               # %cond.store21
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_14:                               # %else22
	slli	a0, s6, 51
	bgez	a0, .LBB0_16
# %bb.15:                               # %cond.store23
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 696(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_16:                               # %else24
	slli	a0, s6, 50
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	lui	a1, 3
	addi	a1, a1, 1056
	add	a1, sp, a1
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	.loc	1 20 0                          # k135114449654864.py:20
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	.loc	1 20 56                         # k135114449654864.py:20:56
	bgez	a0, .LBB0_17
	j	.LBB0_176
.LBB0_17:                               # %else26
	slli	a0, s6, 49
	.loc	1 20 0                          # k135114449654864.py:20
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	.loc	1 20 56                         # k135114449654864.py:20:56
	bgez	a0, .LBB0_18
	j	.LBB0_177
.LBB0_18:                               # %else28
	slli	a0, s6, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_20
.LBB0_19:                               # %cond.store29
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -768
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 336(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_20:                               # %else30
	slli	a0, s6, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_21
	j	.LBB0_178
.LBB0_21:                               # %else32
	slli	a0, s6, 46
	bgez	a0, .LBB0_22
	j	.LBB0_179
.LBB0_22:                               # %else34
	slli	a0, s6, 45
	li	s5, -64
	bgez	a0, .LBB0_23
	j	.LBB0_180
.LBB0_23:                               # %else36
	slli	a0, s6, 44
	bgez	a0, .LBB0_24
	j	.LBB0_181
.LBB0_24:                               # %else38
	slli	a0, s6, 43
	bgez	a0, .LBB0_25
	j	.LBB0_182
.LBB0_25:                               # %else40
	slli	a0, s6, 42
	bgez	a0, .LBB0_26
	j	.LBB0_183
.LBB0_26:                               # %else42
	slli	a0, s6, 41
	lui	a1, 2
	addi	a1, a1, 976
	add	s3, sp, a1
	lui	s8, 1048574
	bgez	a0, .LBB0_27
	j	.LBB0_184
.LBB0_27:                               # %else44
	slli	a0, s6, 40
	bgez	a0, .LBB0_28
	j	.LBB0_185
.LBB0_28:                               # %else46
	slli	a0, s6, 39
	bgez	a0, .LBB0_30
.LBB0_29:                               # %cond.store47
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_30:                               # %else48
	slli	a0, s6, 38
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s7, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_32
# %bb.31:                               # %cond.store49
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1656(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_32:                               # %else50
	.loc	1 0 56                          # k135114449654864.py:0:56
	vsetvli	zero, s7, e32, m8, ta, ma
	vand.vx	v16, v8, s5
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s6, 37
	vsll.vi	v24, v8, 7
	bgez	a0, .LBB0_34
# %bb.33:                               # %cond.store51
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 3
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_34:                               # %else52
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s7, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s6, 36
	vand.vx	v8, v24, s8
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_36
# %bb.35:                               # %cond.store53
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1792
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_36:                               # %else54
	slli	a0, s6, 35
	bgez	a0, .LBB0_38
# %bb.37:                               # %cond.store55
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_38:                               # %else56
	slli	a0, s6, 34
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	bgez	a0, .LBB0_39
	j	.LBB0_186
.LBB0_39:                               # %else58
	slli	a0, s6, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	bgez	a0, .LBB0_40
	j	.LBB0_187
.LBB0_40:                               # %else60
	slli	a0, s6, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_42
.LBB0_41:                               # %cond.store61
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1792
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 936(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_42:                               # %else62
	slli	a0, s6, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_43
	j	.LBB0_188
.LBB0_43:                               # %else64
	slli	a0, s6, 30
	bgez	a0, .LBB0_44
	j	.LBB0_189
.LBB0_44:                               # %else66
	slli	a0, s6, 29
	bgez	a0, .LBB0_45
	j	.LBB0_190
.LBB0_45:                               # %else68
	slli	a0, s6, 28
	bgez	a0, .LBB0_46
	j	.LBB0_191
.LBB0_46:                               # %else70
	slli	a0, s6, 27
	bgez	a0, .LBB0_47
	j	.LBB0_192
.LBB0_47:                               # %else72
	slli	a0, s6, 26
	bgez	a0, .LBB0_48
	j	.LBB0_193
.LBB0_48:                               # %else74
	slli	a0, s6, 25
	bgez	a0, .LBB0_49
	j	.LBB0_194
.LBB0_49:                               # %else76
	slli	a0, s6, 24
	bgez	a0, .LBB0_50
	j	.LBB0_195
.LBB0_50:                               # %else78
	slli	a0, s6, 23
	bgez	a0, .LBB0_51
	j	.LBB0_196
.LBB0_51:                               # %else80
	slli	a0, s6, 22
	bgez	a0, .LBB0_52
	j	.LBB0_197
.LBB0_52:                               # %else82
	slli	a0, s6, 21
	bgez	a0, .LBB0_54
.LBB0_53:                               # %cond.store83
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_54:                               # %else84
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s6, 20
	lui	a1, 2
	addi	a1, a1, -1160
	add	s4, sp, a1
	bgez	a0, .LBB0_56
# %bb.55:                               # %cond.store85
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 216(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 768
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_56:                               # %else86
	slli	a0, s6, 19
	bgez	a0, .LBB0_58
# %bb.57:                               # %cond.store87
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_58:                               # %else88
	slli	a0, s6, 18
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	bgez	a0, .LBB0_59
	j	.LBB0_198
.LBB0_59:                               # %else90
	slli	a0, s6, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	bgez	a0, .LBB0_60
	j	.LBB0_199
.LBB0_60:                               # %else92
	slli	a0, s6, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_62
.LBB0_61:                               # %cond.store93
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 248(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_62:                               # %else94
	slli	a0, s6, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_63
	j	.LBB0_200
.LBB0_63:                               # %else96
	slli	a0, s6, 14
	bgez	a0, .LBB0_64
	j	.LBB0_201
.LBB0_64:                               # %else98
	slli	a0, s6, 13
	bgez	a0, .LBB0_65
	j	.LBB0_202
.LBB0_65:                               # %else100
	slli	a0, s6, 12
	bgez	a0, .LBB0_66
	j	.LBB0_203
.LBB0_66:                               # %else102
	slli	a0, s6, 11
	bgez	a0, .LBB0_67
	j	.LBB0_204
.LBB0_67:                               # %else104
	slli	a0, s6, 10
	bgez	a0, .LBB0_68
	j	.LBB0_205
.LBB0_68:                               # %else106
	slli	a0, s6, 9
	bgez	a0, .LBB0_69
	j	.LBB0_206
.LBB0_69:                               # %else108
	slli	a0, s6, 8
	bgez	a0, .LBB0_70
	j	.LBB0_207
.LBB0_70:                               # %else110
	slli	a0, s6, 7
	bgez	a0, .LBB0_72
.LBB0_71:                               # %cond.store111
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 320(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -384
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_72:                               # %else112
	slli	a0, s6, 6
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s7, e32, m8, ta, ma
	vadd.vv	v8, v16, v8
	bgez	a0, .LBB0_74
# %bb.73:                               # %cond.store113
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 328(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 15
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 720(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_74:                               # %else114
	.loc	1 0 56                          # k135114449654864.py:0:56
	vsetvli	zero, s7, e32, m8, ta, ma
	vand.vx	v16, v8, s5
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s6, 5
	vsll.vi	v24, v8, 7
	bgez	a0, .LBB0_76
# %bb.75:                               # %cond.store115
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 336(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_76:                               # %else116
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, s7, e32, m8, ta, ma
	vsub.vv	v8, v8, v16
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s6, 4
	vand.vx	v8, v24, s8
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_78
# %bb.77:                               # %cond.store117
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 344(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 29
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_78:                               # %else118
	slli	a0, s6, 3
	bgez	a0, .LBB0_80
# %bb.79:                               # %cond.store119
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 352(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_80:                               # %else120
	slli	a0, s6, 2
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	bgez	a0, .LBB0_82
# %bb.81:                               # %cond.store121
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 360(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 7
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_82:                               # %else122
	.loc	1 0 56                          # k135114449654864.py:0:56
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v16, v24, s9
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s6, 1
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl1r.v	v8, (a1)                        # vscale x 8-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v24, v8, 1
	bgez	a0, .LBB0_84
# %bb.83:                               # %cond.store123
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 368(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs1r.v	v24, (a0)                       # vscale x 8-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl1r.v	v24, (a0)                       # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 2
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_84:                               # %else124
	.loc	1 0 56                          # k135114449654864.py:0:56
	vsetivli	zero, 16, e64, m8, ta, ma
	vadd.vv	v8, v16, v16
	.loc	1 20 56                         # k135114449654864.py:20:56
	vmv.x.s	s3, v24
	bgez	s6, .LBB0_86
# %bb.85:                               # %cond.store125
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 376(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 27
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 5
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_86:                               # %else126
	andi	a0, s3, 1
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	beqz	a0, .LBB0_87
	j	.LBB0_208
.LBB0_87:                               # %else128
	andi	a0, s3, 2
	beqz	a0, .LBB0_88
	j	.LBB0_209
.LBB0_88:                               # %else130
	andi	a0, s3, 4
	beqz	a0, .LBB0_89
	j	.LBB0_210
.LBB0_89:                               # %else132
	andi	a0, s3, 8
	beqz	a0, .LBB0_90
	j	.LBB0_211
.LBB0_90:                               # %else134
	andi	a0, s3, 16
	lui	a1, 1
	addi	a1, a1, 824
	add	s4, sp, a1
	beqz	a0, .LBB0_91
	j	.LBB0_212
.LBB0_91:                               # %else136
	andi	a0, s3, 32
	beqz	a0, .LBB0_92
	j	.LBB0_213
.LBB0_92:                               # %else138
	andi	a0, s3, 64
	beqz	a0, .LBB0_93
	j	.LBB0_214
.LBB0_93:                               # %else140
	andi	a0, s3, 128
	beqz	a0, .LBB0_94
	j	.LBB0_215
.LBB0_94:                               # %else142
	andi	a0, s3, 256
	beqz	a0, .LBB0_95
	j	.LBB0_216
.LBB0_95:                               # %else144
	andi	a0, s3, 512
	beqz	a0, .LBB0_96
	j	.LBB0_217
.LBB0_96:                               # %else146
	andi	a0, s3, 1024
	beqz	a0, .LBB0_98
.LBB0_97:                               # %cond.store147
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 10
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_98:                               # %else148
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s3, 52
	csrr	a1, vlenb
	li	a2, 24
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_100
# %bb.99:                               # %cond.store149
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 11
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 23
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_100:                              # %else150
	slli	a0, s3, 51
	bgez	a0, .LBB0_102
# %bb.101:                              # %cond.store151
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 12
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 936(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_102:                              # %else152
	slli	a0, s3, 50
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	bgez	a0, .LBB0_103
	j	.LBB0_218
.LBB0_103:                              # %else154
	slli	a0, s3, 49
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	bgez	a0, .LBB0_104
	j	.LBB0_219
.LBB0_104:                              # %else156
	slli	a0, s3, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_106
.LBB0_105:                              # %cond.store157
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 15
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 21
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 576(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_106:                              # %else158
	slli	a0, s3, 47
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_107
	j	.LBB0_220
.LBB0_107:                              # %else160
	slli	a0, s3, 46
	bgez	a0, .LBB0_108
	j	.LBB0_221
.LBB0_108:                              # %else162
	slli	a0, s3, 45
	bgez	a0, .LBB0_109
	j	.LBB0_222
.LBB0_109:                              # %else164
	slli	a0, s3, 44
	bgez	a0, .LBB0_110
	j	.LBB0_223
.LBB0_110:                              # %else166
	slli	a0, s3, 43
	bgez	a0, .LBB0_111
	j	.LBB0_224
.LBB0_111:                              # %else168
	slli	a0, s3, 42
	bgez	a0, .LBB0_112
	j	.LBB0_225
.LBB0_112:                              # %else170
	slli	a0, s3, 41
	bgez	a0, .LBB0_113
	j	.LBB0_226
.LBB0_113:                              # %else172
	slli	a0, s3, 40
	bgez	a0, .LBB0_114
	j	.LBB0_227
.LBB0_114:                              # %else174
	slli	a0, s3, 39
	addi	s4, sp, 2047
	addi	s4, s4, 737
	bgez	a0, .LBB0_116
.LBB0_115:                              # %cond.store175
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 448(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_116:                              # %else176
	slli	a0, s3, 38
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 48
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetvli	zero, s7, e32, m8, ta, ma
	vadd.vv	v8, v8, v16
	bgez	a0, .LBB0_118
# %bb.117:                              # %cond.store177
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 456(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1896(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_118:                              # %else178
	.loc	1 0 56                          # k135114449654864.py:0:56
	vsetvli	zero, s7, e32, m8, ta, ma
	vand.vx	v16, v8, s5
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s3, 37
	vsll.vi	v8, v8, 7
	bgez	a0, .LBB0_120
# %bb.119:                              # %cond.store179
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 464(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v16, (a0)                       # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v16, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1776(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_120:                              # %else180
	.loc	1 0 56                          # k135114449654864.py:0:56
	vsetvli	zero, s7, e32, m8, ta, ma
	vand.vx	v8, v8, s8
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s3, 36
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsub.vv	v8, v8, v16
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_122
# %bb.121:                              # %cond.store181
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 472(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 17
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_122:                              # %else182
	slli	a0, s3, 35
	bgez	a0, .LBB0_124
# %bb.123:                              # %cond.store183
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 480(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_124:                              # %else184
	slli	a0, s3, 34
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v8, v16
	bgez	a0, .LBB0_125
	j	.LBB0_228
.LBB0_125:                              # %else186
	slli	a0, s3, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	bgez	a0, .LBB0_126
	j	.LBB0_229
.LBB0_126:                              # %else188
	slli	a0, s3, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_128
.LBB0_127:                              # %cond.store189
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 504(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1793
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_128:                              # %else190
	slli	a0, s3, 31
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_129
	j	.LBB0_230
.LBB0_129:                              # %else192
	slli	a0, s3, 30
	bgez	a0, .LBB0_130
	j	.LBB0_231
.LBB0_130:                              # %else194
	slli	a0, s3, 29
	bgez	a0, .LBB0_131
	j	.LBB0_232
.LBB0_131:                              # %else196
	slli	a0, s3, 28
	bgez	a0, .LBB0_132
	j	.LBB0_233
.LBB0_132:                              # %else198
	slli	a0, s3, 27
	bgez	a0, .LBB0_133
	j	.LBB0_234
.LBB0_133:                              # %else200
	slli	a0, s3, 26
	bgez	a0, .LBB0_134
	j	.LBB0_235
.LBB0_134:                              # %else202
	slli	a0, s3, 25
	bgez	a0, .LBB0_135
	j	.LBB0_236
.LBB0_135:                              # %else204
	slli	a0, s3, 24
	bgez	a0, .LBB0_136
	j	.LBB0_237
.LBB0_136:                              # %else206
	slli	a0, s3, 23
	bgez	a0, .LBB0_137
	j	.LBB0_238
.LBB0_137:                              # %else208
	slli	a0, s3, 22
	bgez	a0, .LBB0_138
	j	.LBB0_239
.LBB0_138:                              # %else210
	slli	a0, s3, 21
	bgez	a0, .LBB0_140
.LBB0_139:                              # %cond.store211
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 897
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_140:                              # %else212
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v8, 16
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	slli	a0, s3, 20
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vslidedown.vi	v8, v8, 16
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_142
# %bb.141:                              # %cond.store213
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 769
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_142:                              # %else214
	slli	a0, s3, 19
	bgez	a0, .LBB0_144
# %bb.143:                              # %cond.store215
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 641
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_144:                              # %else216
	slli	a0, s3, 18
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v16, (a1)                       # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e32, m4, ta, ma
	vwadd.vv	v24, v16, v8
	bgez	a0, .LBB0_145
	j	.LBB0_240
.LBB0_145:                              # %else218
	slli	a0, s3, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	bgez	a0, .LBB0_146
	j	.LBB0_241
.LBB0_146:                              # %else220
	slli	a0, s3, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_148
.LBB0_147:                              # %cond.store221
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 632(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 257
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1672(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_148:                              # %else222
	slli	a0, s3, 15
	vadd.vx	v8, v8, s2
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vs8r.v	v8, (a1)                        # vscale x 64-byte Folded Spill
	bgez	a0, .LBB0_149
	j	.LBB0_242
.LBB0_149:                              # %else224
	slli	a0, s3, 14
	bgez	a0, .LBB0_150
	j	.LBB0_243
.LBB0_150:                              # %else226
	slli	a0, s3, 13
	bgez	a0, .LBB0_151
	j	.LBB0_244
.LBB0_151:                              # %else228
	slli	a0, s3, 12
	bgez	a0, .LBB0_152
	j	.LBB0_245
.LBB0_152:                              # %else230
	slli	a0, s3, 11
	bgez	a0, .LBB0_153
	j	.LBB0_246
.LBB0_153:                              # %else232
	slli	a0, s3, 10
	bgez	a0, .LBB0_154
	j	.LBB0_247
.LBB0_154:                              # %else234
	slli	a0, s3, 9
	bgez	a0, .LBB0_155
	j	.LBB0_248
.LBB0_155:                              # %else236
	slli	a0, s3, 8
	bgez	a0, .LBB0_156
	j	.LBB0_249
.LBB0_156:                              # %else238
	slli	a0, s3, 7
	bgez	a0, .LBB0_157
	j	.LBB0_250
.LBB0_157:                              # %else240
	slli	a0, s3, 6
	bgez	a0, .LBB0_158
	j	.LBB0_251
.LBB0_158:                              # %else242
	slli	a0, s3, 5
	bgez	a0, .LBB0_159
	j	.LBB0_252
.LBB0_159:                              # %else244
	slli	a0, s3, 4
	bgez	a0, .LBB0_160
	j	.LBB0_253
.LBB0_160:                              # %else246
	slli	a0, s3, 3
	bgez	a0, .LBB0_161
	j	.LBB0_254
.LBB0_161:                              # %else248
	slli	a0, s3, 2
	bgez	a0, .LBB0_162
	j	.LBB0_255
.LBB0_162:                              # %else250
	slli	a0, s3, 1
	bgez	a0, .LBB0_163
	j	.LBB0_256
.LBB0_163:                              # %else252
	bgez	s3, .LBB0_165
.LBB0_164:                              # %cond.store253
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 760(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 768
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 888(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
.LBB0_165:                              # %else254
	.loc	1 20 4 epilogue_begin           # k135114449654864.py:20:4
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
.LBB0_166:                              # %cond.store
	.cfi_restore_state
	.loc	1 0 4                           # k135114449654864.py:0:4
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 2
	bnez	a0, .LBB0_167
	j	.LBB0_2
.LBB0_167:                              # %cond.store1
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 4
	bnez	a0, .LBB0_168
	j	.LBB0_3
.LBB0_168:                              # %cond.store3
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 8
	bnez	a0, .LBB0_169
	j	.LBB0_4
.LBB0_169:                              # %cond.store5
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s6, 16
	bnez	a0, .LBB0_170
	j	.LBB0_5
.LBB0_170:                              # %cond.store7
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 32
	bnez	a0, .LBB0_171
	j	.LBB0_6
.LBB0_171:                              # %cond.store9
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 64
	bnez	a0, .LBB0_172
	j	.LBB0_7
.LBB0_172:                              # %cond.store11
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 128
	bnez	a0, .LBB0_173
	j	.LBB0_8
.LBB0_173:                              # %cond.store13
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 256
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 256
	bnez	a0, .LBB0_174
	j	.LBB0_9
.LBB0_174:                              # %cond.store15
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 512
	bnez	a0, .LBB0_175
	j	.LBB0_10
.LBB0_175:                              # %cond.store17
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1056(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s6, 1024
	beqz	a0, .LBB0_257
	j	.LBB0_11
.LBB0_257:                              # %cond.store17
	j	.LBB0_12
.LBB0_176:                              # %cond.store25
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 23
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 576(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 49
	.loc	1 20 0                          # k135114449654864.py:20
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	.loc	1 20 56                         # k135114449654864.py:20:56
	bltz	a0, .LBB0_177
	j	.LBB0_18
.LBB0_177:                              # %cond.store27
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -640
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 56
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 456(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_258
	j	.LBB0_19
.LBB0_258:                              # %cond.store27
	j	.LBB0_20
.LBB0_178:                              # %cond.store31
	.loc	1 0 0                           # k135114449654864.py:0
	slli	s5, s5, 16
	fmv.w.x	fa0, s5
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 46
	bltz	a0, .LBB0_179
	j	.LBB0_22
.LBB0_179:                              # %cond.store33
	.loc	1 0 0                           # k135114449654864.py:0
	slli	s11, s11, 16
	fmv.w.x	fa0, s11
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 45
	li	s5, -64
	bltz	a0, .LBB0_180
	j	.LBB0_23
.LBB0_180:                              # %cond.store35
	.loc	1 0 0                           # k135114449654864.py:0
	slli	s3, s3, 16
	fmv.w.x	fa0, s3
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 44
	bltz	a0, .LBB0_181
	j	.LBB0_24
.LBB0_181:                              # %cond.store37
	.loc	1 0 0                           # k135114449654864.py:0
	slli	s10, s10, 16
	fmv.w.x	fa0, s10
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 43
	bltz	a0, .LBB0_182
	j	.LBB0_25
.LBB0_182:                              # %cond.store39
	.loc	1 0 0                           # k135114449654864.py:0
	slli	s8, s8, 16
	fmv.w.x	fa0, s8
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -896
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 42
	bltz	a0, .LBB0_183
	j	.LBB0_26
.LBB0_183:                              # %cond.store41
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 11
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 41
	lui	a1, 2
	addi	a1, a1, 976
	add	s3, sp, a1
	lui	s8, 1048574
	bltz	a0, .LBB0_184
	j	.LBB0_27
.LBB0_184:                              # %cond.store43
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1152
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 2016(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 40
	bltz	a0, .LBB0_185
	j	.LBB0_28
.LBB0_185:                              # %cond.store45
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 3
	addi	a0, a0, -1280
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 39
	bgez	a0, .LBB0_259
	j	.LBB0_29
.LBB0_259:                              # %cond.store45
	j	.LBB0_30
.LBB0_186:                              # %cond.store57
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 5
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1176(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	bltz	a0, .LBB0_187
	j	.LBB0_40
.LBB0_187:                              # %cond.store59
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1920
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1056(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_260
	j	.LBB0_41
.LBB0_260:                              # %cond.store59
	j	.LBB0_42
.LBB0_188:                              # %cond.store63
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 30
	bltz	a0, .LBB0_189
	j	.LBB0_44
.LBB0_189:                              # %cond.store65
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 29
	bltz	a0, .LBB0_190
	j	.LBB0_45
.LBB0_190:                              # %cond.store67
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 28
	bltz	a0, .LBB0_191
	j	.LBB0_46
.LBB0_191:                              # %cond.store69
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 27
	bltz	a0, .LBB0_192
	j	.LBB0_47
.LBB0_192:                              # %cond.store71
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 160(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1664
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 26
	bltz	a0, .LBB0_193
	j	.LBB0_48
.LBB0_193:                              # %cond.store73
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 168(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 25
	bltz	a0, .LBB0_194
	j	.LBB0_49
.LBB0_194:                              # %cond.store75
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 24
	bltz	a0, .LBB0_195
	j	.LBB0_50
.LBB0_195:                              # %cond.store77
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1280
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 23
	bltz	a0, .LBB0_196
	j	.LBB0_51
.LBB0_196:                              # %cond.store79
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 22
	bltz	a0, .LBB0_197
	j	.LBB0_52
.LBB0_197:                              # %cond.store81
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 9
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s3)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 21
	bgez	a0, .LBB0_261
	j	.LBB0_53
.LBB0_261:                              # %cond.store81
	j	.LBB0_54
.LBB0_198:                              # %cond.store89
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 232(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 17
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	bltz	a0, .LBB0_199
	j	.LBB0_60
.LBB0_199:                              # %cond.store91
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 240(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 384
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_262
	j	.LBB0_61
.LBB0_262:                              # %cond.store91
	j	.LBB0_62
.LBB0_200:                              # %cond.store95
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 256(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 14
	bltz	a0, .LBB0_201
	j	.LBB0_64
.LBB0_201:                              # %cond.store97
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 264(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 13
	bltz	a0, .LBB0_202
	j	.LBB0_65
.LBB0_202:                              # %cond.store99
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 272(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 12
	bltz	a0, .LBB0_203
	j	.LBB0_66
.LBB0_203:                              # %cond.store101
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 280(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s6, 11
	bltz	a0, .LBB0_204
	j	.LBB0_67
.LBB0_204:                              # %cond.store103
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 288(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, 128
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1320(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 10
	bltz	a0, .LBB0_205
	j	.LBB0_68
.LBB0_205:                              # %cond.store105
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 296(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1200(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 9
	bltz	a0, .LBB0_206
	j	.LBB0_69
.LBB0_206:                              # %cond.store107
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 304(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -128
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1080(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 8
	bltz	a0, .LBB0_207
	j	.LBB0_70
.LBB0_207:                              # %cond.store109
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 312(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 31
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s6, 7
	bgez	a0, .LBB0_263
	j	.LBB0_71
.LBB0_263:                              # %cond.store109
	j	.LBB0_72
.LBB0_208:                              # %cond.store127
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 2
	bnez	a0, .LBB0_209
	j	.LBB0_88
.LBB0_209:                              # %cond.store129
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 4
	bnez	a0, .LBB0_210
	j	.LBB0_89
.LBB0_210:                              # %cond.store131
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 8
	bnez	a0, .LBB0_211
	j	.LBB0_90
.LBB0_211:                              # %cond.store133
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	andi	a0, s3, 16
	lui	a1, 1
	addi	a1, a1, 824
	add	s4, sp, a1
	bnez	a0, .LBB0_212
	j	.LBB0_91
.LBB0_212:                              # %cond.store135
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 4
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1408
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1896(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 32
	bnez	a0, .LBB0_213
	j	.LBB0_92
.LBB0_213:                              # %cond.store137
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 5
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 13
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1776(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 64
	bnez	a0, .LBB0_214
	j	.LBB0_93
.LBB0_214:                              # %cond.store139
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 6
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1664
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1656(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 128
	bnez	a0, .LBB0_215
	j	.LBB0_94
.LBB0_215:                              # %cond.store141
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m1, ta, ma
	vslidedown.vi	v8, v8, 7
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 25
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1536(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 256
	bnez	a0, .LBB0_216
	j	.LBB0_95
.LBB0_216:                              # %cond.store143
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 8
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 2
	addi	a0, a0, -1920
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 512
	bnez	a0, .LBB0_217
	j	.LBB0_96
.LBB0_217:                              # %cond.store145
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 9
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 3
	slli	a0, a0, 11
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	andi	a0, s3, 1024
	beqz	a0, .LBB0_264
	j	.LBB0_97
.LBB0_264:                              # %cond.store145
	j	.LBB0_98
.LBB0_218:                              # %cond.store153
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 13
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	li	a0, 11
	slli	a0, a0, 9
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 816(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 49
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	bltz	a0, .LBB0_219
	j	.LBB0_104
.LBB0_219:                              # %cond.store155
	.loc	1 0 56                          # k135114449654864.py:0:56
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 72
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, zero, e16, m2, ta, ma
	vslidedown.vi	v8, v8, 14
	vmv.x.s	a0, v8
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1408
	add	a0, sp, a0
	csrr	a1, vlenb
	slli	a1, a1, 6
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 56
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 696(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 48
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_265
	j	.LBB0_105
.LBB0_265:                              # %cond.store155
	j	.LBB0_106
.LBB0_220:                              # %cond.store159
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 384(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 46
	bltz	a0, .LBB0_221
	j	.LBB0_108
.LBB0_221:                              # %cond.store161
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 392(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 45
	bltz	a0, .LBB0_222
	j	.LBB0_109
.LBB0_222:                              # %cond.store163
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 400(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 44
	bltz	a0, .LBB0_223
	j	.LBB0_110
.LBB0_223:                              # %cond.store165
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 408(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 43
	bltz	a0, .LBB0_224
	j	.LBB0_111
.LBB0_224:                              # %cond.store167
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 416(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 1152
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 42
	bltz	a0, .LBB0_225
	j	.LBB0_112
.LBB0_225:                              # %cond.store169
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 424(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 5
	slli	a0, a0, 10
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 240(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 41
	bltz	a0, .LBB0_226
	j	.LBB0_113
.LBB0_226:                              # %cond.store171
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 432(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	lui	a0, 1
	addi	a0, a0, 896
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 120(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 40
	bltz	a0, .LBB0_227
	j	.LBB0_114
.LBB0_227:                              # %cond.store173
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 440(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	li	a0, 19
	slli	a0, a0, 8
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 0(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 39
	addi	s4, sp, 2047
	addi	s4, s4, 737
	bgez	a0, .LBB0_266
	j	.LBB0_115
.LBB0_266:                              # %cond.store173
	j	.LBB0_116
.LBB0_228:                              # %cond.store185
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	lui	a0, 1
	add	a0, sp, a0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1416(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 33
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	bltz	a0, .LBB0_229
	j	.LBB0_126
.LBB0_229:                              # %cond.store187
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 496(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1921
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	ld	a0, 1296(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 32
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_267
	j	.LBB0_127
.LBB0_267:                              # %cond.store187
	j	.LBB0_128
.LBB0_230:                              # %cond.store191
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 512(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 30
	bltz	a0, .LBB0_231
	j	.LBB0_130
.LBB0_231:                              # %cond.store193
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 520(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 29
	bltz	a0, .LBB0_232
	j	.LBB0_131
.LBB0_232:                              # %cond.store195
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 528(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 28
	bltz	a0, .LBB0_233
	j	.LBB0_132
.LBB0_233:                              # %cond.store197
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 536(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 27
	bltz	a0, .LBB0_234
	j	.LBB0_133
.LBB0_234:                              # %cond.store199
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 544(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1665
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 960(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 26
	bltz	a0, .LBB0_235
	j	.LBB0_134
.LBB0_235:                              # %cond.store201
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1537
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 840(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 25
	bltz	a0, .LBB0_236
	j	.LBB0_135
.LBB0_236:                              # %cond.store203
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1409
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 720(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 24
	bltz	a0, .LBB0_237
	j	.LBB0_136
.LBB0_237:                              # %cond.store205
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1281
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 600(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 23
	bltz	a0, .LBB0_238
	j	.LBB0_137
.LBB0_238:                              # %cond.store207
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1153
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 480(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 22
	bltz	a0, .LBB0_239
	j	.LBB0_138
.LBB0_239:                              # %cond.store209
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1025
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 360(s4)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 21
	bgez	a0, .LBB0_268
	j	.LBB0_139
.LBB0_268:                              # %cond.store209
	j	.LBB0_140
.LBB0_240:                              # %cond.store217
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v24, (a0)                       # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v24, (a0)                       # vscale x 64-byte Folded Reload
	addi	a0, sp, 2047
	addi	a0, a0, 513
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1432(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 17
	vsetvli	zero, zero, e64, m8, ta, ma
	vadd.vx	v8, v24, s9
	bltz	a0, .LBB0_241
	j	.LBB0_146
.LBB0_241:                              # %cond.store219
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vs8r.v	v8, (a0)                        # vscale x 64-byte Folded Spill
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 385
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 6
	add	a0, sp, a0
	lui	a1, 3
	addi	a1, a1, 1056
	add	a0, a0, a1
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1552(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 16
	vadd.vv	v8, v8, v8
	bgez	a0, .LBB0_269
	j	.LBB0_147
.LBB0_269:                              # %cond.store219
	j	.LBB0_148
.LBB0_242:                              # %cond.store223
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 640(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 14
	bltz	a0, .LBB0_243
	j	.LBB0_150
.LBB0_243:                              # %cond.store225
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 648(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m1, ta, ma
	vslidedown.vi	v8, v8, 1
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 13
	bltz	a0, .LBB0_244
	j	.LBB0_151
.LBB0_244:                              # %cond.store227
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 656(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 2
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 12
	bltz	a0, .LBB0_245
	j	.LBB0_152
.LBB0_245:                              # %cond.store229
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 664(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	fmv.x.w	a0, fa0
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 1, e64, m2, ta, ma
	vslidedown.vi	v8, v8, 3
	vmv.x.s	a1, v8
	sh	a0, 0(a1)
	slli	a0, s3, 11
	bltz	a0, .LBB0_246
	j	.LBB0_153
.LBB0_246:                              # %cond.store231
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 672(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 129
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1888(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 10
	bltz	a0, .LBB0_247
	j	.LBB0_154
.LBB0_247:                              # %cond.store233
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 680(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 2047
	addi	a0, a0, 1
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -2008(a0)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 9
	bltz	a0, .LBB0_248
	j	.LBB0_155
.LBB0_248:                              # %cond.store235
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 688(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 1920
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1968(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 8
	bltz	a0, .LBB0_249
	j	.LBB0_156
.LBB0_249:                              # %cond.store237
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 696(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 1792
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1848(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 7
	bltz	a0, .LBB0_250
	j	.LBB0_157
.LBB0_250:                              # %cond.store239
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 704(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 1664
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1728(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 6
	bltz	a0, .LBB0_251
	j	.LBB0_158
.LBB0_251:                              # %cond.store241
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 712(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 1536
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1608(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 5
	bltz	a0, .LBB0_252
	j	.LBB0_159
.LBB0_252:                              # %cond.store243
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 720(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 1408
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1488(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 4
	bltz	a0, .LBB0_253
	j	.LBB0_160
.LBB0_253:                              # %cond.store245
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 728(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 1280
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1368(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 3
	bltz	a0, .LBB0_254
	j	.LBB0_161
.LBB0_254:                              # %cond.store247
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 736(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 1152
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1248(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 2
	bltz	a0, .LBB0_255
	j	.LBB0_162
.LBB0_255:                              # %cond.store249
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 744(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 1024
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1128(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	slli	a0, s3, 1
	bltz	a0, .LBB0_256
	j	.LBB0_163
.LBB0_256:                              # %cond.store251
	.loc	1 0 56                          # k135114449654864.py:0:56
	ld	a0, 752(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 16
	fmv.w.x	fa0, a0
	.loc	1 20 56                         # k135114449654864.py:20:56
	call	__truncsfbf2
	addi	a0, sp, 896
	csrr	a1, vlenb
	li	a2, 72
	mul	a1, a1, a2
	add	a1, sp, a1
	lui	a2, 3
	addi	a2, a2, 1056
	add	a1, a1, a2
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	ld	a0, 1008(sp)
	fmv.x.w	a1, fa0
	sh	a1, 0(a0)
	bgez	s3, .LBB0_270
	j	.LBB0_164
.LBB0_270:                              # %cond.store251
	j	.LBB0_165
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused_add_arange_index_copy_transpose_view_6, .Lfunc_end0-triton_poi_fused_add_arange_index_copy_transpose_view_6
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
	.asciz	"k135114449654864.py"           # string offset=7 ; k135114449654864.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
