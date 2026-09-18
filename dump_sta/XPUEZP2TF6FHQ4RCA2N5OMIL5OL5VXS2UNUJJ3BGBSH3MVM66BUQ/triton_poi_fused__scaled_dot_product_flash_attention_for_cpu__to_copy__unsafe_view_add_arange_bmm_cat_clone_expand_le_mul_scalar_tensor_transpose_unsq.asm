	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8 # -- Begin function triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8
	.p2align	2
	.type	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8,@function
triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8: # @triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135112023920016.py"
	.loc	1 2 0                           # k135112023920016.py:2:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -2032
	.cfi_def_cfa_offset 2032
	sd	ra, 2024(sp)                    # 8-byte Folded Spill
	sd	s0, 2016(sp)                    # 8-byte Folded Spill
	sd	s1, 2008(sp)                    # 8-byte Folded Spill
	sd	s2, 2000(sp)                    # 8-byte Folded Spill
	sd	s3, 1992(sp)                    # 8-byte Folded Spill
	sd	s4, 1984(sp)                    # 8-byte Folded Spill
	sd	s5, 1976(sp)                    # 8-byte Folded Spill
	sd	s6, 1968(sp)                    # 8-byte Folded Spill
	sd	s7, 1960(sp)                    # 8-byte Folded Spill
	sd	s8, 1952(sp)                    # 8-byte Folded Spill
	sd	s9, 1944(sp)                    # 8-byte Folded Spill
	sd	s10, 1936(sp)                   # 8-byte Folded Spill
	sd	s11, 1928(sp)                   # 8-byte Folded Spill
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
	addi	sp, sp, -1968
	.cfi_def_cfa_offset 4000
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135112023920016.py:4:33
	slliw	a2, a3, 9
	lui	a3, 1048574
	lui	a4, 599186
	.loc	1 7 19                          # k135112023920016.py:7:19
	sraiw	a5, a2, 31
	addi	a4, a4, 1171
	.loc	1 11 25                         # k135112023920016.py:11:25
	slli	a6, a2, 1
	.loc	1 7 19                          # k135112023920016.py:7:19
	srliw	a5, a5, 19
	.loc	1 10 47                         # k135112023920016.py:10:47
	mul	a4, a2, a4
	.loc	1 11 25                         # k135112023920016.py:11:25
	add	a1, a1, a6
	.loc	1 7 19                          # k135112023920016.py:7:19
	add	a5, a2, a5
	.loc	1 10 47                         # k135112023920016.py:10:47
	srli	a4, a4, 32
	.loc	1 7 19                          # k135112023920016.py:7:19
	and	a3, a5, a3
	.loc	1 10 47                         # k135112023920016.py:10:47
	add	a4, a4, a2
	.loc	1 7 19                          # k135112023920016.py:7:19
	sub	a2, a2, a3
	.loc	1 10 47                         # k135112023920016.py:10:47
	srliw	a3, a4, 31
	sraiw	a4, a4, 15
	add	a3, a4, a3
	.loc	1 10 41 is_stmt 0               # k135112023920016.py:10:41
	slli	a3, a3, 13
	.loc	1 10 35                         # k135112023920016.py:10:35
	addw	a2, a3, a2
	.loc	1 10 30                         # k135112023920016.py:10:30
	slli	a2, a2, 1
	add	a0, a0, a2
	.loc	1 10 52                         # k135112023920016.py:10:52
	lh	a2, 0(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -208(a3)                    # 8-byte Folded Spill
	lh	a2, 2(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -216(a3)                    # 8-byte Folded Spill
	lh	a2, 4(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -224(a3)                    # 8-byte Folded Spill
	lh	a2, 6(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -232(a3)                    # 8-byte Folded Spill
	lh	a2, 8(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -240(a3)                    # 8-byte Folded Spill
	lh	a2, 10(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -248(a3)                    # 8-byte Folded Spill
	lh	a2, 12(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -256(a3)                    # 8-byte Folded Spill
	lh	a2, 14(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -264(a3)                    # 8-byte Folded Spill
	lh	a2, 16(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -272(a3)                    # 8-byte Folded Spill
	lh	a2, 18(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -280(a3)                    # 8-byte Folded Spill
	lh	a2, 20(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -288(a3)                    # 8-byte Folded Spill
	lh	a2, 22(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -296(a3)                    # 8-byte Folded Spill
	lh	a2, 24(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -304(a3)                    # 8-byte Folded Spill
	lh	a2, 26(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -312(a3)                    # 8-byte Folded Spill
	lh	a2, 28(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -320(a3)                    # 8-byte Folded Spill
	lh	a2, 30(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -328(a3)                    # 8-byte Folded Spill
	lh	a2, 32(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -336(a3)                    # 8-byte Folded Spill
	lh	a2, 34(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -344(a3)                    # 8-byte Folded Spill
	lh	a2, 36(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -352(a3)                    # 8-byte Folded Spill
	lh	a2, 38(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -360(a3)                    # 8-byte Folded Spill
	lh	a2, 40(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -368(a3)                    # 8-byte Folded Spill
	lh	a2, 42(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -376(a3)                    # 8-byte Folded Spill
	lh	a2, 44(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -384(a3)                    # 8-byte Folded Spill
	lh	a2, 46(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -392(a3)                    # 8-byte Folded Spill
	lh	a2, 48(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -400(a3)                    # 8-byte Folded Spill
	lh	a2, 50(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -408(a3)                    # 8-byte Folded Spill
	lh	a2, 52(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -416(a3)                    # 8-byte Folded Spill
	lh	a2, 54(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -424(a3)                    # 8-byte Folded Spill
	lh	a2, 56(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -432(a3)                    # 8-byte Folded Spill
	lh	a2, 58(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -440(a3)                    # 8-byte Folded Spill
	lh	a2, 60(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -448(a3)                    # 8-byte Folded Spill
	lh	a2, 62(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -456(a3)                    # 8-byte Folded Spill
	lh	a2, 64(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -464(a3)                    # 8-byte Folded Spill
	lh	a2, 66(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -472(a3)                    # 8-byte Folded Spill
	lh	a2, 68(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -480(a3)                    # 8-byte Folded Spill
	lh	a2, 70(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -488(a3)                    # 8-byte Folded Spill
	lh	a2, 72(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -496(a3)                    # 8-byte Folded Spill
	lh	a2, 74(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -504(a3)                    # 8-byte Folded Spill
	lh	a2, 76(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -512(a3)                    # 8-byte Folded Spill
	lh	a2, 78(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -520(a3)                    # 8-byte Folded Spill
	lh	a2, 80(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -528(a3)                    # 8-byte Folded Spill
	lh	a2, 82(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -536(a3)                    # 8-byte Folded Spill
	lh	a2, 84(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -544(a3)                    # 8-byte Folded Spill
	lh	a2, 86(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -552(a3)                    # 8-byte Folded Spill
	lh	a2, 88(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -560(a3)                    # 8-byte Folded Spill
	lh	a2, 90(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -568(a3)                    # 8-byte Folded Spill
	lh	a2, 92(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -576(a3)                    # 8-byte Folded Spill
	lh	a2, 94(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -584(a3)                    # 8-byte Folded Spill
	lh	a2, 96(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -592(a3)                    # 8-byte Folded Spill
	lh	a2, 98(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -600(a3)                    # 8-byte Folded Spill
	lh	a2, 100(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -608(a3)                    # 8-byte Folded Spill
	lh	a2, 102(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -616(a3)                    # 8-byte Folded Spill
	lh	a2, 104(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -624(a3)                    # 8-byte Folded Spill
	lh	a2, 106(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -632(a3)                    # 8-byte Folded Spill
	lh	a2, 108(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -640(a3)                    # 8-byte Folded Spill
	lh	a2, 110(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -648(a3)                    # 8-byte Folded Spill
	lh	a2, 112(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -656(a3)                    # 8-byte Folded Spill
	lh	a2, 114(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -664(a3)                    # 8-byte Folded Spill
	lh	a2, 116(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -672(a3)                    # 8-byte Folded Spill
	lh	a2, 118(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -680(a3)                    # 8-byte Folded Spill
	lh	a2, 120(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -688(a3)                    # 8-byte Folded Spill
	lh	a2, 122(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -696(a3)                    # 8-byte Folded Spill
	lh	a2, 124(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -704(a3)                    # 8-byte Folded Spill
	lh	a2, 126(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -712(a3)                    # 8-byte Folded Spill
	lh	a2, 128(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -720(a3)                    # 8-byte Folded Spill
	lh	a2, 130(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -728(a3)                    # 8-byte Folded Spill
	lh	a2, 132(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -736(a3)                    # 8-byte Folded Spill
	lh	a2, 134(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -744(a3)                    # 8-byte Folded Spill
	lh	a2, 136(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -752(a3)                    # 8-byte Folded Spill
	lh	a2, 138(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -760(a3)                    # 8-byte Folded Spill
	lh	a2, 140(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -768(a3)                    # 8-byte Folded Spill
	lh	a2, 142(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -776(a3)                    # 8-byte Folded Spill
	lh	a2, 144(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -784(a3)                    # 8-byte Folded Spill
	lh	a2, 146(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -792(a3)                    # 8-byte Folded Spill
	lh	a2, 148(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -800(a3)                    # 8-byte Folded Spill
	lh	a2, 150(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -808(a3)                    # 8-byte Folded Spill
	lh	a2, 152(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -816(a3)                    # 8-byte Folded Spill
	lh	a2, 154(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -824(a3)                    # 8-byte Folded Spill
	lh	a2, 156(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -832(a3)                    # 8-byte Folded Spill
	lh	a2, 158(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -840(a3)                    # 8-byte Folded Spill
	lh	a2, 160(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -848(a3)                    # 8-byte Folded Spill
	lh	a2, 162(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -856(a3)                    # 8-byte Folded Spill
	lh	a2, 164(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -864(a3)                    # 8-byte Folded Spill
	lh	a2, 166(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -872(a3)                    # 8-byte Folded Spill
	lh	a2, 168(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -880(a3)                    # 8-byte Folded Spill
	lh	a2, 170(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -888(a3)                    # 8-byte Folded Spill
	lh	a2, 172(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -896(a3)                    # 8-byte Folded Spill
	lh	a2, 174(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -904(a3)                    # 8-byte Folded Spill
	lh	a2, 176(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -912(a3)                    # 8-byte Folded Spill
	lh	a2, 178(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -920(a3)                    # 8-byte Folded Spill
	lh	a2, 180(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -928(a3)                    # 8-byte Folded Spill
	lh	a2, 182(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -936(a3)                    # 8-byte Folded Spill
	lh	a2, 184(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -944(a3)                    # 8-byte Folded Spill
	lh	a2, 186(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -952(a3)                    # 8-byte Folded Spill
	lh	a2, 188(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -960(a3)                    # 8-byte Folded Spill
	lh	a2, 190(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -968(a3)                    # 8-byte Folded Spill
	lh	a2, 192(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -976(a3)                    # 8-byte Folded Spill
	lh	a2, 194(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -984(a3)                    # 8-byte Folded Spill
	lh	a2, 196(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -992(a3)                    # 8-byte Folded Spill
	lh	a2, 198(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1000(a3)                   # 8-byte Folded Spill
	lh	a2, 200(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1008(a3)                   # 8-byte Folded Spill
	lh	a2, 202(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1016(a3)                   # 8-byte Folded Spill
	lh	a2, 204(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1024(a3)                   # 8-byte Folded Spill
	lh	a2, 206(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1032(a3)                   # 8-byte Folded Spill
	lh	a2, 208(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1040(a3)                   # 8-byte Folded Spill
	lh	a2, 210(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1048(a3)                   # 8-byte Folded Spill
	lh	a2, 212(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1056(a3)                   # 8-byte Folded Spill
	lh	a2, 214(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1064(a3)                   # 8-byte Folded Spill
	lh	a2, 216(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1072(a3)                   # 8-byte Folded Spill
	lh	a2, 218(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1080(a3)                   # 8-byte Folded Spill
	lh	a2, 220(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1088(a3)                   # 8-byte Folded Spill
	lh	a2, 222(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1096(a3)                   # 8-byte Folded Spill
	lh	a2, 224(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1104(a3)                   # 8-byte Folded Spill
	lh	a2, 226(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1112(a3)                   # 8-byte Folded Spill
	lh	a2, 228(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1120(a3)                   # 8-byte Folded Spill
	lh	a2, 230(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1128(a3)                   # 8-byte Folded Spill
	lh	a2, 232(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1136(a3)                   # 8-byte Folded Spill
	lh	a2, 234(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1144(a3)                   # 8-byte Folded Spill
	lh	a2, 236(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1152(a3)                   # 8-byte Folded Spill
	lh	a2, 238(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1160(a3)                   # 8-byte Folded Spill
	lh	a2, 240(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1168(a3)                   # 8-byte Folded Spill
	lh	a2, 242(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1176(a3)                   # 8-byte Folded Spill
	lh	a2, 244(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1184(a3)                   # 8-byte Folded Spill
	lh	a2, 246(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1192(a3)                   # 8-byte Folded Spill
	lh	a2, 248(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1200(a3)                   # 8-byte Folded Spill
	lh	a2, 250(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1208(a3)                   # 8-byte Folded Spill
	lh	a2, 252(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1216(a3)                   # 8-byte Folded Spill
	lh	a2, 254(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1224(a3)                   # 8-byte Folded Spill
	lh	a2, 256(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1232(a3)                   # 8-byte Folded Spill
	lh	a2, 258(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1240(a3)                   # 8-byte Folded Spill
	lh	a2, 260(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1248(a3)                   # 8-byte Folded Spill
	lh	a2, 262(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1256(a3)                   # 8-byte Folded Spill
	lh	a2, 264(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1264(a3)                   # 8-byte Folded Spill
	lh	a2, 266(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1272(a3)                   # 8-byte Folded Spill
	lh	a2, 268(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1280(a3)                   # 8-byte Folded Spill
	lh	a2, 270(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1288(a3)                   # 8-byte Folded Spill
	lh	a2, 272(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1296(a3)                   # 8-byte Folded Spill
	lh	a2, 274(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1304(a3)                   # 8-byte Folded Spill
	lh	a2, 276(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1312(a3)                   # 8-byte Folded Spill
	lh	a2, 278(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1320(a3)                   # 8-byte Folded Spill
	lh	a2, 280(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1328(a3)                   # 8-byte Folded Spill
	lh	a2, 282(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1336(a3)                   # 8-byte Folded Spill
	lh	a2, 284(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1344(a3)                   # 8-byte Folded Spill
	lh	a2, 286(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1352(a3)                   # 8-byte Folded Spill
	lh	a2, 288(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1360(a3)                   # 8-byte Folded Spill
	lh	a2, 290(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1368(a3)                   # 8-byte Folded Spill
	lh	a2, 292(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1376(a3)                   # 8-byte Folded Spill
	lh	a2, 294(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1384(a3)                   # 8-byte Folded Spill
	lh	a2, 296(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1392(a3)                   # 8-byte Folded Spill
	lh	a2, 298(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1400(a3)                   # 8-byte Folded Spill
	lh	a2, 300(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1408(a3)                   # 8-byte Folded Spill
	lh	a2, 302(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1416(a3)                   # 8-byte Folded Spill
	lh	a2, 304(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1424(a3)                   # 8-byte Folded Spill
	lh	a2, 306(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1432(a3)                   # 8-byte Folded Spill
	lh	a2, 308(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1440(a3)                   # 8-byte Folded Spill
	lh	a2, 310(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1448(a3)                   # 8-byte Folded Spill
	lh	a2, 312(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1456(a3)                   # 8-byte Folded Spill
	lh	a2, 314(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1464(a3)                   # 8-byte Folded Spill
	lh	a2, 316(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1472(a3)                   # 8-byte Folded Spill
	lh	a2, 318(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1480(a3)                   # 8-byte Folded Spill
	lh	a2, 320(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1488(a3)                   # 8-byte Folded Spill
	lh	a2, 322(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1496(a3)                   # 8-byte Folded Spill
	lh	a2, 324(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1504(a3)                   # 8-byte Folded Spill
	lh	a2, 326(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1512(a3)                   # 8-byte Folded Spill
	lh	a2, 328(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1520(a3)                   # 8-byte Folded Spill
	lh	a2, 330(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1528(a3)                   # 8-byte Folded Spill
	lh	a2, 332(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1536(a3)                   # 8-byte Folded Spill
	lh	a2, 334(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1544(a3)                   # 8-byte Folded Spill
	lh	a2, 336(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1552(a3)                   # 8-byte Folded Spill
	lh	a2, 338(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1560(a3)                   # 8-byte Folded Spill
	lh	a2, 340(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1568(a3)                   # 8-byte Folded Spill
	lh	a2, 342(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1576(a3)                   # 8-byte Folded Spill
	lh	a2, 344(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1584(a3)                   # 8-byte Folded Spill
	lh	a2, 346(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1592(a3)                   # 8-byte Folded Spill
	lh	a2, 348(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1600(a3)                   # 8-byte Folded Spill
	lh	a2, 350(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1608(a3)                   # 8-byte Folded Spill
	lh	a2, 352(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1616(a3)                   # 8-byte Folded Spill
	lh	a2, 354(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1624(a3)                   # 8-byte Folded Spill
	lh	a2, 356(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1632(a3)                   # 8-byte Folded Spill
	lh	a2, 358(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1640(a3)                   # 8-byte Folded Spill
	lh	a2, 360(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1648(a3)                   # 8-byte Folded Spill
	lh	a2, 362(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1656(a3)                   # 8-byte Folded Spill
	lh	a2, 364(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1664(a3)                   # 8-byte Folded Spill
	lh	a2, 366(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1672(a3)                   # 8-byte Folded Spill
	lh	a2, 368(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1680(a3)                   # 8-byte Folded Spill
	lh	a2, 370(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1688(a3)                   # 8-byte Folded Spill
	lh	a2, 372(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1696(a3)                   # 8-byte Folded Spill
	lh	a2, 374(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1704(a3)                   # 8-byte Folded Spill
	lh	a2, 376(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1712(a3)                   # 8-byte Folded Spill
	lh	a2, 378(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1720(a3)                   # 8-byte Folded Spill
	lh	a2, 380(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1728(a3)                   # 8-byte Folded Spill
	lh	a2, 382(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1736(a3)                   # 8-byte Folded Spill
	lh	a2, 384(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1744(a3)                   # 8-byte Folded Spill
	lh	a2, 386(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1752(a3)                   # 8-byte Folded Spill
	lh	a2, 388(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1760(a3)                   # 8-byte Folded Spill
	lh	a2, 390(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1768(a3)                   # 8-byte Folded Spill
	lh	a2, 392(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1776(a3)                   # 8-byte Folded Spill
	lh	a2, 394(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1784(a3)                   # 8-byte Folded Spill
	lh	a2, 396(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1792(a3)                   # 8-byte Folded Spill
	lh	a2, 398(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1800(a3)                   # 8-byte Folded Spill
	lh	a2, 400(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1808(a3)                   # 8-byte Folded Spill
	lh	a2, 402(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1816(a3)                   # 8-byte Folded Spill
	lh	a2, 404(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1824(a3)                   # 8-byte Folded Spill
	lh	a2, 406(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1832(a3)                   # 8-byte Folded Spill
	lh	a2, 408(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1840(a3)                   # 8-byte Folded Spill
	lh	a2, 410(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1848(a3)                   # 8-byte Folded Spill
	lh	a2, 412(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1856(a3)                   # 8-byte Folded Spill
	lh	a2, 414(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1864(a3)                   # 8-byte Folded Spill
	lh	a2, 416(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1872(a3)                   # 8-byte Folded Spill
	lh	a2, 418(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1880(a3)                   # 8-byte Folded Spill
	lh	a2, 420(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1888(a3)                   # 8-byte Folded Spill
	lh	a2, 422(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1896(a3)                   # 8-byte Folded Spill
	lh	a2, 424(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1904(a3)                   # 8-byte Folded Spill
	lh	a2, 426(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1912(a3)                   # 8-byte Folded Spill
	lh	a2, 428(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1920(a3)                   # 8-byte Folded Spill
	lh	a2, 430(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1928(a3)                   # 8-byte Folded Spill
	lh	a2, 432(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1936(a3)                   # 8-byte Folded Spill
	lh	a2, 434(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1944(a3)                   # 8-byte Folded Spill
	lh	a2, 436(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1952(a3)                   # 8-byte Folded Spill
	lh	a2, 438(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1960(a3)                   # 8-byte Folded Spill
	lh	a2, 440(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1968(a3)                   # 8-byte Folded Spill
	lh	a2, 442(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1976(a3)                   # 8-byte Folded Spill
	lh	a2, 444(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1984(a3)                   # 8-byte Folded Spill
	lh	a2, 446(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -1992(a3)                   # 8-byte Folded Spill
	lh	a2, 448(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -2000(a3)                   # 8-byte Folded Spill
	lh	a2, 450(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -2008(a3)                   # 8-byte Folded Spill
	lh	a2, 452(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -2016(a3)                   # 8-byte Folded Spill
	lh	a2, 454(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -2024(a3)                   # 8-byte Folded Spill
	lh	a2, 456(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -2032(a3)                   # 8-byte Folded Spill
	lh	a2, 458(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -2040(a3)                   # 8-byte Folded Spill
	lh	a2, 460(a0)
	lui	a3, 1
	add	a3, sp, a3
	sd	a2, -2048(a3)                   # 8-byte Folded Spill
	lh	a2, 462(a0)
	sd	a2, 2040(sp)                    # 8-byte Folded Spill
	lh	a2, 464(a0)
	sd	a2, 2032(sp)                    # 8-byte Folded Spill
	lh	a2, 466(a0)
	sd	a2, 2024(sp)                    # 8-byte Folded Spill
	lh	a2, 468(a0)
	sd	a2, 2016(sp)                    # 8-byte Folded Spill
	lh	a2, 470(a0)
	sd	a2, 2008(sp)                    # 8-byte Folded Spill
	lh	a2, 472(a0)
	sd	a2, 2000(sp)                    # 8-byte Folded Spill
	lh	a2, 474(a0)
	sd	a2, 1992(sp)                    # 8-byte Folded Spill
	lh	a2, 476(a0)
	sd	a2, 1984(sp)                    # 8-byte Folded Spill
	lh	a2, 478(a0)
	sd	a2, 1976(sp)                    # 8-byte Folded Spill
	lh	a2, 480(a0)
	sd	a2, 1968(sp)                    # 8-byte Folded Spill
	lh	a2, 482(a0)
	sd	a2, 1960(sp)                    # 8-byte Folded Spill
	lh	a2, 484(a0)
	sd	a2, 1952(sp)                    # 8-byte Folded Spill
	lh	a2, 486(a0)
	sd	a2, 1944(sp)                    # 8-byte Folded Spill
	lh	a2, 488(a0)
	sd	a2, 1936(sp)                    # 8-byte Folded Spill
	lh	a2, 490(a0)
	sd	a2, 1928(sp)                    # 8-byte Folded Spill
	lh	a2, 492(a0)
	sd	a2, 1920(sp)                    # 8-byte Folded Spill
	lh	a2, 494(a0)
	sd	a2, 1912(sp)                    # 8-byte Folded Spill
	lh	a2, 496(a0)
	sd	a2, 1904(sp)                    # 8-byte Folded Spill
	lh	a2, 498(a0)
	sd	a2, 1896(sp)                    # 8-byte Folded Spill
	lh	a2, 500(a0)
	sd	a2, 1888(sp)                    # 8-byte Folded Spill
	lh	a2, 502(a0)
	sd	a2, 1880(sp)                    # 8-byte Folded Spill
	lh	a2, 1016(a0)
	sd	a2, 848(sp)                     # 8-byte Folded Spill
	lh	a2, 1018(a0)
	sd	a2, 840(sp)                     # 8-byte Folded Spill
	lh	a2, 504(a0)
	sd	a2, 1872(sp)                    # 8-byte Folded Spill
	lh	a2, 506(a0)
	sd	a2, 1864(sp)                    # 8-byte Folded Spill
	lh	a2, 508(a0)
	sd	a2, 1856(sp)                    # 8-byte Folded Spill
	lh	a2, 510(a0)
	sd	a2, 1848(sp)                    # 8-byte Folded Spill
	lh	a2, 512(a0)
	sd	a2, 1840(sp)                    # 8-byte Folded Spill
	lh	a2, 514(a0)
	sd	a2, 1832(sp)                    # 8-byte Folded Spill
	lh	a2, 516(a0)
	sd	a2, 1824(sp)                    # 8-byte Folded Spill
	lh	a2, 518(a0)
	sd	a2, 1816(sp)                    # 8-byte Folded Spill
	lh	a2, 520(a0)
	sd	a2, 1808(sp)                    # 8-byte Folded Spill
	lh	a2, 522(a0)
	sd	a2, 1800(sp)                    # 8-byte Folded Spill
	lh	a2, 524(a0)
	sd	a2, 1792(sp)                    # 8-byte Folded Spill
	lh	a2, 526(a0)
	sd	a2, 1784(sp)                    # 8-byte Folded Spill
	lh	a2, 528(a0)
	sd	a2, 1776(sp)                    # 8-byte Folded Spill
	lh	a2, 530(a0)
	sd	a2, 1768(sp)                    # 8-byte Folded Spill
	lh	a2, 532(a0)
	sd	a2, 1760(sp)                    # 8-byte Folded Spill
	lh	a2, 534(a0)
	sd	a2, 1752(sp)                    # 8-byte Folded Spill
	lh	a2, 536(a0)
	sd	a2, 1744(sp)                    # 8-byte Folded Spill
	lh	a2, 538(a0)
	sd	a2, 1736(sp)                    # 8-byte Folded Spill
	lh	a2, 540(a0)
	sd	a2, 1728(sp)                    # 8-byte Folded Spill
	lh	a2, 542(a0)
	sd	a2, 1720(sp)                    # 8-byte Folded Spill
	lh	a2, 544(a0)
	sd	a2, 1712(sp)                    # 8-byte Folded Spill
	lh	a2, 546(a0)
	sd	a2, 1704(sp)                    # 8-byte Folded Spill
	lh	a2, 548(a0)
	sd	a2, 1696(sp)                    # 8-byte Folded Spill
	lh	a2, 550(a0)
	sd	a2, 1688(sp)                    # 8-byte Folded Spill
	lh	a2, 552(a0)
	sd	a2, 1680(sp)                    # 8-byte Folded Spill
	lh	a2, 554(a0)
	sd	a2, 1672(sp)                    # 8-byte Folded Spill
	lh	a2, 556(a0)
	sd	a2, 1664(sp)                    # 8-byte Folded Spill
	lh	a2, 558(a0)
	sd	a2, 1656(sp)                    # 8-byte Folded Spill
	lh	a2, 560(a0)
	sd	a2, 1648(sp)                    # 8-byte Folded Spill
	lh	a2, 562(a0)
	sd	a2, 1640(sp)                    # 8-byte Folded Spill
	lh	a2, 564(a0)
	sd	a2, 1632(sp)                    # 8-byte Folded Spill
	lh	a2, 566(a0)
	sd	a2, 1624(sp)                    # 8-byte Folded Spill
	lh	a2, 568(a0)
	sd	a2, 1616(sp)                    # 8-byte Folded Spill
	lh	a2, 570(a0)
	sd	a2, 1608(sp)                    # 8-byte Folded Spill
	lh	a2, 572(a0)
	sd	a2, 1600(sp)                    # 8-byte Folded Spill
	lh	a2, 574(a0)
	sd	a2, 1592(sp)                    # 8-byte Folded Spill
	lh	a2, 576(a0)
	sd	a2, 1584(sp)                    # 8-byte Folded Spill
	lh	a2, 578(a0)
	sd	a2, 1576(sp)                    # 8-byte Folded Spill
	lh	a2, 580(a0)
	sd	a2, 1568(sp)                    # 8-byte Folded Spill
	lh	a2, 582(a0)
	sd	a2, 1560(sp)                    # 8-byte Folded Spill
	lh	a2, 584(a0)
	sd	a2, 1552(sp)                    # 8-byte Folded Spill
	lh	a2, 586(a0)
	sd	a2, 1544(sp)                    # 8-byte Folded Spill
	lh	a2, 588(a0)
	sd	a2, 1536(sp)                    # 8-byte Folded Spill
	lh	a2, 590(a0)
	sd	a2, 1528(sp)                    # 8-byte Folded Spill
	lh	a2, 592(a0)
	sd	a2, 1520(sp)                    # 8-byte Folded Spill
	lh	a2, 594(a0)
	sd	a2, 1512(sp)                    # 8-byte Folded Spill
	lh	a2, 596(a0)
	sd	a2, 1504(sp)                    # 8-byte Folded Spill
	lh	a2, 598(a0)
	sd	a2, 1496(sp)                    # 8-byte Folded Spill
	lh	a2, 600(a0)
	sd	a2, 1488(sp)                    # 8-byte Folded Spill
	lh	a2, 602(a0)
	sd	a2, 1480(sp)                    # 8-byte Folded Spill
	lh	a2, 604(a0)
	sd	a2, 1472(sp)                    # 8-byte Folded Spill
	lh	a2, 606(a0)
	sd	a2, 1464(sp)                    # 8-byte Folded Spill
	lh	a2, 608(a0)
	sd	a2, 1456(sp)                    # 8-byte Folded Spill
	lh	a2, 610(a0)
	sd	a2, 1448(sp)                    # 8-byte Folded Spill
	lh	a2, 612(a0)
	sd	a2, 1440(sp)                    # 8-byte Folded Spill
	lh	a2, 614(a0)
	sd	a2, 1432(sp)                    # 8-byte Folded Spill
	lh	a2, 616(a0)
	sd	a2, 1424(sp)                    # 8-byte Folded Spill
	lh	a2, 618(a0)
	sd	a2, 1416(sp)                    # 8-byte Folded Spill
	lh	a2, 620(a0)
	sd	a2, 1408(sp)                    # 8-byte Folded Spill
	lh	a2, 622(a0)
	sd	a2, 1400(sp)                    # 8-byte Folded Spill
	lh	a2, 624(a0)
	sd	a2, 1392(sp)                    # 8-byte Folded Spill
	lh	a2, 626(a0)
	sd	a2, 1384(sp)                    # 8-byte Folded Spill
	lh	a2, 628(a0)
	sd	a2, 1376(sp)                    # 8-byte Folded Spill
	lh	a2, 630(a0)
	sd	a2, 1368(sp)                    # 8-byte Folded Spill
	lh	a2, 632(a0)
	sd	a2, 1360(sp)                    # 8-byte Folded Spill
	lh	a2, 634(a0)
	sd	a2, 1352(sp)                    # 8-byte Folded Spill
	lh	a2, 636(a0)
	sd	a2, 1344(sp)                    # 8-byte Folded Spill
	lh	a2, 638(a0)
	sd	a2, 1336(sp)                    # 8-byte Folded Spill
	lh	a2, 640(a0)
	sd	a2, 1328(sp)                    # 8-byte Folded Spill
	lh	a2, 642(a0)
	sd	a2, 1320(sp)                    # 8-byte Folded Spill
	lh	a2, 644(a0)
	sd	a2, 1312(sp)                    # 8-byte Folded Spill
	lh	a2, 646(a0)
	sd	a2, 1304(sp)                    # 8-byte Folded Spill
	lh	a2, 648(a0)
	sd	a2, 1296(sp)                    # 8-byte Folded Spill
	lh	a2, 650(a0)
	sd	a2, 1288(sp)                    # 8-byte Folded Spill
	lh	a2, 652(a0)
	sd	a2, 1280(sp)                    # 8-byte Folded Spill
	lh	a2, 654(a0)
	sd	a2, 1272(sp)                    # 8-byte Folded Spill
	lh	a2, 656(a0)
	sd	a2, 1264(sp)                    # 8-byte Folded Spill
	lh	a2, 658(a0)
	sd	a2, 1256(sp)                    # 8-byte Folded Spill
	lh	a2, 660(a0)
	sd	a2, 1248(sp)                    # 8-byte Folded Spill
	lh	a2, 662(a0)
	sd	a2, 1240(sp)                    # 8-byte Folded Spill
	lh	a2, 664(a0)
	sd	a2, 1232(sp)                    # 8-byte Folded Spill
	lh	a2, 666(a0)
	sd	a2, 1224(sp)                    # 8-byte Folded Spill
	lh	a2, 668(a0)
	sd	a2, 1216(sp)                    # 8-byte Folded Spill
	lh	a2, 670(a0)
	sd	a2, 1208(sp)                    # 8-byte Folded Spill
	lh	a2, 672(a0)
	sd	a2, 1200(sp)                    # 8-byte Folded Spill
	lh	a2, 674(a0)
	sd	a2, 1192(sp)                    # 8-byte Folded Spill
	lh	a2, 676(a0)
	sd	a2, 1184(sp)                    # 8-byte Folded Spill
	lh	a2, 678(a0)
	sd	a2, 1176(sp)                    # 8-byte Folded Spill
	lh	a2, 680(a0)
	sd	a2, 1168(sp)                    # 8-byte Folded Spill
	lh	a2, 682(a0)
	sd	a2, 1160(sp)                    # 8-byte Folded Spill
	lh	a2, 684(a0)
	sd	a2, 1152(sp)                    # 8-byte Folded Spill
	lh	a2, 686(a0)
	sd	a2, 1144(sp)                    # 8-byte Folded Spill
	lh	a2, 688(a0)
	sd	a2, 1136(sp)                    # 8-byte Folded Spill
	lh	a2, 690(a0)
	sd	a2, 1128(sp)                    # 8-byte Folded Spill
	lh	a2, 692(a0)
	sd	a2, 1120(sp)                    # 8-byte Folded Spill
	lh	a2, 694(a0)
	sd	a2, 1112(sp)                    # 8-byte Folded Spill
	lh	a2, 696(a0)
	sd	a2, 1104(sp)                    # 8-byte Folded Spill
	lh	a2, 698(a0)
	sd	a2, 1096(sp)                    # 8-byte Folded Spill
	lh	a2, 700(a0)
	sd	a2, 1088(sp)                    # 8-byte Folded Spill
	lh	a2, 702(a0)
	sd	a2, 1080(sp)                    # 8-byte Folded Spill
	lh	a2, 704(a0)
	sd	a2, 1072(sp)                    # 8-byte Folded Spill
	lh	a2, 706(a0)
	sd	a2, 1064(sp)                    # 8-byte Folded Spill
	lh	a2, 708(a0)
	sd	a2, 1056(sp)                    # 8-byte Folded Spill
	lh	a2, 710(a0)
	sd	a2, 1048(sp)                    # 8-byte Folded Spill
	lh	a2, 712(a0)
	sd	a2, 1040(sp)                    # 8-byte Folded Spill
	lh	a2, 714(a0)
	sd	a2, 1032(sp)                    # 8-byte Folded Spill
	lh	a2, 716(a0)
	sd	a2, 1024(sp)                    # 8-byte Folded Spill
	lh	a2, 718(a0)
	sd	a2, 1016(sp)                    # 8-byte Folded Spill
	lh	a2, 720(a0)
	sd	a2, 1008(sp)                    # 8-byte Folded Spill
	lh	a2, 722(a0)
	sd	a2, 1000(sp)                    # 8-byte Folded Spill
	lh	a2, 724(a0)
	sd	a2, 992(sp)                     # 8-byte Folded Spill
	lh	a2, 726(a0)
	sd	a2, 984(sp)                     # 8-byte Folded Spill
	lh	a2, 728(a0)
	sd	a2, 976(sp)                     # 8-byte Folded Spill
	lh	a2, 730(a0)
	sd	a2, 968(sp)                     # 8-byte Folded Spill
	lh	a2, 732(a0)
	sd	a2, 960(sp)                     # 8-byte Folded Spill
	lh	a2, 734(a0)
	sd	a2, 952(sp)                     # 8-byte Folded Spill
	lh	a2, 736(a0)
	sd	a2, 944(sp)                     # 8-byte Folded Spill
	lh	a2, 738(a0)
	sd	a2, 936(sp)                     # 8-byte Folded Spill
	lh	a2, 740(a0)
	sd	a2, 928(sp)                     # 8-byte Folded Spill
	lh	a2, 742(a0)
	sd	a2, 920(sp)                     # 8-byte Folded Spill
	lh	a2, 744(a0)
	sd	a2, 912(sp)                     # 8-byte Folded Spill
	lh	a2, 746(a0)
	sd	a2, 904(sp)                     # 8-byte Folded Spill
	lh	a2, 748(a0)
	sd	a2, 896(sp)                     # 8-byte Folded Spill
	lh	a2, 750(a0)
	sd	a2, 888(sp)                     # 8-byte Folded Spill
	lh	a2, 752(a0)
	sd	a2, 880(sp)                     # 8-byte Folded Spill
	lh	a2, 754(a0)
	sd	a2, 872(sp)                     # 8-byte Folded Spill
	lh	a2, 756(a0)
	sd	a2, 864(sp)                     # 8-byte Folded Spill
	lh	a2, 758(a0)
	sd	a2, 856(sp)                     # 8-byte Folded Spill
	lh	a2, 760(a0)
	sd	a2, 832(sp)                     # 8-byte Folded Spill
	lh	a2, 762(a0)
	sd	a2, 824(sp)                     # 8-byte Folded Spill
	lh	a2, 764(a0)
	sd	a2, 816(sp)                     # 8-byte Folded Spill
	lh	a2, 766(a0)
	sd	a2, 808(sp)                     # 8-byte Folded Spill
	lh	a2, 768(a0)
	sd	a2, 800(sp)                     # 8-byte Folded Spill
	lh	a2, 770(a0)
	sd	a2, 792(sp)                     # 8-byte Folded Spill
	lh	a2, 772(a0)
	sd	a2, 784(sp)                     # 8-byte Folded Spill
	lh	a2, 774(a0)
	sd	a2, 776(sp)                     # 8-byte Folded Spill
	lh	a2, 776(a0)
	sd	a2, 768(sp)                     # 8-byte Folded Spill
	lh	a2, 778(a0)
	sd	a2, 760(sp)                     # 8-byte Folded Spill
	lh	a2, 780(a0)
	sd	a2, 752(sp)                     # 8-byte Folded Spill
	lh	a2, 782(a0)
	sd	a2, 744(sp)                     # 8-byte Folded Spill
	lh	a2, 784(a0)
	sd	a2, 736(sp)                     # 8-byte Folded Spill
	lh	a2, 786(a0)
	sd	a2, 728(sp)                     # 8-byte Folded Spill
	lh	a2, 788(a0)
	sd	a2, 720(sp)                     # 8-byte Folded Spill
	lh	a2, 790(a0)
	sd	a2, 712(sp)                     # 8-byte Folded Spill
	lh	a2, 792(a0)
	sd	a2, 704(sp)                     # 8-byte Folded Spill
	lh	a2, 794(a0)
	sd	a2, 696(sp)                     # 8-byte Folded Spill
	lh	a2, 796(a0)
	sd	a2, 688(sp)                     # 8-byte Folded Spill
	lh	a2, 798(a0)
	sd	a2, 680(sp)                     # 8-byte Folded Spill
	lh	a2, 800(a0)
	sd	a2, 672(sp)                     # 8-byte Folded Spill
	lh	a2, 802(a0)
	sd	a2, 664(sp)                     # 8-byte Folded Spill
	lh	a2, 804(a0)
	sd	a2, 656(sp)                     # 8-byte Folded Spill
	lh	a2, 806(a0)
	sd	a2, 648(sp)                     # 8-byte Folded Spill
	lh	a2, 808(a0)
	sd	a2, 640(sp)                     # 8-byte Folded Spill
	lh	a2, 810(a0)
	sd	a2, 632(sp)                     # 8-byte Folded Spill
	lh	a2, 812(a0)
	sd	a2, 624(sp)                     # 8-byte Folded Spill
	lh	a2, 814(a0)
	sd	a2, 616(sp)                     # 8-byte Folded Spill
	lh	a2, 816(a0)
	sd	a2, 608(sp)                     # 8-byte Folded Spill
	lh	a2, 818(a0)
	sd	a2, 600(sp)                     # 8-byte Folded Spill
	lh	a2, 820(a0)
	sd	a2, 592(sp)                     # 8-byte Folded Spill
	lh	a2, 822(a0)
	sd	a2, 584(sp)                     # 8-byte Folded Spill
	lh	a2, 824(a0)
	sd	a2, 576(sp)                     # 8-byte Folded Spill
	lh	a2, 826(a0)
	sd	a2, 568(sp)                     # 8-byte Folded Spill
	lh	a2, 828(a0)
	sd	a2, 560(sp)                     # 8-byte Folded Spill
	lh	a2, 830(a0)
	sd	a2, 552(sp)                     # 8-byte Folded Spill
	lh	a2, 832(a0)
	sd	a2, 544(sp)                     # 8-byte Folded Spill
	lh	a2, 834(a0)
	sd	a2, 536(sp)                     # 8-byte Folded Spill
	lh	a2, 836(a0)
	sd	a2, 528(sp)                     # 8-byte Folded Spill
	lh	a2, 838(a0)
	sd	a2, 520(sp)                     # 8-byte Folded Spill
	lh	a2, 840(a0)
	sd	a2, 512(sp)                     # 8-byte Folded Spill
	lh	a2, 842(a0)
	sd	a2, 504(sp)                     # 8-byte Folded Spill
	lh	a2, 844(a0)
	sd	a2, 496(sp)                     # 8-byte Folded Spill
	lh	a2, 846(a0)
	sd	a2, 488(sp)                     # 8-byte Folded Spill
	lh	a2, 848(a0)
	sd	a2, 480(sp)                     # 8-byte Folded Spill
	lh	a2, 850(a0)
	sd	a2, 472(sp)                     # 8-byte Folded Spill
	lh	a2, 852(a0)
	sd	a2, 464(sp)                     # 8-byte Folded Spill
	lh	a2, 854(a0)
	sd	a2, 456(sp)                     # 8-byte Folded Spill
	lh	a2, 856(a0)
	sd	a2, 448(sp)                     # 8-byte Folded Spill
	lh	a2, 858(a0)
	sd	a2, 440(sp)                     # 8-byte Folded Spill
	lh	a2, 860(a0)
	sd	a2, 432(sp)                     # 8-byte Folded Spill
	lh	a2, 862(a0)
	sd	a2, 424(sp)                     # 8-byte Folded Spill
	lh	a2, 864(a0)
	sd	a2, 416(sp)                     # 8-byte Folded Spill
	lh	a2, 866(a0)
	sd	a2, 408(sp)                     # 8-byte Folded Spill
	lh	a2, 868(a0)
	sd	a2, 400(sp)                     # 8-byte Folded Spill
	lh	a2, 870(a0)
	sd	a2, 392(sp)                     # 8-byte Folded Spill
	lh	a2, 872(a0)
	sd	a2, 384(sp)                     # 8-byte Folded Spill
	lh	a2, 874(a0)
	sd	a2, 376(sp)                     # 8-byte Folded Spill
	lh	a2, 876(a0)
	sd	a2, 368(sp)                     # 8-byte Folded Spill
	lh	a2, 878(a0)
	sd	a2, 360(sp)                     # 8-byte Folded Spill
	lh	a2, 880(a0)
	sd	a2, 352(sp)                     # 8-byte Folded Spill
	lh	a2, 882(a0)
	sd	a2, 344(sp)                     # 8-byte Folded Spill
	lh	a2, 884(a0)
	sd	a2, 336(sp)                     # 8-byte Folded Spill
	lh	a2, 886(a0)
	sd	a2, 328(sp)                     # 8-byte Folded Spill
	lh	a2, 888(a0)
	sd	a2, 320(sp)                     # 8-byte Folded Spill
	lh	a2, 890(a0)
	sd	a2, 312(sp)                     # 8-byte Folded Spill
	lh	a2, 892(a0)
	sd	a2, 304(sp)                     # 8-byte Folded Spill
	lh	a2, 894(a0)
	sd	a2, 296(sp)                     # 8-byte Folded Spill
	lh	a2, 896(a0)
	sd	a2, 288(sp)                     # 8-byte Folded Spill
	lh	a2, 898(a0)
	sd	a2, 280(sp)                     # 8-byte Folded Spill
	lh	a2, 900(a0)
	sd	a2, 272(sp)                     # 8-byte Folded Spill
	lh	a2, 902(a0)
	sd	a2, 264(sp)                     # 8-byte Folded Spill
	lh	a2, 904(a0)
	sd	a2, 256(sp)                     # 8-byte Folded Spill
	lh	a2, 906(a0)
	sd	a2, 248(sp)                     # 8-byte Folded Spill
	lh	a2, 908(a0)
	sd	a2, 240(sp)                     # 8-byte Folded Spill
	lh	a2, 910(a0)
	sd	a2, 232(sp)                     # 8-byte Folded Spill
	lh	a2, 912(a0)
	sd	a2, 224(sp)                     # 8-byte Folded Spill
	lh	a2, 914(a0)
	sd	a2, 216(sp)                     # 8-byte Folded Spill
	lh	a2, 916(a0)
	sd	a2, 208(sp)                     # 8-byte Folded Spill
	lh	a2, 918(a0)
	sd	a2, 200(sp)                     # 8-byte Folded Spill
	lh	a2, 920(a0)
	sd	a2, 192(sp)                     # 8-byte Folded Spill
	lh	a2, 922(a0)
	sd	a2, 184(sp)                     # 8-byte Folded Spill
	lh	a2, 924(a0)
	sd	a2, 176(sp)                     # 8-byte Folded Spill
	lh	a2, 926(a0)
	sd	a2, 168(sp)                     # 8-byte Folded Spill
	lh	a2, 928(a0)
	sd	a2, 160(sp)                     # 8-byte Folded Spill
	lh	a2, 930(a0)
	sd	a2, 152(sp)                     # 8-byte Folded Spill
	lh	a2, 932(a0)
	sd	a2, 144(sp)                     # 8-byte Folded Spill
	lh	a2, 934(a0)
	sd	a2, 136(sp)                     # 8-byte Folded Spill
	lh	a2, 936(a0)
	sd	a2, 128(sp)                     # 8-byte Folded Spill
	lh	a2, 938(a0)
	sd	a2, 120(sp)                     # 8-byte Folded Spill
	lh	a2, 940(a0)
	sd	a2, 112(sp)                     # 8-byte Folded Spill
	lh	a2, 942(a0)
	sd	a2, 104(sp)                     # 8-byte Folded Spill
	lh	a2, 944(a0)
	sd	a2, 96(sp)                      # 8-byte Folded Spill
	lh	a2, 946(a0)
	sd	a2, 88(sp)                      # 8-byte Folded Spill
	lh	a2, 948(a0)
	sd	a2, 80(sp)                      # 8-byte Folded Spill
	lh	a2, 950(a0)
	sd	a2, 72(sp)                      # 8-byte Folded Spill
	lh	a2, 952(a0)
	sd	a2, 64(sp)                      # 8-byte Folded Spill
	lh	a2, 954(a0)
	sd	a2, 56(sp)                      # 8-byte Folded Spill
	lh	a2, 956(a0)
	sd	a2, 48(sp)                      # 8-byte Folded Spill
	lh	a2, 958(a0)
	sd	a2, 40(sp)                      # 8-byte Folded Spill
	lh	a2, 960(a0)
	sd	a2, 32(sp)                      # 8-byte Folded Spill
	lh	a2, 962(a0)
	sd	a2, 24(sp)                      # 8-byte Folded Spill
	lh	a2, 964(a0)
	sd	a2, 16(sp)                      # 8-byte Folded Spill
	lh	a2, 966(a0)
	sd	a2, 8(sp)                       # 8-byte Folded Spill
	lh	s11, 968(a0)
	lh	s8, 970(a0)
	lh	s9, 972(a0)
	lh	s10, 974(a0)
	lh	s7, 976(a0)
	lh	s4, 978(a0)
	lh	s5, 980(a0)
	lh	s6, 982(a0)
	lh	s3, 984(a0)
	lh	s0, 986(a0)
	lh	s1, 988(a0)
	lh	s2, 990(a0)
	lh	t6, 992(a0)
	lh	t4, 994(a0)
	lh	t5, 996(a0)
	lh	t2, 998(a0)
	lh	t3, 1000(a0)
	lh	t0, 1002(a0)
	lh	t1, 1004(a0)
	lh	a7, 1006(a0)
	lh	a6, 1008(a0)
	lh	a3, 1010(a0)
	lh	a4, 1012(a0)
	lh	a5, 1014(a0)
	lh	a2, 1020(a0)
	lh	a0, 1022(a0)
	ld	ra, 848(sp)                     # 8-byte Folded Reload
	.loc	1 11 36 is_stmt 1               # k135112023920016.py:11:36
	sh	ra, 1016(a1)
	ld	ra, 840(sp)                     # 8-byte Folded Reload
	sh	ra, 1018(a1)
	sh	a2, 1020(a1)
	sh	a0, 1022(a1)
	sh	a6, 1008(a1)
	sh	a3, 1010(a1)
	sh	a4, 1012(a1)
	sh	a5, 1014(a1)
	sh	t2, 998(a1)
	sh	t0, 1002(a1)
	sh	t1, 1004(a1)
	sh	a7, 1006(a1)
	sh	t6, 992(a1)
	sh	t4, 994(a1)
	sh	t5, 996(a1)
	sh	t3, 1000(a1)
	sh	s3, 984(a1)
	sh	s0, 986(a1)
	sh	s1, 988(a1)
	sh	s2, 990(a1)
	sh	s7, 976(a1)
	sh	s4, 978(a1)
	sh	s5, 980(a1)
	sh	s6, 982(a1)
	sh	s11, 968(a1)
	sh	s8, 970(a1)
	sh	s9, 972(a1)
	sh	s10, 974(a1)
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	sh	a0, 960(a1)
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	sh	a0, 962(a1)
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	sh	a0, 964(a1)
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	sh	a0, 966(a1)
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	sh	a0, 952(a1)
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	sh	a0, 954(a1)
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	sh	a0, 956(a1)
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	sh	a0, 958(a1)
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	sh	a0, 944(a1)
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	sh	a0, 946(a1)
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	sh	a0, 948(a1)
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	sh	a0, 950(a1)
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	sh	a0, 936(a1)
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	sh	a0, 938(a1)
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	sh	a0, 940(a1)
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	sh	a0, 942(a1)
	ld	a0, 160(sp)                     # 8-byte Folded Reload
	sh	a0, 928(a1)
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	sh	a0, 930(a1)
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	sh	a0, 932(a1)
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	sh	a0, 934(a1)
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	sh	a0, 920(a1)
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	sh	a0, 922(a1)
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	sh	a0, 924(a1)
	ld	a0, 168(sp)                     # 8-byte Folded Reload
	sh	a0, 926(a1)
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	sh	a0, 912(a1)
	ld	a0, 216(sp)                     # 8-byte Folded Reload
	sh	a0, 914(a1)
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	sh	a0, 916(a1)
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	sh	a0, 918(a1)
	ld	a0, 256(sp)                     # 8-byte Folded Reload
	sh	a0, 904(a1)
	ld	a0, 248(sp)                     # 8-byte Folded Reload
	sh	a0, 906(a1)
	ld	a0, 240(sp)                     # 8-byte Folded Reload
	sh	a0, 908(a1)
	ld	a0, 232(sp)                     # 8-byte Folded Reload
	sh	a0, 910(a1)
	ld	a0, 288(sp)                     # 8-byte Folded Reload
	sh	a0, 896(a1)
	ld	a0, 280(sp)                     # 8-byte Folded Reload
	sh	a0, 898(a1)
	ld	a0, 272(sp)                     # 8-byte Folded Reload
	sh	a0, 900(a1)
	ld	a0, 264(sp)                     # 8-byte Folded Reload
	sh	a0, 902(a1)
	ld	a0, 320(sp)                     # 8-byte Folded Reload
	sh	a0, 888(a1)
	ld	a0, 312(sp)                     # 8-byte Folded Reload
	sh	a0, 890(a1)
	ld	a0, 304(sp)                     # 8-byte Folded Reload
	sh	a0, 892(a1)
	ld	a0, 296(sp)                     # 8-byte Folded Reload
	sh	a0, 894(a1)
	ld	a0, 352(sp)                     # 8-byte Folded Reload
	sh	a0, 880(a1)
	ld	a0, 344(sp)                     # 8-byte Folded Reload
	sh	a0, 882(a1)
	ld	a0, 336(sp)                     # 8-byte Folded Reload
	sh	a0, 884(a1)
	ld	a0, 328(sp)                     # 8-byte Folded Reload
	sh	a0, 886(a1)
	ld	a0, 384(sp)                     # 8-byte Folded Reload
	sh	a0, 872(a1)
	ld	a0, 376(sp)                     # 8-byte Folded Reload
	sh	a0, 874(a1)
	ld	a0, 368(sp)                     # 8-byte Folded Reload
	sh	a0, 876(a1)
	ld	a0, 360(sp)                     # 8-byte Folded Reload
	sh	a0, 878(a1)
	ld	a0, 416(sp)                     # 8-byte Folded Reload
	sh	a0, 864(a1)
	ld	a0, 408(sp)                     # 8-byte Folded Reload
	sh	a0, 866(a1)
	ld	a0, 400(sp)                     # 8-byte Folded Reload
	sh	a0, 868(a1)
	ld	a0, 392(sp)                     # 8-byte Folded Reload
	sh	a0, 870(a1)
	ld	a0, 448(sp)                     # 8-byte Folded Reload
	sh	a0, 856(a1)
	ld	a0, 440(sp)                     # 8-byte Folded Reload
	sh	a0, 858(a1)
	ld	a0, 432(sp)                     # 8-byte Folded Reload
	sh	a0, 860(a1)
	ld	a0, 424(sp)                     # 8-byte Folded Reload
	sh	a0, 862(a1)
	ld	a0, 480(sp)                     # 8-byte Folded Reload
	sh	a0, 848(a1)
	ld	a0, 472(sp)                     # 8-byte Folded Reload
	sh	a0, 850(a1)
	ld	a0, 464(sp)                     # 8-byte Folded Reload
	sh	a0, 852(a1)
	ld	a0, 456(sp)                     # 8-byte Folded Reload
	sh	a0, 854(a1)
	ld	a0, 512(sp)                     # 8-byte Folded Reload
	sh	a0, 840(a1)
	ld	a0, 504(sp)                     # 8-byte Folded Reload
	sh	a0, 842(a1)
	ld	a0, 496(sp)                     # 8-byte Folded Reload
	sh	a0, 844(a1)
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	sh	a0, 846(a1)
	ld	a0, 544(sp)                     # 8-byte Folded Reload
	sh	a0, 832(a1)
	ld	a0, 536(sp)                     # 8-byte Folded Reload
	sh	a0, 834(a1)
	ld	a0, 528(sp)                     # 8-byte Folded Reload
	sh	a0, 836(a1)
	ld	a0, 520(sp)                     # 8-byte Folded Reload
	sh	a0, 838(a1)
	ld	a0, 576(sp)                     # 8-byte Folded Reload
	sh	a0, 824(a1)
	ld	a0, 568(sp)                     # 8-byte Folded Reload
	sh	a0, 826(a1)
	ld	a0, 560(sp)                     # 8-byte Folded Reload
	sh	a0, 828(a1)
	ld	a0, 552(sp)                     # 8-byte Folded Reload
	sh	a0, 830(a1)
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	sh	a0, 816(a1)
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	sh	a0, 818(a1)
	ld	a0, 592(sp)                     # 8-byte Folded Reload
	sh	a0, 820(a1)
	ld	a0, 584(sp)                     # 8-byte Folded Reload
	sh	a0, 822(a1)
	ld	a0, 640(sp)                     # 8-byte Folded Reload
	sh	a0, 808(a1)
	ld	a0, 632(sp)                     # 8-byte Folded Reload
	sh	a0, 810(a1)
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	sh	a0, 812(a1)
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	sh	a0, 814(a1)
	ld	a0, 672(sp)                     # 8-byte Folded Reload
	sh	a0, 800(a1)
	ld	a0, 664(sp)                     # 8-byte Folded Reload
	sh	a0, 802(a1)
	ld	a0, 656(sp)                     # 8-byte Folded Reload
	sh	a0, 804(a1)
	ld	a0, 648(sp)                     # 8-byte Folded Reload
	sh	a0, 806(a1)
	ld	a0, 704(sp)                     # 8-byte Folded Reload
	sh	a0, 792(a1)
	ld	a0, 696(sp)                     # 8-byte Folded Reload
	sh	a0, 794(a1)
	ld	a0, 688(sp)                     # 8-byte Folded Reload
	sh	a0, 796(a1)
	ld	a0, 680(sp)                     # 8-byte Folded Reload
	sh	a0, 798(a1)
	ld	a0, 736(sp)                     # 8-byte Folded Reload
	sh	a0, 784(a1)
	ld	a0, 728(sp)                     # 8-byte Folded Reload
	sh	a0, 786(a1)
	ld	a0, 720(sp)                     # 8-byte Folded Reload
	sh	a0, 788(a1)
	ld	a0, 712(sp)                     # 8-byte Folded Reload
	sh	a0, 790(a1)
	ld	a0, 768(sp)                     # 8-byte Folded Reload
	sh	a0, 776(a1)
	ld	a0, 760(sp)                     # 8-byte Folded Reload
	sh	a0, 778(a1)
	ld	a0, 752(sp)                     # 8-byte Folded Reload
	sh	a0, 780(a1)
	ld	a0, 744(sp)                     # 8-byte Folded Reload
	sh	a0, 782(a1)
	ld	a0, 800(sp)                     # 8-byte Folded Reload
	sh	a0, 768(a1)
	ld	a0, 792(sp)                     # 8-byte Folded Reload
	sh	a0, 770(a1)
	ld	a0, 784(sp)                     # 8-byte Folded Reload
	sh	a0, 772(a1)
	ld	a0, 776(sp)                     # 8-byte Folded Reload
	sh	a0, 774(a1)
	ld	a0, 832(sp)                     # 8-byte Folded Reload
	sh	a0, 760(a1)
	ld	a0, 824(sp)                     # 8-byte Folded Reload
	sh	a0, 762(a1)
	ld	a0, 816(sp)                     # 8-byte Folded Reload
	sh	a0, 764(a1)
	ld	a0, 808(sp)                     # 8-byte Folded Reload
	sh	a0, 766(a1)
	ld	a0, 880(sp)                     # 8-byte Folded Reload
	sh	a0, 752(a1)
	ld	a0, 872(sp)                     # 8-byte Folded Reload
	sh	a0, 754(a1)
	ld	a0, 864(sp)                     # 8-byte Folded Reload
	sh	a0, 756(a1)
	ld	a0, 856(sp)                     # 8-byte Folded Reload
	sh	a0, 758(a1)
	ld	a0, 912(sp)                     # 8-byte Folded Reload
	sh	a0, 744(a1)
	ld	a0, 904(sp)                     # 8-byte Folded Reload
	sh	a0, 746(a1)
	ld	a0, 896(sp)                     # 8-byte Folded Reload
	sh	a0, 748(a1)
	ld	a0, 888(sp)                     # 8-byte Folded Reload
	sh	a0, 750(a1)
	ld	a0, 944(sp)                     # 8-byte Folded Reload
	sh	a0, 736(a1)
	ld	a0, 936(sp)                     # 8-byte Folded Reload
	sh	a0, 738(a1)
	ld	a0, 928(sp)                     # 8-byte Folded Reload
	sh	a0, 740(a1)
	ld	a0, 920(sp)                     # 8-byte Folded Reload
	sh	a0, 742(a1)
	ld	a0, 976(sp)                     # 8-byte Folded Reload
	sh	a0, 728(a1)
	ld	a0, 968(sp)                     # 8-byte Folded Reload
	sh	a0, 730(a1)
	ld	a0, 960(sp)                     # 8-byte Folded Reload
	sh	a0, 732(a1)
	ld	a0, 952(sp)                     # 8-byte Folded Reload
	sh	a0, 734(a1)
	ld	a0, 1008(sp)                    # 8-byte Folded Reload
	sh	a0, 720(a1)
	ld	a0, 1000(sp)                    # 8-byte Folded Reload
	sh	a0, 722(a1)
	ld	a0, 992(sp)                     # 8-byte Folded Reload
	sh	a0, 724(a1)
	ld	a0, 984(sp)                     # 8-byte Folded Reload
	sh	a0, 726(a1)
	ld	a0, 1040(sp)                    # 8-byte Folded Reload
	sh	a0, 712(a1)
	ld	a0, 1032(sp)                    # 8-byte Folded Reload
	sh	a0, 714(a1)
	ld	a0, 1024(sp)                    # 8-byte Folded Reload
	sh	a0, 716(a1)
	ld	a0, 1016(sp)                    # 8-byte Folded Reload
	sh	a0, 718(a1)
	ld	a0, 1072(sp)                    # 8-byte Folded Reload
	sh	a0, 704(a1)
	ld	a0, 1064(sp)                    # 8-byte Folded Reload
	sh	a0, 706(a1)
	ld	a0, 1056(sp)                    # 8-byte Folded Reload
	sh	a0, 708(a1)
	ld	a0, 1048(sp)                    # 8-byte Folded Reload
	sh	a0, 710(a1)
	ld	a0, 1104(sp)                    # 8-byte Folded Reload
	sh	a0, 696(a1)
	ld	a0, 1096(sp)                    # 8-byte Folded Reload
	sh	a0, 698(a1)
	ld	a0, 1088(sp)                    # 8-byte Folded Reload
	sh	a0, 700(a1)
	ld	a0, 1080(sp)                    # 8-byte Folded Reload
	sh	a0, 702(a1)
	ld	a0, 1136(sp)                    # 8-byte Folded Reload
	sh	a0, 688(a1)
	ld	a0, 1128(sp)                    # 8-byte Folded Reload
	sh	a0, 690(a1)
	ld	a0, 1120(sp)                    # 8-byte Folded Reload
	sh	a0, 692(a1)
	ld	a0, 1112(sp)                    # 8-byte Folded Reload
	sh	a0, 694(a1)
	ld	a0, 1168(sp)                    # 8-byte Folded Reload
	sh	a0, 680(a1)
	ld	a0, 1160(sp)                    # 8-byte Folded Reload
	sh	a0, 682(a1)
	ld	a0, 1152(sp)                    # 8-byte Folded Reload
	sh	a0, 684(a1)
	ld	a0, 1144(sp)                    # 8-byte Folded Reload
	sh	a0, 686(a1)
	ld	a0, 1200(sp)                    # 8-byte Folded Reload
	sh	a0, 672(a1)
	ld	a0, 1192(sp)                    # 8-byte Folded Reload
	sh	a0, 674(a1)
	ld	a0, 1184(sp)                    # 8-byte Folded Reload
	sh	a0, 676(a1)
	ld	a0, 1176(sp)                    # 8-byte Folded Reload
	sh	a0, 678(a1)
	ld	a0, 1232(sp)                    # 8-byte Folded Reload
	sh	a0, 664(a1)
	ld	a0, 1224(sp)                    # 8-byte Folded Reload
	sh	a0, 666(a1)
	ld	a0, 1216(sp)                    # 8-byte Folded Reload
	sh	a0, 668(a1)
	ld	a0, 1208(sp)                    # 8-byte Folded Reload
	sh	a0, 670(a1)
	ld	a0, 1264(sp)                    # 8-byte Folded Reload
	sh	a0, 656(a1)
	ld	a0, 1256(sp)                    # 8-byte Folded Reload
	sh	a0, 658(a1)
	ld	a0, 1248(sp)                    # 8-byte Folded Reload
	sh	a0, 660(a1)
	ld	a0, 1240(sp)                    # 8-byte Folded Reload
	sh	a0, 662(a1)
	ld	a0, 1296(sp)                    # 8-byte Folded Reload
	sh	a0, 648(a1)
	ld	a0, 1288(sp)                    # 8-byte Folded Reload
	sh	a0, 650(a1)
	ld	a0, 1280(sp)                    # 8-byte Folded Reload
	sh	a0, 652(a1)
	ld	a0, 1272(sp)                    # 8-byte Folded Reload
	sh	a0, 654(a1)
	ld	a0, 1328(sp)                    # 8-byte Folded Reload
	sh	a0, 640(a1)
	ld	a0, 1320(sp)                    # 8-byte Folded Reload
	sh	a0, 642(a1)
	ld	a0, 1312(sp)                    # 8-byte Folded Reload
	sh	a0, 644(a1)
	ld	a0, 1304(sp)                    # 8-byte Folded Reload
	sh	a0, 646(a1)
	ld	a0, 1360(sp)                    # 8-byte Folded Reload
	sh	a0, 632(a1)
	ld	a0, 1352(sp)                    # 8-byte Folded Reload
	sh	a0, 634(a1)
	ld	a0, 1344(sp)                    # 8-byte Folded Reload
	sh	a0, 636(a1)
	ld	a0, 1336(sp)                    # 8-byte Folded Reload
	sh	a0, 638(a1)
	ld	a0, 1392(sp)                    # 8-byte Folded Reload
	sh	a0, 624(a1)
	ld	a0, 1384(sp)                    # 8-byte Folded Reload
	sh	a0, 626(a1)
	ld	a0, 1376(sp)                    # 8-byte Folded Reload
	sh	a0, 628(a1)
	ld	a0, 1368(sp)                    # 8-byte Folded Reload
	sh	a0, 630(a1)
	ld	a0, 1424(sp)                    # 8-byte Folded Reload
	sh	a0, 616(a1)
	ld	a0, 1416(sp)                    # 8-byte Folded Reload
	sh	a0, 618(a1)
	ld	a0, 1408(sp)                    # 8-byte Folded Reload
	sh	a0, 620(a1)
	ld	a0, 1400(sp)                    # 8-byte Folded Reload
	sh	a0, 622(a1)
	ld	a0, 1456(sp)                    # 8-byte Folded Reload
	sh	a0, 608(a1)
	ld	a0, 1448(sp)                    # 8-byte Folded Reload
	sh	a0, 610(a1)
	ld	a0, 1440(sp)                    # 8-byte Folded Reload
	sh	a0, 612(a1)
	ld	a0, 1432(sp)                    # 8-byte Folded Reload
	sh	a0, 614(a1)
	ld	a0, 1488(sp)                    # 8-byte Folded Reload
	sh	a0, 600(a1)
	ld	a0, 1480(sp)                    # 8-byte Folded Reload
	sh	a0, 602(a1)
	ld	a0, 1472(sp)                    # 8-byte Folded Reload
	sh	a0, 604(a1)
	ld	a0, 1464(sp)                    # 8-byte Folded Reload
	sh	a0, 606(a1)
	ld	a0, 1520(sp)                    # 8-byte Folded Reload
	sh	a0, 592(a1)
	ld	a0, 1512(sp)                    # 8-byte Folded Reload
	sh	a0, 594(a1)
	ld	a0, 1504(sp)                    # 8-byte Folded Reload
	sh	a0, 596(a1)
	ld	a0, 1496(sp)                    # 8-byte Folded Reload
	sh	a0, 598(a1)
	ld	a0, 1552(sp)                    # 8-byte Folded Reload
	sh	a0, 584(a1)
	ld	a0, 1544(sp)                    # 8-byte Folded Reload
	sh	a0, 586(a1)
	ld	a0, 1536(sp)                    # 8-byte Folded Reload
	sh	a0, 588(a1)
	ld	a0, 1528(sp)                    # 8-byte Folded Reload
	sh	a0, 590(a1)
	ld	a0, 1584(sp)                    # 8-byte Folded Reload
	sh	a0, 576(a1)
	ld	a0, 1576(sp)                    # 8-byte Folded Reload
	sh	a0, 578(a1)
	ld	a0, 1568(sp)                    # 8-byte Folded Reload
	sh	a0, 580(a1)
	ld	a0, 1560(sp)                    # 8-byte Folded Reload
	sh	a0, 582(a1)
	ld	a0, 1616(sp)                    # 8-byte Folded Reload
	sh	a0, 568(a1)
	ld	a0, 1608(sp)                    # 8-byte Folded Reload
	sh	a0, 570(a1)
	ld	a0, 1600(sp)                    # 8-byte Folded Reload
	sh	a0, 572(a1)
	ld	a0, 1592(sp)                    # 8-byte Folded Reload
	sh	a0, 574(a1)
	ld	a0, 1648(sp)                    # 8-byte Folded Reload
	sh	a0, 560(a1)
	ld	a0, 1640(sp)                    # 8-byte Folded Reload
	sh	a0, 562(a1)
	ld	a0, 1632(sp)                    # 8-byte Folded Reload
	sh	a0, 564(a1)
	ld	a0, 1624(sp)                    # 8-byte Folded Reload
	sh	a0, 566(a1)
	ld	a0, 1680(sp)                    # 8-byte Folded Reload
	sh	a0, 552(a1)
	ld	a0, 1672(sp)                    # 8-byte Folded Reload
	sh	a0, 554(a1)
	ld	a0, 1664(sp)                    # 8-byte Folded Reload
	sh	a0, 556(a1)
	ld	a0, 1656(sp)                    # 8-byte Folded Reload
	sh	a0, 558(a1)
	ld	a0, 1712(sp)                    # 8-byte Folded Reload
	sh	a0, 544(a1)
	ld	a0, 1704(sp)                    # 8-byte Folded Reload
	sh	a0, 546(a1)
	ld	a0, 1696(sp)                    # 8-byte Folded Reload
	sh	a0, 548(a1)
	ld	a0, 1688(sp)                    # 8-byte Folded Reload
	sh	a0, 550(a1)
	ld	a0, 1744(sp)                    # 8-byte Folded Reload
	sh	a0, 536(a1)
	ld	a0, 1736(sp)                    # 8-byte Folded Reload
	sh	a0, 538(a1)
	ld	a0, 1728(sp)                    # 8-byte Folded Reload
	sh	a0, 540(a1)
	ld	a0, 1720(sp)                    # 8-byte Folded Reload
	sh	a0, 542(a1)
	ld	a0, 1776(sp)                    # 8-byte Folded Reload
	sh	a0, 528(a1)
	ld	a0, 1768(sp)                    # 8-byte Folded Reload
	sh	a0, 530(a1)
	ld	a0, 1760(sp)                    # 8-byte Folded Reload
	sh	a0, 532(a1)
	ld	a0, 1752(sp)                    # 8-byte Folded Reload
	sh	a0, 534(a1)
	ld	a0, 1808(sp)                    # 8-byte Folded Reload
	sh	a0, 520(a1)
	ld	a0, 1800(sp)                    # 8-byte Folded Reload
	sh	a0, 522(a1)
	ld	a0, 1792(sp)                    # 8-byte Folded Reload
	sh	a0, 524(a1)
	ld	a0, 1784(sp)                    # 8-byte Folded Reload
	sh	a0, 526(a1)
	ld	a0, 1840(sp)                    # 8-byte Folded Reload
	sh	a0, 512(a1)
	ld	a0, 1832(sp)                    # 8-byte Folded Reload
	sh	a0, 514(a1)
	ld	a0, 1824(sp)                    # 8-byte Folded Reload
	sh	a0, 516(a1)
	ld	a0, 1816(sp)                    # 8-byte Folded Reload
	sh	a0, 518(a1)
	ld	a0, 1872(sp)                    # 8-byte Folded Reload
	sh	a0, 504(a1)
	ld	a0, 1864(sp)                    # 8-byte Folded Reload
	sh	a0, 506(a1)
	ld	a0, 1856(sp)                    # 8-byte Folded Reload
	sh	a0, 508(a1)
	ld	a0, 1848(sp)                    # 8-byte Folded Reload
	sh	a0, 510(a1)
	ld	a0, 1904(sp)                    # 8-byte Folded Reload
	sh	a0, 496(a1)
	ld	a0, 1896(sp)                    # 8-byte Folded Reload
	sh	a0, 498(a1)
	ld	a0, 1888(sp)                    # 8-byte Folded Reload
	sh	a0, 500(a1)
	ld	a0, 1880(sp)                    # 8-byte Folded Reload
	sh	a0, 502(a1)
	ld	a0, 1936(sp)                    # 8-byte Folded Reload
	sh	a0, 488(a1)
	ld	a0, 1928(sp)                    # 8-byte Folded Reload
	sh	a0, 490(a1)
	ld	a0, 1920(sp)                    # 8-byte Folded Reload
	sh	a0, 492(a1)
	ld	a0, 1912(sp)                    # 8-byte Folded Reload
	sh	a0, 494(a1)
	ld	a0, 1968(sp)                    # 8-byte Folded Reload
	sh	a0, 480(a1)
	ld	a0, 1960(sp)                    # 8-byte Folded Reload
	sh	a0, 482(a1)
	ld	a0, 1952(sp)                    # 8-byte Folded Reload
	sh	a0, 484(a1)
	ld	a0, 1944(sp)                    # 8-byte Folded Reload
	sh	a0, 486(a1)
	ld	a0, 2000(sp)                    # 8-byte Folded Reload
	sh	a0, 472(a1)
	ld	a0, 1992(sp)                    # 8-byte Folded Reload
	sh	a0, 474(a1)
	ld	a0, 1984(sp)                    # 8-byte Folded Reload
	sh	a0, 476(a1)
	ld	a0, 1976(sp)                    # 8-byte Folded Reload
	sh	a0, 478(a1)
	ld	a0, 2032(sp)                    # 8-byte Folded Reload
	sh	a0, 464(a1)
	ld	a0, 2024(sp)                    # 8-byte Folded Reload
	sh	a0, 466(a1)
	ld	a0, 2016(sp)                    # 8-byte Folded Reload
	sh	a0, 468(a1)
	ld	a0, 2008(sp)                    # 8-byte Folded Reload
	sh	a0, 470(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -2032(a0)                   # 8-byte Folded Reload
	sh	a0, 456(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -2040(a0)                   # 8-byte Folded Reload
	sh	a0, 458(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -2048(a0)                   # 8-byte Folded Reload
	sh	a0, 460(a1)
	ld	a0, 2040(sp)                    # 8-byte Folded Reload
	sh	a0, 462(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -2000(a0)                   # 8-byte Folded Reload
	sh	a0, 448(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -2008(a0)                   # 8-byte Folded Reload
	sh	a0, 450(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -2016(a0)                   # 8-byte Folded Reload
	sh	a0, 452(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -2024(a0)                   # 8-byte Folded Reload
	sh	a0, 454(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1968(a0)                   # 8-byte Folded Reload
	sh	a0, 440(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1976(a0)                   # 8-byte Folded Reload
	sh	a0, 442(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1984(a0)                   # 8-byte Folded Reload
	sh	a0, 444(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1992(a0)                   # 8-byte Folded Reload
	sh	a0, 446(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1936(a0)                   # 8-byte Folded Reload
	sh	a0, 432(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1944(a0)                   # 8-byte Folded Reload
	sh	a0, 434(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1952(a0)                   # 8-byte Folded Reload
	sh	a0, 436(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1960(a0)                   # 8-byte Folded Reload
	sh	a0, 438(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1904(a0)                   # 8-byte Folded Reload
	sh	a0, 424(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1912(a0)                   # 8-byte Folded Reload
	sh	a0, 426(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1920(a0)                   # 8-byte Folded Reload
	sh	a0, 428(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1928(a0)                   # 8-byte Folded Reload
	sh	a0, 430(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1872(a0)                   # 8-byte Folded Reload
	sh	a0, 416(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1880(a0)                   # 8-byte Folded Reload
	sh	a0, 418(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1888(a0)                   # 8-byte Folded Reload
	sh	a0, 420(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1896(a0)                   # 8-byte Folded Reload
	sh	a0, 422(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1840(a0)                   # 8-byte Folded Reload
	sh	a0, 408(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1848(a0)                   # 8-byte Folded Reload
	sh	a0, 410(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1856(a0)                   # 8-byte Folded Reload
	sh	a0, 412(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1864(a0)                   # 8-byte Folded Reload
	sh	a0, 414(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1808(a0)                   # 8-byte Folded Reload
	sh	a0, 400(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1816(a0)                   # 8-byte Folded Reload
	sh	a0, 402(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1824(a0)                   # 8-byte Folded Reload
	sh	a0, 404(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1832(a0)                   # 8-byte Folded Reload
	sh	a0, 406(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1776(a0)                   # 8-byte Folded Reload
	sh	a0, 392(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1784(a0)                   # 8-byte Folded Reload
	sh	a0, 394(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1792(a0)                   # 8-byte Folded Reload
	sh	a0, 396(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1800(a0)                   # 8-byte Folded Reload
	sh	a0, 398(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1744(a0)                   # 8-byte Folded Reload
	sh	a0, 384(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1752(a0)                   # 8-byte Folded Reload
	sh	a0, 386(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1760(a0)                   # 8-byte Folded Reload
	sh	a0, 388(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1768(a0)                   # 8-byte Folded Reload
	sh	a0, 390(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1712(a0)                   # 8-byte Folded Reload
	sh	a0, 376(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1720(a0)                   # 8-byte Folded Reload
	sh	a0, 378(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1728(a0)                   # 8-byte Folded Reload
	sh	a0, 380(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1736(a0)                   # 8-byte Folded Reload
	sh	a0, 382(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1680(a0)                   # 8-byte Folded Reload
	sh	a0, 368(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1688(a0)                   # 8-byte Folded Reload
	sh	a0, 370(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1696(a0)                   # 8-byte Folded Reload
	sh	a0, 372(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1704(a0)                   # 8-byte Folded Reload
	sh	a0, 374(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1648(a0)                   # 8-byte Folded Reload
	sh	a0, 360(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1656(a0)                   # 8-byte Folded Reload
	sh	a0, 362(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1664(a0)                   # 8-byte Folded Reload
	sh	a0, 364(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1672(a0)                   # 8-byte Folded Reload
	sh	a0, 366(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1616(a0)                   # 8-byte Folded Reload
	sh	a0, 352(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1624(a0)                   # 8-byte Folded Reload
	sh	a0, 354(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1632(a0)                   # 8-byte Folded Reload
	sh	a0, 356(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1640(a0)                   # 8-byte Folded Reload
	sh	a0, 358(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1584(a0)                   # 8-byte Folded Reload
	sh	a0, 344(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1592(a0)                   # 8-byte Folded Reload
	sh	a0, 346(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1600(a0)                   # 8-byte Folded Reload
	sh	a0, 348(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1608(a0)                   # 8-byte Folded Reload
	sh	a0, 350(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1552(a0)                   # 8-byte Folded Reload
	sh	a0, 336(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1560(a0)                   # 8-byte Folded Reload
	sh	a0, 338(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1568(a0)                   # 8-byte Folded Reload
	sh	a0, 340(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1576(a0)                   # 8-byte Folded Reload
	sh	a0, 342(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1520(a0)                   # 8-byte Folded Reload
	sh	a0, 328(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1528(a0)                   # 8-byte Folded Reload
	sh	a0, 330(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1536(a0)                   # 8-byte Folded Reload
	sh	a0, 332(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1544(a0)                   # 8-byte Folded Reload
	sh	a0, 334(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1488(a0)                   # 8-byte Folded Reload
	sh	a0, 320(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1496(a0)                   # 8-byte Folded Reload
	sh	a0, 322(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1504(a0)                   # 8-byte Folded Reload
	sh	a0, 324(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1512(a0)                   # 8-byte Folded Reload
	sh	a0, 326(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1456(a0)                   # 8-byte Folded Reload
	sh	a0, 312(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1464(a0)                   # 8-byte Folded Reload
	sh	a0, 314(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1472(a0)                   # 8-byte Folded Reload
	sh	a0, 316(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1480(a0)                   # 8-byte Folded Reload
	sh	a0, 318(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1424(a0)                   # 8-byte Folded Reload
	sh	a0, 304(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1432(a0)                   # 8-byte Folded Reload
	sh	a0, 306(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1440(a0)                   # 8-byte Folded Reload
	sh	a0, 308(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1448(a0)                   # 8-byte Folded Reload
	sh	a0, 310(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1392(a0)                   # 8-byte Folded Reload
	sh	a0, 296(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1400(a0)                   # 8-byte Folded Reload
	sh	a0, 298(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1408(a0)                   # 8-byte Folded Reload
	sh	a0, 300(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1416(a0)                   # 8-byte Folded Reload
	sh	a0, 302(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1360(a0)                   # 8-byte Folded Reload
	sh	a0, 288(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1368(a0)                   # 8-byte Folded Reload
	sh	a0, 290(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1376(a0)                   # 8-byte Folded Reload
	sh	a0, 292(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1384(a0)                   # 8-byte Folded Reload
	sh	a0, 294(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1328(a0)                   # 8-byte Folded Reload
	sh	a0, 280(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1336(a0)                   # 8-byte Folded Reload
	sh	a0, 282(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1344(a0)                   # 8-byte Folded Reload
	sh	a0, 284(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1352(a0)                   # 8-byte Folded Reload
	sh	a0, 286(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1296(a0)                   # 8-byte Folded Reload
	sh	a0, 272(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1304(a0)                   # 8-byte Folded Reload
	sh	a0, 274(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1312(a0)                   # 8-byte Folded Reload
	sh	a0, 276(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1320(a0)                   # 8-byte Folded Reload
	sh	a0, 278(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1264(a0)                   # 8-byte Folded Reload
	sh	a0, 264(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1272(a0)                   # 8-byte Folded Reload
	sh	a0, 266(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1280(a0)                   # 8-byte Folded Reload
	sh	a0, 268(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1288(a0)                   # 8-byte Folded Reload
	sh	a0, 270(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1232(a0)                   # 8-byte Folded Reload
	sh	a0, 256(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1240(a0)                   # 8-byte Folded Reload
	sh	a0, 258(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1248(a0)                   # 8-byte Folded Reload
	sh	a0, 260(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1256(a0)                   # 8-byte Folded Reload
	sh	a0, 262(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1200(a0)                   # 8-byte Folded Reload
	sh	a0, 248(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1208(a0)                   # 8-byte Folded Reload
	sh	a0, 250(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1216(a0)                   # 8-byte Folded Reload
	sh	a0, 252(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1224(a0)                   # 8-byte Folded Reload
	sh	a0, 254(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1168(a0)                   # 8-byte Folded Reload
	sh	a0, 240(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1176(a0)                   # 8-byte Folded Reload
	sh	a0, 242(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1184(a0)                   # 8-byte Folded Reload
	sh	a0, 244(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1192(a0)                   # 8-byte Folded Reload
	sh	a0, 246(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1136(a0)                   # 8-byte Folded Reload
	sh	a0, 232(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1144(a0)                   # 8-byte Folded Reload
	sh	a0, 234(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1152(a0)                   # 8-byte Folded Reload
	sh	a0, 236(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1160(a0)                   # 8-byte Folded Reload
	sh	a0, 238(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1104(a0)                   # 8-byte Folded Reload
	sh	a0, 224(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1112(a0)                   # 8-byte Folded Reload
	sh	a0, 226(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1120(a0)                   # 8-byte Folded Reload
	sh	a0, 228(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1128(a0)                   # 8-byte Folded Reload
	sh	a0, 230(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1072(a0)                   # 8-byte Folded Reload
	sh	a0, 216(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1080(a0)                   # 8-byte Folded Reload
	sh	a0, 218(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1088(a0)                   # 8-byte Folded Reload
	sh	a0, 220(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1096(a0)                   # 8-byte Folded Reload
	sh	a0, 222(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1040(a0)                   # 8-byte Folded Reload
	sh	a0, 208(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1048(a0)                   # 8-byte Folded Reload
	sh	a0, 210(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1056(a0)                   # 8-byte Folded Reload
	sh	a0, 212(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1064(a0)                   # 8-byte Folded Reload
	sh	a0, 214(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1008(a0)                   # 8-byte Folded Reload
	sh	a0, 200(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1016(a0)                   # 8-byte Folded Reload
	sh	a0, 202(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1024(a0)                   # 8-byte Folded Reload
	sh	a0, 204(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1032(a0)                   # 8-byte Folded Reload
	sh	a0, 206(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -976(a0)                    # 8-byte Folded Reload
	sh	a0, 192(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -984(a0)                    # 8-byte Folded Reload
	sh	a0, 194(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -992(a0)                    # 8-byte Folded Reload
	sh	a0, 196(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -1000(a0)                   # 8-byte Folded Reload
	sh	a0, 198(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -944(a0)                    # 8-byte Folded Reload
	sh	a0, 184(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -952(a0)                    # 8-byte Folded Reload
	sh	a0, 186(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -960(a0)                    # 8-byte Folded Reload
	sh	a0, 188(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -968(a0)                    # 8-byte Folded Reload
	sh	a0, 190(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -912(a0)                    # 8-byte Folded Reload
	sh	a0, 176(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -920(a0)                    # 8-byte Folded Reload
	sh	a0, 178(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -928(a0)                    # 8-byte Folded Reload
	sh	a0, 180(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -936(a0)                    # 8-byte Folded Reload
	sh	a0, 182(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -880(a0)                    # 8-byte Folded Reload
	sh	a0, 168(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -888(a0)                    # 8-byte Folded Reload
	sh	a0, 170(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -896(a0)                    # 8-byte Folded Reload
	sh	a0, 172(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -904(a0)                    # 8-byte Folded Reload
	sh	a0, 174(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -848(a0)                    # 8-byte Folded Reload
	sh	a0, 160(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -856(a0)                    # 8-byte Folded Reload
	sh	a0, 162(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -864(a0)                    # 8-byte Folded Reload
	sh	a0, 164(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -872(a0)                    # 8-byte Folded Reload
	sh	a0, 166(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -816(a0)                    # 8-byte Folded Reload
	sh	a0, 152(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -824(a0)                    # 8-byte Folded Reload
	sh	a0, 154(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -832(a0)                    # 8-byte Folded Reload
	sh	a0, 156(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -840(a0)                    # 8-byte Folded Reload
	sh	a0, 158(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -784(a0)                    # 8-byte Folded Reload
	sh	a0, 144(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -792(a0)                    # 8-byte Folded Reload
	sh	a0, 146(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -800(a0)                    # 8-byte Folded Reload
	sh	a0, 148(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -808(a0)                    # 8-byte Folded Reload
	sh	a0, 150(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -752(a0)                    # 8-byte Folded Reload
	sh	a0, 136(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -760(a0)                    # 8-byte Folded Reload
	sh	a0, 138(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -768(a0)                    # 8-byte Folded Reload
	sh	a0, 140(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -776(a0)                    # 8-byte Folded Reload
	sh	a0, 142(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -720(a0)                    # 8-byte Folded Reload
	sh	a0, 128(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -728(a0)                    # 8-byte Folded Reload
	sh	a0, 130(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -736(a0)                    # 8-byte Folded Reload
	sh	a0, 132(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -744(a0)                    # 8-byte Folded Reload
	sh	a0, 134(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -688(a0)                    # 8-byte Folded Reload
	sh	a0, 120(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -696(a0)                    # 8-byte Folded Reload
	sh	a0, 122(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -704(a0)                    # 8-byte Folded Reload
	sh	a0, 124(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -712(a0)                    # 8-byte Folded Reload
	sh	a0, 126(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -656(a0)                    # 8-byte Folded Reload
	sh	a0, 112(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -664(a0)                    # 8-byte Folded Reload
	sh	a0, 114(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -672(a0)                    # 8-byte Folded Reload
	sh	a0, 116(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -680(a0)                    # 8-byte Folded Reload
	sh	a0, 118(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -624(a0)                    # 8-byte Folded Reload
	sh	a0, 104(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -632(a0)                    # 8-byte Folded Reload
	sh	a0, 106(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -640(a0)                    # 8-byte Folded Reload
	sh	a0, 108(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -648(a0)                    # 8-byte Folded Reload
	sh	a0, 110(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -592(a0)                    # 8-byte Folded Reload
	sh	a0, 96(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -600(a0)                    # 8-byte Folded Reload
	sh	a0, 98(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -608(a0)                    # 8-byte Folded Reload
	sh	a0, 100(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -616(a0)                    # 8-byte Folded Reload
	sh	a0, 102(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -560(a0)                    # 8-byte Folded Reload
	sh	a0, 88(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -568(a0)                    # 8-byte Folded Reload
	sh	a0, 90(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -576(a0)                    # 8-byte Folded Reload
	sh	a0, 92(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -584(a0)                    # 8-byte Folded Reload
	sh	a0, 94(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -528(a0)                    # 8-byte Folded Reload
	sh	a0, 80(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -536(a0)                    # 8-byte Folded Reload
	sh	a0, 82(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -544(a0)                    # 8-byte Folded Reload
	sh	a0, 84(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -552(a0)                    # 8-byte Folded Reload
	sh	a0, 86(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -496(a0)                    # 8-byte Folded Reload
	sh	a0, 72(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -504(a0)                    # 8-byte Folded Reload
	sh	a0, 74(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -512(a0)                    # 8-byte Folded Reload
	sh	a0, 76(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -520(a0)                    # 8-byte Folded Reload
	sh	a0, 78(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -464(a0)                    # 8-byte Folded Reload
	sh	a0, 64(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -472(a0)                    # 8-byte Folded Reload
	sh	a0, 66(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -480(a0)                    # 8-byte Folded Reload
	sh	a0, 68(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -488(a0)                    # 8-byte Folded Reload
	sh	a0, 70(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -432(a0)                    # 8-byte Folded Reload
	sh	a0, 56(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -440(a0)                    # 8-byte Folded Reload
	sh	a0, 58(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -448(a0)                    # 8-byte Folded Reload
	sh	a0, 60(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -456(a0)                    # 8-byte Folded Reload
	sh	a0, 62(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -400(a0)                    # 8-byte Folded Reload
	sh	a0, 48(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -408(a0)                    # 8-byte Folded Reload
	sh	a0, 50(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -416(a0)                    # 8-byte Folded Reload
	sh	a0, 52(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -424(a0)                    # 8-byte Folded Reload
	sh	a0, 54(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -368(a0)                    # 8-byte Folded Reload
	sh	a0, 40(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -376(a0)                    # 8-byte Folded Reload
	sh	a0, 42(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -384(a0)                    # 8-byte Folded Reload
	sh	a0, 44(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -392(a0)                    # 8-byte Folded Reload
	sh	a0, 46(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -336(a0)                    # 8-byte Folded Reload
	sh	a0, 32(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -344(a0)                    # 8-byte Folded Reload
	sh	a0, 34(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -352(a0)                    # 8-byte Folded Reload
	sh	a0, 36(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -360(a0)                    # 8-byte Folded Reload
	sh	a0, 38(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -304(a0)                    # 8-byte Folded Reload
	sh	a0, 24(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -312(a0)                    # 8-byte Folded Reload
	sh	a0, 26(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -320(a0)                    # 8-byte Folded Reload
	sh	a0, 28(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -328(a0)                    # 8-byte Folded Reload
	sh	a0, 30(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -272(a0)                    # 8-byte Folded Reload
	sh	a0, 16(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -280(a0)                    # 8-byte Folded Reload
	sh	a0, 18(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -288(a0)                    # 8-byte Folded Reload
	sh	a0, 20(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -296(a0)                    # 8-byte Folded Reload
	sh	a0, 22(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -240(a0)                    # 8-byte Folded Reload
	sh	a0, 8(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -248(a0)                    # 8-byte Folded Reload
	sh	a0, 10(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -256(a0)                    # 8-byte Folded Reload
	sh	a0, 12(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -264(a0)                    # 8-byte Folded Reload
	sh	a0, 14(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -208(a0)                    # 8-byte Folded Reload
	sh	a0, 0(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -216(a0)                    # 8-byte Folded Reload
	sh	a0, 2(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -224(a0)                    # 8-byte Folded Reload
	sh	a0, 4(a1)
	lui	a0, 1
	add	a0, sp, a0
	ld	a0, -232(a0)                    # 8-byte Folded Reload
	sh	a0, 6(a1)
	.loc	1 11 4 epilogue_begin is_stmt 0 # k135112023920016.py:11:4
	addi	sp, sp, 1968
	.cfi_def_cfa_offset 2032
	ld	ra, 2024(sp)                    # 8-byte Folded Reload
	ld	s0, 2016(sp)                    # 8-byte Folded Reload
	ld	s1, 2008(sp)                    # 8-byte Folded Reload
	ld	s2, 2000(sp)                    # 8-byte Folded Reload
	ld	s3, 1992(sp)                    # 8-byte Folded Reload
	ld	s4, 1984(sp)                    # 8-byte Folded Reload
	ld	s5, 1976(sp)                    # 8-byte Folded Reload
	ld	s6, 1968(sp)                    # 8-byte Folded Reload
	ld	s7, 1960(sp)                    # 8-byte Folded Reload
	ld	s8, 1952(sp)                    # 8-byte Folded Reload
	ld	s9, 1944(sp)                    # 8-byte Folded Reload
	ld	s10, 1936(sp)                   # 8-byte Folded Reload
	ld	s11, 1928(sp)                   # 8-byte Folded Reload
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
	addi	sp, sp, 2032
	.cfi_def_cfa_offset 0
	ret
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8, .Lfunc_end0-triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_8
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
	.asciz	"k135112023920016.py"           # string offset=7 ; k135112023920016.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
