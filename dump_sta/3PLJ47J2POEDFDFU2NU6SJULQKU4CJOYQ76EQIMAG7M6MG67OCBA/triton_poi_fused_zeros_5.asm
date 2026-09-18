	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused_zeros_5        # -- Begin function triton_poi_fused_zeros_5
	.p2align	2
	.type	triton_poi_fused_zeros_5,@function
triton_poi_fused_zeros_5:               # @triton_poi_fused_zeros_5
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294389936.py"
	.loc	1 4 33 prologue_end             # k135114294389936.py:4:33
	slliw	a1, a2, 7
	.loc	1 9 25                          # k135114294389936.py:9:25
	slli	a1, a1, 1
	add	a0, a0, a1
	.loc	1 9 36 is_stmt 0                # k135114294389936.py:9:36
	sh	zero, 248(a0)
	sh	zero, 250(a0)
	sh	zero, 252(a0)
	sh	zero, 254(a0)
	sh	zero, 240(a0)
	sh	zero, 242(a0)
	sh	zero, 244(a0)
	sh	zero, 246(a0)
	sh	zero, 232(a0)
	sh	zero, 234(a0)
	sh	zero, 236(a0)
	sh	zero, 238(a0)
	sh	zero, 224(a0)
	sh	zero, 226(a0)
	sh	zero, 228(a0)
	sh	zero, 230(a0)
	sh	zero, 216(a0)
	sh	zero, 218(a0)
	sh	zero, 220(a0)
	sh	zero, 222(a0)
	sh	zero, 208(a0)
	sh	zero, 210(a0)
	sh	zero, 212(a0)
	sh	zero, 214(a0)
	sh	zero, 200(a0)
	sh	zero, 202(a0)
	sh	zero, 204(a0)
	sh	zero, 206(a0)
	sh	zero, 192(a0)
	sh	zero, 194(a0)
	sh	zero, 196(a0)
	sh	zero, 198(a0)
	sh	zero, 184(a0)
	sh	zero, 186(a0)
	sh	zero, 188(a0)
	sh	zero, 190(a0)
	sh	zero, 176(a0)
	sh	zero, 178(a0)
	sh	zero, 180(a0)
	sh	zero, 182(a0)
	sh	zero, 168(a0)
	sh	zero, 170(a0)
	sh	zero, 172(a0)
	sh	zero, 174(a0)
	sh	zero, 160(a0)
	sh	zero, 162(a0)
	sh	zero, 164(a0)
	sh	zero, 166(a0)
	sh	zero, 152(a0)
	sh	zero, 154(a0)
	sh	zero, 156(a0)
	sh	zero, 158(a0)
	sh	zero, 144(a0)
	sh	zero, 146(a0)
	sh	zero, 148(a0)
	sh	zero, 150(a0)
	sh	zero, 136(a0)
	sh	zero, 138(a0)
	sh	zero, 140(a0)
	sh	zero, 142(a0)
	sh	zero, 128(a0)
	sh	zero, 130(a0)
	sh	zero, 132(a0)
	sh	zero, 134(a0)
	sh	zero, 120(a0)
	sh	zero, 122(a0)
	sh	zero, 124(a0)
	sh	zero, 126(a0)
	sh	zero, 112(a0)
	sh	zero, 114(a0)
	sh	zero, 116(a0)
	sh	zero, 118(a0)
	sh	zero, 104(a0)
	sh	zero, 106(a0)
	sh	zero, 108(a0)
	sh	zero, 110(a0)
	sh	zero, 96(a0)
	sh	zero, 98(a0)
	sh	zero, 100(a0)
	sh	zero, 102(a0)
	sh	zero, 88(a0)
	sh	zero, 90(a0)
	sh	zero, 92(a0)
	sh	zero, 94(a0)
	sh	zero, 80(a0)
	sh	zero, 82(a0)
	sh	zero, 84(a0)
	sh	zero, 86(a0)
	sh	zero, 72(a0)
	sh	zero, 74(a0)
	sh	zero, 76(a0)
	sh	zero, 78(a0)
	sh	zero, 64(a0)
	sh	zero, 66(a0)
	sh	zero, 68(a0)
	sh	zero, 70(a0)
	sh	zero, 56(a0)
	sh	zero, 58(a0)
	sh	zero, 60(a0)
	sh	zero, 62(a0)
	sh	zero, 48(a0)
	sh	zero, 50(a0)
	sh	zero, 52(a0)
	sh	zero, 54(a0)
	sh	zero, 40(a0)
	sh	zero, 42(a0)
	sh	zero, 44(a0)
	sh	zero, 46(a0)
	sh	zero, 32(a0)
	sh	zero, 34(a0)
	sh	zero, 36(a0)
	sh	zero, 38(a0)
	sh	zero, 24(a0)
	sh	zero, 26(a0)
	sh	zero, 28(a0)
	sh	zero, 30(a0)
	sh	zero, 16(a0)
	sh	zero, 18(a0)
	sh	zero, 20(a0)
	sh	zero, 22(a0)
	sh	zero, 8(a0)
	sh	zero, 10(a0)
	sh	zero, 12(a0)
	sh	zero, 14(a0)
	sh	zero, 0(a0)
	sh	zero, 2(a0)
	sh	zero, 4(a0)
	sh	zero, 6(a0)
	.loc	1 9 4                           # k135114294389936.py:9:4
	ret
.Ltmp0:
.Lfunc_end0:
	.size	triton_poi_fused_zeros_5, .Lfunc_end0-triton_poi_fused_zeros_5
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
	.asciz	"k135114294389936.py"           # string offset=7 ; k135114294389936.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
