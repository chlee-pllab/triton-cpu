	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_9 # -- Begin function triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_9
	.p2align	2
	.type	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_9,@function
triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_9: # @triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_9
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135112023918768.py"
	.loc	1 2 0                           # k135112023918768.py:2:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	csrr	a2, vlenb
	li	a4, 29
	mul	a2, a2, a4
	sub	sp, sp, a2
	.cfi_escape 0x0f, 0x0d, 0x72, 0x00, 0x11, 0x10, 0x22, 0x11, 0x1d, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 16 + 29 * vlenb
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135112023918768.py:4:33
	slliw	a5, a3, 7
	li	a4, 32
	li	a2, 64
	li	a3, 128
	.loc	1 8 19                          # k135112023918768.py:8:19
	lwu	a6, 0(a0)
	lw	a0, 4(a0)
	.loc	1 5 23                          # k135112023918768.py:5:23
	vsetvli	zero, a4, e32, m8, ta, ma
	vid.v	v8
	vor.vx	v24, v8, a5
	vadd.vx	v16, v8, a4
	vor.vx	v16, v16, a5
	.loc	1 12 19                         # k135112023918768.py:12:19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v0, v24, 16
	.loc	1 8 19                          # k135112023918768.py:8:19
	slli	a0, a0, 32
	.loc	1 12 19                         # k135112023918768.py:12:19
	vsetivli	zero, 16, e64, m8, ta, ma
	vsext.vf2	v8, v0
	.loc	1 8 19                          # k135112023918768.py:8:19
	or	a0, a0, a6
	.loc	1 14 19                         # k135112023918768.py:14:19
	vmsgt.vx	v7, v8, a0
	csrr	a6, vlenb
	slli	a7, a6, 1
	add	a6, a7, a6
	add	a6, sp, a6
	addi	a6, a6, 16
	vs1r.v	v7, (a6)                        # vscale x 8-byte Folded Spill
	.loc	1 6 21                          # k135112023918768.py:6:21
	vsetvli	zero, a4, e32, m8, ta, ma
	vmslt.vx	v8, v24, a3
	csrr	a6, vlenb
	li	a7, 28
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 16
	vs1r.v	v8, (a6)                        # vscale x 8-byte Folded Spill
	.loc	1 12 19                         # k135112023918768.py:12:19
	vsetivli	zero, 16, e64, m8, ta, ma
	vsext.vf2	v8, v24
	.loc	1 14 19                         # k135112023918768.py:14:19
	vmsgt.vx	v24, v8, a0
	csrr	a6, vlenb
	add	a6, sp, a6
	addi	a6, a6, 16
	vs1r.v	v24, (a6)                       # vscale x 8-byte Folded Spill
	csrr	a6, vlenb
	li	a7, 12
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 16
	vs8r.v	v16, (a6)                       # vscale x 64-byte Folded Spill
	.loc	1 12 19                         # k135112023918768.py:12:19
	vsext.vf2	v8, v16
	.loc	1 14 19                         # k135112023918768.py:14:19
	vmsgt.vx	v24, v8, a0
	csrr	a6, vlenb
	slli	a6, a6, 1
	add	a6, sp, a6
	addi	a6, a6, 16
	vs1r.v	v24, (a6)                       # vscale x 8-byte Folded Spill
	.loc	1 12 19                         # k135112023918768.py:12:19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v16, 16
	vsetivli	zero, 16, e64, m8, ta, ma
	vsext.vf2	v24, v8
	.loc	1 14 19                         # k135112023918768.py:14:19
	vmsgt.vx	v8, v24, a0
	csrr	a6, vlenb
	li	a7, 11
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 16
	vs1r.v	v8, (a6)                        # vscale x 8-byte Folded Spill
	vsetvli	zero, a4, e32, m8, ta, ma
	vid.v	v24
	.loc	1 5 23                          # k135112023918768.py:5:23
	vadd.vx	v8, v24, a2
	vor.vx	v16, v8, a5
	csrr	a6, vlenb
	li	a7, 20
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 16
	vs8r.v	v16, (a6)                       # vscale x 64-byte Folded Spill
	.loc	1 12 19                         # k135112023918768.py:12:19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v8, v16, 16
	vsetivli	zero, 16, e64, m8, ta, ma
	vsext.vf2	v0, v8
	.loc	1 14 19                         # k135112023918768.py:14:19
	vmsgt.vx	v8, v0, a0
	addi	a6, sp, 16
	vs1r.v	v8, (a6)                        # vscale x 8-byte Folded Spill
	.loc	1 12 19                         # k135112023918768.py:12:19
	vsext.vf2	v8, v16
	.loc	1 14 19                         # k135112023918768.py:14:19
	vmsgt.vx	v0, v8, a0
	li	a6, 96
	.loc	1 5 23                          # k135112023918768.py:5:23
	vsetvli	zero, a4, e32, m8, ta, ma
	vadd.vx	v8, v24, a6
	vor.vx	v8, v8, a5
	.loc	1 12 19                         # k135112023918768.py:12:19
	vsetivli	zero, 16, e64, m8, ta, ma
	vsext.vf2	v16, v8
	.loc	1 14 19                         # k135112023918768.py:14:19
	vmsgt.vx	v7, v16, a0
	.loc	1 12 19                         # k135112023918768.py:12:19
	vsetivli	zero, 16, e32, m8, ta, ma
	vslidedown.vi	v16, v8, 16
	vsetivli	zero, 16, e64, m8, ta, ma
	vsext.vf2	v24, v16
	.loc	1 14 19                         # k135112023918768.py:14:19
	vmsgt.vx	v16, v24, a0
	csrr	a0, vlenb
	add	a0, sp, a0
	addi	a0, a0, 16
	vl1r.v	v6, (a0)                        # vscale x 8-byte Folded Reload
	csrr	a0, vlenb
	slli	a6, a0, 1
	add	a0, a6, a0
	add	a0, sp, a0
	addi	a0, a0, 16
	vl1r.v	v17, (a0)                       # vscale x 8-byte Folded Reload
	vsetivli	zero, 4, e8, mf2, tu, ma
	vslideup.vi	v6, v17, 2
	li	a0, -128
	addi	a6, sp, 16
	vl1r.v	v17, (a6)                       # vscale x 8-byte Folded Reload
	vslideup.vi	v0, v17, 2
	csrr	a6, vlenb
	slli	a6, a6, 1
	add	a6, sp, a6
	addi	a6, a6, 16
	vl1r.v	v17, (a6)                       # vscale x 8-byte Folded Reload
	vsetivli	zero, 6, e8, mf2, tu, ma
	vslideup.vi	v6, v17, 4
	vslideup.vi	v0, v7, 4
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v0, v16, 6
	.loc	1 17 32                         # k135112023918768.py:17:32
	vsetvli	zero, a2, e16, m8, ta, ma
	vmv.v.i	v16, 0
	vmerge.vxm	v24, v16, a0, v0
	csrr	a6, vlenb
	slli	a7, a6, 1
	add	a6, a7, a6
	add	a6, sp, a6
	addi	a6, a6, 16
	vs8r.v	v24, (a6)                       # vscale x 64-byte Folded Spill
	csrr	a6, vlenb
	li	a7, 11
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 16
	vl1r.v	v25, (a6)                       # vscale x 8-byte Folded Reload
	.loc	1 14 19                         # k135112023918768.py:14:19
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v6, v25, 6
	vmv1r.v	v0, v6
	csrr	a6, vlenb
	li	a7, 12
	mul	a6, a6, a7
	add	a6, sp, a6
	addi	a6, a6, 16
	vl8r.v	v24, (a6)                       # vscale x 64-byte Folded Reload
	.loc	1 6 21                          # k135112023918768.py:6:21
	vsetvli	zero, a4, e32, m8, ta, ma
	vmslt.vx	v7, v24, a3
	.loc	1 17 32                         # k135112023918768.py:17:32
	vsetvli	zero, a2, e16, m8, ta, ma
	vmerge.vxm	v16, v16, a0, v0
	csrr	a0, vlenb
	li	a6, 28
	mul	a0, a0, a6
	add	a0, sp, a0
	addi	a0, a0, 16
	vl1r.v	v0, (a0)                        # vscale x 8-byte Folded Reload
	.loc	1 6 21                          # k135112023918768.py:6:21
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v0, v7, 4
	.loc	1 18 25                         # k135112023918768.py:18:25
	slli	a5, a5, 1
	add	a1, a1, a5
	.loc	1 18 36 is_stmt 0               # k135112023918768.py:18:36
	vsetvli	zero, a2, e16, m8, ta, ma
	vse16.v	v16, (a1), v0.t
	.loc	1 6 21 is_stmt 1                # k135112023918768.py:6:21
	vsetvli	zero, a4, e32, m8, ta, ma
	vmslt.vx	v16, v8, a3
	csrr	a0, vlenb
	li	a4, 20
	mul	a0, a0, a4
	add	a0, sp, a0
	addi	a0, a0, 16
	vl8r.v	v8, (a0)                        # vscale x 64-byte Folded Reload
	vmslt.vx	v0, v8, a3
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v0, v16, 4
	.loc	1 18 36                         # k135112023918768.py:18:36
	addi	a0, a1, 128
	csrr	a1, vlenb
	slli	a3, a1, 1
	add	a1, a3, a1
	add	a1, sp, a1
	addi	a1, a1, 16
	vl8r.v	v8, (a1)                        # vscale x 64-byte Folded Reload
	vsetvli	zero, a2, e16, m8, ta, ma
	vse16.v	v8, (a0), v0.t
	.loc	1 18 4 epilogue_begin is_stmt 0 # k135112023918768.py:18:4
	csrr	a0, vlenb
	li	a1, 29
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 16
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_9, .Lfunc_end0-triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_9
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
	.asciz	"k135112023918768.py"           # string offset=7 ; k135112023918768.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
