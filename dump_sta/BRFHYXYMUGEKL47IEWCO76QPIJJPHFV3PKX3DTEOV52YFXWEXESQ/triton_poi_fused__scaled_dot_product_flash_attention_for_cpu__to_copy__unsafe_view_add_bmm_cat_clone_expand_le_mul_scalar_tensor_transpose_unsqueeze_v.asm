	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_10 # -- Begin function triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_10
	.p2align	2
	.type	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_10,@function
triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_10: # @triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_10
.Lfunc_begin0:
	.file	1 "/home/chlee/qwen_triton_engine/rebuilt_kernels" "k135114294381680.py"
	.loc	1 2 0                           # k135114294381680.py:2:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	csrr	a1, vlenb
	li	a3, 42
	mul	a1, a1, a3
	sub	sp, sp, a1
	.cfi_escape 0x0f, 0x0d, 0x72, 0x00, 0x11, 0x10, 0x22, 0x11, 0x2a, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 16 + 42 * vlenb
.Ltmp0:
	.loc	1 4 33 prologue_end             # k135114294381680.py:4:33
	slliw	a1, a2, 7
	li	a4, 32
	li	a3, 96
	li	a2, -128
	.loc	1 5 23                          # k135114294381680.py:5:23
	vsetvli	zero, a4, e32, m8, ta, ma
	vmv.v.x	v8, a1
	vid.v	v16
	vadd.vx	v16, v16, a3
	.loc	1 8 19                          # k135114294381680.py:8:19
	vsra.vi	v8, v8, 31
	.loc	1 5 23                          # k135114294381680.py:5:23
	vor.vx	v24, v16, a1
	csrr	a3, vlenb
	slli	a5, a3, 5
	add	a3, a5, a3
	add	a3, sp, a3
	addi	a3, a3, 16
	vs8r.v	v24, (a3)                       # vscale x 64-byte Folded Spill
	.loc	1 8 19                          # k135114294381680.py:8:19
	vsrl.vi	v0, v8, 25
	vadd.vv	v8, v24, v0
	vsra.vi	v16, v8, 7
	.loc	1 7 19                          # k135114294381680.py:7:19
	vand.vx	v8, v8, a2
	vsub.vv	v8, v24, v8
	.loc	1 14 19                         # k135114294381680.py:14:19
	vmslt.vv	v24, v16, v8
	csrr	a3, vlenb
	li	a5, 41
	mul	a3, a3, a5
	add	a3, sp, a3
	addi	a3, a3, 16
	vs1r.v	v24, (a3)                       # vscale x 8-byte Folded Spill
	li	a3, 64
	vid.v	v16
	.loc	1 5 23                          # k135114294381680.py:5:23
	vadd.vx	v8, v16, a3
	vor.vx	v24, v8, a1
	csrr	a5, vlenb
	li	a6, 24
	mul	a5, a5, a6
	add	a5, sp, a5
	addi	a5, a5, 16
	vs8r.v	v24, (a5)                       # vscale x 64-byte Folded Spill
	.loc	1 8 19                          # k135114294381680.py:8:19
	vadd.vv	v8, v24, v0
	vsra.vi	v16, v8, 7
	csrr	a5, vlenb
	slli	a5, a5, 4
	add	a5, sp, a5
	addi	a5, a5, 16
	vs8r.v	v16, (a5)                       # vscale x 64-byte Folded Spill
	.loc	1 7 19                          # k135114294381680.py:7:19
	vand.vx	v8, v8, a2
	vsub.vv	v8, v24, v8
	csrr	a5, vlenb
	slli	a5, a5, 4
	add	a5, sp, a5
	addi	a5, a5, 16
	vl8r.v	v16, (a5)                       # vscale x 64-byte Folded Reload
	.loc	1 14 19                         # k135114294381680.py:14:19
	vmslt.vv	v24, v16, v8
	csrr	a5, vlenb
	slli	a5, a5, 5
	add	a5, sp, a5
	addi	a5, a5, 16
	vs1r.v	v24, (a5)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294381680.py:5:23
	vid.v	v8
	vadd.vx	v8, v8, a4
	vor.vx	v16, v8, a1
	addi	a4, sp, 16
	vs8r.v	v16, (a4)                       # vscale x 64-byte Folded Spill
	.loc	1 8 19                          # k135114294381680.py:8:19
	vadd.vv	v8, v16, v0
	vsra.vi	v24, v8, 7
	csrr	a4, vlenb
	slli	a4, a4, 3
	add	a4, sp, a4
	addi	a4, a4, 16
	vs8r.v	v24, (a4)                       # vscale x 64-byte Folded Spill
	.loc	1 7 19                          # k135114294381680.py:7:19
	vand.vx	v8, v8, a2
	vsub.vv	v8, v16, v8
	csrr	a4, vlenb
	slli	a4, a4, 3
	add	a4, sp, a4
	addi	a4, a4, 16
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 14 19                         # k135114294381680.py:14:19
	vmslt.vv	v24, v16, v8
	csrr	a4, vlenb
	slli	a4, a4, 4
	add	a4, sp, a4
	addi	a4, a4, 16
	vs1r.v	v24, (a4)                       # vscale x 8-byte Folded Spill
	.loc	1 5 23                          # k135114294381680.py:5:23
	vid.v	v8
	vor.vx	v8, v8, a1
	.loc	1 8 19                          # k135114294381680.py:8:19
	vadd.vv	v0, v8, v0
	vsra.vi	v24, v0, 7
	csrr	a4, vlenb
	slli	a4, a4, 3
	add	a4, sp, a4
	addi	a4, a4, 16
	vs8r.v	v24, (a4)                       # vscale x 64-byte Folded Spill
	.loc	1 7 19                          # k135114294381680.py:7:19
	vand.vx	v0, v0, a2
	vsub.vv	v24, v8, v0
	csrr	a4, vlenb
	slli	a4, a4, 3
	add	a4, sp, a4
	addi	a4, a4, 16
	vl8r.v	v16, (a4)                       # vscale x 64-byte Folded Reload
	.loc	1 14 19                         # k135114294381680.py:14:19
	vmslt.vv	v0, v16, v24
	li	a4, 1792
	.loc	1 6 21                          # k135114294381680.py:6:21
	vmslt.vx	v7, v8, a4
	csrr	a5, vlenb
	slli	a6, a5, 5
	add	a5, a6, a5
	add	a5, sp, a5
	addi	a5, a5, 16
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	vmslt.vx	v24, v8, a4
	csrr	a5, vlenb
	li	a6, 24
	mul	a5, a5, a6
	add	a5, sp, a5
	addi	a5, a5, 16
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	vmslt.vx	v6, v8, a4
	addi	a5, sp, 16
	vl8r.v	v8, (a5)                        # vscale x 64-byte Folded Reload
	vmslt.vx	v25, v8, a4
	csrr	a4, vlenb
	slli	a4, a4, 4
	add	a4, sp, a4
	addi	a4, a4, 16
	vl1r.v	v8, (a4)                        # vscale x 8-byte Folded Reload
	.loc	1 14 19                         # k135114294381680.py:14:19
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v0, v8, 4
	.loc	1 17 32                         # k135114294381680.py:17:32
	vsetvli	zero, a3, e16, m8, ta, ma
	vmv.v.i	v16, 0
	vmerge.vxm	v8, v16, a2, v0
	csrr	a4, vlenb
	li	a5, 41
	mul	a4, a4, a5
	add	a4, sp, a4
	addi	a4, a4, 16
	vl1r.v	v26, (a4)                       # vscale x 8-byte Folded Reload
	csrr	a4, vlenb
	slli	a4, a4, 5
	add	a4, sp, a4
	addi	a4, a4, 16
	vl1r.v	v0, (a4)                        # vscale x 8-byte Folded Reload
	.loc	1 14 19                         # k135114294381680.py:14:19
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v0, v26, 4
	.loc	1 6 21                          # k135114294381680.py:6:21
	vslideup.vi	v6, v24, 4
	.loc	1 17 32                         # k135114294381680.py:17:32
	vsetvli	zero, a3, e16, m8, ta, ma
	vmerge.vxm	v16, v16, a2, v0
	.loc	1 6 21                          # k135114294381680.py:6:21
	vsetivli	zero, 8, e8, mf2, ta, ma
	vslideup.vi	v7, v25, 4
	.loc	1 18 25                         # k135114294381680.py:18:25
	slli	a1, a1, 1
	add	a0, a0, a1
	.loc	1 18 36 is_stmt 0               # k135114294381680.py:18:36
	addi	a1, a0, 128
	vmv1r.v	v0, v6
	vsetvli	zero, a3, e16, m8, ta, ma
	vse16.v	v16, (a1), v0.t
	vmv1r.v	v0, v7
	vse16.v	v8, (a0), v0.t
	.loc	1 18 4 epilogue_begin           # k135114294381680.py:18:4
	csrr	a0, vlenb
	li	a1, 42
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 16
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Ltmp1:
.Lfunc_end0:
	.size	triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_10, .Lfunc_end0-triton_poi_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_bmm_cat_clone_expand_le_mul_scalar_tensor_transpose_unsqueeze_view_where_10
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
	.asciz	"k135114294381680.py"           # string offset=7 ; k135114294381680.py
.Linfo_string2:
	.asciz	"/home/chlee/qwen_triton_engine/rebuilt_kernels" # string offset=27 ; /home/chlee/qwen_triton_engine/rebuilt_kernels
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
