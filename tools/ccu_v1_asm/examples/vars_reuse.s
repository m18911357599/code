# Variable-aware CCU V1 asm: named resources + auto ID alloc/reuse
#
# Declarations (optional if auto-declared on first use):
#   .xn name
#   .gsa name
#   .ms name
#   .cke name
#   .ch name
#   .sqe name
#   .var xn name = 3     # pinned physical id
#
# Identifiers in resource operand fields are allocated; numbers stay literals.

# Explicit decls
.xn phase_a
.xn phase_b
.gsa src_gsa
.ms  slice0
.cke sync_done
.ch  peer0

# phase_a live only in first block
LOAD_IMD_TO_XN xn=phase_a, imm=0x1000, sec=0
LOAD_XX xd=phase_a, xm=phase_a, xn=phase_a

# phase_b starts after phase_a dies → should reuse same xn id
LOAD_IMD_TO_XN xn=phase_b, imm=0x2000, sec=0
LOAD_SQEARGS_TO_GSA gsa=src_gsa, sqe=0
TRANS_LOC_MEM_TO_LOC_MS ms=slice0, gsa=src_gsa, xn=phase_b, len_xn=phase_b, ch=peer0, clear=0, len_en=1, set_id=sync_done, set_mask=0x1, wait_id=0, wait_mask=0
SET_CKE clear=0, set_id=sync_done, set_mask=0xffff, wait_id=0, wait_mask=0

# Auto-declared on first use (no .decl): tmp_xn, tmp_gsa
LOAD_IMD_TO_GSA gsa=tmp_gsa, imm=0xabcdef00
LOAD_GSA_XN gsad=tmp_gsa, gsam=tmp_gsa, xn=tmp_xn
