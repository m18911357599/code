# Variable-aware CCU V1 asm — positional operands (no field=)
#
# Declarations:
#   .xn name | .gsa name | .ms name | .cke name | .ch name | .sqe name
#   .var xn name = 3     # pinned id
#
# Instruction form:
#   LOAD_IMD_TO_XN phase_b, 0x2000, 0

.xn phase_a
.xn phase_b
.gsa src_gsa
.ms  slice0
.cke sync_done
.ch  peer0

# phase_a live only in first block
LOAD_IMD_TO_XN phase_a, 0x1000, 0
LOAD_XX phase_a, phase_a, phase_a

# phase_b starts after phase_a dies → reuse same xn id
LOAD_IMD_TO_XN phase_b, 0x2000, 0
LOAD_SQEARGS_TO_GSA src_gsa, 0
TRANS_LOC_MEM_TO_LOC_MS slice0, src_gsa, phase_b, phase_b, peer0, 0, 1, sync_done, 0x1, 0, 0
SET_CKE 0, sync_done, 0xffff, 0, 0

# Auto-declared on first use: tmp_xn, tmp_gsa
LOAD_IMD_TO_GSA tmp_gsa, 0xabcdef00
LOAD_GSA_XN tmp_gsa, tmp_gsa, tmp_xn
