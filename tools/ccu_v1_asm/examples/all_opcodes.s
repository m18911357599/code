# CCU V1 sample — positional operands (no field= prefixes)
# Order is canonical per mnemonic; see isa.c operand tables.

# ---- LOAD ----
LOAD_SQEARGS_TO_GSA 1, 2
LOAD_SQEARGS_TO_XN 3, 4
LOAD_IMD_TO_GSA 5, 0x1122334455667788
LOAD_IMD_TO_XN 6, 0x1000, 0
LOAD_GSA_XN 7, 8, 9
LOAD_GSA_GSA 1, 2, 3
LOAD_XX 4, 5, 6

# ---- CTRL ----
LOOP 0, 10, 11
LOOP_GROUP 0, 12, 13, 1
SET_CKE 0, 1, 0xffff, 2, 0x0001
CLEAR_CKE 1, 3, 0x00ff, 4, 0x0002
JMP 14, 15, 0xabcdef

# ---- TRANS ----
TRANS_LOC_MEM_TO_LOC_MS 0, 1, 2, 3, 0, 0, 1, 1, 0x1, 2, 0x2
TRANS_RMT_MEM_TO_LOC_MS 1, 2, 3, 4, 1, 1, 0, 3, 0x3, 4, 0x4
TRANS_LOC_MS_TO_LOC_MEM 5, 6, 7, 8, 2, 0, 1, 5, 0x5, 6, 0x6
TRANS_LOC_MS_TO_RMT_MEM 1, 2, 3, 4, 3, 0, 1, 0, 0, 0, 0
TRANS_RMT_MS_TO_LOC_MEM 2, 3, 4, 5, 4, 1, 1, 7, 0x7, 8, 0x8
TRANS_LOC_MS_TO_LOC_MS 1, 2, 3, 5, 0, 1, 9, 0x9, 10, 0xa
TRANS_RMT_MS_TO_LOC_MS 3, 4, 5, 6, 0, 0, 0, 0, 0, 0
TRANS_LOC_MS_TO_RMT_MS 5, 6, 7, 7, 1, 0xb, 0, 1, 2, 0xc, 3, 0xd
TRANS_RMT_MEM_TO_LOC_MEM 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
TRANS_LOC_MEM_TO_RMT_MEM 2, 3, 4, 5, 6, 1, 0xab, 0xc, 0x5, 1, 1, 1, 4, 0xe, 5, 0xf
TRANS_LOC_MEM_TO_LOC_MEM 1, 2, 3, 4, 5, 2, 0, 1, 6, 0x10, 7, 0x11

# ---- SYNC ----
SYNC_CKE 1, 2, 0xffff, 0, 0, 1, 0x1, 2, 0x2
SYNC_GSA 3, 4, 1, 5, 0x20, 1, 6, 0x21, 7, 0x22
SYNC_XN 8, 9, 2, 10, 0x30, 0, 11, 0x31, 12, 0x32

# ---- REDUCE ----
ADD [0,1,2,0,0,0,0,0], 3, 1, 4, 13, 0, 1, 0x1, 2, 0x2
MAX [4,5,6,7,0,0,0,0], 4, 2, 14, 1, 3, 0x3, 4, 0x4
MIN [8,9,10,0,0,0,0,0], 3, 1, 15, 0, 5, 0x5, 6, 0x6
