# CCU V1 sample covering every opcode (semantic round-trip fixture)
# Operand names are canonical; comments and operand order on input may vary.

# ---- LOAD ----
LOAD_SQEARGS_TO_GSA gsa=1, sqe=2
LOAD_SQEARGS_TO_XN xn=3, sqe=4
LOAD_IMD_TO_GSA gsa=5, imm=0x1122334455667788
LOAD_IMD_TO_XN xn=6, imm=0x1000, sec=0
LOAD_GSA_XN gsad=7, gsam=8, xn=9
LOAD_GSA_GSA gsad=1, gsam=2, gsan=3
LOAD_XX xd=4, xm=5, xn=6

# ---- CTRL ----
LOOP start=0, end=10, xn=11
LOOP_GROUP start_loop=0, xn=12, xm=13, hiperf=1
SET_CKE clear=0, set_id=1, set_mask=0xffff, wait_id=2, wait_mask=0x0001
CLEAR_CKE clear=1, clear_id=3, clear_mask=0x00ff, wait_id=4, wait_mask=0x0002
JMP dst_xn=14, cond_xn=15, expect=0xabcdef

# ---- TRANS ----
TRANS_LOC_MEM_TO_LOC_MS ms=0, gsa=1, xn=2, len_xn=3, ch=0, clear=0, len_en=1, set_id=1, set_mask=0x1, wait_id=2, wait_mask=0x2
TRANS_RMT_MEM_TO_LOC_MS ms=1, gsa=2, xn=3, len_xn=4, ch=1, clear=1, len_en=0, set_id=3, set_mask=0x3, wait_id=4, wait_mask=0x4
TRANS_LOC_MS_TO_LOC_MEM gsa=5, xn=6, ms=7, len_xn=8, ch=2, clear=0, len_en=1, set_id=5, set_mask=0x5, wait_id=6, wait_mask=0x6
TRANS_LOC_MS_TO_RMT_MEM gsa=1, xn=2, ms=3, len_xn=4, ch=3, clear=0, len_en=1, set_id=0, set_mask=0, wait_id=0, wait_mask=0
TRANS_RMT_MS_TO_LOC_MEM gsa=2, xn=3, ms=4, len_xn=5, ch=4, clear=1, len_en=1, set_id=7, set_mask=0x7, wait_id=8, wait_mask=0x8
TRANS_LOC_MS_TO_LOC_MS dst_ms=1, src_ms=2, len_xn=3, ch=5, clear=0, len_en=1, set_id=9, set_mask=0x9, wait_id=10, wait_mask=0xa
TRANS_RMT_MS_TO_LOC_MS loc_ms=3, rmt_ms=4, len_xn=5, ch=6, clear=0, len_en=0, set_id=0, set_mask=0, wait_id=0, wait_mask=0
TRANS_LOC_MS_TO_RMT_MS rmt_ms=5, loc_ms=6, len_xn=7, ch=7, rmt_set_id=1, rmt_set_mask=0xb, clear=0, len_en=1, set_id=2, set_mask=0xc, wait_id=3, wait_mask=0xd
TRANS_RMT_MEM_TO_LOC_MEM loc_gsa=1, loc_xn=2, rmt_gsa=3, rmt_xn=4, len_xn=5, ch=0, udf=0, reduce_dtype=0, reduce_op=0, clear=0, len_en=1, reduce_en=0, set_id=0, set_mask=0, wait_id=0, wait_mask=0
TRANS_LOC_MEM_TO_RMT_MEM rmt_gsa=2, rmt_xn=3, loc_gsa=4, loc_xn=5, len_xn=6, ch=1, udf=0xab, reduce_dtype=0xc, reduce_op=0x5, clear=1, len_en=1, reduce_en=1, set_id=4, set_mask=0xe, wait_id=5, wait_mask=0xf
TRANS_LOC_MEM_TO_LOC_MEM dst_gsa=1, dst_xn=2, src_gsa=3, src_xn=4, len_xn=5, ch=2, clear=0, len_en=1, set_id=6, set_mask=0x10, wait_id=7, wait_mask=0x11

# ---- SYNC ----
SYNC_CKE rmt_cke=1, loc_cke=2, loc_mask=0xffff, ch=0, clear=0, set_id=1, set_mask=0x1, wait_id=2, wait_mask=0x2
SYNC_GSA rmt_gsa=3, loc_gsa=4, ch=1, rmt_set_id=5, rmt_set_mask=0x20, clear=1, set_id=6, set_mask=0x21, wait_id=7, wait_mask=0x22
SYNC_XN rmt_xn=8, loc_xn=9, ch=2, rmt_set_id=10, rmt_set_mask=0x30, clear=0, set_id=11, set_mask=0x31, wait_id=12, wait_mask=0x32

# ---- REDUCE ----
ADD ms=[0,1,2,0,0,0,0,0], count=3, cast=1, dtype=4, len_xn=13, clear=0, set_id=1, set_mask=0x1, wait_id=2, wait_mask=0x2
MAX ms=[4,5,6,7,0,0,0,0], count=4, dtype=2, len_xn=14, clear=1, set_id=3, set_mask=0x3, wait_id=4, wait_mask=0x4
MIN ms=[8,9,10,0,0,0,0,0], count=3, dtype=1, len_xn=15, clear=0, set_id=5, set_mask=0x5, wait_id=6, wait_mask=0x6
