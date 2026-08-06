#include "ccu_v1_isa.h"

#include <ctype.h>
#include <string.h>

/* Positional operand orders (canonical). Source form: MNEMONIC v0, v1, ... */
static const char *const k_ops_load_sqe_gsa[] = {"gsa", "sqe", NULL};
static const char *const k_ops_load_sqe_xn[] = {"xn", "sqe", NULL};
static const char *const k_ops_load_imd_gsa[] = {"gsa", "imm", NULL};
static const char *const k_ops_load_imd_xn[] = {"xn", "imm", "sec", NULL};
static const char *const k_ops_load_gsa_xn[] = {"gsad", "gsam", "xn", NULL};
static const char *const k_ops_load_gsa_gsa[] = {"gsad", "gsam", "gsan", NULL};
static const char *const k_ops_load_xx[] = {"xd", "xm", "xn", NULL};
static const char *const k_ops_loop[] = {"start", "end", "xn", NULL};
static const char *const k_ops_loop_group[] = {"start_loop", "xn", "xm", "hiperf", NULL};
static const char *const k_ops_set_cke[] = {"clear", "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_clear_cke[] = {"clear", "clear_id", "clear_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_jmp[] = {"dst_xn", "cond_xn", "expect", NULL};
static const char *const k_ops_trans_mem_ms[] = {"ms",     "gsa",     "xn",      "len_xn",  "ch",
                                                 "clear",  "len_en",  "set_id",  "set_mask", "wait_id",
                                                 "wait_mask", NULL};
static const char *const k_ops_trans_ms_mem[] = {"gsa",     "xn",      "ms",      "len_xn",  "ch",
                                                 "clear",  "len_en",  "set_id",  "set_mask", "wait_id",
                                                 "wait_mask", NULL};
static const char *const k_ops_trans_loc_ms_loc_ms[] = {"dst_ms", "src_ms", "len_xn", "ch", "clear", "len_en",
                                                        "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_trans_rmt_ms_loc_ms[] = {"loc_ms", "rmt_ms", "len_xn", "ch", "clear", "len_en",
                                                        "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_trans_loc_ms_rmt_ms[] = {
    "rmt_ms", "loc_ms", "len_xn", "ch", "rmt_set_id", "rmt_set_mask", "clear", "len_en",
    "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_trans_rmt_mem_loc_mem[] = {
    "loc_gsa", "loc_xn", "rmt_gsa", "rmt_xn", "len_xn", "ch", "udf", "reduce_dtype", "reduce_op",
    "clear", "len_en", "reduce_en", "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_trans_loc_mem_rmt_mem[] = {
    "rmt_gsa", "rmt_xn", "loc_gsa", "loc_xn", "len_xn", "ch", "udf", "reduce_dtype", "reduce_op",
    "clear", "len_en", "reduce_en", "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_trans_loc_mem_loc_mem[] = {
    "dst_gsa", "dst_xn", "src_gsa", "src_xn", "len_xn", "ch", "clear", "len_en",
    "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_sync_cke[] = {"rmt_cke", "loc_cke", "loc_mask", "ch", "clear",
                                             "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_sync_gsa[] = {"rmt_gsa", "loc_gsa", "ch", "rmt_set_id", "rmt_set_mask", "clear",
                                             "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_sync_xn[] = {"rmt_xn", "loc_xn", "ch", "rmt_set_id", "rmt_set_mask", "clear",
                                            "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_add[] = {"ms", "count", "cast", "dtype", "len_xn", "clear",
                                        "set_id", "set_mask", "wait_id", "wait_mask", NULL};
static const char *const k_ops_maxmin[] = {"ms", "count", "dtype", "len_xn", "clear",
                                           "set_id", "set_mask", "wait_id", "wait_mask", NULL};

#define OD(mnem, type, code, id, ops)                                                                                  \
    {                                                                                                                  \
        mnem, type, code, id, ops, (int)(sizeof(ops) / sizeof((ops)[0]) - 1)                                            \
    }

static const CcuV1OpcodeDesc g_opcodes[CCU_V1_OP_COUNT] = {
    OD("LOAD_SQEARGS_TO_GSA", CCU_V1_LOAD_TYPE, 0x0, CCU_V1_OP_LOAD_SQEARGS_TO_GSA, k_ops_load_sqe_gsa),
    OD("LOAD_SQEARGS_TO_XN", CCU_V1_LOAD_TYPE, 0x1, CCU_V1_OP_LOAD_SQEARGS_TO_XN, k_ops_load_sqe_xn),
    OD("LOAD_IMD_TO_GSA", CCU_V1_LOAD_TYPE, 0x2, CCU_V1_OP_LOAD_IMD_TO_GSA, k_ops_load_imd_gsa),
    OD("LOAD_IMD_TO_XN", CCU_V1_LOAD_TYPE, 0x3, CCU_V1_OP_LOAD_IMD_TO_XN, k_ops_load_imd_xn),
    OD("LOAD_GSA_XN", CCU_V1_LOAD_TYPE, 0x4, CCU_V1_OP_LOAD_GSA_XN, k_ops_load_gsa_xn),
    OD("LOAD_GSA_GSA", CCU_V1_LOAD_TYPE, 0x5, CCU_V1_OP_LOAD_GSA_GSA, k_ops_load_gsa_gsa),
    OD("LOAD_XX", CCU_V1_LOAD_TYPE, 0x6, CCU_V1_OP_LOAD_XX, k_ops_load_xx),
    OD("LOOP", CCU_V1_CTRL_TYPE, 0x0, CCU_V1_OP_LOOP, k_ops_loop),
    OD("LOOP_GROUP", CCU_V1_CTRL_TYPE, 0x1, CCU_V1_OP_LOOP_GROUP, k_ops_loop_group),
    OD("SET_CKE", CCU_V1_CTRL_TYPE, 0x2, CCU_V1_OP_SET_CKE, k_ops_set_cke),
    OD("CLEAR_CKE", CCU_V1_CTRL_TYPE, 0x4, CCU_V1_OP_CLEAR_CKE, k_ops_clear_cke),
    OD("JMP", CCU_V1_CTRL_TYPE, 0x5, CCU_V1_OP_JMP, k_ops_jmp),
    OD("TRANS_LOC_MEM_TO_LOC_MS", CCU_V1_TRANS_TYPE, 0x0, CCU_V1_OP_TRANS_LOC_MEM_TO_LOC_MS, k_ops_trans_mem_ms),
    OD("TRANS_RMT_MEM_TO_LOC_MS", CCU_V1_TRANS_TYPE, 0x1, CCU_V1_OP_TRANS_RMT_MEM_TO_LOC_MS, k_ops_trans_mem_ms),
    OD("TRANS_LOC_MS_TO_LOC_MEM", CCU_V1_TRANS_TYPE, 0x2, CCU_V1_OP_TRANS_LOC_MS_TO_LOC_MEM, k_ops_trans_ms_mem),
    OD("TRANS_LOC_MS_TO_RMT_MEM", CCU_V1_TRANS_TYPE, 0x3, CCU_V1_OP_TRANS_LOC_MS_TO_RMT_MEM, k_ops_trans_ms_mem),
    OD("TRANS_RMT_MS_TO_LOC_MEM", CCU_V1_TRANS_TYPE, 0x4, CCU_V1_OP_TRANS_RMT_MS_TO_LOC_MEM, k_ops_trans_ms_mem),
    OD("TRANS_LOC_MS_TO_LOC_MS", CCU_V1_TRANS_TYPE, 0x5, CCU_V1_OP_TRANS_LOC_MS_TO_LOC_MS, k_ops_trans_loc_ms_loc_ms),
    OD("TRANS_RMT_MS_TO_LOC_MS", CCU_V1_TRANS_TYPE, 0x6, CCU_V1_OP_TRANS_RMT_MS_TO_LOC_MS, k_ops_trans_rmt_ms_loc_ms),
    OD("TRANS_LOC_MS_TO_RMT_MS", CCU_V1_TRANS_TYPE, 0x7, CCU_V1_OP_TRANS_LOC_MS_TO_RMT_MS, k_ops_trans_loc_ms_rmt_ms),
    OD("TRANS_RMT_MEM_TO_LOC_MEM", CCU_V1_TRANS_TYPE, 0x8, CCU_V1_OP_TRANS_RMT_MEM_TO_LOC_MEM,
       k_ops_trans_rmt_mem_loc_mem),
    OD("TRANS_LOC_MEM_TO_RMT_MEM", CCU_V1_TRANS_TYPE, 0x9, CCU_V1_OP_TRANS_LOC_MEM_TO_RMT_MEM,
       k_ops_trans_loc_mem_rmt_mem),
    OD("TRANS_LOC_MEM_TO_LOC_MEM", CCU_V1_TRANS_TYPE, 0xA, CCU_V1_OP_TRANS_LOC_MEM_TO_LOC_MEM,
       k_ops_trans_loc_mem_loc_mem),
    OD("SYNC_CKE", CCU_V1_TRANS_TYPE, 0xB, CCU_V1_OP_SYNC_CKE, k_ops_sync_cke),
    OD("SYNC_GSA", CCU_V1_TRANS_TYPE, 0xC, CCU_V1_OP_SYNC_GSA, k_ops_sync_gsa),
    OD("SYNC_XN", CCU_V1_TRANS_TYPE, 0xD, CCU_V1_OP_SYNC_XN, k_ops_sync_xn),
    OD("ADD", CCU_V1_REDUCE_TYPE, 0x0, CCU_V1_OP_ADD, k_ops_add),
    OD("MAX", CCU_V1_REDUCE_TYPE, 0x1, CCU_V1_OP_MAX, k_ops_maxmin),
    OD("MIN", CCU_V1_REDUCE_TYPE, 0x2, CCU_V1_OP_MIN, k_ops_maxmin),
};

/* Open-addressed mnemonic hash (FNV-1a + linear probe), built once. */
#define MNEM_HASH_SIZE 64
static const CcuV1OpcodeDesc *g_mnem_hash[MNEM_HASH_SIZE];
static int g_hash_ready;

static uint32_t fnv1a(const char *s)
{
    uint32_t h = 2166136261u;
    for (; *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c >= 'a' && c <= 'z') {
            c = (unsigned char)(c - 'a' + 'A');
        }
        h ^= c;
        h *= 16777619u;
    }
    return h;
}

static void ensure_hash(void)
{
    if (g_hash_ready) {
        return;
    }
    memset(g_mnem_hash, 0, sizeof(g_mnem_hash));
    for (size_t i = 0; i < CCU_V1_OP_COUNT; ++i) {
        uint32_t h = fnv1a(g_opcodes[i].mnemonic) & (MNEM_HASH_SIZE - 1);
        while (g_mnem_hash[h]) {
            h = (h + 1) & (MNEM_HASH_SIZE - 1);
        }
        g_mnem_hash[h] = &g_opcodes[i];
    }
    g_hash_ready = 1;
}

const CcuV1OpcodeDesc *ccu_v1_opcodes(void)
{
    return g_opcodes;
}

size_t ccu_v1_opcode_count(void)
{
    return CCU_V1_OP_COUNT;
}

static int mnem_eq(const char *a, const char *b)
{
    for (; *a && *b; ++a, ++b) {
        char ca = *a, cb = *b;
        if (ca >= 'a' && ca <= 'z') {
            ca = (char)(ca - 'a' + 'A');
        }
        if (cb >= 'a' && cb <= 'z') {
            cb = (char)(cb - 'a' + 'A');
        }
        if (ca != cb) {
            return 0;
        }
    }
    return *a == '\0' && *b == '\0';
}

const CcuV1OpcodeDesc *ccu_v1_lookup_mnemonic(const char *mnem)
{
    ensure_hash();
    uint32_t h = fnv1a(mnem) & (MNEM_HASH_SIZE - 1);
    for (int i = 0; i < MNEM_HASH_SIZE; ++i) {
        const CcuV1OpcodeDesc *d = g_mnem_hash[h];
        if (!d) {
            return NULL;
        }
        if (mnem_eq(d->mnemonic, mnem)) {
            return d;
        }
        h = (h + 1) & (MNEM_HASH_SIZE - 1);
    }
    return NULL;
}

const CcuV1OpcodeDesc *ccu_v1_lookup_opcode(uint8_t type, uint16_t code)
{
    for (size_t i = 0; i < CCU_V1_OP_COUNT; ++i) {
        if (g_opcodes[i].type == type && g_opcodes[i].code == code) {
            return &g_opcodes[i];
        }
    }
    return NULL;
}
