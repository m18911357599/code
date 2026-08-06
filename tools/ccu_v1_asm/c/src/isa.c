#include "ccu_v1_isa.h"

#include <ctype.h>
#include <string.h>

static const CcuV1OpcodeDesc g_opcodes[CCU_V1_OP_COUNT] = {
    {"LOAD_SQEARGS_TO_GSA", CCU_V1_LOAD_TYPE, 0x0, CCU_V1_OP_LOAD_SQEARGS_TO_GSA},
    {"LOAD_SQEARGS_TO_XN", CCU_V1_LOAD_TYPE, 0x1, CCU_V1_OP_LOAD_SQEARGS_TO_XN},
    {"LOAD_IMD_TO_GSA", CCU_V1_LOAD_TYPE, 0x2, CCU_V1_OP_LOAD_IMD_TO_GSA},
    {"LOAD_IMD_TO_XN", CCU_V1_LOAD_TYPE, 0x3, CCU_V1_OP_LOAD_IMD_TO_XN},
    {"LOAD_GSA_XN", CCU_V1_LOAD_TYPE, 0x4, CCU_V1_OP_LOAD_GSA_XN},
    {"LOAD_GSA_GSA", CCU_V1_LOAD_TYPE, 0x5, CCU_V1_OP_LOAD_GSA_GSA},
    {"LOAD_XX", CCU_V1_LOAD_TYPE, 0x6, CCU_V1_OP_LOAD_XX},
    {"LOOP", CCU_V1_CTRL_TYPE, 0x0, CCU_V1_OP_LOOP},
    {"LOOP_GROUP", CCU_V1_CTRL_TYPE, 0x1, CCU_V1_OP_LOOP_GROUP},
    {"SET_CKE", CCU_V1_CTRL_TYPE, 0x2, CCU_V1_OP_SET_CKE},
    {"CLEAR_CKE", CCU_V1_CTRL_TYPE, 0x4, CCU_V1_OP_CLEAR_CKE},
    {"JMP", CCU_V1_CTRL_TYPE, 0x5, CCU_V1_OP_JMP},
    {"TRANS_LOC_MEM_TO_LOC_MS", CCU_V1_TRANS_TYPE, 0x0, CCU_V1_OP_TRANS_LOC_MEM_TO_LOC_MS},
    {"TRANS_RMT_MEM_TO_LOC_MS", CCU_V1_TRANS_TYPE, 0x1, CCU_V1_OP_TRANS_RMT_MEM_TO_LOC_MS},
    {"TRANS_LOC_MS_TO_LOC_MEM", CCU_V1_TRANS_TYPE, 0x2, CCU_V1_OP_TRANS_LOC_MS_TO_LOC_MEM},
    {"TRANS_LOC_MS_TO_RMT_MEM", CCU_V1_TRANS_TYPE, 0x3, CCU_V1_OP_TRANS_LOC_MS_TO_RMT_MEM},
    {"TRANS_RMT_MS_TO_LOC_MEM", CCU_V1_TRANS_TYPE, 0x4, CCU_V1_OP_TRANS_RMT_MS_TO_LOC_MEM},
    {"TRANS_LOC_MS_TO_LOC_MS", CCU_V1_TRANS_TYPE, 0x5, CCU_V1_OP_TRANS_LOC_MS_TO_LOC_MS},
    {"TRANS_RMT_MS_TO_LOC_MS", CCU_V1_TRANS_TYPE, 0x6, CCU_V1_OP_TRANS_RMT_MS_TO_LOC_MS},
    {"TRANS_LOC_MS_TO_RMT_MS", CCU_V1_TRANS_TYPE, 0x7, CCU_V1_OP_TRANS_LOC_MS_TO_RMT_MS},
    {"TRANS_RMT_MEM_TO_LOC_MEM", CCU_V1_TRANS_TYPE, 0x8, CCU_V1_OP_TRANS_RMT_MEM_TO_LOC_MEM},
    {"TRANS_LOC_MEM_TO_RMT_MEM", CCU_V1_TRANS_TYPE, 0x9, CCU_V1_OP_TRANS_LOC_MEM_TO_RMT_MEM},
    {"TRANS_LOC_MEM_TO_LOC_MEM", CCU_V1_TRANS_TYPE, 0xA, CCU_V1_OP_TRANS_LOC_MEM_TO_LOC_MEM},
    {"SYNC_CKE", CCU_V1_TRANS_TYPE, 0xB, CCU_V1_OP_SYNC_CKE},
    {"SYNC_GSA", CCU_V1_TRANS_TYPE, 0xC, CCU_V1_OP_SYNC_GSA},
    {"SYNC_XN", CCU_V1_TRANS_TYPE, 0xD, CCU_V1_OP_SYNC_XN},
    {"ADD", CCU_V1_REDUCE_TYPE, 0x0, CCU_V1_OP_ADD},
    {"MAX", CCU_V1_REDUCE_TYPE, 0x1, CCU_V1_OP_MAX},
    {"MIN", CCU_V1_REDUCE_TYPE, 0x2, CCU_V1_OP_MIN},
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
