#include "ccu_v1_asm.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ccu_v1_program_init(CcuV1Program *p)
{
    p->items = NULL;
    p->count = 0;
    p->capacity = 0;
}

void ccu_v1_program_free(CcuV1Program *p)
{
    free(p->items);
    p->items = NULL;
    p->count = 0;
    p->capacity = 0;
}

int ccu_v1_program_reserve(CcuV1Program *p, size_t n)
{
    if (n <= p->capacity) {
        return 0;
    }
    size_t cap = p->capacity ? p->capacity : 2048;
    while (cap < n) {
        cap *= 2;
    }
    CcuV1Instr *ni = (CcuV1Instr *)realloc(p->items, cap * sizeof(CcuV1Instr));
    if (!ni) {
        return -1;
    }
    p->items = ni;
    p->capacity = cap;
    return 0;
}

void ccu_v1_instr_set_opcode(CcuV1Instr *instr, const CcuV1OpcodeDesc *desc)
{
    memset(instr, 0, sizeof(*instr));
    instr->header.raw = ccu_v1_make_header(desc->type, desc->code);
}

/* ---- operand kv helpers ---- */
#define MAX_OPS 24
#define MAX_NAME 32

typedef struct {
    char name[MAX_NAME];
    int is_list;
    uint64_t scalar;
    uint16_t list[CCU_V1_MS_MAX];
    int list_n;
} OpKV;

typedef struct {
    OpKV items[MAX_OPS];
    int count;
} OpMap;

static void opmap_init(OpMap *m)
{
    m->count = 0;
}

static OpKV *opmap_find(OpMap *m, const char *name)
{
    for (int i = 0; i < m->count; ++i) {
        if (strcmp(m->items[i].name, name) == 0) {
            return &m->items[i];
        }
    }
    return NULL;
}

static int opmap_get_u64(OpMap *m, const char *name, uint64_t def, uint64_t *out)
{
    OpKV *kv = opmap_find(m, name);
    if (!kv) {
        *out = def;
        return 0;
    }
    if (kv->is_list) {
        return -1;
    }
    *out = kv->scalar;
    return 0;
}

static int opmap_get_u16(OpMap *m, const char *name, uint16_t def, uint16_t *out)
{
    uint64_t v;
    if (opmap_get_u64(m, name, def, &v) != 0) {
        return -1;
    }
    if (v > 0xFFFFu) {
        return -1;
    }
    *out = (uint16_t)v;
    return 0;
}

static int opmap_get_ms(OpMap *m, const char *name, uint16_t out[CCU_V1_MS_MAX])
{
    OpKV *kv = opmap_find(m, name);
    memset(out, 0, sizeof(uint16_t) * CCU_V1_MS_MAX);
    if (!kv) {
        return 0;
    }
    if (!kv->is_list) {
        out[0] = (uint16_t)kv->scalar;
        return 0;
    }
    for (int i = 0; i < kv->list_n && i < CCU_V1_MS_MAX; ++i) {
        out[i] = kv->list[i];
    }
    return 0;
}

static const char *skip_ws(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\r') {
        ++p;
    }
    return p;
}

static int parse_u64(const char *s, const char *end, uint64_t *out)
{
    if (s >= end) {
        return -1;
    }
    uint64_t v = 0;
    if (end - s >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
        if (s >= end) {
            return -1;
        }
        while (s < end) {
            char c = *s++;
            unsigned d;
            if (c >= '0' && c <= '9') {
                d = (unsigned)(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                d = (unsigned)(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                d = (unsigned)(c - 'A' + 10);
            } else {
                return -1;
            }
            if (v > (0xFFFFFFFFFFFFFFFFull >> 4)) {
                return -1;
            }
            v = (v << 4) | d;
        }
        *out = v;
        return 0;
    }
    while (s < end) {
        char c = *s++;
        if (c < '0' || c > '9') {
            return -1;
        }
        if (v > (0xFFFFFFFFFFFFFFFFull / 10)) {
            return -1;
        }
        v = v * 10u + (uint64_t)(c - '0');
    }
    *out = v;
    return 0;
}

static int parse_list(const char *s, const char *end, OpKV *kv)
{
    kv->is_list = 1;
    kv->list_n = 0;
    const char *p = s;
    if (p >= end || *p != '[') {
        return -1;
    }
    ++p;
    p = skip_ws(p);
    if (p < end && *p == ']') {
        return 0;
    }
    while (p < end) {
        const char *v0 = p;
        while (p < end && *p != ',' && *p != ']') {
            ++p;
        }
        const char *v1 = p;
        while (v1 > v0 && (v1[-1] == ' ' || v1[-1] == '\t')) {
            --v1;
        }
        uint64_t val;
        if (parse_u64(v0, v1, &val) != 0 || val > 0xFFFFu) {
            return -1;
        }
        if (kv->list_n >= CCU_V1_MS_MAX) {
            return -1;
        }
        kv->list[kv->list_n++] = (uint16_t)val;
        p = skip_ws(p);
        if (p < end && *p == ',') {
            ++p;
            p = skip_ws(p);
            continue;
        }
        if (p < end && *p == ']') {
            return 0;
        }
        return -1;
    }
    return -1;
}

static int parse_operands(const char *line, OpMap *map, char *errmsg, size_t errmsg_sz)
{
    opmap_init(map);
    const char *p = skip_ws(line);
    if (*p == '\0' || *p == '#') {
        return 0;
    }
    while (*p) {
        if (map->count >= MAX_OPS) {
            snprintf(errmsg, errmsg_sz, "too many operands");
            return -1;
        }
        const char *n0 = p;
        while (*p && (isalnum((unsigned char)*p) || *p == '_')) {
            ++p;
        }
        if (p == n0 || *p != '=') {
            snprintf(errmsg, errmsg_sz, "expected name=value near '%s'", n0);
            return -1;
        }
        size_t nlen = (size_t)(p - n0);
        if (nlen >= MAX_NAME) {
            snprintf(errmsg, errmsg_sz, "operand name too long");
            return -1;
        }
        OpKV *kv = &map->items[map->count];
        memcpy(kv->name, n0, nlen);
        kv->name[nlen] = '\0';
        kv->is_list = 0;
        kv->scalar = 0;
        kv->list_n = 0;
        ++p;
        p = skip_ws(p);
        if (*p == '[') {
            const char *lb = p;
            while (*p && *p != ']') {
                ++p;
            }
            if (*p != ']') {
                snprintf(errmsg, errmsg_sz, "unclosed list in %s", kv->name);
                return -1;
            }
            ++p;
            if (parse_list(lb, p, kv) != 0) {
                snprintf(errmsg, errmsg_sz, "bad list for %s", kv->name);
                return -1;
            }
        } else {
            const char *v0 = p;
            while (*p && *p != ',' && *p != '#' && *p != ' ' && *p != '\t') {
                ++p;
            }
            if (parse_u64(v0, p, &kv->scalar) != 0) {
                snprintf(errmsg, errmsg_sz, "bad value for %s", kv->name);
                return -1;
            }
        }
        map->count++;
        p = skip_ws(p);
        if (*p == ',') {
            ++p;
            p = skip_ws(p);
            continue;
        }
        if (*p == '\0' || *p == '#') {
            return 0;
        }
        snprintf(errmsg, errmsg_sz, "unexpected text near '%s'", p);
        return -1;
    }
    return 0;
}

#define GET16(name, field)                                                                                             \
    do {                                                                                                               \
        if (opmap_get_u16(map, name, 0, &field) != 0) {                                                                \
            snprintf(errmsg, errmsg_sz, "bad operand %s", name);                                                       \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define GET64(name, field)                                                                                             \
    do {                                                                                                               \
        if (opmap_get_u64(map, name, 0, &field) != 0) {                                                                \
            snprintf(errmsg, errmsg_sz, "bad operand %s", name);                                                       \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)

static int fill_instr(CcuV1OpcodeId id, OpMap *map, CcuV1Instr *instr, char *errmsg, size_t errmsg_sz)
{
    switch (id) {
    case CCU_V1_OP_LOAD_SQEARGS_TO_GSA:
        GET16("gsa", instr->load_sqe_gsa.gsa);
        GET16("sqe", instr->load_sqe_gsa.sqe);
        break;
    case CCU_V1_OP_LOAD_SQEARGS_TO_XN:
        GET16("xn", instr->load_sqe_xn.xn);
        GET16("sqe", instr->load_sqe_xn.sqe);
        break;
    case CCU_V1_OP_LOAD_IMD_TO_GSA:
        GET16("gsa", instr->load_imd_gsa.gsa);
        GET64("imm", instr->load_imd_gsa.imm);
        break;
    case CCU_V1_OP_LOAD_IMD_TO_XN:
        GET16("xn", instr->load_imd_xn.xn);
        GET64("imm", instr->load_imd_xn.imm);
        GET16("sec", instr->load_imd_xn.sec);
        break;
    case CCU_V1_OP_LOAD_GSA_XN:
        GET16("gsad", instr->load_gsa_xn.gsad);
        GET16("gsam", instr->load_gsa_xn.gsam);
        GET16("xn", instr->load_gsa_xn.xn);
        break;
    case CCU_V1_OP_LOAD_GSA_GSA:
        GET16("gsad", instr->load_gsa_gsa.gsad);
        GET16("gsam", instr->load_gsa_gsa.gsam);
        GET16("gsan", instr->load_gsa_gsa.gsan);
        break;
    case CCU_V1_OP_LOAD_XX:
        GET16("xd", instr->load_xx.xd);
        GET16("xm", instr->load_xx.xm);
        GET16("xn", instr->load_xx.xn);
        break;
    case CCU_V1_OP_LOOP:
        GET16("start", instr->loop.start);
        GET16("end", instr->loop.end);
        GET16("xn", instr->loop.xn);
        break;
    case CCU_V1_OP_LOOP_GROUP: {
        uint16_t hiperf = 0;
        GET16("start_loop", instr->loop_group.start_loop);
        GET16("xn", instr->loop_group.xn);
        GET16("xm", instr->loop_group.xm);
        GET16("hiperf", hiperf);
        instr->loop_group.hiperf = hiperf & 1u;
        break;
    }
    case CCU_V1_OP_SET_CKE: {
        uint16_t clear = 0;
        GET16("clear", clear);
        instr->set_cke.clear = clear & 1u;
        GET16("set_id", instr->set_cke.set_id);
        GET16("set_mask", instr->set_cke.set_mask);
        GET16("wait_id", instr->set_cke.wait_id);
        GET16("wait_mask", instr->set_cke.wait_mask);
        break;
    }
    case CCU_V1_OP_CLEAR_CKE: {
        uint16_t clear = 0;
        GET16("clear", clear);
        instr->clear_cke.clear = clear & 1u;
        GET16("clear_id", instr->clear_cke.clear_id);
        GET16("clear_mask", instr->clear_cke.clear_mask);
        GET16("wait_id", instr->clear_cke.wait_id);
        GET16("wait_mask", instr->clear_cke.wait_mask);
        break;
    }
    case CCU_V1_OP_JMP:
        GET16("dst_xn", instr->jmp.dst_xn);
        GET16("cond_xn", instr->jmp.cond_xn);
        GET64("expect", instr->jmp.expect);
        break;
    case CCU_V1_OP_TRANS_LOC_MEM_TO_LOC_MS:
    case CCU_V1_OP_TRANS_RMT_MEM_TO_LOC_MS: {
        uint16_t clear = 0, len_en = 0;
        GET16("ms", instr->trans_mem_to_ms.ms);
        GET16("gsa", instr->trans_mem_to_ms.gsa);
        GET16("xn", instr->trans_mem_to_ms.xn);
        GET16("len_xn", instr->trans_mem_to_ms.len_xn);
        GET16("ch", instr->trans_mem_to_ms.ch);
        GET16("clear", clear);
        GET16("len_en", len_en);
        instr->trans_mem_to_ms.clear = clear & 1u;
        instr->trans_mem_to_ms.len_en = len_en & 1u;
        GET16("set_id", instr->trans_mem_to_ms.set_id);
        GET16("set_mask", instr->trans_mem_to_ms.set_mask);
        GET16("wait_id", instr->trans_mem_to_ms.wait_id);
        GET16("wait_mask", instr->trans_mem_to_ms.wait_mask);
        break;
    }
    case CCU_V1_OP_TRANS_LOC_MS_TO_LOC_MEM:
    case CCU_V1_OP_TRANS_LOC_MS_TO_RMT_MEM:
    case CCU_V1_OP_TRANS_RMT_MS_TO_LOC_MEM: {
        uint16_t clear = 0, len_en = 0;
        GET16("gsa", instr->trans_ms_to_mem.gsa);
        GET16("xn", instr->trans_ms_to_mem.xn);
        GET16("ms", instr->trans_ms_to_mem.ms);
        GET16("len_xn", instr->trans_ms_to_mem.len_xn);
        GET16("ch", instr->trans_ms_to_mem.ch);
        GET16("clear", clear);
        GET16("len_en", len_en);
        instr->trans_ms_to_mem.clear = clear & 1u;
        instr->trans_ms_to_mem.len_en = len_en & 1u;
        GET16("set_id", instr->trans_ms_to_mem.set_id);
        GET16("set_mask", instr->trans_ms_to_mem.set_mask);
        GET16("wait_id", instr->trans_ms_to_mem.wait_id);
        GET16("wait_mask", instr->trans_ms_to_mem.wait_mask);
        break;
    }
    case CCU_V1_OP_TRANS_LOC_MS_TO_LOC_MS: {
        uint16_t clear = 0, len_en = 0;
        GET16("dst_ms", instr->trans_loc_ms_loc_ms.dst_ms);
        GET16("src_ms", instr->trans_loc_ms_loc_ms.src_ms);
        GET16("len_xn", instr->trans_loc_ms_loc_ms.len_xn);
        GET16("ch", instr->trans_loc_ms_loc_ms.ch);
        GET16("clear", clear);
        GET16("len_en", len_en);
        instr->trans_loc_ms_loc_ms.clear = clear & 1u;
        instr->trans_loc_ms_loc_ms.len_en = len_en & 1u;
        GET16("set_id", instr->trans_loc_ms_loc_ms.set_id);
        GET16("set_mask", instr->trans_loc_ms_loc_ms.set_mask);
        GET16("wait_id", instr->trans_loc_ms_loc_ms.wait_id);
        GET16("wait_mask", instr->trans_loc_ms_loc_ms.wait_mask);
        break;
    }
    case CCU_V1_OP_TRANS_RMT_MS_TO_LOC_MS: {
        uint16_t clear = 0, len_en = 0;
        GET16("loc_ms", instr->trans_rmt_ms_loc_ms.loc_ms);
        GET16("rmt_ms", instr->trans_rmt_ms_loc_ms.rmt_ms);
        GET16("len_xn", instr->trans_rmt_ms_loc_ms.len_xn);
        GET16("ch", instr->trans_rmt_ms_loc_ms.ch);
        GET16("clear", clear);
        GET16("len_en", len_en);
        instr->trans_rmt_ms_loc_ms.clear = clear & 1u;
        instr->trans_rmt_ms_loc_ms.len_en = len_en & 1u;
        GET16("set_id", instr->trans_rmt_ms_loc_ms.set_id);
        GET16("set_mask", instr->trans_rmt_ms_loc_ms.set_mask);
        GET16("wait_id", instr->trans_rmt_ms_loc_ms.wait_id);
        GET16("wait_mask", instr->trans_rmt_ms_loc_ms.wait_mask);
        break;
    }
    case CCU_V1_OP_TRANS_LOC_MS_TO_RMT_MS: {
        uint16_t clear = 0, len_en = 0;
        GET16("rmt_ms", instr->trans_loc_ms_rmt_ms.rmt_ms);
        GET16("loc_ms", instr->trans_loc_ms_rmt_ms.loc_ms);
        GET16("len_xn", instr->trans_loc_ms_rmt_ms.len_xn);
        GET16("ch", instr->trans_loc_ms_rmt_ms.ch);
        GET16("rmt_set_id", instr->trans_loc_ms_rmt_ms.rmt_set_id);
        GET16("rmt_set_mask", instr->trans_loc_ms_rmt_ms.rmt_set_mask);
        GET16("clear", clear);
        GET16("len_en", len_en);
        instr->trans_loc_ms_rmt_ms.clear = clear & 1u;
        instr->trans_loc_ms_rmt_ms.len_en = len_en & 1u;
        GET16("set_id", instr->trans_loc_ms_rmt_ms.set_id);
        GET16("set_mask", instr->trans_loc_ms_rmt_ms.set_mask);
        GET16("wait_id", instr->trans_loc_ms_rmt_ms.wait_id);
        GET16("wait_mask", instr->trans_loc_ms_rmt_ms.wait_mask);
        break;
    }
    case CCU_V1_OP_TRANS_RMT_MEM_TO_LOC_MEM: {
        uint16_t clear = 0, len_en = 0, reduce_en = 0, udf = 0, rd = 0, ro = 0;
        GET16("loc_gsa", instr->trans_rmt_mem_loc_mem.loc_gsa);
        GET16("loc_xn", instr->trans_rmt_mem_loc_mem.loc_xn);
        GET16("rmt_gsa", instr->trans_rmt_mem_loc_mem.rmt_gsa);
        GET16("rmt_xn", instr->trans_rmt_mem_loc_mem.rmt_xn);
        GET16("len_xn", instr->trans_rmt_mem_loc_mem.len_xn);
        GET16("ch", instr->trans_rmt_mem_loc_mem.ch);
        GET16("udf", udf);
        GET16("reduce_dtype", rd);
        GET16("reduce_op", ro);
        instr->trans_rmt_mem_loc_mem.udf = udf & 0xFFu;
        instr->trans_rmt_mem_loc_mem.reduce_dtype = rd & 0xFu;
        instr->trans_rmt_mem_loc_mem.reduce_op = ro & 0xFu;
        GET16("clear", clear);
        GET16("len_en", len_en);
        GET16("reduce_en", reduce_en);
        instr->trans_rmt_mem_loc_mem.clear = clear & 1u;
        instr->trans_rmt_mem_loc_mem.len_en = len_en & 1u;
        instr->trans_rmt_mem_loc_mem.reduce_en = reduce_en & 1u;
        GET16("set_id", instr->trans_rmt_mem_loc_mem.set_id);
        GET16("set_mask", instr->trans_rmt_mem_loc_mem.set_mask);
        GET16("wait_id", instr->trans_rmt_mem_loc_mem.wait_id);
        GET16("wait_mask", instr->trans_rmt_mem_loc_mem.wait_mask);
        break;
    }
    case CCU_V1_OP_TRANS_LOC_MEM_TO_RMT_MEM: {
        uint16_t clear = 0, len_en = 0, reduce_en = 0, udf = 0, rd = 0, ro = 0;
        GET16("rmt_gsa", instr->trans_loc_mem_rmt_mem.rmt_gsa);
        GET16("rmt_xn", instr->trans_loc_mem_rmt_mem.rmt_xn);
        GET16("loc_gsa", instr->trans_loc_mem_rmt_mem.loc_gsa);
        GET16("loc_xn", instr->trans_loc_mem_rmt_mem.loc_xn);
        GET16("len_xn", instr->trans_loc_mem_rmt_mem.len_xn);
        GET16("ch", instr->trans_loc_mem_rmt_mem.ch);
        GET16("udf", udf);
        GET16("reduce_dtype", rd);
        GET16("reduce_op", ro);
        instr->trans_loc_mem_rmt_mem.udf = udf & 0xFFu;
        instr->trans_loc_mem_rmt_mem.reduce_dtype = rd & 0xFu;
        instr->trans_loc_mem_rmt_mem.reduce_op = ro & 0xFu;
        GET16("clear", clear);
        GET16("len_en", len_en);
        GET16("reduce_en", reduce_en);
        instr->trans_loc_mem_rmt_mem.clear = clear & 1u;
        instr->trans_loc_mem_rmt_mem.len_en = len_en & 1u;
        instr->trans_loc_mem_rmt_mem.reduce_en = reduce_en & 1u;
        GET16("set_id", instr->trans_loc_mem_rmt_mem.set_id);
        GET16("set_mask", instr->trans_loc_mem_rmt_mem.set_mask);
        GET16("wait_id", instr->trans_loc_mem_rmt_mem.wait_id);
        GET16("wait_mask", instr->trans_loc_mem_rmt_mem.wait_mask);
        break;
    }
    case CCU_V1_OP_TRANS_LOC_MEM_TO_LOC_MEM: {
        uint16_t clear = 0, len_en = 0;
        GET16("dst_gsa", instr->trans_loc_mem_loc_mem.dst_gsa);
        GET16("dst_xn", instr->trans_loc_mem_loc_mem.dst_xn);
        GET16("src_gsa", instr->trans_loc_mem_loc_mem.src_gsa);
        GET16("src_xn", instr->trans_loc_mem_loc_mem.src_xn);
        GET16("len_xn", instr->trans_loc_mem_loc_mem.len_xn);
        GET16("ch", instr->trans_loc_mem_loc_mem.ch);
        GET16("clear", clear);
        GET16("len_en", len_en);
        instr->trans_loc_mem_loc_mem.clear = clear & 1u;
        instr->trans_loc_mem_loc_mem.len_en = len_en & 1u;
        GET16("set_id", instr->trans_loc_mem_loc_mem.set_id);
        GET16("set_mask", instr->trans_loc_mem_loc_mem.set_mask);
        GET16("wait_id", instr->trans_loc_mem_loc_mem.wait_id);
        GET16("wait_mask", instr->trans_loc_mem_loc_mem.wait_mask);
        break;
    }
    case CCU_V1_OP_SYNC_CKE: {
        uint16_t clear = 0;
        GET16("rmt_cke", instr->sync_cke.rmt_cke);
        GET16("loc_cke", instr->sync_cke.loc_cke);
        GET16("loc_mask", instr->sync_cke.loc_mask);
        GET16("ch", instr->sync_cke.ch);
        GET16("clear", clear);
        instr->sync_cke.clear = clear & 1u;
        GET16("set_id", instr->sync_cke.set_id);
        GET16("set_mask", instr->sync_cke.set_mask);
        GET16("wait_id", instr->sync_cke.wait_id);
        GET16("wait_mask", instr->sync_cke.wait_mask);
        break;
    }
    case CCU_V1_OP_SYNC_GSA: {
        uint16_t clear = 0;
        GET16("rmt_gsa", instr->sync_gsa.rmt_gsa);
        GET16("loc_gsa", instr->sync_gsa.loc_gsa);
        GET16("ch", instr->sync_gsa.ch);
        GET16("rmt_set_id", instr->sync_gsa.rmt_set_id);
        GET16("rmt_set_mask", instr->sync_gsa.rmt_set_mask);
        GET16("clear", clear);
        instr->sync_gsa.clear = clear & 1u;
        GET16("set_id", instr->sync_gsa.set_id);
        GET16("set_mask", instr->sync_gsa.set_mask);
        GET16("wait_id", instr->sync_gsa.wait_id);
        GET16("wait_mask", instr->sync_gsa.wait_mask);
        break;
    }
    case CCU_V1_OP_SYNC_XN: {
        uint16_t clear = 0;
        GET16("rmt_xn", instr->sync_xn.rmt_xn);
        GET16("loc_xn", instr->sync_xn.loc_xn);
        GET16("ch", instr->sync_xn.ch);
        GET16("rmt_set_id", instr->sync_xn.rmt_set_id);
        GET16("rmt_set_mask", instr->sync_xn.rmt_set_mask);
        GET16("clear", clear);
        instr->sync_xn.clear = clear & 1u;
        GET16("set_id", instr->sync_xn.set_id);
        GET16("set_mask", instr->sync_xn.set_mask);
        GET16("wait_id", instr->sync_xn.wait_id);
        GET16("wait_mask", instr->sync_xn.wait_mask);
        break;
    }
    case CCU_V1_OP_ADD: {
        uint16_t clear = 0, count = 0, cast = 0, dtype = 0;
        if (opmap_get_ms(map, "ms", instr->add.ms) != 0) {
            snprintf(errmsg, errmsg_sz, "bad operand ms");
            return -1;
        }
        GET16("len_xn", instr->add.len_xn);
        GET16("clear", clear);
        GET16("count", count);
        GET16("cast", cast);
        GET16("dtype", dtype);
        instr->add.clear = clear & 1u;
        instr->add.count = count & 7u;
        instr->add.cast = cast & 3u;
        instr->add.dtype = dtype & 31u;
        GET16("set_id", instr->add.set_id);
        GET16("set_mask", instr->add.set_mask);
        GET16("wait_id", instr->add.wait_id);
        GET16("wait_mask", instr->add.wait_mask);
        break;
    }
    case CCU_V1_OP_MAX:
    case CCU_V1_OP_MIN: {
        uint16_t clear = 0, count = 0, dtype = 0;
        if (opmap_get_ms(map, "ms", instr->maxmin.ms) != 0) {
            snprintf(errmsg, errmsg_sz, "bad operand ms");
            return -1;
        }
        GET16("len_xn", instr->maxmin.len_xn);
        GET16("clear", clear);
        GET16("count", count);
        GET16("dtype", dtype);
        instr->maxmin.clear = clear & 1u;
        instr->maxmin.count = count & 7u;
        instr->maxmin.dtype = dtype & 31u;
        GET16("set_id", instr->maxmin.set_id);
        GET16("set_mask", instr->maxmin.set_mask);
        GET16("wait_id", instr->maxmin.wait_id);
        GET16("wait_mask", instr->maxmin.wait_mask);
        break;
    }
    default:
        snprintf(errmsg, errmsg_sz, "internal: unknown opcode id");
        return -1;
    }
    return 0;
}

static int assemble_line(const char *line, int lineno, CcuV1Instr *out, char *errmsg, size_t errmsg_sz)
{
    const char *p = skip_ws(line);
    if (*p == '\0' || *p == '#') {
        return 1; /* empty */
    }
    /* optional label: */
    const char *q = p;
    while (*q && (isalnum((unsigned char)*q) || *q == '_')) {
        ++q;
    }
    if (*q == ':') {
        p = skip_ws(q + 1);
        if (*p == '\0' || *p == '#') {
            return 1;
        }
    }
    char mnem[64];
    size_t mi = 0;
    while (*p && (isalnum((unsigned char)*p) || *p == '_') && mi + 1 < sizeof(mnem)) {
        mnem[mi++] = *p++;
    }
    mnem[mi] = '\0';
    if (mi == 0) {
        snprintf(errmsg, errmsg_sz, "line %d: missing mnemonic", lineno);
        return -1;
    }
    const CcuV1OpcodeDesc *desc = ccu_v1_lookup_mnemonic(mnem);
    if (!desc) {
        snprintf(errmsg, errmsg_sz, "line %d: unknown mnemonic %s", lineno, mnem);
        return -1;
    }
    OpMap map;
    char local_err[128];
    if (parse_operands(p, &map, local_err, sizeof(local_err)) != 0) {
        snprintf(errmsg, errmsg_sz, "line %d: %s", lineno, local_err);
        return -1;
    }
    ccu_v1_instr_set_opcode(out, desc);
    if (fill_instr(desc->id, &map, out, local_err, sizeof(local_err)) != 0) {
        snprintf(errmsg, errmsg_sz, "line %d: %s", lineno, local_err);
        return -1;
    }
    return 0;
}

int ccu_v1_assemble_text(const char *text, size_t text_len, CcuV1Program *out, char *errmsg, size_t errmsg_sz)
{
    ccu_v1_program_init(out);
    /* estimate lines */
    size_t approx = 16;
    for (size_t i = 0; i < text_len; ++i) {
        if (text[i] == '\n') {
            ++approx;
        }
    }
    if (ccu_v1_program_reserve(out, approx) != 0) {
        snprintf(errmsg, errmsg_sz, "oom");
        return -1;
    }

    int lineno = 1;
    const char *p = text;
    const char *end = text + text_len;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') {
            ++p;
        }
        size_t llen = (size_t)(p - line);
        char buf[4096];
        if (llen >= sizeof(buf)) {
            snprintf(errmsg, errmsg_sz, "line %d: line too long", lineno);
            ccu_v1_program_free(out);
            return -1;
        }
        memcpy(buf, line, llen);
        buf[llen] = '\0';
        /* strip CR */
        if (llen > 0 && buf[llen - 1] == '\r') {
            buf[llen - 1] = '\0';
        }
        if (out->count >= out->capacity) {
            if (ccu_v1_program_reserve(out, out->capacity * 2) != 0) {
                snprintf(errmsg, errmsg_sz, "oom");
                ccu_v1_program_free(out);
                return -1;
            }
        }
        int rc = assemble_line(buf, lineno, &out->items[out->count], errmsg, errmsg_sz);
        if (rc < 0) {
            ccu_v1_program_free(out);
            return -1;
        }
        if (rc == 0) {
            out->count++;
        }
        if (p < end && *p == '\n') {
            ++p;
        }
        ++lineno;
    }
    return 0;
}

/* ---- format ---- */

static int appendf(char *buf, size_t buf_sz, int *pos, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + *pos, (*pos < (int)buf_sz) ? buf_sz - (size_t)*pos : 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        return -1;
    }
    *pos += n;
    return 0;
}

static void fmt_ms(char *tmp, size_t tmp_sz, const uint16_t ms[CCU_V1_MS_MAX])
{
    int pos = 0;
    pos += snprintf(tmp + pos, tmp_sz - (size_t)pos, "[");
    for (int i = 0; i < CCU_V1_MS_MAX; ++i) {
        if (i) {
            pos += snprintf(tmp + pos, tmp_sz - (size_t)pos, ",");
        }
        pos += snprintf(tmp + pos, tmp_sz - (size_t)pos, "%u", (unsigned)ms[i]);
    }
    snprintf(tmp + pos, tmp_sz - (size_t)pos, "]");
}

int ccu_v1_format_instr(const CcuV1Instr *instr, char *buf, size_t buf_sz)
{
    uint8_t type;
    uint16_t code;
    ccu_v1_split_header(instr->header.raw, &type, &code);
    const CcuV1OpcodeDesc *desc = ccu_v1_lookup_opcode(type, code);
    if (!desc) {
        return -1;
    }
    int pos = 0;
    if (appendf(buf, buf_sz, &pos, "%s", desc->mnemonic) != 0) {
        return -1;
    }

#define SEP() appendf(buf, buf_sz, &pos, (pos > (int)strlen(desc->mnemonic)) ? ", " : " ")
#define U16(name, v)                                                                                                   \
    do {                                                                                                               \
        if (SEP() != 0 || appendf(buf, buf_sz, &pos, "%s=%u", name, (unsigned)(v)) != 0)                               \
            return -1;                                                                                                 \
    } while (0)
#define HEX(name, v)                                                                                                   \
    do {                                                                                                               \
        if (SEP() != 0 || appendf(buf, buf_sz, &pos, "%s=0x%llx", name, (unsigned long long)(v)) != 0)                  \
            return -1;                                                                                                 \
    } while (0)

    switch (desc->id) {
    case CCU_V1_OP_LOAD_SQEARGS_TO_GSA:
        U16("gsa", instr->load_sqe_gsa.gsa);
        U16("sqe", instr->load_sqe_gsa.sqe);
        break;
    case CCU_V1_OP_LOAD_SQEARGS_TO_XN:
        U16("xn", instr->load_sqe_xn.xn);
        U16("sqe", instr->load_sqe_xn.sqe);
        break;
    case CCU_V1_OP_LOAD_IMD_TO_GSA:
        U16("gsa", instr->load_imd_gsa.gsa);
        HEX("imm", instr->load_imd_gsa.imm);
        break;
    case CCU_V1_OP_LOAD_IMD_TO_XN:
        U16("xn", instr->load_imd_xn.xn);
        HEX("imm", instr->load_imd_xn.imm);
        U16("sec", instr->load_imd_xn.sec);
        break;
    case CCU_V1_OP_LOAD_GSA_XN:
        U16("gsad", instr->load_gsa_xn.gsad);
        U16("gsam", instr->load_gsa_xn.gsam);
        U16("xn", instr->load_gsa_xn.xn);
        break;
    case CCU_V1_OP_LOAD_GSA_GSA:
        U16("gsad", instr->load_gsa_gsa.gsad);
        U16("gsam", instr->load_gsa_gsa.gsam);
        U16("gsan", instr->load_gsa_gsa.gsan);
        break;
    case CCU_V1_OP_LOAD_XX:
        U16("xd", instr->load_xx.xd);
        U16("xm", instr->load_xx.xm);
        U16("xn", instr->load_xx.xn);
        break;
    case CCU_V1_OP_LOOP:
        U16("start", instr->loop.start);
        U16("end", instr->loop.end);
        U16("xn", instr->loop.xn);
        break;
    case CCU_V1_OP_LOOP_GROUP:
        U16("start_loop", instr->loop_group.start_loop);
        U16("xn", instr->loop_group.xn);
        U16("xm", instr->loop_group.xm);
        U16("hiperf", instr->loop_group.hiperf);
        break;
    case CCU_V1_OP_SET_CKE:
        U16("clear", instr->set_cke.clear);
        U16("set_id", instr->set_cke.set_id);
        HEX("set_mask", instr->set_cke.set_mask);
        U16("wait_id", instr->set_cke.wait_id);
        HEX("wait_mask", instr->set_cke.wait_mask);
        break;
    case CCU_V1_OP_CLEAR_CKE:
        U16("clear", instr->clear_cke.clear);
        U16("clear_id", instr->clear_cke.clear_id);
        HEX("clear_mask", instr->clear_cke.clear_mask);
        U16("wait_id", instr->clear_cke.wait_id);
        HEX("wait_mask", instr->clear_cke.wait_mask);
        break;
    case CCU_V1_OP_JMP:
        U16("dst_xn", instr->jmp.dst_xn);
        U16("cond_xn", instr->jmp.cond_xn);
        HEX("expect", instr->jmp.expect);
        break;
    case CCU_V1_OP_TRANS_LOC_MEM_TO_LOC_MS:
    case CCU_V1_OP_TRANS_RMT_MEM_TO_LOC_MS:
        U16("ms", instr->trans_mem_to_ms.ms);
        U16("gsa", instr->trans_mem_to_ms.gsa);
        U16("xn", instr->trans_mem_to_ms.xn);
        U16("len_xn", instr->trans_mem_to_ms.len_xn);
        U16("ch", instr->trans_mem_to_ms.ch);
        U16("clear", instr->trans_mem_to_ms.clear);
        U16("len_en", instr->trans_mem_to_ms.len_en);
        U16("set_id", instr->trans_mem_to_ms.set_id);
        HEX("set_mask", instr->trans_mem_to_ms.set_mask);
        U16("wait_id", instr->trans_mem_to_ms.wait_id);
        HEX("wait_mask", instr->trans_mem_to_ms.wait_mask);
        break;
    case CCU_V1_OP_TRANS_LOC_MS_TO_LOC_MEM:
    case CCU_V1_OP_TRANS_LOC_MS_TO_RMT_MEM:
    case CCU_V1_OP_TRANS_RMT_MS_TO_LOC_MEM:
        U16("gsa", instr->trans_ms_to_mem.gsa);
        U16("xn", instr->trans_ms_to_mem.xn);
        U16("ms", instr->trans_ms_to_mem.ms);
        U16("len_xn", instr->trans_ms_to_mem.len_xn);
        U16("ch", instr->trans_ms_to_mem.ch);
        U16("clear", instr->trans_ms_to_mem.clear);
        U16("len_en", instr->trans_ms_to_mem.len_en);
        U16("set_id", instr->trans_ms_to_mem.set_id);
        HEX("set_mask", instr->trans_ms_to_mem.set_mask);
        U16("wait_id", instr->trans_ms_to_mem.wait_id);
        HEX("wait_mask", instr->trans_ms_to_mem.wait_mask);
        break;
    case CCU_V1_OP_TRANS_LOC_MS_TO_LOC_MS:
        U16("dst_ms", instr->trans_loc_ms_loc_ms.dst_ms);
        U16("src_ms", instr->trans_loc_ms_loc_ms.src_ms);
        U16("len_xn", instr->trans_loc_ms_loc_ms.len_xn);
        U16("ch", instr->trans_loc_ms_loc_ms.ch);
        U16("clear", instr->trans_loc_ms_loc_ms.clear);
        U16("len_en", instr->trans_loc_ms_loc_ms.len_en);
        U16("set_id", instr->trans_loc_ms_loc_ms.set_id);
        HEX("set_mask", instr->trans_loc_ms_loc_ms.set_mask);
        U16("wait_id", instr->trans_loc_ms_loc_ms.wait_id);
        HEX("wait_mask", instr->trans_loc_ms_loc_ms.wait_mask);
        break;
    case CCU_V1_OP_TRANS_RMT_MS_TO_LOC_MS:
        U16("loc_ms", instr->trans_rmt_ms_loc_ms.loc_ms);
        U16("rmt_ms", instr->trans_rmt_ms_loc_ms.rmt_ms);
        U16("len_xn", instr->trans_rmt_ms_loc_ms.len_xn);
        U16("ch", instr->trans_rmt_ms_loc_ms.ch);
        U16("clear", instr->trans_rmt_ms_loc_ms.clear);
        U16("len_en", instr->trans_rmt_ms_loc_ms.len_en);
        U16("set_id", instr->trans_rmt_ms_loc_ms.set_id);
        HEX("set_mask", instr->trans_rmt_ms_loc_ms.set_mask);
        U16("wait_id", instr->trans_rmt_ms_loc_ms.wait_id);
        HEX("wait_mask", instr->trans_rmt_ms_loc_ms.wait_mask);
        break;
    case CCU_V1_OP_TRANS_LOC_MS_TO_RMT_MS:
        U16("rmt_ms", instr->trans_loc_ms_rmt_ms.rmt_ms);
        U16("loc_ms", instr->trans_loc_ms_rmt_ms.loc_ms);
        U16("len_xn", instr->trans_loc_ms_rmt_ms.len_xn);
        U16("ch", instr->trans_loc_ms_rmt_ms.ch);
        U16("rmt_set_id", instr->trans_loc_ms_rmt_ms.rmt_set_id);
        HEX("rmt_set_mask", instr->trans_loc_ms_rmt_ms.rmt_set_mask);
        U16("clear", instr->trans_loc_ms_rmt_ms.clear);
        U16("len_en", instr->trans_loc_ms_rmt_ms.len_en);
        U16("set_id", instr->trans_loc_ms_rmt_ms.set_id);
        HEX("set_mask", instr->trans_loc_ms_rmt_ms.set_mask);
        U16("wait_id", instr->trans_loc_ms_rmt_ms.wait_id);
        HEX("wait_mask", instr->trans_loc_ms_rmt_ms.wait_mask);
        break;
    case CCU_V1_OP_TRANS_RMT_MEM_TO_LOC_MEM:
        U16("loc_gsa", instr->trans_rmt_mem_loc_mem.loc_gsa);
        U16("loc_xn", instr->trans_rmt_mem_loc_mem.loc_xn);
        U16("rmt_gsa", instr->trans_rmt_mem_loc_mem.rmt_gsa);
        U16("rmt_xn", instr->trans_rmt_mem_loc_mem.rmt_xn);
        U16("len_xn", instr->trans_rmt_mem_loc_mem.len_xn);
        U16("ch", instr->trans_rmt_mem_loc_mem.ch);
        U16("udf", instr->trans_rmt_mem_loc_mem.udf);
        U16("reduce_dtype", instr->trans_rmt_mem_loc_mem.reduce_dtype);
        U16("reduce_op", instr->trans_rmt_mem_loc_mem.reduce_op);
        U16("clear", instr->trans_rmt_mem_loc_mem.clear);
        U16("len_en", instr->trans_rmt_mem_loc_mem.len_en);
        U16("reduce_en", instr->trans_rmt_mem_loc_mem.reduce_en);
        U16("set_id", instr->trans_rmt_mem_loc_mem.set_id);
        HEX("set_mask", instr->trans_rmt_mem_loc_mem.set_mask);
        U16("wait_id", instr->trans_rmt_mem_loc_mem.wait_id);
        HEX("wait_mask", instr->trans_rmt_mem_loc_mem.wait_mask);
        break;
    case CCU_V1_OP_TRANS_LOC_MEM_TO_RMT_MEM:
        U16("rmt_gsa", instr->trans_loc_mem_rmt_mem.rmt_gsa);
        U16("rmt_xn", instr->trans_loc_mem_rmt_mem.rmt_xn);
        U16("loc_gsa", instr->trans_loc_mem_rmt_mem.loc_gsa);
        U16("loc_xn", instr->trans_loc_mem_rmt_mem.loc_xn);
        U16("len_xn", instr->trans_loc_mem_rmt_mem.len_xn);
        U16("ch", instr->trans_loc_mem_rmt_mem.ch);
        U16("udf", instr->trans_loc_mem_rmt_mem.udf);
        U16("reduce_dtype", instr->trans_loc_mem_rmt_mem.reduce_dtype);
        U16("reduce_op", instr->trans_loc_mem_rmt_mem.reduce_op);
        U16("clear", instr->trans_loc_mem_rmt_mem.clear);
        U16("len_en", instr->trans_loc_mem_rmt_mem.len_en);
        U16("reduce_en", instr->trans_loc_mem_rmt_mem.reduce_en);
        U16("set_id", instr->trans_loc_mem_rmt_mem.set_id);
        HEX("set_mask", instr->trans_loc_mem_rmt_mem.set_mask);
        U16("wait_id", instr->trans_loc_mem_rmt_mem.wait_id);
        HEX("wait_mask", instr->trans_loc_mem_rmt_mem.wait_mask);
        break;
    case CCU_V1_OP_TRANS_LOC_MEM_TO_LOC_MEM:
        U16("dst_gsa", instr->trans_loc_mem_loc_mem.dst_gsa);
        U16("dst_xn", instr->trans_loc_mem_loc_mem.dst_xn);
        U16("src_gsa", instr->trans_loc_mem_loc_mem.src_gsa);
        U16("src_xn", instr->trans_loc_mem_loc_mem.src_xn);
        U16("len_xn", instr->trans_loc_mem_loc_mem.len_xn);
        U16("ch", instr->trans_loc_mem_loc_mem.ch);
        U16("clear", instr->trans_loc_mem_loc_mem.clear);
        U16("len_en", instr->trans_loc_mem_loc_mem.len_en);
        U16("set_id", instr->trans_loc_mem_loc_mem.set_id);
        HEX("set_mask", instr->trans_loc_mem_loc_mem.set_mask);
        U16("wait_id", instr->trans_loc_mem_loc_mem.wait_id);
        HEX("wait_mask", instr->trans_loc_mem_loc_mem.wait_mask);
        break;
    case CCU_V1_OP_SYNC_CKE:
        U16("rmt_cke", instr->sync_cke.rmt_cke);
        U16("loc_cke", instr->sync_cke.loc_cke);
        HEX("loc_mask", instr->sync_cke.loc_mask);
        U16("ch", instr->sync_cke.ch);
        U16("clear", instr->sync_cke.clear);
        U16("set_id", instr->sync_cke.set_id);
        HEX("set_mask", instr->sync_cke.set_mask);
        U16("wait_id", instr->sync_cke.wait_id);
        HEX("wait_mask", instr->sync_cke.wait_mask);
        break;
    case CCU_V1_OP_SYNC_GSA:
        U16("rmt_gsa", instr->sync_gsa.rmt_gsa);
        U16("loc_gsa", instr->sync_gsa.loc_gsa);
        U16("ch", instr->sync_gsa.ch);
        U16("rmt_set_id", instr->sync_gsa.rmt_set_id);
        HEX("rmt_set_mask", instr->sync_gsa.rmt_set_mask);
        U16("clear", instr->sync_gsa.clear);
        U16("set_id", instr->sync_gsa.set_id);
        HEX("set_mask", instr->sync_gsa.set_mask);
        U16("wait_id", instr->sync_gsa.wait_id);
        HEX("wait_mask", instr->sync_gsa.wait_mask);
        break;
    case CCU_V1_OP_SYNC_XN:
        U16("rmt_xn", instr->sync_xn.rmt_xn);
        U16("loc_xn", instr->sync_xn.loc_xn);
        U16("ch", instr->sync_xn.ch);
        U16("rmt_set_id", instr->sync_xn.rmt_set_id);
        HEX("rmt_set_mask", instr->sync_xn.rmt_set_mask);
        U16("clear", instr->sync_xn.clear);
        U16("set_id", instr->sync_xn.set_id);
        HEX("set_mask", instr->sync_xn.set_mask);
        U16("wait_id", instr->sync_xn.wait_id);
        HEX("wait_mask", instr->sync_xn.wait_mask);
        break;
    case CCU_V1_OP_ADD: {
        char msbuf[128];
        fmt_ms(msbuf, sizeof(msbuf), instr->add.ms);
        if (SEP() != 0 || appendf(buf, buf_sz, &pos, "ms=%s", msbuf) != 0) {
            return -1;
        }
        U16("count", instr->add.count);
        U16("cast", instr->add.cast);
        U16("dtype", instr->add.dtype);
        U16("len_xn", instr->add.len_xn);
        U16("clear", instr->add.clear);
        U16("set_id", instr->add.set_id);
        HEX("set_mask", instr->add.set_mask);
        U16("wait_id", instr->add.wait_id);
        HEX("wait_mask", instr->add.wait_mask);
        break;
    }
    case CCU_V1_OP_MAX:
    case CCU_V1_OP_MIN: {
        char msbuf[128];
        fmt_ms(msbuf, sizeof(msbuf), instr->maxmin.ms);
        if (SEP() != 0 || appendf(buf, buf_sz, &pos, "ms=%s", msbuf) != 0) {
            return -1;
        }
        U16("count", instr->maxmin.count);
        U16("dtype", instr->maxmin.dtype);
        U16("len_xn", instr->maxmin.len_xn);
        U16("clear", instr->maxmin.clear);
        U16("set_id", instr->maxmin.set_id);
        HEX("set_mask", instr->maxmin.set_mask);
        U16("wait_id", instr->maxmin.wait_id);
        HEX("wait_mask", instr->maxmin.wait_mask);
        break;
    }
    default:
        return -1;
    }
#undef SEP
#undef U16
#undef HEX
    if (pos >= (int)buf_sz) {
        return -1;
    }
    return pos;
}

int ccu_v1_disassemble_program(const CcuV1Program *prog, FILE *out)
{
    char line[1024];
    for (size_t i = 0; i < prog->count; ++i) {
        if (ccu_v1_format_instr(&prog->items[i], line, sizeof(line)) < 0) {
            return -1;
        }
        if (fprintf(out, "%s\n", line) < 0) {
            return -1;
        }
    }
    return 0;
}

int ccu_v1_program_from_binary(const uint8_t *data, size_t len, CcuV1Program *out, char *errmsg, size_t errmsg_sz)
{
    ccu_v1_program_init(out);
    if (len % CCU_V1_INSTR_SIZE != 0) {
        snprintf(errmsg, errmsg_sz, "binary size %zu not multiple of %d", len, CCU_V1_INSTR_SIZE);
        return -1;
    }
    size_t n = len / CCU_V1_INSTR_SIZE;
    if (ccu_v1_program_reserve(out, n) != 0) {
        snprintf(errmsg, errmsg_sz, "oom");
        return -1;
    }
    for (size_t i = 0; i < n; ++i) {
        memcpy(&out->items[i], data + i * CCU_V1_INSTR_SIZE, CCU_V1_INSTR_SIZE);
        uint8_t type;
        uint16_t code;
        ccu_v1_split_header(out->items[i].header.raw, &type, &code);
        if (!ccu_v1_lookup_opcode(type, code)) {
            snprintf(errmsg, errmsg_sz, "unknown opcode at instr %zu: type=%u code=0x%x", i, type, code);
            ccu_v1_program_free(out);
            return -1;
        }
    }
    out->count = n;
    return 0;
}

int ccu_v1_program_to_binary(const CcuV1Program *prog, uint8_t **out_data, size_t *out_len)
{
    size_t len = prog->count * CCU_V1_INSTR_SIZE;
    uint8_t *buf = (uint8_t *)malloc(len ? len : 1);
    if (!buf) {
        return -1;
    }
    for (size_t i = 0; i < prog->count; ++i) {
        memcpy(buf + i * CCU_V1_INSTR_SIZE, &prog->items[i], CCU_V1_INSTR_SIZE);
    }
    *out_data = buf;
    *out_len = len;
    return 0;
}

int ccu_v1_program_semantic_eq(const CcuV1Program *a, const CcuV1Program *b)
{
    if (a->count != b->count) {
        return -1;
    }
    /* Canonical compare via re-format + re-parse would work; byte compare of
     * packed structs is stronger and matches reserved=0 encoding. */
    for (size_t i = 0; i < a->count; ++i) {
        if (memcmp(&a->items[i], &b->items[i], CCU_V1_INSTR_SIZE) != 0) {
            return -1;
        }
    }
    return 0;
}
