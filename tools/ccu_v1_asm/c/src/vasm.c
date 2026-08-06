#include "ccu_v1_vasm.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ccu_v1_vasm_config_default(CcuV1VasmConfig *cfg)
{
    cfg->limits[CCU_V1_RES_XN] = 256;
    cfg->limits[CCU_V1_RES_GSA] = 64;
    cfg->limits[CCU_V1_RES_MS] = 32;
    cfg->limits[CCU_V1_RES_CKE] = 64;
    cfg->limits[CCU_V1_RES_CH] = 32;
    cfg->limits[CCU_V1_RES_SQE] = 32;
}

void ccu_v1_vasm_result_init(CcuV1VasmResult *r)
{
    memset(r, 0, sizeof(*r));
    ccu_v1_program_init(&r->program);
    ccu_v1_vasm_config_default(&r->cfg);
}

void ccu_v1_vasm_result_free(CcuV1VasmResult *r)
{
    free(r->vars);
    free(r->lowered_asm);
    ccu_v1_program_free(&r->program);
    memset(r, 0, sizeof(*r));
}

const char *ccu_v1_res_type_name(CcuV1ResType t)
{
    static const char *names[] = {"xn", "gsa", "ms", "cke", "ch", "sqe"};
    if ((unsigned)t >= CCU_V1_RES_COUNT) {
        return "?";
    }
    return names[t];
}

int ccu_v1_res_type_parse(const char *s, CcuV1ResType *out)
{
    for (int i = 0; i < CCU_V1_RES_COUNT; ++i) {
        if (strcmp(s, ccu_v1_res_type_name((CcuV1ResType)i)) == 0) {
            *out = (CcuV1ResType)i;
            return 0;
        }
    }
    return -1;
}

/* Classify operand field name → resource type, or -1 if immediate/literal field. */
static int field_res_type(const char *field)
{
    if (!strcmp(field, "xn") || !strcmp(field, "xd") || !strcmp(field, "xm") || !strcmp(field, "len_xn") ||
        !strcmp(field, "dst_xn") || !strcmp(field, "cond_xn") || !strcmp(field, "loc_xn") ||
        !strcmp(field, "rmt_xn") || !strcmp(field, "src_xn")) {
        return CCU_V1_RES_XN;
    }
    if (!strcmp(field, "gsa") || !strcmp(field, "gsad") || !strcmp(field, "gsam") || !strcmp(field, "gsan") ||
        !strcmp(field, "loc_gsa") || !strcmp(field, "rmt_gsa") || !strcmp(field, "dst_gsa") ||
        !strcmp(field, "src_gsa")) {
        return CCU_V1_RES_GSA;
    }
    if (!strcmp(field, "ms") || !strcmp(field, "dst_ms") || !strcmp(field, "src_ms") || !strcmp(field, "loc_ms") ||
        !strcmp(field, "rmt_ms")) {
        return CCU_V1_RES_MS;
    }
    if (!strcmp(field, "set_id") || !strcmp(field, "wait_id") || !strcmp(field, "clear_id") ||
        !strcmp(field, "rmt_set_id") || !strcmp(field, "rmt_cke") || !strcmp(field, "loc_cke")) {
        return CCU_V1_RES_CKE;
    }
    if (!strcmp(field, "ch")) {
        return CCU_V1_RES_CH;
    }
    if (!strcmp(field, "sqe")) {
        return CCU_V1_RES_SQE;
    }
    return -1; /* imm / mask / flag / instr index */
}

static const char *skip_ws(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\r') {
        ++p;
    }
    return p;
}

static int is_ident_start(char c)
{
    return isalpha((unsigned char)c) || c == '_';
}

static int is_ident(char c)
{
    return isalnum((unsigned char)c) || c == '_';
}

static int is_number_token(const char *s, const char *end)
{
    if (s >= end) {
        return 0;
    }
    if (s[0] == '0' && end - s >= 2 && (s[1] == 'x' || s[1] == 'X')) {
        return 1;
    }
    if (s[0] >= '0' && s[0] <= '9') {
        return 1;
    }
    return 0;
}

/* ---- front-end IR ---- */

#define MAX_OPS_PER_INSN 24
#define MAX_MS_ELEMS 8
#define MAX_NAME 64

typedef enum { OP_IMM = 0, OP_SYM = 1, OP_MSLIST = 2 } OpValKind;

typedef struct {
    char field[MAX_NAME];
    OpValKind kind;
    uint64_t imm;
    char sym[MAX_NAME];
    char ms_syms[MAX_MS_ELEMS][MAX_NAME];
    int ms_is_imm[MAX_MS_ELEMS];
    uint16_t ms_imm[MAX_MS_ELEMS];
    int ms_n;
} VasmOp;

typedef struct {
    const CcuV1OpcodeDesc *desc;
    VasmOp ops[MAX_OPS_PER_INSN];
    int op_count;
    int lineno;
} VasmInsn;

typedef struct {
    VasmInsn *insns;
    size_t count;
    size_t cap;
} VasmIr;

static void ir_init(VasmIr *ir)
{
    ir->insns = NULL;
    ir->count = 0;
    ir->cap = 0;
}

static void ir_free(VasmIr *ir)
{
    free(ir->insns);
    ir->insns = NULL;
    ir->count = 0;
    ir->cap = 0;
}

static int ir_push(VasmIr *ir, const VasmInsn *insn)
{
    if (ir->count >= ir->cap) {
        size_t ncap = ir->cap ? ir->cap * 2 : 2048;
        VasmInsn *ni = (VasmInsn *)realloc(ir->insns, ncap * sizeof(VasmInsn));
        if (!ni) {
            return -1;
        }
        ir->insns = ni;
        ir->cap = ncap;
    }
    ir->insns[ir->count++] = *insn;
    return 0;
}

static int var_find(CcuV1VasmResult *r, const char *name)
{
    for (size_t i = 0; i < r->var_count; ++i) {
        if (strcmp(r->vars[i].name, name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int var_add(CcuV1VasmResult *r, const char *name, CcuV1ResType type, int pinned, int16_t pin_id,
                   char *errmsg, size_t errmsg_sz)
{
    int idx = var_find(r, name);
    if (idx >= 0) {
        if (r->vars[idx].type != type) {
            snprintf(errmsg, errmsg_sz, "variable '%s' type conflict: %s vs %s", name,
                     ccu_v1_res_type_name(r->vars[idx].type), ccu_v1_res_type_name(type));
            return -1;
        }
        if (pinned) {
            if (r->vars[idx].pinned && r->vars[idx].id != pin_id) {
                snprintf(errmsg, errmsg_sz, "variable '%s' pinned id conflict", name);
                return -1;
            }
            r->vars[idx].pinned = 1;
            r->vars[idx].id = pin_id;
        }
        return idx;
    }
    if (r->var_count >= r->var_cap) {
        size_t ncap = r->var_cap ? r->var_cap * 2 : 128;
        CcuV1VarInfo *nv = (CcuV1VarInfo *)realloc(r->vars, ncap * sizeof(CcuV1VarInfo));
        if (!nv) {
            snprintf(errmsg, errmsg_sz, "oom");
            return -1;
        }
        r->vars = nv;
        r->var_cap = ncap;
    }
    CcuV1VarInfo *v = &r->vars[r->var_count];
    memset(v, 0, sizeof(*v));
    snprintf(v->name, sizeof(v->name), "%s", name);
    v->type = type;
    v->pinned = pinned;
    v->id = pinned ? pin_id : (int16_t)-1;
    v->live_start = -1;
    v->live_end = -1;
    v->use_count = 0;
    return (int)r->var_count++;
}

static int note_use(CcuV1VasmResult *r, const char *name, CcuV1ResType type, int insn_idx, char *errmsg,
                    size_t errmsg_sz)
{
    int idx = var_add(r, name, type, 0, -1, errmsg, errmsg_sz);
    if (idx < 0) {
        return -1;
    }
    CcuV1VarInfo *v = &r->vars[idx];
    if (v->live_start < 0 || insn_idx < v->live_start) {
        v->live_start = insn_idx;
    }
    if (insn_idx > v->live_end) {
        v->live_end = insn_idx;
    }
    v->use_count++;
    return idx;
}

static int parse_decl(const char *line, CcuV1VasmResult *r, char *errmsg, size_t errmsg_sz)
{
    const char *p = skip_ws(line);
    if (*p != '.') {
        return 1; /* not a decl */
    }
    ++p;
    char tname[32];
    size_t ti = 0;
    while (*p && is_ident(*p) && ti + 1 < sizeof(tname)) {
        tname[ti++] = *p++;
    }
    tname[ti] = '\0';
    p = skip_ws(p);

    CcuV1ResType type;
    if (!strcmp(tname, "var")) {
        char rt[32];
        size_t ri = 0;
        while (*p && is_ident(*p) && ri + 1 < sizeof(rt)) {
            rt[ri++] = *p++;
        }
        rt[ri] = '\0';
        if (ccu_v1_res_type_parse(rt, &type) != 0) {
            snprintf(errmsg, errmsg_sz, "unknown resource type '%s'", rt);
            return -1;
        }
        p = skip_ws(p);
    } else {
        if (ccu_v1_res_type_parse(tname, &type) != 0) {
            snprintf(errmsg, errmsg_sz, "unknown directive '.%s'", tname);
            return -1;
        }
    }

    if (!is_ident_start(*p)) {
        snprintf(errmsg, errmsg_sz, "expected variable name after .%s", tname);
        return -1;
    }
    char name[MAX_NAME];
    size_t ni = 0;
    while (*p && is_ident(*p) && ni + 1 < sizeof(name)) {
        name[ni++] = *p++;
    }
    name[ni] = '\0';
    p = skip_ws(p);

    int pinned = 0;
    int16_t pin_id = -1;
    if (*p == '=') {
        ++p;
        p = skip_ws(p);
        char *end = NULL;
        unsigned long v = strtoul(p, &end, 0);
        if (end == p || v > 0xFFFFul) {
            snprintf(errmsg, errmsg_sz, "bad pinned id for %s", name);
            return -1;
        }
        pinned = 1;
        pin_id = (int16_t)v;
        p = skip_ws(end);
    }
    if (*p && *p != '#') {
        snprintf(errmsg, errmsg_sz, "trailing junk in declaration of %s", name);
        return -1;
    }
    if (var_add(r, name, type, pinned, pin_id, errmsg, errmsg_sz) < 0) {
        return -1;
    }
    return 0;
}

static int parse_op_value(const char *v0, const char *v1, VasmOp *op, int expect_ms_list, char *errmsg,
                          size_t errmsg_sz)
{
    const char *s = skip_ws(v0);
    const char *e = v1;
    while (e > s && (e[-1] == ' ' || e[-1] == '\t')) {
        --e;
    }
    if (s >= e) {
        snprintf(errmsg, errmsg_sz, "empty operand value");
        return -1;
    }
    if (*s == '[') {
        op->kind = OP_MSLIST;
        op->ms_n = 0;
        const char *p = s + 1;
        p = skip_ws(p);
        if (p < e && *p == ']') {
            return 0;
        }
        while (p < e) {
            const char *a = p;
            while (p < e && *p != ',' && *p != ']') {
                ++p;
            }
            const char *b = p;
            while (b > a && (b[-1] == ' ' || b[-1] == '\t')) {
                --b;
            }
            if (op->ms_n >= MAX_MS_ELEMS) {
                snprintf(errmsg, errmsg_sz, "ms list too long");
                return -1;
            }
            if (is_number_token(a, b)) {
                char *end = NULL;
                unsigned long v = strtoul(a, &end, 0);
                if (end != b || v > 0xFFFFul) {
                    snprintf(errmsg, errmsg_sz, "bad ms list number");
                    return -1;
                }
                op->ms_is_imm[op->ms_n] = 1;
                op->ms_imm[op->ms_n] = (uint16_t)v;
                op->ms_syms[op->ms_n][0] = '\0';
            } else {
                if (!is_ident_start(*a)) {
                    snprintf(errmsg, errmsg_sz, "bad ms list element");
                    return -1;
                }
                size_t n = (size_t)(b - a);
                if (n >= MAX_NAME) {
                    snprintf(errmsg, errmsg_sz, "symbol too long");
                    return -1;
                }
                memcpy(op->ms_syms[op->ms_n], a, n);
                op->ms_syms[op->ms_n][n] = '\0';
                op->ms_is_imm[op->ms_n] = 0;
            }
            op->ms_n++;
            p = skip_ws(p);
            if (p < e && *p == ',') {
                ++p;
                p = skip_ws(p);
                continue;
            }
            if (p < e && *p == ']') {
                return 0;
            }
            snprintf(errmsg, errmsg_sz, "bad ms list syntax");
            return -1;
        }
        snprintf(errmsg, errmsg_sz, "unclosed ms list");
        return -1;
    }
    (void)expect_ms_list;
    if (is_number_token(s, e)) {
        char *end = NULL;
        unsigned long long v = strtoull(s, &end, 0);
        if (end != e) {
            snprintf(errmsg, errmsg_sz, "bad numeric operand");
            return -1;
        }
        op->kind = OP_IMM;
        op->imm = (uint64_t)v;
        return 0;
    }
    /* optional @name */
    if (*s == '@') {
        ++s;
    }
    if (!is_ident_start(*s)) {
        snprintf(errmsg, errmsg_sz, "expected identifier or number");
        return -1;
    }
    size_t n = (size_t)(e - s);
    if (n >= MAX_NAME) {
        snprintf(errmsg, errmsg_sz, "symbol too long");
        return -1;
    }
    op->kind = OP_SYM;
    memcpy(op->sym, s, n);
    op->sym[n] = '\0';
    return 0;
}

static int parse_insn_line(const char *line, int lineno, VasmInsn *out, char *errmsg, size_t errmsg_sz)
{
    memset(out, 0, sizeof(*out));
    out->lineno = lineno;
    const char *p = skip_ws(line);
    if (*p == '\0' || *p == '#' || *p == '.') {
        return 1;
    }
    /* optional label */
    const char *q = p;
    while (*q && is_ident(*q)) {
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
    while (*p && is_ident(*p) && mi + 1 < sizeof(mnem)) {
        mnem[mi++] = (char)toupper((unsigned char)*p++);
    }
    mnem[mi] = '\0';
    out->desc = ccu_v1_lookup_mnemonic(mnem);
    if (!out->desc) {
        snprintf(errmsg, errmsg_sz, "line %d: unknown mnemonic %s", lineno, mnem);
        return -1;
    }
    p = skip_ws(p);
    while (*p && *p != '#') {
        if (out->op_count >= MAX_OPS_PER_INSN) {
            snprintf(errmsg, errmsg_sz, "line %d: too many operands", lineno);
            return -1;
        }
        VasmOp *op = &out->ops[out->op_count];
        memset(op, 0, sizeof(*op));
        const char *n0 = p;
        while (*p && is_ident(*p)) {
            ++p;
        }
        if (p == n0 || *p != '=') {
            snprintf(errmsg, errmsg_sz, "line %d: expected name=value", lineno);
            return -1;
        }
        size_t nlen = (size_t)(p - n0);
        if (nlen >= MAX_NAME) {
            snprintf(errmsg, errmsg_sz, "line %d: field name too long", lineno);
            return -1;
        }
        memcpy(op->field, n0, nlen);
        op->field[nlen] = '\0';
        ++p;
        p = skip_ws(p);
        const char *v0 = p;
        if (*p == '[') {
            while (*p && *p != ']') {
                ++p;
            }
            if (*p != ']') {
                snprintf(errmsg, errmsg_sz, "line %d: unclosed list", lineno);
                return -1;
            }
            ++p;
        } else {
            while (*p && *p != ',' && *p != '#' && *p != ' ' && *p != '\t') {
                ++p;
            }
        }
        char local[128];
        if (parse_op_value(v0, p, op, 0, local, sizeof(local)) != 0) {
            snprintf(errmsg, errmsg_sz, "line %d: %s", lineno, local);
            return -1;
        }
        out->op_count++;
        p = skip_ws(p);
        if (*p == ',') {
            ++p;
            p = skip_ws(p);
            continue;
        }
        if (*p == '\0' || *p == '#') {
            break;
        }
        snprintf(errmsg, errmsg_sz, "line %d: unexpected text", lineno);
        return -1;
    }
    return 0;
}

/* ---- liveness + linear-scan allocation ---- */

typedef struct {
    int var_idx;
    int start;
    int end;
} Interval;

static int cmp_interval_start(const void *a, const void *b)
{
    const Interval *ia = (const Interval *)a;
    const Interval *ib = (const Interval *)b;
    if (ia->start != ib->start) {
        return ia->start - ib->start;
    }
    return ia->var_idx - ib->var_idx;
}

static int allocate_ids(CcuV1VasmResult *r, char *errmsg, size_t errmsg_sz)
{
    for (int t = 0; t < CCU_V1_RES_COUNT; ++t) {
        uint16_t limit = r->cfg.limits[t];
        if (limit == 0) {
            snprintf(errmsg, errmsg_sz, "resource limit for %s is 0", ccu_v1_res_type_name((CcuV1ResType)t));
            return -1;
        }

        /* Collect intervals of this type */
        Interval *iv = NULL;
        size_t niv = 0, civ = 0;
        uint8_t *pinned_busy = (uint8_t *)calloc(limit, 1);
        if (!pinned_busy) {
            snprintf(errmsg, errmsg_sz, "oom");
            return -1;
        }

        for (size_t i = 0; i < r->var_count; ++i) {
            CcuV1VarInfo *v = &r->vars[i];
            if ((int)v->type != t) {
                continue;
            }
            if (v->live_start < 0) {
                /* declared but unused — still assign if pinned, else skip */
                if (v->pinned) {
                    if ((uint16_t)v->id >= limit) {
                        snprintf(errmsg, errmsg_sz, "pinned id %d for %s exceeds limit %u", v->id, v->name, limit);
                        free(pinned_busy);
                        free(iv);
                        return -1;
                    }
                }
                continue;
            }
            if (v->pinned) {
                if ((uint16_t)v->id >= limit) {
                    snprintf(errmsg, errmsg_sz, "pinned id %d for %s exceeds limit %u", v->id, v->name, limit);
                    free(pinned_busy);
                    free(iv);
                    return -1;
                }
                continue;
            }
            if (niv >= civ) {
                size_t ncap = civ ? civ * 2 : 64;
                Interval *ni = (Interval *)realloc(iv, ncap * sizeof(Interval));
                if (!ni) {
                    free(pinned_busy);
                    free(iv);
                    snprintf(errmsg, errmsg_sz, "oom");
                    return -1;
                }
                iv = ni;
                civ = ncap;
            }
            iv[niv].var_idx = (int)i;
            iv[niv].start = v->live_start;
            iv[niv].end = v->live_end;
            niv++;
        }

        qsort(iv, niv, sizeof(Interval), cmp_interval_start);

        /* Active list of (end, id, var_idx) — simple array */
        typedef struct {
            int end;
            uint16_t id;
        } Active;
        Active *active = (Active *)calloc(limit, sizeof(Active));
        size_t nactive = 0;
        uint8_t *in_use = (uint8_t *)calloc(limit, 1);
        if (!active || !in_use) {
            free(pinned_busy);
            free(iv);
            free(active);
            free(in_use);
            snprintf(errmsg, errmsg_sz, "oom");
            return -1;
        }

        /* Mark pinned IDs as permanently reserved across their live ranges via
         * a timeline check: before assigning, ensure no pinned overlap. */
        uint16_t peak = 0;

        for (size_t i = 0; i < niv; ++i) {
            int cur = iv[i].start;
            /* expire actives that ended before cur (no overlap at cur) */
            size_t w = 0;
            for (size_t a = 0; a < nactive; ++a) {
                if (active[a].end < cur) {
                    in_use[active[a].id] = 0;
                } else {
                    active[w++] = active[a];
                }
            }
            nactive = w;

            /* collect pinned IDs live at cur */
            memset(pinned_busy, 0, limit);
            for (size_t vi = 0; vi < r->var_count; ++vi) {
                CcuV1VarInfo *pv = &r->vars[vi];
                if ((int)pv->type != t || !pv->pinned || pv->live_start < 0) {
                    continue;
                }
                if (pv->live_start <= cur && cur <= pv->live_end) {
                    pinned_busy[pv->id] = 1;
                }
            }

            int chosen = -1;
            for (uint16_t id = 0; id < limit; ++id) {
                if (!in_use[id] && !pinned_busy[id]) {
                    chosen = (int)id;
                    break;
                }
            }
            if (chosen < 0) {
                snprintf(errmsg, errmsg_sz, "out of %s IDs (limit %u) while allocating '%s'",
                         ccu_v1_res_type_name((CcuV1ResType)t), limit, r->vars[iv[i].var_idx].name);
                free(pinned_busy);
                free(iv);
                free(active);
                free(in_use);
                return -1;
            }
            r->vars[iv[i].var_idx].id = (int16_t)chosen;
            in_use[chosen] = 1;
            active[nactive].end = iv[i].end;
            active[nactive].id = (uint16_t)chosen;
            nactive++;
            if ((uint16_t)(chosen + 1) > peak) {
                peak = (uint16_t)(chosen + 1);
            }
        }

        /* also account for pinned peaks */
        for (size_t vi = 0; vi < r->var_count; ++vi) {
            CcuV1VarInfo *pv = &r->vars[vi];
            if ((int)pv->type == t && pv->pinned && pv->id >= 0 && (uint16_t)(pv->id + 1) > peak) {
                peak = (uint16_t)(pv->id + 1);
            }
            if ((int)pv->type == t && pv->pinned && pv->live_start < 0) {
                /* unused pinned still consumes id number in metainfo */
                if ((uint16_t)(pv->id + 1) > peak) {
                    peak = (uint16_t)(pv->id + 1);
                }
            }
        }
        r->used_peak[t] = peak;

        free(pinned_busy);
        free(iv);
        free(active);
        free(in_use);
    }

    /* Ensure every used var has an id */
    for (size_t i = 0; i < r->var_count; ++i) {
        if (r->vars[i].live_start >= 0 && r->vars[i].id < 0) {
            snprintf(errmsg, errmsg_sz, "internal: var %s has no id", r->vars[i].name);
            return -1;
        }
    }
    return 0;
}

static int resolve_sym(CcuV1VasmResult *r, const char *name, uint64_t *out, char *errmsg, size_t errmsg_sz)
{
    int idx = var_find(r, name);
    if (idx < 0 || r->vars[idx].id < 0) {
        snprintf(errmsg, errmsg_sz, "unresolved variable '%s'", name);
        return -1;
    }
    *out = (uint64_t)r->vars[idx].id;
    return 0;
}

static int lower_to_text(CcuV1VasmResult *r, const VasmIr *ir, char *errmsg, size_t errmsg_sz)
{
    size_t cap = 4096;
    char *buf = (char *)malloc(cap);
    if (!buf) {
        snprintf(errmsg, errmsg_sz, "oom");
        return -1;
    }
    size_t len = 0;

    for (size_t i = 0; i < ir->count; ++i) {
        const VasmInsn *ins = &ir->insns[i];
        char line[2048];
        int pos = 0;
        pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "%s", ins->desc->mnemonic);
        for (int o = 0; o < ins->op_count; ++o) {
            const VasmOp *op = &ins->ops[o];
            pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "%s%s=", o ? ", " : " ", op->field);
            if (op->kind == OP_IMM) {
                if (!strcmp(op->field, "imm") || !strcmp(op->field, "expect") || strstr(op->field, "mask")) {
                    pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "0x%llx", (unsigned long long)op->imm);
                } else {
                    pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "%llu", (unsigned long long)op->imm);
                }
            } else if (op->kind == OP_SYM) {
                uint64_t id;
                if (resolve_sym(r, op->sym, &id, errmsg, errmsg_sz) != 0) {
                    free(buf);
                    return -1;
                }
                pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "%llu", (unsigned long long)id);
            } else {
                pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "[");
                for (int m = 0; m < MAX_MS_ELEMS; ++m) {
                    if (m) {
                        pos += snprintf(line + pos, sizeof(line) - (size_t)pos, ",");
                    }
                    if (m < op->ms_n) {
                        if (op->ms_is_imm[m]) {
                            pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "%u", op->ms_imm[m]);
                        } else {
                            uint64_t id;
                            if (resolve_sym(r, op->ms_syms[m], &id, errmsg, errmsg_sz) != 0) {
                                free(buf);
                                return -1;
                            }
                            pos +=
                                snprintf(line + pos, sizeof(line) - (size_t)pos, "%llu", (unsigned long long)id);
                        }
                    } else {
                        pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "0");
                    }
                }
                pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "]");
            }
            if (pos >= (int)sizeof(line) - 4) {
                snprintf(errmsg, errmsg_sz, "lowered line too long at instr %zu", i);
                free(buf);
                return -1;
            }
        }
        pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "\n");
        if (len + (size_t)pos + 1 > cap) {
            while (len + (size_t)pos + 1 > cap) {
                cap *= 2;
            }
            char *nb = (char *)realloc(buf, cap);
            if (!nb) {
                free(buf);
                snprintf(errmsg, errmsg_sz, "oom");
                return -1;
            }
            buf = nb;
        }
        memcpy(buf + len, line, (size_t)pos);
        len += (size_t)pos;
    }
    buf[len] = '\0';
    r->lowered_asm = buf;
    return 0;
}

static int collect_uses(CcuV1VasmResult *r, const VasmIr *ir, char *errmsg, size_t errmsg_sz)
{
    for (size_t i = 0; i < ir->count; ++i) {
        const VasmInsn *ins = &ir->insns[i];
        for (int o = 0; o < ins->op_count; ++o) {
            const VasmOp *op = &ins->ops[o];
            int rtype = field_res_type(op->field);
            if (op->kind == OP_SYM) {
                if (rtype < 0) {
                    snprintf(errmsg, errmsg_sz, "line %d: symbol not allowed for field %s", ins->lineno, op->field);
                    return -1;
                }
                if (note_use(r, op->sym, (CcuV1ResType)rtype, (int)i, errmsg, errmsg_sz) < 0) {
                    return -1;
                }
            } else if (op->kind == OP_MSLIST) {
                if (rtype != CCU_V1_RES_MS && rtype >= 0) {
                    /* ms list only on ms field */
                }
                for (int m = 0; m < op->ms_n; ++m) {
                    if (!op->ms_is_imm[m]) {
                        if (note_use(r, op->ms_syms[m], CCU_V1_RES_MS, (int)i, errmsg, errmsg_sz) < 0) {
                            return -1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int ccu_v1_vasm_assemble(const char *text, size_t text_len, const CcuV1VasmConfig *cfg, CcuV1VasmResult *out,
                         char *errmsg, size_t errmsg_sz)
{
    ccu_v1_vasm_result_init(out);
    if (cfg) {
        out->cfg = *cfg;
    }

    VasmIr ir;
    ir_init(&ir);

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
            snprintf(errmsg, errmsg_sz, "line %d: too long", lineno);
            ir_free(&ir);
            ccu_v1_vasm_result_free(out);
            return -1;
        }
        memcpy(buf, line, llen);
        buf[llen] = '\0';
        if (llen > 0 && buf[llen - 1] == '\r') {
            buf[llen - 1] = '\0';
        }

        char *hash = strchr(buf, '#');
        if (hash) {
            *hash = '\0';
        }

        const char *sp = skip_ws(buf);
        if (*sp == '\0') {
            /* empty */
        } else if (*sp == '.') {
            char local[256];
            int rc = parse_decl(buf, out, local, sizeof(local));
            if (rc < 0) {
                snprintf(errmsg, errmsg_sz, "line %d: %s", lineno, local);
                ir_free(&ir);
                ccu_v1_vasm_result_free(out);
                return -1;
            }
        } else {
            VasmInsn insn;
            int rc = parse_insn_line(buf, lineno, &insn, errmsg, errmsg_sz);
            if (rc < 0) {
                ir_free(&ir);
                ccu_v1_vasm_result_free(out);
                return -1;
            }
            if (rc == 0) {
                if (ir_push(&ir, &insn) != 0) {
                    snprintf(errmsg, errmsg_sz, "oom");
                    ir_free(&ir);
                    ccu_v1_vasm_result_free(out);
                    return -1;
                }
            }
        }
        if (p < end && *p == '\n') {
            ++p;
        }
        ++lineno;
    }

    out->instr_count = ir.count;
    if (collect_uses(out, &ir, errmsg, errmsg_sz) != 0) {
        ir_free(&ir);
        ccu_v1_vasm_result_free(out);
        return -1;
    }
    if (allocate_ids(out, errmsg, errmsg_sz) != 0) {
        ir_free(&ir);
        ccu_v1_vasm_result_free(out);
        return -1;
    }
    if (lower_to_text(out, &ir, errmsg, errmsg_sz) != 0) {
        ir_free(&ir);
        ccu_v1_vasm_result_free(out);
        return -1;
    }
    if (ccu_v1_assemble_text(out->lowered_asm, strlen(out->lowered_asm), &out->program, errmsg, errmsg_sz) != 0) {
        ir_free(&ir);
        ccu_v1_vasm_result_free(out);
        return -1;
    }
    ir_free(&ir);
    return 0;
}

int ccu_v1_vasm_write_metainfo(const CcuV1VasmResult *r, FILE *out)
{
    fprintf(out, "{\n");
    fprintf(out, "  \"version\": 1,\n");
    fprintf(out, "  \"instr_count\": %zu,\n", r->instr_count);
    fprintf(out, "  \"resources\": {\n");
    for (int t = 0; t < CCU_V1_RES_COUNT; ++t) {
        fprintf(out, "    \"%s\": {\"limit\": %u, \"peak_used\": %u}%s\n", ccu_v1_res_type_name((CcuV1ResType)t),
                (unsigned)r->cfg.limits[t], (unsigned)r->used_peak[t], (t + 1 < CCU_V1_RES_COUNT) ? "," : "");
    }
    fprintf(out, "  },\n");
    fprintf(out, "  \"variables\": [\n");
    for (size_t i = 0; i < r->var_count; ++i) {
        const CcuV1VarInfo *v = &r->vars[i];
        fprintf(out,
                "    {\"name\": \"%s\", \"type\": \"%s\", \"id\": %d, \"pinned\": %s, "
                "\"live_start\": %d, \"live_end\": %d, \"use_count\": %d}%s\n",
                v->name, ccu_v1_res_type_name(v->type), (int)v->id, v->pinned ? "true" : "false", v->live_start,
                v->live_end, v->use_count, (i + 1 < r->var_count) ? "," : "");
    }
    fprintf(out, "  ]\n");
    fprintf(out, "}\n");
    return 0;
}
