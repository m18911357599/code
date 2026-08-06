/**
 * C-style assembler: context + builtins that fill packed CcuV1Instr binary.
 */
#include "ccu_v1_casm.h"

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__thread CcuV1CasmCtx *ccu_v1_casm_tls;

void ccu_v1_casm_init(CcuV1CasmCtx *ctx)
{
    assert(ctx);
    ccu_v1_program_init(&ctx->program);
    ctx->inst = NULL;
    ctx->errmsg[0] = '\0';
    ctx->failed = 0;
}

void ccu_v1_casm_free(CcuV1CasmCtx *ctx)
{
    assert(ctx);
    ccu_v1_program_free(&ctx->program);
    ctx->inst = NULL;
    ctx->errmsg[0] = '\0';
    ctx->failed = 0;
}

void ccu_v1_casm_begin(CcuV1CasmCtx *ctx)
{
    assert(ctx);
    assert(ccu_v1_casm_tls == NULL);
    /* Pre-reserve zeroed slots once — must always run (not inside assert). */
    if (ccu_v1_program_reserve(&ctx->program, CCU_V1_CASM_INIT_CAP) != 0) {
        abort();
    }
    ccu_v1_casm_tls = ctx;
}

void ccu_v1_casm_end(void)
{
    assert(ccu_v1_casm_tls != NULL);
    ccu_v1_casm_tls = NULL;
}

void ccu_v1_casm_run(CcuV1CasmCtx *ctx, void (*entry)(void))
{
    assert(ctx);
    assert(entry);
    ccu_v1_casm_begin(ctx);
    entry();
    ccu_v1_casm_end();
    assert(!ctx->failed);
}

void ccu_v1_casm_grow(CcuV1CasmCtx *ctx)
{
    assert(ctx);
    if (ccu_v1_program_reserve(&ctx->program, ctx->program.count + 1) != 0) {
        abort();
    }
}

int ccu_v1_casm_write_file(const CcuV1CasmCtx *ctx, const char *path)
{
    assert(ctx);
    assert(path);
    FILE *f = fopen(path, "wb");
    if (!f) {
        return -1;
    }
    size_t n = ctx->program.count;
    if (n > 0) {
        if (!ctx->program.items || fwrite(ctx->program.items, CCU_V1_INSTR_SIZE, n, f) != n) {
            fclose(f);
            return -1;
        }
    }
    if (fclose(f) != 0) {
        return -1;
    }
    return 0;
}

static void append_instr(CcuV1CasmCtx *ctx, const CcuV1Instr *instr)
{
    assert(ctx);
    assert(instr);
    CcuV1Instr *dst = ccu_v1_casm_emit(ctx, 0, 0);
    *dst = *instr;
}

static void name_to_mnemonic(const char *name, char *out, size_t out_sz)
{
    size_t i = 0;
    for (; name[i] && i + 1 < out_sz; ++i) {
        char c = name[i];
        if (c >= 'a' && c <= 'z') {
            out[i] = (char)(c - 'a' + 'A');
        } else {
            out[i] = c;
        }
    }
    out[i] = '\0';
}

static int format_arg(const CcuV1CasmArg *a, char *buf, size_t buf_sz)
{
    if (!a->is_list) {
        return snprintf(buf, buf_sz, "%llu", (unsigned long long)a->scalar);
    }
    size_t n = 0;
    if (n + 1 >= buf_sz) {
        return -1;
    }
    buf[n++] = '[';
    for (int i = 0; i < a->list_n; ++i) {
        int w = snprintf(buf + n, buf_sz - n, "%s%u", i ? "," : "", (unsigned)a->list[i]);
        if (w < 0 || (size_t)w >= buf_sz - n) {
            return -1;
        }
        n += (size_t)w;
    }
    if (n + 1 >= buf_sz) {
        return -1;
    }
    buf[n++] = ']';
    buf[n] = '\0';
    return (int)n;
}

int ccu_v1_casm_call(CcuV1CasmCtx *ctx, const char *name, const CcuV1CasmArg *args, int nargs)
{
    if (!ctx || !name) {
        return -1;
    }

    char mnem[64];
    name_to_mnemonic(name, mnem, sizeof(mnem));

    /* Dedicated path: fill LOOP binary in-context (no text round-trip). */
    if (strcmp(mnem, "LOOP") == 0) {
        if (nargs != 3 || (args && (args[0].is_list || args[1].is_list || args[2].is_list))) {
            snprintf(ctx->errmsg, sizeof(ctx->errmsg), "loop(start, end, xn) expects 3 scalars");
            return -1;
        }
        if (args[0].scalar > 0xffffu || args[1].scalar > 0xffffu || args[2].scalar > 0xffffu) {
            snprintf(ctx->errmsg, sizeof(ctx->errmsg), "loop: argument out of u16 range");
            return -1;
        }
        ccu_v1_casm_loop(ctx, (uint16_t)args[0].scalar, (uint16_t)args[1].scalar,
                         (uint16_t)args[2].scalar);
        return 0;
    }

    const CcuV1OpcodeDesc *desc = ccu_v1_lookup_mnemonic(mnem);
    if (!desc) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "unknown builtin '%s'", name);
        return -1;
    }
    if (nargs != desc->nop) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "%s: expected %d args, got %d", name, desc->nop, nargs);
        return -1;
    }

    /* Lower to one numeric asm line and reuse the core encoder. */
    char line[2048];
    size_t pos = 0;
    int w = snprintf(line + pos, sizeof(line) - pos, "%s", mnem);
    if (w < 0 || (size_t)w >= sizeof(line) - pos) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "line buffer overflow");
        return -1;
    }
    pos += (size_t)w;
    for (int i = 0; i < nargs; ++i) {
        char abuf[256];
        if (format_arg(&args[i], abuf, sizeof(abuf)) < 0) {
            snprintf(ctx->errmsg, sizeof(ctx->errmsg), "%s: bad argument %d", name, i);
            return -1;
        }
        w = snprintf(line + pos, sizeof(line) - pos, "%s%s", i == 0 ? " " : ", ", abuf);
        if (w < 0 || (size_t)w >= sizeof(line) - pos) {
            snprintf(ctx->errmsg, sizeof(ctx->errmsg), "line buffer overflow");
            return -1;
        }
        pos += (size_t)w;
    }

    CcuV1Program tmp;
    char err[256];
    if (ccu_v1_assemble_text(line, pos, &tmp, err, sizeof(err)) != 0) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "%s: %s", name, err);
        return -1;
    }
    if (tmp.count != 1) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "%s: expected 1 instr, got %zu", name, tmp.count);
        ccu_v1_program_free(&tmp);
        return -1;
    }
    append_instr(ctx, &tmp.items[0]);
    ccu_v1_program_free(&tmp);
    return 0;
}

/* ---------------- C-style parser ---------------- */

typedef struct {
    const char *p;
    const char *end;
    int line;
    CcuV1CasmCtx *ctx;
} Parser;

static void skip_ws_comments(Parser *ps)
{
    for (;;) {
        while (ps->p < ps->end) {
            char c = *ps->p;
            if (c == ' ' || c == '\t' || c == '\r') {
                ++ps->p;
            } else if (c == '\n') {
                ++ps->p;
                ++ps->line;
            } else {
                break;
            }
        }
        if (ps->p + 1 < ps->end && ps->p[0] == '/' && ps->p[1] == '/') {
            while (ps->p < ps->end && *ps->p != '\n') {
                ++ps->p;
            }
            continue;
        }
        if (ps->p + 1 < ps->end && ps->p[0] == '/' && ps->p[1] == '*') {
            ps->p += 2;
            while (ps->p + 1 < ps->end && !(ps->p[0] == '*' && ps->p[1] == '/')) {
                if (*ps->p == '\n') {
                    ++ps->line;
                }
                ++ps->p;
            }
            if (ps->p + 1 < ps->end) {
                ps->p += 2;
            }
            continue;
        }
        /* Skip preprocessor lines (#include, ...) */
        if (ps->p < ps->end && *ps->p == '#') {
            while (ps->p < ps->end && *ps->p != '\n') {
                ++ps->p;
            }
            continue;
        }
        break;
    }
}

static int match_kw(Parser *ps, const char *kw)
{
    skip_ws_comments(ps);
    size_t n = strlen(kw);
    if ((size_t)(ps->end - ps->p) < n) {
        return 0;
    }
    if (strncmp(ps->p, kw, n) != 0) {
        return 0;
    }
    if (ps->p + n < ps->end) {
        char c = ps->p[n];
        if (isalnum((unsigned char)c) || c == '_') {
            return 0;
        }
    }
    ps->p += n;
    return 1;
}

static int expect_char(Parser *ps, char ch)
{
    skip_ws_comments(ps);
    if (ps->p >= ps->end || *ps->p != ch) {
        snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "line %d: expected '%c'", ps->line, ch);
        return -1;
    }
    ++ps->p;
    return 0;
}

static int parse_ident(Parser *ps, char *out, size_t out_sz)
{
    skip_ws_comments(ps);
    if (ps->p >= ps->end || !(isalpha((unsigned char)*ps->p) || *ps->p == '_')) {
        snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "line %d: expected identifier", ps->line);
        return -1;
    }
    size_t i = 0;
    while (ps->p < ps->end && (isalnum((unsigned char)*ps->p) || *ps->p == '_')) {
        if (i + 1 >= out_sz) {
            snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "line %d: identifier too long", ps->line);
            return -1;
        }
        out[i++] = *ps->p++;
    }
    out[i] = '\0';
    return 0;
}

static int parse_u64_lit(Parser *ps, uint64_t *out)
{
    skip_ws_comments(ps);
    if (ps->p >= ps->end || !isdigit((unsigned char)*ps->p)) {
        snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "line %d: expected number", ps->line);
        return -1;
    }
    uint64_t v = 0;
    if (ps->p + 1 < ps->end && ps->p[0] == '0' && (ps->p[1] == 'x' || ps->p[1] == 'X')) {
        ps->p += 2;
        if (ps->p >= ps->end || !isxdigit((unsigned char)*ps->p)) {
            snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "line %d: bad hex literal", ps->line);
            return -1;
        }
        while (ps->p < ps->end && isxdigit((unsigned char)*ps->p)) {
            char c = *ps->p++;
            unsigned d = (c <= '9') ? (unsigned)(c - '0')
                                    : (unsigned)((c | 32) - 'a' + 10);
            v = (v << 4) | d;
        }
    } else {
        while (ps->p < ps->end && isdigit((unsigned char)*ps->p)) {
            v = v * 10u + (uint64_t)(*ps->p++ - '0');
        }
    }
    /* Optional C integer suffixes: u, l, ull, ... */
    while (ps->p < ps->end) {
        char c = (char)(*ps->p | 32);
        if (c == 'u' || c == 'l') {
            ++ps->p;
        } else {
            break;
        }
    }
    *out = v;
    return 0;
}

static int parse_list_body(Parser *ps, CcuV1CasmArg *arg, char close)
{
    arg->is_list = 1;
    arg->list_n = 0;
    skip_ws_comments(ps);
    if (ps->p < ps->end && *ps->p == close) {
        ++ps->p;
        return 0;
    }
    for (;;) {
        uint64_t v;
        if (parse_u64_lit(ps, &v) != 0) {
            return -1;
        }
        if (v > 0xffffu || arg->list_n >= CCU_V1_MS_MAX) {
            snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "line %d: list element out of range", ps->line);
            return -1;
        }
        arg->list[arg->list_n++] = (uint16_t)v;
        skip_ws_comments(ps);
        if (ps->p < ps->end && *ps->p == ',') {
            ++ps->p;
            continue;
        }
        break;
    }
    if (expect_char(ps, close) != 0) {
        return -1;
    }
    return 0;
}

static int parse_arg(Parser *ps, CcuV1CasmArg *arg)
{
    skip_ws_comments(ps);
    memset(arg, 0, sizeof(*arg));
    /* MS(...) list helper used by native C API */
    if (ps->p + 2 < ps->end && ps->p[0] == 'M' && ps->p[1] == 'S' && ps->p[2] == '(') {
        ps->p += 3;
        return parse_list_body(ps, arg, ')');
    }
    if (ps->p < ps->end && (*ps->p == '{' || *ps->p == '[')) {
        char open = *ps->p++;
        char close = (open == '{') ? '}' : ']';
        return parse_list_body(ps, arg, close);
    }
    if (parse_u64_lit(ps, &arg->scalar) != 0) {
        return -1;
    }
    return 0;
}

static int parse_call_stmt(Parser *ps)
{
    char name[64];
    if (parse_ident(ps, name, sizeof(name)) != 0) {
        return -1;
    }
    if (expect_char(ps, '(') != 0) {
        return -1;
    }

    CcuV1CasmArg args[CCU_V1_CASM_MAX_ARGS];
    int nargs = 0;
    skip_ws_comments(ps);
    if (!(ps->p < ps->end && *ps->p == ')')) {
        for (;;) {
            if (nargs >= CCU_V1_CASM_MAX_ARGS) {
                snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "line %d: too many arguments", ps->line);
                return -1;
            }
            if (parse_arg(ps, &args[nargs]) != 0) {
                return -1;
            }
            ++nargs;
            skip_ws_comments(ps);
            if (ps->p < ps->end && *ps->p == ',') {
                ++ps->p;
                continue;
            }
            break;
        }
    }
    if (expect_char(ps, ')') != 0) {
        return -1;
    }
    if (expect_char(ps, ';') != 0) {
        return -1;
    }

    if (ccu_v1_casm_call(ps->ctx, name, args, nargs) != 0) {
        /* prefix line number if not already present */
        char tmp[CCU_V1_CASM_ERRMSG];
        snprintf(tmp, sizeof(tmp), "line %d: %s", ps->line, ps->ctx->errmsg);
        snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "%s", tmp);
        return -1;
    }
    return 0;
}

static int parse_main(Parser *ps)
{
    skip_ws_comments(ps);
    if (!match_kw(ps, "void") && !match_kw(ps, "int")) {
        snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "line %d: expected void/int main()", ps->line);
        return -1;
    }
    if (!match_kw(ps, "main")) {
        snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "line %d: expected main", ps->line);
        return -1;
    }
    if (expect_char(ps, '(') != 0) {
        return -1;
    }
    skip_ws_comments(ps);
    /* allow empty params or void */
    if (match_kw(ps, "void")) {
        /* ok */
    }
    if (expect_char(ps, ')') != 0) {
        return -1;
    }
    if (expect_char(ps, '{') != 0) {
        return -1;
    }

    for (;;) {
        skip_ws_comments(ps);
        if (ps->p >= ps->end) {
            snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg), "line %d: unexpected EOF in main", ps->line);
            return -1;
        }
        if (*ps->p == '}') {
            ++ps->p;
            break;
        }
        if (parse_call_stmt(ps) != 0) {
            return -1;
        }
    }
    skip_ws_comments(ps);
    if (ps->p < ps->end) {
        snprintf(ps->ctx->errmsg, sizeof(ps->ctx->errmsg),
                 "line %d: trailing tokens after main", ps->line);
        return -1;
    }
    return 0;
}

int ccu_v1_casm_compile(const char *text, size_t text_len, CcuV1CasmCtx *ctx)
{
    if (!ctx) {
        return -1;
    }
    ccu_v1_casm_free(ctx);
    ccu_v1_casm_init(ctx);
    if (ccu_v1_program_reserve(&ctx->program, CCU_V1_CASM_INIT_CAP) != 0) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "oom");
        return -1;
    }

    Parser ps = {.p = text, .end = text + text_len, .line = 1, .ctx = ctx};
    if (parse_main(&ps) != 0) {
        ccu_v1_program_free(&ctx->program);
        return -1;
    }
    return 0;
}
