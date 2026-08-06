/**
 * C-style CCU V1 assembler.
 *
 * Binary generation flow for programs such as loop_main.c:
 *   1. Create CcuV1CasmCtx          (before calling main)
 *   2. ccu_v1_casm_begin(&ctx)      — pre-reserve slots (zeroed once)
 *   3. Call user main()             — inline emit = bump + set header
 *   4. ccu_v1_casm_end()
 *   5. ccu_v1_casm_write_file(...)  — fwrite packed 32B*N bytes
 *
 * Hot path uses assert (compiles out with -DNDEBUG); no null/error branches.
 */
#ifndef CCU_V1_CASM_H
#define CCU_V1_CASM_H

#include "ccu_v1_asm.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CCU_V1_CASM_MAX_ARGS     24
#define CCU_V1_CASM_ERRMSG       512
#define CCU_V1_CASM_INIT_CAP     2048

typedef struct {
    int is_list;
    uint64_t scalar;
    uint16_t list[CCU_V1_MS_MAX];
    int list_n;
} CcuV1CasmArg;

/* Assembler context: accumulates packed instructions. */
typedef struct {
    CcuV1Program program;
    CcuV1Instr *inst; /* current instruction slot (set by ccu_v1_casm_emit) */
    char errmsg[CCU_V1_CASM_ERRMSG];
    int failed; /* text-interpreter path only */
} CcuV1CasmCtx;

/* Active context for the current thread (set by begin). */
extern __thread CcuV1CasmCtx *ccu_v1_casm_tls;

void ccu_v1_casm_init(CcuV1CasmCtx *ctx);
void ccu_v1_casm_free(CcuV1CasmCtx *ctx);

/* Install / clear thread-local current context (must wrap user main). */
void ccu_v1_casm_begin(CcuV1CasmCtx *ctx);
void ccu_v1_casm_end(void);

static inline CcuV1CasmCtx *ccu_v1_casm_current(void)
{
    return ccu_v1_casm_tls;
}

/* begin → entry() → end. */
void ccu_v1_casm_run(CcuV1CasmCtx *ctx, void (*entry)(void));

/* Cold path: grow zeroed capacity (not on the per-instruction hot path). */
void ccu_v1_casm_grow(CcuV1CasmCtx *ctx);

/**
 * Hot path: bump one pre-zeroed slot, set header, return it.
 * Capacity is pre-reserved in begin(); grow only if exhausted.
 */
static inline CcuV1Instr *ccu_v1_casm_emit(CcuV1CasmCtx *ctx, uint8_t type, uint16_t code)
{
    assert(ctx);
    CcuV1Program *p = &ctx->program;
    if (__builtin_expect(p->count >= p->capacity, 0)) {
        ccu_v1_casm_grow(ctx);
    }
    CcuV1Instr *inst = &p->items[p->count++];
    inst->header.raw = ccu_v1_make_header(type, code);
    ctx->inst = inst;
    return inst;
}

/* fwrite program.items as raw 32B * count. */
void ccu_v1_casm_write_file(const CcuV1CasmCtx *ctx, const char *path);

/** Emit CTRL/LOOP and fill loop fields. */
static inline void ccu_v1_casm_loop(CcuV1CasmCtx *ctx, uint16_t start, uint16_t end, uint16_t xn)
{
    CcuV1Instr *inst = ccu_v1_casm_emit(ctx, CCU_V1_CTRL_TYPE, 0x0);
    inst->loop.start = start;
    inst->loop.end = end;
    inst->loop.xn = xn;
}

/**
 * Generic dispatcher for the text interpreter path.
 * Native intrinsics do NOT use this — they write through ccu_v1_casm_emit.
 */
int ccu_v1_casm_call(CcuV1CasmCtx *ctx, const char *name, const CcuV1CasmArg *args, int nargs);

/**
 * Compile C-style text (void main() { ... }) into ctx->program (interpreter).
 * Native path for loop_main.c: compile as C + ccu_v1_casm_run + write_file.
 */
int ccu_v1_casm_compile(const char *text, size_t text_len, CcuV1CasmCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* CCU_V1_CASM_H */
