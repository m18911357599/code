/**
 * C-style CCU V1 assembler.
 *
 * Binary generation flow for programs such as loop_main.c:
 *   1. Create CcuV1CasmCtx          (before calling main)
 *   2. ccu_v1_casm_begin(&ctx)
 *   3. Call user main()             — each instr (loop, ...) fills packed binary
 *   4. ccu_v1_casm_end()
 *   5. ccu_v1_casm_write_file(...)  — write raw 32B*N bytes to file
 */
#ifndef CCU_V1_CASM_H
#define CCU_V1_CASM_H

#include "ccu_v1_asm.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CCU_V1_CASM_MAX_ARGS 24
#define CCU_V1_CASM_ERRMSG   512

typedef struct {
    int is_list;
    uint64_t scalar;
    uint16_t list[CCU_V1_MS_MAX];
    int list_n;
} CcuV1CasmArg;

/* Assembler context: accumulates packed instructions. */
typedef struct {
    CcuV1Program program;
    char errmsg[CCU_V1_CASM_ERRMSG];
    int failed;
} CcuV1CasmCtx;

void ccu_v1_casm_init(CcuV1CasmCtx *ctx);
void ccu_v1_casm_free(CcuV1CasmCtx *ctx);

/* Install / clear thread-local current context (must wrap user main). */
int ccu_v1_casm_begin(CcuV1CasmCtx *ctx);
void ccu_v1_casm_end(void);
CcuV1CasmCtx *ccu_v1_casm_current(void);

/* begin → entry() → end. Returns -1 if entry set ctx->failed. */
int ccu_v1_casm_run(CcuV1CasmCtx *ctx, void (*entry)(void));

/* Encode program and write raw binary bytes to path. */
int ccu_v1_casm_write_file(const CcuV1CasmCtx *ctx, const char *path);

/**
 * Builtin: emit CTRL/LOOP binary into ctx.
 * Layout: start, end, xn  (matches CcuV1Loop / mnemonic LOOP).
 */
int ccu_v1_casm_loop(CcuV1CasmCtx *ctx, uint16_t start, uint16_t end, uint16_t xn);

/**
 * Generic builtin dispatcher: name is C-style (loop, load_imd_to_xn) or
 * ISA mnemonic (LOOP, LOAD_IMD_TO_XN). Fills one instruction into ctx.
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
