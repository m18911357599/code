/**
 * C-style CCU V1 assembler.
 *
 * Source form:
 *   void main() {
 *       loop(0, 10, 11);           // fills LOOP binary into assembler context
 *       load_imd_to_xn(6, 0x1000, 0);
 *   }
 *
 * The assembler owns a CcuV1CasmCtx; builtins such as loop() write one
 * packed CcuV1Instr (32B) into that context.
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
} CcuV1CasmCtx;

void ccu_v1_casm_init(CcuV1CasmCtx *ctx);
void ccu_v1_casm_free(CcuV1CasmCtx *ctx);

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
 * Compile C-style text (void main() { ... }) into ctx->program.
 * Returns 0 on success; on failure fills ctx->errmsg.
 */
int ccu_v1_casm_compile(const char *text, size_t text_len, CcuV1CasmCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* CCU_V1_CASM_H */
